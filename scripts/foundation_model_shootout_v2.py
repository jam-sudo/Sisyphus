"""Foundation Model Shootout v2: Fix methodology issues.

Changes from v1:
- Ridge regression (linear probe) + MLP in addition to XGBoost
- Mean pooling in addition to CLS token
- Verify ChemBERTa model variant
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
HEP_AZ_PATH = ROOT / "data" / "clearance_hepatocyte_az.tab"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_holdout_smiles() -> set[str]:
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    all_names = set(holdout_data.get("holdout", [])) | set(holdout_data.get("train", []))
    with open(CLINICAL_PK_JSON) as f:
        cpk = json.load(f)
    holdout_smiles = set()
    for name, data in cpk["drugs"].items():
        if name in all_names and "smiles" in data:
            mol = Chem.MolFromSmiles(data["smiles"])
            if mol:
                holdout_smiles.add(Chem.MolToSmiles(mol))
    return holdout_smiles


def load_data() -> tuple[list[str], np.ndarray]:
    holdout = load_holdout_smiles()
    smiles_list, targets = [], []
    with open(HEP_AZ_PATH) as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            smi = parts[1].strip('"')
            clint = float(parts[2])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in holdout:
                continue
            smiles_list.append(smi)
            targets.append(np.log10(max(clint, 0.1)))
    return smiles_list, np.array(targets, dtype=np.float64)


# ---------------------------------------------------------------------------
# Embedding extractors — CLS + mean pooling
# ---------------------------------------------------------------------------

def extract_morgan(smiles_list: list[str]) -> np.ndarray:
    from sisyphus.descriptors import compute_features
    return np.array([compute_features(s) for s in smiles_list], dtype=np.float64)


def _extract_transformer(smiles_list: list[str], model_name: str,
                         batch_size: int = 64, trust_remote: bool = False
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLS and mean-pooled embeddings from a HF transformer."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote)
    model = model.to(DEVICE).eval()

    cls_list, mean_list = [], []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # CLS token
        cls_embed = hidden[:, 0, :].cpu().numpy()
        cls_list.append(cls_embed)

        # Mean pooling (masked)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        masked_hidden = hidden * attention_mask
        mean_embed = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        mean_list.append(mean_embed.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return np.vstack(cls_list).astype(np.float64), np.vstack(mean_list).astype(np.float64)


def extract_molformer(smiles_list):
    return _extract_transformer(smiles_list, "ibm-research/MoLFormer-XL-both-10pct",
                                trust_remote=True)


def extract_chemberta(smiles_list):
    # Try the larger model first
    return _extract_transformer(smiles_list, "DeepChem/ChemBERTa-77M-MTR")


def extract_unimol(smiles_list: list[str], batch_size: int = 64) -> np.ndarray:
    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type="molecule")
    all_embeds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        reprs = clf.get_repr(batch)
        all_embeds.extend(reprs)
    return np.array(all_embeds, dtype=np.float64)


# ---------------------------------------------------------------------------
# CV evaluation — 3 downstream models
# ---------------------------------------------------------------------------

def scaffold_split(smiles_list, n_folds=5, seed=42):
    scaffold_to_idx: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        scaffold_to_idx.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_idx.keys())
    rng.shuffle(scaffolds)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scaffolds):
        folds[i % n_folds].extend(scaffold_to_idx[sc])
    return folds


def evaluate_multi(X: np.ndarray, y: np.ndarray, smiles_list: list[str], label: str) -> dict:
    """5-fold scaffold CV with Ridge, MLP, and XGBoost."""
    folds = scaffold_split(smiles_list)

    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                                     early_stopping=True, random_state=42, verbose=False),
        "XGB": lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, n_jobs=4, verbosity=0),
    }

    results = {}
    for model_name, model_fn in models.items():
        all_preds = np.zeros(len(y))
        fold_r2s = []

        for test_idx in folds:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False

            X_train, X_test = X[train_mask], X[test_idx]
            y_train, y_test = y[train_mask], y[test_idx]

            # Standardize for Ridge/MLP (important for dense embeddings)
            if model_name in ("Ridge", "MLP"):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            m = model_fn()
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            all_preds[test_idx] = preds
            fold_r2s.append(r2_score(y_test, preds))

        r2 = r2_score(y, all_preds)
        mae = mean_absolute_error(y, all_preds)
        results[model_name] = {"r2": r2, "mae": mae, "fold_r2s": fold_r2s}

    best_model = max(results, key=lambda k: results[k]["r2"])
    best_r2 = results[best_model]["r2"]
    log.info(f"[{label}] Best: {best_model} R²={best_r2:.4f}")

    return {"label": label, "n_features": X.shape[1], "results": results, "best_model": best_model, "best_r2": best_r2}


def main():
    smiles_list, y = load_data()
    all_results = []

    # A. Morgan FP baseline
    log.info("=== A. Morgan FP ===")
    X_morgan = extract_morgan(smiles_list)
    all_results.append(evaluate_multi(X_morgan, y, smiles_list, "Morgan_FP"))

    # B/C. MoLFormer
    log.info("=== B/C. MoLFormer-XL ===")
    X_mf_cls, X_mf_mean = extract_molformer(smiles_list)
    log.info(f"  CLS: {X_mf_cls.shape}, Mean: {X_mf_mean.shape}")
    all_results.append(evaluate_multi(X_mf_cls, y, smiles_list, "MoLFormer_CLS"))
    all_results.append(evaluate_multi(X_mf_mean, y, smiles_list, "MoLFormer_Mean"))

    # D/E. ChemBERTa-2
    log.info("=== D/E. ChemBERTa-2 ===")
    X_cb_cls, X_cb_mean = extract_chemberta(smiles_list)
    log.info(f"  CLS: {X_cb_cls.shape}, Mean: {X_cb_mean.shape}")
    all_results.append(evaluate_multi(X_cb_cls, y, smiles_list, "ChemBERTa_CLS"))
    all_results.append(evaluate_multi(X_cb_mean, y, smiles_list, "ChemBERTa_Mean"))

    # F. Uni-Mol
    log.info("=== F. Uni-Mol 3D ===")
    try:
        X_unimol = extract_unimol(smiles_list)
        log.info(f"  Shape: {X_unimol.shape}")
        all_results.append(evaluate_multi(X_unimol, y, smiles_list, "UniMol_3D"))
    except Exception as e:
        log.error(f"  Uni-Mol failed: {e}")

    # G. Best transformer + Morgan (concatenated)
    log.info("=== G. MoLFormer_Mean + Morgan ===")
    X_mf_morgan = np.hstack([X_mf_mean, X_morgan])
    all_results.append(evaluate_multi(X_mf_morgan, y, smiles_list, "MoLFormer+Morgan"))

    # Summary
    print("\n" + "=" * 95)
    print("  FOUNDATION MODEL SHOOTOUT v2 — Ridge / MLP / XGBoost × Embedding")
    print("=" * 95)
    print(f"  N={len(y)} compounds, target=log10(CLint)\n")

    baseline_r2 = all_results[0]["best_r2"]

    print(f"  {'Representation':<22s} {'Dim':>5s} | {'Ridge':>8s} {'MLP':>8s} {'XGB':>8s} | {'Best':>8s} {'ΔR²':>8s}")
    print("  " + "-" * 85)

    for r in all_results:
        ridge_r2 = r["results"]["Ridge"]["r2"]
        mlp_r2 = r["results"]["MLP"]["r2"]
        xgb_r2 = r["results"]["XGB"]["r2"]
        best = r["best_r2"]
        delta = best - baseline_r2
        marker = " ***" if delta > 0.05 else " **" if delta > 0.03 else " *" if delta > 0.01 else ""
        print(f"  {r['label']:<22s} {r['n_features']:>5d} | {ridge_r2:>8.4f} {mlp_r2:>8.4f} {xgb_r2:>8.4f} | {best:>8.4f} {delta:>+8.4f}{marker}")

    best_overall = max(all_results, key=lambda r: r["best_r2"])
    delta_best = best_overall["best_r2"] - baseline_r2

    print()
    if delta_best > 0.05:
        print(f"  PASS: {best_overall['label']} ({best_overall['best_model']}) ΔR²={delta_best:+.4f}")
    elif delta_best > 0.03:
        print(f"  MARGINAL: {best_overall['label']} ({best_overall['best_model']}) ΔR²={delta_best:+.4f}")
    else:
        print(f"  FAIL: Best={best_overall['label']} ({best_overall['best_model']}) ΔR²={delta_best:+.4f}")
        print(f"  → Foundation model embeddings do not improve CLint prediction")


if __name__ == "__main__":
    sys.exit(main())
