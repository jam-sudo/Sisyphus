"""Foundation Model Shootout: frozen embeddings → XGBoost → CLint R².

Compares 5 representations:
  A. Morgan FP 2048 + RDKit 9 (baseline)
  B. MoLFormer-XL 768-dim
  C. ChemBERTa-2 768-dim
  D. Uni-Mol 512-dim (3D conformer)
  E. MoLFormer 768 + Morgan FP 2057 (local + global)

All use the same XGBoost, same 5-fold scaffold CV, same TDC Hepatocyte_AZ data.
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
from sklearn.metrics import mean_absolute_error, r2_score

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
HEP_AZ_PATH = ROOT / "data" / "clearance_hepatocyte_az.tab"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    """Load TDC Hepatocyte_AZ CLint data (holdout excluded)."""
    holdout = load_holdout_smiles()
    smiles_list = []
    targets = []
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
    y = np.array(targets, dtype=np.float64)
    log.info(f"Loaded {len(smiles_list)} compounds (holdout excluded)")
    return smiles_list, y


# ---------------------------------------------------------------------------
# Embedding extractors
# ---------------------------------------------------------------------------

def extract_morgan(smiles_list: list[str]) -> np.ndarray:
    """Baseline: Morgan FP 2048 + 9 RDKit descriptors = 2057."""
    from sisyphus.descriptors import compute_features
    feats = []
    for smi in smiles_list:
        feats.append(compute_features(smi))
    return np.array(feats, dtype=np.float64)


def extract_molformer(smiles_list: list[str], batch_size: int = 64) -> np.ndarray:
    """MoLFormer-XL frozen embeddings (768-dim CLS token)."""
    from transformers import AutoModel, AutoTokenizer

    model_name = "ibm-research/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(DEVICE).eval()

    all_embeds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeds.append(cls_embed)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embeds).astype(np.float64)


def extract_chemberta(smiles_list: list[str], batch_size: int = 64) -> np.ndarray:
    """ChemBERTa-2 frozen embeddings (768-dim CLS token)."""
    from transformers import AutoModel, AutoTokenizer

    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(DEVICE).eval()

    all_embeds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeds.append(cls_embed)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embeds).astype(np.float64)


def extract_unimol(smiles_list: list[str], batch_size: int = 64) -> np.ndarray:
    """Uni-Mol frozen embeddings (512-dim CLS repr, 3D conformer)."""
    from unimol_tools import UniMolRepr

    clf = UniMolRepr(data_type="molecule")

    all_embeds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        reprs = clf.get_repr(batch)
        # reprs is list of (512,) arrays
        all_embeds.extend(reprs)

    return np.array(all_embeds, dtype=np.float64)


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def scaffold_split(smiles_list: list[str], n_folds: int = 5, seed: int = 42) -> list[list[int]]:
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
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scaffolds):
        folds[i % n_folds].extend(scaffold_to_idx[sc])
    return folds


def evaluate(X: np.ndarray, y: np.ndarray, smiles_list: list[str], label: str) -> dict:
    """5-fold scaffold CV with XGBoost."""
    folds = scaffold_split(smiles_list)
    all_preds = np.zeros(len(y))
    fold_r2s = []

    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )

    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds
        fold_r2s.append(r2_score(y[test_idx], preds))

    overall_r2 = r2_score(y, all_preds)
    overall_mae = mean_absolute_error(y, all_preds)

    log.info(f"[{label}] R²={overall_r2:.4f} (folds: {[f'{r:.3f}' for r in fold_r2s]}), MAE={overall_mae:.4f}")
    return {
        "label": label,
        "n_features": X.shape[1],
        "r2": overall_r2,
        "mae": overall_mae,
        "fold_r2s": fold_r2s,
        "r2_mean": np.mean(fold_r2s),
        "r2_std": np.std(fold_r2s),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    smiles_list, y = load_data()

    results = []

    # A. Morgan FP baseline
    log.info("=== A. Morgan FP 2048 + RDKit 9 ===")
    t0 = time.time()
    X_morgan = extract_morgan(smiles_list)
    log.info(f"  Extracted in {time.time()-t0:.1f}s, shape={X_morgan.shape}")
    results.append(evaluate(X_morgan, y, smiles_list, "Morgan_FP"))

    # B. MoLFormer-XL
    log.info("=== B. MoLFormer-XL 768 ===")
    t0 = time.time()
    X_molformer = extract_molformer(smiles_list)
    log.info(f"  Extracted in {time.time()-t0:.1f}s, shape={X_molformer.shape}")
    results.append(evaluate(X_molformer, y, smiles_list, "MoLFormer"))

    # C. ChemBERTa-2
    log.info("=== C. ChemBERTa-2 768 ===")
    t0 = time.time()
    X_chemberta = extract_chemberta(smiles_list)
    log.info(f"  Extracted in {time.time()-t0:.1f}s, shape={X_chemberta.shape}")
    results.append(evaluate(X_chemberta, y, smiles_list, "ChemBERTa2"))

    # D. Uni-Mol 512
    log.info("=== D. Uni-Mol 512 (3D) ===")
    t0 = time.time()
    try:
        X_unimol = extract_unimol(smiles_list)
        log.info(f"  Extracted in {time.time()-t0:.1f}s, shape={X_unimol.shape}")
        results.append(evaluate(X_unimol, y, smiles_list, "UniMol_3D"))
    except Exception as e:
        log.error(f"  Uni-Mol failed: {e}")

    # E. MoLFormer + Morgan FP (concatenated)
    log.info("=== E. MoLFormer + Morgan FP ===")
    X_combined = np.hstack([X_molformer, X_morgan])
    log.info(f"  Combined shape={X_combined.shape}")
    results.append(evaluate(X_combined, y, smiles_list, "MoLFormer+Morgan"))

    # F. Uni-Mol + Morgan FP
    if "X_unimol" in dir():
        log.info("=== F. Uni-Mol + Morgan FP ===")
        X_uni_morgan = np.hstack([X_unimol, X_morgan])
        log.info(f"  Combined shape={X_uni_morgan.shape}")
        results.append(evaluate(X_uni_morgan, y, smiles_list, "UniMol+Morgan"))

    # Summary table
    print("\n" + "=" * 80)
    print("  FOUNDATION MODEL SHOOTOUT — CLint Prediction (5-fold Scaffold CV)")
    print("=" * 80)
    print(f"\n  N compounds: {len(y)}")
    print(f"  Target: log10(CLint), TDC Hepatocyte_AZ")
    print()
    print(f"  {'Model':<25s} {'Dim':>6s} {'R²':>8s} {'MAE':>8s} {'R² folds':>25s} {'ΔR²':>8s}")
    print("  " + "-" * 80)

    baseline_r2 = results[0]["r2"]
    for r in results:
        delta = r["r2"] - baseline_r2
        fold_str = ", ".join(f"{f:.3f}" for f in r["fold_r2s"])
        marker = " ***" if delta > 0.05 else " **" if delta > 0.03 else " *" if delta > 0.01 else ""
        print(f"  {r['label']:<25s} {r['n_features']:>6d} {r['r2']:>8.4f} {r['mae']:>8.4f} {fold_str:>25s} {delta:>+8.4f}{marker}")

    # Verdict
    best = max(results, key=lambda r: r["r2"])
    delta_best = best["r2"] - baseline_r2

    print()
    if delta_best > 0.05:
        print(f"  PASS: {best['label']} beats Morgan FP by ΔR²={delta_best:+.4f} (>0.05)")
        print(f"  → Proceed to fine-tuning (Phase 1)")
    elif delta_best > 0.03:
        print(f"  MARGINAL: {best['label']} ΔR²={delta_best:+.4f} (0.03-0.05)")
        print(f"  → Fine-tuning may help, proceed cautiously")
    else:
        print(f"  FAIL: Best model {best['label']} ΔR²={delta_best:+.4f} (<0.03)")
        print(f"  → Frozen embeddings do not improve over Morgan FP")
        print(f"  → Fine-tuning unlikely to overcome this gap")

    return 0


if __name__ == "__main__":
    sys.exit(main())
