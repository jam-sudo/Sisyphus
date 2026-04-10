"""Phase 3: CLint model retraining with docking features.

Compares:
- model_base: Morgan FP (2048) + RDKit desc (9) = 2057 features
- model_enriched: Morgan FP (2048) + RDKit desc (9) + docking features = 2057 + N

Uses the same TDC Hepatocyte_AZ dataset, same 5-fold scaffold CV.
Gate: ΔR² > 0.05 to proceed to Phase 4.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.metrics import mean_absolute_error, r2_score

from sisyphus.descriptors import compute_features

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_VINA = ROOT / "data" / "docking" / "cache"
CACHE_DIFFDOCK = ROOT / "data" / "docking" / "cache_diffdock"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
HEP_AZ_PATH = ROOT / "data" / "clearance_hepatocyte_az.tab"

CYPS = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"]

# Docking feature names per CYP
DOCK_FEATURE_NAMES = [
    "docking_score",
    "heme_fe_distance_min",
    "pose_centroid_fe_dist",
    "n_heavy_atoms",
]


def inchikey_14(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        import hashlib
        return hashlib.md5(smiles.encode()).hexdigest()[:14]
    return Chem.inchi.MolToInchiKey(mol)[:14]


def load_holdout_smiles() -> set[str]:
    """Load holdout drug SMILES to exclude from training."""
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    holdout_names = set(holdout_data.get("holdout", []))
    train_names = set(holdout_data.get("train", []))
    all_names = holdout_names | train_names

    with open(CLINICAL_PK_JSON) as f:
        cpk = json.load(f)

    holdout_smiles = set()
    for name, data in cpk["drugs"].items():
        if name in all_names and "smiles" in data:
            mol = Chem.MolFromSmiles(data["smiles"])
            if mol:
                holdout_smiles.add(Chem.MolToSmiles(mol))
    return holdout_smiles


def load_docking_features(smiles: str, cache_dir: Path, cyps: list[str]) -> dict[str, float]:
    """Load docking features from cache for a SMILES."""
    key14 = inchikey_14(smiles)
    features = {}

    for cyp in cyps:
        cache_path = cache_dir / f"{key14}_{cyp}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            for feat_name in DOCK_FEATURE_NAMES:
                val = data.get(feat_name, float("nan"))
                if val is None:
                    val = float("nan")
                features[f"{cyp}_{feat_name}"] = float(val)
        else:
            for feat_name in DOCK_FEATURE_NAMES:
                features[f"{cyp}_{feat_name}"] = float("nan")

    return features


def load_tdc_data() -> pd.DataFrame:
    """Load TDC Hepatocyte_AZ data with holdout exclusion."""
    holdout_smiles = load_holdout_smiles()

    rows = []
    with open(HEP_AZ_PATH) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            drug_id = parts[0].strip('"')
            smiles = parts[1].strip('"')
            clint = float(parts[2])

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)

            if canon in holdout_smiles:
                continue

            rows.append({"drug_id": drug_id, "smiles": smiles, "canon_smiles": canon, "clint": clint})

    df = pd.DataFrame(rows)
    log.info(f"TDC Hepatocyte_AZ: {len(df)} drugs (after holdout exclusion)")
    return df


def build_feature_matrices(
    df: pd.DataFrame,
    use_diffdock: bool = True,
    cyps: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Build base and enriched feature matrices.

    Returns:
        X_base: (N, 2057) base features
        X_enriched: (N, 2057 + n_dock_feats) enriched features
        y: (N,) log10(CLint)
        smiles_list: list of canonical SMILES
        dock_feature_names: names of docking features
    """
    if cyps is None:
        cyps = CYPS

    cache_dir = CACHE_DIFFDOCK if use_diffdock else CACHE_VINA

    base_features = []
    dock_features = []
    targets = []
    smiles_list = []
    failed = 0

    dock_feat_names = [f"{cyp}_{fn}" for cyp in cyps for fn in DOCK_FEATURE_NAMES]

    for _, row in df.iterrows():
        smi = row["smiles"]
        try:
            base = compute_features(smi)
        except Exception:
            failed += 1
            continue

        dock = load_docking_features(smi, cache_dir, cyps)
        dock_vec = [dock[name] for name in dock_feat_names]

        base_features.append(base)
        dock_features.append(dock_vec)
        targets.append(np.log10(max(row["clint"], 0.1)))
        smiles_list.append(row["canon_smiles"])

    X_base = np.array(base_features, dtype=np.float64)
    X_dock = np.array(dock_features, dtype=np.float64)
    X_enriched = np.hstack([X_base, X_dock])
    y = np.array(targets, dtype=np.float64)

    # Count docking coverage
    n_with_dock = int(np.sum(~np.isnan(X_dock[:, 0])))
    log.info(
        f"Features: base={X_base.shape}, enriched={X_enriched.shape}, "
        f"docking coverage={n_with_dock}/{len(y)} ({100*n_with_dock/len(y):.0f}%), "
        f"failed={failed}"
    )

    return X_base, X_enriched, y, smiles_list, dock_feat_names


def scaffold_split(smiles_list: list[str], n_folds: int = 5, seed: int = 42) -> list[list[int]]:
    """Murcko scaffold split."""
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            scaffold = ""
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, scaffold in enumerate(scaffolds):
        folds[i % n_folds].extend(scaffold_to_indices[scaffold])

    return folds


def train_cv(
    X: np.ndarray, y: np.ndarray, smiles_list: list[str], label: str
) -> tuple[xgb.XGBRegressor, dict]:
    """5-fold scaffold CV, return final model + metrics."""
    folds = scaffold_split(smiles_list)

    all_preds = np.zeros(len(y))
    fold_r2s = []
    fold_maes = []

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

    for fold_idx, test_indices in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_indices] = False

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_indices])

        all_preds[test_indices] = preds

        fold_r2 = r2_score(y[test_indices], preds)
        fold_mae = mean_absolute_error(y[test_indices], preds)
        fold_r2s.append(fold_r2)
        fold_maes.append(fold_mae)

    overall_r2 = r2_score(y, all_preds)
    overall_mae = mean_absolute_error(y, all_preds)

    log.info(
        f"[{label}] CV R²={overall_r2:.4f} (folds: {[f'{r:.3f}' for r in fold_r2s]}), "
        f"MAE={overall_mae:.4f}"
    )

    # Train final model on all data
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X, y)

    metrics = {
        "label": label,
        "n": len(y),
        "n_features": X.shape[1],
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "fold_r2s": fold_r2s,
        "fold_maes": fold_maes,
        "cv_r2_mean": np.mean(fold_r2s),
        "cv_r2_std": np.std(fold_r2s),
    }

    return final_model, metrics


def analyze_feature_importance(model: xgb.XGBRegressor, feature_names: list[str], n_top: int = 30):
    """Print top feature importances, highlighting docking features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'Rank':>4s}  {'Feature':>40s}  {'Importance':>10s}  {'Type':>10s}")
    print("-" * 70)

    dock_in_top = 0
    for rank, idx in enumerate(indices[:n_top]):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        imp = importances[idx]
        is_dock = any(cyp in name for cyp in CYPS)
        tag = "DOCKING" if is_dock else "base"
        if is_dock:
            dock_in_top += 1
        print(f"{rank+1:>4d}  {name:>40s}  {imp:>10.4f}  {tag:>10s}")

    print(f"\nDocking features in top {n_top}: {dock_in_top}")

    # Total importance by category
    base_imp = sum(importances[:2057])
    dock_imp = sum(importances[2057:])
    total_imp = sum(importances)
    print(f"Total importance: base={base_imp/total_imp:.1%}, docking={dock_imp/total_imp:.1%}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Check which docking cache is available
    n_diffdock = len(list(CACHE_DIFFDOCK.glob("*_CYP3A4.json"))) if CACHE_DIFFDOCK.exists() else 0
    n_vina = len(list(CACHE_VINA.glob("*_CYP3A4.json"))) if CACHE_VINA.exists() else 0

    log.info(f"Docking cache: DiffDock={n_diffdock}, Vina={n_vina} (CYP3A4)")

    # Decide which cache to use and which CYPs
    if n_diffdock > n_vina:
        use_diffdock = True
        # Check which CYPs have good coverage
        avail_cyps = []
        for cyp in CYPS:
            n = len(list(CACHE_DIFFDOCK.glob(f"*_{cyp}.json")))
            if n > 500:
                avail_cyps.append(cyp)
                log.info(f"  {cyp}: {n} cached (DiffDock)")
        if not avail_cyps:
            avail_cyps = ["CYP3A4"]
    else:
        use_diffdock = False
        avail_cyps = ["CYP3A4"]
        log.info(f"Using Vina cache (CYP3A4 only): {n_vina} cached")

    # Load data
    df = load_tdc_data()

    # Build features
    X_base, X_enriched, y, smiles_list, dock_feat_names = build_feature_matrices(
        df, use_diffdock=use_diffdock, cyps=avail_cyps
    )

    # Build feature name list
    base_names = [f"morgan_{i}" for i in range(2048)] + [
        "logP", "tpsa", "mw", "hba", "hbd", "rotbonds", "rings", "fsp3", "molmr"
    ]
    all_names = base_names + dock_feat_names

    # Train and compare
    print("\n" + "=" * 70)
    print("  CLint MODEL COMPARISON: Base vs Enriched (Docking Features)")
    print("=" * 70)

    model_base, metrics_base = train_cv(X_base, y, smiles_list, "CLint_base")
    model_enriched, metrics_enriched = train_cv(X_enriched, y, smiles_list, "CLint_enriched")

    delta_r2 = metrics_enriched["overall_r2"] - metrics_base["overall_r2"]

    print()
    print(f"{'':25s} {'Base':>12s}  {'Enriched':>12s}  {'Delta':>8s}")
    print("-" * 65)
    print(f"{'N features':25s} {metrics_base['n_features']:>12d}  {metrics_enriched['n_features']:>12d}")
    print(f"{'N compounds':25s} {metrics_base['n']:>12d}  {metrics_enriched['n']:>12d}")
    print(f"{'CV R²':25s} {metrics_base['overall_r2']:>12.4f}  {metrics_enriched['overall_r2']:>12.4f}  {delta_r2:>+8.4f}")
    print(f"{'CV MAE':25s} {metrics_base['overall_mae']:>12.4f}  {metrics_enriched['overall_mae']:>12.4f}  {metrics_enriched['overall_mae']-metrics_base['overall_mae']:>+8.4f}")
    print(f"{'CV R² mean±std':25s} {metrics_base['cv_r2_mean']:>.3f}±{metrics_base['cv_r2_std']:.3f}    {metrics_enriched['cv_r2_mean']:>.3f}±{metrics_enriched['cv_r2_std']:.3f}")
    print()

    # Feature importance analysis
    print("\n--- Feature Importance (Enriched Model) ---")
    analyze_feature_importance(model_enriched, all_names, n_top=30)

    # Gate check
    print("\n" + "=" * 70)
    print("  GATE CHECK")
    print("=" * 70)
    if delta_r2 > 0.05:
        print(f"PASS: ΔR² = {delta_r2:+.4f} > 0.05 threshold")
        print("Docking features significantly improve CLint prediction.")
        print("Proceed to Phase 4 (holdout benchmark).")

        # Save enriched model
        model_path = ROOT / "models" / "adme" / "xgboost_clint_docking.json"
        model_enriched.save_model(str(model_path))
        log.info(f"Enriched model saved: {model_path}")

        # Save metadata
        meta = {
            "tool": "diffdock" if use_diffdock else "vina",
            "cyps": avail_cyps,
            "n_dock_features": len(dock_feat_names),
            "dock_feature_names": dock_feat_names,
            "metrics_base": {k: v for k, v in metrics_base.items() if k != "label"},
            "metrics_enriched": {k: v for k, v in metrics_enriched.items() if k != "label"},
            "delta_r2": delta_r2,
        }
        meta_path = ROOT / "models" / "adme" / "clint_docking_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

        return 0

    elif delta_r2 > 0.03:
        print(f"MARGINAL: ΔR² = {delta_r2:+.4f} (between 0.03 and 0.05)")
        print("Docking features show weak improvement. Consider expanding CYP coverage.")
        return 0

    else:
        print(f"FAIL: ΔR² = {delta_r2:+.4f} < 0.03 threshold")
        print("Docking features are uninformative for CLint prediction.")
        print("STOP: Do not proceed to Phase 4.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
