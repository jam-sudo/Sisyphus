"""Phase 5: Surrogate model for docking features.

Trains Morgan FP + RDKit desc → predicted docking features.
This eliminates the need for actual docking at inference time.

Only runs if Phase 4 succeeded.

Usage:
    python scripts/train_docking_surrogate.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score

from sisyphus.descriptors import compute_features

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_DIFFDOCK = ROOT / "data" / "docking" / "cache_diffdock"
CACHE_VINA = ROOT / "data" / "docking" / "cache"

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


def load_docking_meta() -> dict:
    meta_path = ROOT / "models" / "adme" / "clint_docking_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def collect_training_data(meta: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Collect all drugs with both base features and docking features.

    Returns:
        X: (N, 2057) base features
        Y: (N, n_dock_features) docking feature targets
        smiles_list: SMILES for each row
        target_names: names of docking feature columns
    """
    cyps = meta["cyps"]
    tool = meta["tool"]
    cache_dir = CACHE_DIFFDOCK if tool == "diffdock" else CACHE_VINA

    target_names = [f"{cyp}_{fn}" for cyp in cyps for fn in DOCK_FEATURE_NAMES]

    # Collect all drugs with cached docking results
    X_list = []
    Y_list = []
    smiles_list = []

    # Scan cache directory for CYP3A4 (first CYP) to get all docked drugs
    first_cyp = cyps[0]
    cache_files = list(cache_dir.glob(f"*_{first_cyp}.json"))
    log.info(f"Found {len(cache_files)} cached docking results for {first_cyp}")

    for cache_file in cache_files:
        with open(cache_file) as f:
            data = json.load(f)

        smiles = data["smiles"]
        key14 = data["inchikey_14"]

        # Check all CYPs are available
        all_cyp_available = True
        dock_vec = []
        for cyp in cyps:
            cyp_path = cache_dir / f"{key14}_{cyp}.json"
            if cyp_path.exists():
                with open(cyp_path) as f:
                    cyp_data = json.load(f)
                for fn in DOCK_FEATURE_NAMES:
                    val = cyp_data.get(fn, float("nan"))
                    dock_vec.append(float(val) if val is not None else float("nan"))
            else:
                all_cyp_available = False
                break

        if not all_cyp_available:
            continue

        # Skip if any NaN in docking features
        if any(np.isnan(v) for v in dock_vec):
            continue

        # Compute base features
        try:
            base_feats = compute_features(smiles)
        except Exception:
            continue

        X_list.append(base_feats)
        Y_list.append(dock_vec)
        smiles_list.append(smiles)

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    log.info(f"Surrogate training data: X={X.shape}, Y={Y.shape}")
    return X, Y, smiles_list, target_names


def train_surrogate(
    X: np.ndarray, Y: np.ndarray, target_names: list[str]
) -> tuple[list[xgb.XGBRegressor], dict]:
    """Train per-feature XGBoost surrogate models."""
    from sklearn.model_selection import KFold

    n_targets = Y.shape[1]
    models = []
    results = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for j in range(n_targets):
        y_j = Y[:, j]
        name = target_names[j]

        # 5-fold CV
        all_preds = np.zeros(len(y_j))
        for train_idx, test_idx in kf.split(X):
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.5,
                random_state=42,
                n_jobs=4,
                verbosity=0,
            )
            model.fit(X[train_idx], y_j[train_idx])
            all_preds[test_idx] = model.predict(X[test_idx])

        r2 = r2_score(y_j, all_preds)
        results[name] = {"r2": r2, "mean": float(np.mean(y_j)), "std": float(np.std(y_j))}

        log.info(f"  {name}: surrogate R²={r2:.3f}")

        # Train final model
        final_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )
        final_model.fit(X, y_j)
        models.append(final_model)

    return models, results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    meta = load_docking_meta()
    X, Y, smiles_list, target_names = collect_training_data(meta)

    if len(X) < 100:
        log.error(f"Insufficient data: {len(X)} drugs with complete docking features")
        return 1

    print("=" * 70)
    print("  Phase 5: Docking Feature Surrogate Model")
    print("=" * 70)
    print(f"  N training drugs: {len(X)}")
    print(f"  N base features: {X.shape[1]}")
    print(f"  N docking features (targets): {Y.shape[1]}")
    print()

    models, results = train_surrogate(X, Y, target_names)

    # Summary
    print()
    print(f"  {'Feature':>40s}  {'R²':>8s}  {'Verdict':>10s}")
    print("  " + "-" * 65)

    pass_count = 0
    for name, res in results.items():
        verdict = "GOOD" if res["r2"] > 0.5 else "WEAK" if res["r2"] > 0.3 else "FAIL"
        if res["r2"] > 0.5:
            pass_count += 1
        print(f"  {name:>40s}  {res['r2']:>8.3f}  {verdict:>10s}")

    mean_r2 = np.mean([r["r2"] for r in results.values()])
    print(f"\n  Mean surrogate R²: {mean_r2:.3f}")
    print(f"  Features with R² > 0.5: {pass_count}/{len(results)}")

    # Gate check
    print()
    if mean_r2 > 0.5:
        print("  PASS: Surrogate models are adequate (mean R² > 0.5)")
        print("  Docking features can be approximated from structure alone at inference time.")

        # Save models
        model_dir = ROOT / "models" / "adme" / "docking_surrogates"
        model_dir.mkdir(parents=True, exist_ok=True)
        for model, name in zip(models, target_names):
            model.save_model(str(model_dir / f"surrogate_{name}.json"))
        log.info(f"Saved {len(models)} surrogate models to {model_dir}")

        # Save metadata
        surr_meta = {
            "target_names": target_names,
            "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            "n_training": len(X),
            "mean_r2": float(mean_r2),
        }
        with open(model_dir / "surrogate_meta.json", "w") as f:
            json.dump(surr_meta, f, indent=2)

    elif mean_r2 > 0.3:
        print("  MARGINAL: Surrogate models are weak (mean R² 0.3-0.5)")
        print("  Consider using real docking features where cached, NaN otherwise.")

    else:
        print("  FAIL: Surrogate models are inadequate (mean R² < 0.3)")
        print("  Docking features cannot be predicted from structure — use cache or NaN.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
