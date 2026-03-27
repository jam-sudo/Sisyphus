#!/usr/bin/env python3
"""Phase 1 + 1b: Train CL/F and Vd/F XGBoost prediction models.

Phase 1: CL/F model
  Target: log10(CL/F in mL/min/kg)
  Features: Morgan FP 2048 + 9 RDKit descriptors = 2057
  5-fold CV, report R², RMSE, %2-fold, in-sample R²
  Gate: CV R² >= 0.15

Phase 1b: Vd/F model
  Target: log10(Vd/F in L/kg)
  Same features and approach
  Gate: CV R² >= 0.20

N > 300, so full 2057 features are acceptable per spec.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# CL/F sanity bounds (mL/min/kg) — filter extremes for training
CLF_MIN = 0.1
CLF_MAX = 200.0

# Vd/F sanity bounds (L/kg)
VDF_MIN = 0.03
VDF_MAX = 50.0


def compute_features(smiles: str) -> np.ndarray:
    """Compute 2057-element feature vector (Morgan FP + RDKit descriptors)."""
    from sisyphus.descriptors import compute_features as _compute
    return _compute(smiles)


def load_data() -> tuple[list[dict], list[dict]]:
    """Load CL/F and Vd/F training datasets from Phase 0 output.

    Returns:
        clf_data: list of dicts with smiles, log10_clf (in-range only)
        vdf_data: list of dicts with smiles, log10_vdf (in-range only)
    """
    path = ROOT / "data" / "training" / "clf_training.csv"
    clf_data = []
    vdf_data = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["smiles"].strip()
            clf = float(row["clf_mL_min_kg"])
            log10_clf = float(row["log10_clf"])

            # Filter in-range CL/F for training
            if CLF_MIN <= clf <= CLF_MAX:
                clf_data.append({"smiles": smiles, "name": row["name"], "log10_clf": log10_clf})

            # Vd/F: filter in-range and non-empty
            if row["log10_vdf"].strip():
                vdf = float(row["vdf_L_kg"])
                log10_vdf = float(row["log10_vdf"])
                if VDF_MIN <= vdf <= VDF_MAX:
                    vdf_data.append({"smiles": smiles, "name": row["name"], "log10_vdf": log10_vdf})

    return clf_data, vdf_data


def featurize(data: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Compute features for all drugs. Returns (X, valid_smiles)."""
    X_list = []
    valid = []
    failed = 0
    for d in data:
        try:
            feat = compute_features(d["smiles"])
            X_list.append(feat)
            valid.append(d["smiles"])
        except Exception:
            failed += 1
    if failed:
        logger.warning("Failed to featurize %d drugs", failed)
    return np.array(X_list), valid


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Train XGBoost with 5-fold CV. Returns metrics dict."""

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # XGBoost hyperparameters (conservative to avoid overfitting)
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_weight": 5,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": 0,
        "tree_method": "hist",
    }

    cv_predictions = np.zeros(len(y))
    fold_r2 = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        cv_predictions[val_idx] = y_pred

        # Per-fold R²
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fold_r2.append(r2)

    # Overall CV metrics
    ss_res = np.sum((y - cv_predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    cv_rmse = float(np.sqrt(np.mean((y - cv_predictions) ** 2)))

    # %2-fold: percentage of predictions within 2-fold of observed
    # In log10 space: |pred - obs| < log10(2) ≈ 0.301
    abs_errors = np.abs(y - cv_predictions)
    pct_2fold = float(np.mean(abs_errors < np.log10(2)) * 100)

    # In-sample R² (overfitting check)
    model_full = xgb.XGBRegressor(**params)
    model_full.fit(X, y, verbose=False)
    y_insample = model_full.predict(X)
    ss_res_in = np.sum((y - y_insample) ** 2)
    insample_r2 = 1 - ss_res_in / ss_tot if ss_tot > 0 else 0

    logger.info("\n=== %s Model Results ===", target_name)
    logger.info("N training:        %d", len(y))
    logger.info("N features:        %d", X.shape[1])
    logger.info("5-fold CV R²:      %.3f", cv_r2)
    logger.info("5-fold CV RMSE:    %.3f", cv_rmse)
    logger.info("%%2-fold:           %.1f%%", pct_2fold)
    logger.info("In-sample R²:      %.3f", insample_r2)
    logger.info("Overfitting gap:   %.3f (in-sample - CV)", insample_r2 - cv_r2)
    logger.info("Per-fold R²:       %s", [f"{r:.3f}" for r in fold_r2])

    if insample_r2 - cv_r2 > 0.15:
        logger.warning("⚠ OVERFITTING DETECTED: gap %.3f > 0.15", insample_r2 - cv_r2)

    return {
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "pct_2fold": pct_2fold,
        "insample_r2": insample_r2,
        "fold_r2": fold_r2,
        "model": model_full,
        "cv_predictions": cv_predictions,
    }


def main() -> None:
    # Load data
    clf_data, vdf_data = load_data()
    logger.info("CL/F training data: %d drugs (in-range)", len(clf_data))
    logger.info("Vd/F training data: %d drugs (in-range)", len(vdf_data))

    # Feature computation (shared)
    logger.info("\nComputing features for CL/F dataset...")
    all_smiles_clf = [d["smiles"] for d in clf_data]
    X_clf_list = []
    y_clf_list = []
    clf_names = []
    failed_clf = 0
    for d in clf_data:
        try:
            feat = compute_features(d["smiles"])
            X_clf_list.append(feat)
            y_clf_list.append(d["log10_clf"])
            clf_names.append(d["name"])
        except Exception:
            failed_clf += 1

    X_clf = np.array(X_clf_list)
    y_clf = np.array(y_clf_list)
    logger.info("CL/F featurized: %d OK, %d failed", len(X_clf), failed_clf)

    logger.info("\nComputing features for Vd/F dataset...")
    X_vdf_list = []
    y_vdf_list = []
    vdf_names = []
    failed_vdf = 0
    for d in vdf_data:
        try:
            feat = compute_features(d["smiles"])
            X_vdf_list.append(feat)
            y_vdf_list.append(d["log10_vdf"])
            vdf_names.append(d["name"])
        except Exception:
            failed_vdf += 1

    X_vdf = np.array(X_vdf_list)
    y_vdf = np.array(y_vdf_list)
    logger.info("Vd/F featurized: %d OK, %d failed", len(X_vdf), failed_vdf)

    # --- Phase 1: CL/F model ---
    clf_result = train_and_evaluate(X_clf, y_clf, "CL/F")

    # Gate check
    logger.info("\n=== CL/F GATE CHECK ===")
    if clf_result["cv_r2"] >= 0.15:
        logger.info("PASS: CL/F CV R² = %.3f >= 0.15", clf_result["cv_r2"])
    else:
        logger.error("FAIL: CL/F CV R² = %.3f < 0.15. Ceiling reached.", clf_result["cv_r2"])
        sys.exit(1)

    # --- Phase 1b: Vd/F model ---
    vdf_result = train_and_evaluate(X_vdf, y_vdf, "Vd/F")

    # Gate check
    logger.info("\n=== Vd/F GATE CHECK ===")
    if vdf_result["cv_r2"] >= 0.20:
        logger.info("PASS: Vd/F CV R² = %.3f >= 0.20", vdf_result["cv_r2"])
    else:
        logger.warning("FAIL: Vd/F CV R² = %.3f < 0.20. Will use engine Vd fallback.", vdf_result["cv_r2"])

    # --- Redundancy test: CL/F predicted Cmax vs ML direct Cmax ---
    logger.info("\n=== Redundancy Test ===")
    logger.info("Loading existing ML Cmax model for correlation check...")
    try:
        from sisyphus.ml.models import PKPredictor
        predictor = PKPredictor()

        # Get ML predictions for CL/F training drugs
        ml_predictions = []
        clf_cv_preds = clf_result["cv_predictions"]
        valid_pairs = []

        for i, name in enumerate(clf_names):
            smiles = clf_data[i]["smiles"] if i < len(clf_data) else None
            if smiles is None:
                continue
            try:
                ml_cmax = predictor.predict_cmax(smiles, 100.0)  # normalize at 100mg
                # CL/F predicted: log10_clf → CL/F → approximate Cmax
                # For correlation, just use log10 CL/F as proxy (inverse relationship)
                ml_predictions.append(np.log10(max(ml_cmax.mean, 1e-10)))
                valid_pairs.append(i)
            except Exception:
                continue

        if len(valid_pairs) > 30:
            ml_arr = np.array(ml_predictions)
            clf_cv_arr = clf_cv_preds[valid_pairs]

            # CL/F is inversely related to Cmax, so use negative CL/F
            r, p = pearsonr(-clf_cv_arr, ml_arr)
            logger.info("Pearson r (CL/F predicted vs ML Cmax):    %.3f (p=%.2e)", r, p)
            logger.info("  N pairs: %d", len(valid_pairs))
            if abs(r) > 0.90:
                logger.warning("r > 0.90: CL/F and ML Cmax are highly correlated — low additive value")
            elif abs(r) < 0.70:
                logger.info("r < 0.70: CL/F and ML Cmax are orthogonal — good additive value expected")
            else:
                logger.info("0.70 < r < 0.90: moderate correlation — some additive value expected")
        else:
            logger.warning("Too few valid pairs (%d) for correlation test", len(valid_pairs))
    except Exception as e:
        logger.warning("Redundancy test failed: %s", e)

    # --- Save models ---
    model_dir = ROOT / "models" / "direct_pk"
    model_dir.mkdir(parents=True, exist_ok=True)

    clf_model_path = model_dir / "xgboost_clf.json"
    clf_result["model"].save_model(str(clf_model_path))
    logger.info("\nCL/F model saved to %s", clf_model_path)

    vdf_model_path = model_dir / "xgboost_vdf.json"
    vdf_result["model"].save_model(str(vdf_model_path))
    logger.info("Vd/F model saved to %s", vdf_model_path)

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("CL/F:  N=%d  CV R²=%.3f  RMSE=%.3f  %%2-fold=%.1f%%  Gap=%.3f",
                len(y_clf), clf_result["cv_r2"], clf_result["cv_rmse"],
                clf_result["pct_2fold"], clf_result["insample_r2"] - clf_result["cv_r2"])
    logger.info("Vd/F:  N=%d  CV R²=%.3f  RMSE=%.3f  %%2-fold=%.1f%%  Gap=%.3f",
                len(y_vdf), vdf_result["cv_r2"], vdf_result["cv_rmse"],
                vdf_result["pct_2fold"], vdf_result["insample_r2"] - vdf_result["cv_r2"])


if __name__ == "__main__":
    main()
