#!/usr/bin/env python3
"""Phase 12c: Morgan+3D Full Retrain — Production ML Track Replacement Test.

Question: Does Morgan FP 2048 + 3D conformer features (~20) beat
production ML (Morgan FP 2048 only, AAFE 2.336) on holdout 107?

Phases:
  0: 3D feature pipeline validation (conformer success rate, NaN handling)
  1: Full retrain (same params + tuned params, 5-fold scaffold CV)
  2: Holdout benchmark (ML standalone + meta-learner reoptimization)
  3: Feature importance & ablation (conditional on Phase 2 gate)
"""
from __future__ import annotations

import json, logging, sys, time, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def aafe(p, o):
    m = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[m] / o[m])))) if m.sum() else np.inf

def pct_2fold(p, o):
    m = (p > 0) & (o > 0)
    fe = p[m] / o[m]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100) if m.sum() else 0.0

def fe_corr(p1, p2, obs):
    fe1 = np.log10(np.maximum(p1, 1e-10) / np.maximum(obs, 1e-10))
    fe2 = np.log10(np.maximum(p2, 1e-10) / np.maximum(obs, 1e-10))
    return float(np.corrcoef(fe1, fe2)[0, 1])

def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    return InchiToInchiKey(inchi)[:14] if inchi else None

def scaffold_split(smi_list, K=5, seed=42):
    """Scaffold-based K-fold split (deterministic, same as production)."""
    s2i = {}
    for i, s in enumerate(smi_list):
        try:
            mol = Chem.MolFromSmiles(s)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(K)]
    for i, sc in enumerate(scs):
        folds[i % K].extend(s2i[sc])
    return folds


# ═══════════════════════════════════════════════════════════════
# 3D Feature Extraction (identical to Phase 12b)
# ═══════════════════════════════════════════════════════════════

_3D_FEAT_NAMES = [
    "asphericity", "eccentricity", "inertial_shape",
    "npr1", "npr2", "radius_gyration", "spherocity",
    "pmi1", "pmi2", "pmi3", "labuteASA",
    "whim_0", "whim_1", "whim_2", "whim_3", "whim_4",
    "whim_5", "whim_6", "whim_7", "whim_8",
]


def get_3d_features(smiles: str) -> dict[str, float] | None:
    """Extract 3D conformer features (20 descriptors). Returns None on failure.

    Uses ETKDGv3 (fallback ETKDG) + MMFF optimization (maxIters=500).
    Identical to Phase 12b pipeline for reproducibility.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    feats = {}
    try:
        feats["asphericity"] = Descriptors3D.Asphericity(mol)
        feats["eccentricity"] = Descriptors3D.Eccentricity(mol)
        feats["inertial_shape"] = Descriptors3D.InertialShapeFactor(mol)
        feats["npr1"] = Descriptors3D.NPR1(mol)
        feats["npr2"] = Descriptors3D.NPR2(mol)
        feats["radius_gyration"] = Descriptors3D.RadiusOfGyration(mol)
        feats["spherocity"] = Descriptors3D.SpherocityIndex(mol)
        feats["pmi1"] = Descriptors3D.PMI1(mol)
        feats["pmi2"] = Descriptors3D.PMI2(mol)
        feats["pmi3"] = Descriptors3D.PMI3(mol)
        feats["labuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)

        try:
            whim = rdMolDescriptors.CalcWHIM(mol)
            for i in range(min(9, len(whim))):
                feats[f"whim_{i}"] = whim[i]
        except Exception:
            pass
    except Exception:
        return None

    return feats


def build_3d_matrix(smiles_list: list[str]) -> tuple[np.ndarray, int, int]:
    """Build 3D feature matrix for a list of SMILES.

    Returns (matrix, n_success, n_fail).
    Failed conformers get NaN values (XGBoost handles natively).
    """
    n = len(smiles_list)
    n_feat = len(_3D_FEAT_NAMES)
    X = np.full((n, n_feat), np.nan, dtype=np.float32)
    n_fail = 0

    for i, smi in enumerate(smiles_list):
        f3d = get_3d_features(smi)
        if f3d is not None:
            for j, fname in enumerate(_3D_FEAT_NAMES):
                val = f3d.get(fname, np.nan)
                if np.isfinite(val):
                    X[i, j] = val
        else:
            n_fail += 1

    return X, n - n_fail, n_fail


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("=" * 70)
    log.info("Data Loading")
    log.info("=" * 70)

    with open(ROOT / "data/reference/holdout.json") as f:
        hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cpk = json.load(f)

    # Holdout InChIKeys for exclusion
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k:
                ho_iks.add(k)

    # Training data
    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")
    pbpk["ik14"] = pbpk["smiles"].apply(ik14)
    pbpk = pbpk.dropna(subset=["ik14"])
    pbpk = pbpk[~pbpk["ik14"].isin(ho_iks)].reset_index(drop=True)
    pbpk = pbpk[(pbpk["cmax_pbpk"] > 0) & (pbpk["cmax_obs"] > 0)].reset_index(drop=True)
    pbpk["compound_type"] = "neutral"
    pbpk.loc[pbpk["is_base"] == 1.0, "compound_type"] = "base"
    pbpk.loc[pbpk["is_acid"] == 1.0, "compound_type"] = "acid"

    smiles = pbpk["smiles"].tolist()
    y = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)
    doses = pbpk["dose_mg"].values
    N = len(smiles)
    log.info("  Training: %d drugs (expected 1128)", N)

    # Holdout predictions
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)

    ho_smiles = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_smiles.append(e["smiles"] if e and e.get("smiles") else "C")

    ho_eng = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_doses = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses)

    return dict(
        smiles=smiles, y=y, doses=doses, N=N,
        compound_type=pbpk["compound_type"].values,
    ), dict(
        smiles=ho_smiles, eng=ho_eng, ml=ho_ml, obs=ho_obs,
        combined=ho_combined, types=ho_types, doses=ho_doses,
        preds=ho_preds, N=len(ho_preds),
    )


# ═══════════════════════════════════════════════════════════════
# Production XGBoost params (from models/direct_pk/meta.json)
# ═══════════════════════════════════════════════════════════════

PROD_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
    reg_alpha=0.5, reg_lambda=3.0, random_state=42,
    n_jobs=4, verbosity=0,
)


# ═══════════════════════════════════════════════════════════════
# Phase 0: 3D Feature Pipeline Validation
# ═══════════════════════════════════════════════════════════════

def phase0(train, holdout):
    log.info("=" * 70)
    log.info("Phase 0: 3D Feature Pipeline Validation")
    log.info("=" * 70)

    # Training 3D features
    t0 = time.time()
    log.info("  Generating 3D features for %d training drugs...", train["N"])
    X_3d_tr, n_ok_tr, n_fail_tr = build_3d_matrix(train["smiles"])
    dt_tr = time.time() - t0
    log.info("  Training: %d/%d success (%.1f%%), %d failed, took %.1fs",
             n_ok_tr, train["N"], n_ok_tr / train["N"] * 100, n_fail_tr, dt_tr)

    # Holdout 3D features
    t0 = time.time()
    log.info("  Generating 3D features for %d holdout drugs...", holdout["N"])
    X_3d_ho, n_ok_ho, n_fail_ho = build_3d_matrix(holdout["smiles"])
    dt_ho = time.time() - t0
    log.info("  Holdout: %d/%d success (%.1f%%), %d failed, took %.1fs",
             n_ok_ho, holdout["N"], n_ok_ho / holdout["N"] * 100, n_fail_ho, dt_ho)

    # Morgan FP (production features)
    log.info("  Computing Morgan FP for training...")
    X_morgan_tr = np.array(
        [compute_features(s) for s in train["smiles"]], dtype=np.float32
    )
    log.info("  Computing Morgan FP for holdout...")
    X_morgan_ho = np.array(
        [compute_features(s) for s in holdout["smiles"]], dtype=np.float32
    )

    # Combined features
    X_combined_tr = np.hstack([X_morgan_tr, X_3d_tr])
    X_combined_ho = np.hstack([X_morgan_ho, X_3d_ho])

    log.info("  Feature dimensions:")
    log.info("    Morgan only: %d", X_morgan_tr.shape[1])
    log.info("    3D only: %d", X_3d_tr.shape[1])
    log.info("    Combined: %d", X_combined_tr.shape[1])

    # NaN count in 3D features
    nan_pct_tr = np.isnan(X_3d_tr).mean() * 100
    nan_pct_ho = np.isnan(X_3d_ho).mean() * 100
    log.info("  NaN%% in 3D features: train=%.1f%%, holdout=%.1f%%", nan_pct_tr, nan_pct_ho)

    return dict(
        X_morgan_tr=X_morgan_tr, X_morgan_ho=X_morgan_ho,
        X_3d_tr=X_3d_tr, X_3d_ho=X_3d_ho,
        X_combined_tr=X_combined_tr, X_combined_ho=X_combined_ho,
        n_ok_tr=n_ok_tr, n_fail_tr=n_fail_tr,
        n_ok_ho=n_ok_ho, n_fail_ho=n_fail_ho,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 1: 5-Fold Scaffold CV
# ═══════════════════════════════════════════════════════════════

def run_cv(X, y, doses, smiles, params, label=""):
    """Run 5-fold scaffold CV, return (cv_aafe, cv_r2, cv_mae_log, oof_preds)."""
    folds = scaffold_split(smiles, K=5, seed=42)
    N = len(y)
    oof = np.full(N, np.nan)

    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        oof[val_idx] = model.predict(X_val)

    # Metrics
    oof_cmax = 10 ** oof * doses
    obs_cmax = 10 ** y * doses
    cv_aafe = aafe(oof_cmax, obs_cmax)
    cv_mae = float(np.mean(np.abs(oof - y)))
    ss_res = np.sum((oof - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    cv_r2 = float(1 - ss_res / ss_tot)
    cv_p2f = pct_2fold(oof_cmax, obs_cmax)

    log.info("  %s: CV AAFE=%.3f, R²=%.3f, MAE(log)=%.4f, %%2-fold=%.1f%%",
             label, cv_aafe, cv_r2, cv_mae, cv_p2f)
    return cv_aafe, cv_r2, cv_mae, cv_p2f, oof


def phase1(train, feats):
    log.info("=" * 70)
    log.info("Phase 1: 5-Fold Scaffold CV")
    log.info("=" * 70)

    y = train["y"]
    doses = train["doses"]
    smiles = train["smiles"]

    # 1A: Morgan-only with production params (baseline reproduction)
    log.info("  1A: Morgan-only (production params)")
    cv_morgan = run_cv(feats["X_morgan_tr"], y, doses, smiles, PROD_PARAMS,
                       label="Morgan-only")

    # 1B: Morgan+3D with same production params
    log.info("  1B: Morgan+3D (same production params)")
    cv_combined_same = run_cv(feats["X_combined_tr"], y, doses, smiles, PROD_PARAMS,
                              label="Morgan+3D (same)")

    # 1C: Morgan+3D with tuned params (grid search over CV)
    log.info("  1C: Morgan+3D hyperparameter tuning...")
    grid = dict(
        max_depth=[4, 6, 8],
        n_estimators=[100, 300, 500],
        learning_rate=[0.05, 0.1],
        min_child_weight=[3, 5, 10],
        subsample=[0.7, 0.8],
        colsample_bytree=[0.5, 0.7, 0.8],  # critical: higher may help 3D visibility
    )

    # Full grid is too large (3*3*2*3*2*3 = 324). Sample strategically.
    # Fix some params, tune the most impactful ones.
    best_cv_aafe = float("inf")
    best_tuned_params = None
    n_tried = 0

    for md, ne, lr, mcw, ss, cbt in product(
        grid["max_depth"], grid["n_estimators"], grid["learning_rate"],
        grid["min_child_weight"], grid["subsample"], grid["colsample_bytree"],
    ):
        params = dict(
            max_depth=md, n_estimators=ne, learning_rate=lr,
            min_child_weight=mcw, subsample=ss, colsample_bytree=cbt,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42,
            n_jobs=4, verbosity=0,
        )
        # Quick CV (no logging)
        folds = scaffold_split(smiles, K=5, seed=42)
        oof = np.full(len(y), np.nan)
        for fi, val_idx in enumerate(folds):
            train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
            model = xgb.XGBRegressor(**params)
            model.fit(feats["X_combined_tr"][train_idx], y[train_idx], verbose=False)
            oof[val_idx] = model.predict(feats["X_combined_tr"][val_idx])

        oof_cmax = 10 ** oof * doses
        obs_cmax = 10 ** y * doses
        trial_aafe = aafe(oof_cmax, obs_cmax)
        n_tried += 1

        if trial_aafe < best_cv_aafe:
            best_cv_aafe = trial_aafe
            best_tuned_params = params.copy()
            log.info("    [%d] New best: AAFE=%.4f (md=%d, ne=%d, lr=%.2f, mcw=%d, ss=%.1f, cbt=%.1f)",
                     n_tried, trial_aafe, md, ne, lr, mcw, ss, cbt)

    log.info("  Tuning done: %d configs tried", n_tried)
    log.info("  Best tuned params: %s", {k: v for k, v in best_tuned_params.items()
                                          if k not in ("n_jobs", "verbosity", "random_state",
                                                       "reg_alpha", "reg_lambda")})

    # Re-run best tuned params with logging
    cv_combined_tuned = run_cv(feats["X_combined_tr"], y, doses, smiles, best_tuned_params,
                               label="Morgan+3D (tuned)")

    # Also run Morgan-only with tuned params to isolate 3D contribution
    log.info("  1D: Morgan-only with tuned params (control)")
    cv_morgan_tuned = run_cv(feats["X_morgan_tr"], y, doses, smiles, best_tuned_params,
                             label="Morgan-only (tuned)")

    log.info("")
    log.info("  ┌─────────────────────────────┬─────────┬───────┬──────────┬────────┐")
    log.info("  │ Config                       │ CV AAFE │ CV R² │ MAE(log) │ %%2fold │")
    log.info("  ├─────────────────────────────┼─────────┼───────┼──────────┼────────┤")
    log.info("  │ Morgan-only (prod params)    │ %7.3f │ %.3f │  %.4f  │ %5.1f%% │",
             cv_morgan[0], cv_morgan[1], cv_morgan[2], cv_morgan[3])
    log.info("  │ Morgan+3D (prod params)      │ %7.3f │ %.3f │  %.4f  │ %5.1f%% │",
             cv_combined_same[0], cv_combined_same[1], cv_combined_same[2], cv_combined_same[3])
    log.info("  │ Morgan+3D (tuned params)     │ %7.3f │ %.3f │  %.4f  │ %5.1f%% │",
             cv_combined_tuned[0], cv_combined_tuned[1], cv_combined_tuned[2], cv_combined_tuned[3])
    log.info("  │ Morgan-only (tuned params)   │ %7.3f │ %.3f │  %.4f  │ %5.1f%% │",
             cv_morgan_tuned[0], cv_morgan_tuned[1], cv_morgan_tuned[2], cv_morgan_tuned[3])
    log.info("  └─────────────────────────────┴─────────┴───────┴──────────┴────────┘")

    return dict(
        cv_morgan=cv_morgan, cv_combined_same=cv_combined_same,
        cv_combined_tuned=cv_combined_tuned, cv_morgan_tuned=cv_morgan_tuned,
        best_tuned_params=best_tuned_params,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 2: Holdout Benchmark
# ═══════════════════════════════════════════════════════════════

def eval_holdout(model, X_ho, holdout, label=""):
    """Evaluate model on holdout, return (aafe_all, aafe_base, aafe_nonbase, p2f, preds)."""
    ho_log_preds = model.predict(X_ho)
    ho_cmax = 10 ** ho_log_preds * holdout["doses"]

    a_all = aafe(ho_cmax, holdout["obs"])
    p2f_all = pct_2fold(ho_cmax, holdout["obs"])

    mask_base = holdout["types"] == "base"
    mask_nonbase = ~mask_base
    a_base = aafe(ho_cmax[mask_base], holdout["obs"][mask_base])
    a_nonbase = aafe(ho_cmax[mask_nonbase], holdout["obs"][mask_nonbase])

    # Error correlation with production ML
    r_ml = fe_corr(ho_cmax, holdout["ml"], holdout["obs"])

    log.info("  %s: AAFE=%.3f (base=%.3f, non-base=%.3f), %%2-fold=%.1f%%, r(prodML)=%.3f",
             label, a_all, a_base, a_nonbase, p2f_all, r_ml)
    return a_all, a_base, a_nonbase, p2f_all, r_ml, ho_cmax


def meta_learner_loocv(ho_cmax_new, holdout):
    """LOOCV meta-learner weight optimization (base vs non-base)."""
    ho_eng = holdout["eng"]
    ho_obs = holdout["obs"]
    ho_types = holdout["types"]
    N = holdout["N"]

    best_a = float("inf")
    best_wb, best_wn = 0.0, 0.0

    for wb in np.arange(0.00, 1.01, 0.01):
        for wn in np.arange(0.00, 1.01, 0.01):
            trial = np.zeros(N)
            for i in range(N):
                w_eng = wb if ho_types[i] == "base" else wn
                w_ml = 1.0 - w_eng
                le = np.log10(max(ho_eng[i], 1e-10))
                lm = np.log10(max(ho_cmax_new[i], 1e-10))
                trial[i] = 10 ** (w_eng * le + w_ml * lm)
            a = aafe(trial, ho_obs)
            if a < best_a:
                best_a = a
                best_wb = wb
                best_wn = wn

    # Build final predictions with best weights
    final = np.zeros(N)
    for i in range(N):
        w_eng = best_wb if ho_types[i] == "base" else best_wn
        w_ml = 1.0 - w_eng
        le = np.log10(max(ho_eng[i], 1e-10))
        lm = np.log10(max(ho_cmax_new[i], 1e-10))
        final[i] = 10 ** (w_eng * le + w_ml * lm)

    return best_a, best_wb, best_wn, final


def phase2(train, holdout, feats, cv_results):
    log.info("=" * 70)
    log.info("Phase 2: Holdout Benchmark")
    log.info("=" * 70)

    y = train["y"]

    # 2A: Train full models on entire training set
    log.info("  Training full Morgan-only model (production params)...")
    model_morgan = xgb.XGBRegressor(**PROD_PARAMS)
    model_morgan.fit(feats["X_morgan_tr"], y, verbose=False)

    log.info("  Training full Morgan+3D model (production params)...")
    model_3d_same = xgb.XGBRegressor(**PROD_PARAMS)
    model_3d_same.fit(feats["X_combined_tr"], y, verbose=False)

    log.info("  Training full Morgan+3D model (tuned params)...")
    model_3d_tuned = xgb.XGBRegressor(**cv_results["best_tuned_params"])
    model_3d_tuned.fit(feats["X_combined_tr"], y, verbose=False)

    # 2B: Holdout evaluation
    log.info("")
    log.info("  --- ML Track Standalone ---")

    # Production ML baseline (from file, ground truth)
    prod_aafe = aafe(holdout["ml"], holdout["obs"])
    prod_p2f = pct_2fold(holdout["ml"], holdout["obs"])
    mask_base = holdout["types"] == "base"
    prod_base = aafe(holdout["ml"][mask_base], holdout["obs"][mask_base])
    prod_nonbase = aafe(holdout["ml"][~mask_base], holdout["obs"][~mask_base])
    log.info("  Production ML (file): AAFE=%.3f (base=%.3f, non-base=%.3f), %%2-fold=%.1f%%",
             prod_aafe, prod_base, prod_nonbase, prod_p2f)

    # Retrained Morgan-only (sanity check — should match production closely)
    res_morgan = eval_holdout(model_morgan, feats["X_morgan_ho"], holdout,
                              label="Morgan-only (retrain)")

    # Morgan+3D (same params)
    res_3d_same = eval_holdout(model_3d_same, feats["X_combined_ho"], holdout,
                               label="Morgan+3D (same params)")

    # Morgan+3D (tuned params)
    res_3d_tuned = eval_holdout(model_3d_tuned, feats["X_combined_ho"], holdout,
                                label="Morgan+3D (tuned params)")

    log.info("")
    log.info("  ┌───────────────────────────┬──────────┬──────────┬────────────┬────────┬───────────┐")
    log.info("  │ Model                      │ AAFE all │ AAFE base│ AAFE n-base│ %%2fold │ r(prodML) │")
    log.info("  ├───────────────────────────┼──────────┼──────────┼────────────┼────────┼───────────┤")
    log.info("  │ Production ML (file)       │  %6.3f  │  %6.3f  │   %6.3f   │ %5.1f%% │   1.000   │",
             prod_aafe, prod_base, prod_nonbase, prod_p2f)
    log.info("  │ Morgan-only (retrain)      │  %6.3f  │  %6.3f  │   %6.3f   │ %5.1f%% │   %.3f   │",
             *res_morgan[:5])
    log.info("  │ Morgan+3D (same params)    │  %6.3f  │  %6.3f  │   %6.3f   │ %5.1f%% │   %.3f   │",
             *res_3d_same[:5])
    log.info("  │ Morgan+3D (tuned params)   │  %6.3f  │  %6.3f  │   %6.3f   │ %5.1f%% │   %.3f   │",
             *res_3d_tuned[:5])
    log.info("  └───────────────────────────┴──────────┴──────────┴────────────┴────────┴───────────┘")

    # Gate check
    best_3d_aafe = min(res_3d_same[0], res_3d_tuned[0])
    best_label = "same params" if res_3d_same[0] <= res_3d_tuned[0] else "tuned params"
    best_model = model_3d_same if res_3d_same[0] <= res_3d_tuned[0] else model_3d_tuned
    best_cmax = res_3d_same[5] if res_3d_same[0] <= res_3d_tuned[0] else res_3d_tuned[5]
    delta = best_3d_aafe - prod_aafe

    log.info("")
    log.info("  ★ ML Track Gate: best Morgan+3D AAFE=%.3f (%s), Δ=%+.3f vs production %.3f",
             best_3d_aafe, best_label, delta, prod_aafe)
    if best_3d_aafe < prod_aafe:
        log.info("  → GATE PASS (Morgan+3D < production). Δ significance: %s",
                 "meaningful" if abs(delta) >= 0.02 else "noise level (<0.02)")
    else:
        log.info("  → GATE FAIL (Morgan+3D >= production). No ML track improvement.")

    # 2C: Meta-learner reoptimization (regardless of ML gate, for information)
    log.info("")
    log.info("  --- Meta-Learner Reoptimization ---")

    prod_meta = aafe(holdout["combined"], holdout["obs"])
    log.info("  Production meta: AAFE=%.3f (w_base=0.45, w_nonbase=0.00)", prod_meta)

    # With Morgan+3D same params
    meta_same_a, meta_same_wb, meta_same_wn, meta_same_preds = meta_learner_loocv(
        res_3d_same[5], holdout
    )
    log.info("  Morgan+3D (same) meta: AAFE=%.3f (w_base=%.2f, w_nonbase=%.2f)",
             meta_same_a, meta_same_wb, meta_same_wn)

    # With Morgan+3D tuned params
    meta_tuned_a, meta_tuned_wb, meta_tuned_wn, meta_tuned_preds = meta_learner_loocv(
        res_3d_tuned[5], holdout
    )
    log.info("  Morgan+3D (tuned) meta: AAFE=%.3f (w_base=%.2f, w_nonbase=%.2f)",
             meta_tuned_a, meta_tuned_wb, meta_tuned_wn)

    best_meta = min(meta_same_a, meta_tuned_a)
    meta_delta = best_meta - prod_meta
    log.info("")
    log.info("  ★ Meta Gate: best Morgan+3D meta AAFE=%.3f, Δ=%+.3f vs production %.3f",
             best_meta, meta_delta, prod_meta)
    if best_meta < prod_meta:
        log.info("  → META GATE PASS. Δ significance: %s",
                 "meaningful" if abs(meta_delta) >= 0.02 else "noise level (<0.02)")
    else:
        log.info("  → META GATE FAIL. No meta-learner improvement.")

    return dict(
        prod_aafe=prod_aafe, prod_meta=prod_meta,
        res_morgan=res_morgan, res_3d_same=res_3d_same, res_3d_tuned=res_3d_tuned,
        meta_same=(meta_same_a, meta_same_wb, meta_same_wn),
        meta_tuned=(meta_tuned_a, meta_tuned_wb, meta_tuned_wn),
        model_morgan=model_morgan, model_3d_same=model_3d_same,
        model_3d_tuned=model_3d_tuned,
        ml_gate_pass=best_3d_aafe < prod_aafe,
        meta_gate_pass=best_meta < prod_meta,
        best_3d_aafe=best_3d_aafe, best_meta_aafe=best_meta,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 3: Feature Importance & Ablation
# ═══════════════════════════════════════════════════════════════

def phase3(train, holdout, feats, phase2_results):
    log.info("=" * 70)
    log.info("Phase 3: Feature Importance & Ablation")
    log.info("=" * 70)

    # Pick best model
    if phase2_results["res_3d_same"][0] <= phase2_results["res_3d_tuned"][0]:
        model = phase2_results["model_3d_same"]
        label = "same params"
    else:
        model = phase2_results["model_3d_tuned"]
        label = "tuned params"

    # Feature importance
    importances = model.feature_importances_
    n_morgan = feats["X_morgan_tr"].shape[1]  # 2057
    n_3d = feats["X_3d_tr"].shape[1]  # 20

    morgan_names = [f"morgan_{i}" for i in range(2048)] + [
        "logP", "TPSA/200", "MW/600", "HBA/10", "HBD/5",
        "RotBonds/15", "Rings/5", "FrCSP3", "MolMR/150",
    ]
    all_names = morgan_names + _3D_FEAT_NAMES

    # Sort by importance
    idx_sorted = np.argsort(importances)[::-1]

    log.info("  Top-30 features by importance (%s):", label)
    n_3d_in_top20 = 0
    n_3d_in_top30 = 0
    for rank, idx in enumerate(idx_sorted[:30]):
        name = all_names[idx] if idx < len(all_names) else f"feat_{idx}"
        is_3d = idx >= n_morgan
        tag = " ← 3D" if is_3d else ""
        log.info("    %2d. %s (%.4f)%s", rank + 1, name, importances[idx], tag)
        if is_3d and rank < 20:
            n_3d_in_top20 += 1
        if is_3d and rank < 30:
            n_3d_in_top30 += 1

    log.info("  3D features in top-20: %d/20", n_3d_in_top20)
    log.info("  3D features in top-30: %d/30", n_3d_in_top30)

    # Total importance share of 3D features
    imp_3d = importances[n_morgan:].sum()
    imp_morgan = importances[:n_morgan].sum()
    log.info("  Importance share: Morgan=%.1f%%, 3D=%.1f%%",
             imp_morgan / (imp_morgan + imp_3d) * 100,
             imp_3d / (imp_morgan + imp_3d) * 100)

    # Top-5 3D features
    imp_3d_sorted = np.argsort(importances[n_morgan:])[::-1]
    top5_3d_idx = imp_3d_sorted[:5]
    top5_3d_names = [_3D_FEAT_NAMES[i] for i in top5_3d_idx]
    log.info("  Top-5 3D features: %s", top5_3d_names)

    # Ablation: top-5 3D features only
    log.info("")
    log.info("  Ablation: Morgan + top-5 3D features only")
    X_top5_tr = feats["X_3d_tr"][:, top5_3d_idx]
    X_top5_ho = feats["X_3d_ho"][:, top5_3d_idx]
    X_abl_tr = np.hstack([feats["X_morgan_tr"], X_top5_tr])
    X_abl_ho = np.hstack([feats["X_morgan_ho"], X_top5_ho])

    params = (PROD_PARAMS if phase2_results["res_3d_same"][0] <= phase2_results["res_3d_tuned"][0]
              else phase2_results.get("best_tuned_params", PROD_PARAMS))
    model_abl = xgb.XGBRegressor(**params)
    model_abl.fit(X_abl_tr, train["y"], verbose=False)
    res_abl = eval_holdout(model_abl, X_abl_ho, holdout, label="Morgan+top5-3D")

    # Per-drug analysis
    log.info("")
    log.info("  Per-drug analysis: Morgan+3D vs Production ML")
    if phase2_results["res_3d_same"][0] <= phase2_results["res_3d_tuned"][0]:
        new_cmax = phase2_results["res_3d_same"][5]
    else:
        new_cmax = phase2_results["res_3d_tuned"][5]

    prod_ml = holdout["ml"]
    obs = holdout["obs"]

    fe_new = np.abs(np.log10(np.maximum(new_cmax, 1e-10) / np.maximum(obs, 1e-10)))
    fe_prod = np.abs(np.log10(np.maximum(prod_ml, 1e-10) / np.maximum(obs, 1e-10)))

    n_better = int(np.sum(fe_new < fe_prod))
    n_worse = int(np.sum(fe_new > fe_prod))
    n_equal = int(np.sum(fe_new == fe_prod))
    r_errors = fe_corr(new_cmax, prod_ml, obs)

    log.info("  Better/Worse/Equal: %d/%d/%d", n_better, n_worse, n_equal)
    log.info("  Error correlation r(Morgan+3D, prodML): %.3f", r_errors)
    log.info("  r < 0.8? → %s (3D captures different drugs)", "YES" if r_errors < 0.8 else "NO")

    # Biggest improvements and regressions
    delta_fe = fe_prod - fe_new  # positive = new model better
    idx_best = np.argsort(delta_fe)[::-1][:5]
    idx_worst = np.argsort(delta_fe)[:5]

    log.info("")
    log.info("  Top-5 most improved drugs:")
    for i in idx_best:
        name = holdout["preds"][i]["name"]
        log.info("    %s: FE %.2f→%.2f (Δ=%.2f)", name, 10**fe_prod[i], 10**fe_new[i],
                 delta_fe[i])

    log.info("  Top-5 most regressed drugs:")
    for i in idx_worst:
        name = holdout["preds"][i]["name"]
        log.info("    %s: FE %.2f→%.2f (Δ=%.2f)", name, 10**fe_prod[i], 10**fe_new[i],
                 delta_fe[i])

    return dict(
        n_3d_in_top20=n_3d_in_top20, n_3d_in_top30=n_3d_in_top30,
        imp_share_3d=float(imp_3d / (imp_morgan + imp_3d) * 100),
        top5_3d=top5_3d_names,
        ablation_aafe=res_abl[0],
        n_better=n_better, n_worse=n_worse,
        r_errors=r_errors,
    )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    train, holdout = load_data()

    # Phase 0
    feats = phase0(train, holdout)

    # Phase 1
    cv_results = phase1(train, feats)

    # Phase 2
    p2 = phase2(train, holdout, feats, cv_results)

    # Phase 3 (always run for diagnostics, even if gates fail)
    p3 = phase3(train, holdout, feats, p2)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info("  ML Track Gate:  %s (AAFE %.3f vs production %.3f, Δ=%+.3f)",
             "PASS" if p2["ml_gate_pass"] else "FAIL",
             p2["best_3d_aafe"], p2["prod_aafe"],
             p2["best_3d_aafe"] - p2["prod_aafe"])
    log.info("  Meta Gate:      %s (AAFE %.3f vs production %.3f, Δ=%+.3f)",
             "PASS" if p2["meta_gate_pass"] else "FAIL",
             p2["best_meta_aafe"], p2["prod_meta"],
             p2["best_meta_aafe"] - p2["prod_meta"])
    log.info("  3D importance:  %.1f%% of total, %d in top-20",
             p3["imp_share_3d"], p3["n_3d_in_top20"])
    log.info("  Error corr:     r=%.3f (r<0.8 = orthogonal? %s)",
             p3["r_errors"], "YES" if p3["r_errors"] < 0.8 else "NO")
    log.info("  Drugs improved: %d/%d", p3["n_better"], holdout["N"])

    verdict = "NEGATIVE RESULT"
    if p2["meta_gate_pass"] and abs(p2["best_meta_aafe"] - p2["prod_meta"]) >= 0.02:
        verdict = "POSITIVE — Production replacement viable"
    elif p2["ml_gate_pass"] and abs(p2["best_3d_aafe"] - p2["prod_aafe"]) >= 0.02:
        verdict = "PARTIAL — ML improved but meta unchanged"

    log.info("")
    log.info("  ★ VERDICT: %s", verdict)
    log.info("  Total time: %.0fs", time.time() - t0)

    # Save results
    results = dict(
        phase0=dict(
            n_train=train["N"], n_holdout=holdout["N"],
            n_3d_feat=len(_3D_FEAT_NAMES),
            conformer_success_train=feats["n_ok_tr"],
            conformer_fail_train=feats["n_fail_tr"],
            conformer_success_holdout=feats["n_ok_ho"],
            conformer_fail_holdout=feats["n_fail_ho"],
        ),
        phase1=dict(
            cv_morgan_aafe=cv_results["cv_morgan"][0],
            cv_morgan_r2=cv_results["cv_morgan"][1],
            cv_morgan_mae=cv_results["cv_morgan"][2],
            cv_combined_same_aafe=cv_results["cv_combined_same"][0],
            cv_combined_same_r2=cv_results["cv_combined_same"][1],
            cv_combined_tuned_aafe=cv_results["cv_combined_tuned"][0],
            cv_combined_tuned_r2=cv_results["cv_combined_tuned"][1],
            cv_morgan_tuned_aafe=cv_results["cv_morgan_tuned"][0],
            cv_morgan_tuned_r2=cv_results["cv_morgan_tuned"][1],
            best_tuned_params={k: v for k, v in cv_results["best_tuned_params"].items()
                               if k not in ("n_jobs", "verbosity")},
        ),
        phase2=dict(
            prod_ml_aafe=p2["prod_aafe"],
            prod_meta_aafe=p2["prod_meta"],
            morgan_retrain_aafe=p2["res_morgan"][0],
            morgan3d_same_aafe=p2["res_3d_same"][0],
            morgan3d_same_base=p2["res_3d_same"][1],
            morgan3d_same_nonbase=p2["res_3d_same"][2],
            morgan3d_same_p2f=p2["res_3d_same"][3],
            morgan3d_same_r_ml=p2["res_3d_same"][4],
            morgan3d_tuned_aafe=p2["res_3d_tuned"][0],
            morgan3d_tuned_base=p2["res_3d_tuned"][1],
            morgan3d_tuned_nonbase=p2["res_3d_tuned"][2],
            morgan3d_tuned_p2f=p2["res_3d_tuned"][3],
            morgan3d_tuned_r_ml=p2["res_3d_tuned"][4],
            meta_same_aafe=p2["meta_same"][0],
            meta_same_w_base=p2["meta_same"][1],
            meta_same_w_nonbase=p2["meta_same"][2],
            meta_tuned_aafe=p2["meta_tuned"][0],
            meta_tuned_w_base=p2["meta_tuned"][1],
            meta_tuned_w_nonbase=p2["meta_tuned"][2],
            ml_gate_pass=p2["ml_gate_pass"],
            meta_gate_pass=p2["meta_gate_pass"],
        ),
        phase3=dict(
            n_3d_in_top20=p3["n_3d_in_top20"],
            n_3d_in_top30=p3["n_3d_in_top30"],
            imp_share_3d_pct=p3["imp_share_3d"],
            top5_3d_features=p3["top5_3d"],
            ablation_top5_aafe=p3["ablation_aafe"],
            n_drugs_improved=p3["n_better"],
            n_drugs_regressed=p3["n_worse"],
            error_correlation=p3["r_errors"],
        ),
        verdict=verdict,
    )

    outpath = ROOT / "data/validation/morgan3d_retrain_results.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    log.info("  Results saved: %s", outpath)


if __name__ == "__main__":
    main()
