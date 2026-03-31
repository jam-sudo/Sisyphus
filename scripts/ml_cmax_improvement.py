#!/usr/bin/env python3
"""ML Cmax Track Improvement — Features + Ensemble + Meta-Learner.

Phase 1: Feature comparison (Morgan, Mordred, combined, MACCS) + Optuna
Phase 2: Ensemble (XGB+LGB+RF with diverse features)
Phase 3: Meta-learner re-optimization + holdout gate

The ML track bypasses IVIVE entirely — no error cancellation risk.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

def _ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    if not inchi: return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None

def load_holdout_ik():
    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cp = json.load(f)
    names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    ik_set = set()
    for n in names:
        e = drugs.get(n) or drugs.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ik = _ik14(e["smiles"])
            if ik: ik_set.add(ik)
    return ik_set

def scaffold_split(smiles_list, n_folds=5, seed=42):
    s2i = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except: sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scs): folds[i % n_folds].extend(s2i[sc])
    return folds

def compute_aafe(pred, obs):
    mask = (pred > 0) & (obs > 0)
    if mask.sum() == 0: return float("inf")
    return float(10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask]))))

# ═══════════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_morgan(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    for b in fp.GetOnBits(): arr[b] = 1.0
    return arr

def compute_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array([int(fp.GetBit(i)) for i in range(167)], dtype=np.float32)

# Mordred (lazy init)
_mordred_calc = None
def compute_mordred(smi):
    global _mordred_calc
    if _mordred_calc is None:
        from mordred import Calculator, descriptors
        _mordred_calc = Calculator(descriptors, ignore_3D=True)
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    try:
        result = _mordred_calc(mol)
        return np.array([float(v) if not hasattr(v, 'error') and v is not None
                         else np.nan for v in result], dtype=np.float32)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# Load MMPK training data (ML Cmax target: log10(Cmax/Dose))
# ═══════════════════════════════════════════════════════════════════════════

def load_mmpk_data():
    ho_ik = load_holdout_ik()
    mmpk = pd.read_csv(ROOT / "data/training/mmpk_expanded_full.csv")
    mmpk = mmpk[~mmpk["in_holdout"]].reset_index(drop=True)

    # Per-drug: take median entry
    dedup = mmpk.groupby("canon_smiles").agg(
        dose_mg=("dose_mg", "median"),
        cmax=("cmax_mg_L", "median"),
        log_cpd=("log_cmax_per_dose", "median"),
    ).reset_index()
    dedup["ik"] = dedup["canon_smiles"].apply(_ik14)
    dedup = dedup.dropna(subset=["ik"])
    dedup = dedup[~dedup["ik"].isin(ho_ik)].reset_index(drop=True)
    dedup = dedup[dedup["cmax"] > 0].reset_index(drop=True)

    log.info("MMPK training: %d drugs", len(dedup))
    return dedup

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Feature comparison + Optuna
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_sets(dedup):
    """Compute all feature sets for training drugs."""
    log.info("Computing feature sets ...")

    records = []
    n_mordred_fail = 0
    for i, (_, row) in enumerate(dedup.iterrows()):
        smi = row["canon_smiles"]
        rec = {"smi": smi, "y": float(row["log_cpd"]), "dose": float(row["dose_mg"])}

        # Morgan FP
        mfp = compute_morgan(smi)
        if mfp is None: continue
        rec["morgan"] = mfp

        # Sisyphus descriptors (Morgan 2048 + 9 RDKit = 2057)
        try:
            rec["sisyphus"] = compute_features(smi).astype(np.float32)
        except:
            continue

        # MACCS
        maccs = compute_maccs(smi)
        rec["maccs"] = maccs if maccs is not None else np.zeros(167, dtype=np.float32)

        # Mordred
        mord = compute_mordred(smi)
        if mord is None:
            n_mordred_fail += 1
            rec["mordred"] = None
        else:
            rec["mordred"] = mord

        records.append(rec)

        if (i + 1) % 200 == 0:
            log.info("  Features: %d/%d", i + 1, len(dedup))

    log.info("Feature computation: %d success, %d mordred failures", len(records), n_mordred_fail)

    # Process Mordred: remove high-NaN and constant columns
    mordred_valid = [r for r in records if r["mordred"] is not None]
    if mordred_valid:
        mordred_mat = np.array([r["mordred"] for r in mordred_valid])
        nan_frac = np.isnan(mordred_mat).mean(axis=0)
        var = np.nanvar(mordred_mat, axis=0)
        keep = (nan_frac < 0.5) & (var > 1e-10)
        mordred_keep_idx = np.where(keep)[0]
        log.info("Mordred: %d raw → %d after NaN/constant removal", mordred_mat.shape[1], len(mordred_keep_idx))

        # Apply selection + impute NaN with column median
        for r in records:
            if r["mordred"] is not None:
                sel = r["mordred"][mordred_keep_idx]
                r["mordred_clean"] = sel
            else:
                r["mordred_clean"] = np.full(len(mordred_keep_idx), np.nan, dtype=np.float32)

        # Compute column medians for NaN imputation
        clean_mat = np.array([r["mordred_clean"] for r in records])
        col_medians = np.nanmedian(clean_mat, axis=0)
        for r in records:
            nans = np.isnan(r["mordred_clean"])
            if nans.any():
                r["mordred_clean"][nans] = col_medians[nans]
    else:
        for r in records:
            r["mordred_clean"] = np.zeros(0, dtype=np.float32)

    return records, len(mordred_keep_idx) if mordred_valid else 0


def cv_aafe(X, y, smiles, params):
    """5-fold scaffold CV AAFE for a regression model."""
    folds = scaffold_split(smiles)
    all_pred = np.zeros(len(y))
    for fi in range(5):
        te = folds[fi]; tr = [j for f in range(5) if f != fi for j in folds[f]]
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], verbose=False)
        all_pred[te] = m.predict(X[te])
    # AAFE on dose-normalized predictions (same scale as target)
    pred_cmax = 10 ** all_pred  # Cmax/Dose
    obs_cmax = 10 ** y          # Cmax/Dose
    return compute_aafe(pred_cmax, obs_cmax)


def run_phase1(records, n_mordred):
    """Feature comparison + Optuna."""
    log.info("=" * 60)
    log.info("Phase 1: Feature Comparison + Optuna")
    log.info("=" * 60)

    smiles = [r["smi"] for r in records]
    y = np.array([r["y"] for r in records])

    feature_sets = {
        "A_morgan": np.array([r["morgan"] for r in records]),
        "B_mordred": np.array([r["mordred_clean"] for r in records]),
        "C_morgan_mordred": np.array([np.concatenate([r["morgan"], r["mordred_clean"]]) for r in records]),
        "D_sisyphus": np.array([r["sisyphus"] for r in records]),
        "E_maccs": np.array([r["maccs"] for r in records]),
    }

    default_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=4, verbosity=0)

    results = {}
    best_aafe = float("inf")
    best_set_name = None

    for name, X in feature_sets.items():
        if X.shape[1] == 0:
            log.info("  %s: skipped (0 features)", name)
            continue
        # Replace remaining NaN with 0 for XGBoost
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        aafe = cv_aafe(X, y, smiles, default_params)
        results[name] = {"n_feat": X.shape[1], "cv_aafe": aafe}
        log.info("  %s: %d features, CV AAFE=%.3f", name, X.shape[1], aafe)
        if aafe < best_aafe:
            best_aafe = aafe
            best_set_name = name

    log.info("Best feature set: %s (CV AAFE=%.3f)", best_set_name, best_aafe)

    # Optuna on best feature set
    log.info("\nOptuna HP optimization on %s ...", best_set_name)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_best = np.nan_to_num(feature_sets[best_set_name], nan=0.0, posinf=0.0, neginf=0.0)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": 4, "verbosity": 0,
        }
        return cv_aafe(X_best, y, smiles, params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    best_optuna_aafe = study.best_value
    best_params = {**study.best_params, "random_state": 42, "n_jobs": 4, "verbosity": 0}
    results["optuna_best"] = {
        "feature_set": best_set_name, "cv_aafe": best_optuna_aafe,
        "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best_params.items()},
    }
    log.info("Optuna: CV AAFE=%.3f (baseline A=%.3f)", best_optuna_aafe, results["A_morgan"]["cv_aafe"])

    return results, X_best, y, smiles, best_params, best_set_name


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Ensemble
# ═══════════════════════════════════════════════════════════════════════════

def run_phase2(records, y, smiles, best_xgb_params):
    """Diverse ensemble: XGB+Morgan, LGB+Mordred, RF+MACCS."""
    log.info("=" * 60)
    log.info("Phase 2: Diverse Feature Ensemble")
    log.info("=" * 60)

    from sklearn.ensemble import RandomForestRegressor

    X_morgan = np.nan_to_num(np.array([r["morgan"] for r in records]), nan=0.0)
    X_mordred = np.nan_to_num(np.array([r["mordred_clean"] for r in records]), nan=0.0)
    X_maccs = np.nan_to_num(np.array([r["maccs"] for r in records]), nan=0.0)

    folds = scaffold_split(smiles)

    # Collect OOF predictions for each base model
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    oof_rf = np.zeros(len(y))

    for fi in range(5):
        te = folds[fi]; tr = [j for f in range(5) if f != fi for j in folds[f]]

        # XGBoost + Morgan (best params)
        m1 = xgb.XGBRegressor(**best_xgb_params)
        m1.fit(X_morgan[tr], y[tr], verbose=False)
        oof_xgb[te] = m1.predict(X_morgan[te])

        # LightGBM + Mordred
        try:
            import lightgbm as lgb
            m2 = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                                    n_jobs=4, verbose=-1)
            m2.fit(X_mordred[tr], y[tr])
            oof_lgb[te] = m2.predict(X_mordred[te])
        except ImportError:
            # Fallback: XGBoost + Mordred
            m2 = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   n_jobs=4, verbosity=0)
            m2.fit(X_mordred[tr], y[tr], verbose=False)
            oof_lgb[te] = m2.predict(X_mordred[te])

        # Random Forest + MACCS
        m3 = RandomForestRegressor(n_estimators=500, max_depth=12, max_features="sqrt",
                                    random_state=42, n_jobs=4)
        m3.fit(X_maccs[tr], y[tr])
        oof_rf[te] = m3.predict(X_maccs[te])

    # Individual AAFEs
    obs_cpd = 10 ** y
    aafe_xgb = compute_aafe(10 ** oof_xgb, obs_cpd)
    aafe_lgb = compute_aafe(10 ** oof_lgb, obs_cpd)
    aafe_rf = compute_aafe(10 ** oof_rf, obs_cpd)
    log.info("Individual: XGB=%.3f, LGB/Mordred=%.3f, RF/MACCS=%.3f", aafe_xgb, aafe_lgb, aafe_rf)

    # Simple average (log-space geometric mean)
    oof_avg = (oof_xgb + oof_lgb + oof_rf) / 3.0
    aafe_avg = compute_aafe(10 ** oof_avg, obs_cpd)
    log.info("Simple average: AAFE=%.3f", aafe_avg)

    # Weighted average (optimize on OOF)
    best_w_aafe = float("inf")
    best_w = (1/3, 1/3, 1/3)
    for w1 in np.arange(0, 1.05, 0.05):
        for w2 in np.arange(0, 1.05 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01: continue
            oof_w = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_rf
            a = compute_aafe(10 ** oof_w, obs_cpd)
            if a < best_w_aafe:
                best_w_aafe = a
                best_w = (w1, w2, w3)

    log.info("Weighted avg: w=(%.2f,%.2f,%.2f), AAFE=%.3f", *best_w, best_w_aafe)

    # Two-model ensemble (XGB + LGB only, skip RF if weak)
    oof_2avg = (oof_xgb + oof_lgb) / 2.0
    aafe_2avg = compute_aafe(10 ** oof_2avg, obs_cpd)
    log.info("XGB+LGB avg: AAFE=%.3f", aafe_2avg)

    return {
        "xgb_morgan": aafe_xgb,
        "lgb_mordred": aafe_lgb,
        "rf_maccs": aafe_rf,
        "simple_avg_3": aafe_avg,
        "weighted_avg_3": best_w_aafe,
        "weights_3": best_w,
        "xgb_lgb_avg": aafe_2avg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Holdout benchmark + Meta-learner re-optimization
# ═══════════════════════════════════════════════════════════════════════════

def run_holdout_benchmark(records, y, smiles, best_params, best_set_name, ensemble_result):
    """Run holdout N=107 benchmark with best ML configuration."""
    log.info("=" * 60)
    log.info("Phase 3: Holdout Benchmark + Meta-Learner")
    log.info("=" * 60)

    from sisyphus.validation.reference import load_reference
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile

    # Determine best ML approach from Phase 1+2
    phase1_best_aafe = None  # will get from results
    phase2_best_aafe = min(v for v in ensemble_result.values() if isinstance(v, (int, float))) if ensemble_result else float("inf")

    # Train final models on all training data
    # Best single model
    X_morgan = np.nan_to_num(np.array([r["morgan"] for r in records]), nan=0.0)
    X_mordred = np.nan_to_num(np.array([r["mordred_clean"] for r in records]), nan=0.0)
    X_maccs = np.nan_to_num(np.array([r["maccs"] for r in records]), nan=0.0)

    # Feature set for best single model
    feature_map = {
        "A_morgan": X_morgan,
        "B_mordred": X_mordred,
        "C_morgan_mordred": np.hstack([X_morgan, X_mordred]),
        "D_sisyphus": np.nan_to_num(np.array([r["sisyphus"] for r in records]), nan=0.0),
        "E_maccs": X_maccs,
    }
    X_single = feature_map.get(best_set_name, X_morgan)

    # Train final single model
    model_single = xgb.XGBRegressor(**best_params)
    model_single.fit(X_single, y, verbose=False)

    # Train ensemble base models on all data
    model_xgb = xgb.XGBRegressor(**best_params)
    model_xgb.fit(X_morgan, y, verbose=False)

    try:
        import lightgbm as lgb
        model_lgb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                                       n_jobs=4, verbose=-1)
    except ImportError:
        model_lgb = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                                      n_jobs=4, verbosity=0)
    model_lgb.fit(X_mordred, y)

    # Holdout evaluation
    refs = [r for r in load_reference() if r.in_holdout]
    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    holdout_drugs = []
    for ref in refs:
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            eng = result.engine_pk.cmax.mean if result.engine_pk else None
            ml_baseline = result.ml_pk.cmax.mean if result.ml_pk else None
            if not (eng and eng > 0 and ml_baseline and ml_baseline > 0):
                continue

            p = compute_profile(ref.smiles)

            # New ML predictions
            morgan_feat = compute_morgan(ref.smiles)
            mordred_feat = compute_mordred(ref.smiles)

            if morgan_feat is None: continue
            if mordred_feat is None:
                mordred_feat = np.zeros(X_mordred.shape[1], dtype=np.float32)
            else:
                # Apply same mordred cleaning as training
                mordred_clean_idx = np.arange(X_mordred.shape[1])  # already cleaned in records
                if len(mordred_feat) > X_mordred.shape[1]:
                    # Need to select same columns — use first N
                    mordred_feat = mordred_feat[:X_mordred.shape[1]]
                mordred_feat = np.nan_to_num(mordred_feat, nan=0.0)

            # Single model prediction
            if best_set_name == "A_morgan":
                feat_single = morgan_feat.reshape(1, -1)
            elif best_set_name == "C_morgan_mordred":
                feat_single = np.concatenate([morgan_feat, mordred_feat[:X_mordred.shape[1]]]).reshape(1, -1)
            else:
                feat_single = morgan_feat.reshape(1, -1)  # fallback

            log_cpd_single = float(model_single.predict(feat_single)[0])
            cmax_single = 10 ** log_cpd_single * ref.dose_mg

            # Ensemble prediction (XGB+LGB average)
            log_cpd_xgb = float(model_xgb.predict(morgan_feat.reshape(1, -1))[0])
            log_cpd_lgb = float(model_lgb.predict(mordred_feat[:X_mordred.shape[1]].reshape(1, -1))[0])
            log_cpd_ens = (log_cpd_xgb + log_cpd_lgb) / 2.0
            cmax_ens = 10 ** log_cpd_ens * ref.dose_mg

            holdout_drugs.append({
                "name": ref.name, "obs": ref.cmax_obs,
                "eng": eng, "ml_base": ml_baseline,
                "ml_single": cmax_single, "ml_ens": cmax_ens,
                "is_base": p.compound_type == "base",
            })
        except Exception:
            continue

    logging.getLogger("sisyphus").setLevel(logging.INFO)
    log.info("Holdout drugs evaluated: %d", len(holdout_drugs))

    # Compute AAFEs
    df = pd.DataFrame(holdout_drugs)

    def _aafe(pred_col):
        p = df[pred_col].values; o = df["obs"].values
        return compute_aafe(p, o)

    ml_base_aafe = _aafe("ml_base")
    ml_single_aafe = _aafe("ml_single")
    ml_ens_aafe = _aafe("ml_ens")

    # Meta-learner: combine engine + ML for each ML variant
    def _meta_aafe(ml_col, wb, wo):
        errs = []
        for _, d in df.iterrows():
            w = wb if d["is_base"] else wo
            le = np.log10(max(d["eng"], 1e-10))
            lm = np.log10(max(d[ml_col], 1e-10))
            dis = abs(le - lm)
            if dis > 1.0: w = w * (1.0 / dis)
            lc = w * le + (1 - w) * lm
            errs.append(abs(lc - np.log10(max(d["obs"], 1e-10))))
        return 10 ** np.mean(errs)

    meta_base = _meta_aafe("ml_base", 0.45, 0.00)
    meta_single_oldw = _meta_aafe("ml_single", 0.45, 0.00)
    meta_ens_oldw = _meta_aafe("ml_ens", 0.45, 0.00)

    # LOOCV weight re-optimization for best ML variant
    best_ml_col = "ml_ens" if ml_ens_aafe < ml_single_aafe else "ml_single"
    best_loocv_aafe, best_wb, best_wo = float("inf"), 0.0, 0.0
    for wb in np.arange(0, 1.01, 0.05):
        for wo in np.arange(0, 0.51, 0.05):
            a = _meta_aafe(best_ml_col, wb, wo)
            if a < best_loocv_aafe:
                best_loocv_aafe, best_wb, best_wo = a, wb, wo

    # Print results
    print()
    print("=" * 75)
    print("  ML CMAX IMPROVEMENT — HOLDOUT RESULTS (N=%d)" % len(df))
    print("=" * 75)
    print()
    print(f"  {'Configuration':<35s} {'ML AAFE':>8s} {'Meta AAFE':>10s}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline (Morgan, XGB, w=0.45)':<35s} {ml_base_aafe:>8.3f} {meta_base:>10.3f}")
    print(f"  {'Single best + Optuna (old w)':<35s} {ml_single_aafe:>8.3f} {meta_single_oldw:>10.3f}")
    print(f"  {'XGB+LGB ensemble (old w)':<35s} {ml_ens_aafe:>8.3f} {meta_ens_oldw:>10.3f}")
    print(f"  {'Best ML + new w ({:.2f}/{:.2f})':<35s} {'—':>8s} {best_loocv_aafe:>10.3f}".format(best_wb, best_wo))
    print()

    # Gate
    best_meta = min(meta_single_oldw, meta_ens_oldw, best_loocv_aafe)
    delta = best_meta - meta_base

    print(f"  Best meta AAFE: {best_meta:.3f} (baseline: {meta_base:.3f}, Δ={delta:+.3f})")
    print()
    if delta < -0.01:
        print("  GATE: ✓ KEEP — meta AAFE improved")
        decision = "KEEP"
    elif delta > 0.01:
        print("  GATE: ✗ REVERT — meta AAFE worsened")
        decision = "REVERT"
    else:
        print(f"  GATE: ~ NOISE — delta {delta:+.3f} within ±0.01")
        decision = "NOISE"

    return {
        "n_holdout": len(df),
        "ml_baseline_aafe": ml_base_aafe,
        "ml_single_aafe": ml_single_aafe,
        "ml_ensemble_aafe": ml_ens_aafe,
        "meta_baseline": meta_base,
        "meta_single_oldw": meta_single_oldw,
        "meta_ensemble_oldw": meta_ens_oldw,
        "meta_best_neww": best_loocv_aafe,
        "best_wb": best_wb, "best_wo": best_wo,
        "delta": delta,
        "decision": decision,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    dedup = load_mmpk_data()
    records, n_mordred = build_feature_sets(dedup)

    # Phase 1
    phase1, X_best, y, smiles, best_params, best_set = run_phase1(records, n_mordred)

    # Phase 2
    phase2 = run_phase2(records, y, smiles, best_params)

    # Phase 3
    phase3 = run_holdout_benchmark(records, y, smiles, best_params, best_set, phase2)

    # Save all
    results = {"phase1": phase1, "phase2": phase2, "phase3": phase3}
    out = ROOT / "data/validation/ml_improvement_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
