#!/usr/bin/env python3
"""Meta-Learner Tournament: 10 methods compete on holdout.

5 PK-domain + 5 other-domain-inspired approaches.
No engine/ML internals modified — post-hoc combination only.

Methods:
  M1  Isotonic Engine Calibration    (PK: non-parametric monotone calibration)
  M2  ER-Proxy Routing               (PK: extraction ratio → continuous weight)
  M3  Error Direction Classifier     (PK: binary XGBoost on sign(engine_error))
  M4  CLint-Stratified Meta          (PK: per-CLint-tertile weights)
  M5  AAFE-Direct Optimization       (PK: Nelder-Mead on training AAFE)
  M6  Quantile XGBoost               (Econometrics: median regression)
  M7  Local BMA                      (Weather: neighbor-based probabilistic weighting)
  M8  Caruana Ensemble Selection     (AutoML: greedy forward selection)
  M9  Disagreement-Sigmoid           (Signal Processing: smooth logistic routing)
  M10 Trimmed AAFE Meta              (Robust Stats: outlier-robust optimization)
"""
from __future__ import annotations

import json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_aafe(pred, obs):
    mask = (pred > 0) & (obs > 0)
    if mask.sum() == 0: return float("inf")
    return float(10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask]))))


def pct_2fold(pred, obs):
    mask = (pred > 0) & (obs > 0)
    if mask.sum() == 0: return 0.0
    fe = pred[mask] / obs[mask]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100)


def geo_blend(cmax_eng, cmax_ml, w_eng):
    """Geometric blend in log space: cmax = eng^w * ml^(1-w)."""
    le = np.log10(np.maximum(cmax_eng, 1e-10))
    lm = np.log10(np.maximum(cmax_ml, 1e-10))
    return 10 ** (w_eng * le + (1 - w_eng) * lm)


def scaffold_split(smiles_list, n_folds=5, seed=42):
    s2i = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception: sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scs): folds[i % n_folds].extend(s2i[sc])
    return folds


def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    if not inchi: return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading data...")

    # Training data with engine predictions + physicochemical features
    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")

    # Holdout exclusion (holdout only, not train reference)
    with open(ROOT / "data/reference/holdout.json") as f:
        hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cpk = json.load(f)
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: ho_iks.add(k)

    pbpk["ik14"] = pbpk["smiles"].apply(ik14)
    pbpk = pbpk.dropna(subset=["ik14"])
    pbpk = pbpk[~pbpk["ik14"].isin(ho_iks)].reset_index(drop=True)
    pbpk = pbpk[(pbpk["cmax_pbpk"] > 0) & (pbpk["cmax_obs"] > 0)].reset_index(drop=True)

    # Compound type
    pbpk["compound_type"] = "neutral"
    pbpk.loc[pbpk["is_base"] == 1.0, "compound_type"] = "base"
    pbpk.loc[pbpk["is_acid"] == 1.0, "compound_type"] = "acid"

    N_train = len(pbpk)
    log.info("  Training drugs: %d", N_train)

    # Compute ML OOF predictions via scaffold CV
    smiles_list = pbpk["smiles"].tolist()
    X_all = np.array([compute_features(s) for s in smiles_list], dtype=np.float32)
    y_all = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)

    xgb_params = dict(n_estimators=200, max_depth=4, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                      reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=4, verbosity=0)

    folds = scaffold_split(smiles_list)
    ml_oof_log = np.full(N_train, np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_all[train_idx], y_all[train_idx])
        ml_oof_log[val_idx] = m.predict(X_all[val_idx])

    ml_oof_cmax = 10 ** ml_oof_log * pbpk["dose_mg"].values
    ml_oof_cmax = np.maximum(ml_oof_cmax, 1e-10)

    # Full-model ML (for methods that need consistent scale with holdout)
    m_full = xgb.XGBRegressor(**xgb_params)
    m_full.fit(X_all, y_all)
    ml_full_cmax = 10 ** m_full.predict(X_all) * pbpk["dose_mg"].values

    # Holdout data
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)

    ho_smiles = {}
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ho_smiles[n] = e["smiles"]

    # Holdout features
    ho_feats = []
    for h in ho_preds:
        smi = ho_smiles.get(h["name"])
        if smi:
            try: ho_feats.append(compute_features(smi))
            except Exception: ho_feats.append(np.zeros(2057))
        else:
            ho_feats.append(np.zeros(2057))
    ho_X = np.array(ho_feats, dtype=np.float32)

    ho_engine = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_clf = np.array([h["cmax_clf"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])

    # Morgan FPs for neighbor computation
    train_fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), radius=2, nBits=2048))

    ho_fps = []
    for h in ho_preds:
        smi = ho_smiles.get(h["name"])
        mol = Chem.MolFromSmiles(smi) if smi else None
        ho_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), radius=2, nBits=2048))

    train = {
        "engine": pbpk["cmax_pbpk"].values,
        "obs": pbpk["cmax_obs"].values,
        "ml_oof": ml_oof_cmax,
        "ml_full": ml_full_cmax,
        "dose": pbpk["dose_mg"].values,
        "compound_type": pbpk["compound_type"].values,
        "fup": pbpk["fup"].values,
        "clint": pbpk["clint"].values,
        "logP": pbpk["logP"].values,
        "TPSA": pbpk["TPSA_norm"].values * 200,
        "MW": pbpk["MW_norm"].values * 600,
        "smiles": smiles_list,
        "X": X_all,
        "y": y_all,
        "fps": train_fps,
        "folds": folds,
        "N": N_train,
    }

    holdout = {
        "engine": ho_engine,
        "ml": ho_ml,
        "clf": ho_clf,
        "obs": ho_obs,
        "combined": ho_combined,
        "types": ho_types,
        "X": ho_X,
        "fps": ho_fps,
        "preds": ho_preds,
        "N": len(ho_preds),
    }

    return train, holdout


# ═══════════════════════════════════════════════════════════════════════════
# Method 1: Isotonic Engine Calibration
# ═══════════════════════════════════════════════════════════════════════════

def m1_isotonic(train, holdout):
    """Calibrate engine via isotonic regression, then blend with ML."""
    log_eng_tr = np.log10(np.maximum(train["engine"], 1e-10))
    log_obs_tr = np.log10(np.maximum(train["obs"], 1e-10))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(log_eng_tr, log_obs_tr)

    # Apply to holdout engine
    log_eng_ho = np.log10(np.maximum(holdout["engine"], 1e-10))
    cal_eng_ho = 10 ** iso.predict(log_eng_ho)

    # Blend calibrated engine with ML using standard weights
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo_blend(cal_eng_ho[i], holdout["ml"][i], w)

    # Also report calibrated engine standalone
    cal_eng_aafe = compute_aafe(cal_eng_ho, holdout["obs"])
    log.info("  M1: Calibrated engine standalone AAFE: %.3f (raw: %.3f)",
             cal_eng_aafe, compute_aafe(holdout["engine"], holdout["obs"]))

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 2: ER-Proxy Routing
# ═══════════════════════════════════════════════════════════════════════════

def m2_er_routing(train, holdout):
    """Use fup×CLint proxy for extraction ratio to modulate engine weight."""
    # On training: learn optimal weight as function of ER proxy
    # Using OOF predictions for honest evaluation
    er_tr = np.log10(train["fup"] * train["clint"] + 1e-6)

    # Split into 3 ER groups, find optimal w_engine for each
    tertiles = np.percentile(er_tr, [33.3, 66.7])
    groups = np.digitize(er_tr, tertiles)  # 0, 1, 2

    # For each group, grid search optimal w_engine
    best_w = {}
    for g in [0, 1, 2]:
        mask = (groups == g) & (train["compound_type"] == "base")
        if mask.sum() < 10:
            best_w[g] = 0.45
            continue
        best_aafe = float("inf")
        for w in np.arange(0, 0.8, 0.05):
            pred = geo_blend(train["engine"][mask], train["ml_oof"][mask], w)
            a = compute_aafe(pred, train["obs"][mask])
            if a < best_aafe:
                best_aafe = a
                best_w[g] = w
        log.info("  M2: ER group %d (base): optimal w=%.2f, AAFE=%.3f (n=%d)",
                 g, best_w[g], best_aafe, mask.sum())

    # Apply to holdout: need ER proxy for holdout drugs
    # Get fup and clint from engine predictions (via predict layer)
    # Approximate: use holdout engine Cmax as proxy for ER (higher Cmax ≈ lower ER)
    # Actually, we don't have fup/clint for holdout directly.
    # Use log(engine/ml) as a proxy: high disagreement ≈ high ER ≈ CLint sensitive
    log_disag = np.abs(np.log10(np.maximum(holdout["engine"], 1e-10)) -
                        np.log10(np.maximum(holdout["ml"], 1e-10)))

    # Map holdout drugs to ER groups based on disagreement tertiles
    # (approximate: high disagreement ≈ high ER)
    disag_tertiles = np.percentile(
        np.abs(np.log10(np.maximum(train["engine"], 1e-10)) -
               np.log10(np.maximum(train["ml_oof"], 1e-10))),
        [33.3, 66.7])
    ho_groups = np.digitize(log_disag, disag_tertiles)

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            w = best_w.get(ho_groups[i], 0.45)
        else:
            w = 0.00
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 3: Error Direction Classifier
# ═══════════════════════════════════════════════════════════════════════════

def m3_error_direction(train, holdout):
    """Binary classifier: does engine over-predict (1) or under-predict (0)?
    If over-predicting → lower engine weight. If under → raise it."""
    eng_fe = np.log10(train["engine"] / train["obs"])
    labels = (eng_fe > 0).astype(int)  # 1 = over-predict

    # Scaffold CV to estimate accuracy
    folds = train["folds"]
    cv_correct = 0
    cv_total = 0
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.5,
                                 random_state=42, n_jobs=4, verbosity=0,
                                 use_label_encoder=False, eval_metric="logloss")
        clf.fit(train["X"][train_idx], labels[train_idx])
        pred = clf.predict(train["X"][val_idx])
        cv_correct += (pred == labels[val_idx]).sum()
        cv_total += len(val_idx)
    cv_acc = cv_correct / cv_total
    log.info("  M3: CV accuracy: %.1f%% (majority class: %.1f%%)",
             cv_acc * 100, max(labels.mean(), 1 - labels.mean()) * 100)

    # Train full model
    clf_full = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.5,
                                  random_state=42, n_jobs=4, verbosity=0,
                                  use_label_encoder=False, eval_metric="logloss")
    clf_full.fit(train["X"], labels)
    prob_over = clf_full.predict_proba(holdout["X"])[:, 1]

    # Routing: when engine likely over-predicts, trust ML more
    # w_engine = w_base * (1 - prob_over) / 0.5  (scale so prob_over=0.5 → w unchanged)
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            w_base = 0.45
            # Scale engine weight by confidence in correct direction
            # prob_over > 0.5 → reduce engine weight
            # prob_over < 0.5 → increase engine weight
            w = w_base * 2 * (1 - prob_over[i])
            w = np.clip(w, 0, 0.9)
        else:
            w = 0.00
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 4: CLint-Stratified Meta
# ═══════════════════════════════════════════════════════════════════════════

def m4_clint_stratified(train, holdout):
    """Different meta-learner weights per predicted CLint range."""
    clint = train["clint"]
    tertiles = np.percentile(clint, [33.3, 66.7])
    groups = np.digitize(clint, tertiles)

    # For each group × base/non-base, find optimal w_engine
    best_w = {}
    for g in [0, 1, 2]:
        mask_base = (groups == g) & (train["compound_type"] == "base")
        if mask_base.sum() < 5:
            best_w[(g, True)] = 0.45
        else:
            best_a = float("inf")
            for w in np.arange(0, 0.9, 0.05):
                pred = geo_blend(train["engine"][mask_base], train["ml_oof"][mask_base], w)
                a = compute_aafe(pred, train["obs"][mask_base])
                if a < best_a:
                    best_a = a
                    best_w[(g, True)] = w
        best_w[(g, False)] = 0.00  # non-base always ML

    log.info("  M4: CLint tertile weights (base): low=%.2f, mid=%.2f, high=%.2f",
             best_w[(0, True)], best_w[(1, True)], best_w[(2, True)])

    # For holdout: estimate CLint group from engine prediction proxy
    # Higher engine Cmax / dose → lower CLint → group 0
    # Actually, we need CLint for holdout. Use the predict layer.
    # Faster: use a simple logP → CLint proxy
    # Or: map holdout to nearest training drug's CLint group
    ho_groups = np.zeros(holdout["N"], dtype=int)
    for i in range(holdout["N"]):
        sims = DataStructs.BulkTanimotoSimilarity(holdout["fps"][i], train["fps"])
        nn = np.argmax(sims)
        ho_groups[i] = np.digitize(train["clint"][nn], tertiles)

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        is_base = holdout["types"][i] == "base"
        w = best_w.get((ho_groups[i], is_base), 0.00)
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 5: AAFE-Direct Optimization (Nelder-Mead)
# ═══════════════════════════════════════════════════════════════════════════

def m5_aafe_direct(train, holdout):
    """Directly minimize AAFE via Nelder-Mead on meta-learner parameters."""
    eng_tr = train["engine"]
    ml_tr = train["ml_oof"]
    obs_tr = train["obs"]
    types_tr = train["compound_type"]
    base_mask_tr = types_tr == "base"

    def neg_aafe(params):
        w_base, w_other = params
        w_base = np.clip(w_base, 0, 1)
        w_other = np.clip(w_other, 0, 1)
        w = np.where(base_mask_tr, w_base, w_other)
        pred = geo_blend(eng_tr, ml_tr, w)
        return compute_aafe(pred, obs_tr)

    # Multi-start Nelder-Mead
    best_result = None
    best_aafe = float("inf")
    for w0_base in [0.0, 0.2, 0.45, 0.7]:
        for w0_other in [0.0, 0.15, 0.3]:
            res = minimize(neg_aafe, [w0_base, w0_other], method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 0.001, "fatol": 0.001})
            if res.fun < best_aafe:
                best_aafe = res.fun
                best_result = res

    w_base_opt = np.clip(best_result.x[0], 0, 1)
    w_other_opt = np.clip(best_result.x[1], 0, 1)
    log.info("  M5: AAFE-optimal weights: w_base=%.3f, w_other=%.3f (train AAFE=%.3f)",
             w_base_opt, w_other_opt, best_aafe)

    # Apply to holdout
    w = np.where(holdout["types"] == "base", w_base_opt, w_other_opt)
    preds = geo_blend(holdout["engine"], holdout["ml"], w)
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 6: Quantile XGBoost (Median Regression)
# ═══════════════════════════════════════════════════════════════════════════

def m6_quantile_xgb(train, holdout):
    """Median regression: directly optimizes L1 loss (AAFE-aligned)."""
    # Standalone quantile ML model
    qr_params = dict(n_estimators=200, max_depth=4, learning_rate=0.08,
                     subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                     reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                     n_jobs=4, verbosity=0,
                     objective="reg:quantileerror", quantile_alpha=0.5)

    # CV sanity check
    folds = train["folds"]
    cv_aafe_mean = []
    cv_aafe_median = []
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]

        # Standard (mean) model
        m_mean = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                   subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                   reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                                   n_jobs=4, verbosity=0)
        m_mean.fit(train["X"][train_idx], train["y"][train_idx])
        p_mean = 10 ** m_mean.predict(train["X"][val_idx]) * train["dose"][val_idx]
        cv_aafe_mean.append(compute_aafe(p_mean, train["obs"][val_idx]))

        # Quantile (median) model
        m_q = xgb.XGBRegressor(**qr_params)
        m_q.fit(train["X"][train_idx], train["y"][train_idx])
        p_q = 10 ** m_q.predict(train["X"][val_idx]) * train["dose"][val_idx]
        cv_aafe_median.append(compute_aafe(p_q, train["obs"][val_idx]))

    log.info("  M6: CV AAFE — mean loss: %.3f, quantile loss: %.3f (Δ=%.4f)",
             np.mean(cv_aafe_mean), np.mean(cv_aafe_median),
             np.mean(cv_aafe_median) - np.mean(cv_aafe_mean))

    # Full quantile model for holdout
    m_q_full = xgb.XGBRegressor(**qr_params)
    m_q_full.fit(train["X"], train["y"])
    ho_doses = np.array([
        next((cpk_drugs.get(h["name"], {}).get("dose_mg", 100.0)
              for cpk_drugs in [{}]), 100.0) for h in holdout["preds"]])
    # Get actual doses from clinical_pk
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cpk = json.load(f)
    ho_doses = []
    for h in holdout["preds"]:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses, dtype=float)

    q_pred = 10 ** m_q_full.predict(holdout["X"]) * ho_doses
    q_pred = np.maximum(q_pred, 1e-10)

    # Standalone quantile AAFE
    q_standalone = compute_aafe(q_pred, holdout["obs"])
    log.info("  M6: Holdout quantile standalone AAFE: %.3f (baseline ML: %.3f)",
             q_standalone, compute_aafe(holdout["ml"], holdout["obs"]))

    # Blend quantile ML with engine using standard meta weights
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo_blend(holdout["engine"][i], q_pred[i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 7: Local BMA (Bayesian Model Averaging)
# ═══════════════════════════════════════════════════════════════════════════

def m7_local_bma(train, holdout):
    """Per-drug weighting based on local error variance of neighbors."""
    K = 15

    # Engine fold errors for training drugs (no OOF issue)
    eng_fe = np.log10(train["engine"] / train["obs"])
    # ML fold errors (use OOF for honest estimate)
    ml_fe = np.log10(train["ml_oof"] / train["obs"])

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(holdout["fps"][i], train["fps"]))
        top_k = np.argsort(sims)[-K:]

        weights = sims[top_k]
        weights = np.maximum(weights, 0.01)

        # Local engine error variance (weighted)
        local_eng_fe = eng_fe[top_k]
        local_ml_fe = ml_fe[top_k]

        # Weighted variance
        wmean_eng = np.average(local_eng_fe, weights=weights)
        wvar_eng = np.average((local_eng_fe - wmean_eng) ** 2, weights=weights) + 0.01
        wmean_ml = np.average(local_ml_fe, weights=weights)
        wvar_ml = np.average((local_ml_fe - wmean_ml) ** 2, weights=weights) + 0.01

        # Local bias
        local_bias_eng = abs(wmean_eng)
        local_bias_ml = abs(wmean_ml)

        # BMA: weight inversely proportional to MSE = bias² + variance
        mse_eng = local_bias_eng ** 2 + wvar_eng
        mse_ml = local_bias_ml ** 2 + wvar_ml

        w_eng = (1 / mse_eng) / (1 / mse_eng + 1 / mse_ml)

        if holdout["types"][i] != "base":
            w_eng = min(w_eng, 0.15)  # cap non-base engine weight

        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w_eng)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 8: Caruana Ensemble Selection
# ═══════════════════════════════════════════════════════════════════════════

def m8_caruana(train, holdout):
    """Greedy forward selection from a library of blend combinations."""
    # Build library of blends on training data (using OOF ML)
    eng_tr = train["engine"]
    ml_tr = train["ml_oof"]
    obs_tr = train["obs"]

    # Library: various fixed blends (in log space)
    w_grid = np.arange(0, 1.05, 0.1)  # 0.0, 0.1, ..., 1.0
    library_train = {}
    for w in w_grid:
        library_train[f"w{w:.1f}"] = geo_blend(eng_tr, ml_tr, w)

    # Also add per-type best blends
    for ctype in ["base", "acid", "neutral"]:
        mask = train["compound_type"] == ctype
        best_a, best_w_ct = float("inf"), 0.0
        for w in np.arange(0, 0.9, 0.05):
            pred = geo_blend(eng_tr[mask], ml_tr[mask], w)
            a = compute_aafe(pred, obs_tr[mask])
            if a < best_a:
                best_a, best_w_ct = a, w
        # Create type-conditional blend
        type_pred = np.copy(ml_tr)
        type_pred[mask] = geo_blend(eng_tr[mask], ml_tr[mask], best_w_ct)
        library_train[f"type_{ctype}_{best_w_ct:.2f}"] = type_pred

    # Greedy selection with replacement
    ensemble_preds = []
    selected = []

    for round_i in range(10):
        best_name = None
        best_aafe_ens = float("inf")

        for name, pred in library_train.items():
            # Candidate ensemble: average of existing + new (in log space)
            trial = list(ensemble_preds) + [np.log10(np.maximum(pred, 1e-10))]
            avg_log = np.mean(trial, axis=0)
            avg_cmax = 10 ** avg_log
            a = compute_aafe(avg_cmax, obs_tr)
            if a < best_aafe_ens:
                best_aafe_ens = a
                best_name = name

        selected.append(best_name)
        ensemble_preds.append(np.log10(np.maximum(library_train[best_name], 1e-10)))
        if round_i < 5:
            log.info("  M8: Round %d: selected '%s', train AAFE=%.3f",
                     round_i + 1, best_name, best_aafe_ens)

    # Count selections
    from collections import Counter
    counts = Counter(selected)
    log.info("  M8: Final ensemble: %s", dict(counts))

    # Apply same ensemble to holdout
    ho_eng = holdout["engine"]
    ho_ml = holdout["ml"]

    library_ho = {}
    for w in w_grid:
        library_ho[f"w{w:.1f}"] = geo_blend(ho_eng, ho_ml, w)

    # Type-conditional blends need holdout compound types
    for ctype in ["base", "acid", "neutral"]:
        mask = holdout["types"] == ctype
        # Find the best_w for this type from training (parse from selected name)
        matching = [s for s in counts if s.startswith(f"type_{ctype}")]
        if matching:
            w_ct = float(matching[0].split("_")[-1])
            type_pred = np.copy(ho_ml)
            type_pred[mask] = geo_blend(ho_eng[mask], ho_ml[mask], w_ct)
            library_ho[matching[0]] = type_pred

    ho_ensemble = []
    for name in selected:
        if name in library_ho:
            ho_ensemble.append(np.log10(np.maximum(library_ho[name], 1e-10)))

    if ho_ensemble:
        avg_log = np.mean(ho_ensemble, axis=0)
        preds = 10 ** avg_log
    else:
        preds = ho_ml  # fallback

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 9: Disagreement-Sigmoid Routing
# ═══════════════════════════════════════════════════════════════════════════

def m9_disagreement_sigmoid(train, holdout):
    """Smooth logistic weighting based on |log(engine) - log(ML)|."""
    eng_tr = train["engine"]
    ml_tr = train["ml_oof"]
    obs_tr = train["obs"]
    base_tr = train["compound_type"] == "base"

    le = np.log10(np.maximum(eng_tr, 1e-10))
    lm = np.log10(np.maximum(ml_tr, 1e-10))
    disag = np.abs(le - lm)

    # Optimize sigmoid parameters: w = max_w * sigmoid(a + b * disag)
    # Only for base drugs (non-base always w=0)
    def objective(params):
        max_w, a, b = params
        max_w = np.clip(max_w, 0.01, 0.9)
        w = np.zeros(len(eng_tr))
        w[base_tr] = max_w / (1 + np.exp(-(a + b * disag[base_tr])))
        pred = geo_blend(eng_tr, ml_tr, w)
        return compute_aafe(pred, obs_tr)

    best_result = None
    best_val = float("inf")
    for mw0 in [0.3, 0.5, 0.7]:
        for a0 in [-1, 0, 1]:
            for b0 in [-2, -1, 0, 1]:
                res = minimize(objective, [mw0, a0, b0], method="Nelder-Mead",
                               options={"maxiter": 300})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res

    max_w, a, b = best_result.x
    max_w = np.clip(max_w, 0.01, 0.9)
    log.info("  M9: Sigmoid params: max_w=%.3f, a=%.3f, b=%.3f (train AAFE=%.3f)",
             max_w, a, b, best_val)

    # Apply to holdout
    le_ho = np.log10(np.maximum(holdout["engine"], 1e-10))
    lm_ho = np.log10(np.maximum(holdout["ml"], 1e-10))
    disag_ho = np.abs(le_ho - lm_ho)

    w_ho = np.zeros(holdout["N"])
    base_ho = holdout["types"] == "base"
    w_ho[base_ho] = max_w / (1 + np.exp(-(a + b * disag_ho[base_ho])))

    preds = geo_blend(holdout["engine"], holdout["ml"], w_ho)
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Method 10: Trimmed AAFE Meta (Robust Statistics)
# ═══════════════════════════════════════════════════════════════════════════

def m10_trimmed_aafe(train, holdout):
    """Optimize meta weights on trimmed AAFE (exclude top 10% worst)."""
    eng_tr = train["engine"]
    ml_tr = train["ml_oof"]
    obs_tr = train["obs"]
    base_tr = train["compound_type"] == "base"

    def trimmed_aafe(params, trim_pct=0.10):
        w_base, w_other = params
        w_base = np.clip(w_base, 0, 1)
        w_other = np.clip(w_other, 0, 1)
        w = np.where(base_tr, w_base, w_other)
        pred = geo_blend(eng_tr, ml_tr, w)
        abs_fe = np.abs(np.log10(np.maximum(pred, 1e-10) / np.maximum(obs_tr, 1e-10)))
        # Trim top 10% worst
        cutoff = np.percentile(abs_fe, (1 - trim_pct) * 100)
        mask = abs_fe <= cutoff
        if mask.sum() == 0: return float("inf")
        return float(10 ** np.mean(abs_fe[mask]))

    best_result = None
    best_val = float("inf")
    for w0_base in [0.0, 0.2, 0.45, 0.7]:
        for w0_other in [0.0, 0.15, 0.3]:
            res = minimize(trimmed_aafe, [w0_base, w0_other], method="Nelder-Mead",
                           options={"maxiter": 500})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res

    w_base_opt = np.clip(best_result.x[0], 0, 1)
    w_other_opt = np.clip(best_result.x[1], 0, 1)
    log.info("  M10: Trimmed AAFE weights: w_base=%.3f, w_other=%.3f (trimmed train=%.3f)",
             w_base_opt, w_other_opt, best_val)

    w = np.where(holdout["types"] == "base", w_base_opt, w_other_opt)
    preds = geo_blend(holdout["engine"], holdout["ml"], w)
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Tournament
# ═══════════════════════════════════════════════════════════════════════════

def run_tournament(train, holdout):
    log.info("=" * 70)
    log.info("TOURNAMENT: 10 Methods Compete")
    log.info("=" * 70)

    methods = [
        ("M1  Isotonic Engine Cal.", m1_isotonic),
        ("M2  ER-Proxy Routing", m2_er_routing),
        ("M3  Error Direction Clf", m3_error_direction),
        ("M4  CLint-Stratified", m4_clint_stratified),
        ("M5  AAFE-Direct Optim", m5_aafe_direct),
        ("M6  Quantile XGBoost", m6_quantile_xgb),
        ("M7  Local BMA", m7_local_bma),
        ("M8  Caruana Ensemble", m8_caruana),
        ("M9  Disagree-Sigmoid", m9_disagreement_sigmoid),
        ("M10 Trimmed AAFE", m10_trimmed_aafe),
    ]

    obs = holdout["obs"]
    types = holdout["types"]
    base_mask = types == "base"

    # Baselines
    baseline_ml = compute_aafe(holdout["ml"], obs)
    baseline_meta = compute_aafe(holdout["combined"], obs)
    baseline_meta_base = compute_aafe(holdout["combined"][base_mask], obs[base_mask])
    baseline_meta_nb = compute_aafe(holdout["combined"][~base_mask], obs[~base_mask])
    baseline_meta_p2f = pct_2fold(holdout["combined"], obs)

    results = []
    for name, fn in methods:
        log.info("─── %s ───", name)
        t0 = time.time()
        try:
            pred = fn(train, holdout)
            aafe_all = compute_aafe(pred, obs)
            aafe_base = compute_aafe(pred[base_mask], obs[base_mask])
            aafe_nb = compute_aafe(pred[~base_mask], obs[~base_mask])
            p2f = pct_2fold(pred, obs)
            elapsed = time.time() - t0
            delta = aafe_all - baseline_meta
            results.append((name, aafe_all, aafe_base, aafe_nb, p2f, delta, pred))
            log.info("  → AAFE=%.3f (Δ=%+.3f), base=%.3f, ~base=%.3f, %%2f=%.1f%% [%.1fs]",
                     aafe_all, delta, aafe_base, aafe_nb, p2f, elapsed)
        except Exception as e:
            log.error("  → FAILED: %s", e)
            results.append((name, float("inf"), float("inf"), float("inf"), 0.0, float("inf"), None))

    # ── Leaderboard ──
    log.info("")
    log.info("=" * 80)
    log.info("LEADERBOARD")
    log.info("=" * 80)
    header = f"{'Rank':<5} {'Method':<25} {'AAFE':>7} {'Δ':>8} {'Base':>7} {'~Base':>7} {'%2f':>6}"
    log.info(header)
    log.info("-" * 80)

    # Add baselines
    all_results = [
        ("--- Baseline ML", baseline_ml, compute_aafe(holdout["ml"][base_mask], obs[base_mask]),
         compute_aafe(holdout["ml"][~base_mask], obs[~base_mask]),
         pct_2fold(holdout["ml"], obs), baseline_ml - baseline_meta, None),
        ("--- Baseline Meta", baseline_meta, baseline_meta_base, baseline_meta_nb,
         baseline_meta_p2f, 0.0, None),
    ] + results

    sorted_results = sorted(all_results, key=lambda x: x[1])

    for rank, (name, aafe, ab, anb, p2f, delta, _) in enumerate(sorted_results, 1):
        marker = " ★" if delta < -0.02 else ""
        log.info(f"  {rank:<4} {name:<25} {aafe:>7.3f} {delta:>+8.3f} {ab:>7.3f} {anb:>7.3f} {p2f:>5.1f}%{marker}")

    # ── Winner analysis ──
    best = sorted_results[0]
    if best[5] < -0.05:
        log.info("\n  ★★★ MEANINGFUL IMPROVEMENT: %s (Δ=%.3f)", best[0], best[5])
    elif best[5] < -0.02:
        log.info("\n  ★ MARGINAL: %s (Δ=%.3f)", best[0], best[5])
    elif best[5] < 0:
        log.info("\n  ~ NOISE LEVEL: %s (Δ=%.3f)", best[0], best[5])
    else:
        log.info("\n  ✗ NO IMPROVEMENT — all methods ≥ baseline meta")

    # ── Error correlation of top methods ──
    method_preds = [(n, p) for n, _, _, _, _, _, p in results if p is not None]
    if len(method_preds) >= 3:
        log.info("\nError correlations (top 5 vs baseline meta):")
        fe_meta = np.log10(np.maximum(holdout["combined"], 1e-10) /
                           np.maximum(obs, 1e-10))
        for name, pred in method_preds[:5]:
            fe = np.log10(np.maximum(pred, 1e-10) / np.maximum(obs, 1e-10))
            r = np.corrcoef(fe_meta, fe)[0, 1]
            log.info("  %s: r=%.3f with baseline", name, r)

    # ── Save results ──
    out = {
        "baseline_ml_aafe": baseline_ml,
        "baseline_meta_aafe": baseline_meta,
        "methods": [
            {"name": name, "aafe": aafe, "aafe_base": ab, "aafe_nonbase": anb,
             "pct2fold": p2f, "delta": delta}
            for name, aafe, ab, anb, p2f, delta, _ in results
        ],
        "leaderboard": [
            {"rank": rank, "name": name, "aafe": aafe, "delta": delta}
            for rank, (name, aafe, _, _, _, delta, _) in enumerate(sorted_results, 1)
        ],
    }
    outpath = ROOT / "data/validation/meta_tournament_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    log.info("\nResults saved to %s", outpath)

    return results


# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    train, holdout = load_data()
    results = run_tournament(train, holdout)
    log.info("\nTotal time: %.1f seconds", time.time() - t_start)


if __name__ == "__main__":
    main()
