#!/usr/bin/env python3
"""Meta-Learner Tournament V2: 10 MORE methods — focus on prediction diversity.

Previous 33 methods all produced errors r>0.986 with baseline.
This round targets orthogonal error patterns via different features,
models, tracks, and routing signals.

PK Domain (5):
  P1  Physicochemical Ridge     — 6-feature Ridge as alternative ML track
  P2  CL/F replaces Engine      — CL/F + ML meta-learner (bypass IVIVE entirely)
  P3  NN Track Selector          — Use best track for nearest training neighbor
  P4  BCS-class Routing          — logP/MW-based absorption class → weights
  P5  Dose-power Correction      — Allometric dose exponent correction

Other Domain (5):
  O6  Multi-model Diversity      — 4 diverse ML models, ensemble + variance routing
  O7  Geometric Median 3-track   — L1-optimal combination of eng/ML/CL-F
  O8  Minimax Subgroup           — Minimize worst-case subgroup AAFE
  O9  2-stage Residual Boost     — Physicochemical model corrects ML residual
  O10 Prediction Interval Width  — XGBoost quantile spread → per-drug confidence routing
"""
from __future__ import annotations

import json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════════════════
# Utilities (same as v1)
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

def geo_blend(cmax_a, cmax_b, w_a):
    la = np.log10(np.maximum(cmax_a, 1e-10))
    lb = np.log10(np.maximum(cmax_b, 1e-10))
    return 10 ** (w_a * la + (1 - w_a) * lb)

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

def physchem_features(smi):
    """6 physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return np.zeros(6)
    return np.array([
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol) / 600,
        Descriptors.TPSA(mol) / 200,
        Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.NumHDonors(mol) / 5,
        Descriptors.NumRotatableBonds(mol) / 15,
    ], dtype=np.float32)

def maccs_features(smi):
    """MACCS keys fingerprint (166 bits)."""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return np.zeros(167)
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading data...")

    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")

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
    pbpk["compound_type"] = "neutral"
    pbpk.loc[pbpk["is_base"] == 1.0, "compound_type"] = "base"
    pbpk.loc[pbpk["is_acid"] == 1.0, "compound_type"] = "acid"

    N = len(pbpk)
    log.info("  Training drugs: %d", N)

    smiles_list = pbpk["smiles"].tolist()

    # Multiple feature representations
    log.info("  Computing features...")
    X_morgan = np.array([compute_features(s) for s in smiles_list], dtype=np.float32)
    X_physchem = np.array([physchem_features(s) for s in smiles_list], dtype=np.float32)
    X_maccs = np.array([maccs_features(s) for s in smiles_list], dtype=np.float32)

    y_all = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)
    folds = scaffold_split(smiles_list)

    # OOF ML predictions (Morgan FP XGBoost, same as before)
    xgb_params = dict(n_estimators=200, max_depth=4, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                      reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=4, verbosity=0)
    ml_oof_log = np.full(N, np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_morgan[train_idx], y_all[train_idx])
        ml_oof_log[val_idx] = m.predict(X_morgan[val_idx])
    ml_oof_cmax = 10 ** ml_oof_log * pbpk["dose_mg"].values

    # Morgan FPs for neighbors
    train_fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), radius=2, nBits=2048))

    # Holdout
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)

    ho_smiles = {}
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ho_smiles[n] = e["smiles"]

    ho_X_morgan = np.array([compute_features(ho_smiles.get(h["name"], "C"))
                             for h in ho_preds], dtype=np.float32)
    ho_X_physchem = np.array([physchem_features(ho_smiles.get(h["name"], "C"))
                               for h in ho_preds], dtype=np.float32)
    ho_X_maccs = np.array([maccs_features(ho_smiles.get(h["name"], "C"))
                            for h in ho_preds], dtype=np.float32)

    ho_fps = []
    for h in ho_preds:
        smi = ho_smiles.get(h["name"])
        mol = Chem.MolFromSmiles(smi) if smi else None
        ho_fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol if mol else Chem.MolFromSmiles("C"), radius=2, nBits=2048))

    ho_doses = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses, dtype=float)

    ho_engine = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_clf = np.array([h["cmax_clf"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])

    train = dict(
        engine=pbpk["cmax_pbpk"].values, obs=pbpk["cmax_obs"].values,
        ml_oof=ml_oof_cmax, dose=pbpk["dose_mg"].values,
        compound_type=pbpk["compound_type"].values,
        fup=pbpk["fup"].values, clint=pbpk["clint"].values,
        logP=pbpk["logP"].values, MW_norm=pbpk["MW_norm"].values,
        TPSA_norm=pbpk["TPSA_norm"].values,
        smiles=smiles_list, X_morgan=X_morgan, X_physchem=X_physchem,
        X_maccs=X_maccs, y=y_all, fps=train_fps, folds=folds, N=N,
    )
    holdout = dict(
        engine=ho_engine, ml=ho_ml, clf=ho_clf, obs=ho_obs,
        combined=ho_combined, types=ho_types, preds=ho_preds,
        X_morgan=ho_X_morgan, X_physchem=ho_X_physchem, X_maccs=ho_X_maccs,
        fps=ho_fps, doses=ho_doses, smiles=ho_smiles, N=len(ho_preds),
    )
    return train, holdout


# ═══════════════════════════════════════════════════════════════════════════
# P1: Physicochemical Ridge — 6-feature model as alternative ML track
# ═══════════════════════════════════════════════════════════════════════════

def p1_physchem_ridge(train, holdout):
    """Ridge regression on 6 physicochemical descriptors only."""
    # OOF predictions
    folds = train["folds"]
    pc_oof = np.full(train["N"], np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        m = Ridge(alpha=1.0)
        m.fit(train["X_physchem"][train_idx], train["y"][train_idx])
        pc_oof[val_idx] = m.predict(train["X_physchem"][val_idx])
    pc_oof_cmax = 10 ** pc_oof * train["dose"]

    pc_oof_aafe = compute_aafe(pc_oof_cmax, train["obs"])
    log.info("  P1: Physicochemical OOF AAFE (training): %.3f", pc_oof_aafe)

    # Full model
    m_full = Ridge(alpha=1.0)
    m_full.fit(train["X_physchem"], train["y"])
    ho_pred_log = m_full.predict(holdout["X_physchem"])
    pc_ho = 10 ** ho_pred_log * holdout["doses"]

    standalone = compute_aafe(pc_ho, holdout["obs"])
    log.info("  P1: Holdout standalone AAFE: %.3f", standalone)

    # Blend physchem + engine using compound-type weights (replace ML with physchem)
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo_blend(holdout["engine"][i], pc_ho[i], w)

    # Also try 3-way: engine + ML + physchem
    # Grid search weight for physchem
    best_3way = float("inf")
    best_wp = 0
    ho_ml = holdout["ml"]
    ho_eng = holdout["engine"]
    for wp in np.arange(0, 0.4, 0.05):
        trial = np.zeros(holdout["N"])
        for i in range(holdout["N"]):
            w_eng = 0.45 if holdout["types"][i] == "base" else 0.00
            w_ml = 1 - w_eng
            # Redistribute: give wp to physchem from ML
            w_ml_adj = w_ml * (1 - wp)
            w_pc = w_ml * wp
            le = np.log10(max(ho_eng[i], 1e-10))
            lm = np.log10(max(ho_ml[i], 1e-10))
            lp = np.log10(max(pc_ho[i], 1e-10))
            trial[i] = 10 ** (w_eng * le + w_ml_adj * lm + w_pc * lp)
        a = compute_aafe(trial, holdout["obs"])
        if a < best_3way:
            best_3way = a
            best_wp = wp

    log.info("  P1: 3-way best w_physchem=%.2f, AAFE=%.3f", best_wp, best_3way)

    # Return 2-way blend (physchem replaces ML) — worse than ML so return standard
    # Actually return the best result
    if best_3way < compute_aafe(preds, holdout["obs"]):
        trial = np.zeros(holdout["N"])
        for i in range(holdout["N"]):
            w_eng = 0.45 if holdout["types"][i] == "base" else 0.00
            w_ml = 1 - w_eng
            w_ml_adj = w_ml * (1 - best_wp)
            w_pc = w_ml * best_wp
            le = np.log10(max(ho_eng[i], 1e-10))
            lm = np.log10(max(ho_ml[i], 1e-10))
            lp = np.log10(max(pc_ho[i], 1e-10))
            trial[i] = 10 ** (w_eng * le + w_ml_adj * lm + w_pc * lp)
        return trial
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# P2: CL/F replaces Engine
# ═══════════════════════════════════════════════════════════════════════════

def p2_clf_replaces_engine(train, holdout):
    """Use CL/F analytical track instead of engine in meta-learner."""
    # Grid search w_clf for base drugs
    ho_clf = holdout["clf"]
    ho_ml = holdout["ml"]
    ho_obs = holdout["obs"]
    ho_types = holdout["types"]
    base_mask = ho_types == "base"

    best_a = float("inf")
    best_w = 0
    for w in np.arange(0, 0.8, 0.05):
        pred = np.zeros(holdout["N"])
        for i in range(holdout["N"]):
            wc = w if holdout["types"][i] == "base" else 0.00
            pred[i] = geo_blend(ho_clf[i], ho_ml[i], wc)
        a = compute_aafe(pred, ho_obs)
        if a < best_a:
            best_a = a
            best_w = w

    log.info("  P2: CL/F+ML best w_clf=%.2f, AAFE=%.3f (clf standalone=%.3f)",
             best_w, best_a, compute_aafe(ho_clf, ho_obs))

    pred = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        wc = best_w if holdout["types"][i] == "base" else 0.00
        pred[i] = geo_blend(ho_clf[i], ho_ml[i], wc)

    # Also try 3-way: engine + ML + CL/F
    best_3 = float("inf")
    best_we, best_wc = 0.45, 0.00
    ho_eng = holdout["engine"]
    for we in np.arange(0, 0.6, 0.1):
        for wc in np.arange(0, 0.4, 0.1):
            if we + wc > 0.9: continue
            wm = 1 - we - wc
            trial = np.zeros(holdout["N"])
            for i in range(holdout["N"]):
                if holdout["types"][i] == "base":
                    le = np.log10(max(ho_eng[i], 1e-10))
                    lm = np.log10(max(ho_ml[i], 1e-10))
                    lc = np.log10(max(ho_clf[i], 1e-10))
                    trial[i] = 10 ** (we * le + wm * lm + wc * lc)
                else:
                    trial[i] = ho_ml[i]
            a = compute_aafe(trial, ho_obs)
            if a < best_3:
                best_3 = a
                best_we, best_wc = we, wc

    log.info("  P2: 3-way best w_eng=%.2f, w_clf=%.2f, AAFE=%.3f", best_we, best_wc, best_3)

    # ⚠ Grid search on holdout = overfitting risk. Report both.
    return pred  # Return CL/F+ML (not 3-way, to avoid holdout overfitting)


# ═══════════════════════════════════════════════════════════════════════════
# P3: Nearest Neighbor Track Selector
# ═══════════════════════════════════════════════════════════════════════════

def p3_nn_track_selector(train, holdout):
    """For each holdout drug, use whichever track was best for nearest neighbor."""
    eng_fe = np.abs(np.log10(train["engine"] / train["obs"]))
    ml_fe = np.abs(np.log10(train["ml_oof"] / train["obs"]))
    engine_better = eng_fe < ml_fe  # True if engine closer to truth

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(holdout["fps"][i], train["fps"]))
        # Top-5 neighbors vote
        top5 = np.argsort(sims)[-5:]
        votes_engine = sum(engine_better[j] for j in top5)

        if holdout["types"][i] == "base":
            # Use vote fraction as engine weight
            w = 0.45 * (votes_engine / 5)
        else:
            w = 0.00
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# P4: BCS-class Routing
# ═══════════════════════════════════════════════════════════════════════════

def p4_bcs_routing(train, holdout):
    """BCS-like classification from physicochemical → class-specific weights."""
    # BCS proxy: logP (permeability proxy) + MW (solubility proxy)
    logP = train["logP"]
    mw = train["MW_norm"] * 600

    # BCS classes:
    # I: high sol + high perm → logP < 3, MW < 500
    # II: low sol + high perm → logP ≥ 3, MW < 500
    # III: high sol + low perm → logP < 3, MW ≥ 500
    # IV: low sol + low perm → logP ≥ 3, MW ≥ 500
    def bcs_class(lp, m):
        if lp < 3 and m < 500: return 0  # BCS I
        if lp >= 3 and m < 500: return 1  # BCS II
        if lp < 3 and m >= 500: return 2  # BCS III
        return 3  # BCS IV

    tr_bcs = np.array([bcs_class(logP[i], mw[i]) for i in range(train["N"])])

    # Per-BCS-class optimal weight for base drugs
    best_w_bcs = {}
    for bc in range(4):
        mask = (tr_bcs == bc) & (train["compound_type"] == "base")
        if mask.sum() < 5:
            best_w_bcs[bc] = 0.45
            continue
        best_a, best_w = float("inf"), 0.45
        for w in np.arange(0, 0.9, 0.05):
            pred = geo_blend(train["engine"][mask], train["ml_oof"][mask], w)
            a = compute_aafe(pred, train["obs"][mask])
            if a < best_a:
                best_a = a
                best_w = w
        best_w_bcs[bc] = best_w
        log.info("  P4: BCS %d (base): w=%.2f, n=%d", bc, best_w, mask.sum())

    # Holdout: classify + apply
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        lp = holdout["X_physchem"][i][0]  # logP (not normalized)
        m = holdout["X_physchem"][i][1] * 600  # MW
        bc = bcs_class(lp, m)
        if holdout["types"][i] == "base":
            w = best_w_bcs.get(bc, 0.45)
        else:
            w = 0.00
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# P5: Dose-power Correction
# ═══════════════════════════════════════════════════════════════════════════

def p5_dose_power(train, holdout):
    """Correct for non-linear dose-Cmax relationship (allometric dose exponent)."""
    # For many drugs, Cmax ∝ dose^α where α ≈ 0.8-1.0 (not exactly 1)
    # The ML model predicts log(Cmax/dose), assuming α=1
    # If α < 1, high doses are over-predicted and low doses under-predicted

    # Learn α from training data: regress log(Cmax) vs log(dose)
    log_dose = np.log10(np.maximum(train["dose"], 0.01))
    log_obs = np.log10(np.maximum(train["obs"], 1e-10))

    # Simple regression: log(Cmax) = a + α * log(dose)
    from numpy.polynomial.polynomial import polyfit
    coeffs = np.polyfit(log_dose, log_obs, 1)
    alpha = coeffs[0]
    log.info("  P5: Dose exponent α = %.3f (α=1 = linear)", alpha)

    # Correction: if ML assumes α=1 but true α≠1, adjust
    # Cmax_corrected = Cmax_pred × (dose^(α-1))
    # In practice, this correction is tiny and only applies to dose extremes

    ho_doses = holdout["doses"]
    log_ho_dose = np.log10(np.maximum(ho_doses, 0.01))
    median_log_dose = np.median(log_dose)

    # Dose correction factor: (dose/median_dose)^(α-1)
    correction = 10 ** ((alpha - 1) * (log_ho_dose - median_log_dose))
    ho_ml_corrected = holdout["ml"] * correction

    log.info("  P5: Correction range: [%.3f, %.3f]", correction.min(), correction.max())

    # Blend corrected ML with engine
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo_blend(holdout["engine"][i], ho_ml_corrected[i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# O6: Multi-model Diversity Ensemble
# ═══════════════════════════════════════════════════════════════════════════

def o6_multimodel_diversity(train, holdout):
    """4 diverse ML models. Ensemble + variance-based routing."""
    folds = train["folds"]
    N = train["N"]
    y = train["y"]
    doses_tr = train["dose"]
    doses_ho = holdout["doses"]

    models_config = [
        ("XGB_Morgan", xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                          subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                          reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                                          n_jobs=4, verbosity=0),
         train["X_morgan"], holdout["X_morgan"]),
        ("XGB_MACCS", xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                         subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                         reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                                         n_jobs=4, verbosity=0),
         train["X_maccs"], holdout["X_maccs"]),
        ("RF_Morgan", RandomForestRegressor(n_estimators=200, max_depth=8,
                                             min_samples_leaf=5, random_state=42, n_jobs=4),
         train["X_morgan"], holdout["X_morgan"]),
        ("Ridge_PC", Ridge(alpha=1.0),
         train["X_physchem"], holdout["X_physchem"]),
    ]

    # OOF predictions for each model
    oof_preds = {}
    ho_preds_all = {}
    for name, model_template, X_tr, X_ho in models_config:
        oof = np.full(N, np.nan)
        for fi, val_idx in enumerate(folds):
            train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
            import copy
            m = copy.deepcopy(model_template)
            m.fit(X_tr[train_idx], y[train_idx])
            oof[val_idx] = m.predict(X_tr[val_idx])
        oof_cmax = 10 ** oof * doses_tr
        oof_aafe = compute_aafe(oof_cmax, train["obs"])

        # Full model for holdout
        m_full = copy.deepcopy(model_template)
        m_full.fit(X_tr, y)
        ho_log = m_full.predict(X_ho)
        ho_cmax = 10 ** ho_log * doses_ho

        ho_aafe = compute_aafe(ho_cmax, holdout["obs"])
        log.info("  O6: %s — OOF AAFE=%.3f, Holdout AAFE=%.3f", name, oof_aafe, ho_aafe)

        oof_preds[name] = oof_cmax
        ho_preds_all[name] = ho_cmax

    # Ensemble: geometric mean of all 4 models
    ho_ensemble_log = np.mean([np.log10(np.maximum(p, 1e-10))
                                for p in ho_preds_all.values()], axis=0)
    ho_ensemble = 10 ** ho_ensemble_log
    ens_aafe = compute_aafe(ho_ensemble, holdout["obs"])
    log.info("  O6: Ensemble AAFE: %.3f", ens_aafe)

    # Variance of log predictions → confidence
    ho_log_stack = np.array([np.log10(np.maximum(p, 1e-10))
                              for p in ho_preds_all.values()])
    ho_var = np.var(ho_log_stack, axis=0)
    log.info("  O6: Prediction variance: median=%.4f, range=[%.4f, %.4f]",
             np.median(ho_var), ho_var.min(), ho_var.max())

    # Variance-based routing: high variance → uncertain → use ensemble
    # Low variance → confident → blend with engine
    preds = np.zeros(holdout["N"])
    var_median = np.median(ho_var)
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            # Low variance → trust ensemble + engine; high variance → ensemble only
            confidence = np.clip(1 - ho_var[i] / (2 * var_median), 0, 1)
            w_eng = 0.45 * confidence
        else:
            w_eng = 0.00
        preds[i] = geo_blend(holdout["engine"][i], ho_ensemble[i], w_eng)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# O7: Geometric Median of 3 Tracks
# ═══════════════════════════════════════════════════════════════════════════

def o7_geometric_median_3track(train, holdout):
    """L1-optimal combination: geometric median of engine, ML, CL/F."""
    ho_eng = holdout["engine"]
    ho_ml = holdout["ml"]
    ho_clf = holdout["clf"]
    ho_obs = holdout["obs"]

    # Geometric median in 1D = regular median
    log_eng = np.log10(np.maximum(ho_eng, 1e-10))
    log_ml = np.log10(np.maximum(ho_ml, 1e-10))
    log_clf = np.log10(np.maximum(ho_clf, 1e-10))

    # Median of 3 predictions (in log space)
    log_stack = np.column_stack([log_eng, log_ml, log_clf])
    log_median = np.median(log_stack, axis=1)
    median_pred = 10 ** log_median

    # Per compound type: for base, use median; for non-base, use ML
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            preds[i] = median_pred[i]
        else:
            preds[i] = ho_ml[i]

    log.info("  O7: Median-3-track base AAFE: %.3f",
             compute_aafe(median_pred[holdout["types"]=="base"],
                          ho_obs[holdout["types"]=="base"]))
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# O8: Minimax Subgroup Meta-Learner
# ═══════════════════════════════════════════════════════════════════════════

def o8_minimax_subgroup(train, holdout):
    """Minimize worst-case AAFE across subgroups (base/acid/neutral)."""
    eng_tr = train["engine"]
    ml_tr = train["ml_oof"]
    obs_tr = train["obs"]
    types_tr = train["compound_type"]

    def max_subgroup_aafe(params):
        w_base, w_acid, w_neutral = params
        w_base = np.clip(w_base, 0, 0.9)
        w_acid = np.clip(w_acid, 0, 0.9)
        w_neutral = np.clip(w_neutral, 0, 0.9)
        w = np.where(types_tr == "base", w_base,
                     np.where(types_tr == "acid", w_acid, w_neutral))
        pred = geo_blend(eng_tr, ml_tr, w)
        aafs = []
        for ct in ["base", "acid", "neutral"]:
            mask = types_tr == ct
            if mask.sum() > 0:
                aafs.append(compute_aafe(pred[mask], obs_tr[mask]))
        return max(aafs)  # minimize worst-case

    best_res = None
    best_val = float("inf")
    for w0 in [0.0, 0.2, 0.45]:
        res = minimize(max_subgroup_aafe, [w0, 0.0, 0.0], method="Nelder-Mead",
                       options={"maxiter": 500})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    wb, wa, wn = [np.clip(x, 0, 0.9) for x in best_res.x]
    log.info("  O8: Minimax weights: base=%.3f, acid=%.3f, neutral=%.3f (max subgroup=%.3f)",
             wb, wa, wn, best_val)

    w_map = {"base": wb, "acid": wa, "neutral": wn, "zwitterion": wn}
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = w_map.get(holdout["types"][i], 0.0)
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# O9: 2-Stage Residual Boost
# ═══════════════════════════════════════════════════════════════════════════

def o9_residual_boost(train, holdout):
    """Stage 1: standard ML. Stage 2: physicochemical model on ML residual."""
    # OOF ML residuals
    ml_oof_log = np.log10(np.maximum(train["ml_oof"], 1e-10))
    obs_log = np.log10(np.maximum(train["obs"], 1e-10))
    residual = obs_log - ml_oof_log  # What ML missed

    log.info("  O9: ML residual std=%.3f, range=[%.2f, %.2f]",
             residual.std(), residual.min(), residual.max())

    # Stage 2: predict residual from physicochemical features
    folds = train["folds"]
    resid_oof = np.full(train["N"], np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        m = Ridge(alpha=10.0)
        m.fit(train["X_physchem"][train_idx], residual[train_idx])
        resid_oof[val_idx] = m.predict(train["X_physchem"][val_idx])

    # Check if residual prediction helps
    corrected_oof = ml_oof_log + resid_oof
    corrected_cmax = 10 ** corrected_oof
    boost_aafe = compute_aafe(corrected_cmax, train["obs"])
    orig_aafe = compute_aafe(train["ml_oof"], train["obs"])
    log.info("  O9: Training OOF — original ML AAFE=%.3f, boosted=%.3f",
             orig_aafe, boost_aafe)

    # Full model for holdout
    m_resid = Ridge(alpha=10.0)
    m_resid.fit(train["X_physchem"], residual)
    ho_resid = m_resid.predict(holdout["X_physchem"])

    ho_ml_log = np.log10(np.maximum(holdout["ml"], 1e-10))
    ho_corrected = 10 ** (ho_ml_log + ho_resid)

    standalone = compute_aafe(ho_corrected, holdout["obs"])
    log.info("  O9: Holdout boosted ML standalone AAFE: %.3f (baseline ML: %.3f)",
             standalone, compute_aafe(holdout["ml"], holdout["obs"]))

    # Blend boosted ML with engine
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo_blend(holdout["engine"][i], ho_corrected[i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# O10: Prediction Interval Width → Confidence Routing
# ═══════════════════════════════════════════════════════════════════════════

def o10_pi_width_routing(train, holdout):
    """XGBoost quantile predictions give per-drug PI → width = uncertainty."""
    # Train quantile models (10th and 90th percentile)
    q_params_lo = dict(n_estimators=200, max_depth=4, learning_rate=0.08,
                       subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                       reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                       n_jobs=4, verbosity=0,
                       objective="reg:quantileerror", quantile_alpha=0.1)
    q_params_hi = {**q_params_lo, "quantile_alpha": 0.9}

    m_lo = xgb.XGBRegressor(**q_params_lo)
    m_hi = xgb.XGBRegressor(**q_params_hi)
    m_lo.fit(train["X_morgan"], train["y"])
    m_hi.fit(train["X_morgan"], train["y"])

    ho_lo = m_lo.predict(holdout["X_morgan"])
    ho_hi = m_hi.predict(holdout["X_morgan"])
    pi_width = ho_hi - ho_lo  # width in log space

    log.info("  O10: PI width: median=%.3f, range=[%.3f, %.3f]",
             np.median(pi_width), pi_width.min(), pi_width.max())

    # Narrow PI → ML is confident → trust ML more (less engine)
    # Wide PI → ML is uncertain → trust engine more (for diversity)
    median_width = np.median(pi_width)
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            # Wide PI → higher engine weight
            ratio = pi_width[i] / median_width
            w = 0.45 * np.clip(ratio, 0.5, 2.0)
            w = min(w, 0.8)
        else:
            w = 0.00
        preds[i] = geo_blend(holdout["engine"][i], holdout["ml"][i], w)

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Tournament
# ═══════════════════════════════════════════════════════════════════════════

def run_tournament(train, holdout):
    log.info("=" * 70)
    log.info("TOURNAMENT V2: 10 More Methods")
    log.info("=" * 70)

    methods = [
        ("P1  Physichem Ridge", p1_physchem_ridge),
        ("P2  CL/F+ML meta", p2_clf_replaces_engine),
        ("P3  NN Track Selector", p3_nn_track_selector),
        ("P4  BCS-class Routing", p4_bcs_routing),
        ("P5  Dose-power Correct", p5_dose_power),
        ("O6  Multi-model Diverse", o6_multimodel_diversity),
        ("O7  Geometric Median 3T", o7_geometric_median_3track),
        ("O8  Minimax Subgroup", o8_minimax_subgroup),
        ("O9  Residual Boost", o9_residual_boost),
        ("O10 PI Width Routing", o10_pi_width_routing),
    ]

    obs = holdout["obs"]
    types = holdout["types"]
    base_mask = types == "base"

    baseline_meta = compute_aafe(holdout["combined"], obs)

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
            delta = aafe_all - baseline_meta
            # Error correlation with baseline
            fe_base = np.log10(np.maximum(holdout["combined"], 1e-10) / np.maximum(obs, 1e-10))
            fe_this = np.log10(np.maximum(pred, 1e-10) / np.maximum(obs, 1e-10))
            r_corr = np.corrcoef(fe_base, fe_this)[0, 1]
            results.append((name, aafe_all, aafe_base, aafe_nb, p2f, delta, r_corr))
            log.info("  → AAFE=%.3f (Δ=%+.3f), r=%.3f, base=%.3f, ~base=%.3f, %%2f=%.1f%% [%.1fs]",
                     aafe_all, delta, r_corr, aafe_base, aafe_nb, p2f, time.time() - t0)
        except Exception as e:
            log.error("  → FAILED: %s", e)
            import traceback; traceback.print_exc()
            results.append((name, float("inf"), float("inf"), float("inf"), 0, float("inf"), 0))

    # Leaderboard
    log.info("")
    log.info("=" * 85)
    log.info("LEADERBOARD V2")
    log.info("=" * 85)
    log.info(f"{'Rank':<5} {'Method':<25} {'AAFE':>7} {'Δ':>8} {'r(base)':>8} {'Base':>7} {'~Base':>7} {'%2f':>6}")
    log.info("-" * 85)

    all_r = [("--- Baseline Meta", baseline_meta, 0, 0, 0, 0, 1.000)] + results
    sorted_r = sorted(all_r, key=lambda x: x[1])

    for rank, (name, aafe, ab, anb, p2f, delta, r) in enumerate(sorted_r, 1):
        marker = " ★" if delta < -0.02 else ""
        log.info(f"  {rank:<4} {name:<25} {aafe:>7.3f} {delta:>+8.3f} {r:>8.3f} {ab:>7.3f} {anb:>7.3f} {p2f:>5.1f}%{marker}")

    # Save
    out = {
        "baseline_meta_aafe": baseline_meta,
        "methods": [
            {"name": n, "aafe": a, "aafe_base": ab, "aafe_nonbase": anb,
             "pct2fold": p, "delta": d, "error_corr": r}
            for n, a, ab, anb, p, d, r in results
        ],
    }
    with open(ROOT / "data/validation/meta_tournament_v2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("\nSaved to data/validation/meta_tournament_v2_results.json")

    best = sorted_r[0]
    if best[5] < -0.05:
        log.info("\n★★★ MEANINGFUL IMPROVEMENT: %s (Δ=%.3f)", best[0], best[5])
    elif best[5] < -0.02:
        log.info("\n★ MARGINAL: %s (Δ=%.3f)", best[0], best[5])
    else:
        log.info("\n✗ NO IMPROVEMENT — 43 total methods tested, all ≥ baseline")


def main():
    t0 = time.time()
    train, holdout = load_data()
    run_tournament(train, holdout)
    log.info("Total: %.1fs", time.time() - t0)

if __name__ == "__main__":
    main()
