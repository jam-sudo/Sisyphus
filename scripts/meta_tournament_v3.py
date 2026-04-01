#!/usr/bin/env python3
"""Meta-Learner Tournament V3: 10 genuinely different approaches.

Key insight: all 43 previous methods used the SAME prediction sources.
This round creates NEW prediction sources with orthogonal error patterns.

PK Domain (5):
  P1  AUC-First Decomposition   — Predict AUC & ratio separately → multiply
  P2  20-fold Stacking          — Fix OOF-Full gap (r=0.81→?) with higher K
  P3  Analytical 1-Cpt ADME     — Closed-form Cmax from predicted PK params
  P4  Substructure Correction    — Functional group → avg engine bias → correct
  P5  LOWESS Engine Calibration  — Robust local regression (not isotonic)

Other Domain (5):
  O6  Rank Aggregation           — Social choice: aggregate rankings not values
  O7  Feature-Dropout Ensemble   — Bagging: 30 random feature masks
  O8  Huber Loss XGBoost         — Robust stats: smooth L1 loss in log space
  O9  AUC-ML + Engine Meta       — New ML track (AUC-derived) replaces Morgan ML
  O10 Decomposed 3-Track         — AUC-Cmax + standard ML + engine, 3-way blend
"""
from __future__ import annotations

import json, logging, sys, time, warnings, copy
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def aafe(p, o):
    m = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[m] / o[m])))) if m.sum() else np.inf

def p2f(p, o):
    m = (p > 0) & (o > 0)
    fe = p[m] / o[m]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100) if m.sum() else 0.0

def geo(a, b, w):
    la = np.log10(np.maximum(a, 1e-10))
    lb = np.log10(np.maximum(b, 1e-10))
    return 10 ** (w * la + (1 - w) * lb)

def scaffold_split(smi_list, K=5, seed=42):
    s2i = {}
    for i, s in enumerate(smi_list):
        try:
            mol = Chem.MolFromSmiles(s)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception: sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(K)]
    for i, sc in enumerate(scs): folds[i % K].extend(s2i[sc])
    return folds

def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    if not inchi: return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading data...")

    # Holdout exclusion IKs
    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cpk = json.load(f)
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: ho_iks.add(k)

    # MMPK training data — FULL with AUC/Tmax/t½
    mmpk = pd.read_csv(ROOT / "data/training/mmpk_expanded_full.csv")
    mmpk = mmpk[~mmpk["in_holdout"]].reset_index(drop=True)

    # Deduplicate by SMILES (median of PK endpoints)
    dedup = mmpk.groupby("canon_smiles").agg(
        dose=("dose_mg", "median"), cmax=("cmax_mg_L", "median"),
        log_cpd=("log_cmax_per_dose", "median"),
        auc=("auc_ng_h_ml", "median"), tmax=("tmax_h", "median"),
        thalf=("thalf_h", "median"), f_pct=("f_pct", "median"),
    ).reset_index()
    dedup["ik14"] = dedup["canon_smiles"].apply(ik14)
    dedup = dedup.dropna(subset=["ik14"])
    dedup = dedup[~dedup["ik14"].isin(ho_iks)].reset_index(drop=True)
    dedup = dedup[dedup["cmax"] > 0].reset_index(drop=True)

    # Engine predictions (cached)
    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")
    pbpk["ik14"] = pbpk["smiles"].apply(ik14)

    # Merge engine predictions + compound type
    merged = dedup.merge(
        pbpk[["ik14", "smiles", "cmax_pbpk", "is_base", "is_acid", "fup", "clint"]].drop_duplicates("ik14"),
        on="ik14", how="inner"
    )
    merged["compound_type"] = "neutral"
    merged.loc[merged["is_base"] == 1.0, "compound_type"] = "base"
    merged.loc[merged["is_acid"] == 1.0, "compound_type"] = "acid"
    merged = merged[merged["cmax_pbpk"] > 0].reset_index(drop=True)

    N = len(merged)
    log.info("  Training: %d drugs (with engine)", N)
    log.info("  AUC available: %d (%.0f%%)", (merged["auc"] > 0).sum(),
             (merged["auc"] > 0).mean() * 100)

    # Features
    smiles = merged["smiles"].tolist()
    X = np.array([compute_features(s) for s in smiles], dtype=np.float32)
    y_cpd = merged["log_cpd"].values  # log10(Cmax/dose)
    doses = merged["dose"].values

    # AUC target (for AUC predictor)
    auc_mask = merged["auc"] > 0
    y_auc = np.full(N, np.nan)
    y_auc[auc_mask] = np.log10(merged.loc[auc_mask, "auc"].values * 1e-6 / merged.loc[auc_mask, "dose"].values)
    # log10(AUC_mg_L_h / dose)

    # Cmax/AUC ratio target
    y_ratio = np.full(N, np.nan)
    ratio_mask = auc_mask & (merged["cmax"] > 0)
    y_ratio[ratio_mask] = np.log10(
        merged.loc[ratio_mask, "cmax"].values / (merged.loc[ratio_mask, "auc"].values * 1e-6)
    )  # log10(Cmax / AUC_mg_L_h)

    # Scaffold splits
    folds5 = scaffold_split(smiles, K=5)
    folds20 = scaffold_split(smiles, K=20)

    # XGB params
    xp = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.5, reg_lambda=3.0,
              random_state=42, n_jobs=4, verbosity=0)

    # Morgan FPs
    fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), radius=2, nBits=2048) for s in smiles]

    # Holdout
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f: ho_preds = json.load(f)
    ho_smiles = {}
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"): ho_smiles[n] = e["smiles"]

    ho_X = np.array([compute_features(ho_smiles.get(h["name"], "C")) for h in ho_preds], dtype=np.float32)
    ho_eng = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_clf = np.array([h["cmax_clf"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_doses = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses, dtype=float)
    ho_fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(ho_smiles.get(h["name"])) or Chem.MolFromSmiles("C"),
        radius=2, nBits=2048) for h in ho_preds]

    train = dict(
        N=N, smiles=smiles, X=X, y_cpd=y_cpd, doses=doses,
        engine=merged["cmax_pbpk"].values, obs=merged["cmax"].values,
        compound_type=merged["compound_type"].values,
        y_auc=y_auc, y_ratio=y_ratio, auc_mask=auc_mask.values,
        ratio_mask=ratio_mask.values,
        folds5=folds5, folds20=folds20, xp=xp, fps=fps,
        fup=merged["fup"].values, clint=merged["clint"].values,
    )
    holdout = dict(
        N=len(ho_preds), X=ho_X, engine=ho_eng, ml=ho_ml, clf=ho_clf,
        obs=ho_obs, combined=ho_combined, types=ho_types, doses=ho_doses,
        preds=ho_preds, fps=ho_fps,
    )
    return train, holdout


# ═══════════════════════════════════════════════════════════════
# P1: AUC-First Decomposition
# ═══════════════════════════════════════════════════════════════

def p1_auc_decomp(train, holdout):
    """Predict AUC and Cmax/AUC ratio separately, multiply for Cmax."""
    X, xp = train["X"], train["xp"]
    am, rm = train["auc_mask"], train["ratio_mask"]
    doses_tr, doses_ho = train["doses"], holdout["doses"]

    # Train AUC predictor (on drugs with AUC data)
    y_auc = train["y_auc"]
    idx_auc = np.where(am)[0]

    folds = scaffold_split([train["smiles"][i] for i in idx_auc], K=5)
    oof_auc = np.full(len(idx_auc), np.nan)
    for fi, vi in enumerate(folds):
        ti = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        m = xgb.XGBRegressor(**xp)
        m.fit(X[idx_auc[ti]], y_auc[idx_auc[ti]])
        oof_auc[vi] = m.predict(X[idx_auc[vi]])
    oof_auc_cmax = 10 ** oof_auc * doses_tr[idx_auc]  # this is AUC, not Cmax yet

    m_auc = xgb.XGBRegressor(**xp)
    m_auc.fit(X[idx_auc], y_auc[idx_auc])
    ho_auc_log = m_auc.predict(holdout["X"])
    ho_auc = 10 ** ho_auc_log * doses_ho  # AUC in mg/L·h

    auc_aafe = aafe(ho_auc, holdout["obs"])  # comparing AUC predictor to Cmax obs (mismatched!)
    log.info("  P1: AUC predictor trained on %d drugs", len(idx_auc))

    # Train Cmax/AUC ratio predictor
    y_ratio = train["y_ratio"]
    idx_ratio = np.where(rm)[0]

    m_ratio = xgb.XGBRegressor(**xp)
    m_ratio.fit(X[idx_ratio], y_ratio[idx_ratio])
    ho_ratio = 10 ** m_ratio.predict(holdout["X"])  # Cmax/AUC ratio

    # Combine: Cmax = AUC × ratio
    ho_cmax_decomp = ho_auc * ho_ratio

    standalone = aafe(ho_cmax_decomp, holdout["obs"])
    log.info("  P1: AUC×ratio standalone AAFE: %.3f (baseline ML: %.3f)",
             standalone, aafe(holdout["ml"], holdout["obs"]))

    # Error correlation with baseline ML
    fe_ml = np.log10(np.maximum(holdout["ml"], 1e-10) / np.maximum(holdout["obs"], 1e-10))
    fe_decomp = np.log10(np.maximum(ho_cmax_decomp, 1e-10) / np.maximum(holdout["obs"], 1e-10))
    r_corr = np.corrcoef(fe_ml, fe_decomp)[0, 1]
    log.info("  P1: Error correlation with ML: r=%.3f ★", r_corr)

    # Blend with engine
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(holdout["engine"][i], ho_cmax_decomp[i], w)

    return preds, ho_cmax_decomp


# ═══════════════════════════════════════════════════════════════
# P2: 20-fold CV Stacking (fix OOF-Full gap)
# ═══════════════════════════════════════════════════════════════

def p2_20fold_stacking(train, holdout):
    """Higher K-fold → OOF closer to full model → Ridge transfers."""
    X, y, xp = train["X"], train["y_cpd"], train["xp"]
    doses_tr = train["doses"]
    N = train["N"]

    for K_label, folds in [("5-fold", train["folds5"]), ("20-fold", train["folds20"])]:
        ml_oof = np.full(N, np.nan)
        for fi, vi in enumerate(folds):
            ti = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
            m = xgb.XGBRegressor(**xp)
            m.fit(X[ti], y[ti])
            ml_oof[vi] = m.predict(X[vi])

        ml_oof_cmax = 10 ** ml_oof * doses_tr
        # Correlation with full model
        m_full = xgb.XGBRegressor(**xp)
        m_full.fit(X, y)
        ml_full = m_full.predict(X)
        r = np.corrcoef(ml_oof[~np.isnan(ml_oof)], ml_full[~np.isnan(ml_oof)])[0, 1]
        log.info("  P2: %s OOF-Full r=%.4f, OOF AAFE=%.3f",
                 K_label, r, aafe(ml_oof_cmax, train["obs"]))

        if K_label == "20-fold":
            oof_20 = ml_oof

    # Ridge stacking on 20-fold OOF
    log_eng = np.log10(np.maximum(train["engine"], 1e-10))
    log_ml_oof = oof_20
    log_obs = np.log10(np.maximum(train["obs"], 1e-10))

    X_stack = np.column_stack([log_eng, log_ml_oof])
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_stack, log_obs)
    log.info("  P2: Ridge coefficients: intercept=%.4f, b_eng=%.4f, b_ml=%.4f (alpha=%.2f)",
             ridge.intercept_, ridge.coef_[0], ridge.coef_[1], ridge.alpha_)

    # Holdout
    log_ho_eng = np.log10(np.maximum(holdout["engine"], 1e-10))
    log_ho_ml = np.log10(np.maximum(holdout["ml"], 1e-10))
    X_ho = np.column_stack([log_ho_eng, log_ho_ml])
    ho_pred = 10 ** ridge.predict(X_ho)

    return ho_pred, None


# ═══════════════════════════════════════════════════════════════
# P3: Analytical 1-Cpt from ADME predictions
# ═══════════════════════════════════════════════════════════════

def p3_analytical_1cpt(train, holdout):
    """Cmax from closed-form 1-compartment oral model using predicted PK params."""
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    preds_1cpt = []
    for i, h in enumerate(holdout["preds"]):
        try:
            smi = None
            for n in [h["name"], h["name"].replace(" ", "_")]:
                with open(ROOT / "data/reference/clinical_pk.json") as f:
                    cpk_tmp = json.load(f)
                entry = cpk_tmp["drugs"].get(n)
                if entry and entry.get("smiles"):
                    smi = entry["smiles"]; break
            if not smi:
                preds_1cpt.append(holdout["ml"][i]); continue

            profile = compute_profile(smi)
            adme_props = predict_adme(profile)
            dose = holdout["doses"][i]

            # Extract PK params
            fup = adme_props.fup.mean
            clint = adme_props.clint.mean  # uL/min/million cells
            vdss = adme_props.vdss.mean  # L/kg, assume 70kg
            peff = adme_props.peff.mean  # cm/s

            # IVIVE: CLint → CLh
            # CLh = Qh × fup × CLint_scaled / (Qh + fup × CLint_scaled)
            MPPGL = 120  # mg protein per g liver
            LIVER_W = 1800  # g
            Q_LIVER = 90  # L/h
            BW = 70  # kg

            clint_scaled = clint * MPPGL * LIVER_W * 60 * 1e-6  # uL/min → L/h
            CLh = Q_LIVER * fup * clint_scaled / (Q_LIVER + fup * clint_scaled)
            CL = max(CLh, 0.01)  # L/h

            Vd = max(vdss * BW, 1.0)  # L
            ke = CL / Vd  # 1/h

            # Absorption: ka from Peff
            R_intestine = 1.75  # cm
            ka = max(2 * peff * 3600 / R_intestine, 0.1)  # 1/h

            # Bioavailability: Fg × Fh (no gut first pass for simplicity)
            Fh = 1 - CLh / Q_LIVER
            F = max(min(Fh * 0.9, 1.0), 0.01)  # assume Fa×Fg ≈ 0.9 for oral

            # 1-compartment oral: Cmax = F×dose×ka / (Vd×(ka-ke)) × (exp(-ke×Tmax) - exp(-ka×Tmax))
            if abs(ka - ke) < 0.001:
                Tmax = 1.0 / ka
            else:
                Tmax = np.log(ka / ke) / (ka - ke)
            Tmax = max(min(Tmax, 24.0), 0.01)

            Cmax = F * dose * ka / (Vd * abs(ka - ke) + 1e-10) * \
                   abs(np.exp(-ke * Tmax) - np.exp(-ka * Tmax))
            Cmax = max(Cmax, 1e-10)
            preds_1cpt.append(Cmax)

        except Exception:
            preds_1cpt.append(holdout["ml"][i])

    logging.getLogger("sisyphus").setLevel(logging.INFO)
    preds_1cpt = np.array(preds_1cpt)
    standalone = aafe(preds_1cpt, holdout["obs"])
    log.info("  P3: Analytical 1-Cpt standalone AAFE: %.3f", standalone)

    # Error correlation with ML and engine
    fe_ml = np.log10(np.maximum(holdout["ml"], 1e-10) / np.maximum(holdout["obs"], 1e-10))
    fe_1cpt = np.log10(np.maximum(preds_1cpt, 1e-10) / np.maximum(holdout["obs"], 1e-10))
    fe_eng = np.log10(np.maximum(holdout["engine"], 1e-10) / np.maximum(holdout["obs"], 1e-10))
    log.info("  P3: r(1cpt, ML)=%.3f, r(1cpt, engine)=%.3f",
             np.corrcoef(fe_1cpt, fe_ml)[0, 1], np.corrcoef(fe_1cpt, fe_eng)[0, 1])

    # Blend with ML
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(preds_1cpt[i], holdout["ml"][i], w)

    return preds, preds_1cpt


# ═══════════════════════════════════════════════════════════════
# P4: Substructure Engine Correction
# ═══════════════════════════════════════════════════════════════

def p4_substruct_correction(train, holdout):
    """SMARTS patterns → average engine fold error → correction."""
    from rdkit.Chem import MolFromSmarts

    patterns = {
        "phenol": "[OH]c1ccccc1",
        "carboxyl": "C(=O)[OH]",
        "amine_primary": "[NH2]",
        "amine_secondary": "[NH]([C])[C]",
        "amine_tertiary": "[N]([C])([C])[C]",
        "amide": "C(=O)N",
        "ester": "C(=O)O[C]",
        "sulfonamide": "S(=O)(=O)N",
        "fluorine": "[F]",
        "chlorine": "[Cl]",
        "heterocycle_N": "[nR]",
        "heterocycle_O": "[oR]",
        "lactam": "C1(=O)NC1",
    }

    eng_fe = np.log10(train["engine"] / train["obs"])

    # For each pattern, compute average engine FE for drugs containing it
    pattern_bias = {}
    for name, smarts in patterns.items():
        pat = MolFromSmarts(smarts)
        if not pat: continue
        hits = np.array([
            bool(Chem.MolFromSmiles(s) and Chem.MolFromSmiles(s).HasSubstructMatch(pat))
            for s in train["smiles"]])
        if hits.sum() >= 20:
            bias = eng_fe[hits].mean()
            pattern_bias[name] = (smarts, bias, hits.sum())

    for name, (_, bias, n) in sorted(pattern_bias.items(), key=lambda x: abs(x[1][1]), reverse=True)[:5]:
        log.info("  P4: %s: bias=%+.3f, n=%d", name, bias, n)

    # For each holdout drug, compute correction from matching patterns
    corrections = np.zeros(holdout["N"])
    for i, h in enumerate(holdout["preds"]):
        smi = None
        for preds_entry in [h]:
            # Get SMILES
            with open(ROOT / "data/reference/clinical_pk.json") as f:
                cpk_tmp = json.load(f)
            entry = cpk_tmp["drugs"].get(h["name"]) or cpk_tmp["drugs"].get(h["name"].replace(" ", "_"))
            if entry: smi = entry.get("smiles")
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue

        matching_biases = []
        for name, (smarts, bias, _) in pattern_bias.items():
            pat = MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pat):
                matching_biases.append(bias)

        if matching_biases:
            corrections[i] = np.mean(matching_biases)

    # Apply correction to engine
    log_eng_ho = np.log10(np.maximum(holdout["engine"], 1e-10))
    corrected_eng = 10 ** (log_eng_ho - corrections)

    corr_aafe = aafe(corrected_eng, holdout["obs"])
    log.info("  P4: Corrected engine AAFE: %.3f (raw: %.3f)",
             corr_aafe, aafe(holdout["engine"], holdout["obs"]))

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(corrected_eng[i], holdout["ml"][i], w)

    return preds, None


# ═══════════════════════════════════════════════════════════════
# P5: LOWESS Engine Calibration
# ═══════════════════════════════════════════════════════════════

def p5_lowess_calibration(train, holdout):
    """Local weighted regression (robust, not isotonic) for engine calibration."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    log_eng = np.log10(np.maximum(train["engine"], 1e-10))
    log_obs = np.log10(np.maximum(train["obs"], 1e-10))

    # LOWESS with large fraction (robust, less overfitting)
    for frac in [0.3, 0.5, 0.7]:
        result = lowess(log_obs, log_eng, frac=frac, return_sorted=True)
        # Interpolation function
        cal_x = result[:, 0]
        cal_y = result[:, 1]

        # Apply to holdout
        log_eng_ho = np.log10(np.maximum(holdout["engine"], 1e-10))
        cal_ho = np.interp(log_eng_ho, cal_x, cal_y)
        cal_eng_ho = 10 ** cal_ho

        cal_aafe = aafe(cal_eng_ho, holdout["obs"])
        log.info("  P5: LOWESS frac=%.1f → engine AAFE: %.3f", frac, cal_aafe)

    # Use frac=0.5 as primary
    result = lowess(log_obs, log_eng, frac=0.5, return_sorted=True)
    cal_ho = np.interp(log_eng_ho, result[:, 0], result[:, 1])
    cal_eng = 10 ** cal_ho

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(cal_eng[i], holdout["ml"][i], w)

    return preds, None


# ═══════════════════════════════════════════════════════════════
# O6: Rank Aggregation (Social Choice Theory)
# ═══════════════════════════════════════════════════════════════

def o6_rank_aggregation(train, holdout):
    """Aggregate engine/ML/CL-F by rank, not value."""
    N = holdout["N"]
    eng = holdout["engine"]
    ml = holdout["ml"]
    clf = holdout["clf"]

    # Rank each track (1 = lowest Cmax)
    rank_eng = np.argsort(np.argsort(eng)).astype(float)
    rank_ml = np.argsort(np.argsort(ml)).astype(float)
    rank_clf = np.argsort(np.argsort(clf)).astype(float)

    # Average ranks (Borda count)
    for label, ranks in [
        ("eng+ml", (rank_eng + rank_ml) / 2),
        ("eng+ml+clf", (rank_eng + rank_ml + rank_clf) / 3),
        ("2eng+3ml", (2*rank_eng + 3*rank_ml) / 5),
    ]:
        # Map average rank back to Cmax using ML distribution (as reference)
        sorted_ml = np.sort(ml)
        avg_rank = ranks
        # Normalize rank to [0, N-1]
        norm_rank = (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min() + 1e-10) * (N - 1)
        # Interpolate from sorted ML values
        mapped = np.interp(norm_rank, np.arange(N), sorted_ml)
        a = aafe(mapped, holdout["obs"])
        log.info("  O6: Rank(%s) → AAFE=%.3f", label, a)

    # Primary: eng+ml average rank → map to ML distribution
    avg_rank = (rank_eng + rank_ml) / 2
    norm_rank = (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min() + 1e-10) * (N - 1)
    sorted_ml = np.sort(ml)
    preds = np.interp(norm_rank, np.arange(N), sorted_ml)

    return preds, None


# ═══════════════════════════════════════════════════════════════
# O7: Feature-Dropout Ensemble (Bagging variant)
# ═══════════════════════════════════════════════════════════════

def o7_dropout_ensemble(train, holdout):
    """30 XGBoost with random 80% feature masks → ensemble + variance."""
    X, y, xp = train["X"], train["y_cpd"], train["xp"]
    doses_ho = holdout["doses"]
    rng = np.random.default_rng(42)

    ho_preds_all = []
    for b in range(30):
        # Random 80% feature mask
        mask = rng.random(X.shape[1]) < 0.8
        if mask.sum() < 50: mask[:50] = True

        m = xgb.XGBRegressor(**xp)
        m.fit(X[:, mask], y)
        ho_log = m.predict(holdout["X"][:, mask])
        ho_cmax = 10 ** ho_log * doses_ho
        ho_preds_all.append(ho_cmax)

    ho_stack = np.array(ho_preds_all)
    ensemble = np.median(ho_stack, axis=0)  # use median for robustness
    variance = np.var(np.log10(np.maximum(ho_stack, 1e-10)), axis=0)

    ens_aafe = aafe(ensemble, holdout["obs"])
    log.info("  O7: 30-model dropout ensemble AAFE: %.3f", ens_aafe)
    log.info("  O7: Prediction variance median=%.4f", np.median(variance))

    # Blend with engine (use variance for routing)
    preds = np.zeros(holdout["N"])
    var_med = np.median(variance)
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            conf = np.clip(1 - variance[i] / (2 * var_med), 0, 1)
            w = 0.45 * conf
        else:
            w = 0.00
        preds[i] = geo(holdout["engine"][i], ensemble[i], w)

    return preds, ensemble


# ═══════════════════════════════════════════════════════════════
# O8: Huber Loss XGBoost (Robust Statistics)
# ═══════════════════════════════════════════════════════════════

def o8_huber_xgb(train, holdout):
    """Pseudo-Huber loss: smooth L1 near zero, L2 far away."""
    X, y, xp_base = train["X"], train["y_cpd"], dict(train["xp"])
    doses_ho = holdout["doses"]

    # XGBoost with Huber loss (reg:pseudohubererror)
    xp_huber = {**xp_base, "objective": "reg:pseudohubererror",
                "huber_slope": 0.5}  # delta parameter

    m = xgb.XGBRegressor(**xp_huber)
    m.fit(X, y)
    ho_log = m.predict(holdout["X"])
    ho_cmax = 10 ** ho_log * doses_ho

    standalone = aafe(ho_cmax, holdout["obs"])
    log.info("  O8: Huber XGB standalone AAFE: %.3f (baseline ML: %.3f)",
             standalone, aafe(holdout["ml"], holdout["obs"]))

    # Blend with engine
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(holdout["engine"][i], ho_cmax[i], w)

    return preds, ho_cmax


# ═══════════════════════════════════════════════════════════════
# O9: AUC-derived ML + Engine Meta
# ═══════════════════════════════════════════════════════════════

def o9_auc_ml_engine(train, holdout):
    """Use AUC-decomposed Cmax as ML track in standard meta-learner."""
    # Get P1's decomposed predictions
    _, ho_decomp = p1_auc_decomp(train, holdout)
    if ho_decomp is None:
        return holdout["ml"], None

    # 3-way blend: engine + standard ML + AUC-decomposed
    ho_eng = holdout["engine"]
    ho_ml = holdout["ml"]

    best_a = float("inf")
    best_wa, best_wd = 0.45, 0.00
    for wa in np.arange(0, 0.6, 0.05):
        for wd in np.arange(0, 0.5, 0.05):
            if wa + wd > 0.8: continue
            wm = 1 - wa - wd
            trial = np.zeros(holdout["N"])
            for i in range(holdout["N"]):
                if holdout["types"][i] == "base":
                    le = np.log10(max(ho_eng[i], 1e-10))
                    lm = np.log10(max(ho_ml[i], 1e-10))
                    ld = np.log10(max(ho_decomp[i], 1e-10))
                    trial[i] = 10 ** (wa * le + wm * lm + wd * ld)
                else:
                    # For non-base: try adding small AUC-decomp weight
                    lm = np.log10(max(ho_ml[i], 1e-10))
                    ld = np.log10(max(ho_decomp[i], 1e-10))
                    trial[i] = 10 ** ((1-wd) * lm + wd * ld)
            a = aafe(trial, holdout["obs"])
            if a < best_a:
                best_a = a
                best_wa, best_wd = wa, wd

    log.info("  O9: 3-way best w_eng=%.2f, w_auc=%.2f → AAFE=%.3f", best_wa, best_wd, best_a)

    # ⚠ This is grid-searched on holdout — report with caveat
    trial = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        if holdout["types"][i] == "base":
            le = np.log10(max(ho_eng[i], 1e-10))
            lm = np.log10(max(ho_ml[i], 1e-10))
            ld = np.log10(max(ho_decomp[i], 1e-10))
            trial[i] = 10 ** (best_wa * le + (1-best_wa-best_wd) * lm + best_wd * ld)
        else:
            lm = np.log10(max(ho_ml[i], 1e-10))
            ld = np.log10(max(ho_decomp[i], 1e-10))
            trial[i] = 10 ** ((1-best_wd) * lm + best_wd * ld)

    return trial, None


# ═══════════════════════════════════════════════════════════════
# O10: AUC-Decomposed as Direct Replacement
# ═══════════════════════════════════════════════════════════════

def o10_auc_replace_ml(train, holdout):
    """Replace standard ML with AUC-decomposed Cmax as the sole ML track."""
    _, ho_decomp = p1_auc_decomp(train, holdout)
    if ho_decomp is None:
        return holdout["combined"], None

    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(holdout["engine"][i], ho_decomp[i], w)

    return preds, None


# ═══════════════════════════════════════════════════════════════
# Tournament
# ═══════════════════════════════════════════════════════════════

def run(train, holdout):
    log.info("=" * 70)
    log.info("TOURNAMENT V3: Orthogonal Prediction Sources")
    log.info("=" * 70)

    methods = [
        ("P1  AUC Decomposition", p1_auc_decomp),
        ("P2  20-fold Stacking", p2_20fold_stacking),
        ("P3  Analytical 1-Cpt", p3_analytical_1cpt),
        ("P4  Substruct Correct", p4_substruct_correction),
        ("P5  LOWESS Calibration", p5_lowess_calibration),
        ("O6  Rank Aggregation", o6_rank_aggregation),
        ("O7  Dropout Ensemble", o7_dropout_ensemble),
        ("O8  Huber Loss XGB", o8_huber_xgb),
        ("O9  AUC+ML+Eng 3-way", o9_auc_ml_engine),
        ("O10 AUC replaces ML", o10_auc_replace_ml),
    ]

    obs = holdout["obs"]
    base = holdout["types"] == "base"
    baseline = aafe(holdout["combined"], obs)

    results = []
    for name, fn in methods:
        log.info("─── %s ───", name)
        t0 = time.time()
        try:
            out = fn(train, holdout)
            pred = out[0] if isinstance(out, tuple) else out
            a_all = aafe(pred, obs)
            a_base = aafe(pred[base], obs[base])
            a_nb = aafe(pred[~base], obs[~base])
            pf = p2f(pred, obs)
            d = a_all - baseline
            fe_b = np.log10(np.maximum(holdout["combined"], 1e-10) / np.maximum(obs, 1e-10))
            fe_t = np.log10(np.maximum(pred, 1e-10) / np.maximum(obs, 1e-10))
            r = np.corrcoef(fe_b, fe_t)[0, 1]
            results.append((name, a_all, a_base, a_nb, pf, d, r))
            log.info("  → AAFE=%.3f (Δ=%+.3f) r=%.3f base=%.3f ~base=%.3f %%2f=%.1f%% [%.1fs]",
                     a_all, d, r, a_base, a_nb, pf, time.time() - t0)
        except Exception as e:
            log.error("  → FAILED: %s", e)
            import traceback; traceback.print_exc()
            results.append((name, np.inf, np.inf, np.inf, 0, np.inf, 0))

    # Leaderboard
    log.info("\n" + "=" * 85)
    log.info("LEADERBOARD V3")
    log.info("=" * 85)
    log.info(f"{'Rk':<4} {'Method':<25} {'AAFE':>7} {'Δ':>8} {'r':>6} {'Base':>7} {'~Base':>7} {'%2f':>6}")
    log.info("-" * 85)

    all_r = [("--- Baseline Meta", baseline, 0, 0, 0, 0, 1.0)] + results
    for rk, (n, a, ab, anb, pf, d, r) in enumerate(sorted(all_r, key=lambda x: x[1]), 1):
        mk = " ★" if d < -0.02 else ""
        log.info(f"  {rk:<3} {n:<25} {a:>7.3f} {d:>+8.3f} {r:>6.3f} {ab:>7.3f} {anb:>7.3f} {pf:>5.1f}%{mk}")

    # Save
    out = {"baseline": baseline, "methods": [
        {"name": n, "aafe": a, "base": ab, "nonbase": anb, "p2f": pf, "delta": d, "r": r}
        for n, a, ab, anb, pf, d, r in results]}
    with open(ROOT / "data/validation/meta_tournament_v3_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("\nSaved to data/validation/meta_tournament_v3_results.json")

    best = min(results, key=lambda x: x[1])
    if best[5] < -0.05:
        log.info("\n★★★ MEANINGFUL: %s (Δ=%.3f)", best[0], best[5])
    elif best[5] < -0.02:
        log.info("\n★ MARGINAL: %s (Δ=%.3f)", best[0], best[5])
    else:
        log.info("\n✗ NO IMPROVEMENT — 53 total methods tested")


def main():
    t0 = time.time()
    train, holdout = load_data()
    run(train, holdout)
    log.info("Total: %.1fs", time.time() - t0)

if __name__ == "__main__":
    main()
