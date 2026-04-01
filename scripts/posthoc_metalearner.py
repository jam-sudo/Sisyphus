#!/usr/bin/env python3
"""Post-Hoc Meta-Learner: OOF Stacking + ACF + Winsorized.

3 independent post-hoc corrections on existing engine/ML outputs.
No engine or ML internals are modified.

Phase 0: Data Preparation (engine + OOF ML + neighbors)
Phase 1: Independent tests (Stacking, ACF, Winsorized)
Phase 2: Integration (Stacking → ACF)
Phase 3: Error analysis
"""
from __future__ import annotations

import json, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    if not inchi: return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


def compute_aafe(pred, obs):
    mask = (pred > 0) & (obs > 0)
    if mask.sum() == 0: return float("inf")
    return float(10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask]))))


def pct_2fold(pred, obs):
    mask = (pred > 0) & (obs > 0)
    if mask.sum() == 0: return 0.0
    fe = pred[mask] / obs[mask]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100)


def scaffold_split(smiles_list, n_folds=5, seed=42):
    """Stratified k-fold split by Murcko generic scaffold."""
    s2i = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scs):
        folds[i % n_folds].extend(s2i[sc])
    return folds


def weighted_median(values, weights):
    """Compute weighted median."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumw = np.cumsum(sorted_weights)
    cutoff = cumw[-1] * 0.5
    idx = np.searchsorted(cumw, cutoff)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def load_holdout_iks(holdout_only=True):
    """Load InChIKey-14 set for holdout drugs.

    Args:
        holdout_only: If True, only exclude 107 holdout drugs.
                      If False, exclude holdout + train reference drugs (183 total).
    """
    with open(ROOT / "data/reference/holdout.json") as f:
        hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cp = json.load(f)
    if holdout_only:
        names = hd.get("holdout", [])
    else:
        names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    iks = set()
    for n in names:
        e = drugs.get(n) or drugs.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: iks.add(k)
    return iks


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

def phase0():
    log.info("=" * 70)
    log.info("Phase 0: Data Preparation")
    log.info("=" * 70)

    # ── 0A: Load cached engine predictions, exclude holdout ──
    log.info("─── 0A: Engine predictions (cached) ───")
    ho_iks = load_holdout_iks(holdout_only=True)
    log.info("  Holdout IK14 set: %d entries (holdout only)", len(ho_iks))

    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")
    log.info("  MMPK PBPK features loaded: %d drugs", len(pbpk))

    # Compute IK14 for exclusion
    pbpk["ik14"] = pbpk["smiles"].apply(ik14)
    before = len(pbpk)
    pbpk = pbpk.dropna(subset=["ik14"]).reset_index(drop=True)
    pbpk = pbpk[~pbpk["ik14"].isin(ho_iks)].reset_index(drop=True)
    pbpk = pbpk[pbpk["cmax_pbpk"] > 0].reset_index(drop=True)
    pbpk = pbpk[pbpk["cmax_obs"] > 0].reset_index(drop=True)
    log.info("  After holdout exclusion: %d → %d drugs (excluded %d)",
             before, len(pbpk), before - len(pbpk))

    # Derive compound_type
    pbpk["compound_type"] = "neutral"
    pbpk.loc[pbpk["is_base"] == 1.0, "compound_type"] = "base"
    pbpk.loc[pbpk["is_acid"] == 1.0, "compound_type"] = "acid"

    # ── 0B: ML OOF predictions ──
    log.info("─── 0B: ML OOF predictions (5-fold scaffold CV) ───")

    smiles_list = pbpk["smiles"].tolist()
    N = len(smiles_list)

    # Compute features
    t0 = time.time()
    feat_list = []
    valid_mask = np.ones(N, dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            feat_list.append(compute_features(smi))
        except Exception:
            feat_list.append(np.zeros(2057))
            valid_mask[i] = False
    X_all = np.array(feat_list, dtype=np.float32)
    log.info("  Features computed: %d drugs (%.1fs), %d invalid",
             N, time.time() - t0, (~valid_mask).sum())

    # Target: log10(cmax / dose)
    y_all = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)
    doses = pbpk["dose_mg"].values

    # XGBoost hyperparameters (from production model meta.json)
    xgb_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42,
        n_jobs=4, verbosity=0,
    )

    # 5-fold scaffold split
    folds = scaffold_split(smiles_list)
    fold_sizes = [len(f) for f in folds]
    log.info("  Scaffold folds: %s (total %d)", fold_sizes, sum(fold_sizes))

    # OOF predictions
    ml_oof = np.full(N, np.nan)
    t0 = time.time()
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_all[train_idx], y_all[train_idx])
        preds = model.predict(X_all[val_idx])
        ml_oof[val_idx] = preds
        log.info("    Fold %d: train=%d, val=%d", fi, len(train_idx), len(val_idx))
    log.info("  ML OOF complete (%.1fs)", time.time() - t0)

    # Convert OOF log predictions to Cmax (mg/L)
    cmax_ml_oof = 10 ** ml_oof * doses
    cmax_ml_oof = np.maximum(cmax_ml_oof, 1e-10)

    # Also compute full-model ML predictions for OOF-vs-full correlation
    model_full = xgb.XGBRegressor(**xgb_params)
    model_full.fit(X_all, y_all)
    ml_full_log = model_full.predict(X_all)
    cmax_ml_full = 10 ** ml_full_log * doses

    r_oof_full = np.corrcoef(ml_oof[~np.isnan(ml_oof)],
                              ml_full_log[~np.isnan(ml_oof)])[0, 1]
    log.info("  ML OOF vs Full correlation: r = %.4f", r_oof_full)
    if r_oof_full < 0.90:
        log.warning("  ⚠ OOF-Full correlation < 0.90 — transfer risk!")

    # OOF ML AAFE (sanity check)
    aafe_ml_oof = compute_aafe(cmax_ml_oof, pbpk["cmax_obs"].values)
    aafe_ml_full = compute_aafe(cmax_ml_full, pbpk["cmax_obs"].values)
    log.info("  ML OOF AAFE (training): %.3f", aafe_ml_oof)
    log.info("  ML Full AAFE (training, in-sample): %.3f", aafe_ml_full)

    # ── 0C: Nearest neighbor precomputation ──
    log.info("─── 0C: Nearest neighbor precomputation ───")
    t0 = time.time()

    # Morgan FPs for training drugs
    train_fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        else:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), radius=2, nBits=2048))

    # LOO neighbors within training set (for ACF training evaluation)
    K_MAX = 20
    train_neighbors = []  # list of list of (idx, sim)
    for i in range(N):
        sims = DataStructs.BulkTanimotoSimilarity(train_fps[i], train_fps)
        sims[i] = -1.0  # exclude self
        top_idx = np.argsort(sims)[-K_MAX:][::-1]
        train_neighbors.append([(int(j), float(sims[j])) for j in top_idx])

    # Holdout FPs and neighbors
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        holdout_preds = json.load(f)

    holdout_fps = []
    holdout_valid = []
    for h in holdout_preds:
        # Need SMILES for holdout drugs
        # Get from clinical_pk.json
        pass

    # Load holdout SMILES from clinical_pk
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cpk = json.load(f)
    with open(ROOT / "data/reference/holdout.json") as f:
        ho_json = json.load(f)
    ho_names = ho_json.get("holdout", [])

    holdout_smiles = {}
    for name in ho_names:
        entry = cpk["drugs"].get(name) or cpk["drugs"].get(name.replace(" ", "_"))
        if entry and entry.get("smiles"):
            holdout_smiles[name] = entry["smiles"]

    holdout_neighbors = {}  # name → list of (train_idx, sim)
    for h in holdout_preds:
        name = h["name"]
        smi = holdout_smiles.get(name)
        if not smi:
            holdout_neighbors[name] = []
            continue
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            holdout_neighbors[name] = []
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        top_idx = np.argsort(sims)[-K_MAX:][::-1]
        holdout_neighbors[name] = [(int(j), float(sims[j])) for j in top_idx]

    log.info("  Neighbors computed (%.1fs): %d training, %d holdout",
             time.time() - t0, N, len(holdout_neighbors))

    # ── Compute meta-learner predictions for training (using OOF ML) ──
    cmax_engine = pbpk["cmax_pbpk"].values
    compound_types = pbpk["compound_type"].values

    # Apply meta-learner formula with OOF ML
    cmax_meta_oof = np.zeros(N)
    for i in range(N):
        ce = cmax_engine[i]
        cm = cmax_ml_oof[i]
        ct = compound_types[i]
        is_base = (ct == "base")
        w_eng = 0.45 if is_base else 0.00
        w_ml = 0.55 if is_base else 1.00

        if w_eng > 0 and ce > 0 and cm > 0:
            log_e = np.log10(max(ce, 1e-10))
            log_m = np.log10(max(cm, 1e-10))
            disag = abs(log_e - log_m)
            if disag > 1.0:
                scale = 1.0 / disag
                w_eng *= scale
                total = w_eng + w_ml
                w_eng /= total
                w_ml /= total
            log_cmax = w_eng * log_e + w_ml * log_m
            cmax_meta_oof[i] = 10 ** log_cmax
        else:
            cmax_meta_oof[i] = cm

    aafe_meta_oof = compute_aafe(cmax_meta_oof, pbpk["cmax_obs"].values)
    log.info("  Meta (OOF ML) AAFE (training): %.3f", aafe_meta_oof)

    # Assemble training data dict
    train_data = {
        "smiles": smiles_list,
        "cmax_obs": pbpk["cmax_obs"].values,
        "cmax_engine": cmax_engine,
        "cmax_ml_oof": cmax_ml_oof,
        "cmax_ml_full": cmax_ml_full,
        "cmax_meta_oof": cmax_meta_oof,
        "compound_type": compound_types,
        "dose_mg": doses,
        "X_features": X_all,
        "y_log_cpd": y_all,
        "folds": folds,
        "train_neighbors": train_neighbors,
        "r_oof_full": r_oof_full,
        "N": N,
    }

    holdout_data = {
        "preds": holdout_preds,
        "neighbors": holdout_neighbors,
    }

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "smiles": smiles_list,
        "dose_mg": doses,
        "cmax_obs": pbpk["cmax_obs"].values,
        "cmax_engine": cmax_engine,
        "cmax_ml_oof": cmax_ml_oof,
        "cmax_ml_full": cmax_ml_full,
        "cmax_meta_oof": cmax_meta_oof,
        "compound_type": compound_types,
    })
    oof_df.to_csv(ROOT / "data/training/oof_predictions.csv", index=False)
    log.info("  Saved: data/training/oof_predictions.csv")

    return train_data, holdout_data


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1A: OOF Stacking
# ═══════════════════════════════════════════════════════════════════════════

def phase1a_stacking(train_data, holdout_data):
    log.info("=" * 70)
    log.info("Phase 1A: OOF Stacking (Ridge)")
    log.info("=" * 70)

    N = train_data["N"]
    cmax_engine_train = train_data["cmax_engine"]
    cmax_ml_oof_train = train_data["cmax_ml_oof"]
    cmax_obs_train = train_data["cmax_obs"]
    doses_train = train_data["dose_mg"]
    compound_types_train = train_data["compound_type"]
    folds = train_data["folds"]

    # Training features (log space)
    log_eng = np.log10(np.maximum(cmax_engine_train, 1e-10))
    log_ml = np.log10(np.maximum(cmax_ml_oof_train, 1e-10))
    log_obs = np.log10(np.maximum(cmax_obs_train, 1e-10))

    # ── V1: Minimal (2 features) ──
    X_v1 = np.column_stack([log_eng, log_ml])

    # Ridge alpha selection via 5-fold CV on training set
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv.fit(X_v1, log_obs)
    best_alpha_v1 = ridge_cv.alpha_
    log.info("  V1 (2-feat) best alpha: %.2f", best_alpha_v1)

    ridge_v1 = Ridge(alpha=best_alpha_v1)
    ridge_v1.fit(X_v1, log_obs)
    log.info("  V1 coefficients: intercept=%.4f, b_engine=%.4f, b_ml=%.4f",
             ridge_v1.intercept_, ridge_v1.coef_[0], ridge_v1.coef_[1])
    log.info("  V1 training R²: %.4f", ridge_v1.score(X_v1, log_obs))

    # ── V2: + dose (3 features) ──
    log_dose = np.log10(np.maximum(doses_train, 0.01))
    X_v2 = np.column_stack([log_eng, log_ml, log_dose])

    ridge_cv2 = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv2.fit(X_v2, log_obs)
    best_alpha_v2 = ridge_cv2.alpha_
    log.info("  V2 (3-feat) best alpha: %.2f", best_alpha_v2)

    ridge_v2 = Ridge(alpha=best_alpha_v2)
    ridge_v2.fit(X_v2, log_obs)
    log.info("  V2 coefficients: intercept=%.4f, b_engine=%.4f, b_ml=%.4f, b_dose=%.4f",
             ridge_v2.intercept_, ridge_v2.coef_[0], ridge_v2.coef_[1], ridge_v2.coef_[2])
    log.info("  V2 training R²: %.4f", ridge_v2.score(X_v2, log_obs))

    # ── V3: + compound_type indicators (4 features) ──
    is_base_train = (compound_types_train == "base").astype(float)
    is_acid_train = (compound_types_train == "acid").astype(float)
    X_v3 = np.column_stack([log_eng, log_ml, is_base_train, is_acid_train])

    ridge_cv3 = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv3.fit(X_v3, log_obs)
    best_alpha_v3 = ridge_cv3.alpha_
    log.info("  V3 (4-feat +type) best alpha: %.2f", best_alpha_v3)

    ridge_v3 = Ridge(alpha=best_alpha_v3)
    ridge_v3.fit(X_v3, log_obs)
    log.info("  V3 coefficients: intercept=%.4f, b_eng=%.4f, b_ml=%.4f, b_base=%.4f, b_acid=%.4f",
             ridge_v3.intercept_, ridge_v3.coef_[0], ridge_v3.coef_[1],
             ridge_v3.coef_[2], ridge_v3.coef_[3])
    log.info("  V3 training R²: %.4f", ridge_v3.score(X_v3, log_obs))

    # ── V4: Use ml_full (in-sample) instead of OOF — sensitivity analysis ──
    log_ml_full = np.log10(np.maximum(train_data["cmax_ml_full"], 1e-10))
    X_v4 = np.column_stack([log_eng, log_ml_full])

    ridge_cv4 = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv4.fit(X_v4, log_obs)
    best_alpha_v4 = ridge_cv4.alpha_

    ridge_v4 = Ridge(alpha=best_alpha_v4)
    ridge_v4.fit(X_v4, log_obs)
    log.info("  V4 (ml_full, alpha=%.2f): intercept=%.4f, b_eng=%.4f, b_ml=%.4f, R²=%.4f",
             best_alpha_v4, ridge_v4.intercept_, ridge_v4.coef_[0],
             ridge_v4.coef_[1], ridge_v4.score(X_v4, log_obs))

    # ── OOF predictions for training set (for Phase 2 ACF) ──
    stacking_oof = np.full(N, np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        ridge_fold = Ridge(alpha=best_alpha_v1)
        ridge_fold.fit(X_v1[train_idx], log_obs[train_idx])
        stacking_oof[val_idx] = ridge_fold.predict(X_v1[val_idx])
    stacking_oof_cmax = 10 ** stacking_oof
    aafe_stacking_oof_train = compute_aafe(stacking_oof_cmax, cmax_obs_train)
    log.info("  Stacking OOF AAFE (training): %.3f", aafe_stacking_oof_train)

    # ── Holdout evaluation ──
    ho_preds = holdout_data["preds"]
    ho_engine = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])

    log_ho_eng = np.log10(np.maximum(ho_engine, 1e-10))
    log_ho_ml = np.log10(np.maximum(ho_ml, 1e-10))

    # V1 holdout
    X_ho_v1 = np.column_stack([log_ho_eng, log_ho_ml])
    ho_stacking_v1_log = ridge_v1.predict(X_ho_v1)
    ho_stacking_v1 = 10 ** ho_stacking_v1_log

    # V2 holdout (+ dose)
    ho_doses = []
    with open(ROOT / "data/reference/clinical_pk.json") as f:
        cpk = json.load(f)
    for h in ho_preds:
        entry = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        if entry:
            ho_doses.append(entry.get("dose_mg", 100.0))
        else:
            ho_doses.append(100.0)
    ho_doses = np.array(ho_doses, dtype=float)
    log_ho_dose = np.log10(np.maximum(ho_doses, 0.01))
    X_ho_v2 = np.column_stack([log_ho_eng, log_ho_ml, log_ho_dose])
    ho_stacking_v2_log = ridge_v2.predict(X_ho_v2)
    ho_stacking_v2 = 10 ** ho_stacking_v2_log

    # V3 holdout (+ compound_type)
    ho_is_base = (ho_types == "base").astype(float)
    ho_is_acid = (ho_types == "acid").astype(float)
    X_ho_v3 = np.column_stack([log_ho_eng, log_ho_ml, ho_is_base, ho_is_acid])
    ho_stacking_v3_log = ridge_v3.predict(X_ho_v3)
    ho_stacking_v3 = 10 ** ho_stacking_v3_log

    # V4 holdout (ml_full-trained Ridge applied to holdout ml)
    # Note: V4 was trained on ml_full (in-sample), so it expects ml_full-scale inputs
    # Holdout ML predictions are from the full model, so this should transfer cleanly
    X_ho_v4 = np.column_stack([log_ho_eng, log_ho_ml])
    ho_stacking_v4_log = ridge_v4.predict(X_ho_v4)
    ho_stacking_v4 = 10 ** ho_stacking_v4_log

    # Metrics
    base_mask = ho_types == "base"
    nonbase_mask = ~base_mask

    def report(label, pred):
        aafe_all = compute_aafe(pred, ho_obs)
        aafe_base = compute_aafe(pred[base_mask], ho_obs[base_mask])
        aafe_nonbase = compute_aafe(pred[nonbase_mask], ho_obs[nonbase_mask])
        p2f = pct_2fold(pred, ho_obs)
        log.info("  %s: AAFE=%.3f (base=%.3f, non-base=%.3f), %%2-fold=%.1f%%",
                 label, aafe_all, aafe_base, aafe_nonbase, p2f)
        return {"aafe": aafe_all, "aafe_base": aafe_base,
                "aafe_nonbase": aafe_nonbase, "pct2fold": p2f}

    log.info("─── Holdout results ───")
    r_baseline_meta = report("Baseline Meta", ho_combined)
    r_stacking_v1 = report("Stacking V1 (eng+ml_oof)", ho_stacking_v1)
    r_stacking_v2 = report("Stacking V2 (+dose)", ho_stacking_v2)
    r_stacking_v3 = report("Stacking V3 (+type)", ho_stacking_v3)
    r_stacking_v4 = report("Stacking V4 (ml_full)", ho_stacking_v4)

    return {
        "ridge_v1": ridge_v1,
        "ridge_v2": ridge_v2,
        "ridge_v3": ridge_v3,
        "ridge_v4": ridge_v4,
        "best_alpha_v1": best_alpha_v1,
        "best_alpha_v2": best_alpha_v2,
        "best_alpha_v3": best_alpha_v3,
        "best_alpha_v4": best_alpha_v4,
        "ho_stacking_v1": ho_stacking_v1,
        "ho_stacking_v2": ho_stacking_v2,
        "ho_stacking_v3": ho_stacking_v3,
        "ho_stacking_v4": ho_stacking_v4,
        "stacking_oof": stacking_oof,  # log10 OOF predictions for Phase 2
        "stacking_oof_cmax": stacking_oof_cmax,
        "metrics_v1": r_stacking_v1,
        "metrics_v2": r_stacking_v2,
        "metrics_v3": r_stacking_v3,
        "metrics_v4": r_stacking_v4,
        "metrics_baseline": r_baseline_meta,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1B: Analog Correction Factor (ACF)
# ═══════════════════════════════════════════════════════════════════════════

def phase1b_acf(train_data, holdout_data):
    log.info("=" * 70)
    log.info("Phase 1B: Analog Correction Factor (ACF)")
    log.info("=" * 70)

    N = train_data["N"]
    cmax_meta_oof = train_data["cmax_meta_oof"]
    cmax_obs_train = train_data["cmax_obs"]
    train_neighbors = train_data["train_neighbors"]

    # Fold errors of meta-learner (using OOF ML) on training set
    # fold_error > 0 means meta over-predicts
    meta_fe = np.log10(np.maximum(cmax_meta_oof, 1e-10) /
                        np.maximum(cmax_obs_train, 1e-10))

    log.info("  Meta fold error (training, OOF): mean=%.3f, std=%.3f, range=[%.2f, %.2f]",
             meta_fe.mean(), meta_fe.std(), meta_fe.min(), meta_fe.max())

    # Holdout ACF
    ho_preds = holdout_data["preds"]
    ho_neighbors = holdout_data["neighbors"]
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])

    base_mask = ho_types == "base"
    nonbase_mask = ~base_mask

    k_values = [3, 5, 7, 10]
    results = {}

    for k in k_values:
        corrections_wmedian = []
        corrections_wmean = []

        for h in ho_preds:
            name = h["name"]
            nbs = ho_neighbors.get(name, [])[:k]
            if not nbs:
                corrections_wmedian.append(0.0)
                corrections_wmean.append(0.0)
                continue

            nb_idx = [n[0] for n in nbs]
            nb_sim = np.array([n[1] for n in nbs])

            # Neighbor fold errors
            nb_fe = meta_fe[nb_idx]

            # Weighted median
            if nb_sim.sum() > 0:
                corr_wmed = weighted_median(nb_fe, nb_sim)
            else:
                corr_wmed = 0.0
            corrections_wmedian.append(corr_wmed)

            # Weighted mean
            if nb_sim.sum() > 0:
                corr_wmean = float(np.average(nb_fe, weights=nb_sim))
            else:
                corr_wmean = 0.0
            corrections_wmean.append(corr_wmean)

        corrections_wmedian = np.array(corrections_wmedian)
        corrections_wmean = np.array(corrections_wmean)

        # Apply correction: log10(cmax_corrected) = log10(cmax_meta) - correction
        log_meta = np.log10(np.maximum(ho_combined, 1e-10))
        cmax_acf_wmedian = 10 ** (log_meta - corrections_wmedian)
        cmax_acf_wmean = 10 ** (log_meta - corrections_wmean)

        aafe_wmed = compute_aafe(cmax_acf_wmedian, ho_obs)
        aafe_wmean = compute_aafe(cmax_acf_wmean, ho_obs)
        p2f_wmed = pct_2fold(cmax_acf_wmedian, ho_obs)
        p2f_wmean = pct_2fold(cmax_acf_wmean, ho_obs)

        aafe_wmed_base = compute_aafe(cmax_acf_wmedian[base_mask], ho_obs[base_mask])
        aafe_wmed_nonbase = compute_aafe(cmax_acf_wmedian[nonbase_mask], ho_obs[nonbase_mask])

        log.info("  k=%d: wmedian AAFE=%.3f (base=%.3f, nonbase=%.3f, %%2f=%.1f%%)  |  "
                 "wmean AAFE=%.3f (%%2f=%.1f%%)",
                 k, aafe_wmed, aafe_wmed_base, aafe_wmed_nonbase, p2f_wmed,
                 aafe_wmean, p2f_wmean)

        results[k] = {
            "cmax_wmedian": cmax_acf_wmedian,
            "cmax_wmean": cmax_acf_wmean,
            "corrections_wmedian": corrections_wmedian,
            "corrections_wmean": corrections_wmean,
            "aafe_wmedian": aafe_wmed,
            "aafe_wmean": aafe_wmean,
            "aafe_wmedian_base": aafe_wmed_base,
            "aafe_wmedian_nonbase": aafe_wmed_nonbase,
            "pct2fold_wmedian": p2f_wmed,
        }

    log.info("  Correction distribution (k=5, wmedian): mean=%.3f, std=%.3f",
             results[5]["corrections_wmedian"].mean(),
             results[5]["corrections_wmedian"].std())

    # Primary: k=5, weighted median
    primary = results[5]
    log.info("  ★ Primary (k=5, wmedian): AAFE=%.3f, %%2-fold=%.1f%%",
             primary["aafe_wmedian"], primary["pct2fold_wmedian"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1C: Winsorized Meta-Learner
# ═══════════════════════════════════════════════════════════════════════════

def phase1c_winsorized(holdout_data):
    log.info("=" * 70)
    log.info("Phase 1C: Winsorized Meta-Learner")
    log.info("=" * 70)

    ho_preds = holdout_data["preds"]
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_engine = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])

    base_mask = ho_types == "base"
    nonbase_mask = ~base_mask
    n_base = base_mask.sum()
    n_nonbase = nonbase_mask.sum()

    cap_values = [0.3, 0.5, 0.7, 1.0]
    results = {}

    for cap_log in cap_values:
        cmax_winsorized = np.zeros(len(ho_preds))
        n_clipped = 0

        for i, h in enumerate(ho_preds):
            ce = h["cmax_engine"]
            cm = h["cmax_ml"]
            ct = h["compound_type"]
            is_base = (ct == "base")
            w_eng = 0.45 if is_base else 0.00
            w_ml = 0.55 if is_base else 1.00

            if w_eng > 0 and ce > 0 and cm > 0:
                log_e = np.log10(max(ce, 1e-10))
                log_m = np.log10(max(cm, 1e-10))
                disagreement = log_e - log_m

                if abs(disagreement) > cap_log:
                    n_clipped += 1

                clipped = np.clip(disagreement, -cap_log, cap_log)
                log_e_clipped = log_m + clipped
                e_clipped = 10 ** log_e_clipped

                # Also apply disagreement penalty (as in production)
                disag_abs = abs(clipped)
                w_e, w_m = w_eng, w_ml
                if disag_abs > 1.0:
                    scale = 1.0 / disag_abs
                    w_e *= scale
                    total = w_e + w_m
                    w_e /= total
                    w_m /= total

                log_cmax = w_e * log_e_clipped + w_m * log_m
                cmax_winsorized[i] = 10 ** log_cmax
            else:
                cmax_winsorized[i] = cm

        aafe_all = compute_aafe(cmax_winsorized, ho_obs)
        aafe_base = compute_aafe(cmax_winsorized[base_mask], ho_obs[base_mask])
        aafe_nonbase = compute_aafe(cmax_winsorized[nonbase_mask], ho_obs[nonbase_mask])
        p2f = pct_2fold(cmax_winsorized, ho_obs)

        log.info("  cap=%.1f (%.1fx): AAFE=%.3f (base=%.3f, nonbase=%.3f), "
                 "%%2-fold=%.1f%%, clipped=%d/%d base",
                 cap_log, 10**cap_log, aafe_all, aafe_base, aafe_nonbase,
                 p2f, n_clipped, n_base)

        results[cap_log] = {
            "cmax": cmax_winsorized,
            "aafe": aafe_all,
            "aafe_base": aafe_base,
            "aafe_nonbase": aafe_nonbase,
            "pct2fold": p2f,
            "n_clipped": n_clipped,
        }

    primary = results[0.5]
    log.info("  ★ Primary (cap=0.5): AAFE=%.3f, %%2-fold=%.1f%%",
             primary["aafe"], primary["pct2fold"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1D: Comparison Table
# ═══════════════════════════════════════════════════════════════════════════

def phase1d_comparison(results_1a, results_1b, results_1c, holdout_data):
    log.info("=" * 70)
    log.info("Phase 1D: Independent Results Comparison")
    log.info("=" * 70)

    ho_preds = holdout_data["preds"]
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    base_mask = ho_types == "base"

    baseline_ml_aafe = compute_aafe(ho_ml, ho_obs)
    baseline_ml_p2f = pct_2fold(ho_ml, ho_obs)
    baseline_meta_aafe = compute_aafe(ho_combined, ho_obs)
    baseline_meta_p2f = pct_2fold(ho_combined, ho_obs)

    rows = [
        ("Baseline ML", baseline_ml_aafe,
         compute_aafe(ho_ml[base_mask], ho_obs[base_mask]),
         compute_aafe(ho_ml[~base_mask], ho_obs[~base_mask]),
         baseline_ml_p2f, 0),
        ("Baseline Meta", baseline_meta_aafe,
         compute_aafe(ho_combined[base_mask], ho_obs[base_mask]),
         compute_aafe(ho_combined[~base_mask], ho_obs[~base_mask]),
         baseline_meta_p2f, 2),
        ("OOF Stacking V1", results_1a["metrics_v1"]["aafe"],
         results_1a["metrics_v1"]["aafe_base"],
         results_1a["metrics_v1"]["aafe_nonbase"],
         results_1a["metrics_v1"]["pct2fold"], 3),
        ("OOF Stacking V2 (+dose)", results_1a["metrics_v2"]["aafe"],
         results_1a["metrics_v2"]["aafe_base"],
         results_1a["metrics_v2"]["aafe_nonbase"],
         results_1a["metrics_v2"]["pct2fold"], 4),
        ("OOF Stacking V3 (+type)", results_1a["metrics_v3"]["aafe"],
         results_1a["metrics_v3"]["aafe_base"],
         results_1a["metrics_v3"]["aafe_nonbase"],
         results_1a["metrics_v3"]["pct2fold"], 5),
        ("Stacking V4 (ml_full)", results_1a["metrics_v4"]["aafe"],
         results_1a["metrics_v4"]["aafe_base"],
         results_1a["metrics_v4"]["aafe_nonbase"],
         results_1a["metrics_v4"]["pct2fold"], 3),
        ("ACF k=5 wmedian", results_1b[5]["aafe_wmedian"],
         results_1b[5]["aafe_wmedian_base"],
         results_1b[5]["aafe_wmedian_nonbase"],
         results_1b[5]["pct2fold_wmedian"], 0),
        ("Winsorized cap=0.5", results_1c[0.5]["aafe"],
         results_1c[0.5]["aafe_base"],
         results_1c[0.5]["aafe_nonbase"],
         results_1c[0.5]["pct2fold"], 1),
    ]

    header = f"{'Method':<25} {'AAFE(all)':>10} {'AAFE(base)':>11} {'AAFE(~base)':>12} {'%2-fold':>8} {'Params':>6}"
    log.info("")
    log.info(header)
    log.info("-" * len(header))
    for name, aafe_all, aafe_base, aafe_nb, p2f, params in rows:
        delta = aafe_all - baseline_meta_aafe
        sign = "+" if delta >= 0 else ""
        log.info(f"  {name:<23} {aafe_all:>10.3f} {aafe_base:>11.3f} {aafe_nb:>12.3f} "
                 f"{p2f:>7.1f}% {params:>6d}  (Δ={sign}{delta:.3f})")
    log.info("")

    # Any beat baseline?
    best_method = min(rows, key=lambda x: x[1])
    if best_method[1] < baseline_meta_aafe:
        log.info("  ✓ Best method: %s (AAFE %.3f, Δ=%.3f)",
                 best_method[0], best_method[1], best_method[1] - baseline_meta_aafe)
    else:
        log.info("  ✗ No method beat baseline meta (%.3f)", baseline_meta_aafe)

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Integration — Stacking → ACF
# ═══════════════════════════════════════════════════════════════════════════

def phase2_integration(train_data, holdout_data, results_1a):
    log.info("=" * 70)
    log.info("Phase 2: Integration — Stacking → ACF")
    log.info("=" * 70)

    N = train_data["N"]
    cmax_obs_train = train_data["cmax_obs"]
    train_neighbors = train_data["train_neighbors"]
    stacking_oof = results_1a["stacking_oof"]  # log10 predictions
    stacking_oof_cmax = results_1a["stacking_oof_cmax"]

    # Stacking OOF fold errors on training set
    stacking_fe = stacking_oof - np.log10(np.maximum(cmax_obs_train, 1e-10))
    log.info("  Stacking OOF fold error: mean=%.3f, std=%.3f", stacking_fe.mean(), stacking_fe.std())

    # Holdout: Stacking V1 predictions + ACF correction
    ho_preds = holdout_data["preds"]
    ho_neighbors = holdout_data["neighbors"]
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_stacking_v1 = results_1a["ho_stacking_v1"]

    base_mask = ho_types == "base"

    k_values = [3, 5, 7, 10]
    results = {}

    for k in k_values:
        corrections = []
        for h in ho_preds:
            name = h["name"]
            nbs = ho_neighbors.get(name, [])[:k]
            if not nbs:
                corrections.append(0.0)
                continue
            nb_idx = [n[0] for n in nbs]
            nb_sim = np.array([n[1] for n in nbs])
            nb_fe = stacking_fe[nb_idx]
            if nb_sim.sum() > 0:
                corr = weighted_median(nb_fe, nb_sim)
            else:
                corr = 0.0
            corrections.append(corr)

        corrections = np.array(corrections)
        log_stacking = np.log10(np.maximum(ho_stacking_v1, 1e-10))
        cmax_combined = 10 ** (log_stacking - corrections)

        aafe_all = compute_aafe(cmax_combined, ho_obs)
        aafe_base = compute_aafe(cmax_combined[base_mask], ho_obs[base_mask])
        aafe_nonbase = compute_aafe(cmax_combined[~base_mask], ho_obs[~base_mask])
        p2f = pct_2fold(cmax_combined, ho_obs)

        log.info("  Stacking+ACF k=%d: AAFE=%.3f (base=%.3f, nonbase=%.3f), %%2-fold=%.1f%%",
                 k, aafe_all, aafe_base, aafe_nonbase, p2f)

        results[k] = {
            "cmax": cmax_combined,
            "aafe": aafe_all,
            "aafe_base": aafe_base,
            "aafe_nonbase": aafe_nonbase,
            "pct2fold": p2f,
            "corrections": corrections,
        }

    # Summary table
    stacking_aafe = results_1a["metrics_v1"]["aafe"]
    baseline_aafe = results_1a["metrics_baseline"]["aafe"]
    log.info("")
    log.info("  %-25s  AAFE   Δ vs Meta", "Method")
    log.info("  " + "-" * 50)
    log.info("  %-25s  %.3f  (baseline)", "Baseline Meta", baseline_aafe)
    log.info("  %-25s  %.3f  %+.3f", "OOF Stacking V1", stacking_aafe, stacking_aafe - baseline_aafe)
    for k in k_values:
        r = results[k]
        log.info("  %-25s  %.3f  %+.3f", f"Stacking+ACF k={k}",
                 r["aafe"], r["aafe"] - baseline_aafe)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Error Analysis
# ═══════════════════════════════════════════════════════════════════════════

def phase3_error_analysis(train_data, holdout_data, results_1a, results_1b, results_2):
    log.info("=" * 70)
    log.info("Phase 3: Error Analysis")
    log.info("=" * 70)

    ho_preds = holdout_data["preds"]
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])

    ho_stacking = results_1a["ho_stacking_v1"]
    ho_acf_k5 = results_1b[5]["cmax_wmedian"]
    ho_stack_acf_k5 = results_2[5]["cmax"]

    # Per-drug fold errors (log10 scale)
    fe_meta = np.log10(np.maximum(ho_combined, 1e-10) / np.maximum(ho_obs, 1e-10))
    fe_stacking = np.log10(np.maximum(ho_stacking, 1e-10) / np.maximum(ho_obs, 1e-10))
    fe_acf = np.log10(np.maximum(ho_acf_k5, 1e-10) / np.maximum(ho_obs, 1e-10))
    fe_combined = np.log10(np.maximum(ho_stack_acf_k5, 1e-10) / np.maximum(ho_obs, 1e-10))

    # Error correlation matrix
    errors = np.column_stack([fe_meta, fe_stacking, fe_acf, fe_combined])
    names = ["Meta", "Stacking", "ACF", "Stack+ACF"]
    log.info("  Error correlation matrix (log fold error):")
    corr = np.corrcoef(errors.T)
    header = "          " + "".join(f"{n:>10}" for n in names)
    log.info("  %s", header)
    for i, n in enumerate(names):
        row = f"  {n:<10}" + "".join(f"{corr[i,j]:>10.3f}" for j in range(len(names)))
        log.info("%s", row)

    # Improved vs worsened drugs (Stacking V1 vs Meta)
    improved = np.abs(fe_stacking) < np.abs(fe_meta)
    worsened = np.abs(fe_stacking) > np.abs(fe_meta)
    log.info("")
    log.info("  Stacking V1 vs Meta: %d improved, %d worsened, %d unchanged",
             improved.sum(), worsened.sum(), (~improved & ~worsened).sum())

    # Characteristics of improved vs worsened
    for label, mask in [("Improved", improved), ("Worsened", worsened)]:
        drugs = [ho_preds[i] for i in range(len(ho_preds)) if mask[i]]
        types = [d["compound_type"] for d in drugs]
        ad_flags = sum(1 for d in drugs if d.get("ad_flags"))
        log.info("  %s (%d drugs): base=%d, acid=%d, neutral=%d, AD-flagged=%d",
                 label, len(drugs),
                 sum(1 for t in types if t == "base"),
                 sum(1 for t in types if t == "acid"),
                 sum(1 for t in types if t == "neutral"),
                 ad_flags)

    # Nearest-neighbor similarity statistics for improved vs worsened
    ho_neighbors = holdout_data["neighbors"]
    for label, mask in [("Improved", improved), ("Worsened", worsened)]:
        sims = []
        for i, h in enumerate(ho_preds):
            if mask[i]:
                nbs = ho_neighbors.get(h["name"], [])
                if nbs:
                    sims.append(nbs[0][1])  # top-1 similarity
        if sims:
            log.info("  %s top-1 Tanimoto: median=%.3f, mean=%.3f",
                     label, np.median(sims), np.mean(sims))

    # Per-drug results table (first 20 drugs sorted by improvement)
    improvement = np.abs(fe_meta) - np.abs(fe_stacking)
    sorted_idx = np.argsort(improvement)[::-1]
    log.info("")
    log.info("  Top 10 most improved (Stacking vs Meta):")
    log.info("  %-20s %8s %8s %8s %8s %8s", "Drug", "Obs", "Meta", "Stack", "FE_meta", "FE_stack")
    for i in sorted_idx[:10]:
        h = ho_preds[i]
        log.info("  %-20s %8.3f %8.3f %8.3f %+8.3f %+8.3f",
                 h["name"][:20], ho_obs[i], ho_combined[i], ho_stacking[i],
                 fe_meta[i], fe_stacking[i])

    log.info("")
    log.info("  Top 10 most worsened (Stacking vs Meta):")
    for i in sorted_idx[-10:]:
        h = ho_preds[i]
        log.info("  %-20s %8.3f %8.3f %8.3f %+8.3f %+8.3f",
                 h["name"][:20], ho_obs[i], ho_combined[i], ho_stacking[i],
                 fe_meta[i], fe_stacking[i])

    return {
        "error_corr": corr.tolist(),
        "n_improved": int(improved.sum()),
        "n_worsened": int(worsened.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results_1a, results_1b, results_1c, comparison, results_2, results_3):
    out = {
        "phase1a_stacking": {
            "v1": {
                "intercept": float(results_1a["ridge_v1"].intercept_),
                "coef_engine": float(results_1a["ridge_v1"].coef_[0]),
                "coef_ml": float(results_1a["ridge_v1"].coef_[1]),
                "alpha": results_1a["best_alpha_v1"],
                **results_1a["metrics_v1"],
            },
            "v2": {
                "intercept": float(results_1a["ridge_v2"].intercept_),
                "coef_engine": float(results_1a["ridge_v2"].coef_[0]),
                "coef_ml": float(results_1a["ridge_v2"].coef_[1]),
                "coef_dose": float(results_1a["ridge_v2"].coef_[2]),
                "alpha": results_1a["best_alpha_v2"],
                **results_1a["metrics_v2"],
            },
            "v3": {
                "intercept": float(results_1a["ridge_v3"].intercept_),
                "coef_engine": float(results_1a["ridge_v3"].coef_[0]),
                "coef_ml": float(results_1a["ridge_v3"].coef_[1]),
                "coef_base": float(results_1a["ridge_v3"].coef_[2]),
                "coef_acid": float(results_1a["ridge_v3"].coef_[3]),
                "alpha": results_1a["best_alpha_v3"],
                **results_1a["metrics_v3"],
            },
            "v4_ml_full": {
                "intercept": float(results_1a["ridge_v4"].intercept_),
                "coef_engine": float(results_1a["ridge_v4"].coef_[0]),
                "coef_ml": float(results_1a["ridge_v4"].coef_[1]),
                "alpha": results_1a["best_alpha_v4"],
                **results_1a["metrics_v4"],
            },
        },
        "phase1b_acf": {
            str(k): {
                "aafe_wmedian": v["aafe_wmedian"],
                "aafe_wmean": v["aafe_wmean"],
                "aafe_wmedian_base": v["aafe_wmedian_base"],
                "aafe_wmedian_nonbase": v["aafe_wmedian_nonbase"],
                "pct2fold_wmedian": v["pct2fold_wmedian"],
            }
            for k, v in results_1b.items()
        },
        "phase1c_winsorized": {
            str(k): {
                "aafe": v["aafe"],
                "aafe_base": v["aafe_base"],
                "aafe_nonbase": v["aafe_nonbase"],
                "pct2fold": v["pct2fold"],
                "n_clipped": v["n_clipped"],
            }
            for k, v in results_1c.items()
        },
        "phase1d_comparison": [
            {"method": row[0], "aafe_all": row[1], "aafe_base": row[2],
             "aafe_nonbase": row[3], "pct2fold": row[4], "params": row[5]}
            for row in comparison
        ],
        "phase2_stacking_acf": {
            str(k): {
                "aafe": v["aafe"],
                "aafe_base": v["aafe_base"],
                "aafe_nonbase": v["aafe_nonbase"],
                "pct2fold": v["pct2fold"],
            }
            for k, v in results_2.items()
        },
        "phase3_error_analysis": results_3,
    }

    outpath = ROOT / "data/validation/posthoc_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Results saved to %s", outpath)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # Phase 0
    train_data, holdout_data = phase0()

    # Phase 1
    results_1a = phase1a_stacking(train_data, holdout_data)
    results_1b = phase1b_acf(train_data, holdout_data)
    results_1c = phase1c_winsorized(holdout_data)
    comparison = phase1d_comparison(results_1a, results_1b, results_1c, holdout_data)

    # Phase 2 (always run, even if individual methods didn't beat baseline)
    results_2 = phase2_integration(train_data, holdout_data, results_1a)

    # Phase 3
    results_3 = phase3_error_analysis(train_data, holdout_data,
                                       results_1a, results_1b, results_2)

    # Save
    save_results(results_1a, results_1b, results_1c, comparison, results_2, results_3)

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("Total time: %.1f seconds", elapsed)

    # Final verdict
    baseline = results_1a["metrics_baseline"]["aafe"]
    best_aafe = min(
        results_1a["metrics_v1"]["aafe"],
        results_1a["metrics_v2"]["aafe"],
        results_1a["metrics_v3"]["aafe"],
        results_1a["metrics_v4"]["aafe"],
        results_1b[5]["aafe_wmedian"],
        results_1c[0.5]["aafe"],
        results_2[5]["aafe"],
    )
    delta = best_aafe - baseline
    log.info("Baseline Meta AAFE: %.3f", baseline)
    log.info("Best AAFE achieved: %.3f (Δ = %+.3f)", best_aafe, delta)

    if delta < -0.05:
        log.info("▶ MEANINGFUL IMPROVEMENT (Δ > 0.05)")
    elif delta < -0.02:
        log.info("▶ MARGINAL (0.02 < Δ < 0.05) — report but don't replace")
    else:
        log.info("▶ NOISE LEVEL (Δ < 0.02) — 23rd negative result")


if __name__ == "__main__":
    main()
