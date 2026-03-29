#!/usr/bin/env python3
"""BDE Reactivity Features for CLint Prediction — Phase 1 complete pipeline.

Phase 1A: ALFABET BDE calculation for TDC training compounds
Phase 1B: BDE feature engineering (10 features × 2 sets)
Phase 1C: Redundancy check (SMARTS proxy vs Morgan FP)
Phase 1D: BDE ↔ CLint correlation analysis (gate: |r| > 0.15)
Phase 1E: CLint prediction model comparison (if gate passes)

Usage:
    TF_USE_LEGACY_KERAS=1 python3 scripts/bde_clint_validation.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TDC_HEP_TAB = ROOT / "data" / "training" / "clearance_hepatocyte_az.tab"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
OUTPUT_DIR = ROOT / "data" / "validation"
BDE_FEATURES_PATH = ROOT / "data" / "features" / "bde_features.csv"
BDE_RESULTS_PATH = OUTPUT_DIR / "bde_correlation.json"


# ---------------------------------------------------------------------------
# Holdout exclusion (same as clint_classification.py)
# ---------------------------------------------------------------------------
def _canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def _inchikey_prefix(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    inchi = MolToInchi(mol)
    if not inchi:
        return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


def build_holdout_keys():
    with open(HOLDOUT_JSON) as f:
        hd = json.load(f)
    with open(CLINICAL_PK_JSON) as f:
        cp = json.load(f)
    all_names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    ho_smiles, ho_ik = set(), set()
    for name in all_names:
        e = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if not e:
            continue
        s = e.get("smiles", "")
        if not s:
            continue
        c = _canonical_smiles(s)
        if c:
            ho_smiles.add(c)
        i = _inchikey_prefix(s)
        if i:
            ho_ik.add(i)
    return {"smiles": ho_smiles, "ik": ho_ik}


def is_holdout(smiles, ho):
    c = _canonical_smiles(smiles)
    if c and c in ho["smiles"]:
        return True
    i = _inchikey_prefix(smiles)
    if i and i in ho["ik"]:
        return True
    return False


def scaffold_split(smiles_list, n_folds=5, seed=42):
    s2i = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys())
    rng.shuffle(scs)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scs):
        folds[i % n_folds].extend(s2i[sc])
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1A: ALFABET BDE Calculation
# ═══════════════════════════════════════════════════════════════════════════
def compute_bde_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BDE features for all compounds using ALFABET."""
    log.info("=" * 60)
    log.info("Phase 1A: ALFABET BDE Calculation")
    log.info("=" * 60)

    from alfabet import model as alfabet_model

    smiles_list = df["canon"].tolist()
    n_total = len(smiles_list)

    # Process in batches to avoid memory issues
    batch_size = 100
    all_features = []
    n_failed = 0
    n_no_ch = 0

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_smiles = smiles_list[batch_start:batch_end]

        try:
            results = alfabet_model.predict(batch_smiles)
        except Exception as e:
            log.warning("ALFABET batch failed [%d:%d]: %s", batch_start, batch_end, e)
            for smi in batch_smiles:
                all_features.append({"canon": smi, "bde_failed": True})
                n_failed += 1
            continue

        for smi in batch_smiles:
            mol_results = results[results["molecule"] == smi]
            if len(mol_results) == 0:
                all_features.append({"canon": smi, "bde_failed": True})
                n_failed += 1
                continue

            valid = mol_results[mol_results["is_valid"] == True]

            # All C-H bonds
            ch_all = valid[valid["bond_type"] == "C-H"]

            # Aliphatic C-H bonds (BDE < 105 kcal/mol as proxy — aromatic C-H > 108)
            ch_aliph = ch_all[ch_all["bde_pred"] < 105]

            feat = {"canon": smi, "bde_failed": False}

            for suffix, ch_df in [("_all", ch_all), ("_aliph", ch_aliph)]:
                bde_vals = ch_df["bde_pred"].values if len(ch_df) > 0 else np.array([])

                if len(bde_vals) == 0:
                    # No C-H bonds of this type
                    feat[f"BDE_min{suffix}"] = np.nan
                    feat[f"BDE_2nd{suffix}"] = np.nan
                    feat[f"BDE_3rd{suffix}"] = np.nan
                    feat[f"BDE_mean{suffix}"] = np.nan
                    feat[f"BDE_std{suffix}"] = np.nan
                    feat[f"BDE_range{suffix}"] = np.nan
                    feat[f"N_vuln_85{suffix}"] = 0
                    feat[f"N_vuln_90{suffix}"] = 0
                    feat[f"N_CH{suffix}"] = 0
                    feat[f"BDE_min_frac{suffix}"] = np.nan
                else:
                    sorted_bde = np.sort(bde_vals)
                    feat[f"BDE_min{suffix}"] = float(sorted_bde[0])
                    feat[f"BDE_2nd{suffix}"] = float(sorted_bde[1]) if len(sorted_bde) > 1 else float(sorted_bde[0])
                    feat[f"BDE_3rd{suffix}"] = float(sorted_bde[2]) if len(sorted_bde) > 2 else float(sorted_bde[-1])
                    feat[f"BDE_mean{suffix}"] = float(np.mean(bde_vals))
                    feat[f"BDE_std{suffix}"] = float(np.std(bde_vals)) if len(bde_vals) > 1 else 0.0
                    feat[f"BDE_range{suffix}"] = float(np.max(bde_vals) - np.min(bde_vals))
                    feat[f"N_vuln_85{suffix}"] = int(np.sum(bde_vals < 85))
                    feat[f"N_vuln_90{suffix}"] = int(np.sum(bde_vals < 90))
                    feat[f"N_CH{suffix}"] = len(bde_vals)
                    feat[f"BDE_min_frac{suffix}"] = float(sorted_bde[0] / np.mean(bde_vals))

            if feat.get("N_CH_all", 0) == 0:
                n_no_ch += 1

            all_features.append(feat)

        if (batch_end) % 200 == 0 or batch_end == n_total:
            log.info("  BDE: %d/%d (failed=%d, no_CH=%d)", batch_end, n_total, n_failed, n_no_ch)

    bde_df = pd.DataFrame(all_features)
    log.info("BDE computation complete: %d total, %d failed, %d no C-H bonds",
             n_total, n_failed, n_no_ch)

    return bde_df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1C: Redundancy Check
# ═══════════════════════════════════════════════════════════════════════════
def redundancy_check(df: pd.DataFrame) -> dict:
    """Check if Morgan FP already encodes BDE-vulnerable substructures."""
    log.info("=" * 60)
    log.info("Phase 1C: Redundancy Check (SMARTS proxy)")
    log.info("=" * 60)

    # SMARTS for BDE-vulnerable substructures
    smarts_patterns = {
        "benzylic_CH": "[c]~[CH2,CH]",
        "allylic_CH": "[C]=[C]~[CH]",
        "alpha_heteroatom": "[N,O,S]~[CH]",
        "tertiary_CH": "[CH]([C])([C])[C]",
        "aldehyde_CH": "[CH]=O",
    }

    features = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canon"])
        if mol is None:
            features.append({k: 0 for k in smarts_patterns})
            continue
        feat = {}
        for name, smarts in smarts_patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            feat[name] = 1 if mol.HasSubstructMatch(pat) else 0
        features.append(feat)

    proxy_df = pd.DataFrame(features)
    X_proxy = proxy_df.values.astype(float)
    y = np.log10(np.maximum(df["Y"].values, 0.1))

    # Linear regression R²
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    lr = LinearRegression()
    # Simple 5-fold CV (not scaffold — just for quick proxy check)
    scores = cross_val_score(lr, X_proxy, y, cv=5, scoring="r2")
    r2_proxy = float(np.mean(scores))

    log.info("SMARTS proxy R² (5-fold CV): %.4f", r2_proxy)
    for name in smarts_patterns:
        prevalence = proxy_df[name].mean()
        log.info("  %s: %.1f%% prevalence", name, prevalence * 100)

    if r2_proxy > 0.10:
        log.info("  → R²>0.10: Morgan FP likely already has BDE proxy information")
    else:
        log.info("  → R²<0.10: BDE proxy weak in Morgan FP. ALFABET BDE may add new info.")

    return {"r2_proxy": r2_proxy, "pattern_prevalence": {k: float(proxy_df[k].mean()) for k in smarts_patterns}}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1D: BDE ↔ CLint Correlation
# ═══════════════════════════════════════════════════════════════════════════
def bde_clint_correlation(df: pd.DataFrame, bde_df: pd.DataFrame) -> dict:
    """Analyze BDE ↔ CLint correlation."""
    log.info("=" * 60)
    log.info("Phase 1D: BDE ↔ CLint Correlation Analysis")
    log.info("=" * 60)

    # Merge
    merged = df.merge(bde_df, on="canon", how="inner")
    merged = merged[merged["bde_failed"] == False].reset_index(drop=True)
    merged = merged.dropna(subset=["BDE_min_all"]).reset_index(drop=True)

    log.info("Merged dataset: %d compounds (with valid BDE)", len(merged))

    log_clint = np.log10(np.maximum(merged["Y"].values, 0.1))

    results = {}

    # Overall correlation
    for feat in ["BDE_min_all", "BDE_min_aliph", "BDE_mean_all", "BDE_mean_aliph",
                 "N_vuln_90_all", "N_vuln_90_aliph"]:
        vals = merged[feat].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        from scipy.stats import pearsonr
        r, p = pearsonr(vals[mask], log_clint[mask])
        results[f"r_{feat}"] = float(r)
        results[f"p_{feat}"] = float(p)
        log.info("  %s vs log10(CLint): r=%.3f, p=%.2e (N=%d)", feat, r, p, mask.sum())

    # Check for sign: expect NEGATIVE (lower BDE → higher CLint)
    r_main = results.get("r_BDE_min_all", 0)
    if r_main > 0:
        log.warning("  ⚠ POSITIVE correlation for BDE_min_all (r=%.3f) — unexpected!", r_main)
        log.warning("  Expected negative: lower BDE → more reactive → higher CLint")
        log.warning("  Check: are BDE units kcal/mol? Is CLint direction correct?")

    # CYP-dominant subset (CLint > 10 AND logP > 1 as proxy)
    log.info("-" * 40)
    log.info("CYP-dominant subset analysis:")

    # Proxy 1: CLint > 10 (measurable metabolism)
    cyp_mask1 = merged["Y"] > 10
    log.info("  Proxy 1 (CLint > 10): N=%d", cyp_mask1.sum())
    if cyp_mask1.sum() >= 20:
        vals = merged.loc[cyp_mask1, "BDE_min_all"].values
        y_sub = log_clint[cyp_mask1]
        mask = ~np.isnan(vals)
        if mask.sum() >= 10:
            r, p = pearsonr(vals[mask], y_sub[mask])
            results["r_BDE_min_cyp1"] = float(r)
            results["p_BDE_min_cyp1"] = float(p)
            log.info("    BDE_min vs log10(CLint): r=%.3f, p=%.2e", r, p)

    # Proxy 2: CLint > 10 AND molecular weight from features
    # Use logP > 1 proxy — compute from RDKit
    from rdkit.Chem import Descriptors
    logp_vals = []
    for smi in merged["canon"]:
        mol = Chem.MolFromSmiles(smi)
        logp_vals.append(Descriptors.MolLogP(mol) if mol else 0)
    merged["logP"] = logp_vals

    cyp_mask2 = (merged["Y"] > 10) & (merged["logP"] > 1)
    log.info("  Proxy 2 (CLint>10 & logP>1): N=%d", cyp_mask2.sum())
    if cyp_mask2.sum() >= 20:
        vals = merged.loc[cyp_mask2, "BDE_min_all"].values
        y_sub = log_clint[cyp_mask2]
        mask = ~np.isnan(vals)
        if mask.sum() >= 10:
            r, p = pearsonr(vals[mask], y_sub[mask])
            results["r_BDE_min_cyp2"] = float(r)
            results["p_BDE_min_cyp2"] = float(p)
            log.info("    BDE_min vs log10(CLint): r=%.3f, p=%.2e", r, p)

    # Gate check
    r_overall = abs(results.get("r_BDE_min_all", 0))
    r_cyp = max(abs(results.get("r_BDE_min_cyp1", 0)), abs(results.get("r_BDE_min_cyp2", 0)))

    log.info("-" * 40)
    log.info("GATE CHECK:")
    log.info("  |r| overall: %.3f (threshold: 0.15)", r_overall)
    log.info("  |r| CYP subset: %.3f (threshold: 0.15)", r_cyp)

    gate_passed = r_overall > 0.15 or r_cyp > 0.15
    results["gate_passed"] = gate_passed

    if gate_passed:
        log.info("  ✓ GATE PASSED — proceeding to Phase 1E")
    else:
        log.info("  ✗ GATE FAILED — |r| < 0.15. BDE not informative for CLint.")
        log.info("  This is a valid negative finding for Pharos design.")

    return results, merged


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1E: Model Comparison (only if gate passes)
# ═══════════════════════════════════════════════════════════════════════════
def model_comparison(merged: pd.DataFrame, ho_keys: dict) -> dict:
    """Compare regression/classification models with/without BDE features."""
    log.info("=" * 60)
    log.info("Phase 1E: Model Comparison")
    log.info("=" * 60)

    # Prepare data
    smiles_list, X_morgan_list, X_bde_list, y_list = [], [], [], []
    bde_feat_cols = [c for c in merged.columns if c.startswith("BDE_") or c.startswith("N_")]
    bde_feat_cols = [c for c in bde_feat_cols if c != "bde_failed"]

    for _, row in merged.iterrows():
        try:
            fp = compute_features(row["canon"])
            bde_feats = [float(row[c]) if not np.isnan(row[c]) else 0.0 for c in bde_feat_cols]
            smiles_list.append(row["canon"])
            X_morgan_list.append(fp)
            X_bde_list.append(bde_feats)
            y_list.append(np.log10(max(float(row["Y"]), 0.1)))
        except Exception:
            pass

    X_morgan = np.array(X_morgan_list, dtype=np.float64)
    X_bde = np.array(X_bde_list, dtype=np.float64)
    X_combined = np.hstack([X_morgan, X_bde])
    y = np.array(y_list, dtype=np.float64)
    y_class = np.array([0 if v < np.log10(10) else (1 if v <= np.log10(50) else 2) for v in y])

    log.info("Features: Morgan=%s, BDE=%s, Combined=%s", X_morgan.shape, X_bde.shape, X_combined.shape)

    folds = scaffold_split(smiles_list)

    xgb_reg_p = dict(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                     colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0)
    xgb_cls_p = dict(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                     colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0,
                     objective="multi:softprob", num_class=3, eval_metric="mlogloss")

    def _cv_regression(X, y_reg, label):
        preds = np.zeros(len(y_reg))
        for fi in range(5):
            te = folds[fi]
            tr = [j for f in range(5) if f != fi for j in folds[f]]
            m = xgb.XGBRegressor(**xgb_reg_p)
            m.fit(X[tr], y_reg[tr], verbose=False)
            preds[te] = m.predict(X[te])
        ss_res = np.sum((y_reg - preds) ** 2)
        ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        return r2

    def _cv_classification(X, y_cls, label):
        from sklearn.metrics import accuracy_score, cohen_kappa_score
        preds = np.zeros(len(y_cls), dtype=int)
        for fi in range(5):
            te = folds[fi]
            tr = [j for f in range(5) if f != fi for j in folds[f]]
            m = xgb.XGBClassifier(**xgb_cls_p)
            m.fit(X[tr], y_cls[tr])
            preds[te] = m.predict(X[te])
        acc = accuracy_score(y_cls, preds)
        kappa = cohen_kappa_score(y_cls, preds)
        return acc, kappa

    results = {}

    # A. Morgan FP only, Regression
    r2_a = _cv_regression(X_morgan, y, "A: Morgan Regr")
    results["A_morgan_regr_r2"] = r2_a

    # B. Morgan FP only, Classification
    acc_b, kap_b = _cv_classification(X_morgan, y_class, "B: Morgan Class")
    results["B_morgan_class_acc"] = acc_b
    results["B_morgan_class_kappa"] = kap_b

    # C. BDE only, Regression
    r2_c = _cv_regression(X_bde, y, "C: BDE Regr")
    results["C_bde_regr_r2"] = r2_c

    # D. BDE only, Classification
    acc_d, kap_d = _cv_classification(X_bde, y_class, "D: BDE Class")
    results["D_bde_class_acc"] = acc_d
    results["D_bde_class_kappa"] = kap_d

    # E. Morgan + BDE, Regression
    r2_e = _cv_regression(X_combined, y, "E: Morgan+BDE Regr")
    results["E_combined_regr_r2"] = r2_e

    # F. Morgan + BDE, Classification
    acc_f, kap_f = _cv_classification(X_combined, y_class, "F: Morgan+BDE Class")
    results["F_combined_class_acc"] = acc_f
    results["F_combined_class_kappa"] = kap_f

    # Delta
    results["delta_r2"] = r2_e - r2_a
    results["delta_acc"] = acc_f - acc_b

    # Feature importance for combined model
    m_full = xgb.XGBRegressor(**xgb_reg_p)
    m_full.fit(X_combined, y, verbose=False)
    importances = m_full.feature_importances_
    n_morgan = X_morgan.shape[1]
    bde_importances = importances[n_morgan:]
    top_30_idx = np.argsort(importances)[-30:][::-1]
    bde_in_top30 = sum(1 for i in top_30_idx if i >= n_morgan)
    results["bde_in_top30"] = bde_in_top30

    if bde_in_top30 > 0:
        bde_top = [(bde_feat_cols[i - n_morgan], float(importances[i]))
                   for i in top_30_idx if i >= n_morgan]
        results["bde_top_features"] = bde_top

    # Print
    print()
    print("=" * 70)
    print("  MODEL COMPARISON (5-fold Murcko scaffold CV)")
    print("=" * 70)
    print()
    print(f"{'Model':<5s} {'Type':<6s} {'Features':<12s} {'N_feat':>7s} {'R²/Acc':>8s} {'Kappa':>7s}")
    print("-" * 55)
    print(f"{'A':<5s} {'Regr':<6s} {'Morgan FP':<12s} {X_morgan.shape[1]:>7d} {r2_a:>8.3f} {'-':>7s}")
    print(f"{'B':<5s} {'Class':<6s} {'Morgan FP':<12s} {X_morgan.shape[1]:>7d} {acc_b:>8.3f} {kap_b:>7.3f}")
    print(f"{'C':<5s} {'Regr':<6s} {'BDE only':<12s} {X_bde.shape[1]:>7d} {r2_c:>8.3f} {'-':>7s}")
    print(f"{'D':<5s} {'Class':<6s} {'BDE only':<12s} {X_bde.shape[1]:>7d} {acc_d:>8.3f} {kap_d:>7.3f}")
    print(f"{'E':<5s} {'Regr':<6s} {'Morgan+BDE':<12s} {X_combined.shape[1]:>7d} {r2_e:>8.3f} {'-':>7s}")
    print(f"{'F':<5s} {'Class':<6s} {'Morgan+BDE':<12s} {X_combined.shape[1]:>7d} {acc_f:>8.3f} {kap_f:>7.3f}")
    print()
    print(f"ΔR² (E-A):  {r2_e - r2_a:+.3f}  {'✓ BDE adds value' if r2_e - r2_a > 0.03 else '✗ BDE redundant'}")
    print(f"ΔAcc (F-B): {acc_f - acc_b:+.3f}  {'✓ BDE adds value' if acc_f - acc_b > 0.03 else '✗ BDE redundant'}")
    print(f"BDE features in top 30: {bde_in_top30}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "features").mkdir(parents=True, exist_ok=True)

    # Load TDC data
    df = pd.read_csv(TDC_HEP_TAB, sep="\t")
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canonical_smiles)
    df = df.dropna(subset=["canon"]).reset_index(drop=True)
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()

    ho_keys = build_holdout_keys()
    mask = df["canon"].apply(lambda s: is_holdout(s, ho_keys))
    df = df[~mask].reset_index(drop=True)
    log.info("Training data: %d compounds (after holdout exclusion)", len(df))

    # Phase 1A: BDE computation
    bde_df = compute_bde_features(df)

    # Save BDE features
    bde_out = df.merge(bde_df, on="canon", how="inner")
    bde_out.to_csv(BDE_FEATURES_PATH, index=False)
    log.info("BDE features saved to %s", BDE_FEATURES_PATH)

    # Phase 1C: Redundancy check
    redundancy = redundancy_check(df)

    # Phase 1D: Correlation
    correlation, merged = bde_clint_correlation(df, bde_df)

    all_results = {
        "n_total": len(df),
        "n_bde_valid": int((~bde_df["bde_failed"]).sum()),
        "n_bde_failed": int(bde_df["bde_failed"].sum()),
        "redundancy": redundancy,
        "correlation": correlation,
    }

    # Phase 1E: Model comparison (only if gate passes)
    if correlation.get("gate_passed", False):
        model_results = model_comparison(merged, ho_keys)
        all_results["model_comparison"] = model_results
    else:
        log.info("=" * 60)
        log.info("Phase 1E: SKIPPED (gate failed)")
        log.info("=" * 60)
        print()
        print("Phase 1E SKIPPED: BDE ↔ CLint correlation below threshold (|r| < 0.15)")
        print("BDE is not informative for CLint prediction.")

    # Save
    with open(BDE_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", BDE_RESULTS_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
