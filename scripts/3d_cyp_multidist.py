#!/usr/bin/env python3
"""Phase 12b: 3D Conformer Features + CYP Stratification + Multi-Distribution Averaging.

Experiment A: 3D conformer-based features for ML Cmax prediction
Experiment B: CYP enzyme-stratified CLint models
Experiment C: Rodgers-Rowland + Poulin-Theil Kp model averaging
"""
from __future__ import annotations
import json, logging, sys, time, warnings
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

def p2f(p, o):
    m = (p > 0) & (o > 0); fe = p[m] / o[m]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100) if m.sum() else 0.0

def geo(a, b, w):
    return 10 ** (w * np.log10(np.maximum(a, 1e-10)) + (1-w) * np.log10(np.maximum(b, 1e-10)))

def scaffold_split(smi_list, K=5, seed=42):
    s2i = {}
    for i, s in enumerate(smi_list):
        try:
            mol = Chem.MolFromSmiles(s)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except: sc = ""
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
    return InchiToInchiKey(inchi)[:14] if inchi else None

def fe_corr(p1, p2, obs):
    fe1 = np.log10(np.maximum(p1, 1e-10) / np.maximum(obs, 1e-10))
    fe2 = np.log10(np.maximum(p2, 1e-10) / np.maximum(obs, 1e-10))
    return float(np.corrcoef(fe1, fe2)[0, 1])


# ═══════════════════════════════════════════════════════════════
# 3D Feature Extraction
# ═══════════════════════════════════════════════════════════════

def get_3d_features(smiles):
    """Extract 3D conformer features (max 20). Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
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

        # WHIM descriptors (first 9 only, cap at 20 total)
        try:
            whim = rdMolDescriptors.CalcWHIM(mol)
            for i in range(min(9, len(whim))):
                feats[f"whim_{i}"] = whim[i]
        except Exception:
            pass
    except Exception:
        return None

    return feats


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("=" * 70)
    log.info("Data Loading")
    log.info("=" * 70)

    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cpk = json.load(f)
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: ho_iks.add(k)

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
    N = len(smiles)
    log.info("  Training: %d drugs", N)

    # Holdout
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)
    ho_smiles = {}
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"): ho_smiles[n] = e["smiles"]

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

    train = dict(N=N, smiles=smiles, y=y, doses=pbpk["dose_mg"].values,
                 engine=pbpk["cmax_pbpk"].values, obs=pbpk["cmax_obs"].values,
                 compound_type=pbpk["compound_type"].values)
    holdout = dict(N=len(ho_preds), engine=ho_eng, ml=ho_ml, obs=ho_obs,
                   combined=ho_combined, types=ho_types, doses=ho_doses,
                   preds=ho_preds, smiles=ho_smiles, cpk=cpk)
    return train, holdout


# ═══════════════════════════════════════════════════════════════
# Experiment A: 3D Conformer Features
# ═══════════════════════════════════════════════════════════════

def experiment_a(train, holdout):
    log.info("=" * 70)
    log.info("Experiment A: 3D Conformer Features")
    log.info("=" * 70)

    xp = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.5, reg_lambda=3.0,
              random_state=42, n_jobs=4, verbosity=0)

    # A1: Generate 3D features for training
    log.info("  A1: Generating 3D features for %d training drugs...", train["N"])
    t0 = time.time()
    tr_3d_list = []
    n_fail = 0
    for smi in train["smiles"]:
        f3d = get_3d_features(smi)
        if f3d is None:
            n_fail += 1
            tr_3d_list.append(None)
        else:
            tr_3d_list.append(f3d)
    log.info("  3D conformer success: %d/%d (%.1f%%), took %.1fs",
             train["N"] - n_fail, train["N"], (1 - n_fail/train["N"])*100, time.time()-t0)

    # Determine feature names from first successful entry
    feat_names = None
    for f3d in tr_3d_list:
        if f3d is not None:
            feat_names = sorted(f3d.keys())
            break
    if feat_names is None:
        log.error("  No 3D features generated — Experiment A ABORT")
        return {"gate_pass": False, "reason": "no 3D features"}

    n_3d = len(feat_names)
    log.info("  3D feature count: %d (%s...)", n_3d, feat_names[:5])

    # Build 3D feature matrix (fill zeros for failures)
    X_3d_tr = np.zeros((train["N"], n_3d), dtype=np.float32)
    for i, f3d in enumerate(tr_3d_list):
        if f3d is not None:
            for j, fname in enumerate(feat_names):
                X_3d_tr[i, j] = f3d.get(fname, 0.0)

    # Replace NaN/Inf with 0
    X_3d_tr = np.nan_to_num(X_3d_tr, nan=0.0, posinf=0.0, neginf=0.0)

    # Morgan FP
    X_morgan_tr = np.array([compute_features(s) for s in train["smiles"]], dtype=np.float32)

    # Combined features
    X_combined_tr = np.hstack([X_morgan_tr, X_3d_tr])

    # Holdout features
    ho_smi_list = [holdout["smiles"].get(h["name"], "C") for h in holdout["preds"]]
    X_morgan_ho = np.array([compute_features(s) for s in ho_smi_list], dtype=np.float32)

    ho_3d_list = [get_3d_features(s) for s in ho_smi_list]
    X_3d_ho = np.zeros((holdout["N"], n_3d), dtype=np.float32)
    ho_3d_fail = 0
    for i, f3d in enumerate(ho_3d_list):
        if f3d is not None:
            for j, fname in enumerate(feat_names):
                X_3d_ho[i, j] = f3d.get(fname, 0.0)
        else:
            ho_3d_fail += 1
    X_3d_ho = np.nan_to_num(X_3d_ho, nan=0.0, posinf=0.0, neginf=0.0)
    X_combined_ho = np.hstack([X_morgan_ho, X_3d_ho])
    log.info("  Holdout 3D fail: %d/%d", ho_3d_fail, holdout["N"])

    # A2: Train and evaluate configs
    y = train["y"]
    doses_tr, doses_ho = train["doses"], holdout["doses"]
    folds = scaffold_split(train["smiles"])

    configs = [
        ("Morgan_only (baseline)", X_morgan_tr, X_morgan_ho),
        ("3D_only", X_3d_tr, X_3d_ho),
        ("Morgan+3D", X_combined_tr, X_combined_ho),
    ]

    results_a = {}
    for name, X_tr, X_ho in configs:
        # CV
        oof = np.full(train["N"], np.nan)
        for fi, vi in enumerate(folds):
            ti = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
            m = xgb.XGBRegressor(**xp)
            m.fit(X_tr[ti], y[ti])
            oof[vi] = m.predict(X_tr[vi])
        cv_cmax = 10 ** oof * doses_tr
        cv_aafe = aafe(cv_cmax, train["obs"])

        # Holdout
        m_full = xgb.XGBRegressor(**xp)
        m_full.fit(X_tr, y)
        ho_log = m_full.predict(X_ho)
        ho_cmax = np.maximum(10 ** ho_log * doses_ho, 1e-10)
        ho_aafe = aafe(ho_cmax, holdout["obs"])
        r_ml = fe_corr(ho_cmax, holdout["ml"], holdout["obs"])

        # Blend with engine
        ho_blend = np.zeros(holdout["N"])
        for i in range(holdout["N"]):
            w = 0.45 if holdout["types"][i] == "base" else 0.00
            ho_blend[i] = geo(holdout["engine"][i], ho_cmax[i], w)
        blend_aafe = aafe(ho_blend, holdout["obs"])

        log.info("  %s: CV=%.3f, Holdout=%.3f, r(ML)=%.3f, Blended=%.3f",
                 name, cv_aafe, ho_aafe, r_ml, blend_aafe)
        results_a[name] = {
            "cv_aafe": cv_aafe, "holdout_aafe": ho_aafe,
            "r_ml": r_ml, "blend_aafe": blend_aafe,
        }

    baseline_ml = results_a["Morgan_only (baseline)"]["holdout_aafe"]
    combined_ml = results_a["Morgan+3D"]["holdout_aafe"]
    gate = combined_ml < baseline_ml
    log.info("  Gate A: Morgan+3D (%.3f) < Morgan (%.3f)? → %s",
             combined_ml, baseline_ml, "PASS" if gate else "FAIL")

    results_a["gate_pass"] = gate
    results_a["n_3d_features"] = n_3d
    results_a["conformer_success_rate"] = (train["N"] - n_fail) / train["N"]
    return results_a


# ═══════════════════════════════════════════════════════════════
# Experiment B: CYP Enzyme Stratification
# ═══════════════════════════════════════════════════════════════

def experiment_b(train, holdout):
    log.info("=" * 70)
    log.info("Experiment B: CYP Enzyme Stratification")
    log.info("=" * 70)

    # B1: Load CYP annotations
    from sisyphus.predict.drugbank import DrugBankLookup
    _db = DrugBankLookup()
    def get_substrate_enzymes(smi):
        return _db.get_substrate_enzymes(smi)

    # Annotate training drugs
    cyp_labels = []
    for smi in train["smiles"]:
        enzymes = get_substrate_enzymes(smi)
        if not enzymes:
            cyp_labels.append("Unknown")
        else:
            # Primary = first one (arbitrary but consistent)
            cyp_labels.append(sorted(enzymes)[0])
    cyp_labels = np.array(cyp_labels)

    # Distribution
    from collections import Counter
    dist = Counter(cyp_labels)
    log.info("  B1: CYP distribution (training):")
    for cyp, n in sorted(dist.items(), key=lambda x: -x[1]):
        log.info("    %s: %d (%.1f%%)", cyp, n, n/train["N"]*100)

    annotated = sum(1 for c in cyp_labels if c != "Unknown")
    n_3a4 = dist.get("CYP3A4", 0)
    log.info("  Annotated: %d/%d (%.1f%%)", annotated, train["N"], annotated/train["N"]*100)
    log.info("  Gate B1: CYP3A4 >= 200? %s (%d). Annotated >= 600? %s (%d)",
             "YES" if n_3a4 >= 200 else "NO", n_3a4,
             "YES" if annotated >= 600 else "NO", annotated)

    if n_3a4 < 200 or annotated < 600:
        log.info("  Gate B1 FAIL — insufficient CYP annotations")
        return {"gate_pass": False, "reason": "insufficient CYP annotations",
                "n_annotated": annotated, "n_3a4": n_3a4}

    # B2: Load CLint training data
    clint_path = ROOT / "data/training" / "clearance_hepatocyte_az.tab"
    if not clint_path.exists():
        # Try TDC default location
        from sisyphus.predict.adme import _load_tdc_dataset
        log.info("  Loading CLint training data...")

    # Use the MMPK drugs instead — we have engine predictions, can compare
    # Actually, we need CLint data to train stratified models.
    # Let me check what CLint training data is available.
    tdc_path = ROOT / "data" / "clearance_hepatocyte_az.tab"
    if tdc_path.exists():
        clint_df = pd.read_csv(tdc_path, sep="\t")
        log.info("  CLint data: %d compounds", len(clint_df))
    else:
        log.info("  CLint data not found at %s", tdc_path)
        # Try alternative
        for p in [ROOT / "data/training/clearance_hepatocyte_az.tab",
                  ROOT / "data/clearance_hepatocyte_az.tab"]:
            if p.exists():
                clint_df = pd.read_csv(p, sep="\t")
                log.info("  CLint data found: %s (%d compounds)", p, len(clint_df))
                tdc_path = p
                break
        else:
            log.info("  Gate B2 FAIL — CLint training data not found")
            return {"gate_pass": False, "reason": "CLint data not found"}

    # CLint data columns
    cols = list(clint_df.columns)
    log.info("  CLint columns: %s", cols)

    # Map: Drug_ID or SMILES column
    smi_col = None
    y_col = None
    for c in cols:
        if "smiles" in c.lower() or "drug" in c.lower():
            smi_col = c
        if c == "Y" or "clearance" in c.lower():
            y_col = c
    if smi_col is None or y_col is None:
        # TDC format: Drug_ID, Drug, Y
        if "Drug" in cols and "Y" in cols:
            smi_col, y_col = "Drug", "Y"
        else:
            log.info("  Cannot identify CLint data columns — Experiment B ABORT")
            return {"gate_pass": False, "reason": "CLint column identification failed"}

    clint_smiles = clint_df[smi_col].tolist()
    clint_y = np.log10(clint_df[y_col].values.clip(0.01))

    # Annotate CLint drugs with CYP
    clint_cyp = []
    for smi in clint_smiles:
        enzymes = get_substrate_enzymes(smi)
        if not enzymes:
            clint_cyp.append("Unknown")
        else:
            clint_cyp.append(sorted(enzymes)[0])
    clint_cyp = np.array(clint_cyp)

    clint_dist = Counter(clint_cyp)
    log.info("  CLint CYP distribution:")
    for cyp, n in sorted(clint_dist.items(), key=lambda x: -x[1]):
        log.info("    %s: %d", cyp, n)

    # Merge small classes into "Other"
    for cyp in list(clint_dist.keys()):
        if clint_dist[cyp] < 50:
            clint_cyp[clint_cyp == cyp] = "Other"
    clint_dist_merged = Counter(clint_cyp)
    log.info("  After merging (<50): %s", dict(clint_dist_merged))

    # Features
    X_clint = np.array([compute_features(s) for s in clint_smiles], dtype=np.float32)

    # B2: Stratified vs Global CLint models
    xp_clint = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
                     colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.5,
                     reg_lambda=3.0, random_state=42, n_jobs=4, verbosity=0)

    folds_clint = scaffold_split(clint_smiles)

    # Global model CV
    oof_global = np.full(len(clint_smiles), np.nan)
    for fi, vi in enumerate(folds_clint):
        ti = [j for fj, idxs in enumerate(folds_clint) if fj != fi for j in idxs]
        m = xgb.XGBRegressor(**xp_clint)
        m.fit(X_clint[ti], clint_y[ti])
        oof_global[vi] = m.predict(X_clint[vi])
    r2_global = 1 - np.sum((clint_y - oof_global)**2) / np.sum((clint_y - clint_y.mean())**2)
    log.info("  Global CLint CV R²: %.3f", r2_global)

    # Stratified model CV
    oof_strat = np.full(len(clint_smiles), np.nan)
    unique_cyps = sorted(set(clint_cyp))
    for cyp in unique_cyps:
        mask = clint_cyp == cyp
        idx = np.where(mask)[0]
        if len(idx) < 30:
            # Too few: use global predictions
            oof_strat[idx] = oof_global[idx]
            continue

        # Per-CYP folds
        cyp_smiles = [clint_smiles[i] for i in idx]
        cyp_folds = scaffold_split(cyp_smiles)
        for fi, vi in enumerate(cyp_folds):
            ti = [j for fj, idxs in enumerate(cyp_folds) if fj != fi for j in idxs]
            global_ti = idx[ti]
            global_vi = idx[vi]
            if len(global_ti) < 20:
                oof_strat[global_vi] = oof_global[global_vi]
                continue
            from sklearn.linear_model import Ridge
            if len(global_ti) < 100:
                m = Ridge(alpha=1.0)
            else:
                m = xgb.XGBRegressor(**xp_clint)
            m.fit(X_clint[global_ti], clint_y[global_ti])
            oof_strat[global_vi] = m.predict(X_clint[global_vi])

    valid = ~np.isnan(oof_strat)
    r2_strat = 1 - np.sum((clint_y[valid] - oof_strat[valid])**2) / \
               np.sum((clint_y[valid] - clint_y[valid].mean())**2)
    log.info("  Stratified CLint CV R²: %.3f (global: %.3f)", r2_strat, r2_global)

    gate_b2 = r2_strat > 0.25
    log.info("  Gate B2: Stratified R² > 0.25? → %s", "PASS" if gate_b2 else "FAIL")

    return {
        "gate_pass": gate_b2,
        "r2_global": r2_global, "r2_stratified": r2_strat,
        "cyp_distribution": dict(clint_dist_merged),
        "n_clint_drugs": len(clint_smiles),
    }


# ═══════════════════════════════════════════════════════════════
# Experiment C: Multi-Distribution Kp Averaging
# ═══════════════════════════════════════════════════════════════

def experiment_c(train, holdout):
    log.info("=" * 70)
    log.info("Experiment C: Rodgers-Rowland + Poulin-Theil Kp Averaging")
    log.info("=" * 70)

    from sisyphus.pipeline.predict import predict

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    # We need to run engine with BOTH Kp methods for holdout drugs.
    # The pipeline defaults to RR. To get PT, we need to modify the call.
    # Check if we can pass kp_method through predict()

    # Quick check: can we get PT predictions?
    from sisyphus.predict.ivive import _compute_all_kp, _KP_FUNCTIONS
    log.info("  Available Kp methods: %s", list(_KP_FUNCTIONS.keys()))

    # Strategy: For each holdout drug, run predict() twice:
    # 1. With default (RR) — already have these in 3track JSON
    # 2. With PT — need to hack the kp_method

    # The kp_method is set in build_drug_on_graph() in ivive.py.
    # We can monkey-patch it temporarily.
    from sisyphus.predict import ivive as ivive_module

    ho_cmax_rr = holdout["engine"]  # Already computed
    ho_cmax_pt = np.zeros(holdout["N"])

    # Monkey-patch to force Poulin-Theil
    original_build = ivive_module.build_drug_on_graph

    def build_pt_patched(*args, **kwargs):
        kwargs["kp_method"] = "poulin_theil"
        return original_build(*args, **kwargs)

    # Check if the original function accepts kp_method
    import inspect
    sig = inspect.signature(original_build)
    if "kp_method" not in sig.parameters:
        log.info("  build_drug_on_graph doesn't accept kp_method — checking alternatives")
        # Try to find where kp_method is set
        log.info("  Cannot switch Kp method programmatically — Experiment C ABORT")
        logging.getLogger("sisyphus").setLevel(logging.INFO)
        return {"gate_pass": False, "reason": "kp_method not switchable"}

    log.info("  Running engine with Poulin-Theil Kp for %d holdout drugs...", holdout["N"])
    ivive_module.build_drug_on_graph = build_pt_patched

    t0 = time.time()
    n_fail = 0
    for i, h in enumerate(holdout["preds"]):
        smi = holdout["smiles"].get(h["name"])
        if not smi:
            ho_cmax_pt[i] = ho_cmax_rr[i]
            n_fail += 1
            continue
        try:
            result = predict(smi, holdout["doses"][i], "oral")
            ho_cmax_pt[i] = result.engine_pk.cmax.mean if result.engine_pk else ho_cmax_rr[i]
        except Exception:
            ho_cmax_pt[i] = ho_cmax_rr[i]
            n_fail += 1

    # Restore original
    ivive_module.build_drug_on_graph = original_build
    logging.getLogger("sisyphus").setLevel(logging.INFO)

    log.info("  PT engine complete (%.1fs, %d failures)", time.time()-t0, n_fail)

    # Compare
    aafe_rr = aafe(ho_cmax_rr, holdout["obs"])
    aafe_pt = aafe(ho_cmax_pt, holdout["obs"])

    # Geometric mean
    ho_cmax_avg = np.sqrt(ho_cmax_rr * ho_cmax_pt)  # geometric mean of 2
    aafe_avg = aafe(ho_cmax_avg, holdout["obs"])

    # Error correlation
    r_rr_pt = fe_corr(ho_cmax_rr, ho_cmax_pt, holdout["obs"])

    log.info("  Results:")
    log.info("    RR engine AAFE:  %.3f", aafe_rr)
    log.info("    PT engine AAFE:  %.3f", aafe_pt)
    log.info("    Avg engine AAFE: %.3f", aafe_avg)
    log.info("    r(RR_error, PT_error): %.3f", r_rr_pt)

    # Blend with ML
    baseline = aafe(holdout["combined"], holdout["obs"])
    for label, eng_pred in [("RR (current)", ho_cmax_rr), ("PT", ho_cmax_pt), ("RR+PT avg", ho_cmax_avg)]:
        blend = np.zeros(holdout["N"])
        for i in range(holdout["N"]):
            w = 0.45 if holdout["types"][i] == "base" else 0.00
            blend[i] = geo(eng_pred[i], holdout["ml"][i], w)
        a = aafe(blend, holdout["obs"])
        log.info("    Meta (%s): %.3f (Δ=%+.3f)", label, a, a - baseline)

    gate = aafe_avg < aafe_rr
    log.info("  Gate C: Avg (%.3f) < RR (%.3f)? → %s", aafe_avg, aafe_rr, "PASS" if gate else "FAIL")

    return {
        "aafe_rr": aafe_rr, "aafe_pt": aafe_pt, "aafe_avg": aafe_avg,
        "r_rr_pt": r_rr_pt, "gate_pass": gate,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    train, holdout = load_data()

    res_a = experiment_a(train, holdout)
    res_b = experiment_b(train, holdout)
    res_c = experiment_c(train, holdout)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY (%.0fs)", time.time()-t0)
    log.info("=" * 70)
    log.info("  Exp A (3D Features): gate=%s", "PASS" if res_a.get("gate_pass") else "FAIL")
    log.info("  Exp B (CYP Stratif): gate=%s", "PASS" if res_b.get("gate_pass") else "FAIL")
    log.info("  Exp C (Kp Average):  gate=%s", "PASS" if res_c.get("gate_pass") else "FAIL")

    # Save
    out = {"experiment_a": {k: v for k, v in res_a.items() if not isinstance(v, np.ndarray)},
           "experiment_b": {k: v for k, v in res_b.items() if not isinstance(v, np.ndarray)},
           "experiment_c": {k: v for k, v in res_c.items() if not isinstance(v, np.ndarray)}}
    with open(ROOT / "data/validation/3d_cyp_multidist_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("  Saved to data/validation/3d_cyp_multidist_results.json")


if __name__ == "__main__":
    main()
