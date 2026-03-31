#!/usr/bin/env python3
"""CLint Descriptor Upgrade — Sequential experiments with error cancellation gate.

Exp 1: Base descriptors 9→22
Exp 2: + VSA descriptors (36)
Exp 3: Morgan FP radius 2→3
Exp 4: Feature selection (top 300/500/800)
Exp 5: Optuna hyperparameter search

Each step: 5-fold scaffold CV R², then holdout AAFE gate if R² improves.
"""
from __future__ import annotations

import json, logging, sys, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MolSurf
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

# ─── Paths ───
TDC_HEP = ROOT / "data/training/clearance_hepatocyte_az.tab"
HOLDOUT_JSON = ROOT / "data/reference/holdout.json"
CLINICAL_PK = ROOT / "data/reference/clinical_pk.json"
MODEL_DIR = ROOT / "models/adme"

# ─── Holdout exclusion ───
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
    with open(HOLDOUT_JSON) as f: hd = json.load(f)
    with open(CLINICAL_PK) as f: cp = json.load(f)
    names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    ik_set = set()
    for n in names:
        e = drugs.get(n) or drugs.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ik = _ik14(e["smiles"])
            if ik: ik_set.add(ik)
    return ik_set

# ─── Descriptor computation ───
def compute_morgan(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    for b in fp.GetOnBits(): arr[b] = 1.0
    return arr

def compute_base_descriptors(mol):
    """Original 9 descriptors (normalized)."""
    return [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol) / 200.0,
        Descriptors.MolWt(mol) / 600.0,
        Descriptors.NumHAcceptors(mol) / 10.0,
        Descriptors.NumHDonors(mol) / 5.0,
        Descriptors.NumRotatableBonds(mol) / 15.0,
        Descriptors.RingCount(mol) / 5.0,
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol) / 150.0,
    ]

def compute_extended_descriptors(mol):
    """Base 9 + 13 new = 22 descriptors."""
    base = compute_base_descriptors(mol)
    try:
        ext = [
            Descriptors.BertzCT(mol) / 1000.0,
            Descriptors.NumAromaticRings(mol) / 5.0,
            Descriptors.NumAliphaticRings(mol) / 5.0,
            Descriptors.NumAromaticHeterocycles(mol) / 3.0,
            Descriptors.NumSaturatedHeterocycles(mol) / 3.0,
            Descriptors.NumHeteroatoms(mol) / 10.0,
            Descriptors.NumAmideBonds(mol) / 5.0,
            Descriptors.LabuteASA(mol) / 200.0,
            Descriptors.MaxPartialCharge(mol) or 0.0,
            Descriptors.MinPartialCharge(mol) or 0.0,
            Descriptors.MaxAbsPartialCharge(mol) or 0.0,
            Descriptors.MinAbsPartialCharge(mol) or 0.0,
            Descriptors.NHOHCount(mol) / 5.0,
        ]
    except Exception:
        ext = [0.0] * 13
    return base + ext

def compute_vsa_descriptors(mol):
    """36 VSA descriptors."""
    vsa = []
    for i in range(1, 13):
        try: vsa.append(getattr(Descriptors, f'SlogP_VSA{i}')(mol) / 100.0)
        except: vsa.append(0.0)
    for i in range(1, 11):
        try: vsa.append(getattr(Descriptors, f'SMR_VSA{i}')(mol) / 100.0)
        except: vsa.append(0.0)
    for i in range(1, 15):
        try: vsa.append(getattr(Descriptors, f'PEOE_VSA{i}')(mol) / 100.0)
        except: vsa.append(0.0)
    return vsa

# ─── Scaffold split ───
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

# ─── CV evaluation ───
def scaffold_cv_r2(X, y, smiles, xgb_params=None):
    if xgb_params is None:
        xgb_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=4, verbosity=0)
    folds = scaffold_split(smiles)
    preds = np.zeros(len(y))
    for fi in range(5):
        te = folds[fi]; tr = [j for f in range(5) if f != fi for j in folds[f]]
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X[tr], y[tr], verbose=False)
        preds[te] = m.predict(X[te])
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

# ─── Load TDC data ───
def load_data():
    ho_ik = load_holdout_ik()
    df = pd.read_csv(TDC_HEP, sep="\t")
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canon)
    df = df.dropna(subset=["canon"])
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()
    df["ik"] = df["canon"].apply(_ik14)
    df = df.dropna(subset=["ik"])
    df = df[~df["ik"].isin(ho_ik)].reset_index(drop=True)
    log.info("Training data: %d compounds", len(df))
    return df

# ─── Build feature matrices for each experiment ───
def build_features(df, desc_func, radius=2, nbits=2048):
    X_list, y_list, smi_list = [], [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canon"])
        if not mol: continue
        fp = compute_morgan(row["canon"], radius=radius, nbits=nbits)
        if fp is None: continue
        desc = desc_func(mol)
        desc = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in desc]
        feat = np.concatenate([fp, np.array(desc, dtype=np.float32)])
        X_list.append(feat)
        y_list.append(np.log10(max(float(row["Y"]), 0.1)))
        smi_list.append(row["canon"])
    return np.array(X_list), np.array(y_list), smi_list

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    df = load_data()
    results = {}
    best_r2 = 0.0
    best_config = "baseline"
    best_X, best_y, best_smi = None, None, None
    best_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       n_jobs=4, verbosity=0)

    # ─── Baseline ───
    log.info("Baseline: Morgan FP (r=2, 2048) + 9 descriptors")
    X0, y0, smi0 = build_features(df, compute_base_descriptors, radius=2)
    r2_base = scaffold_cv_r2(X0, y0, smi0)
    results["baseline"] = {"r2": r2_base, "n_feat": X0.shape[1], "n": len(y0)}
    log.info("  R²=%.3f, N=%d, features=%d", r2_base, len(y0), X0.shape[1])
    best_r2 = r2_base
    best_X, best_y, best_smi = X0, y0, smi0

    # ─── Exp 1: Extended descriptors (9→22) ───
    log.info("\nExp 1: Extended descriptors (9→22)")
    X1, y1, smi1 = build_features(df, compute_extended_descriptors, radius=2)
    r2_1 = scaffold_cv_r2(X1, y1, smi1)
    delta_1 = r2_1 - r2_base
    results["exp1_extended"] = {"r2": r2_1, "delta": delta_1, "n_feat": X1.shape[1]}
    log.info("  R²=%.3f (Δ=%+.3f), features=%d", r2_1, delta_1, X1.shape[1])
    if r2_1 > best_r2:
        best_r2 = r2_1; best_config = "exp1"; best_X, best_y, best_smi = X1, y1, smi1

    # ─── Exp 2: + VSA descriptors (22→58) ───
    log.info("\nExp 2: + VSA descriptors (22→58)")
    def desc_full(mol):
        return compute_extended_descriptors(mol) + compute_vsa_descriptors(mol)
    X2, y2, smi2 = build_features(df, desc_full, radius=2)
    r2_2 = scaffold_cv_r2(X2, y2, smi2)
    delta_2 = r2_2 - r2_base
    results["exp2_vsa"] = {"r2": r2_2, "delta": delta_2, "n_feat": X2.shape[1]}
    log.info("  R²=%.3f (Δ=%+.3f), features=%d", r2_2, delta_2, X2.shape[1])
    if r2_2 > best_r2:
        best_r2 = r2_2; best_config = "exp2"; best_X, best_y, best_smi = X2, y2, smi2

    # ─── Exp 3: Radius 3 (ECFP6) ───
    log.info("\nExp 3: Morgan radius=3 (ECFP6) + best descriptors")
    # Use whichever descriptor set was best so far
    if best_config == "exp2":
        desc_fn = desc_full
    elif best_config == "exp1":
        desc_fn = compute_extended_descriptors
    else:
        desc_fn = compute_base_descriptors
    X3, y3, smi3 = build_features(df, desc_fn, radius=3, nbits=2048)
    r2_3 = scaffold_cv_r2(X3, y3, smi3)
    delta_3 = r2_3 - r2_base
    results["exp3_radius3"] = {"r2": r2_3, "delta": delta_3, "n_feat": X3.shape[1]}
    log.info("  R²=%.3f (Δ=%+.3f), features=%d", r2_3, delta_3, X3.shape[1])
    if r2_3 > best_r2:
        best_r2 = r2_3; best_config = "exp3"; best_X, best_y, best_smi = X3, y3, smi3

    # Also try radius=3 with 4096 bits
    X3b, y3b, smi3b = build_features(df, desc_fn, radius=3, nbits=4096)
    r2_3b = scaffold_cv_r2(X3b, y3b, smi3b)
    results["exp3b_radius3_4096"] = {"r2": r2_3b, "delta": r2_3b - r2_base, "n_feat": X3b.shape[1]}
    log.info("  r=3 + 4096 bits: R²=%.3f (Δ=%+.3f), features=%d", r2_3b, r2_3b - r2_base, X3b.shape[1])
    if r2_3b > best_r2:
        best_r2 = r2_3b; best_config = "exp3b"; best_X, best_y, best_smi = X3b, y3b, smi3b

    # ─── Exp 4: Feature selection ───
    log.info("\nExp 4: Feature selection on best config (%s)", best_config)
    from sklearn.feature_selection import SelectFromModel
    X_for_sel = best_X.copy()  # preserve original for all iterations
    n_feat_total = X_for_sel.shape[1]
    best_sel_X = None
    for max_f in [300, 500, 800]:
        if max_f > n_feat_total:
            continue
        m_sel = xgb.XGBRegressor(**best_params)
        m_sel.fit(X_for_sel, best_y, verbose=False)
        sel = SelectFromModel(m_sel, max_features=max_f, threshold=-np.inf)
        sel.fit(X_for_sel, best_y)
        X_sel = sel.transform(X_for_sel)
        r2_sel = scaffold_cv_r2(X_sel, best_y, best_smi)
        results[f"exp4_select_{max_f}"] = {"r2": r2_sel, "delta": r2_sel - r2_base, "n_feat": X_sel.shape[1]}
        log.info("  top %d: R²=%.3f (Δ=%+.3f)", max_f, r2_sel, r2_sel - r2_base)
        if r2_sel > best_r2:
            best_r2 = r2_sel; best_sel_X = X_sel
    if best_sel_X is not None:
        best_X = best_sel_X

    # ─── Exp 5: Optuna hyperparameter search ───
    log.info("\nExp 5: Optuna hyperparameter search")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42, "n_jobs": 4, "verbosity": 0,
            }
            return scaffold_cv_r2(best_X, best_y, best_smi, params)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        r2_optuna = study.best_value
        results["exp5_optuna"] = {
            "r2": r2_optuna, "delta": r2_optuna - r2_base,
            "best_params": study.best_params,
        }
        log.info("  Best R²=%.3f (Δ=%+.3f)", r2_optuna, r2_optuna - r2_base)
        log.info("  Params: %s", study.best_params)
        if r2_optuna > best_r2:
            best_r2 = r2_optuna
            best_params = {**study.best_params, "random_state": 42, "n_jobs": 4, "verbosity": 0}
    except ImportError:
        log.warning("Optuna not installed, skipping Exp 5")
        # Manual grid search fallback
        log.info("  Falling back to manual grid search")
        best_grid_r2 = best_r2
        for md in [4, 5, 6, 7, 8]:
            for ne in [300, 500, 700, 1000]:
                for lr in [0.01, 0.05, 0.1]:
                    p = dict(max_depth=md, n_estimators=ne, learning_rate=lr,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=4, verbosity=0)
                    r2 = scaffold_cv_r2(best_X, best_y, best_smi, p)
                    if r2 > best_grid_r2:
                        best_grid_r2 = r2; best_params = p
        results["exp5_grid"] = {"r2": best_grid_r2, "delta": best_grid_r2 - r2_base}
        log.info("  Grid best R²=%.3f (Δ=%+.3f)", best_grid_r2, best_grid_r2 - r2_base)
        best_r2 = max(best_r2, best_grid_r2)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  CLint DESCRIPTOR UPGRADE — RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Experiment':<30s} {'R²':>8s} {'Δ':>8s} {'Features':>9s}")
    print(f"  {'-'*60}")
    for k, v in results.items():
        d = v.get("delta", 0)
        d_str = f"{d:+.3f}" if d != 0 else "—"
        nf = v.get("n_feat", "?")
        print(f"  {k:<30s} {v['r2']:>8.3f} {d_str:>8s} {nf:>9}")
    print()
    print(f"  Best CLint R²: {best_r2:.3f} (baseline: {r2_base:.3f}, Δ={best_r2 - r2_base:+.3f})")

    # ═══════════════════════════════════════════════════════════════════════
    # Error cancellation gate: holdout AAFE if R² improved > 0.01
    # ═══════════════════════════════════════════════════════════════════════
    if best_r2 - r2_base > 0.01:
        print(f"\n  R² improved by {best_r2 - r2_base:+.3f} > 0.01 — running holdout AAFE gate")

        # Train final model with best features/params on all training data
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(best_X, best_y, verbose=False)

        # Save temporarily and run benchmark
        tmp_model = MODEL_DIR / "xgboost_clint_desc_upgrade.json"
        backup_model = MODEL_DIR / "xgboost_clint.json"
        backup_copy = MODEL_DIR / "xgboost_clint_pre_desc.json.bak"

        if not backup_copy.exists():
            shutil.copy2(backup_model, backup_copy)

        # NOTE: We can't simply swap the model because the feature computation
        # in adme.py uses the OLD descriptor function. To properly test, we'd
        # need to also modify descriptors.py. For now, report CLint R² only
        # and note that AAFE gate requires code change.
        print(f"\n  ⚠ AAFE gate requires modifying descriptors.py to match new features.")
        print(f"  CLint R² improvement: {r2_base:.3f} → {best_r2:.3f} (Δ={best_r2 - r2_base:+.3f})")
        print(f"  Based on 16 prior experiments, CLint R² improvement does NOT")
        print(f"  guarantee AAFE improvement (error cancellation).")
        print(f"  Previous: R² 0.279→0.333 (+0.054) → meta AAFE +0.038 (worse)")
    else:
        print(f"\n  R² improvement {best_r2 - r2_base:+.3f} ≤ 0.01 — no AAFE gate needed")

    # Save results
    out = ROOT / "data/validation/clint_descriptor_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
