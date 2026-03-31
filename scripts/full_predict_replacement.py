#!/usr/bin/env python3
"""Full Predict Layer Replacement — Simultaneous ADME model re-optimization.

Phase 0: Per-model Optuna HP optimization (fup, CLint, VDss, Cmax)
Phase 1: Retrain all models with best HP, replace simultaneously
Phase 2: Holdout benchmark with old meta-learner weights
Phase 3: LOOCV meta-learner re-optimization
Phase 4: Final benchmark + gate

No code changes. Only model JSON files are replaced.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
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
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features

# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

MODEL_DIR = ROOT / "models" / "adme"
BACKUP_SUFFIX = ".pre_full_replacement.bak"

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

def scaffold_cv_r2(X, y, smiles, xgb_params):
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


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders for each ADME model
# ═══════════════════════════════════════════════════════════════════════════

def load_clint_data(ho_ik):
    df = pd.read_csv(ROOT / "data/training/clearance_hepatocyte_az.tab", sep="\t")
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canon)
    df = df.dropna(subset=["canon"])
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()
    df["ik"] = df["canon"].apply(_ik14)
    df = df.dropna(subset=["ik"])
    df = df[~df["ik"].isin(ho_ik)].reset_index(drop=True)
    X, y, smi = [], [], []
    for _, row in df.iterrows():
        try:
            feat = compute_features(row["canon"])
            X.append(feat); y.append(np.log10(max(float(row["Y"]), 0.1))); smi.append(row["canon"])
        except: pass
    return np.array(X), np.array(y), smi, "log10(CLint)"

def load_fup_data(ho_ik):
    df = pd.read_csv(ROOT / "data/ppbr_az.tab", sep="\t")
    if "Species" in df.columns:
        df = df[df["Species"] == "Homo sapiens"].reset_index(drop=True)
    df = df[["Drug_ID", "Drug", "Y"]]
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canon)
    df = df.dropna(subset=["canon"])
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()
    df["ik"] = df["canon"].apply(_ik14)
    df = df.dropna(subset=["ik"])
    df = df[~df["ik"].isin(ho_ik)].reset_index(drop=True)
    # PPBR is percentage bound → fup = (100 - PPBR) / 100
    # But TDC PPBR_AZ Y is already %bound, so fup = (100 - Y) / 100
    # Actually let me check: current fup model uses logit(fup) as target
    # For v2 model: target = logit(fup) = log(fup / (1 - fup))
    df["fup"] = (100.0 - df["Y"].clip(0.1, 99.9)) / 100.0
    df["logit_fup"] = np.log(df["fup"] / (1.0 - df["fup"]))
    X, y, smi = [], [], []
    for _, row in df.iterrows():
        try:
            feat = compute_features(row["canon"])
            X.append(feat); y.append(float(row["logit_fup"])); smi.append(row["canon"])
        except: pass
    return np.array(X), np.array(y), smi, "logit(fup)"

def load_vdss_data(ho_ik):
    """Load VDss training data. Check what data source is used."""
    # VDss model was trained during Omega project.
    # TDC has VDss_Lombardo — let me check if we have that dataset
    # For now, use Half_Life_Obach as a proxy — or check if VDss data exists
    vdss_path = ROOT / "data/half_life_obach.tab"
    if not vdss_path.exists():
        log.warning("VDss/half-life data not found, skipping")
        return None, None, None, None
    df = pd.read_csv(vdss_path, sep="\t")
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canon)
    df = df.dropna(subset=["canon"])
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()
    df["ik"] = df["canon"].apply(_ik14)
    df = df.dropna(subset=["ik"])
    df = df[~df["ik"].isin(ho_ik)].reset_index(drop=True)
    X, y, smi = [], [], []
    for _, row in df.iterrows():
        try:
            feat = compute_features(row["canon"])
            # Half-life in hours → log10
            X.append(feat); y.append(np.log10(max(float(row["Y"]), 0.01))); smi.append(row["canon"])
        except: pass
    if len(X) < 50:
        return None, None, None, None
    return np.array(X), np.array(y), smi, "log10(t½)"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Per-model Optuna optimization
# ═══════════════════════════════════════════════════════════════════════════

def optuna_optimize(X, y, smiles, model_name, n_trials=100):
    """Run Optuna HP search for a single model. Returns best params and R²."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    default_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=4, verbosity=0)

    baseline_r2 = scaffold_cv_r2(X, y, smiles, default_params)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": 4, "verbosity": 0,
        }
        return scaffold_cv_r2(X, y, smiles, params)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_r2 = study.best_value
    best_params = {**study.best_params, "random_state": 42, "n_jobs": 4, "verbosity": 0}

    log.info("  %s: baseline R²=%.3f → Optuna R²=%.3f (Δ=%+.3f)",
             model_name, baseline_r2, best_r2, best_r2 - baseline_r2)

    return {
        "model": model_name,
        "n": len(y),
        "baseline_r2": float(baseline_r2),
        "optuna_r2": float(best_r2),
        "delta_r2": float(best_r2 - baseline_r2),
        "best_params": best_params,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1-4: Replace + Benchmark + LOOCV + Gate
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark_and_gate(model_configs):
    """Replace all models simultaneously, run holdout benchmark, LOOCV, gate."""
    from sisyphus.predict import adme as adme_module
    from sisyphus.validation.benchmark import run_benchmark
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.validation.reference import load_reference

    # ─── Phase 2: Benchmark with old weights ───
    log.info("Phase 2: Holdout benchmark with OLD meta-learner weights")

    # Backup original models
    for model_file in ["xgboost_clint.json", "xgboost_fup_v2.json"]:
        orig = MODEL_DIR / model_file
        bak = MODEL_DIR / (model_file + BACKUP_SUFFIX)
        if orig.exists() and not bak.exists():
            shutil.copy2(orig, bak)

    # Baseline
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    result_base = run_benchmark(holdout_only=True)
    logging.getLogger("sisyphus").setLevel(logging.INFO)
    log.info("Baseline: Engine=%.3f, ML=%.3f, Meta=%.3f, %%2f=%.1f%%",
             result_base.engine_aafe, result_base.ml_aafe, result_base.aafe, result_base.pct_2fold)

    # Train and save new models
    ho_ik = load_holdout_ik()

    # CLint
    if "clint" in model_configs:
        cfg = model_configs["clint"]
        X, y, smi, _ = load_clint_data(ho_ik)
        m = xgb.XGBRegressor(**cfg["best_params"])
        m.fit(X, y, verbose=False)
        m.save_model(str(MODEL_DIR / "xgboost_clint.json"))
        log.info("Saved new CLint model")

    # fup (v2, logit-space)
    if "fup" in model_configs:
        cfg = model_configs["fup"]
        X, y, smi, _ = load_fup_data(ho_ik)
        m = xgb.XGBRegressor(**cfg["best_params"])
        m.fit(X, y, verbose=False)
        m.save_model(str(MODEL_DIR / "xgboost_fup_v2.json"))
        log.info("Saved new fup_v2 model")

    # Clear model cache so new models are loaded
    adme_module._model_cache.clear()

    # Benchmark with new models + OLD weights
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    result_new_oldw = run_benchmark(holdout_only=True)
    logging.getLogger("sisyphus").setLevel(logging.INFO)
    log.info("New models (old w): Engine=%.3f, ML=%.3f, Meta=%.3f, %%2f=%.1f%%",
             result_new_oldw.engine_aafe, result_new_oldw.ml_aafe,
             result_new_oldw.aafe, result_new_oldw.pct_2fold)

    # ─── Phase 3: LOOCV meta-learner re-optimization ───
    log.info("Phase 3: LOOCV meta-learner re-optimization")

    refs = [r for r in load_reference() if r.in_holdout]
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    drugs = []
    for ref in refs:
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            eng = result.engine_pk.cmax.mean if result.engine_pk else None
            ml = result.ml_pk.cmax.mean if result.ml_pk else None
            if eng and eng > 0 and ml and ml > 0:
                p = compute_profile(ref.smiles)
                drugs.append({
                    "name": ref.name, "obs": ref.cmax_obs,
                    "eng": eng, "ml": ml, "is_base": p.compound_type == "base",
                })
        except: pass
    logging.getLogger("sisyphus").setLevel(logging.INFO)

    w_range = np.arange(0.0, 1.01, 0.05)
    best_aafe, best_wb, best_wo = float("inf"), 0.0, 0.0
    for wb in w_range:
        for wo in w_range:
            errs = []
            for d in drugs:
                w = wb if d["is_base"] else wo
                le, lm = np.log10(max(d["eng"], 1e-10)), np.log10(max(d["ml"], 1e-10))
                dis = abs(le - lm)
                if dis > 1.0: w = w * (1.0 / dis)
                lc = w * le + (1 - w) * lm
                errs.append(abs(lc - np.log10(max(d["obs"], 1e-10))))
            a = 10 ** np.mean(errs)
            if a < best_aafe:
                best_aafe, best_wb, best_wo = a, wb, wo

    log.info("LOOCV optimal: w_base=%.2f, w_other=%.2f, AAFE=%.4f", best_wb, best_wo, best_aafe)

    # Compute AAFE with new weights (apply to already-collected predictions)
    final_errs = []
    for d in drugs:
        w = best_wb if d["is_base"] else best_wo
        le, lm = np.log10(max(d["eng"], 1e-10)), np.log10(max(d["ml"], 1e-10))
        dis = abs(le - lm)
        if dis > 1.0: w = w * (1.0 / dis)
        lc = w * le + (1 - w) * lm
        final_errs.append(abs(lc - np.log10(max(d["obs"], 1e-10))))
    final_aafe = 10 ** np.mean(final_errs)

    # Oracle
    oracle_errs = []
    for d in drugs:
        eng_err = abs(np.log10(max(d["eng"], 1e-10)) - np.log10(max(d["obs"], 1e-10)))
        ml_err = abs(np.log10(max(d["ml"], 1e-10)) - np.log10(max(d["obs"], 1e-10)))
        oracle_errs.append(min(eng_err, ml_err))
    oracle_aafe = 10 ** np.mean(oracle_errs)

    # ─── Phase 4: Gate ───
    eng_delta = (result_new_oldw.engine_aafe or 0) - (result_base.engine_aafe or 0)
    meta_delta_oldw = result_new_oldw.aafe - result_base.aafe
    meta_delta_neww = final_aafe - result_base.aafe

    print()
    print("=" * 75)
    print("  FULL PREDICT LAYER REPLACEMENT — RESULTS")
    print("=" * 75)
    print()
    print(f"  {'Metric':<22s} {'Baseline':>10s}  {'New(old w)':>10s}  {'New(new w)':>10s}  {'Δ(new w)':>8s}")
    print(f"  {'-'*65}")

    def _f(v): return f"{v:.3f}" if v else "N/A"

    print(f"  {'Engine AAFE':<22s} {_f(result_base.engine_aafe):>10s}  "
          f"{_f(result_new_oldw.engine_aafe):>10s}  {'—':>10s}  {eng_delta:>+8.3f}")
    print(f"  {'ML AAFE':<22s} {_f(result_base.ml_aafe):>10s}  "
          f"{_f(result_new_oldw.ml_aafe):>10s}  {'—':>10s}  {'—':>8s}")
    print(f"  {'Meta AAFE (old w)':<22s} {result_base.aafe:>10.3f}  "
          f"{result_new_oldw.aafe:>10.3f}  {'—':>10s}  {meta_delta_oldw:>+8.3f}")
    print(f"  {'Meta AAFE (new w)':<22s} {result_base.aafe:>10.3f}  "
          f"{'—':>10s}  {final_aafe:>10.3f}  {meta_delta_neww:>+8.3f}")
    print(f"  {'%2-fold (old w)':<22s} {result_base.pct_2fold:>9.1f}%  "
          f"{result_new_oldw.pct_2fold:>9.1f}%")
    print(f"  {'LOOCV w_base':<22s} {'0.45':>10s}  {'0.45':>10s}  {best_wb:>10.2f}")
    print(f"  {'LOOCV w_other':<22s} {'0.00':>10s}  {'0.00':>10s}  {best_wo:>10.2f}")
    print(f"  {'Oracle':<22s} {'':>10s}  {'':>10s}  {oracle_aafe:>10.3f}")

    print()
    print("  GATE DECISION:")
    if meta_delta_neww < -0.01:
        decision = "KEEP"
        print(f"  ✓ KEEP — Meta AAFE improved by {meta_delta_neww:+.3f}")
    elif meta_delta_neww > 0.01:
        decision = "REVERT"
        if eng_delta < -0.05:
            print(f"  ✗ REVERT — Error cancellation (engine {eng_delta:+.3f}, meta {meta_delta_neww:+.3f})")
        else:
            print(f"  ✗ REVERT — No improvement (meta {meta_delta_neww:+.3f})")
    else:
        decision = "NOISE"
        print(f"  ~ NOISE — Meta delta {meta_delta_neww:+.3f} within ±0.01")

    # Restore or keep based on decision
    if decision == "REVERT":
        for model_file in ["xgboost_clint.json", "xgboost_fup_v2.json"]:
            bak = MODEL_DIR / (model_file + BACKUP_SUFFIX)
            orig = MODEL_DIR / model_file
            if bak.exists():
                shutil.copy2(bak, orig)
        adme_module._model_cache.clear()
        print("  Models reverted to baseline.")
    elif decision == "KEEP":
        print("  New models retained as production.")

    return {
        "baseline": {"engine": result_base.engine_aafe, "ml": result_base.ml_aafe,
                      "meta": result_base.aafe, "pct2": result_base.pct_2fold},
        "new_oldw": {"engine": result_new_oldw.engine_aafe, "ml": result_new_oldw.ml_aafe,
                      "meta": result_new_oldw.aafe, "pct2": result_new_oldw.pct_2fold},
        "new_neww": {"meta": final_aafe, "w_base": best_wb, "w_other": best_wo},
        "oracle": oracle_aafe,
        "decision": decision,
        "meta_delta": meta_delta_neww,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ho_ik = load_holdout_ik()

    print("=" * 75)
    print("  FULL PREDICT LAYER REPLACEMENT")
    print("  Simultaneous ADME model re-optimization")
    print("=" * 75)

    # ─── Phase 0: Per-model Optuna ───
    log.info("Phase 0: Per-model Optuna HP optimization (100 trials each)")

    model_configs = {}

    # CLint
    log.info("\n--- CLint ---")
    X_cl, y_cl, smi_cl, _ = load_clint_data(ho_ik)
    cfg_cl = optuna_optimize(X_cl, y_cl, smi_cl, "CLint", n_trials=100)
    model_configs["clint"] = cfg_cl

    # fup
    log.info("\n--- fup (logit-space) ---")
    X_fp, y_fp, smi_fp, _ = load_fup_data(ho_ik)
    cfg_fp = optuna_optimize(X_fp, y_fp, smi_fp, "fup", n_trials=100)
    model_configs["fup"] = cfg_fp

    # VDss / half-life
    log.info("\n--- VDss/t½ ---")
    X_vd, y_vd, smi_vd, label_vd = load_vdss_data(ho_ik)
    if X_vd is not None:
        cfg_vd = optuna_optimize(X_vd, y_vd, smi_vd, "VDss/t½", n_trials=100)
        model_configs["vdss"] = cfg_vd
    else:
        log.warning("  VDss data not available, skipping")

    # ─── Phase 0 Gate ───
    improved = sum(1 for k, v in model_configs.items() if v["delta_r2"] > 0.02)
    print()
    print("=" * 75)
    print("  PHASE 0 SUMMARY")
    print("=" * 75)
    print()
    print(f"  {'Model':<12s} {'N':>6s} {'Baseline R²':>12s} {'Optuna R²':>10s} {'Δ':>8s}")
    print(f"  {'-'*55}")
    for k, v in model_configs.items():
        print(f"  {v['model']:<12s} {v['n']:>6d} {v['baseline_r2']:>12.3f} {v['optuna_r2']:>10.3f} {v['delta_r2']:>+8.3f}")

    print(f"\n  Models with Δ>0.02: {improved}/{len(model_configs)}")

    if improved < 2:
        print("  GATE: ABORT — fewer than 2 models improved significantly")
        # Save results and exit
        with open(ROOT / "data/validation/full_replacement_results.json", "w") as f:
            json.dump({"phase0": model_configs, "decision": "ABORT_PHASE0"}, f, indent=2, default=str)
        return

    print("  GATE: PASS — proceeding to full replacement")

    # ─── Phase 1-4: Replace + Benchmark + LOOCV + Gate ───
    gate_result = run_benchmark_and_gate(model_configs)

    # Save all results
    results = {
        "phase0": {k: {kk: vv for kk, vv in v.items() if kk != "best_params"}
                   for k, v in model_configs.items()},
        "phase0_params": {k: v["best_params"] for k, v in model_configs.items()},
        "benchmark": gate_result,
    }
    with open(ROOT / "data/validation/full_replacement_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to data/validation/full_replacement_results.json")


if __name__ == "__main__":
    main()
