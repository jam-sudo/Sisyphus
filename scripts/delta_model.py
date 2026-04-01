#!/usr/bin/env python3
"""Delta Model: Residual Correction + ADME-Augmented Features.

Engine output as physics prior, ML corrects the residual.
log10(Cmax_final) = log10(Cmax_engine) + Delta(SMILES, ADME_features)

Phase 0: Compute engine predictions + delta targets for MMPK training drugs
Phase 1: 4-way comparison (baseline ML / ADME feat / delta / delta+ADME)
Phase 2: Holdout benchmark + gate
"""
from __future__ import annotations

import json, logging, sys
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
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features

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
    s = set()
    for n in names:
        e = drugs.get(n) or drugs.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ik = _ik14(e["smiles"])
            if ik: s.add(ik)
    return s

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

def compute_aafe(p, o):
    mask = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[mask] / o[mask])))) if mask.sum() > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Compute engine predictions + ADME features + delta targets
# ═══════════════════════════════════════════════════════════════════════════

def phase0():
    log.info("=" * 60)
    log.info("Phase 0: Data Preparation")
    log.info("=" * 60)

    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme

    ho_ik = load_holdout_ik()

    mmpk = pd.read_csv(ROOT / "data/training/mmpk_expanded_full.csv")
    mmpk = mmpk[~mmpk["in_holdout"]].reset_index(drop=True)
    dedup = mmpk.groupby("canon_smiles").agg(
        dose_mg=("dose_mg", "median"), cmax_obs=("cmax_mg_L", "median"),
        log_cpd=("log_cmax_per_dose", "median"),
    ).reset_index()
    dedup["ik"] = dedup["canon_smiles"].apply(_ik14)
    dedup = dedup.dropna(subset=["ik"])
    dedup = dedup[~dedup["ik"].isin(ho_ik)].reset_index(drop=True)
    dedup = dedup[dedup["cmax_obs"] > 0].reset_index(drop=True)

    log.info("MMPK training drugs: %d", len(dedup))

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    records = []
    n_fail = 0
    for i, (_, row) in enumerate(dedup.iterrows()):
        smi = row["canon_smiles"]
        dose = float(row["dose_mg"])
        cmax_obs = float(row["cmax_obs"])

        try:
            result = predict(smi, dose, "oral")
            cmax_eng = result.engine_pk.cmax.mean if result.engine_pk else None
            cmax_ml = result.ml_pk.cmax.mean if result.ml_pk else None

            if not cmax_eng or cmax_eng <= 0 or not cmax_ml or cmax_ml <= 0:
                n_fail += 1; continue

            # Delta target
            delta = np.log10(cmax_obs / cmax_eng)

            # Morgan FP
            fp = compute_features(smi)

            # ADME features
            profile = compute_profile(smi)
            adme = predict_adme(profile)
            mol = Chem.MolFromSmiles(smi)

            adme_feats = {
                "clint_log": np.log10(max(adme.clint.mean, 1e-6)),
                "fup": adme.fup.mean,
                "peff_log": np.log10(max(adme.peff.mean, 1e-6)),
                "sol_log": np.log10(max(adme.solubility.mean, 1e-6)),
                "vdss_log": np.log10(max(adme.vdss.mean, 1e-6)),
                "rbp": adme.rbp.mean,
                "mw": Descriptors.MolWt(mol) / 600.0 if mol else 0,
                "logp": Descriptors.MolLogP(mol) if mol else 0,
                "tpsa": Descriptors.TPSA(mol) / 200.0 if mol else 0,
                "hba": Descriptors.NumHAcceptors(mol) / 10.0 if mol else 0,
                "hbd": Descriptors.NumHDonors(mol) / 5.0 if mol else 0,
                "rotb": Descriptors.NumRotatableBonds(mol) / 15.0 if mol else 0,
                "is_base": 1.0 if profile.compound_type == "base" else 0.0,
                "is_acid": 1.0 if profile.compound_type == "acid" else 0.0,
            }

            records.append({
                "smi": smi, "dose": dose, "cmax_obs": cmax_obs,
                "cmax_eng": cmax_eng, "cmax_ml": cmax_ml,
                "delta": delta, "log_cpd": float(row["log_cpd"]),
                "fp": fp.astype(np.float32),
                "adme": adme_feats,
                "is_base": profile.compound_type == "base",
            })
        except Exception:
            n_fail += 1

        if (i + 1) % 200 == 0:
            log.info("  Progress: %d/%d (fail=%d)", i + 1, len(dedup), n_fail)

    logging.getLogger("sisyphus").setLevel(logging.INFO)
    log.info("Phase 0 complete: %d records, %d failures", len(records), n_fail)

    # Delta stats
    deltas = np.array([r["delta"] for r in records])
    log_cmax = np.log10(np.array([r["cmax_obs"] for r in records]))
    log.info("  Var(log10 Cmax): %.3f", np.var(log_cmax))
    log.info("  Var(delta):      %.3f (%.1f%% of Cmax var)",
             np.var(deltas), np.var(deltas) / np.var(log_cmax) * 100)
    log.info("  Delta range: [%.2f, %.2f], mean=%.3f, std=%.3f",
             deltas.min(), deltas.max(), deltas.mean(), deltas.std())

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: 4-Way Comparison
# ═══════════════════════════════════════════════════════════════════════════

def phase1(records):
    log.info("=" * 60)
    log.info("Phase 1: 4-Way Model Comparison")
    log.info("=" * 60)

    smiles = [r["smi"] for r in records]
    fp_mat = np.array([r["fp"] for r in records])
    adme_mat = np.array([[r["adme"][k] for k in sorted(r["adme"].keys())] for r in records], dtype=np.float32)
    fp_adme = np.hstack([fp_mat, adme_mat])

    y_cmax = np.array([r["log_cpd"] for r in records])  # log10(Cmax/Dose)
    y_delta = np.array([r["delta"] for r in records])    # log10(obs/engine)
    cmax_eng = np.array([r["cmax_eng"] for r in records])
    cmax_obs = np.array([r["cmax_obs"] for r in records])
    doses = np.array([r["dose"] for r in records])

    xgb_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      n_jobs=4, verbosity=0)

    folds = scaffold_split(smiles)
    configs = {
        "A_baseline":   (fp_mat, y_cmax, "cmax"),
        "B_adme_feat":  (fp_adme, y_cmax, "cmax"),
        "C_delta":      (fp_mat, y_delta, "delta"),
        "D_delta_adme": (fp_adme, y_delta, "delta"),
    }

    results = {}
    for name, (X, y, mode) in configs.items():
        preds = np.zeros(len(y))
        for fi in range(5):
            te = folds[fi]; tr = [j for f in range(5) if f != fi for j in folds[f]]
            m = xgb.XGBRegressor(**xgb_params)
            m.fit(X[tr], y[tr], verbose=False)
            preds[te] = m.predict(X[te])

        # Convert to Cmax for AAFE
        if mode == "cmax":
            cmax_pred = 10 ** preds * doses
        else:  # delta
            cmax_pred = cmax_eng * (10 ** preds)

        aafe = compute_aafe(cmax_pred, cmax_obs)

        # R² on own target
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        mae = float(np.mean(np.abs(y - preds)))

        results[name] = {"aafe": aafe, "r2": r2, "mae": mae, "n_feat": X.shape[1]}
        log.info("  %s: AAFE=%.3f, R²=%.3f, MAE=%.3f (features=%d)",
                 name, aafe, r2, mae, X.shape[1])

    # Print comparison
    print()
    print("=" * 70)
    print("  4-WAY CV COMPARISON")
    print("=" * 70)
    print()
    print(f"  {'Config':<20s} {'Features':>8s} {'Target':<6s} {'CV AAFE':>8s} {'R²':>6s}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        target = "delta" if "delta" in name else "Cmax"
        print(f"  {name:<20s} {r['n_feat']:>8d} {target:<6s} {r['aafe']:>8.3f} {r['r2']:>6.3f}")

    return results, fp_mat, fp_adme, y_delta, folds


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Holdout Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def phase2(records, cv_results, fp_mat, fp_adme, y_delta):
    log.info("=" * 60)
    log.info("Phase 2: Holdout Benchmark")
    log.info("=" * 60)

    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme

    # Find best CV config
    best_config = min(cv_results, key=lambda k: cv_results[k]["aafe"])
    log.info("Best CV config: %s (AAFE=%.3f)", best_config, cv_results[best_config]["aafe"])

    # Train delta+ADME and delta-only on ALL training data
    xgb_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      n_jobs=4, verbosity=0)

    model_delta = xgb.XGBRegressor(**xgb_params)
    model_delta.fit(fp_mat, y_delta, verbose=False)

    model_delta_adme = xgb.XGBRegressor(**xgb_params)
    model_delta_adme.fit(fp_adme, y_delta, verbose=False)

    # Load holdout predictions (engine Cmax already computed)
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_pred = json.load(f)
    ho_dict = {d["name"]: d for d in ho_pred}

    from sisyphus.validation.reference import load_reference
    refs = [r for r in load_reference() if r.in_holdout]

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    holdout = []
    for ref in refs:
        try:
            hp = ho_dict.get(ref.name)
            if not hp or hp["cmax_engine"] <= 0:
                continue

            cmax_eng = hp["cmax_engine"]
            cmax_ml = hp["cmax_combined"]  # baseline meta prediction

            # Compute features for delta model
            fp = compute_features(ref.smiles).astype(np.float32)
            profile = compute_profile(ref.smiles)
            adme = predict_adme(profile)
            mol = Chem.MolFromSmiles(ref.smiles)

            adme_feats = np.array([
                np.log10(max(adme.clint.mean, 1e-6)),
                adme.fup.mean,
                Descriptors.NumHDonors(mol) / 5.0 if mol else 0,
                1.0 if profile.compound_type == "acid" else 0.0,
                1.0 if profile.compound_type == "base" else 0.0,
                Descriptors.MolLogP(mol) if mol else 0,
                Descriptors.MolWt(mol) / 600.0 if mol else 0,
                np.log10(max(adme.peff.mean, 1e-6)),
                adme.rbp.mean,
                Descriptors.NumRotatableBonds(mol) / 15.0 if mol else 0,
                np.log10(max(adme.solubility.mean, 1e-6)),
                Descriptors.TPSA(mol) / 200.0 if mol else 0,
                np.log10(max(adme.vdss.mean, 1e-6)),
                Descriptors.NumHAcceptors(mol) / 10.0 if mol else 0,
            ], dtype=np.float32)

            # Delta predictions
            delta_pred = float(model_delta.predict(fp.reshape(1, -1))[0])
            cmax_delta = cmax_eng * (10 ** delta_pred)

            delta_adme_pred = float(model_delta_adme.predict(
                np.concatenate([fp, adme_feats]).reshape(1, -1))[0])
            cmax_delta_adme = cmax_eng * (10 ** delta_adme_pred)

            # Baseline ML prediction
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            cmax_baseline_ml = result.ml_pk.cmax.mean if result.ml_pk else cmax_ml
            cmax_baseline_meta = result.pk.cmax.mean

            holdout.append({
                "name": ref.name, "obs": ref.cmax_obs,
                "eng": cmax_eng, "ml_base": cmax_baseline_ml,
                "meta_base": cmax_baseline_meta,
                "delta": cmax_delta, "delta_adme": cmax_delta_adme,
                "is_base": profile.compound_type == "base",
            })
        except Exception:
            continue

    logging.getLogger("sisyphus").setLevel(logging.INFO)

    df = pd.DataFrame(holdout)
    log.info("Holdout evaluated: %d drugs", len(df))

    def _aafe(col): return compute_aafe(df[col].values, df["obs"].values)
    def _aafe_sub(col, mask):
        sub = df[mask]
        return compute_aafe(sub[col].values, sub["obs"].values) if len(sub) > 3 else float("inf")

    base_mask = df["is_base"]
    nonbase_mask = ~df["is_base"]

    # Hybrid: base=delta_adme, non-base=ml_base
    df["hybrid"] = np.where(base_mask, df["delta_adme"], df["ml_base"])

    # Print results
    print()
    print("=" * 75)
    print("  HOLDOUT BENCHMARK (N=%d)" % len(df))
    print("=" * 75)
    print()
    print(f"  {'Configuration':<30s} {'All':>7s} {'Base(%d)' % base_mask.sum():>9s} {'NonBase(%d)' % nonbase_mask.sum():>11s}")
    print(f"  {'-'*60}")
    for col, label in [
        ("ml_base", "Baseline ML"),
        ("meta_base", "Baseline Meta"),
        ("eng", "Engine only"),
        ("delta", "Delta (FP only)"),
        ("delta_adme", "Delta (FP+ADME)"),
        ("hybrid", "Hybrid (base=Δ,nb=ML)"),
    ]:
        a_all = _aafe(col)
        a_base = _aafe_sub(col, base_mask)
        a_nb = _aafe_sub(col, nonbase_mask)
        print(f"  {label:<30s} {a_all:>7.3f} {a_base:>9.3f} {a_nb:>11.3f}")

    # Gate
    baseline_meta = _aafe("meta_base")
    best_delta = min(_aafe("delta"), _aafe("delta_adme"), _aafe("hybrid"))
    delta_val = best_delta - baseline_meta

    print()
    print(f"  Baseline meta: {baseline_meta:.3f}")
    print(f"  Best delta:    {best_delta:.3f}")
    print(f"  Delta:         {delta_val:+.3f}")
    print()

    if delta_val < -0.01:
        print("  GATE: ✓ KEEP")
        decision = "KEEP"
    elif delta_val > 0.01:
        print("  GATE: ✗ REVERT")
        decision = "REVERT"
    else:
        print(f"  GATE: ~ NOISE ({delta_val:+.3f})")
        decision = "NOISE"

    # Feature importance for delta+ADME model
    importances = model_delta_adme.feature_importances_
    n_fp = fp_mat.shape[1]
    adme_keys = sorted(records[0]["adme"].keys())
    adme_imp = {k: float(importances[n_fp + i]) for i, k in enumerate(adme_keys)}
    top_adme = sorted(adme_imp.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top ADME features in delta model:")
    for k, v in top_adme:
        print(f"    {k:<15s}: importance={v:.4f}")

    return {
        "n_holdout": len(df),
        "aafe_ml_baseline": _aafe("ml_base"),
        "aafe_meta_baseline": baseline_meta,
        "aafe_engine": _aafe("eng"),
        "aafe_delta_fp": _aafe("delta"),
        "aafe_delta_adme": _aafe("delta_adme"),
        "aafe_hybrid": _aafe("hybrid"),
        "aafe_delta_base": _aafe_sub("delta_adme", base_mask),
        "aafe_delta_nonbase": _aafe_sub("delta_adme", nonbase_mask),
        "aafe_ml_base": _aafe_sub("ml_base", base_mask),
        "aafe_ml_nonbase": _aafe_sub("ml_base", nonbase_mask),
        "best_delta": best_delta,
        "delta_vs_meta": delta_val,
        "decision": decision,
        "top_adme_features": top_adme,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    records = phase0()

    if len(records) < 200:
        log.error("Too few records (%d). Aborting.", len(records))
        return

    cv_results, fp_mat, fp_adme, y_delta, folds = phase1(records)
    holdout_results = phase2(records, cv_results, fp_mat, fp_adme, y_delta)

    # Save
    all_results = {
        "phase0": {"n": len(records),
                   "var_log_cmax": float(np.var([np.log10(r["cmax_obs"]) for r in records])),
                   "var_delta": float(np.var([r["delta"] for r in records]))},
        "phase1_cv": {k: {kk: vv for kk, vv in v.items()} for k, v in cv_results.items()},
        "phase2_holdout": holdout_results,
    }
    out = ROOT / "data/validation/delta_model_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
