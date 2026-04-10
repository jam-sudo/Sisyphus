"""Test class-aware meta weights on holdout N=107 + prospective N=15.

Hypothesis: For kinase-class drugs (MW>450 + logP 3-5), the ML model regresses
toward class mean and under-predicts tail cases. Engine-weighted blend
(or engine-only) may outperform the current compound_type-based adaptive weights.

This script:
1. Caches engine + ml Cmax predictions for all holdout + prospective drugs
2. Sweeps weight configurations for the kinase class
3. Reports AAFE under each configuration vs baseline
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path

logging.disable(logging.WARNING)

from sisyphus.pipeline.predict import predict
from sisyphus.predict.chemistry import compute_profile
from sisyphus.validation.reference import load_reference


def is_kinase_class(mw: float, logp: float) -> bool:
    return (mw > 450.0) and (2.5 <= logp <= 5.5)


def cache_predictions(drugs: list[tuple[str, str, float, float, bool]]) -> list[dict]:
    """Run pipeline and cache engine + ml + meta Cmax for each drug."""
    cache = []
    for i, (name, smi, dose, obs, in_holdout) in enumerate(drugs):
        try:
            prof = compute_profile(smi)
            r = predict(smi, dose)
            eng = float(r.engine_pk.cmax.mean) if r.engine_pk else None
            ml = float(r.ml_pk.cmax.mean) if r.ml_pk else None
            meta = float(r.pk.cmax.mean)
            cache.append({
                "name": name, "dose_mg": dose, "obs": obs,
                "mw": prof.mw, "logp": prof.logp, "ctype": prof.compound_type,
                "engine": eng, "ml": ml, "meta_baseline": meta,
                "ad_flags": list(r.ad_flags) if r.ad_flags else [],
                "is_kinase": is_kinase_class(prof.mw, prof.logp),
                "in_holdout": in_holdout,
            })
            if i % 20 == 0:
                print(f"  cached {i+1}/{len(drugs)}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    return cache


def compute_aafe(drugs: list[dict], pred_key: str) -> tuple[float, float, int]:
    """Compute AAFE, %2-fold, N for drugs where pred_key is available."""
    valid = [d for d in drugs if d.get(pred_key) is not None and d.get("obs") is not None]
    if not valid:
        return float("nan"), float("nan"), 0
    logs = [abs(math.log(d[pred_key] / d["obs"])) for d in valid if d[pred_key] > 0 and d["obs"] > 0]
    aafe = math.exp(sum(logs) / len(logs))
    folds = [d[pred_key] / d["obs"] for d in valid if d[pred_key] > 0 and d["obs"] > 0]
    pct_2f = sum(1 for f in folds if 0.5 <= f <= 2.0) / len(folds)
    return aafe, pct_2f, len(valid)


def apply_weights(drug: dict, w_eng_kinase: float, w_ml_kinase: float) -> float:
    """Geometric-weighted Meta for kinase drugs; baseline weights for non-kinase."""
    eng = drug.get("engine")
    ml = drug.get("ml")
    if eng is None or ml is None or eng <= 0 or ml <= 0:
        return drug.get("meta_baseline")
    if drug["is_kinase"]:
        log_meta = w_eng_kinase * math.log10(eng) + w_ml_kinase * math.log10(ml)
        return 10 ** log_meta
    # Non-kinase: use baseline meta (already computed with adaptive weights)
    return drug.get("meta_baseline")


def sweep(cache: list[dict], holdout_only: bool = True) -> dict:
    """Sweep weight configs and report AAFE."""
    drugs = [d for d in cache if d["in_holdout"] == holdout_only or not holdout_only]
    drugs_in_domain = [d for d in drugs if not d.get("ad_flags")]
    # Exclude extreme outliers (montelukast etc) for in-domain stats — but don't cherry-pick
    configs = [
        ("baseline (adaptive)", None, None),
        ("kinase: 1.00/0.00 (engine only)", 1.00, 0.00),
        ("kinase: 0.80/0.20", 0.80, 0.20),
        ("kinase: 0.70/0.30", 0.70, 0.30),
        ("kinase: 0.60/0.40", 0.60, 0.40),  # current base weights
        ("kinase: 0.50/0.50", 0.50, 0.50),
        ("kinase: 0.35/0.65 (current 'other')", 0.35, 0.65),
    ]
    n_kinase = sum(1 for d in drugs if d["is_kinase"])
    n_kinase_id = sum(1 for d in drugs_in_domain if d["is_kinase"])
    results = []
    for label, w_e, w_m in configs:
        # Apply weights
        for d in drugs:
            d["meta_test"] = apply_weights(d, w_e, w_m) if w_e is not None else d["meta_baseline"]
        aafe_all, pct_all, n_all = compute_aafe(drugs, "meta_test")
        aafe_id, pct_id, n_id = compute_aafe(drugs_in_domain, "meta_test")
        # Kinase subset
        kinase_drugs = [d for d in drugs if d["is_kinase"]]
        aafe_k, pct_k, n_k = compute_aafe(kinase_drugs, "meta_test")
        results.append({
            "config": label,
            "w_eng_kinase": w_e, "w_ml_kinase": w_m,
            "aafe_all": aafe_all, "pct_2f_all": pct_all, "n_all": n_all,
            "aafe_id": aafe_id, "pct_2f_id": pct_id, "n_id": n_id,
            "aafe_kinase": aafe_k, "pct_2f_kinase": pct_k, "n_kinase": n_k,
        })
    return {
        "n_kinase_in_cohort": n_kinase,
        "n_kinase_in_domain": n_kinase_id,
        "configs": results,
    }


def main():
    # Load holdout drugs with cmax
    refs = load_reference()
    holdout_drugs = [
        (r.name, r.smiles, r.dose_mg, r.cmax_obs, True)
        for r in refs if r.in_holdout and r.cmax_obs is not None and r.smiles
    ]
    print(f"Loaded {len(holdout_drugs)} holdout drugs")

    # Load prospective N=15 (10 original + 5 kinase batch)
    with open("data/validation/prospective_2024_2025_N10.json") as f:
        n10 = json.load(f)
    with open("data/validation/prospective_batch_N5_kinase.json") as f:
        n5 = json.load(f)
    # Need SMILES and doses for these
    # Map names from drugbank
    import pandas as pd
    db = pd.read_csv("data/drugbank/drugs.csv")
    db_map = {row["name"].lower(): row["smiles"] for _, row in db.iterrows() if isinstance(row.get("name"), str)}
    # N=10 drug curation (we have obs from the JSON)
    doses_n10 = {
        "resmetirom": 100.0, "aficamten": 20.0, "elafibranor": 120.0,
        "inavolisib": 9.0, "danicopan": 150.0, "berotralstat": 150.0,
        "vimseltinib": 30.0, "gepotidacin": 1500.0, "taletrectinib": 600.0,
        "suzetrigine": 100.0,
    }
    pros_drugs = []
    for d in n10["drugs"]:
        smi = db_map.get(d["name"].lower())
        dose = doses_n10.get(d["name"].lower())
        if smi and dose:
            pros_drugs.append((d["name"], smi, dose, d["obs"], False))
    for d in n5["drugs"]:
        if "fold" in d:
            # Read from n5 cache - already has smiles via curation
            smi = db_map.get(d["name"].lower())
            if smi:
                pros_drugs.append((d["name"], smi, d["dose_mg"], d["obs"], False))
    print(f"Loaded {len(pros_drugs)} prospective drugs")

    all_drugs = holdout_drugs + pros_drugs
    print(f"Total: {len(all_drugs)}")
    print()
    print("=== Caching predictions ===")
    cache = cache_predictions(all_drugs)

    # Save cache
    Path("data/validation/meta_weight_sweep_cache.json").write_text(json.dumps(cache, indent=2))
    print(f"Cached {len(cache)} drug predictions")

    # Sweep holdout
    print()
    print("=== Holdout sweep ===")
    holdout_sweep = sweep(cache, holdout_only=True)
    print(f"Kinase class in holdout: {holdout_sweep['n_kinase_in_cohort']} "
          f"({holdout_sweep['n_kinase_in_domain']} in-domain)")
    print()
    print(f"{'config':40s} {'AAFE_all':>9s} {'pct_2f':>7s} {'AAFE_id':>9s} {'AAFE_kin':>9s} {'%2f_kin':>8s}")
    for r in holdout_sweep["configs"]:
        print(f"{r['config']:40s} {r['aafe_all']:9.3f} {r['pct_2f_all']*100:6.1f}% "
              f"{r['aafe_id']:9.3f} {r['aafe_kinase']:9.3f} {r['pct_2f_kinase']*100:7.1f}%")

    # Sweep prospective
    print()
    print("=== Prospective N=15 sweep ===")
    pros_sweep = sweep(cache, holdout_only=False)
    # Filter to prospective-only
    for d in cache: d["is_holdout_d"] = d["in_holdout"]
    pros_cache = [d for d in cache if not d["in_holdout"]]
    # Re-do sweep only on prospective
    configs_pros = []
    for label, w_e, w_m in [
        ("baseline (adaptive)", None, None),
        ("kinase: 1.00/0.00 (engine)", 1.00, 0.00),
        ("kinase: 0.80/0.20", 0.80, 0.20),
        ("kinase: 0.70/0.30", 0.70, 0.30),
        ("kinase: 0.60/0.40", 0.60, 0.40),
    ]:
        for d in pros_cache:
            d["meta_test"] = apply_weights(d, w_e, w_m) if w_e is not None else d["meta_baseline"]
        aafe_all, pct_all, n_all = compute_aafe(pros_cache, "meta_test")
        k_drugs = [d for d in pros_cache if d["is_kinase"]]
        aafe_k, pct_k, n_k = compute_aafe(k_drugs, "meta_test")
        configs_pros.append({
            "config": label, "aafe_all": aafe_all, "pct_2f_all": pct_all, "n_all": n_all,
            "aafe_kinase": aafe_k, "pct_2f_kinase": pct_k, "n_kinase": n_k,
        })
    print(f"{'config':40s} {'AAFE_all':>9s} {'pct_2f':>7s} {'AAFE_kin':>9s}")
    for r in configs_pros:
        print(f"{r['config']:40s} {r['aafe_all']:9.3f} {r['pct_2f_all']*100:6.1f}% "
              f"{r['aafe_kinase']:9.3f} (n={r['n_kinase']})")

    # Save all results
    out = {
        "holdout_sweep": holdout_sweep,
        "prospective_sweep": configs_pros,
        "kinase_class_def": "MW > 450 AND 2.5 <= logP <= 5.5",
    }
    Path("data/validation/class_aware_meta_results.json").write_text(json.dumps(out, indent=2))
    print()
    print("Saved to data/validation/class_aware_meta_results.json")


if __name__ == "__main__":
    main()
