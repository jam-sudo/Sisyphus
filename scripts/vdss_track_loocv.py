"""LOOCV for 4-track meta (engine + ml + vdss_v1/v2 analytical) on holdout.

Tests whether adding VDss-analytical Cmax as a new track survives LOOCV,
and whether adaptive weights (base vs other) can also include VDss track.
"""
from __future__ import annotations

import json
import math
import logging
from itertools import product

logging.disable(logging.WARNING)


def aafe(drugs, key, obs_key="obs"):
    folds = [d[key] / d[obs_key] for d in drugs
             if d.get(key) is not None and d.get(obs_key) and d[key] > 0]
    if not folds:
        return None
    return math.exp(sum(abs(math.log(f)) for f in folds) / len(folds))


def blend_4track(d, w_e, w_m, w_v, vdss_key="cmax_vdss_v1"):
    if not (d.get("engine") and d.get("ml") and d.get(vdss_key)
            and d["engine"] > 0 and d["ml"] > 0 and d[vdss_key] > 0):
        return d.get("meta_baseline")
    s = w_e + w_m + w_v
    if s == 0:
        return d.get("meta_baseline")
    w_e, w_m, w_v = w_e / s, w_m / s, w_v / s
    log_meta = (w_e * math.log10(d["engine"]) + w_m * math.log10(d["ml"])
                + w_v * math.log10(d[vdss_key]))
    return 10 ** log_meta


def main():
    with open("data/validation/meta_weight_sweep_cache.json") as f:
        cache = json.load(f)

    # Merge in VDss analytical predictions from earlier script
    # Need to recompute - the results were not saved per-drug
    # Just run inline
    import xgboost as xgb
    from sisyphus.descriptors import compute_features
    v1 = xgb.Booster(); v1.load_model("/home/jam/Sisyphus/models/adme/xgboost_vdss.json")
    v2 = xgb.Booster(); v2.load_model("/home/jam/Sisyphus/models/adme/xgboost_vdss_v2.json")
    with open("data/reference/clinical_pk.json") as f:
        pk = json.load(f)["drugs"]
    import pandas as pd
    db = pd.read_csv("data/drugbank/drugs.csv")
    db_map = {str(row["name"]).lower(): row["smiles"] for _, row in db.iterrows()}

    BW = 70.0
    for d in cache:
        name = d["name"].lower()
        smi = None
        if name in pk:
            smi = pk[name].get("smiles")
        if not smi:
            smi = db_map.get(name)
        if not smi:
            d["cmax_vdss_v1"] = None
            d["cmax_vdss_v2"] = None
            continue
        try:
            f = compute_features(smi).reshape(1, -1)
            vd1 = max(10 ** float(v1.predict(xgb.DMatrix(f))[0]), 0.01)
            vd2 = max(10 ** float(v2.predict(xgb.DMatrix(f))[0]), 0.01)
            d["cmax_vdss_v1"] = d["dose_mg"] / (vd1 * BW)
            d["cmax_vdss_v2"] = d["dose_mg"] / (vd2 * BW)
            d["vdss_v1"] = vd1
            d["vdss_v2"] = vd2
        except Exception:
            d["cmax_vdss_v1"] = None
            d["cmax_vdss_v2"] = None

    # Holdout cohort
    holdout = [d for d in cache if d["in_holdout"]]
    print(f"Holdout N={len(holdout)}")

    # Weight grid search with LOOCV (fixed weights, not compound-type-adaptive)
    print()
    print("=== Grid search with LOOCV (holdout N=107) ===")
    print(f"{'cfg':35s} {'AAFE_train':>11s} {'AAFE_loo':>10s} {'stability':>10s}")
    best_loo = None
    for w_e, w_m, w_v in [
        (0.60, 0.40, 0.00),  # baseline no vdss
        (0.50, 0.40, 0.10), (0.50, 0.30, 0.20), (0.50, 0.20, 0.30),
        (0.40, 0.40, 0.20), (0.40, 0.30, 0.30), (0.40, 0.20, 0.40),
        (0.45, 0.35, 0.20), (0.45, 0.30, 0.25), (0.45, 0.25, 0.30),
        (0.35, 0.35, 0.30), (0.35, 0.30, 0.35), (0.35, 0.25, 0.40),
        (0.30, 0.30, 0.40),
    ]:
        for d in holdout:
            d["meta_4t"] = blend_4track(d, w_e, w_m, w_v, "cmax_vdss_v1")
        aafe_train = aafe(holdout, "meta_4t")
        # LOOCV: drop each drug, compute AAFE on remaining N-1
        loo_aafes = []
        for i in range(len(holdout)):
            rest = holdout[:i] + holdout[i+1:]
            a = aafe(rest, "meta_4t")
            if a is not None:
                loo_aafes.append(a)
        loo_mean = sum(loo_aafes) / len(loo_aafes) if loo_aafes else None
        stability = 1 - (max(loo_aafes) - min(loo_aafes)) / loo_mean if loo_mean else 0
        cfg = f"{w_e:.2f}/{w_m:.2f}/{w_v:.2f}"
        print(f"{cfg:35s} {aafe_train:11.3f} {loo_mean:10.3f} {stability:10.3f}")
        if best_loo is None or aafe_train < best_loo[0]:
            best_loo = (aafe_train, w_e, w_m, w_v)

    # Report best config on prospective
    print()
    print(f"Best train AAFE config: {best_loo[1]:.2f}/{best_loo[2]:.2f}/{best_loo[3]:.2f} → {best_loo[0]:.3f}")
    pros = [d for d in cache if not d["in_holdout"]]
    for d in pros:
        d["meta_4t"] = blend_4track(d, best_loo[1], best_loo[2], best_loo[3], "cmax_vdss_v1")
    a_pros = aafe(pros, "meta_4t")
    print(f"Prospective N=15 with best config: {a_pros:.3f}")

    # Adaptive 4-track: separate weights for base vs other (like current meta)
    print()
    print("=== Adaptive 4-track (base vs other) ===")
    # Use current weights as base (0.60, 0.40, 0.00) for base; (0.35, 0.50, 0.15) for other — then add vdss
    print(f"{'cfg':35s} {'AAFE_hold':>10s} {'AAFE_id':>10s} {'AAFE_pros':>10s}")
    for w_eb, w_mb, w_vb, w_eo, w_mo, w_vo in [
        (0.60, 0.40, 0.00, 0.35, 0.50, 0.15),  # current baseline (no vdss)
        (0.50, 0.30, 0.20, 0.30, 0.40, 0.30),  # add vdss 0.2 base, 0.3 other
        (0.40, 0.30, 0.30, 0.25, 0.35, 0.40),  # heavier vdss
        (0.40, 0.20, 0.40, 0.25, 0.30, 0.45),  # dominant vdss
        (0.30, 0.30, 0.40, 0.20, 0.30, 0.50),  # very heavy vdss
    ]:
        for d in cache:
            if d["ctype"] == "base":
                d["meta_a4t"] = blend_4track(d, w_eb, w_mb, w_vb, "cmax_vdss_v1")
            else:
                d["meta_a4t"] = blend_4track(d, w_eo, w_mo, w_vo, "cmax_vdss_v1")
        h = [d for d in cache if d["in_holdout"]]
        h_id = [d for d in cache if d["in_holdout"] and not d.get("ad_flags")]
        p = [d for d in cache if not d["in_holdout"]]
        a_h = aafe(h, "meta_a4t")
        a_id = aafe(h_id, "meta_a4t")
        a_p = aafe(p, "meta_a4t")
        cfg = f"base:{w_eb}/{w_mb}/{w_vb} other:{w_eo}/{w_mo}/{w_vo}"
        print(f"{cfg:60s} {a_h:10.3f} {a_id:10.3f} {a_p:10.3f}")


if __name__ == "__main__":
    main()
