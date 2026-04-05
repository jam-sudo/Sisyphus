"""Full LOOCV for adaptive 4-track meta-learner (adds VDss analytical track).

Tests robustness of best adaptive config against leave-one-out cross-validation.
"""
from __future__ import annotations

import json
import math
import logging

logging.disable(logging.WARNING)


def aafe_compute(folds):
    if not folds:
        return None
    return math.exp(sum(abs(math.log(f)) for f in folds) / len(folds))


def blend_4track(d, w_e, w_m, w_v):
    if not (d.get("engine") and d.get("ml") and d.get("cmax_vdss_v1")
            and d["engine"] > 0 and d["ml"] > 0 and d["cmax_vdss_v1"] > 0):
        return d.get("meta_baseline")
    s = w_e + w_m + w_v
    if s == 0:
        return d.get("meta_baseline")
    log_meta = ((w_e/s) * math.log10(d["engine"]) + (w_m/s) * math.log10(d["ml"])
                + (w_v/s) * math.log10(d["cmax_vdss_v1"]))
    return 10 ** log_meta


def objective(holdout, w_eb, w_mb, w_vb, w_eo, w_mo, w_vo):
    folds = []
    for d in holdout:
        if d["ctype"] == "base":
            pred = blend_4track(d, w_eb, w_mb, w_vb)
        else:
            pred = blend_4track(d, w_eo, w_mo, w_vo)
        if pred and d.get("obs") and pred > 0:
            folds.append(pred / d["obs"])
    return aafe_compute(folds)


def main():
    with open("data/validation/meta_weight_sweep_cache.json") as f:
        cache = json.load(f)

    # Recompute VDss analytical
    import xgboost as xgb
    from sisyphus.descriptors import compute_features
    v1 = xgb.Booster(); v1.load_model("/home/jam/Sisyphus/models/adme/xgboost_vdss.json")
    with open("data/reference/clinical_pk.json") as f:
        pk = json.load(f)["drugs"]
    import pandas as pd
    db = pd.read_csv("data/drugbank/drugs.csv")
    db_map = {str(row["name"]).lower(): row["smiles"] for _, row in db.iterrows()}

    BW = 70.0
    for d in cache:
        name = d["name"].lower()
        smi = pk.get(name, {}).get("smiles") or db_map.get(name)
        if not smi:
            d["cmax_vdss_v1"] = None
            continue
        try:
            f = compute_features(smi).reshape(1, -1)
            vd = max(10 ** float(v1.predict(xgb.DMatrix(f))[0]), 0.01)
            d["cmax_vdss_v1"] = d["dose_mg"] / (vd * BW)
        except Exception:
            d["cmax_vdss_v1"] = None

    holdout = [d for d in cache if d["in_holdout"]]
    print(f"Holdout N={len(holdout)}")

    # Grid search over (w_eb, w_mb, w_vb) and (w_eo, w_mo, w_vo)
    # Keep weights ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, sum = 1.0
    import itertools
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    valid_triples = [
        (a, b, c) for a in weights for b in weights for c in weights
        if abs(a + b + c - 1.0) < 0.01
    ]
    print(f"Valid weight triples: {len(valid_triples)}")

    # Search 2D: best for base, best for other
    print()
    print("=== Grid search: best triple per compound_type ===")
    base_drugs = [d for d in holdout if d["ctype"] == "base"]
    other_drugs = [d for d in holdout if d["ctype"] != "base"]
    print(f"Base N={len(base_drugs)}, Other N={len(other_drugs)}")

    def best_triple(subset):
        best = None
        for w_e, w_m, w_v in valid_triples:
            folds = []
            for d in subset:
                p = blend_4track(d, w_e, w_m, w_v)
                if p and d.get("obs") and p > 0:
                    folds.append(p / d["obs"])
            a = aafe_compute(folds)
            if a and (best is None or a < best[0]):
                best = (a, w_e, w_m, w_v)
        return best

    best_base = best_triple(base_drugs)
    best_other = best_triple(other_drugs)
    print(f"Best BASE:  AAFE={best_base[0]:.3f} @ eng={best_base[1]}/ml={best_base[2]}/vdss={best_base[3]}")
    print(f"Best OTHER: AAFE={best_other[0]:.3f} @ eng={best_other[1]}/ml={best_other[2]}/vdss={best_other[3]}")

    w_eb, w_mb, w_vb = best_base[1:]
    w_eo, w_mo, w_vo = best_other[1:]
    print()
    print(f"=== Full-data AAFE with best adaptive config ===")
    a_full = objective(holdout, w_eb, w_mb, w_vb, w_eo, w_mo, w_vo)
    print(f"Full holdout AAFE: {a_full:.3f}")

    # LOOCV: drop each drug, optimize on remaining, test on dropped
    print()
    print("=== LOOCV: drop-one, refit, test on held-out ===")
    loo_folds = []
    loo_weight_votes = {"base": [], "other": []}
    for i in range(len(holdout)):
        train = holdout[:i] + holdout[i+1:]
        test = holdout[i]
        train_base = [d for d in train if d["ctype"] == "base"]
        train_other = [d for d in train if d["ctype"] != "base"]
        bb = best_triple(train_base)
        bo = best_triple(train_other)
        if bb is None or bo is None:
            continue
        loo_weight_votes["base"].append((bb[1], bb[2], bb[3]))
        loo_weight_votes["other"].append((bo[1], bo[2], bo[3]))
        # Predict test
        if test["ctype"] == "base":
            pred = blend_4track(test, bb[1], bb[2], bb[3])
        else:
            pred = blend_4track(test, bo[1], bo[2], bo[3])
        if pred and test.get("obs") and pred > 0:
            loo_folds.append(pred / test["obs"])
        if i % 20 == 0:
            print(f"  LOOCV {i+1}/{len(holdout)}...")

    a_loo = aafe_compute(loo_folds)
    print()
    print(f"LOOCV AAFE: {a_loo:.3f}")
    # Stability: how often does each weight get chosen?
    from collections import Counter
    bc = Counter(loo_weight_votes["base"])
    oc = Counter(loo_weight_votes["other"])
    print(f"BASE weight stability (top 3):")
    for t, c in bc.most_common(3):
        print(f"  {t}: {c}/{len(loo_weight_votes['base'])} ({c/len(loo_weight_votes['base'])*100:.0f}%)")
    print(f"OTHER weight stability (top 3):")
    for t, c in oc.most_common(3):
        print(f"  {t}: {c}/{len(loo_weight_votes['other'])} ({c/len(loo_weight_votes['other'])*100:.0f}%)")

    # Apply to prospective
    pros = [d for d in cache if not d["in_holdout"]]
    for d in pros:
        if d["ctype"] == "base":
            d["meta_4t"] = blend_4track(d, w_eb, w_mb, w_vb)
        else:
            d["meta_4t"] = blend_4track(d, w_eo, w_mo, w_vo)
    folds_p = [d["meta_4t"] / d["obs"] for d in pros if d.get("meta_4t") and d.get("obs") and d["meta_4t"] > 0]
    print(f"\nProspective N={len(folds_p)} with best config: AAFE={aafe_compute(folds_p):.3f}")

    # Save
    out = {
        "baseline_holdout_aafe": 2.808,
        "best_adaptive_4track_full": float(a_full),
        "best_adaptive_4track_loocv": float(a_loo),
        "best_base_weights": {"eng": w_eb, "ml": w_mb, "vdss": w_vb, "aafe": best_base[0]},
        "best_other_weights": {"eng": w_eo, "ml": w_mo, "vdss": w_vo, "aafe": best_other[0]},
        "prospective_aafe": aafe_compute(folds_p),
        "baseline_prospective_aafe": 2.546,
    }
    import pathlib
    pathlib.Path("data/validation/vdss_track_loocv_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved results.")


if __name__ == "__main__":
    main()
