"""Test VDss-analytical Cmax as a new track.

Formula: Cmax_vdss = dose_mg / (VDss_predicted_L_per_kg * 70 kg)
This assumes F=1 and instant absorption. Tests whether including this as
a new track in the meta-learner improves holdout AAFE.
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path

logging.disable(logging.WARNING)

import numpy as np
import xgboost as xgb

from sisyphus.descriptors import compute_features

# Load BOTH VDss models
BW_KG = 70.0
V1_PATH = "/home/jam/Sisyphus/models/adme/xgboost_vdss.json"
V2_PATH = "/home/jam/Sisyphus/models/adme/xgboost_vdss_v2.json"

v1 = xgb.Booster()
v1.load_model(V1_PATH)
v2 = xgb.Booster()
v2.load_model(V2_PATH)


def predict_vdss(smi: str, model: xgb.Booster) -> float:
    f = compute_features(smi).reshape(1, -1)
    log_v = float(model.predict(xgb.DMatrix(f))[0])
    return max(10 ** log_v, 0.01)


def cmax_vdss(smi: str, dose_mg: float, model: xgb.Booster) -> float:
    v = predict_vdss(smi, model)
    return dose_mg / (v * BW_KG)


def main():
    # Load cached predictions from earlier sweep
    with open("data/validation/meta_weight_sweep_cache.json") as f:
        cache = json.load(f)

    # We need SMILES for each drug. Let me load from clinical_pk.json and curation.
    with open("data/reference/clinical_pk.json") as f:
        pk = json.load(f)["drugs"]

    # Map names in cache to smiles via clinical_pk
    for d in cache:
        name = d["name"].lower()
        for k, v in pk.items():
            if k.lower() == name:
                d["smiles"] = v.get("smiles")
                break
        # Try DrugBank for prospective drugs
        if not d.get("smiles"):
            import pandas as pd
            db = pd.read_csv("data/drugbank/drugs.csv")
            match = db[db.name.str.lower() == name]
            if len(match) > 0:
                d["smiles"] = match.iloc[0]["smiles"]

    # Compute VDss-analytical Cmax for each drug (both v1 and v2)
    for d in cache:
        smi = d.get("smiles")
        if not smi:
            d["cmax_vdss_v1"] = None
            d["cmax_vdss_v2"] = None
            continue
        try:
            d["cmax_vdss_v1"] = cmax_vdss(smi, d["dose_mg"], v1)
            d["cmax_vdss_v2"] = cmax_vdss(smi, d["dose_mg"], v2)
        except Exception as e:
            d["cmax_vdss_v1"] = None
            d["cmax_vdss_v2"] = None

    # Compute standalone AAFE for each track
    def aafe(drugs, key):
        folds = [d[key] / d["obs"] for d in drugs
                 if d.get(key) is not None and d.get("obs") and d[key] > 0]
        if not folds:
            return None, 0
        logs = [abs(math.log(f)) for f in folds]
        aafe = math.exp(sum(logs) / len(logs))
        pct = sum(1 for f in folds if 0.5 <= f <= 2.0) / len(folds)
        return aafe, len(folds), pct

    print("=== Standalone track AAFE ===")
    print(f"{'cohort':20s} {'engine':>10s} {'ml':>10s} {'meta':>10s} {'vdss_v1':>10s} {'vdss_v2':>10s}")
    for cohort_name, cohort in [
        ("All holdout N=107", [d for d in cache if d["in_holdout"]]),
        ("Holdout in-domain", [d for d in cache if d["in_holdout"] and not d.get("ad_flags")]),
        ("Prospective N=15", [d for d in cache if not d["in_holdout"]]),
        ("Kinase class (all)", [d for d in cache if d["is_kinase"]]),
        ("Kinase class (prosp)", [d for d in cache if d["is_kinase"] and not d["in_holdout"]]),
    ]:
        a_e, n_e, _ = aafe(cohort, "engine")
        a_m, n_m, _ = aafe(cohort, "ml")
        a_meta, n_meta, _ = aafe(cohort, "meta_baseline")
        a_v1, n_v1, _ = aafe(cohort, "cmax_vdss_v1")
        a_v2, n_v2, _ = aafe(cohort, "cmax_vdss_v2")
        print(f"{cohort_name:20s} {a_e:>10.3f} {a_m:>10.3f} {a_meta:>10.3f} {a_v1:>10.3f} {a_v2:>10.3f}")

    # Try 4-track meta with vdss_v1 as new track
    print()
    print("=== Try 4-track Meta (engine + ml + vdss_v1) ===")
    print(f"{'config':40s} {'AAFE_hold':>10s} {'AAFE_id':>10s} {'AAFE_pros':>10s}")
    configs = [
        ("baseline (engine 0.60 + ml 0.40)", 0.60, 0.40, 0.00),
        ("0.50 eng + 0.40 ml + 0.10 vdss", 0.50, 0.40, 0.10),
        ("0.50 eng + 0.30 ml + 0.20 vdss", 0.50, 0.30, 0.20),
        ("0.40 eng + 0.30 ml + 0.30 vdss", 0.40, 0.30, 0.30),
        ("0.35 eng + 0.30 ml + 0.35 vdss", 0.35, 0.30, 0.35),
    ]
    for label, w_e, w_m, w_v in configs:
        for d in cache:
            if (d.get("engine") and d.get("ml") and d.get("cmax_vdss_v1")
                    and d["engine"] > 0 and d["ml"] > 0 and d["cmax_vdss_v1"] > 0):
                log_meta = (w_e * math.log10(d["engine"]) + w_m * math.log10(d["ml"])
                            + w_v * math.log10(d["cmax_vdss_v1"]))
                d["meta_4track"] = 10 ** log_meta
            else:
                d["meta_4track"] = d.get("meta_baseline")
        holdout = [d for d in cache if d["in_holdout"]]
        holdout_id = [d for d in cache if d["in_holdout"] and not d.get("ad_flags")]
        pros = [d for d in cache if not d["in_holdout"]]
        a_h, _, _ = aafe(holdout, "meta_4track")
        a_id, _, _ = aafe(holdout_id, "meta_4track")
        a_p, _, _ = aafe(pros, "meta_4track")
        print(f"{label:40s} {a_h:10.3f} {a_id:10.3f} {a_p:10.3f}")

    # Save
    Path("data/validation/vdss_analytical_track_results.json").write_text(json.dumps({
        "baseline_engine_aafe_holdout": aafe([d for d in cache if d["in_holdout"]], "engine")[0],
        "baseline_meta_aafe_holdout": aafe([d for d in cache if d["in_holdout"]], "meta_baseline")[0],
        "vdss_v1_standalone_aafe_holdout": aafe([d for d in cache if d["in_holdout"]], "cmax_vdss_v1")[0],
        "vdss_v2_standalone_aafe_holdout": aafe([d for d in cache if d["in_holdout"]], "cmax_vdss_v2")[0],
    }, indent=2))


if __name__ == "__main__":
    main()
