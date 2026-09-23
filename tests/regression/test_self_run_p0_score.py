"""Independent arithmetic check of the frozen P0 compound-cluster score."""

import json
import math
from collections import defaultdict
from pathlib import Path


def test_p0_cluster_score():
    root = Path(__file__).resolve().parents[2] / "data/validation"
    predictions = json.loads((root / "self_run_p0_predictions.json").read_text())
    rulings = json.loads((root / "self_run_p0_source_adjudication.json").read_text())
    reported = json.loads((root / "self_run_p0_results.json").read_text())
    lookup = {(r["candidate_id"], r["arm_id"]): r for r in predictions["rows"]}
    errors = defaultdict(lambda: ([], []))
    n_arms = 0
    for ruling in rulings["candidates"]:
        for evidence in ruling["included_arm_evidence"]:
            prediction = lookup[ruling["candidate_id"], evidence["arm_id"]]
            observed = evidence["observed_cmax_mg_l"]
            errors[ruling["candidate_id"]][0].append(abs(math.log(prediction["meta_cmax_mg_l"] / observed)))
            errors[ruling["candidate_id"]][1].append(abs(math.log(prediction["ml_cmax_mg_l"] / observed)))
            n_arms += 1
    assert len(errors) == reported["n_compounds"] == 18
    assert n_arms == reported["n_arms"] == 58
    mean_meta = sum(sum(v[0]) / len(v[0]) for v in errors.values()) / len(errors)
    mean_ml = sum(sum(v[1]) / len(v[1]) for v in errors.values()) / len(errors)
    assert math.isclose(math.exp(mean_meta), reported["meta_aafe"], rel_tol=1e-12)
    assert math.isclose(math.exp(mean_ml), reported["ml_aafe"], rel_tol=1e-12)
    assert math.isclose(math.exp(mean_meta - mean_ml), reported["meta_ml_aafe_ratio"], rel_tol=1e-12)
