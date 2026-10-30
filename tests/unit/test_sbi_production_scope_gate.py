"""Check the predeclared gate selects pairs by sampler behavior."""

from scripts.sbi_production_scope_gate import production_scope_gate


def test_production_scope_excludes_exact_and_uncomputable_rows():
    valid = {
        "name": "routed", "pass": True, "n_valid": 100,
        "ks_pvalues": [0.5, 0.5, 0.5],
        "coverage": {str(level): [level / 100] * 3 for level in (50, 80, 90, 95)},
        "rejection_fallback_cases": 0, "sampling_error_blocks": 0,
        "size_calibrated_rejections": 0,
    }
    exact = {**valid, "name": "exact", "rejection_fallback_cases": 1}
    mcmc = {**valid, "name": "mcmc", "mcmc_fallback_cases": 1}
    uncomputable = {**valid, "name": "uncomputable", "pass": None,
                    "sampling_error_blocks": 1}
    rejected = {**valid, "name": "rejected", "size_calibrated_rejections": 1}
    aggregate = {
        "drugs": [valid, exact, mcmc, uncomputable, rejected], "levels": [50, 80, 90, 95],
        "n_calibration": 100, "calibrated_null_assumptions": "exact ranks",
    }
    result = production_scope_gate(aggregate, "multi_drug", repetitions=20)
    assert result["selected_pairs"] == ["routed"]
    assert result["excluded_pairs"] == ["exact", "mcmc", "uncomputable", "rejected"]
    assert result["selected_count"] == 1
    assert result["test_count"] == 15
    assert result["status"] == "PASS"
    assert 0 <= result["calibrated_null_pass_probability"] <= 1

    valid["ks_pvalues"][0] = 0.000001
    assert production_scope_gate(aggregate, "multi_drug", repetitions=20)["status"] == "FAIL"
