"""The calibrated-rank gate simulation preserves the current test shape."""

import numpy as np

from scripts.sbi_global_gate_null import _inside_quantile_interval, gate_probability


def test_interpolated_coverage_is_calibrated_for_exact_ranks():
    rng = np.random.default_rng(7)
    ranks = rng.integers(0, 501, size=10000)
    inside = _inside_quantile_interval(ranks, 90, rng)
    assert 0.88 < inside.mean() < 0.92


def test_exact_rank_null_uses_all_three_parameters():
    result = gate_probability(2, 50, repetitions=100, seed=7)
    assert result["simultaneous_ks_tests"] == 6
    assert result["pairs"] == 2
    assert 0 <= result["passed"] <= result["repetitions"]
    assert result["ci_95"][0] <= result["pass_probability"] <= result["ci_95"][1]
    assert result["holm"]["passed"] >= result["passed"]
