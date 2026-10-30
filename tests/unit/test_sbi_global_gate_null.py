"""The calibrated-rank gate simulation preserves the current test shape."""

import numpy as np

from scripts.sbi_global_gate_null import (
    _inside_quantile_interval,
    gate_probability,
    production_scope_probability,
)


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
    assert result["size_calibrated"]["simultaneous_tests"] == 30
    assert 0.9 < result["size_calibrated"]["pass_probability"] <= 1


def test_production_null_repeats_full_family_screening():
    result = production_scope_probability(
        [True, False], [False, False], 50, repetitions=100, seed=7,
    )
    assert result["tests_per_pair"] == 15
    assert 0.9 < result["pass_probability"] <= result["nonempty_probability"] <= 1
    empty = production_scope_probability([False], [True], 50, repetitions=20, seed=7)
    assert empty["pass_probability"] == 0
    assert empty["nonempty_probability"] == 0
    continuous = production_scope_probability(
        [True, True], [False, False], 50, continuous_coverage=True,
        repetitions=20, seed=7,
    )
    assert continuous["tests_per_pair"] == 7
    assert 0 <= continuous["pass_probability"] <= 1
