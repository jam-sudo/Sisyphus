"""The calibrated-rank gate simulation preserves the current test shape."""

from scripts.sbi_global_gate_null import gate_probability


def test_exact_rank_null_uses_all_three_parameters():
    result = gate_probability(2, 50, repetitions=100, seed=7)
    assert result["simultaneous_ks_tests"] == 6
    assert result["pairs"] == 2
    assert 0 <= result["passed"] <= result["repetitions"]
    assert result["ci_95"][0] <= result["pass_probability"] <= result["ci_95"][1]
