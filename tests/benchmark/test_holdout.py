"""Holdout benchmark tests.

These tests run the full pipeline on holdout drugs and check
that acceptance criteria are met.  Marked ``slow`` because each
drug takes 2-5 seconds.
"""

from __future__ import annotations

import pytest

from sisyphus.validation.benchmark import run_benchmark


class TestHoldoutBenchmark:
    @pytest.mark.slow
    def test_holdout_aafe(self) -> None:
        """Run benchmark on first 10 holdout drugs.

        Acceptance criterion: AAFE < 10.0 (lenient for small sample).
        Target for full holdout: AAFE <= 2.5.
        """
        result = run_benchmark(holdout_only=True, max_drugs=10)
        assert result.n_drugs >= 5, f"Too few drugs evaluated: {result.n_drugs}"
        assert result.aafe < 10.0, f"AAFE too high: {result.aafe}"

        print(f"\nHoldout benchmark (N={result.n_drugs}):")
        print(f"  AAFE: {result.aafe:.3f}")
        print(f"  %2-fold: {result.pct_2fold:.1f}%")

    @pytest.mark.slow
    def test_benchmark_result_fields(self) -> None:
        """Verify BenchmarkResult has all expected fields."""
        result = run_benchmark(holdout_only=True, max_drugs=3)
        assert isinstance(result.n_drugs, int)
        assert isinstance(result.aafe, float)
        assert isinstance(result.pct_2fold, float)
        # Default run (compute_pi=False): PI coverage not computed.
        assert result.pi_coverage_90 is None
        assert result.n_pi_computed is None

    @pytest.mark.slow
    def test_pi_coverage_wired_when_compute_pi_true(self) -> None:
        """compute_pi=True should populate pi_coverage_90 as a float in [0, 1]."""
        # Small n_mc_samples + max_drugs keeps runtime low; we only verify the
        # wiring, not statistical validity of the interval.
        result = run_benchmark(
            holdout_only=True,
            max_drugs=3,
            compute_pi=True,
            n_mc_samples=50,
        )
        # Every evaluated drug should have produced an interval (MC enabled,
        # no infusion cases in the first 3 holdout drugs).
        assert result.n_pi_computed is not None
        assert result.n_pi_computed >= 1, (
            f"Expected >=1 drug with PI, got {result.n_pi_computed}"
        )
        assert result.pi_coverage_90 is not None
        assert isinstance(result.pi_coverage_90, float)
        assert 0.0 <= result.pi_coverage_90 <= 1.0, (
            f"Coverage out of [0, 1]: {result.pi_coverage_90}"
        )
