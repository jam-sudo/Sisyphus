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
        assert result.pi_coverage_90 is None  # Not yet implemented
