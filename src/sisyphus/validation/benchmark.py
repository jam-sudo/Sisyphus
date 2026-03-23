"""Holdout benchmark runner.

Runs predictions on the holdout set and computes acceptance metrics.
Enforces the invariant that holdout drugs are never used for training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from a holdout benchmark run.

    Attributes:
        n_drugs: Number of drugs evaluated.
        aafe: Absolute Average Fold Error.
        pct_2fold: Percentage of predictions within 2-fold.
        pi_coverage_90: 90% prediction interval coverage.
    """

    n_drugs: int
    aafe: float
    pct_2fold: float
    pi_coverage_90: float | None


def run_benchmark() -> BenchmarkResult:
    """Run predictions on holdout drugs and compute metrics.

    Returns:
        BenchmarkResult with acceptance metrics.
    """
    raise NotImplementedError
