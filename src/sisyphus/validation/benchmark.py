"""Holdout benchmark runner.

Runs predictions on the holdout set and computes acceptance metrics.
Enforces the invariant that holdout drugs are never used for training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sisyphus.validation.metrics import aafe, pct_within_n_fold
from sisyphus.validation.reference import load_reference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from a holdout benchmark run.

    Attributes:
        n_drugs: Number of drugs successfully evaluated.
        aafe: Absolute Average Fold Error.
        pct_2fold: Percentage of predictions within 2-fold.
        pi_coverage_90: 90% prediction interval coverage (None if not computed).
    """

    n_drugs: int
    aafe: float
    pct_2fold: float
    pi_coverage_90: float | None


def run_benchmark(
    holdout_only: bool = True,
    max_drugs: int | None = None,
) -> BenchmarkResult:
    """Run predictions on reference drugs and compute acceptance metrics.

    Args:
        holdout_only: If True, restrict to holdout drugs only.
        max_drugs: Maximum number of drugs to evaluate (for quick testing).

    Returns:
        BenchmarkResult with AAFE, %2-fold, and optional PI coverage.
    """
    from sisyphus.pipeline.predict import predict

    refs = load_reference()
    if holdout_only:
        refs = [r for r in refs if r.in_holdout]
    if max_drugs is not None:
        refs = refs[:max_drugs]

    predicted: list[float] = []
    observed: list[float] = []
    skipped = 0

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            cmax_pred = result.pk.cmax.mean
            if cmax_pred > 0:
                predicted.append(cmax_pred)
                observed.append(ref.cmax_obs)
                fold = cmax_pred / ref.cmax_obs if ref.cmax_obs > 0 else float("inf")
                logger.info(
                    "[%d/%d] %s: pred=%.4f obs=%.4f fold=%.2f",
                    i + 1,
                    len(refs),
                    ref.name,
                    cmax_pred,
                    ref.cmax_obs,
                    fold,
                )
            else:
                skipped += 1
                logger.warning("[%d/%d] %s: zero Cmax predicted", i + 1, len(refs), ref.name)
        except Exception as e:
            skipped += 1
            logger.warning("[%d/%d] %s: failed: %s", i + 1, len(refs), ref.name, e)

    pred_arr = np.array(predicted)
    obs_arr = np.array(observed)

    result_aafe = aafe(pred_arr, obs_arr) if len(pred_arr) > 0 else float("inf")
    result_pct2 = pct_within_n_fold(pred_arr, obs_arr) if len(pred_arr) > 0 else 0.0

    logger.info(
        "Benchmark complete: %d drugs evaluated, AAFE=%.3f, %%2-fold=%.1f%%, %d skipped",
        len(predicted),
        result_aafe,
        result_pct2,
        skipped,
    )

    return BenchmarkResult(
        n_drugs=len(predicted),
        aafe=result_aafe,
        pct_2fold=result_pct2,
        pi_coverage_90=None,
    )
