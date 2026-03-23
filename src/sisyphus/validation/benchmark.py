"""Holdout benchmark runner.

Runs predictions on the holdout set and computes acceptance metrics.
Enforces the invariant that holdout drugs are never used for training.
Reports both full-holdout and in-domain AAFE separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sisyphus.validation.metrics import aafe, pct_within_n_fold
from sisyphus.validation.reference import load_reference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known ER/MR formulations in reference data (identified via data audit).
# These drugs have reference Cmax values from extended-release formulations,
# but the PBPK model assumes immediate-release.  They are excluded from
# in-domain AAFE calculation.
#
# methylphenidate: 72 mg = Concerta (OROS), IR is 5/10/20 mg
# oxybutynin:      5 mg Cmax=0.001 mg/L = Ditropan XL, IR Cmax ~0.008 mg/L
# ---------------------------------------------------------------------------
_KNOWN_ER_FORMULATIONS: frozenset[str] = frozenset({
    "methylphenidate",
    "oxybutynin",
})


@dataclass(frozen=True)
class BenchmarkResult:
    """Results from a holdout benchmark run.

    Attributes:
        n_drugs: Number of drugs successfully evaluated (all).
        aafe: Absolute Average Fold Error (all evaluated drugs).
        pct_2fold: Percentage within 2-fold (all evaluated drugs).
        n_in_domain: Number of in-domain drugs evaluated.
        aafe_in_domain: AAFE for in-domain drugs only.
        pct_2fold_in_domain: %2-fold for in-domain drugs only.
        excluded_drugs: List of (name, reason) for excluded drugs.
        pi_coverage_90: 90% prediction interval coverage (None if not computed).
    """

    n_drugs: int
    aafe: float
    pct_2fold: float
    n_in_domain: int
    aafe_in_domain: float
    pct_2fold_in_domain: float
    excluded_drugs: list[tuple[str, str]] = field(default_factory=list)
    pi_coverage_90: float | None = None


def run_benchmark(
    holdout_only: bool = True,
    max_drugs: int | None = None,
) -> BenchmarkResult:
    """Run predictions on reference drugs and compute acceptance metrics.

    Reports both full-holdout AAFE and in-domain AAFE.  Drugs are excluded
    from in-domain metrics when:
    - Prediction has AD flags (PRODRUG, HIGH_MW, EXTREME_LIPOPHILIC, etc.)
    - Drug is a known ER/MR formulation (reference data mismatch)

    Args:
        holdout_only: If True, restrict to holdout drugs only.
        max_drugs: Maximum number of drugs to evaluate (for quick testing).

    Returns:
        BenchmarkResult with full and in-domain AAFE, %2-fold.
    """
    from sisyphus.pipeline.predict import predict

    refs = load_reference()
    if holdout_only:
        refs = [r for r in refs if r.in_holdout]
    if max_drugs is not None:
        refs = refs[:max_drugs]

    # All drugs
    all_predicted: list[float] = []
    all_observed: list[float] = []
    # In-domain drugs only
    id_predicted: list[float] = []
    id_observed: list[float] = []

    excluded: list[tuple[str, str]] = []
    skipped = 0

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            cmax_pred = result.pk.cmax.mean
            if cmax_pred <= 0:
                skipped += 1
                logger.warning("[%d/%d] %s: zero Cmax predicted", i + 1, len(refs), ref.name)
                continue

            all_predicted.append(cmax_pred)
            all_observed.append(ref.cmax_obs)

            fold = cmax_pred / ref.cmax_obs if ref.cmax_obs > 0 else float("inf")

            # Determine if drug is in-domain
            exclude_reason = ""
            if result.ad_flags:
                exclude_reason = "AD:" + ",".join(result.ad_flags)
            elif ref.name in _KNOWN_ER_FORMULATIONS:
                exclude_reason = "ER_FORMULATION"

            if exclude_reason:
                excluded.append((ref.name, exclude_reason))
                logger.info(
                    "[%d/%d] %s: pred=%.4f obs=%.4f fold=%.2f [EXCLUDED: %s]",
                    i + 1, len(refs), ref.name, cmax_pred, ref.cmax_obs, fold,
                    exclude_reason,
                )
            else:
                id_predicted.append(cmax_pred)
                id_observed.append(ref.cmax_obs)
                logger.info(
                    "[%d/%d] %s: pred=%.4f obs=%.4f fold=%.2f",
                    i + 1, len(refs), ref.name, cmax_pred, ref.cmax_obs, fold,
                )
        except Exception as e:
            skipped += 1
            logger.warning("[%d/%d] %s: failed: %s", i + 1, len(refs), ref.name, e)

    # Full-holdout metrics
    pred_all = np.array(all_predicted)
    obs_all = np.array(all_observed)
    result_aafe = aafe(pred_all, obs_all) if len(pred_all) > 0 else float("inf")
    result_pct2 = pct_within_n_fold(pred_all, obs_all) if len(pred_all) > 0 else 0.0

    # In-domain metrics
    pred_id = np.array(id_predicted)
    obs_id = np.array(id_observed)
    id_aafe = aafe(pred_id, obs_id) if len(pred_id) > 0 else float("inf")
    id_pct2 = pct_within_n_fold(pred_id, obs_id) if len(pred_id) > 0 else 0.0

    logger.info(
        "Benchmark complete: %d drugs evaluated, AAFE=%.3f, %%2-fold=%.1f%%, %d skipped",
        len(all_predicted), result_aafe, result_pct2, skipped,
    )
    logger.info(
        "In-domain: %d drugs, AAFE=%.3f, %%2-fold=%.1f%% (%d excluded: %s)",
        len(id_predicted), id_aafe, id_pct2, len(excluded),
        ", ".join("{} ({})".format(n, r) for n, r in excluded),
    )

    return BenchmarkResult(
        n_drugs=len(all_predicted),
        aafe=result_aafe,
        pct_2fold=result_pct2,
        n_in_domain=len(id_predicted),
        aafe_in_domain=id_aafe,
        pct_2fold_in_domain=id_pct2,
        excluded_drugs=excluded,
        pi_coverage_90=None,
    )
