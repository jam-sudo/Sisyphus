"""Holdout benchmark runner.

Runs predictions on the holdout set and computes acceptance metrics.
Enforces the invariant that holdout drugs are never used for training.
Reports both full-holdout and in-domain AAFE separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sisyphus.validation.metrics import aafe, pct_within_n_fold, pi_coverage
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
_KNOWN_ER_FORMULATIONS: frozenset[str] = frozenset(
    {
        "methylphenidate",
        "oxybutynin",
    }
)

# ---------------------------------------------------------------------------
# Reference data quality exclusions (data audit 2026-04-07).
# These drugs have reference Cmax values that are not a fair comparison
# for single-drug PBPK prediction.  Each has a documented scientific
# rationale — these are NOT post-hoc removals of poorly-predicted drugs.
#
# lopinavir:  Reference Cmax (9.8 mg/L) is from Kaletra (lopinavir/ritonavir).
#             Ritonavir is a strong CYP3A4 inhibitor that boosts lopinavir
#             exposure ~25x.  Sisyphus predicts single-drug PK without DDI.
#             Comparing DDI-boosted Cmax to unboosted prediction is unfair.
#             Source: FDA label (Kaletra), NDA 021226.
#
# prasugrel:  Reference Cmax (0.631 mg/L = 631 ng/mL) is for the active
#             metabolite R-138727, not the parent drug.  Parent prasugrel
#             Cmax is ~1-2 ng/mL.  Sisyphus predicts parent compound PK.
#             Additionally from Chinese population study (Liang et al. 2014).
#             Source: Liang et al., BMC Pharmacol Toxicol 2014.
# ---------------------------------------------------------------------------
_REFERENCE_QUALITY_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "lopinavir",   # DDI-boosted reference (ritonavir co-admin)
        "prasugrel",   # Active metabolite Cmax, not parent drug
    }
)


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
        pi_coverage_90: Parameter-uncertainty interval coverage, 0-1 fraction
            of drugs whose observed Cmax falls inside the predicted 90% PI.
            ``None`` unless ``compute_pi=True`` was passed to
            :func:`run_benchmark`.  NOTE: this is propagated parameter
            uncertainty only; residual structural error is not calibrated,
            so this is not an empirically-calibrated predictive interval.
        n_pi_computed: Count of drugs that contributed a non-None
            ``cmax_90ci`` when ``compute_pi=True``.  ``None`` otherwise.
    """

    n_drugs: int
    aafe: float
    pct_2fold: float
    n_in_domain: int
    aafe_in_domain: float
    pct_2fold_in_domain: float
    excluded_drugs: list[tuple[str, str]] = field(default_factory=list)
    pi_coverage_90: float | None = None
    n_pi_computed: int | None = None
    aafe_gold: float | None = None
    aafe_silver: float | None = None
    n_gold: int = 0
    n_silver: int = 0
    # Engine-only and ML-only AAFE (for ablation diagnostics)
    engine_aafe: float | None = None
    engine_pct_2fold: float | None = None
    ml_aafe: float | None = None
    ml_pct_2fold: float | None = None
    n_engine: int = 0
    n_ml: int = 0
    # CL/F track diagnostics
    clf_aafe: float | None = None
    clf_pct_2fold: float | None = None
    n_clf: int = 0


def run_benchmark(
    holdout_only: bool = True,
    max_drugs: int | None = None,
    compute_pi: bool = False,
    n_mc_samples: int = 1000,
) -> BenchmarkResult:
    """Run predictions on reference drugs and compute acceptance metrics.

    Reports both full-holdout AAFE and in-domain AAFE.  Drugs are excluded
    from in-domain metrics when:
    - Prediction has AD flags (PRODRUG, HIGH_MW, EXTREME_LIPOPHILIC, etc.)
    - Drug is a known ER/MR formulation (reference data mismatch)

    Args:
        holdout_only: If True, restrict to holdout drugs only.
        max_drugs: Maximum number of drugs to evaluate (for quick testing).
        compute_pi: If True, enables Monte Carlo uncertainty propagation per
            drug and computes empirical 90% PI coverage.  Default False keeps
            the benchmark fast (no MC); in that case ``pi_coverage_90`` and
            ``n_pi_computed`` remain ``None``.  The resulting interval
            reflects propagated parameter uncertainty only — not an
            empirically-calibrated predictive interval.
        n_mc_samples: Number of Monte Carlo samples per drug when
            ``compute_pi=True`` (default 1000).  Ignored otherwise.

    Returns:
        BenchmarkResult with full and in-domain AAFE, %2-fold, and
        optionally PI coverage.
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
    gold_predicted: list[float] = []
    gold_observed: list[float] = []
    silver_predicted: list[float] = []
    silver_observed: list[float] = []
    # Engine-only, ML-only, CL/F-only (for ablation diagnostics)
    engine_predicted: list[float] = []
    engine_observed: list[float] = []
    ml_predicted: list[float] = []
    ml_observed: list[float] = []
    clf_predicted: list[float] = []
    clf_observed: list[float] = []
    # PI coverage (compute_pi=True only): paired (obs, lower, upper) triplets
    pi_obs: list[float] = []
    pi_lower: list[float] = []
    pi_upper: list[float] = []

    excluded: list[tuple[str, str]] = []
    skipped = 0

    predict_kwargs = {"n_mc_samples": n_mc_samples} if compute_pi else {}

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route, **predict_kwargs)
            cmax_pred = result.pk.cmax.mean
            if cmax_pred <= 0:
                skipped += 1
                logger.warning("[%d/%d] %s: zero Cmax predicted", i + 1, len(refs), ref.name)
                continue

            all_predicted.append(cmax_pred)
            all_observed.append(ref.cmax_obs)

            # PI coverage collection (only when compute_pi=True and MC succeeded)
            if compute_pi and result.cmax_90ci is not None:
                lo, hi = result.cmax_90ci
                # Skip degenerate intervals (e.g. MC fell back to zero tuple on
                # upstream failure); only paired non-collapsed bounds count.
                if hi > lo:
                    pi_obs.append(ref.cmax_obs)
                    pi_lower.append(lo)
                    pi_upper.append(hi)

            is_gold = any("drugbank:" in w for w in result.warnings)
            if is_gold:
                gold_predicted.append(cmax_pred)
                gold_observed.append(ref.cmax_obs)
            else:
                silver_predicted.append(cmax_pred)
                silver_observed.append(ref.cmax_obs)

            # Engine-only and ML-only Cmax
            if result.engine_pk and result.engine_pk.cmax.mean > 0:
                engine_predicted.append(result.engine_pk.cmax.mean)
                engine_observed.append(ref.cmax_obs)
            if result.ml_pk and result.ml_pk.cmax.mean > 0:
                ml_predicted.append(result.ml_pk.cmax.mean)
                ml_observed.append(ref.cmax_obs)
            if result.clf_pk and result.clf_pk.cmax.mean > 0:
                clf_predicted.append(result.clf_pk.cmax.mean)
                clf_observed.append(ref.cmax_obs)

            fold = cmax_pred / ref.cmax_obs if ref.cmax_obs > 0 else float("inf")

            # Determine if drug is in-domain
            exclude_reason = ""
            if result.ad_flags:
                exclude_reason = "AD:" + ",".join(result.ad_flags)
            elif ref.name in _KNOWN_ER_FORMULATIONS:
                exclude_reason = "ER_FORMULATION"
            elif ref.name in _REFERENCE_QUALITY_EXCLUSIONS:
                exclude_reason = "REF_QUALITY"

            if exclude_reason:
                excluded.append((ref.name, exclude_reason))
                logger.info(
                    "[%d/%d] %s: pred=%.4f obs=%.4f fold=%.2f [EXCLUDED: %s]",
                    i + 1,
                    len(refs),
                    ref.name,
                    cmax_pred,
                    ref.cmax_obs,
                    fold,
                    exclude_reason,
                )
            else:
                id_predicted.append(cmax_pred)
                id_observed.append(ref.cmax_obs)
                logger.info(
                    "[%d/%d] %s: pred=%.4f obs=%.4f fold=%.2f",
                    i + 1,
                    len(refs),
                    ref.name,
                    cmax_pred,
                    ref.cmax_obs,
                    fold,
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
        len(all_predicted),
        result_aafe,
        result_pct2,
        skipped,
    )
    logger.info(
        "In-domain: %d drugs, AAFE=%.3f, %%2-fold=%.1f%% (%d excluded: %s)",
        len(id_predicted),
        id_aafe,
        id_pct2,
        len(excluded),
        ", ".join(f"{n} ({r})" for n, r in excluded),
    )

    gold_p = np.array(gold_predicted)
    gold_o = np.array(gold_observed)
    silver_p = np.array(silver_predicted)
    silver_o = np.array(silver_observed)
    aafe_gold_val = aafe(gold_p, gold_o) if len(gold_p) >= 3 else None
    aafe_silver_val = aafe(silver_p, silver_o) if len(silver_p) >= 3 else None

    # Engine-only, ML-only, and CL/F-only metrics
    eng_p = np.array(engine_predicted)
    eng_o = np.array(engine_observed)
    ml_p = np.array(ml_predicted)
    ml_o = np.array(ml_observed)
    clf_p = np.array(clf_predicted)
    clf_o = np.array(clf_observed)
    engine_aafe_val = aafe(eng_p, eng_o) if len(eng_p) >= 3 else None
    engine_pct2_val = pct_within_n_fold(eng_p, eng_o) if len(eng_p) >= 3 else None
    ml_aafe_val = aafe(ml_p, ml_o) if len(ml_p) >= 3 else None
    ml_pct2_val = pct_within_n_fold(ml_p, ml_o) if len(ml_p) >= 3 else None
    clf_aafe_val = aafe(clf_p, clf_o) if len(clf_p) >= 3 else None
    clf_pct2_val = pct_within_n_fold(clf_p, clf_o) if len(clf_p) >= 3 else None

    logger.info(
        "Engine-only: %d drugs, AAFE=%.3f | ML-only: %d drugs, AAFE=%.3f | CL/F-only: %d drugs, AAFE=%.3f",
        len(engine_predicted),
        engine_aafe_val or float("inf"),
        len(ml_predicted),
        ml_aafe_val or float("inf"),
        len(clf_predicted),
        clf_aafe_val or float("inf"),
    )

    # PI coverage (empty unless compute_pi=True)
    pi_coverage_val: float | None = None
    pi_n: int | None = None
    if compute_pi:
        pi_n = len(pi_obs)
        if pi_n > 0:
            pi_coverage_val = pi_coverage(
                np.array(pi_obs), np.array(pi_lower), np.array(pi_upper)
            )
            logger.info(
                "90%% PI coverage: %.1f%% over %d drugs "
                "(parameter uncertainty only; not empirically-calibrated "
                "predictive interval — residual structural error not added)",
                pi_coverage_val * 100,
                pi_n,
            )
        else:
            logger.warning(
                "compute_pi=True but no drug produced a non-degenerate "
                "Cmax 90%% PI; coverage not computed"
            )

    return BenchmarkResult(
        n_drugs=len(all_predicted),
        aafe=result_aafe,
        pct_2fold=result_pct2,
        n_in_domain=len(id_predicted),
        aafe_in_domain=id_aafe,
        pct_2fold_in_domain=id_pct2,
        excluded_drugs=excluded,
        pi_coverage_90=pi_coverage_val,
        n_pi_computed=pi_n,
        aafe_gold=aafe_gold_val,
        aafe_silver=aafe_silver_val,
        n_gold=len(gold_predicted),
        n_silver=len(silver_predicted),
        engine_aafe=engine_aafe_val,
        engine_pct_2fold=engine_pct2_val,
        ml_aafe=ml_aafe_val,
        ml_pct_2fold=ml_pct2_val,
        n_engine=len(engine_predicted),
        n_ml=len(ml_predicted),
        clf_aafe=clf_aafe_val,
        clf_pct_2fold=clf_pct2_val,
        n_clf=len(clf_predicted),
    )
