"""Model-Informed Precision Dosing (MIPD).

Given TDM observations, recommends an adjusted dose to achieve a
target steady-state concentration.  Uses the Bayesian posterior from
``tdm.bayesian_update()`` with linear dose scaling (standard clinical
approach for linear PK drugs).

Algorithm:
    1. Run Bayesian update at the *observed* dose → posterior Cmax.
    2. Linear scaling: dose_new = dose_current × (target / posterior_Cmax).
    3. Clamp to clinical dose range and round to practical increment.
    4. Report predicted Css at recommended dose (with uncertainty from
       posterior CV).

For nonlinear PK (saturable metabolism), linear scaling is an
approximation.  A future extension could add grid-search verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sisyphus.core import Distribution
from sisyphus.engine.compiler import CompiledODE
from sisyphus.graph.body import BodyGraph
from sisyphus.core import DrugOnGraph
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update
from sisyphus.regimen.tdm_enkf import EnKFResult, enkf_update
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default dose rounding increment (mg)
DEFAULT_ROUND_INCREMENT: float = 25.0

# Minimum practical dose (mg)
DEFAULT_DOSE_MIN: float = 25.0

# Maximum practical dose (mg)
DEFAULT_DOSE_MAX: float = 5000.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseRecommendation:
    """Result of MIPD dose optimization.

    Attributes:
        current_dose_mg: The dose at which observations were collected.
        recommended_dose_mg: Adjusted dose to achieve target_css.
        target_css: Target steady-state Cmax (mg/L).
        posterior_css_current: Bayesian posterior Css at current dose (mg/L).
        predicted_css_recommended: Predicted Css at recommended dose (mg/L).
        posterior_cv: Posterior CV of Cmax prediction.
        scaling_ratio: recommended_dose / current_dose.
        tdm_result: Full TDMResult or EnKFResult from the Bayesian update.
        method: Scaling method used.
    """

    current_dose_mg: float
    recommended_dose_mg: float
    target_css: float
    posterior_css_current: float
    predicted_css_recommended: float
    posterior_cv: float
    scaling_ratio: float
    tdm_result: TDMResult | EnKFResult
    method: str = "linear_scaling"


# ---------------------------------------------------------------------------
# Dose rounding
# ---------------------------------------------------------------------------


def _round_dose(
    dose_mg: float,
    increment: float = DEFAULT_ROUND_INCREMENT,
    dose_min: float = DEFAULT_DOSE_MIN,
    dose_max: float = DEFAULT_DOSE_MAX,
) -> float:
    """Round dose to nearest practical increment and clamp to range.

    Args:
        dose_mg: Raw calculated dose (mg).
        increment: Rounding increment (mg).
        dose_min: Minimum allowed dose (mg).
        dose_max: Maximum allowed dose (mg).

    Returns:
        Rounded and clamped dose (mg).
    """
    rounded = round(dose_mg / increment) * increment
    return float(np.clip(rounded, dose_min, dose_max))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_dose(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: list[Observation] | tuple[Observation, ...],
    target_css: float,
    dose_range: tuple[float, float] = (DEFAULT_DOSE_MIN, DEFAULT_DOSE_MAX),
    round_increment: float = DEFAULT_ROUND_INCREMENT,
    n_prior: int = 2000,
    seed: int = 42,
    method: str = "is",
) -> DoseRecommendation:
    """Recommend an adjusted dose to achieve a target concentration.

    Runs Bayesian TDM update at the current dose, then uses linear
    dose scaling to compute the dose that would produce ``target_css``
    at steady state.

    Linear scaling assumption: Css ∝ dose (valid for linear PK, which
    covers the vast majority of drugs at therapeutic concentrations).

    Args:
        compiled: Pre-compiled ODE skeleton.
        graph: BodyGraph with distributional parameters.
        drug: DrugOnGraph at the *current* (observed) dose.
        regimen: Dosing regimen the patient received.
        observations: Measured concentrations with times.
        target_css: Desired steady-state Cmax (mg/L).
        dose_range: (min, max) allowed dose in mg.
        round_increment: Round recommended dose to this increment (mg).
        n_prior: Number of prior MC samples for TDM.
        seed: RNG seed.
        method: TDM method — ``"is"`` for importance sampling,
            ``"enkf"`` for Ensemble Kalman Filter.

    Returns:
        DoseRecommendation with adjusted dose and predictions.

    Raises:
        ValueError: If target_css <= 0 or posterior is degenerate.
    """
    if target_css <= 0:
        raise ValueError(f"target_css must be positive, got {target_css}")

    # Step 1: Bayesian update at current dose
    if method == "enkf":
        tdm = enkf_update(
            compiled, graph, drug, regimen,
            observations=observations,
            n_ensemble=n_prior,
            seed=seed,
        )
    else:
        tdm = bayesian_update(
            compiled, graph, drug, regimen,
            observations=observations,
            n_prior=n_prior,
            seed=seed,
        )

    posterior_css = tdm.posterior_cmax.mean
    if posterior_css <= 0:
        raise ValueError(
            f"Posterior Cmax is non-positive ({posterior_css:.6f}). "
            "Cannot compute dose recommendation — check observations and drug model."
        )

    # Step 2: Linear dose scaling
    current_dose = drug.dose_mg
    scaling_ratio = target_css / posterior_css
    raw_dose = current_dose * scaling_ratio

    # Step 3: Clamp and round
    recommended = _round_dose(raw_dose, round_increment, dose_range[0], dose_range[1])

    # Step 4: Predict Css at recommended dose (linear scaling from posterior)
    actual_ratio = recommended / current_dose if current_dose > 0 else 1.0
    predicted_css = posterior_css * actual_ratio

    logger.info(
        "MIPD: current=%.0fmg (Css=%.4f), target=%.4f, "
        "recommended=%.0fmg (predicted Css=%.4f, ratio=%.2f)",
        current_dose, posterior_css, target_css,
        recommended, predicted_css, actual_ratio,
    )

    return DoseRecommendation(
        current_dose_mg=current_dose,
        recommended_dose_mg=recommended,
        target_css=target_css,
        posterior_css_current=posterior_css,
        predicted_css_recommended=predicted_css,
        posterior_cv=tdm.posterior_cmax.cv,
        scaling_ratio=actual_ratio,
        tdm_result=tdm,
    )
