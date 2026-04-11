"""Therapeutic Drug Monitoring — Bayesian update via importance sampling.

Given a dosing regimen and one or more observed concentrations, refines
the prior parameter distributions (from predict layer) into tighter
posteriors.  This is the mechanism that bypasses the CLint R²=0.24
ceiling — observed drug levels directly correct inaccurate priors.

Algorithm:
    1. Draw N prior samples from drug + physiology distributions.
    2. For each sample, simulate the dosing regimen via solve_regimen.
    3. Interpolate predicted concentration at each observation time.
    4. Compute likelihood: L = Π Normal(obs | pred, σ_assay).
    5. Importance weights w_i ∝ L_i. Normalize so Σw = 1.
    6. ESS = 1 / Σ(w²). If ESS < threshold, warn (may need more samples).
    7. Weighted statistics → posterior distributions.

Does NOT modify pipeline weights or model parameters (invariant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sisyphus.core import Distribution, DrugOnGraph, SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum effective sample size (fraction of N) before warning
MIN_ESS_FRACTION = 0.10

# Default assay CV for observations
DEFAULT_ASSAY_CV = 0.10


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """A single TDM observation (measured drug concentration).

    Attributes:
        time_h: Time of measurement (hours from first dose).
        concentration: Measured concentration (mg/L).
        node: Observation compartment (default: venous_blood).
        cv: Assay coefficient of variation (default: 10%).
    """
    time_h: float
    concentration: float
    node: str = "venous_blood"
    cv: float = DEFAULT_ASSAY_CV

    @property
    def sigma(self) -> float:
        """Assay standard deviation (mg/L)."""
        return max(self.concentration * self.cv, 1e-12)


@dataclass(frozen=True)
class TDMResult:
    """Result of Bayesian TDM update.

    Attributes:
        prior_cmax: Prior Cmax distribution (from population model).
        posterior_cmax: Posterior Cmax distribution (after observations).
        prior_params: Prior parameter means {name: value}.
        posterior_params: Posterior parameter means {name: value}.
        n_prior: Number of prior samples drawn.
        n_successful: Number of successful simulations.
        ess: Effective sample size after reweighting.
        observations: The observations used.
        weights: Normalized importance weights (length n_successful).
        log_likelihoods: Log-likelihoods per sample (length n_successful).
    """
    prior_cmax: Distribution
    posterior_cmax: Distribution
    prior_params: dict[str, float] = field(default_factory=dict)
    posterior_params: dict[str, float] = field(default_factory=dict)
    n_prior: int = 0
    n_successful: int = 0
    ess: float = 0.0
    observations: tuple[Observation, ...] = ()
    weights: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    log_likelihoods: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    # Optional 90 % credible interval on posterior Cmax. IBIS / EnKF / SBI
    # dispatch paths compute this directly from their weighted (or equally
    # weighted) sample arrays. The legacy importance-sampling path also
    # populates it from raw weighted quantiles on the prior Cmax samples.
    # None when not computed (older callers can still rely on the lognormal
    # approximation at `.posterior_cmax.mean`/`.cv`).
    cmax_ci_90: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def apply_ci_floor(
    result: "TDMResult", min_ci_half_width_fraction: float
) -> "TDMResult":
    """Public alias for the conformal CI floor helper.

    Useful for post-hoc rescoring of an existing TDMResult without
    re-running the full importance-sampling / IBIS / EnKF pipeline.
    """
    return _apply_ci_floor(result, min_ci_half_width_fraction)


def _apply_ci_floor(
    result: "TDMResult", min_ci_half_width_fraction: float
) -> "TDMResult":
    """Widen the cmax_ci_90 if its half-width is below a fraction of the posterior mean.

    Used as a conformal-style guard against over-tight posteriors that
    pathologically exclude truth (e.g. rivaroxaban under multi-observation
    TDM, see docs/tdm_ci_calibration.md). The new CI is centred on the
    posterior mean and has half-width = ``frac × posterior_mean``.
    """
    if min_ci_half_width_fraction <= 0.0 or result.cmax_ci_90 is None:
        return result
    post_mean = float(result.posterior_cmax.mean)
    if post_mean <= 0:
        return result
    floor = float(min_ci_half_width_fraction) * post_mean
    lo, hi = result.cmax_ci_90
    half = max(post_mean - lo, hi - post_mean)
    if half >= floor:
        return result
    new_lo = max(post_mean - floor, 0.0)
    new_hi = post_mean + floor
    from dataclasses import replace as _replace
    return _replace(result, cmax_ci_90=(new_lo, new_hi))


def _interpolate_concentration(
    sim: SimResult,
    t_obs: float,
    node: str,
) -> float:
    """Linearly interpolate concentration at a specific time point.

    Args:
        sim: SimResult from solve_regimen.
        t_obs: Observation time (hours).
        node: Observation node name.

    Returns:
        Interpolated concentration (mg/L). 0.0 if out of range.
    """
    conc = sim.concentrations.get(node)
    if conc is None:
        return 0.0
    return float(np.interp(t_obs, sim.time_h, conc))


def _log_likelihood(
    sim: SimResult,
    observations: tuple[Observation, ...],
) -> float:
    """Compute log-likelihood of observations given a simulation.

    L = Σ log N(obs_i | pred_i, σ_i)
      = Σ [-0.5 * ((obs - pred) / σ)² - log(σ) - 0.5*log(2π)]

    Args:
        sim: SimResult from a single MC realization.
        observations: Tuple of TDM observations.

    Returns:
        Total log-likelihood (sum over all observations).
    """
    total = 0.0
    for obs in observations:
        pred = _interpolate_concentration(sim, obs.time_h, obs.node)
        sigma = obs.sigma
        residual = (obs.concentration - pred) / sigma
        total += -0.5 * residual * residual - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return total


def _extract_params(drug: DrugOnGraph) -> dict[str, float]:
    """Extract key parameter means from a DrugOnGraph."""
    params = {
        "fup": drug.fup.mean,
        "peff": drug.peff.mean,
    }
    # Total enzyme affinity (sum across all enzymes)
    if drug.enzyme_affinity:
        total_aff = sum(d.mean for d in drug.enzyme_affinity.values())
        params["total_enzyme_affinity"] = total_aff
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bayesian_update(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: tuple[Observation, ...] | list[Observation],
    n_prior: int = 2000,
    seed: int = 42,
    observation_node: str = "venous_blood",
    method: str = "importance_sampling",
    sbi_posterior_path: "str | None" = None,
    sbi_fallback: bool = True,
    sbi_use_surrogate: bool = False,
    sbi_surrogate_std_threshold: float = 0.02,
    logp_hint: float | None = None,
    min_ci_half_width_fraction: float = 0.0,
) -> TDMResult:
    """Run Bayesian TDM update.

    Args:
        method: ``"importance_sampling"`` (default, original), ``"ibis"``
            (sequential MC with resampling + MCMC rejuvenation), ``"enkf"``
            (Ensemble Kalman Filter), or ``"sbi"`` (amortized simulation-
            based inference via the Track A multi-drug conditional NSF).
        sbi_posterior_path: Path to the trained amortized posterior. If
            None, defaults to ``models/sbi/multi_drug_nsf.pt``.
        sbi_fallback: If ``True`` (default) and the SBI posterior file is
            missing, fall back to ``ibis`` rather than raising. Set to
            ``False`` to fail loudly in CI or production configs that
            require SBI to be available.
        logp_hint: Optional logP injection for SBI drug feature extraction.
            Unused by other methods.
        min_ci_half_width_fraction: Optional empirical-conformal floor on
            the posterior 90 % credible-interval half-width, expressed as
            a fraction of the posterior mean. The Track D2 motivating
            case was rivaroxaban under multi-observation TDM, where the
            posterior collapses to ~5 % CV and the empirical CI is too
            narrow to cover the true Cmax even when the posterior mean
            is close to truth. Setting this to e.g. ``0.20`` widens any
            CI half-width below ``0.20 × posterior_mean`` to that floor,
            recovering coverage on over-tightened posteriors without
            affecting well-calibrated runs. Default ``0.0`` (no floor)
            preserves backwards compatibility.

    For ``"ibis"``, ``"enkf"``, and ``"sbi"`` see
    :mod:`sisyphus.regimen.tdm_ibis`, :mod:`sisyphus.regimen.tdm_enkf`,
    and :mod:`sisyphus.regimen.tdm_sbi` respectively.

    Original docstring follows.

    Run Bayesian TDM update via importance sampling.

    Draws ``n_prior`` samples from the joint prior (drug + physiology
    distributions), simulates each, and reweights by the likelihood of
    the observed concentrations.

    Args:
        compiled: Pre-compiled ODE skeleton.
        graph: BodyGraph with distributional parameters.
        drug: DrugOnGraph with distributional parameters (the prior).
        regimen: Dosing regimen the patient received.
        observations: Observed concentrations with times and assay CV.
        n_prior: Number of prior MC samples.
        seed: Base RNG seed.
        observation_node: Default node for observations without explicit node.

    Returns:
        TDMResult with prior/posterior distributions, ESS, weights.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("At least one observation is required")

    def _finalize(res: "TDMResult") -> "TDMResult":
        return _apply_ci_floor(res, min_ci_half_width_fraction)

    # ── Dispatch to alternative methods ──
    if method == "ibis":
        from sisyphus.regimen.tdm_ibis import ibis_update, IBISResult

        ibis_result = ibis_update(
            compiled, graph, drug, regimen, observations,
            n_particles=n_prior, seed=seed,
            observation_node=observation_node,
        )
        # Convert IBISResult → TDMResult for API compatibility
        return _finalize(TDMResult(
            prior_cmax=ibis_result.prior_cmax,
            posterior_cmax=ibis_result.posterior_cmax,
            prior_params=ibis_result.prior_params,
            posterior_params=ibis_result.posterior_params,
            n_prior=ibis_result.n_particles,
            n_successful=ibis_result.n_successful,
            ess=ibis_result.ess_history[-1] if ibis_result.ess_history else 0.0,
            observations=ibis_result.observations,
            weights=ibis_result.weights,
            log_likelihoods=np.log(np.maximum(ibis_result.weights, 1e-300)),
            cmax_ci_90=tuple(ibis_result.cmax_ci_90),
        ))

    if method == "enkf":
        from sisyphus.regimen.tdm_enkf import enkf_update

        enkf_result = enkf_update(
            compiled, graph, drug, regimen, observations,
            n_ensemble=n_prior, seed=seed,
            observation_node=observation_node,
        )
        n_post = max(enkf_result.n_successful_post, 1)
        uniform_weights = np.full(n_post, 1.0 / n_post)
        return _finalize(TDMResult(
            prior_cmax=enkf_result.prior_cmax,
            posterior_cmax=enkf_result.posterior_cmax,
            prior_params=enkf_result.prior_params,
            posterior_params=enkf_result.posterior_params,
            n_prior=enkf_result.n_prior,
            n_successful=enkf_result.n_successful,
            ess=enkf_result.ess,
            observations=enkf_result.observations,
            weights=uniform_weights,
            log_likelihoods=np.zeros(n_post),
            cmax_ci_90=tuple(enkf_result.cmax_ci_90),
        ))

    if method == "sbi":
        from sisyphus.regimen.tdm_sbi import sbi_update

        try:
            return _finalize(sbi_update(
                compiled, graph, drug, regimen, observations,
                posterior_path=sbi_posterior_path,
                n_samples=max(n_prior, 1000),  # amortized sampling is cheap
                n_predictive_sims=min(n_prior, 200),  # bound the forward cost
                seed=seed,
                observation_node=observation_node,
                logp_hint=logp_hint,
                use_surrogate=sbi_use_surrogate,
                surrogate_ensemble_std_threshold=sbi_surrogate_std_threshold,
            ))
        except FileNotFoundError as exc:
            if not sbi_fallback:
                raise
            logger.warning(
                "SBI posterior unavailable (%s), falling back to IBIS",
                exc,
            )
            return bayesian_update(
                compiled, graph, drug, regimen, observations,
                n_prior=n_prior, seed=seed,
                observation_node=observation_node,
                method="ibis",
                min_ci_half_width_fraction=min_ci_half_width_fraction,
            )

    # ── Default: importance sampling (original method) ──

    # Determine simulation window: cover all observation times + one interval
    max_obs_t = max(obs.time_h for obs in observations)
    t_total = max(max_obs_t + 24.0, regimen.last_dose_time_h + 24.0)

    # Collect prior samples and simulation results
    cmax_samples: list[float] = []
    param_samples: list[dict[str, float]] = []
    log_liks: list[float] = []
    n_failures = 0

    for i in range(n_prior):
        rng = np.random.default_rng(seed + i)

        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        try:
            sim = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.25)
            if not sim.solver_success:
                n_failures += 1
                continue
        except Exception:
            n_failures += 1
            continue

        # Extract Cmax from venous blood
        conc = sim.concentrations.get(observation_node)
        if conc is None or len(conc) == 0:
            n_failures += 1
            continue
        cmax = float(np.max(conc))
        if cmax <= 0:
            n_failures += 1
            continue

        # Compute log-likelihood
        ll = _log_likelihood(sim, observations)

        cmax_samples.append(cmax)
        param_samples.append(_extract_params(realized_drug))
        log_liks.append(ll)

    n_ok = len(cmax_samples)
    if n_ok == 0:
        logger.warning("TDM: all %d prior samples failed", n_prior)
        return _finalize(TDMResult(
            prior_cmax=Distribution(0.0),
            posterior_cmax=Distribution(0.0),
            n_prior=n_prior,
            n_successful=0,
            ess=0.0,
            observations=observations,
        ))

    cmax_arr = np.array(cmax_samples)
    ll_arr = np.array(log_liks)

    # Importance weights (log-space for numerical stability)
    ll_max = np.max(ll_arr)
    log_weights = ll_arr - ll_max
    weights = np.exp(log_weights)
    weights /= np.sum(weights)

    # Effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    ess_frac = ess / n_ok

    if ess_frac < MIN_ESS_FRACTION:
        logger.warning(
            "TDM: low ESS = %.1f (%.1f%% of %d). Consider increasing n_prior.",
            ess, 100 * ess_frac, n_ok,
        )

    # Prior statistics
    prior_cmax_mean = float(np.mean(cmax_arr))
    prior_cmax_cv = float(np.std(cmax_arr) / prior_cmax_mean) if prior_cmax_mean > 0 else 0.0

    # Posterior statistics (weighted)
    post_cmax_mean = float(np.average(cmax_arr, weights=weights))
    # Weighted variance: Var = Σw(x - μ)² / (1 - Σw²) — but with normalized weights
    post_cmax_var = float(np.average((cmax_arr - post_cmax_mean) ** 2, weights=weights))
    post_cmax_std = np.sqrt(post_cmax_var)
    post_cmax_cv = float(post_cmax_std / post_cmax_mean) if post_cmax_mean > 0 else 0.0

    # 90 % credible interval via weighted quantiles on the raw Cmax samples
    # (not a lognormal-from-CV approximation — the latter can under-cover when
    # the posterior is asymmetric or when ESS is small relative to n_ok).
    sorted_idx = np.argsort(cmax_arr)
    sorted_cmax = cmax_arr[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    lo_idx = int(np.searchsorted(cum_weights, 0.05))
    hi_idx = int(np.searchsorted(cum_weights, 0.95))
    lo_idx = min(max(lo_idx, 0), len(sorted_cmax) - 1)
    hi_idx = min(max(hi_idx, 0), len(sorted_cmax) - 1)
    ci90 = (float(sorted_cmax[lo_idx]), float(sorted_cmax[hi_idx]))

    # Prior/posterior for tracked parameters
    prior_params: dict[str, float] = {}
    posterior_params: dict[str, float] = {}
    if param_samples:
        param_keys = param_samples[0].keys()
        for key in param_keys:
            vals = np.array([p[key] for p in param_samples])
            prior_params[key] = float(np.mean(vals))
            posterior_params[key] = float(np.average(vals, weights=weights))

    logger.info(
        "TDM update: %d/%d successful, ESS=%.1f (%.1f%%), "
        "Cmax prior=%.4f (CV=%.0f%%) → posterior=%.4f (CV=%.0f%%)",
        n_ok, n_prior, ess, 100 * ess_frac,
        prior_cmax_mean, 100 * prior_cmax_cv,
        post_cmax_mean, 100 * post_cmax_cv,
    )

    return _finalize(TDMResult(
        prior_cmax=Distribution(prior_cmax_mean, cv=prior_cmax_cv),
        posterior_cmax=Distribution(post_cmax_mean, cv=post_cmax_cv),
        prior_params=prior_params,
        posterior_params=posterior_params,
        n_prior=n_prior,
        n_successful=n_ok,
        ess=ess,
        observations=observations,
        weights=weights,
        log_likelihoods=ll_arr,
        cmax_ci_90=ci90,
    ))
