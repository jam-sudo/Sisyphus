"""Therapeutic Drug Monitoring — Bayesian update via Ensemble Kalman Filter.

Phase 3 of the UDE roadmap (docs/breakthrough_path.md).
Replaces importance-sampling in regimen/tdm.py with EnKF to fix particle
degeneracy documented in TDM v2.1 benchmark (ketorolac ESS=2.5, rivaroxaban
ESS=1.0 across 15 runs).

Algorithm:
    1. Draw N prior parameter samples.
    2. Forward simulate each → get predicted observation + Cmax.
    3. Assemble ensemble state X (N, D) and predicted obs Y (N, M) in log-space.
    4. Compute Kalman gain K = P_xy @ (P_yy + R)^-1.
    5. Update each sample: X'_i = X_i + K @ (log_obs + ε_i - Y_i).
    6. Re-simulate with updated parameters → posterior Cmax samples.

Unlike importance sampling, EnKF shifts all ensemble members toward observations
rather than reweighting; this cannot degenerate regardless of prior/observation
mismatch. Assumes approximate log-Gaussian relationship between parameters and
log-concentrations (appropriate for well-stirred 1-compartment PK).

State vector (3D): [log(total_enzyme_affinity), log(fup), log(peff)]
- Captures CL (via enzyme affinity scaling) + protein binding + absorption rate
- Vss not updated (graph-level parameter, weak sensitivity under perturbation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.tdm import Observation, _interpolate_concentration
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)

# Observation-space floor to protect log-transform
_LOG_FLOOR = 1e-12


@dataclass(frozen=True)
class EnKFResult:
    """Result of EnKF TDM update.

    Attributes:
        prior_cmax: Prior Cmax distribution.
        posterior_cmax: Posterior Cmax distribution (after EnKF update + re-simulation).
        cmax_ci_90: 90% credible interval for posterior Cmax (empirical).
        prior_params: Prior parameter means.
        posterior_params: Posterior parameter means.
        n_prior: Number of prior samples requested.
        n_successful: Number of successful prior simulations.
        n_successful_post: Number of successful posterior simulations.
        ess: Effective sample size (EnKF preserves all — equals n_successful).
        importance_ess: ESS as would be computed by importance sampling (diagnostic).
        observations: Observations used.
        posterior_samples: Posterior Cmax samples after re-simulation.
    """
    prior_cmax: Distribution
    posterior_cmax: Distribution
    cmax_ci_90: tuple[float, float]
    prior_params: dict[str, float] = field(default_factory=dict)
    posterior_params: dict[str, float] = field(default_factory=dict)
    n_prior: int = 0
    n_successful: int = 0
    n_successful_post: int = 0
    ess: float = 0.0
    importance_ess: float = 0.0
    observations: tuple[Observation, ...] = ()
    posterior_samples: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )


# ---------------------------------------------------------------------------
# State extraction / application
# ---------------------------------------------------------------------------


def _extract_state(drug: DrugOnGraph) -> np.ndarray:
    """Map a realized DrugOnGraph → log-parameter state vector (3D).

    Returns [log(total_enzyme_affinity), log(fup), log(peff)].
    """
    total_enz = sum(d.mean for d in drug.enzyme_affinity.values())
    return np.array([
        np.log(max(total_enz, _LOG_FLOOR)),
        np.log(max(drug.fup.mean, _LOG_FLOOR)),
        np.log(max(drug.peff.mean, _LOG_FLOOR)),
    ])


def _apply_multipliers(
    drug: DrugOnGraph,
    clint_mult: float,
    fup_mult: float,
    peff_mult: float,
) -> DrugOnGraph:
    """Return a new DrugOnGraph with scalar multipliers applied to key params.

    Keeps CVs unchanged — only rescales means. Used to apply EnKF updates.
    """
    new_fup = Distribution(drug.fup.mean * fup_mult, drug.fup.cv)
    new_peff = Distribution(drug.peff.mean * peff_mult, drug.peff.cv)
    new_enz = {
        tag: Distribution(d.mean * clint_mult, d.cv)
        for tag, d in drug.enzyme_affinity.items()
    }
    return replace(drug, fup=new_fup, peff=new_peff, enzyme_affinity=new_enz)


# ---------------------------------------------------------------------------
# Forward pass: simulate prior ensemble
# ---------------------------------------------------------------------------


def _simulate_prior(
    compiled, graph, drug, regimen, observations, n_ensemble, seed, observation_node,
):
    """Simulate prior ensemble and extract state + predicted observations."""
    t_total = max(
        max(obs.time_h for obs in observations) + 24.0,
        regimen.last_dose_time_h + 24.0,
    )

    n_obs = len(observations)
    X = np.zeros((n_ensemble, 3))
    Y = np.zeros((n_ensemble, n_obs))
    cmax = np.zeros(n_ensemble)
    baseline_clint = np.zeros(n_ensemble)
    baseline_fup = np.zeros(n_ensemble)
    baseline_peff = np.zeros(n_ensemble)
    ok = np.zeros(n_ensemble, dtype=bool)

    for i in range(n_ensemble):
        rng = np.random.default_rng(seed + i)
        rg = graph.sample(rng)
        rd = drug.sample(rng)
        try:
            params = ResolvedParams(rg, rd)
            sim = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.25)
            if not sim.solver_success:
                continue
            conc = sim.concentrations.get(observation_node)
            if conc is None or len(conc) == 0:
                continue
            cm = float(np.max(conc))
            if cm <= 0:
                continue
            for j, obs in enumerate(observations):
                Y[i, j] = _interpolate_concentration(sim, obs.time_h, obs.node)
            X[i] = _extract_state(rd)
            cmax[i] = cm
            baseline_clint[i] = sum(d.mean for d in rd.enzyme_affinity.values())
            baseline_fup[i] = rd.fup.mean
            baseline_peff[i] = rd.peff.mean
            ok[i] = True
        except Exception:
            continue

    return X, Y, cmax, baseline_clint, baseline_fup, baseline_peff, ok, t_total


# ---------------------------------------------------------------------------
# EnKF analysis step
# ---------------------------------------------------------------------------


def _enkf_analysis(X, Y, log_obs, log_sigmas, seed):
    """Stochastic EnKF update: shift ensemble toward observations.

    Args:
        X: (N, D) prior ensemble of log-parameters.
        Y: (N, M) prior predicted log-observations.
        log_obs: (M,) observed log-concentrations.
        log_sigmas: (M,) observation noise SDs in log-space.
        seed: RNG seed for observation perturbation.

    Returns:
        X_post: (N, D) posterior ensemble.
    """
    n, _ = X.shape
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    dX = X - X_mean
    dY = Y - Y_mean

    # Covariances (unbiased, denom n-1)
    denom = max(n - 1, 1)
    P_xy = dX.T @ dY / denom
    P_yy = dY.T @ dY / denom
    R = np.diag(log_sigmas ** 2)

    # Kalman gain K (D, M)
    try:
        K = P_xy @ np.linalg.inv(P_yy + R)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if P_yy + R is singular
        K = P_xy @ np.linalg.pinv(P_yy + R)

    # Perturbed observations (stochastic EnKF)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, log_sigmas[None, :], size=Y.shape)
    innovation = (log_obs[None, :] + eps) - Y  # (N, M)
    X_post = X + innovation @ K.T  # (N, D)

    return X_post


# ---------------------------------------------------------------------------
# Posterior re-simulation
# ---------------------------------------------------------------------------


def _simulate_posterior(
    compiled, graph, drug, regimen, X_post, X_prior,
    baseline_clint, baseline_fup, baseline_peff, t_total, seed,
    observation_node, ok_mask,
):
    """Re-simulate each updated ensemble member and extract Cmax."""
    n = len(X_post)
    cmax_post = np.zeros(n)
    ok_post = np.zeros(n, dtype=bool)

    for i in range(n):
        if not ok_mask[i]:
            continue
        # Compute multipliers from log-space shift
        # new_param = exp(X_post) / exp(X_prior) × baseline = exp(ΔX) × baseline
        d_clint = np.exp(X_post[i, 0] - X_prior[i, 0])
        d_fup = np.exp(X_post[i, 1] - X_prior[i, 1])
        d_peff = np.exp(X_post[i, 2] - X_prior[i, 2])
        # Clip to avoid catastrophic shifts (safety net; Kalman should handle but numerics...)
        d_clint = float(np.clip(d_clint, 0.01, 100.0))
        d_fup = float(np.clip(d_fup, 0.01, 100.0))
        d_peff = float(np.clip(d_peff, 0.01, 100.0))

        rng = np.random.default_rng(seed + i)
        rg = graph.sample(rng)
        rd = drug.sample(rng)
        rd_new = _apply_multipliers(rd, d_clint, d_fup, d_peff)
        # Clamp fup to (0, 1]
        if rd_new.fup.mean > 1.0:
            rd_new = replace(rd_new, fup=Distribution(1.0, rd_new.fup.cv))
        try:
            params = ResolvedParams(rg, rd_new)
            sim = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.25)
            if not sim.solver_success:
                continue
            conc = sim.concentrations.get(observation_node)
            if conc is None or len(conc) == 0:
                continue
            cm = float(np.max(conc))
            if cm > 0:
                cmax_post[i] = cm
                ok_post[i] = True
        except Exception:
            continue

    return cmax_post, ok_post


# ---------------------------------------------------------------------------
# Importance ESS diagnostic (for comparison)
# ---------------------------------------------------------------------------


def _importance_ess(Y, log_obs, log_sigmas):
    """Compute ESS as would be reported by importance sampling."""
    n, m = Y.shape
    log_lik = -0.5 * ((Y - log_obs) / log_sigmas) ** 2
    ll = log_lik.sum(axis=1)
    ll_max = ll.max()
    w = np.exp(ll - ll_max)
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    w /= w_sum
    return float(1.0 / (w * w).sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enkf_update(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: tuple[Observation, ...] | list[Observation],
    n_ensemble: int = 2000,
    seed: int = 42,
    observation_node: str = "venous_blood",
    inflation: float = 1.4,
) -> EnKFResult:
    """Bayesian TDM update via Ensemble Kalman Filter.

    Drop-in alternative to ``regimen.tdm.bayesian_update``. Fixes particle
    degeneracy by shifting the parameter ensemble toward observations
    instead of reweighting.

    Args:
        compiled: Pre-compiled ODE skeleton.
        graph: BodyGraph with distributional parameters.
        drug: DrugOnGraph with distributional parameters (the prior).
        regimen: Dosing regimen the patient received.
        observations: Observed concentrations.
        n_ensemble: Number of ensemble members.
        seed: Base RNG seed.
        observation_node: Default node for observations.

    Returns:
        EnKFResult with prior/posterior distributions, posterior samples,
        and ESS diagnostics.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("At least one observation is required")

    # 1. Simulate prior ensemble
    X, Y, cmax, b_clint, b_fup, b_peff, ok_mask, t_total = _simulate_prior(
        compiled, graph, drug, regimen, observations, n_ensemble, seed, observation_node,
    )
    n_ok = int(ok_mask.sum())
    if n_ok < 10:
        logger.warning("EnKF: only %d/%d prior simulations succeeded", n_ok, n_ensemble)
        return EnKFResult(
            prior_cmax=Distribution(0.0),
            posterior_cmax=Distribution(0.0),
            cmax_ci_90=(0.0, 0.0),
            n_prior=n_ensemble,
            n_successful=n_ok,
            ess=float(n_ok),
            observations=observations,
        )

    # Restrict to successful samples
    X_ok = X[ok_mask]
    Y_ok = Y[ok_mask]
    cmax_ok = cmax[ok_mask]

    # 2. EnKF analysis in log-space
    log_obs = np.log(np.maximum(np.array([o.concentration for o in observations]), _LOG_FLOOR))
    # For log-normal noise with CV c, SD in log-space ≈ c (for small c).
    # Floor at 1e-6 so an Observation(cv=0) cannot produce a zero noise SD, which
    # would divide-by-zero in the ESS / EnKF Kalman gain (R = diag(sigma^2)).
    log_sigmas = np.maximum(np.array([o.cv for o in observations]), 1e-6)
    log_Y = np.log(np.maximum(Y_ok, _LOG_FLOOR))
    importance_ess = _importance_ess(log_Y, log_obs, log_sigmas)
    X_post = _enkf_analysis(X_ok, log_Y, log_obs, log_sigmas, seed + 100_000)

    # Covariance inflation (addresses ensemble collapse / filter divergence).
    # Standard weather-EnKF practice: multiply deviations from mean by inflation.
    if inflation != 1.0:
        X_post_mean = X_post.mean(axis=0)
        X_post = X_post_mean + inflation * (X_post - X_post_mean)

    # 3. Re-simulate posterior
    # Need full-size arrays aligned with ok_mask=True indices
    full_ok = np.where(ok_mask)[0]
    X_post_full = X.copy()
    X_post_full[full_ok] = X_post
    cmax_post_full, ok_post = _simulate_posterior(
        compiled, graph, drug, regimen, X_post_full, X,
        b_clint, b_fup, b_peff, t_total, seed, observation_node, ok_mask,
    )
    cmax_post = cmax_post_full[ok_post]
    n_post = len(cmax_post)
    if n_post < 10:
        logger.warning("EnKF: only %d posterior simulations succeeded", n_post)
        return EnKFResult(
            prior_cmax=Distribution(float(cmax_ok.mean()), float(cmax_ok.std()/cmax_ok.mean())),
            posterior_cmax=Distribution(0.0),
            cmax_ci_90=(0.0, 0.0),
            n_prior=n_ensemble,
            n_successful=n_ok,
            n_successful_post=n_post,
            ess=float(n_post),
            importance_ess=importance_ess,
            observations=observations,
        )

    # 4. Posterior statistics
    prior_mean = float(cmax_ok.mean())
    prior_cv = float(cmax_ok.std() / prior_mean) if prior_mean > 0 else 0.0
    post_mean = float(cmax_post.mean())
    post_cv = float(cmax_post.std() / post_mean) if post_mean > 0 else 0.0
    ci_lo = float(np.percentile(cmax_post, 5.0))
    ci_hi = float(np.percentile(cmax_post, 95.0))

    # Prior/posterior parameter means (in linear space)
    prior_params = {
        "total_enzyme_affinity": float(b_clint[ok_mask].mean()),
        "fup": float(b_fup[ok_mask].mean()),
        "peff": float(b_peff[ok_mask].mean()),
    }
    # Posterior: average the updated multipliers
    d_clint_mean = float(np.mean(np.exp(X_post[:, 0] - X_ok[:, 0])))
    d_fup_mean = float(np.mean(np.exp(X_post[:, 1] - X_ok[:, 1])))
    d_peff_mean = float(np.mean(np.exp(X_post[:, 2] - X_ok[:, 2])))
    posterior_params = {
        "total_enzyme_affinity": prior_params["total_enzyme_affinity"] * d_clint_mean,
        "fup": prior_params["fup"] * d_fup_mean,
        "peff": prior_params["peff"] * d_peff_mean,
    }

    logger.info(
        "EnKF TDM update: %d/%d prior, %d posterior sims. "
        "Cmax prior=%.4f (CV=%.0f%%) → posterior=%.4f (CV=%.0f%%), 90%% CI=[%.4f, %.4f]. "
        "Importance ESS would be %.1f (EnKF ESS=%d).",
        n_ok, n_ensemble, n_post,
        prior_mean, 100 * prior_cv, post_mean, 100 * post_cv, ci_lo, ci_hi,
        importance_ess, n_post,
    )

    return EnKFResult(
        prior_cmax=Distribution(prior_mean, cv=prior_cv),
        posterior_cmax=Distribution(post_mean, cv=post_cv),
        cmax_ci_90=(ci_lo, ci_hi),
        prior_params=prior_params,
        posterior_params=posterior_params,
        n_prior=n_ensemble,
        n_successful=n_ok,
        n_successful_post=n_post,
        ess=float(n_post),
        importance_ess=importance_ess,
        observations=observations,
        posterior_samples=cmax_post,
    )
