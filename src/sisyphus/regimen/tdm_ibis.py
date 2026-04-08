"""Therapeutic Drug Monitoring — IBIS (Iterated Batch Importance Sampling).

Solves the ESS degeneration problem in standard importance sampling TDM
(tdm.py) by processing observations *sequentially* with systematic
resampling and MCMC rejuvenation when ESS drops below a threshold.

Standard IS computes the joint likelihood L = prod(L_k) for all
observations simultaneously.  With >3 observations the weight
distribution collapses (e.g. ketorolac ESS=2.5 with N=2000).

IBIS processes observations one at a time:
    1. Initialize N particles from the prior.
    2. For each observation (sorted by time):
       a. Incremental reweight: w_i *= L_k(particle_i).
       b. Compute ESS = 1 / sum(w^2).
       c. If ESS < threshold: systematic resample + MCMC rejuvenation.
    3. Extract weighted posterior from final particle cloud.

The MCMC rejuvenation step perturbs particles in a 3D log-space
(log-fup, log-CLint_total, log-peff), re-solves the ODE, and applies
a Metropolis-Hastings accept/reject criterion.  This diversifies the
particle cloud after resampling, preventing degeneracy.

References:
    Chopin (2002) "A sequential particle filter method for static models"
    Del Moral, Doucet & Jasra (2006) "Sequential Monte Carlo samplers"
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sisyphus.core import Distribution, DrugOnGraph, SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.tdm import Observation, _interpolate_concentration
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBISResult:
    """Result of IBIS-based TDM update.

    Attributes:
        prior_cmax: Prior Cmax distribution (from population model).
        posterior_cmax: Posterior Cmax distribution (after all observations).
        cmax_ci_90: 90% credible interval for posterior Cmax (5th, 95th).
        prior_params: Prior parameter means {name: value}.
        posterior_params: Posterior parameter means {name: value}.
        n_particles: Number of particles requested.
        n_successful: Number of particles with successful simulations.
        ess_history: ESS after each observation's incremental reweight.
        n_rejuvenation_steps: Total MCMC rejuvenation steps performed.
        acceptance_rates: MCMC acceptance rate per rejuvenation event.
        observations: The observations used (sorted by time_h).
        weights: Final normalized importance weights (length n_successful).
    """

    prior_cmax: Distribution
    posterior_cmax: Distribution
    cmax_ci_90: tuple[float, float]
    prior_params: dict[str, float] = field(default_factory=dict)
    posterior_params: dict[str, float] = field(default_factory=dict)
    n_particles: int = 0
    n_successful: int = 0
    ess_history: list[float] = field(default_factory=list)
    n_rejuvenation_steps: int = 0
    acceptance_rates: list[float] = field(default_factory=list)
    observations: tuple[Observation, ...] = ()
    weights: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Systematic resampling.  Returns index array of length N.

    Systematic resampling uses a single uniform draw to generate N
    evenly-spaced positions along the CDF, producing a low-variance
    index set proportional to the weights.

    Args:
        weights: Normalized weights (sum to 1), shape (N,).
        rng: NumPy random generator.

    Returns:
        Integer index array of length N (indices into original particles).
    """
    N = len(weights)
    positions = (np.arange(N) + rng.random()) / N
    cumsum = np.cumsum(weights)
    # Ensure final cumsum is exactly 1 to avoid edge-case misses
    cumsum[-1] = 1.0
    indices = np.searchsorted(cumsum, positions)
    # Clip to valid range (numerical edge case)
    np.clip(indices, 0, N - 1, out=indices)
    return indices


def _extract_3d_state(drug: DrugOnGraph) -> NDArray[np.float64]:
    """Extract [log(fup), log(clint_total), log(peff)] from a realized drug.

    These three parameters dominate PK variability (binding, metabolism,
    absorption) and form the natural state space for MCMC rejuvenation.

    Args:
        drug: A realized (point-value, cv=0) DrugOnGraph.

    Returns:
        Array of shape (3,): [log(fup), log(CLint_total), log(peff)].
    """
    fup = max(drug.fup.mean, 1e-6)
    peff = max(drug.peff.mean, 1e-6)
    clint_total = sum(d.mean for d in drug.enzyme_affinity.values())
    clint_total = max(clint_total, 1e-6)
    return np.array([np.log(fup), np.log(clint_total), np.log(peff)])


def _build_drug_from_3d(
    drug_template: DrugOnGraph,
    state_3d: NDArray[np.float64],
) -> DrugOnGraph:
    """Create a new DrugOnGraph with updated fup, enzyme_affinity, peff.

    Reconstructs the drug from a 3D log-space state vector.  Enzyme
    affinities are scaled proportionally so that individual enzyme
    contributions maintain their original ratios.

    Args:
        drug_template: Original realized DrugOnGraph (cv=0 values).
        state_3d: Array [log(fup), log(CLint_total), log(peff)].

    Returns:
        New DrugOnGraph with updated parameter values (cv=0).
    """
    new_fup = float(np.exp(state_3d[0]))
    new_clint_total = float(np.exp(state_3d[1]))
    new_peff = float(np.exp(state_3d[2]))

    # Clamp fup to (0, 1] — fraction unbound cannot exceed 1
    new_fup = min(max(new_fup, 1e-6), 1.0)
    new_peff = max(new_peff, 1e-6)

    # Scale enzyme affinities proportionally
    orig_total = sum(d.mean for d in drug_template.enzyme_affinity.values())
    if orig_total > 1e-12:
        scale = new_clint_total / orig_total
    else:
        scale = 1.0

    new_enzyme_affinity = {
        tag: Distribution(mean=max(d.mean * scale, 0.0), cv=0.0)
        for tag, d in drug_template.enzyme_affinity.items()
    }

    return dataclasses.replace(
        drug_template,
        fup=Distribution(mean=new_fup, cv=0.0),
        peff=Distribution(mean=new_peff, cv=0.0),
        enzyme_affinity=new_enzyme_affinity,
    )


def _single_obs_log_likelihood(
    pred_concentration: float,
    obs: Observation,
) -> float:
    """Gaussian log-likelihood for a single observation.

    L = -0.5 * ((obs - pred) / sigma)^2 - log(sigma) - 0.5*log(2*pi)

    Args:
        pred_concentration: Predicted concentration at observation time (mg/L).
        obs: The observation with measured concentration and assay CV.

    Returns:
        Log-likelihood (scalar).
    """
    sigma = obs.sigma
    residual = (obs.concentration - pred_concentration) / sigma
    return -0.5 * residual * residual - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def _multi_obs_log_likelihood(
    sim: SimResult,
    observations: tuple[Observation, ...],
) -> float:
    """Compute log-likelihood over multiple observations.

    Args:
        sim: SimResult from solve_regimen.
        observations: Observations to evaluate against.

    Returns:
        Sum of log-likelihoods.
    """
    total = 0.0
    for obs in observations:
        pred = _interpolate_concentration(sim, obs.time_h, obs.node)
        total += _single_obs_log_likelihood(pred, obs)
    return total


def _extract_params(drug: DrugOnGraph) -> dict[str, float]:
    """Extract key parameter means from a DrugOnGraph."""
    params = {
        "fup": drug.fup.mean,
        "peff": drug.peff.mean,
    }
    if drug.enzyme_affinity:
        total_aff = sum(d.mean for d in drug.enzyme_affinity.values())
        params["total_enzyme_affinity"] = total_aff
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ibis_update(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: tuple[Observation, ...] | list[Observation],
    n_particles: int = 2000,
    seed: int = 42,
    observation_node: str = "venous_blood",
    ess_threshold_fraction: float = 0.5,
    n_mcmc_moves: int = 5,
    mcmc_scale: float = 0.1,
) -> IBISResult:
    """Run IBIS-based TDM update with sequential reweighting and rejuvenation.

    Processes observations one-by-one (sorted by time), resampling and
    applying MCMC rejuvenation whenever ESS drops below the threshold.
    This maintains a diverse particle cloud even with many observations,
    overcoming the ESS degeneracy of standard importance sampling.

    Args:
        compiled: Pre-compiled ODE skeleton.
        graph: BodyGraph with distributional parameters (the prior).
        drug: DrugOnGraph with distributional parameters (the prior).
        regimen: Dosing regimen the patient received.
        observations: Observed concentrations with times and assay CV.
        n_particles: Number of prior particles (MC samples).
        seed: Base RNG seed for reproducibility.
        observation_node: Default compartment for observations.
        ess_threshold_fraction: Resample when ESS < n_particles * fraction.
        n_mcmc_moves: Number of Metropolis-Hastings steps per rejuvenation.
        mcmc_scale: Std dev of log-space random walk proposal (controls
            step size in log-fup, log-CLint, log-peff space).

    Returns:
        IBISResult with prior/posterior distributions, ESS history,
        rejuvenation diagnostics, and final particle weights.
    """
    observations = tuple(sorted(observations, key=lambda o: o.time_h))
    if not observations:
        raise ValueError("At least one observation is required")

    # -----------------------------------------------------------------------
    # Step 1: Initialize particles from the prior
    # -----------------------------------------------------------------------
    max_obs_t = max(obs.time_h for obs in observations)
    t_total = max(max_obs_t + 24.0, regimen.last_dose_time_h + 24.0)

    # Storage for particles
    particle_drugs: list[DrugOnGraph] = []       # realized drug per particle
    particle_graphs: list[BodyGraph] = []         # realized graph per particle
    particle_sims: list[SimResult] = []           # simulation per particle
    particle_cmax: list[float] = []               # Cmax per particle
    # Pre-computed predicted concentrations at ALL observation times
    particle_pred_conc: list[NDArray[np.float64]] = []
    particle_3d_states: list[NDArray[np.float64]] = []

    n_failures = 0

    for i in range(n_particles):
        rng = np.random.default_rng(seed + i)

        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        try:
            sim = solve_regimen(
                compiled, params, regimen, t_total_h=t_total, dt_output=0.25
            )
            if not sim.solver_success:
                n_failures += 1
                continue
        except Exception:
            n_failures += 1
            continue

        # Extract Cmax from observation node
        conc = sim.concentrations.get(observation_node)
        if conc is None or len(conc) == 0:
            n_failures += 1
            continue
        cmax = float(np.max(conc))
        if cmax <= 0:
            n_failures += 1
            continue

        # Pre-compute predicted concentrations at all observation times
        pred = np.array([
            _interpolate_concentration(sim, obs.time_h, obs.node)
            for obs in observations
        ])

        particle_drugs.append(realized_drug)
        particle_graphs.append(realized_graph)
        particle_sims.append(sim)
        particle_cmax.append(cmax)
        particle_pred_conc.append(pred)
        particle_3d_states.append(_extract_3d_state(realized_drug))

    n_ok = len(particle_drugs)
    if n_ok == 0:
        logger.warning("IBIS: all %d particles failed simulation", n_particles)
        return IBISResult(
            prior_cmax=Distribution(0.0),
            posterior_cmax=Distribution(0.0),
            cmax_ci_90=(0.0, 0.0),
            n_particles=n_particles,
            n_successful=0,
            observations=observations,
        )

    # Prior statistics (before any reweighting)
    cmax_arr = np.array(particle_cmax)
    prior_cmax_mean = float(np.mean(cmax_arr))
    prior_cmax_cv = (
        float(np.std(cmax_arr) / prior_cmax_mean)
        if prior_cmax_mean > 0 else 0.0
    )

    prior_params: dict[str, float] = {}
    param_samples = [_extract_params(d) for d in particle_drugs]
    if param_samples:
        for key in param_samples[0]:
            vals = np.array([p[key] for p in param_samples])
            prior_params[key] = float(np.mean(vals))

    # -----------------------------------------------------------------------
    # Step 2: Sequential observation loop
    # -----------------------------------------------------------------------
    log_weights = np.zeros(n_ok)  # start uniform (log(1/N) offset cancels)
    ess_history: list[float] = []
    acceptance_rates: list[float] = []
    total_rejuvenation_steps = 0
    ess_threshold = n_ok * ess_threshold_fraction

    # Track which observations have been processed (for MCMC full log-lik)
    obs_seen: list[Observation] = []

    for k, obs_k in enumerate(observations):
        obs_seen.append(obs_k)

        # (a) Incremental reweight: add log-likelihood of obs_k
        for i in range(n_ok):
            pred_k = particle_pred_conc[i][k]
            ll_k = _single_obs_log_likelihood(pred_k, obs_k)
            log_weights[i] += ll_k

        # (b) Normalize weights
        lw_max = np.max(log_weights)
        weights = np.exp(log_weights - lw_max)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum
        else:
            # Degenerate: reset to uniform
            weights = np.ones(n_ok) / n_ok

        # (c) Compute ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_history.append(float(ess))

        logger.debug(
            "IBIS obs %d/%d (t=%.1fh, C=%.3f): ESS=%.1f (%.1f%%)",
            k + 1, len(observations),
            obs_k.time_h, obs_k.concentration,
            ess, 100.0 * ess / n_ok,
        )

        # (d) Resample + rejuvenate if ESS below threshold
        if ess < ess_threshold:
            logger.info(
                "IBIS: ESS=%.1f < threshold=%.1f at obs %d; resampling + rejuvenation",
                ess, ess_threshold, k + 1,
            )

            # Systematic resampling
            resample_rng = np.random.default_rng(seed + n_particles + k)
            indices = _systematic_resample(weights, resample_rng)

            # Duplicate particles according to resampled indices
            new_drugs = [particle_drugs[j] for j in indices]
            new_graphs = [particle_graphs[j] for j in indices]
            new_sims = [particle_sims[j] for j in indices]
            new_cmax = [particle_cmax[j] for j in indices]
            new_pred = [particle_pred_conc[j].copy() for j in indices]
            new_3d = [particle_3d_states[j].copy() for j in indices]

            # Reset log-weights to uniform after resampling
            log_weights = np.zeros(n_ok)

            # MCMC rejuvenation
            mcmc_rng = np.random.default_rng(seed + 2 * n_particles + k)
            n_accepted = 0
            n_proposed = 0
            obs_seen_tuple = tuple(obs_seen)

            for i in range(n_ok):
                # Current log-likelihood (all observations seen so far)
                current_ll = _multi_obs_log_likelihood(
                    new_sims[i], obs_seen_tuple
                )

                for _step in range(n_mcmc_moves):
                    n_proposed += 1

                    # Propose in 3D log-space
                    proposal_3d = new_3d[i] + mcmc_rng.normal(
                        0.0, mcmc_scale, size=3
                    )

                    # Build proposed drug
                    proposed_drug = _build_drug_from_3d(
                        new_drugs[i], proposal_3d
                    )

                    # Re-solve with proposed drug parameters
                    proposed_params = ResolvedParams(
                        new_graphs[i], proposed_drug
                    )
                    try:
                        proposed_sim = solve_regimen(
                            compiled, proposed_params, regimen,
                            t_total_h=t_total, dt_output=0.25,
                        )
                        if not proposed_sim.solver_success:
                            continue  # reject (solver failed)
                    except Exception:
                        continue  # reject

                    # Check Cmax > 0
                    proposed_conc = proposed_sim.concentrations.get(
                        observation_node
                    )
                    if (proposed_conc is None or len(proposed_conc) == 0
                            or float(np.max(proposed_conc)) <= 0):
                        continue  # reject

                    # Compute proposed log-likelihood
                    proposed_ll = _multi_obs_log_likelihood(
                        proposed_sim, obs_seen_tuple
                    )

                    # Metropolis-Hastings accept/reject
                    log_alpha = proposed_ll - current_ll
                    if np.log(mcmc_rng.random()) < log_alpha:
                        # Accept
                        n_accepted += 1
                        new_drugs[i] = proposed_drug
                        new_sims[i] = proposed_sim
                        new_cmax[i] = float(np.max(proposed_conc))
                        new_3d[i] = proposal_3d.copy()
                        current_ll = proposed_ll

                        # Update predicted concentrations for remaining obs
                        new_pred[i] = np.array([
                            _interpolate_concentration(
                                proposed_sim, obs.time_h, obs.node
                            )
                            for obs in observations
                        ])

            # Record acceptance rate
            acc_rate = n_accepted / n_proposed if n_proposed > 0 else 0.0
            acceptance_rates.append(acc_rate)
            total_rejuvenation_steps += n_proposed

            logger.info(
                "IBIS rejuvenation: %d/%d accepted (%.1f%%)",
                n_accepted, n_proposed, 100.0 * acc_rate,
            )

            # Update particle storage
            particle_drugs = new_drugs
            particle_graphs = new_graphs
            particle_sims = new_sims
            particle_cmax = new_cmax
            particle_pred_conc = new_pred
            particle_3d_states = new_3d

    # -----------------------------------------------------------------------
    # Step 3: Extract posterior
    # -----------------------------------------------------------------------

    # Final weight normalization
    lw_max = np.max(log_weights)
    final_weights = np.exp(log_weights - lw_max)
    fw_sum = np.sum(final_weights)
    if fw_sum > 0:
        final_weights /= fw_sum
    else:
        final_weights = np.ones(n_ok) / n_ok

    cmax_arr = np.array(particle_cmax)

    # Posterior Cmax statistics (weighted)
    post_cmax_mean = float(np.average(cmax_arr, weights=final_weights))
    post_cmax_var = float(
        np.average((cmax_arr - post_cmax_mean) ** 2, weights=final_weights)
    )
    post_cmax_std = float(np.sqrt(post_cmax_var))
    post_cmax_cv = (
        post_cmax_std / post_cmax_mean if post_cmax_mean > 0 else 0.0
    )

    # 90% credible interval via weighted quantiles
    sorted_idx = np.argsort(cmax_arr)
    sorted_cmax = cmax_arr[sorted_idx]
    sorted_weights = final_weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    ci_lo = float(sorted_cmax[np.searchsorted(cum_weights, 0.05)])
    ci_hi = float(sorted_cmax[np.searchsorted(cum_weights, 0.95)])

    # Posterior parameter statistics (weighted)
    posterior_params: dict[str, float] = {}
    param_samples_final = [_extract_params(d) for d in particle_drugs]
    if param_samples_final:
        for key in param_samples_final[0]:
            vals = np.array([p[key] for p in param_samples_final])
            posterior_params[key] = float(
                np.average(vals, weights=final_weights)
            )

    logger.info(
        "IBIS complete: %d/%d particles, %d observations, "
        "Cmax prior=%.4f (CV=%.0f%%) -> posterior=%.4f (CV=%.0f%%), "
        "90%% CI=[%.4f, %.4f], %d rejuvenation steps, "
        "ESS history: %s",
        n_ok, n_particles, len(observations),
        prior_cmax_mean, 100 * prior_cmax_cv,
        post_cmax_mean, 100 * post_cmax_cv,
        ci_lo, ci_hi,
        total_rejuvenation_steps,
        [f"{e:.0f}" for e in ess_history],
    )

    return IBISResult(
        prior_cmax=Distribution(prior_cmax_mean, cv=prior_cmax_cv),
        posterior_cmax=Distribution(post_cmax_mean, cv=post_cmax_cv),
        cmax_ci_90=(ci_lo, ci_hi),
        prior_params=prior_params,
        posterior_params=posterior_params,
        n_particles=n_particles,
        n_successful=n_ok,
        ess_history=ess_history,
        n_rejuvenation_steps=total_rejuvenation_steps,
        acceptance_rates=acceptance_rates,
        observations=observations,
        weights=final_weights,
    )
