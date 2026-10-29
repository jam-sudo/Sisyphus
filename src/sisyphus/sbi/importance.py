"""Likelihood correction for amortized SBI posteriors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import logsumexp

from sisyphus.engine.compiler import ResolvedParams
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.types import DosingRegimen
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW

# Source: pilot in data/validation/sbi_importance_pilot.json (ODE budget versus ESS).
DEFAULT_NUISANCE_DRAW_COUNT = 8
# Source: pilot in data/validation/sbi_importance_pilot.json (ODE budget versus ESS).
DEFAULT_PROPOSAL_COUNT = 256
# Source: an ESS below 32 cannot support a stable 90% tail estimate.
MIN_IMPORTANCE_ESS_COUNT = 32.0
# Source: 5% prior proposals protect against flow support and tail errors.
DEFENSIVE_PRIOR_FRACTION = 0.05
# Source: scripts/sbi_exact_rejection.py uses rejection from the same joint prior.
MAX_REJECTION_ATTEMPT_COUNT = 100000


class LowImportanceESS(RuntimeError):
    """Raised when correction weights cannot support posterior summaries."""


def sample_corrected(
    posterior: object,
    x: np.ndarray,
    log_likelihood_draw: Callable[[np.ndarray, np.random.Generator], float],
    *,
    n_draws: int,
    n_proposals: int = DEFAULT_PROPOSAL_COUNT,
    n_nuisance: int = DEFAULT_NUISANCE_DRAW_COUNT,
    seed: int = 42,
    ess_floor: float = MIN_IMPORTANCE_ESS_COUNT,
    defensive_fraction: float = DEFENSIVE_PRIOR_FRACTION,
) -> tuple[np.ndarray, float]:
    """Resample from a flow after unbiased nuisance-marginal likelihood weighting.

    ``log_likelihood_draw`` returns the joint log density of all observations
    under one independent nuisance realization. Averaging happens in density
    space, preserving the pseudo-marginal likelihood estimate.
    """
    import torch

    if n_draws < 1 or n_proposals < 1 or n_nuisance < 1:
        raise ValueError("Sample and nuisance counts must be positive")
    if not 0 <= defensive_fraction < 1:
        raise ValueError("defensive_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(1, -1))
    n_prior = int(round(n_proposals * defensive_fraction))
    actual_fraction = n_prior / n_proposals
    with torch.no_grad():
        flow = posterior.sample(
            (n_proposals - n_prior,), x=x_t, show_progress_bars=False,
        ).detach().cpu().numpy()
        prior = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n_prior, len(PRIOR_LOW)))
        theta = np.concatenate((flow, prior), axis=0)
        log_q_flow = posterior.log_prob(
            torch.as_tensor(theta, dtype=torch.float32), x=x_t,
        ).detach().cpu().numpy().reshape(-1)
    log_p = -np.log(np.prod(PRIOR_HIGH - PRIOR_LOW))
    log_q = np.logaddexp(
        np.log1p(-actual_fraction) + log_q_flow,
        np.log(actual_fraction) + log_p if actual_fraction else -np.inf,
    )
    log_likelihood = np.empty(n_proposals)
    for i, row in enumerate(theta):
        draws = [log_likelihood_draw(row, rng) for _ in range(n_nuisance)]
        log_likelihood[i] = logsumexp(draws) - np.log(n_nuisance)
    log_weights = log_likelihood + log_p - log_q
    if not np.any(np.isfinite(log_weights)):
        raise LowImportanceESS("All corrected likelihood weights are zero")
    weights = np.exp(log_weights - logsumexp(log_weights))
    ess = float(1.0 / np.sum(weights**2))
    if ess < ess_floor:
        raise LowImportanceESS(f"Corrected SBI ESS {ess:.1f} below floor {ess_floor:.1f}")
    return theta[rng.choice(n_proposals, size=n_draws, replace=True, p=weights)], ess


class CorrectedSBCPosterior:
    """Drop-in SBC sampler that retains each script's rank and coverage gate."""

    def __init__(
        self,
        posterior: object,
        simulator: object,
        *,
        seed: int,
        multi_observation: bool = False,
    ) -> None:
        self.posterior = posterior
        self.simulator = simulator
        self.rng = np.random.default_rng(seed)
        self.multi_observation = multi_observation
        self.ess: list[float] = []
        self.rejection_count = 0
        self.proposal_counts: list[int] = []

    def sample(self, shape: tuple[int, ...], *, x: object, **_: object) -> object:
        """Draw corrected particles for one packed SBC observation."""
        import torch

        context = x.detach().cpu().numpy().reshape(-1)
        sim = self.simulator
        seed = int(self.rng.integers(0, 2**31 - 1))
        if self.multi_observation:
            from sisyphus.sbi.multi_observation import (
                ASSAY_SIGMA_LOG10,
                HOURS_PER_DAY_H,
                MAX_DOSE_COUNT,
            )
            from sisyphus.sbi.simulator import apply_theta_to_drug

            obs = context[:9].reshape(3, 3)
            used = obs[:, 2] > 0
            times = obs[used, 0] * HOURS_PER_DAY_H
            log_obs = obs[used, 1]
            dose = sim.nominal_drug.dose_mg
            interval = float(context[10]) * HOURS_PER_DAY_H
            n_doses = int(round(float(context[11]) * MAX_DOSE_COUNT))
            regimen = DosingRegimen.oral_repeated(dose, interval, n_doses)
            sigma = ASSAY_SIGMA_LOG10
            max_log_likelihood = len(times) * (
                -np.log(sigma) - 0.5 * np.log(2 * np.pi)
            )

            def log_draw(theta: np.ndarray, rng: np.random.Generator) -> float:
                overridden = apply_theta_to_drug(sim.nominal_drug, theta, sim.nominal_peff)
                realized = ResolvedParams(sim._sample_graph(rng), overridden.sample(rng))
                try:
                    output = solve_regimen(
                        sim.compiled, realized, regimen,
                        t_total_h=float(max(times)) + 1.0, dt_output=0.25,
                    )
                    if not output.solver_success:
                        return -np.inf
                    conc = output.concentrations[sim.obs_node]
                    predicted = np.interp(times, output.time_h, conc)
                    if np.any(predicted <= 0) or np.any(~np.isfinite(predicted)):
                        return -np.inf
                    residual = (log_obs - np.log10(predicted)) / sigma
                    return float(np.sum(-0.5 * residual**2
                                        - np.log(sigma) - 0.5 * np.log(2 * np.pi)))
                except (KeyError, ValueError, RuntimeError):
                    return -np.inf
        else:
            sigma = sim.obs_sigma_log10
            observed = float(context[0])
            max_log_likelihood = -np.log(sigma) - 0.5 * np.log(2 * np.pi)

            def log_draw(theta: np.ndarray, rng: np.random.Generator) -> float:
                nuisance_seed = int(rng.integers(0, 2**31 - 1))
                predicted = sim.simulate_single(theta, nuisance_seed, add_noise=False)
                if not np.isfinite(predicted):
                    return -np.inf
                residual = (observed - predicted) / sigma
                return float(-0.5 * residual**2
                             - np.log(sigma) - 0.5 * np.log(2 * np.pi))

        samples = None
        for proposal_count in (DEFAULT_PROPOSAL_COUNT, 2 * DEFAULT_PROPOSAL_COUNT,
                               4 * DEFAULT_PROPOSAL_COUNT):
            try:
                samples, ess = sample_corrected(
                    self.posterior, context, log_draw, n_draws=shape[0],
                    n_proposals=proposal_count, seed=seed + proposal_count,
                )
                self.ess.append(ess)
                self.proposal_counts.append(proposal_count)
                break
            except LowImportanceESS:
                continue
        if samples is None:
            # Joint-prior rejection is exact for this theta prior and simulator.
            accepted = []
            rng = np.random.default_rng(seed + 1)
            for _ in range(MAX_REJECTION_ATTEMPT_COUNT):
                theta = rng.uniform(PRIOR_LOW, PRIOR_HIGH)
                log_likelihood = log_draw(theta, rng)
                if rng.random() < np.exp(log_likelihood - max_log_likelihood):
                    accepted.append(theta)
                    if len(accepted) == shape[0]:
                        break
            if len(accepted) < shape[0]:
                raise LowImportanceESS(
                    f"Exact rejection yielded {len(accepted)}/{shape[0]} samples"
                ) from None
            samples = np.asarray(accepted)
            self.rejection_count += 1
        return torch.as_tensor(samples, dtype=torch.float32)
