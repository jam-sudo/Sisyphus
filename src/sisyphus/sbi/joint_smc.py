"""Adaptive tempering on the joint theta and nuisance posterior for hard SBC cases."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from multiprocessing import get_context

import numpy as np
from scipy.special import logsumexp, ndtr

from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW, build_box_prior

log = logging.getLogger(__name__)

# Source: data/validation/sbi_joint_smc_policy.json; Del Moral et al. (2006).
SMC_PARTICLE_COUNT = 1000
# Source: data/validation/sbi_joint_smc_policy.json.
SMC_INCREMENTAL_ESS_FRACTION = 0.5
# Source: data/validation/sbi_joint_smc_policy.json.
SMC_REJUVENATION_SWEEP_COUNT = 2
# Source: data/validation/sbi_joint_smc_policy.json.
SMC_MAX_STAGE_COUNT = 100
# Source: data/validation/sbi_joint_smc_policy.json; Roberts et al. (1997).
SMC_THETA_PROPOSAL_SCALE_RATIO = 2.38 / np.sqrt(3)
# Source: data/validation/sbi_joint_smc_policy.json; numerical covariance floor.
SMC_COVARIANCE_JITTER_FRACTION = 1e-4
# Source: data/validation/sbi_joint_smc_policy.json; Cotter et al. (2013).
SMC_INITIAL_NUISANCE_PCN_FRACTION = 0.25
# Source: data/validation/sbi_joint_smc_policy.json.
SMC_TARGET_ACCEPTANCE_FRACTION = 0.234
# Source: data/validation/sbi_joint_smc_policy.json.
SMC_ADAPTATION_GAIN_FRACTION = 0.5
# Source: data/validation/sbi_joint_smc_policy.json; bound for nuisance-dimension probe.
SMC_PROBE_NORMAL_COUNT = 1024

_joint_log_likelihood: Callable[[np.ndarray, _GaussianStream], float] | None = None


class _GaussianStream:
    """Map independent standard normals to the simulator's nuisance draws."""

    def __init__(self, normals: np.ndarray) -> None:
        self.normals = np.asarray(normals, dtype=np.float64)
        self.offset = 0

    def _take(self, count: int) -> np.ndarray:
        end = self.offset + count
        if end > len(self.normals):
            raise ValueError("Nuisance stream used more normals than its probe")
        values = self.normals[self.offset:end]
        self.offset = end
        return values

    def normal(self, loc: float, scale: float) -> float:
        return float(loc + scale * self._take(1)[0])

    def lognormal(self, mean: float, sigma: float) -> float:
        return float(np.exp(mean + sigma * self._take(1)[0]))

    def uniform(self, low: float, high: float) -> float:
        return float(low + (high - low) * ndtr(self._take(1)[0]))

    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        mean = np.asarray(mean, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.min(eigenvalues) < -1e-8:
            raise ValueError("Nuisance covariance is not positive semidefinite")
        return mean + eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0))
                                      * self._take(len(mean)))


def _evaluate(task: tuple[np.ndarray, np.ndarray]) -> float:
    if _joint_log_likelihood is None:
        raise RuntimeError("Joint likelihood is not initialized")
    theta, normals = task
    return float(_joint_log_likelihood(theta, _GaussianStream(normals)))


def _normalized_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.isfinite(log_weights).any():
        raise RuntimeError("Every joint SMC particle has zero likelihood")
    weights = np.exp(log_weights - logsumexp(log_weights))
    return weights, float(1 / np.sum(weights**2))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def sample_joint_smc(
    log_likelihood: Callable[[np.ndarray, _GaussianStream], float],
    *,
    n_draws: int,
    seed: int,
    n_particles: int = SMC_PARTICLE_COUNT,
    workers: int = 1,
    max_stages: int = SMC_MAX_STAGE_COUNT,
    rejuvenation_sweeps: int = SMC_REJUVENATION_SWEEP_COUNT,
) -> tuple[np.ndarray, dict]:
    """Sample theta from the joint posterior by adaptive tempering and MH moves.

    The Gaussian stream has the same nuisance prior as the simulator. The
    pCN nuisance move preserves that prior, so its MH ratio only uses the
    tempered observation likelihood. This leaves the original SBC gate intact.
    """
    if min(n_draws, n_particles, workers, max_stages, rejuvenation_sweeps) < 1:
        raise ValueError("Joint SMC counts must be positive")
    if n_draws > n_particles:
        raise ValueError("Joint SMC needs at least as many particles as returned draws")

    global _joint_log_likelihood
    _joint_log_likelihood = log_likelihood
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    prior = build_box_prior()
    theta = prior.sample_numpy(n_particles, rng)
    probe = _GaussianStream(np.zeros(SMC_PROBE_NORMAL_COUNT))
    log_likelihood(theta[0], probe)
    nuisance_dim = probe.offset
    if nuisance_dim < 1:
        raise ValueError("Joint SMC found no stochastic nuisance dimensions")
    normals = rng.normal(size=(n_particles, nuisance_dim))
    evaluations = 1
    beta = 0.0
    lineage = np.arange(n_particles)
    theta_scale = SMC_THETA_PROPOSAL_SCALE_RATIO
    nuisance_scale = SMC_INITIAL_NUISANCE_PCN_FRACTION
    stages = []
    try:
        with get_context("fork").Pool(workers) as pool:
            log_l = np.asarray(pool.map(_evaluate, zip(theta, normals)))
            evaluations += n_particles
            for stage in range(max_stages):
                target_ess = SMC_INCREMENTAL_ESS_FRACTION * n_particles

                def incremental_ess(next_beta: float) -> float:
                    return _normalized_weights((next_beta - beta) * log_l)[1]

                if incremental_ess(1.0) >= target_ess:
                    next_beta = 1.0
                else:
                    low, high = beta, 1.0
                    for _ in range(40):
                        middle = (low + high) / 2
                        if incremental_ess(middle) < target_ess:
                            high = middle
                        else:
                            low = middle
                    next_beta = low
                if next_beta - beta < 1e-10:
                    raise RuntimeError("Joint SMC tempering stalled before beta=1")
                weights, ess = _normalized_weights((next_beta - beta) * log_l)
                indices = _systematic_resample(weights, rng)
                theta, normals, log_l = theta[indices], normals[indices], log_l[indices]
                lineage = lineage[indices]
                beta = next_beta
                covariance = np.cov(theta, rowvar=False)
                covariance += np.diag(
                    SMC_COVARIANCE_JITTER_FRACTION * (PRIOR_HIGH - PRIOR_LOW) ** 2
                )
                root = np.linalg.cholesky(covariance)
                theta_accepts = nuisance_accepts = 0
                for _ in range(rejuvenation_sweeps):
                    proposed_theta = theta + theta_scale * rng.normal(size=theta.shape) @ root.T
                    valid = np.all((proposed_theta >= PRIOR_LOW) & (proposed_theta <= PRIOR_HIGH),
                                   axis=1)
                    proposed_log_l = np.full(n_particles, -np.inf)
                    proposed_log_l[valid] = pool.map(
                        _evaluate, zip(proposed_theta[valid], normals[valid])
                    )
                    evaluations += int(valid.sum())
                    accepted = valid & (np.log(rng.random(n_particles))
                                        < beta * (proposed_log_l - log_l))
                    theta[accepted] = proposed_theta[accepted]
                    log_l[accepted] = proposed_log_l[accepted]
                    theta_accepts += int(accepted.sum())

                    proposed_normals = (np.sqrt(1 - nuisance_scale**2) * normals
                                        + nuisance_scale * rng.normal(size=normals.shape))
                    proposed_log_l = np.asarray(pool.map(
                        _evaluate, zip(theta, proposed_normals)
                    ))
                    evaluations += n_particles
                    accepted = np.log(rng.random(n_particles)) < beta * (proposed_log_l - log_l)
                    normals[accepted] = proposed_normals[accepted]
                    log_l[accepted] = proposed_log_l[accepted]
                    nuisance_accepts += int(accepted.sum())

                theta_rate = theta_accepts / (n_particles * rejuvenation_sweeps)
                nuisance_rate = nuisance_accepts / (n_particles * rejuvenation_sweeps)
                stages.append({"beta": beta, "incremental_ess": ess,
                               "theta_acceptance": theta_rate,
                               "nuisance_acceptance": nuisance_rate})
                log.info(
                    "Joint SMC stage %d beta %.5f ESS %.1f theta accept %.3f "
                    "nuisance accept %.3f", stage + 1, beta, ess, theta_rate, nuisance_rate,
                )
                theta_scale = float(np.clip(
                    theta_scale * np.exp(SMC_ADAPTATION_GAIN_FRACTION
                                         * (theta_rate - SMC_TARGET_ACCEPTANCE_FRACTION)),
                    0.01, 5.0,
                ))
                nuisance_scale = float(np.clip(
                    nuisance_scale * np.exp(SMC_ADAPTATION_GAIN_FRACTION
                                            * (nuisance_rate - SMC_TARGET_ACCEPTANCE_FRACTION)),
                    0.01, 0.9,
                ))
                if beta == 1.0:
                    break
    finally:
        _joint_log_likelihood = None
    if beta < 1.0:
        raise RuntimeError(f"Joint SMC did not reach beta=1 after {max_stages} stages")
    selected = rng.choice(n_particles, size=n_draws, replace=False)
    diagnostics = {
        "sampler": "joint_smc", "particles": n_particles,
        "nuisance_dimensions": nuisance_dim, "stages": stages,
        "unique_initial_ancestors": int(len(np.unique(lineage))),
        "simulator_evaluations": evaluations,
        "elapsed_s": time.monotonic() - started,
    }
    return theta[selected].copy(), diagnostics
