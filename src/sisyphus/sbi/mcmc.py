"""Pseudo-marginal MCMC for SBC cases where exact rejection is too rare."""

from __future__ import annotations

import logging
from collections.abc import Callable
from multiprocessing import get_context

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW

log = logging.getLogger(__name__)

# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_CHAIN_COUNT = 4
# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_BURNIN_STEP_COUNT = 500
# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_CHUNK_STEP_COUNT = 1000
# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_MAX_RETAINED_STEP_COUNT = 8000
# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_MAX_RHAT_RATIO = 1.01
# Source: data/validation/sbi_mcmc_fallback_policy.json.
MCMC_MIN_BULK_ESS_COUNT = 500
# Source: data/validation/sbi_mcmc_fallback_policy.json; initial 10% of each prior span.
MCMC_INITIAL_RW_SCALE_FRACTION = 0.10
# Source: data/validation/sbi_mcmc_fallback_policy.json; 3-D random-walk target.
MCMC_TARGET_ACCEPTANCE_FRACTION = 0.234
# Source: data/validation/sbi_mcmc_fallback_policy.json; damped burn-in adaptation.
MCMC_ADAPTATION_GAIN_FRACTION = 0.5

_log_draw: Callable[[np.ndarray, np.random.Generator], float] | None = None
_posterior: object | None = None


def _log_likelihood_estimate(
    theta: np.ndarray, rng: np.random.Generator, n_nuisance: int,
) -> float:
    if _log_draw is None:
        raise RuntimeError("Pseudo-marginal likelihood is not initialized")
    draws = np.asarray([_log_draw(theta, rng) for _ in range(n_nuisance)])
    if np.isnan(draws).any() or np.isneginf(draws).all():
        return -np.inf
    return float(logsumexp(draws) - np.log(n_nuisance))


def _run_chain(task: tuple) -> tuple[np.ndarray, dict]:
    chain, seed, context, n_nuisance, n_steps, state = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, chain]))
    if state is None:
        import torch

        if _posterior is None:
            raise RuntimeError("Pseudo-marginal proposal is not initialized")
        torch.set_num_threads(1)
        torch.manual_seed(seed + chain)
        x_t = torch.as_tensor(context.reshape(1, -1), dtype=torch.float32)
        theta = None
        log_likelihood = -np.inf
        for attempt in range(200):
            if attempt < 100:
                with torch.no_grad():
                    sample = _posterior.sample(
                        (1,), x=x_t, show_progress_bars=False,
                        **({"reject_outside_prior": False}
                           if hasattr(_posterior, "posterior_estimator") else {}),
                    )
                candidate = sample.detach().cpu().numpy().reshape(-1)
            else:
                candidate = rng.uniform(PRIOR_LOW, PRIOR_HIGH)
            if not np.all((candidate >= PRIOR_LOW) & (candidate <= PRIOR_HIGH)):
                continue
            candidate_log_likelihood = _log_likelihood_estimate(candidate, rng, n_nuisance)
            if np.isfinite(candidate_log_likelihood):
                theta = candidate
                log_likelihood = candidate_log_likelihood
                break
        if theta is None:
            raise RuntimeError("Pseudo-marginal chain could not find a finite initial state")
        state = {
            "theta": theta,
            "log_likelihood": log_likelihood,
            "scale": MCMC_INITIAL_RW_SCALE_FRACTION * (PRIOR_HIGH - PRIOR_LOW),
            "step": 0,
            "accepted": 0,
            "window_accepted": 0,
        }
    else:
        rng.bit_generator.state = state["rng_state"]
    kept = []
    for _ in range(n_steps):
        candidate = state["theta"] + rng.normal(scale=state["scale"])
        accepted = False
        if np.all((candidate >= PRIOR_LOW) & (candidate <= PRIOR_HIGH)):
            proposed_log_likelihood = _log_likelihood_estimate(candidate, rng, n_nuisance)
            accepted = np.log(rng.random()) < proposed_log_likelihood - state["log_likelihood"]
            if accepted:
                state["theta"] = candidate
                state["log_likelihood"] = proposed_log_likelihood
        state["accepted"] += int(accepted)
        state["step"] += 1
        if state["step"] <= MCMC_BURNIN_STEP_COUNT:
            state["window_accepted"] += int(accepted)
            if state["step"] % 50 == 0:
                rate = state["window_accepted"] / 50
                multiplier = np.exp(
                    MCMC_ADAPTATION_GAIN_FRACTION
                    * (rate - MCMC_TARGET_ACCEPTANCE_FRACTION)
                )
                state["scale"] = np.clip(
                    state["scale"] * multiplier,
                    0.001 * (PRIOR_HIGH - PRIOR_LOW),
                    0.5 * (PRIOR_HIGH - PRIOR_LOW),
                )
                state["window_accepted"] = 0
        else:
            kept.append(state["theta"].copy())
    state["rng_state"] = rng.bit_generator.state
    return np.asarray(kept), state


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values.ravel()).reshape(values.shape)
    return norm.ppf((ranks - 0.375) / (values.size + 0.25))


def _split_rhat(values: np.ndarray) -> float:
    n = values.shape[1]
    within = np.var(values, axis=1, ddof=1).mean()
    if within <= 0:
        return float("inf")
    between = n * np.var(values.mean(axis=1), ddof=1)
    return float(np.sqrt(((n - 1) * within + between) / (n * within)))


def _bulk_ess(values: np.ndarray) -> float:
    chains, n = values.shape
    within = np.var(values, axis=1, ddof=1).mean()
    between = n * np.var(values.mean(axis=1), ddof=1)
    variance = (n - 1) / n * within + between / n
    if variance <= 0:
        return 0.0
    centered = values - values.mean(axis=1, keepdims=True)
    transformed = np.fft.rfft(centered, n=2 * n, axis=1)
    autocov = np.fft.irfft(np.abs(transformed) ** 2, n=2 * n, axis=1)[:, :n]
    autocov /= np.arange(n, 0, -1)[None, :]
    rho = 1 - (within - autocov.mean(axis=0)) / variance
    rho[0] = 1.0
    pairs = rho[0::2][:n // 2] + rho[1::2][:n // 2]
    positive = []
    for pair in pairs:
        if pair <= 0:
            break
        positive.append(float(pair))
    if not positive:
        return float(chains * n)
    monotone = np.minimum.accumulate(positive)
    tau = max(1.0, -1 + 2 * float(np.sum(monotone)))
    return float(min(chains * n, chains * n / tau))


def mcmc_diagnostics(draws: np.ndarray) -> dict[str, list[float]]:
    """Return rank-normalized split R-hat and bulk ESS for each theta parameter."""
    if draws.ndim != 3 or draws.shape[0] < 4 or draws.shape[1] < 20:
        raise ValueError("MCMC diagnostics need at least four chains and 20 draws each")
    half = draws.shape[1] // 2
    split = np.concatenate((draws[:, :half], draws[:, half:2 * half]), axis=0)
    rhats = []
    ess = []
    for parameter in range(draws.shape[2]):
        values = split[:, :, parameter]
        ranked = _rank_normalize(values)
        folded = _rank_normalize(np.abs(values - np.median(values)))
        rhats.append(max(_split_rhat(ranked), _split_rhat(folded)))
        ess.append(_bulk_ess(ranked))
    return {"rhat": rhats, "bulk_ess": ess}


def sample_pseudo_marginal_mcmc(
    posterior: object,
    x: np.ndarray,
    log_likelihood_draw: Callable[[np.ndarray, np.random.Generator], float],
    *,
    n_draws: int,
    n_nuisance: int = 8,
    seed: int = 42,
    max_retained_steps: int = MCMC_MAX_RETAINED_STEP_COUNT,
    min_bulk_ess: float = MCMC_MIN_BULK_ESS_COUNT,
    max_rhat: float = MCMC_MAX_RHAT_RATIO,
) -> tuple[np.ndarray, dict]:
    """Draw theta by four-chain pseudo-marginal random-walk Metropolis-Hastings.

    Each chain retains its current unbiased nuisance likelihood estimate on
    rejection. Burn-in step-size adaptation stops before diagnostic draws.
    """
    if n_draws < 1 or n_nuisance < 1 or max_retained_steps < MCMC_CHUNK_STEP_COUNT:
        raise ValueError("MCMC draw, nuisance, and retained-step counts must be positive")
    global _log_draw, _posterior
    _log_draw = log_likelihood_draw
    _posterior = posterior
    states = [None] * MCMC_CHAIN_COUNT
    traces: list[list[np.ndarray]] = [[] for _ in states]
    diagnostics = None
    try:
        with get_context("fork").Pool(MCMC_CHAIN_COUNT) as pool:
            for chunk in range(max_retained_steps // MCMC_CHUNK_STEP_COUNT):
                steps = MCMC_CHUNK_STEP_COUNT + (MCMC_BURNIN_STEP_COUNT if chunk == 0 else 0)
                results = pool.map(_run_chain, [
                    (chain, seed, np.asarray(x, dtype=np.float32), n_nuisance,
                     steps, states[chain])
                    for chain in range(MCMC_CHAIN_COUNT)
                ])
                for chain, (trace, state) in enumerate(results):
                    traces[chain].append(trace)
                    states[chain] = state
                draws = np.stack([np.concatenate(parts) for parts in traces])
                diagnostics = mcmc_diagnostics(draws)
                log.info(
                    "Pseudo-marginal MCMC: %d retained steps/chain, max R-hat %.3f, "
                    "min bulk ESS %.1f",
                    draws.shape[1], max(diagnostics["rhat"]), min(diagnostics["bulk_ess"]),
                )
                if (max(diagnostics["rhat"]) <= max_rhat
                        and min(diagnostics["bulk_ess"]) >= min_bulk_ess):
                    break
    finally:
        _log_draw = None
        _posterior = None
    assert diagnostics is not None
    if max(diagnostics["rhat"]) > max_rhat or min(diagnostics["bulk_ess"]) < min_bulk_ess:
        raise RuntimeError(f"Pseudo-marginal MCMC diagnostics did not pass: {diagnostics}")
    per_chain = np.full(MCMC_CHAIN_COUNT, n_draws // MCMC_CHAIN_COUNT)
    per_chain[:n_draws % MCMC_CHAIN_COUNT] += 1
    thinned = np.concatenate([
        draws[chain, np.linspace(0, len(draws[chain]) - 1, count, dtype=int)]
        for chain, count in enumerate(per_chain) if count
    ])
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(thinned)
    diagnostics.update({
        "chains": MCMC_CHAIN_COUNT,
        "retained_steps_per_chain": draws.shape[1],
        "burnin_steps_per_chain": MCMC_BURNIN_STEP_COUNT,
        "nuisance_draws_per_likelihood": n_nuisance,
        "acceptance_fraction": [
            state["accepted"] / state["step"] for state in states
        ],
    })
    return thinned, diagnostics
