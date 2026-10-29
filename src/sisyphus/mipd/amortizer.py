"""Posterior backends for the MIPD core.

``SIRAmortizer`` is the working reference: for the analytic F/Cmax/AUC regime the
engine is linear, so sampling-importance-resampling is exact and instant — no
training needed.

``NeuralAmortizer`` uses a trained SNPE posterior as a proposal for the (F,
metabolic CLint scale) CL-grid regime, then corrects its draws with the same
prior and likelihood as grid SIR. A prior proposal component covers the atoms
created by clipping F at 1 and CLint at its bounds.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.special import log_ndtr, logsumexp

from sisyphus.mipd.clgrid import CLGridForward, CLPrior, MeasuredConc, sir_posterior_2d
from sisyphus.mipd.core import (
    AnalyticForward,
    APrioriPK,
    FPrior,
    MeasuredAUC,
    MeasuredCmax,
    MeasuredF,
    Observation,
    Posterior,
    PosteriorPK,
    sir_posterior,
)
from sisyphus.regimen.types import DEFAULT_ORAL_NODE, DosingRegimen

# Fixed log-scale design spans CLPrior's 0.05–20 support (mipd.clgrid.CLPrior).
_DESIGN_CLINT_SCALE_RATIO = np.array([0.05, 0.25, 1.0, 4.0, 20.0])
# Training's sparse concentration observations use 4 and 16 h (train_mipd_neural_amortizer).
_DESIGN_CONC_TIME_H = (4.0, 16.0)
# Project release floor from train_mipd_neural_amortizer._MIN_SIR_ESS_COUNT.
_MIN_NEURAL_ESS_COUNT = 200
# Defensive importance sampling (Hesterberg, 1995) covers clipped-prior boundary atoms.
_PRIOR_PROPOSAL_FRACTION = 0.5
# At least ten proposals per ESS-floor particle; grid SIR uses 20,000 initial draws.
_MIN_NEURAL_PROPOSALS_COUNT = 2000
# Same adaptive grid-SIR bounds as train_mipd_neural_amortizer._sir_reference.
_INITIAL_GRID_FALLBACK_DRAWS_COUNT = 20000
_MAX_GRID_FALLBACK_DRAWS_COUNT = 320000


class Amortizer(Protocol):
    """Maps (a-priori engine state, observations) -> posterior PK."""

    def posterior(
        self,
        apriori: APrioriPK,
        observations,
        *,
        rng: np.random.Generator | None = None,
    ) -> PosteriorPK:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class SIRAmortizer:
    """Exact SIR posterior for the analytic (linear-engine) F regime."""

    prior_cv: float = 1.0
    n_samples: int = 20000

    def posterior(
        self,
        apriori: APrioriPK,
        observations,
        *,
        rng: np.random.Generator | None = None,
    ) -> PosteriorPK:
        return sir_posterior(
            FPrior(apriori.f_engine, self.prior_cv),
            AnalyticForward(apriori),
            list(observations),
            n_samples=self.n_samples,
            rng=rng,
        )


class NeuralAmortizer:
    """SNPE posterior over the same F and metabolic CLint latents as grid SIR.

    ``model`` is a validated ``NPETrainingResult`` from ``sisyphus.sbi.amortizer``.
    The context contains the grid response at fixed CLint scales, not molecular identity.
    The engine grid remains necessary for likelihood evaluation and PK outputs.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def load(cls, path: Path) -> NeuralAmortizer:
        """Load a trusted, validated model artifact; torch and sbi are optional."""
        try:
            import sbi  # noqa: F401
            import torch  # noqa: F401

            from sisyphus.ml.registry import verify_model_artifact
            from sisyphus.sbi.amortizer import load_result
        except ImportError as exc:
            raise ImportError(
                "NeuralAmortizer.load requires the optional sbi extra (torch+sbi)"
            ) from exc
        verify_model_artifact(path)
        return cls(load_result(path))

    def posterior(
        self,
        f_prior: FPrior,
        cl_prior: CLPrior,
        forward: CLGridForward,
        observations: Sequence[Observation],
        *,
        regimen: DosingRegimen,
        n_samples: int = 20000,
        rng: np.random.Generator | None = None,
        raw_flow: bool = False,
    ) -> PosteriorPK:
        """Correct neural draws with the grid likelihood/prior, or explicitly use raw flow."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng() if rng is None else rng
        obs = tuple(observations)
        context = _neural_context(
            f_prior, cl_prior, forward, obs, regimen
        )
        if obs:
            if self.model is None:
                raise ValueError("a trained neural model is required with observations")
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "NeuralAmortizer.posterior requires the optional sbi extra (torch+sbi)"
                ) from exc
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
            n_proposals = n_samples if raw_flow else max(n_samples, _MIN_NEURAL_PROPOSALS_COUNT)
            n_prior = 0 if raw_flow else int(n_proposals * _PRIOR_PROPOSAL_FRACTION)
            theta = np.asarray(
                self.model.sample(
                    context, n=n_proposals - n_prior,
                    **({"reject_outside_prior": False} if not raw_flow else {}),
                ),
                dtype=float,
            )
            if theta.shape != (n_proposals - n_prior, 2) or not np.isfinite(theta).all():
                raise ValueError("neural posterior returned invalid (F, CLint scale) samples")
            if n_prior:
                prior_theta = np.column_stack((
                    f_prior.sample(n_prior, rng), cl_prior.sample(n_prior, rng)
                ))
                theta = np.concatenate((theta, prior_theta))
            f, s = theta.T
            inside = (f >= 1e-6) & (f <= 1) & (s >= cl_prior.s_min) & (s <= cl_prior.s_max)
            if raw_flow and not np.all(inside):
                raise ValueError("neural posterior samples are outside the engine prior")
        else:
            f = f_prior.sample(n_samples, rng)
            s = cl_prior.sample(n_samples, rng)
        state = forward(np.clip(f, 1e-6, 1.0), np.clip(s, cl_prior.s_min, cl_prior.s_max))
        if obs and not raw_flow:
            log_prior = np.full(n_proposals, -np.inf)
            log_prior[inside] = (
                _clipped_log_normal_logprob(
                    f[inside], min(max(f_prior.f_engine, 1e-4), 1.0),
                    f_prior.cv, 1e-6, 1.0,
                )
                + _clipped_log_normal_logprob(
                    s[inside], 1.0, cl_prior.cv, cl_prior.s_min, cl_prior.s_max
                )
            )
            boundary = (f == 1e-6) | (f == 1.0) | (s == cl_prior.s_min) | (s == cl_prior.s_max)
            log_flow = np.full(n_proposals, -np.inf)
            if np.any(inside & ~boundary):
                device = next(self.model.density_estimator.parameters()).device
                # Untruncated flow draws require its base density, not sbi's leakage correction.
                values = self.model.posterior.log_prob(
                    torch.as_tensor(theta[inside & ~boundary], dtype=torch.float32, device=device),
                    x=torch.as_tensor(context, dtype=torch.float32, device=device),
                    norm_posterior=False,
                )
                log_flow[inside & ~boundary] = np.asarray(values.detach().cpu(), dtype=float)
            log_proposal = np.logaddexp(
                math.log(n_prior / n_proposals) + log_prior,
                math.log(1 - n_prior / n_proposals) + log_flow,
            )
            log_weight = np.full(n_proposals, -np.inf)
            log_weight[inside] = log_prior[inside] - log_proposal[inside]
            for observation in obs:
                log_weight[inside] += observation.log_likelihood(state)[inside]
            weights = np.exp(log_weight - logsumexp(log_weight))
            n_eff = float(1.0 / np.sum(weights**2)) if np.isfinite(weights).all() else 0.0
            if n_eff < _MIN_NEURAL_ESS_COUNT:
                n_grid_draws = max(n_samples, _INITIAL_GRID_FALLBACK_DRAWS_COUNT)
                while True:
                    fallback = sir_posterior_2d(
                        f_prior, cl_prior, forward, obs,
                        n_samples=n_grid_draws, rng=rng,
                    )
                    if (
                        fallback.n_eff >= _MIN_NEURAL_ESS_COUNT
                        or n_grid_draws >= _MAX_GRID_FALLBACK_DRAWS_COUNT
                    ):
                        break
                    n_grid_draws *= 2
                fallback_idx = rng.choice(len(fallback.f.samples), size=n_samples, replace=False)
                return PosteriorPK(
                    f=Posterior(fallback.f.samples[fallback_idx]),
                    cmax=Posterior(fallback.cmax.samples[fallback_idx]),
                    auc=Posterior(fallback.auc.samples[fallback_idx]),
                    cl_scale=Posterior(fallback.cl_scale.samples[fallback_idx]),
                    n_eff=fallback.n_eff,
                    proposal_n_eff=n_eff,
                    warnings=(f"neural IS ESS {n_eff:.1f} below 200; used grid SIR",),
                )
            idx = rng.choice(n_proposals, size=n_samples, p=weights)
            return PosteriorPK(
                f=Posterior(f[idx]), cmax=Posterior(state["cmax"][idx]),
                auc=Posterior(state["auc"][idx]), cl_scale=Posterior(s[idx]), n_eff=n_eff,
                proposal_n_eff=n_eff,
            )
        return PosteriorPK(
            f=Posterior(f), cmax=Posterior(state["cmax"]),
            auc=Posterior(state["auc"]), cl_scale=Posterior(s),
            n_eff=float(n_samples),
            warnings=() if not obs or not raw_flow else (
                "neural posterior is a calibrated approximation that does not match "
                "exact SIR within Monte Carlo error",
            ),
        )


def _clipped_log_normal_logprob(
    values: np.ndarray, median: float, cv: float, lower: float, upper: float,
) -> np.ndarray:
    """Density in the interior and probability mass at each clipped endpoint."""
    sigma = math.sqrt(math.log1p(cv * cv))
    z = (np.log(values) - math.log(median)) / sigma
    result = -0.5 * z**2 - np.log(values * sigma * math.sqrt(2 * math.pi))
    result[values == lower] = log_ndtr((math.log(lower) - math.log(median)) / sigma)
    result[values == upper] = log_ndtr(-(math.log(upper) - math.log(median)) / sigma)
    return result


def _neural_context(
    f_prior: FPrior,
    cl_prior: CLPrior,
    forward: CLGridForward,
    observations,
    regimen: DosingRegimen,
) -> np.ndarray:
    # shortcut: CLGridForward models one oral dose; extend the simulator before repeated regimens.
    # shortcut: training uses CV 1 priors; simulate other widths before accepting them here.
    if f_prior.cv != 1.0 or cl_prior.cv != 1.0:
        raise ValueError("neural posterior was trained with F and CLint prior CV 1")
    if not np.isclose(cl_prior.s_min, _DESIGN_CLINT_SCALE_RATIO[0]) or (
        not np.isclose(cl_prior.s_max, _DESIGN_CLINT_SCALE_RATIO[-1])
    ):
        raise ValueError("neural posterior requires the trained CLint prior bounds")
    if not np.isfinite(f_prior.f_engine) or not 0 < f_prior.f_engine <= 1:
        raise ValueError("F prior centre must be finite and in (0, 1]")
    if len(regimen.events) != 1 or regimen.events[0].node != DEFAULT_ORAL_NODE:
        raise ValueError("neural CL-grid posterior supports one oral dose")
    dose = regimen.events[0].dose_mg
    if dose <= 0 or regimen.events[0].duration_h != 0:
        raise ValueError("neural CL-grid posterior requires a positive oral bolus")
    if len(observations) > 2:
        raise ValueError("neural posterior supports at most two observations")
    if forward.grid.t_grid[0] > _DESIGN_CONC_TIME_H[0] or (
        forward.grid.t_grid[-1] < _DESIGN_CONC_TIME_H[-1]
    ):
        raise ValueError("engine grid must cover the neural design times")
    design = forward(np.ones(len(_DESIGN_CLINT_SCALE_RATIO)), _DESIGN_CLINT_SCALE_RATIO)
    context = [
        *np.log(design["cmax"]), *np.log(design["auc"]),
        *(value for t in _DESIGN_CONC_TIME_H for value in np.log(design["conc_at"](t))),
        np.log(dose), f_prior.f_engine, f_prior.cv, cl_prior.cv,
    ]
    if not np.isfinite(context).all():
        raise ValueError("engine forward-map summary must be finite")
    for obs in (*observations, *([None] * (2 - len(observations)))):
        slot = [0.0] * 7
        if obs is not None:
            for kind, cls in enumerate((MeasuredF, MeasuredCmax, MeasuredAUC, MeasuredConc)):
                if isinstance(obs, cls):
                    slot[kind] = 1.0
                    break
            else:
                raise TypeError(f"unsupported neural observation: {type(obs).__name__}")
            if not np.isfinite(obs.value) or not np.isfinite(obs.cv) or obs.cv <= 0:
                raise ValueError("observation value and cv must be finite and cv > 0")
            slot[4] = np.log(obs.value)
            slot[5] = np.log(obs.cv)
            if isinstance(obs, MeasuredConc):
                if not np.isfinite(obs.t) or not (
                    forward.grid.t_grid[0] <= obs.t <= forward.grid.t_grid[-1]
                ):
                    raise ValueError("concentration time is outside the engine grid")
                slot[6] = obs.t / forward.grid.t_grid[-1]
        context.extend(slot)
    return np.asarray(context, dtype=np.float32)
