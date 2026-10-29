"""Posterior backends for the MIPD core.

``SIRAmortizer`` is the working reference: for the analytic F/Cmax/AUC regime the
engine is linear, so sampling-importance-resampling is exact and instant — no
training needed.

``NeuralAmortizer`` conditions a trained SNPE posterior on the engine forward map,
single-dose regimen and sparse observations for the (F, metabolic CLint scale)
CL-grid regime. Training data come from the same engine-grid forward and priors
as ``sir_posterior_2d``. The shipped model is opt-in because it failed strict
agreement with grid SIR.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sisyphus.mipd.clgrid import CLGridForward, CLPrior, MeasuredConc
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
    The engine grid remains necessary to report Cmax/AUC from posterior draws;
    amortization replaces per-query latent likelihood weighting, not grid building.
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
    ) -> PosteriorPK:
        """Sample a conditioned two-latent posterior and map it through the grid."""
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
            theta = np.asarray(self.model.sample(context, n=n_samples), dtype=float)
            if theta.shape != (n_samples, 2) or not np.isfinite(theta).all():
                raise ValueError("neural posterior returned invalid (F, CLint scale) samples")
            f, s = theta.T
            if np.any((f <= 0) | (f > 1) | (s < cl_prior.s_min) | (s > cl_prior.s_max)):
                raise ValueError("neural posterior samples are outside the engine prior")
        else:
            f = f_prior.sample(n_samples, rng)
            s = cl_prior.sample(n_samples, rng)
        state = forward(f, s)
        return PosteriorPK(
            f=Posterior(f), cmax=Posterior(state["cmax"]),
            auc=Posterior(state["auc"]), cl_scale=Posterior(s),
            n_eff=float(n_samples),
            warnings=(
                "neural posterior is a calibrated approximation that does not match "
                "exact SIR within Monte Carlo error",
            ),
        )


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
