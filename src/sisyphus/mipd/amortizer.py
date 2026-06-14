"""Posterior backends for the MIPD core.

``SIRAmortizer`` is the working reference: for the analytic F/Cmax/AUC regime the
engine is linear, so sampling-importance-resampling is exact and instant — no
training needed.

``NeuralAmortizer`` is the placeholder for the re-simulation regime (clint/peff/
concentration-time latents, dose/regimen extrapolation), where the forward map
is not analytic and a trained amortized posterior (SNPE over molecular features +
observation) would replace per-query simulation. It requires torch + sbi, which
are not installed; it raises with guidance until that dependency is added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sisyphus.mipd.core import (
    AnalyticForward,
    APrioriPK,
    FPrior,
    PosteriorPK,
    sir_posterior,
)


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
    """Amortized neural posterior (SNPE) — re-simulation regime. Requires torch+sbi."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "NeuralAmortizer requires torch + sbi, which are not installed. It "
            "targets the re-simulation regime (concentration-time / dose-regimen "
            "extrapolation) where the forward map is not analytic; it would be "
            "trained on simulated (molecular-features, observation) -> latent pairs. "
            "For the analytic measured-F / Cmax / AUC regime use SIRAmortizer, which "
            "is exact and needs no training."
        )
