"""MIPD inference core: engine-as-prior Bayesian update of bioavailability F.

The engine produces an a-priori PK prediction whose dominant structural error is
bioavailability F (= fa*Fg*Fh). This module treats the *true* F as a latent with
a wide prior centered on the engine's emergent ``F_engine`` and updates it from
any measured observation (measured F, Cmax, or AUC) via sampling-importance-
resampling (SIR).

Because the production engine is **linear in dose** (verified: Cmax proportional
to dose), the forward map ``Cmax(F) = Cmax0 * F / F_engine`` is *exact*. SIR
therefore needs no per-sample engine re-simulation and runs over tens of
thousands of samples instantly — the same mechanism the validated measured-F
routing uses, generalized to a full posterior with credible intervals and
heterogeneous observation types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class APrioriPK:
    """The engine's a-priori prediction — the anchor the F prior is built on.

    Attributes:
        cmax0: a-priori engine Cmax (at ``f_engine``).
        auc0: a-priori engine AUC(0-t) (at ``f_engine``).
        f_engine: the engine's emergent oral bioavailability (AUC_oral/AUC_iv).
    """

    cmax0: float
    auc0: float
    f_engine: float

    def __post_init__(self) -> None:
        if self.f_engine <= 0:
            raise ValueError(f"f_engine must be > 0, got {self.f_engine}")


class AnalyticForward:
    """Linear-engine forward map F -> PK state. Exact (engine is first-order)."""

    def __init__(self, apriori: APrioriPK) -> None:
        self.apriori = apriori

    def __call__(self, f: np.ndarray) -> dict:
        scale = f / self.apriori.f_engine
        return {
            "f": f,
            "cmax": self.apriori.cmax0 * scale,
            "auc": self.apriori.auc0 * scale,
        }


def _lognormal_logpdf(value: float, mean: np.ndarray, cv: float) -> np.ndarray:
    """log p(value | lognormal(median=mean, cv)). Vectorized over ``mean``."""
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mean = np.asarray(mean, dtype=float)
    mu = np.log(np.where(mean > 0.0, mean, 1e-300))
    x = math.log(value) if value > 0.0 else -700.0
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma)


class Observation(Protocol):
    """A measured datum that constrains the latent via a likelihood."""

    def log_likelihood(self, state: dict) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class MeasuredF:
    """Measured absolute oral bioavailability (0 < F <= 1)."""

    value: float
    cv: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.value <= 1.0:
            raise ValueError(f"measured F must be in (0, 1], got {self.value}")

    def log_likelihood(self, state: dict) -> np.ndarray:
        return _lognormal_logpdf(self.value, state["f"], self.cv)


@dataclass(frozen=True)
class MeasuredCmax:
    """Measured Cmax (mg/L) at the predicted dose/route."""

    value: float
    cv: float = 0.30

    def log_likelihood(self, state: dict) -> np.ndarray:
        return _lognormal_logpdf(self.value, state["cmax"], self.cv)


@dataclass(frozen=True)
class MeasuredAUC:
    """Measured AUC(0-t) (mg*h/L) at the predicted dose/route."""

    value: float
    cv: float = 0.30

    def log_likelihood(self, state: dict) -> np.ndarray:
        return _lognormal_logpdf(self.value, state["auc"], self.cv)


@dataclass(frozen=True)
class FPrior:
    """Wide prior over the true F, centered on ``f_engine``, truncated to (0, 1].

    ``cv`` defaults wide (1.0) because the engine's emergent F is the dominant
    structural error — the prior should not over-trust it.
    """

    f_engine: float
    cv: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        center = min(max(self.f_engine, 1e-4), 1.0)
        sigma = math.sqrt(math.log(1.0 + self.cv * self.cv))
        s = rng.lognormal(mean=math.log(center), sigma=sigma, size=n)
        return np.clip(s, 1e-6, 1.0)


@dataclass(frozen=True)
class Posterior:
    """Posterior samples of a scalar PK quantity."""

    samples: np.ndarray

    @property
    def point(self) -> float:
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    def ci(self, level: float = 0.90) -> tuple[float, float]:
        a = (1.0 - level) / 2.0
        lo, hi = np.percentile(self.samples, [100.0 * a, 100.0 * (1.0 - a)])
        return (float(lo), float(hi))

    @property
    def ci90(self) -> tuple[float, float]:
        return self.ci(0.90)


@dataclass(frozen=True)
class PosteriorPK:
    """The posterior over PK after conditioning on observations."""

    f: Posterior
    cmax: Posterior
    auc: Posterior
    n_eff: float


def sir_posterior(
    prior: FPrior,
    forward: AnalyticForward,
    observations,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> PosteriorPK:
    """Sampling-importance-resampling posterior over F and the resulting PK.

    Draws ``n_samples`` from the prior, weights them by the joint likelihood of
    the observations, and resamples. With no observations it returns the prior
    (i.e. the a-priori engine prediction). Reports the effective sample size
    ``n_eff`` (1/sum(w^2)) as a degeneracy diagnostic.
    """
    if rng is None:
        rng = np.random.default_rng()
    f = prior.sample(n_samples, rng)
    state = forward(f)
    loglik = np.zeros(n_samples)
    for obs in observations:
        loglik = loglik + obs.log_likelihood(state)
    w = np.exp(loglik - loglik.max())
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        w = np.full(n_samples, 1.0 / n_samples)
    else:
        w = w / w_sum
    n_eff = float(1.0 / np.sum(w ** 2))
    idx = rng.choice(n_samples, size=n_samples, p=w)
    return PosteriorPK(
        f=Posterior(state["f"][idx]),
        cmax=Posterior(state["cmax"][idx]),
        auc=Posterior(state["auc"][idx]),
        n_eff=n_eff,
    )
