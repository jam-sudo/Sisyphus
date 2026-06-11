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

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Effective-sample-size floor (as a fraction of the draws) below which the
# resampled posterior is too degenerate — too few distinct particles — for its
# credible interval to be trusted; the SIR resample logs a warning when crossed.
_N_EFF_WARN_FRACTION = 0.005  # 0.5% of the draws


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
    structural error — the prior should not over-trust it. Note: the (0, 1] clip
    truncates the upper tail, so for near-fully-bioavailable drugs (F -> 1) the
    posterior piles at the ceiling and its upper credible bound can be one-sided
    (degenerate). A logit/Beta latent would keep near-boundary intervals
    two-sided; the F-clip is a known limitation of the lognormal-clip prior.
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


def ci_floor(
    ci: tuple[float, float] | None, mean: float, frac: float
) -> tuple[float, float] | None:
    """Widen a 90% interval to half-width ``frac*mean`` if it is narrower.

    Opt-in guard (``frac<=0`` is a no-op) against an over-tight conditioned
    posterior that pathologically excludes truth. Mirrors the formula of
    ``regimen.tdm._apply_ci_floor`` (which takes a TDMResult, so it is not a
    drop-in here). The widened interval is centered on ``mean``.
    """
    if frac <= 0.0 or ci is None or mean <= 0.0:
        return ci
    lo, hi = ci
    floor = frac * mean
    half = max(mean - lo, hi - mean)
    if half >= floor:
        return ci
    return (max(mean - floor, 0.0), mean + floor)


@dataclass(frozen=True)
class PosteriorPK:
    """The posterior over PK after conditioning on observations.

    ``cmax``/``auc`` are the engine-track posterior and ``meta_cmax`` (populated
    by the API) is the production meta blend routed through the same posterior.
    These are **parameter-uncertainty** bands (bioavailability F only) — they do
    NOT carry calibrated predictive coverage, because structural model error is
    not in them (the F-only ``meta_cmax.ci90`` is narrow and under-covers the
    observable Cmax). ``cmax_90ci`` (API-populated) is the train-calibrated
    split-conformal band placed around the posterior meta point, and is the
    user-facing 90% Cmax interval.

    Caveat (review finding #6): the conformal q90 is calibrated on the **a-priori**
    (unconditioned, SMILES-only) prediction error — it is NOT re-calibrated for the
    conditioned posterior. So when an informative Cmax-bearing observation is
    supplied, conditioning has already reduced the true error and the a-priori band
    is conservative (over-wide). A conditioned-case recalibration is future work; the
    band is honest-but-conservative, never anti-conservative.
    """

    f: Posterior
    cmax: Posterior
    auc: Posterior
    n_eff: float
    meta_cmax: Posterior | None = None
    cmax_90ci: tuple[float, float] | None = None
    # metabolic clint-scale latent posterior (CL-grid path); scales enzyme
    # (CYP/UGT/NAT) clearance only — renal/biliary CL is held fixed.
    cl_scale: Posterior | None = None
    # Structured non-fatal flags (e.g. extreme CrCl). Empty by default
    # (additive — preserves the prior PosteriorPK contract). Project doctrine:
    # never silently drop a warning.
    warnings: tuple[str, ...] = ()
    # Renal-CL latent posterior (IV steady-state TDM path, mipd.tdm.predict_tdm):
    # the individualized renal-clearance scale relative to the CrCl-implied value.
    # None on every other path. Additive — preserves the prior contract.
    renal_scale: Posterior | None = None


def _softmax_resample(
    loglik: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Importance weights from log-likelihoods -> resample indices + n_eff.

    Stable softmax; falls back to uniform weights if the sum is degenerate.
    Shared by the F-only (sir_posterior) and 2-latent (sir_posterior_2d) paths.
    """
    n = loglik.size
    w = np.exp(loglik - loglik.max())
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        w = np.full(n, 1.0 / n)
    else:
        w = w / w_sum
    n_eff = float(1.0 / np.sum(w ** 2))
    if n_eff < _N_EFF_WARN_FRACTION * n:
        logger.warning(
            "SIR posterior is degenerate: n_eff=%.1f of %d draws (<%.1f%% of the "
            "prior); the observation is far from the prior or its cv is too tight — "
            "widen the prior or treat the credible interval with caution.",
            n_eff, n, 100.0 * _N_EFF_WARN_FRACTION,
        )
    idx = rng.choice(n, size=n, p=w)
    return idx, n_eff


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
    idx, n_eff = _softmax_resample(loglik, rng)
    return PosteriorPK(
        f=Posterior(state["f"][idx]),
        cmax=Posterior(state["cmax"][idx]),
        auc=Posterior(state["auc"][idx]),
        n_eff=n_eff,
    )
