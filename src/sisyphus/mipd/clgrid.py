"""CL-grid surrogate forward: a clint-scale (CL) latent + MeasuredConc likelihood.

A bioavailability (F) latent is a pure vertical scale on the engine output, so it
is analytic. A clearance latent is not: scaling clint changes the curve *shape*
(elimination rate ke = CL/V), and a single measured concentration constrains that
shape. To keep SIR fast while staying faithful to the engine, the engine is solved
once on a small clint-scale grid (compile-once / parameterize-many), and the
forward interpolates the precomputed response in log-log space:

    forward(F, s):  c(t) = (F / F_engine(s)) * interp_s( c(t; s) )

so AUC = F*Dose/CL(s) and Cmax = (F/F_engine(s))*Cmax(s) — the standard (F, CL)
decomposition, with the engine providing the s -> {c(t), Cmax, AUC, F_engine} map.
The grid itself is built by ``sisyphus.mipd.grid`` (engine solves); this module is
pure numpy over a given grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd.core import (
    FPrior,
    Posterior,
    PosteriorPK,
    _lognormal_logpdf,
    _softmax_resample,
)


@dataclass(frozen=True)
class CLGrid:
    """Precomputed engine response over a clint-scale grid (ascending ``s_grid``).

    ``conc`` is the venous concentration-time curve per scale, shape (G, H) over
    the common ``t_grid`` (H,). ``cmax``/``auc``/``f_engine`` are (G,).
    """

    s_grid: np.ndarray
    t_grid: np.ndarray
    conc: np.ndarray
    cmax: np.ndarray
    auc: np.ndarray
    f_engine: np.ndarray

    def conc_at(self, s: np.ndarray, t: float) -> np.ndarray:
        """Model venous concentration at time ``t`` for each clint-scale in ``s``.

        Interpolates each grid curve at ``t`` (linear in time), then interpolates
        across scale in log-log space. ``s`` is clipped to the grid range.
        """
        s = np.asarray(s, dtype=float)
        c_at_t = np.array(
            [np.interp(t, self.t_grid, self.conc[g]) for g in range(self.s_grid.size)]
        )
        ls = np.log(np.clip(s, self.s_grid[0], self.s_grid[-1]))
        return np.exp(
            np.interp(ls, np.log(self.s_grid), np.log(np.maximum(c_at_t, 1e-300)))
        )


class CLGridForward:
    """Forward map (F, clint-scale) -> PK state over a precomputed ``CLGrid``."""

    def __init__(self, grid: CLGrid) -> None:
        self.grid = grid
        self._ls = np.log(grid.s_grid)
        self._lcmax = np.log(grid.cmax)
        self._lauc = np.log(grid.auc)
        self._lfe = np.log(grid.f_engine)

    def _interp(self, ltable: np.ndarray, s: np.ndarray) -> np.ndarray:
        ls = np.log(np.clip(s, self.grid.s_grid[0], self.grid.s_grid[-1]))
        return np.exp(np.interp(ls, self._ls, ltable))

    def __call__(self, f: np.ndarray, s: np.ndarray) -> dict:
        f = np.asarray(f, dtype=float)
        s = np.asarray(s, dtype=float)
        f_engine = self._interp(self._lfe, s)
        scale = f / f_engine
        grid = self.grid

        def conc_at(t: float, _scale: np.ndarray = scale, _s: np.ndarray = s) -> np.ndarray:
            return _scale * grid.conc_at(_s, t)

        return {
            "f": f,
            "cl_scale": s,
            "cmax": scale * self._interp(self._lcmax, s),
            "auc": scale * self._interp(self._lauc, s),
            "conc_at": conc_at,
        }


@dataclass(frozen=True)
class CLPrior:
    """Prior over the clint-scale latent, centered at 1 (the engine's a-priori).

    ``cv`` defaults wide (1.0) because CLint is the engine's weakest link
    (``_CLINT_CV=1.0``, R^2~0.24). Samples are clipped to the grid range so the
    forward never extrapolates outside the precomputed scales.
    """

    cv: float = 1.0
    s_min: float = 0.05
    s_max: float = 20.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        sigma = math.sqrt(math.log(1.0 + self.cv * self.cv))
        s = rng.lognormal(mean=0.0, sigma=sigma, size=n)  # median 1.0
        return np.clip(s, self.s_min, self.s_max)


@dataclass(frozen=True)
class MeasuredConc:
    """A measured plasma concentration ``value`` (mg/L) at time ``t`` (h)."""

    value: float
    t: float
    cv: float = 0.25

    def log_likelihood(self, state: dict) -> np.ndarray:
        return _lognormal_logpdf(self.value, state["conc_at"](self.t), self.cv)


def sir_posterior_2d(
    f_prior: FPrior,
    cl_prior: CLPrior,
    forward: CLGridForward,
    observations,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> PosteriorPK:
    """SIR posterior over (F, clint-scale) given observations, via the CL grid.

    Handles MeasuredF / MeasuredCmax / MeasuredAUC (from core) and MeasuredConc.
    Reports the clint-scale posterior on ``PosteriorPK.cl_scale``.
    """
    if rng is None:
        rng = np.random.default_rng()
    f = f_prior.sample(n_samples, rng)
    s = cl_prior.sample(n_samples, rng)
    state = forward(f, s)
    loglik = np.zeros(n_samples)
    for obs in observations:
        loglik = loglik + obs.log_likelihood(state)
    idx, n_eff = _softmax_resample(loglik, rng)
    return PosteriorPK(
        f=Posterior(state["f"][idx]),
        cmax=Posterior(state["cmax"][idx]),
        auc=Posterior(state["auc"][idx]),
        n_eff=n_eff,
        cl_scale=Posterior(s[idx]),
    )
