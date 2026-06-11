"""IV steady-state renal-CL grid: a free renal-clearance latent + multi-dose solve.

For an IV drug the engine-as-prior's dominant structural error (bioavailability F)
is absent (F == 1). The latent that a steady-state trough constrains is renal
clearance. This module builds the engine over a renal-CL scale ``r`` (scaling
``drug.renal_clearance``) by re-solving the multi-dose regimen with
``regimen.solver.solve_regimen``, and runs a low-D SIR over ``r`` conditioned on a
``MeasuredConc`` trough. ``r`` multiplies the CrCl-set renal CL, so its prior is
centered on the CrCl-implied value (r=1.0). See the design spec
docs/superpowers/specs/2026-06-11-mipd-steady-state-iv-tdm-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd.core import Posterior, PosteriorPK, _softmax_resample


@dataclass(frozen=True)
class RenalCLPrior:
    """Prior over the renal-CL scale ``r``, centered at 1.0 (= CrCl-implied CL).

    ``cv`` defaults wide (renal CL is the engine's individual-level unknown).
    Samples are clipped to the grid range so the forward never extrapolates.
    """

    cv: float = 1.0
    r_min: float = 0.2
    r_max: float = 5.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        sigma = math.sqrt(math.log(1.0 + self.cv * self.cv))
        r = rng.lognormal(mean=0.0, sigma=sigma, size=n)  # median 1.0
        return np.clip(r, self.r_min, self.r_max)


@dataclass(frozen=True)
class RenalCLGrid:
    """Precomputed engine response over a renal-CL grid (ascending ``r_grid``).

    ``conc`` is the venous concentration-time curve per scale, shape (G, H) over
    the common ``t_grid`` (H,). ``cmax``/``auc`` are the steady-state final-interval
    quantities (G,). No ``f_engine`` — F == 1 for IV.
    """

    r_grid: np.ndarray
    t_grid: np.ndarray
    conc: np.ndarray
    cmax: np.ndarray
    auc: np.ndarray

    def conc_at(self, r: np.ndarray, t: float) -> np.ndarray:
        """Model venous concentration at time ``t`` for each renal-scale in ``r``.

        Interpolates each grid curve at ``t`` (linear in time), then across scale
        in log-log space. ``r`` is clipped to the grid range. ``t`` must lie within
        the simulated horizon — a later time (a trough past the regimen) raises.
        """
        if t < self.t_grid[0] or t > self.t_grid[-1]:
            raise ValueError(
                f"observation time t={t} h is outside the engine grid "
                f"[{self.t_grid[0]}, {self.t_grid[-1]}] h; extend the regimen to cover it"
            )
        r = np.asarray(r, dtype=float)
        c_at_t = np.array(
            [np.interp(t, self.t_grid, self.conc[g]) for g in range(self.r_grid.size)]
        )
        lr = np.log(np.clip(r, self.r_grid[0], self.r_grid[-1]))
        return np.exp(
            np.interp(lr, np.log(self.r_grid), np.log(np.maximum(c_at_t, 1e-300)))
        )


class RenalCLForward:
    """Forward map renal-scale ``r`` -> PK state over a precomputed ``RenalCLGrid``."""

    def __init__(self, grid: RenalCLGrid) -> None:
        self.grid = grid
        self._lr = np.log(grid.r_grid)
        self._lcmax = np.log(grid.cmax)
        self._lauc = np.log(grid.auc)

    def _interp(self, ltable: np.ndarray, r: np.ndarray) -> np.ndarray:
        lr = np.log(np.clip(r, self.grid.r_grid[0], self.grid.r_grid[-1]))
        return np.exp(np.interp(lr, self._lr, ltable))

    def __call__(self, r: np.ndarray) -> dict:
        r = np.asarray(r, dtype=float)
        grid = self.grid

        def conc_at(t: float, _r: np.ndarray = r) -> np.ndarray:
            return grid.conc_at(_r, t)

        return {
            "renal_scale": r,
            "cmax": self._interp(self._lcmax, r),
            "auc": self._interp(self._lauc, r),
            "conc_at": conc_at,
        }
