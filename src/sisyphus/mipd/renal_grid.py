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

# numpy 2.0+ renamed trapz -> trapezoid; match the codebase idiom (pk/nca.py).
_trapz = getattr(np, "trapezoid", np.trapz)


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


def _regimen_interval_h(regimen) -> float:
    """Dosing interval tau from a regimen (events[1]-events[0]); 24.0 if single-dose."""
    ev = regimen.events
    return float(ev[1].time_h - ev[0].time_h) if len(ev) >= 2 else 24.0


def build_renal_cl_grid(
    smiles: str,
    regimen,
    *,
    n_grid: int = 13,
    r_range: tuple[float, float] = (0.2, 5.0),
    renal_factor: float = 1.0,
    kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> RenalCLGrid:
    """Solve the IV regimen over a renal-CL grid and return a ``RenalCLGrid``.

    Reuses ``grid._build_grid_engine`` (engine setup, CrCl renal_factor) and
    ``regimen.solver.solve_regimen`` (multi-dose solve). ``cmax``/``auc`` are the
    steady-state final-dosing-interval quantities. F == 1 (no f_engine column).
    """
    import dataclasses

    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.mipd.grid import (
        _build_grid_engine,
        _fill_nan_log_s,
        _nearest_finite_backfill,
    )
    from sisyphus.regimen.solver import solve_regimen

    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, regimen.events[0].dose_mg, "iv", renal_factor, kp_method
    )
    r_grid = np.geomspace(r_range[0], r_range[1], n_grid)

    last = float(regimen.last_dose_time_h)
    tau = _regimen_interval_h(regimen)
    t_total = last + max(tau, 24.0)
    n_points = max(2, int(round(t_total / dt_output)) + 1)
    t_grid = np.linspace(0.0, t_total, n_points)

    conc_rows: list[np.ndarray] = []
    cmaxs: list[float] = []
    aucs: list[float] = []
    for r in r_grid:
        drug_r = dataclasses.replace(
            drug,
            renal_clearance=Distribution(
                mean=drug.renal_clearance.mean * float(r), cv=drug.renal_clearance.cv
            ),
        )
        params_r = ResolvedParams(realized_graph, drug_r.realize_means())
        sim = solve_regimen(compiled, params_r, regimen, t_total_h=t_total, dt_output=dt_output)
        if not sim.solver_success:
            conc_rows.append(np.full(t_grid.size, np.nan))
            cmaxs.append(np.nan)
            aucs.append(np.nan)
            continue
        t_native = sim.time_h
        c_native = sim.concentrations[obs_node]
        conc_rows.append(np.interp(t_grid, t_native, c_native))
        mask = (t_native >= last - 1e-9) & (t_native <= last + tau + 1e-9)
        if mask.sum() >= 2:
            cmaxs.append(float(np.max(c_native[mask])))
            aucs.append(float(_trapz(c_native[mask], t_native[mask])))
        else:
            cmaxs.append(np.nan)
            aucs.append(np.nan)

    cmaxs_arr = np.array(cmaxs)
    if not np.isfinite(cmaxs_arr).any():
        raise ValueError(
            f"engine failed at all {n_grid} renal-scale grid points; cannot build the grid"
        )
    cmax = _fill_nan_log_s(cmaxs_arr, r_grid)
    auc = _fill_nan_log_s(np.array(aucs), r_grid)
    conc = _nearest_finite_backfill(np.array(conc_rows))
    return RenalCLGrid(r_grid=r_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc)


def sir_posterior_renal(
    prior: RenalCLPrior,
    forward: RenalCLForward,
    observations,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> PosteriorPK:
    """SIR posterior over the renal-CL scale ``r`` given observations.

    Draws ``r`` from the prior, weights by the joint observation likelihood
    (e.g. a steady-state ``MeasuredConc`` trough via ``conc_at``), resamples.
    F is degenerate (== 1) for the IV path. Reports ``n_eff``.
    """
    if rng is None:
        rng = np.random.default_rng()
    r = prior.sample(n_samples, rng)
    state = forward(r)
    loglik = np.zeros(n_samples)
    for obs in observations:
        loglik = loglik + obs.log_likelihood(state)
    idx, n_eff = _softmax_resample(loglik, rng)
    return PosteriorPK(
        f=Posterior(np.ones(idx.size)),
        cmax=Posterior(state["cmax"][idx]),
        auc=Posterior(state["auc"][idx]),
        n_eff=n_eff,
        renal_scale=Posterior(r[idx]),
    )
