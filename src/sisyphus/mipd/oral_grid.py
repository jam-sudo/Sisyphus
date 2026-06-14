"""Oral steady-state clint-scale grid for the engine-as-prior TDM stack.

Composes build_cl_grid (metabolic-s scaling + single-dose f_engine) and
build_renal_cl_grid (multi-dose SS final-interval extraction). Per grid point s:
(1) single-dose oral solve -> oral AUC + endpoints; (2) single-dose IV reference inside
_engine_oral_bioavailability -> f_engine; (3) multi-dose solve_regimen -> SS cmax/auc/conc
on the final interval. Returns the existing CLGrid plus the SS flag and terminal t1/2
(computed at s~=1), so no grid type changes.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

from sisyphus.mipd.clgrid import CLGrid

logger = logging.getLogger(__name__)


def build_oral_cl_grid(
    smiles: str,
    regimen,
    *,
    n_grid: int = 13,
    s_range: tuple[float, float] = (0.05, 20.0),
    renal_factor: float = 1.0,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
    kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> tuple[CLGrid, bool, float | None]:
    """Build the oral SS clint grid. Returns ``(grid, is_steady_state, t_half_h)``."""
    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.mipd._regimen import _regimen_interval_h
    from sisyphus.mipd.grid import _build_grid_engine, _fill_nan_log_s, _nearest_finite_backfill
    from sisyphus.mipd.renal_grid import _trapz
    from sisyphus.pipeline.predict import _engine_oral_bioavailability
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.regimen.profile import compute_steady_state_metrics
    from sisyphus.regimen.solver import solve_regimen

    dose_mg = float(regimen.events[0].dose_mg)
    s_grid = np.geomspace(s_range[0], s_range[1], n_grid)
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, dose_mg, "oral", renal_factor, kp_method, body_weight_kg, age_years
    )
    admin_idx = compiled.state_index[drug.administration_node]

    last = float(regimen.last_dose_time_h)
    tau = _regimen_interval_h(regimen)
    t_total = last + max(tau, 24.0)
    n_points = max(2, int(round(t_total / dt_output)) + 1)
    t_grid = np.linspace(0.0, t_total, n_points)
    i1 = int(np.argmin(np.abs(np.log(s_grid))))

    conc_rows: list[np.ndarray] = []
    cmaxs: list[float] = []
    aucs: list[float] = []
    fengs: list[float] = []
    is_ss = False
    t_half_h: float | None = None

    for gi, s in enumerate(s_grid):
        # Scale METABOLIC intrinsic clearance only (enzyme_affinity: CYP/UGT/NAT);
        # renal/biliary terms are held fixed, so ``s`` is a metabolic-CL scale.
        drug_s = dataclasses.replace(
            drug,
            enzyme_affinity={
                k: Distribution(mean=v.mean * float(s), cv=v.cv)
                for k, v in drug.enzyme_affinity.items()
            },
        )
        realized_drug_s = drug_s.realize_means()
        params_s = ResolvedParams(realized_graph, realized_drug_s)

        # (1)+(2) single-dose oral solve -> oral AUC, endpoints, f_engine.
        y0 = np.zeros(compiled.n_states)
        y0[admin_idx] = dose_mg
        sim_sd = solve(compiled, params_s, y0, t_span=(0, 24), t_min_h=0.0)
        feng = np.nan
        if sim_sd.solver_success:
            pk_sd = compute_endpoints(sim_sd, observation_node=obs_node, t_min_h=0.0)
            feng = _engine_oral_bioavailability(
                compiled, params_s, realized_drug_s, pk_sd.auc_0t.mean, obs_node
            )
            feng = feng if (feng is not None and feng > 0) else np.nan
            if gi == i1 and pk_sd.t_half is not None:
                t_half_h = float(pk_sd.t_half.mean)
        fengs.append(feng)

        # (3) multi-dose SS solve -> final-interval cmax/auc/conc.
        sim_ss = solve_regimen(compiled, params_s, regimen, t_total_h=t_total, dt_output=dt_output)
        if not sim_ss.solver_success:
            conc_rows.append(np.full(t_grid.size, np.nan))
            cmaxs.append(np.nan)
            aucs.append(np.nan)
            continue
        t_native = sim_ss.time_h
        c_native = sim_ss.concentrations[obs_node]
        conc_rows.append(np.interp(t_grid, t_native, c_native))
        mask = (t_native >= last - 1e-9) & (t_native <= last + tau + 1e-9)
        if mask.sum() >= 2:
            cmaxs.append(float(np.max(c_native[mask])))
            aucs.append(float(_trapz(c_native[mask], t_native[mask])))
        else:
            cmaxs.append(np.nan)
            aucs.append(np.nan)
        if gi == i1:
            try:
                is_ss = bool(
                    compute_steady_state_metrics(sim_ss, regimen, node=obs_node).is_steady_state
                )
            except ValueError:
                is_ss = False

    cmaxs_arr = np.array(cmaxs)
    fengs_arr = np.array(fengs)
    if not np.isfinite(cmaxs_arr).any():
        raise ValueError(
            f"engine failed at all {n_grid} oral clint-scale grid points; cannot build grid"
        )
    if not np.isfinite(fengs_arr).any():
        raise ValueError(
            f"engine produced no valid oral bioavailability at any of {n_grid} grid points"
        )
    cmax = _fill_nan_log_s(cmaxs_arr, s_grid)
    auc = _fill_nan_log_s(np.array(aucs), s_grid)
    f_engine = np.clip(_fill_nan_log_s(fengs_arr, s_grid), 1e-4, 1.0)
    conc = _nearest_finite_backfill(np.array(conc_rows))
    grid = CLGrid(s_grid=s_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc, f_engine=f_engine)
    return grid, is_ss, t_half_h
