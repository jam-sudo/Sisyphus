"""Tests for build_oral_cl_grid: oral steady-state clint-scale grid.

Load-bearing tests: LTI exactness (dose-linearity), grid faithfulness at s~=1,
n_grid=1 single-slice, and the all-fail -> ValueError path.
"""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import CLGrid, CLGridForward
from sisyphus.mipd.oral_grid import build_oral_cl_grid
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine — in-domain, fast


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def test_returns_tuple_grid_bool_thalf():
    grid, is_ss, t_half = build_oral_cl_grid(SMILES, _reg(), n_grid=5)
    assert isinstance(grid, CLGrid)
    assert isinstance(is_ss, bool)
    assert t_half is None or t_half > 0.0
    assert grid.f_engine.shape == grid.s_grid.shape
    assert np.all(grid.f_engine > 0) and np.all(grid.f_engine <= 1.0)


def test_lti_exactness_dose_linear():
    g1, _, _ = build_oral_cl_grid(SMILES, _reg(dose=100.0), n_grid=5)
    g2, _, _ = build_oral_cl_grid(SMILES, _reg(dose=200.0), n_grid=5)
    np.testing.assert_allclose(g2.cmax, 2.0 * g1.cmax, rtol=1e-6)
    np.testing.assert_allclose(g2.auc, 2.0 * g1.auc, rtol=1e-6)


def test_n_grid_1_s1_slice():
    grid, _, _ = build_oral_cl_grid(SMILES, _reg(), n_grid=1, s_range=(1.0, 1.0))
    assert grid.s_grid.shape == (1,)
    fwd = CLGridForward(grid)
    state = fwd(np.array([grid.f_engine[0]]), np.ones(1))
    last = float(_reg().last_dose_time_h)
    assert state["conc_at"](last + 12.0) > 0.0


def test_grid_faithful_at_s1_matches_direct_solve():
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.mipd.grid import _build_grid_engine
    from sisyphus.regimen.solver import solve_regimen
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=3, s_range=(0.5, 2.0))
    i1 = int(np.argmin(np.abs(np.log(grid.s_grid))))
    compiled, rgraph, drug, obs = _build_grid_engine(
        SMILES, reg.events[0].dose_mg, "oral", 1.0, "rodgers_rowland", None, None
    )
    sim = solve_regimen(compiled, ResolvedParams(rgraph, drug.realize_means()), reg,
                        t_total_h=float(reg.last_dose_time_h) + 24.0, dt_output=0.1)
    last = float(reg.last_dose_time_h)
    m = (sim.time_h >= last - 1e-9) & (sim.time_h <= last + 12.0 + 1e-9)
    direct_cmax = float(np.max(sim.concentrations[obs][m]))
    assert grid.cmax[i1] == pytest.approx(direct_cmax, rel=0.05)


def test_all_grid_points_fail_raises():
    # An unparseable SMILES makes the engine build fail at every grid point, so no
    # grid can be assembled -> ValueError (the all-fail guard's intent).
    with pytest.raises(ValueError):
        build_oral_cl_grid("not_a_valid_smiles", _reg(), n_grid=3)
