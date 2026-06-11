"""Tests for mipd.renal_grid (IV steady-state renal-CL grid + SIR)."""
import numpy as np

from sisyphus.mipd.renal_grid import RenalCLForward, RenalCLGrid, RenalCLPrior


def _toy_grid() -> RenalCLGrid:
    # 3-point renal grid; lower r (lower renal CL) -> higher exposure.
    r_grid = np.array([0.5, 1.0, 2.0])
    t_grid = np.array([0.0, 1.0, 2.0, 3.0])
    # conc curves decreasing in r (more CL -> lower conc), decaying in t
    conc = np.array([
        [10.0, 6.0, 4.0, 2.0],   # r=0.5
        [8.0, 4.0, 2.0, 1.0],    # r=1.0
        [6.0, 2.0, 1.0, 0.5],    # r=2.0
    ])
    cmax = conc.max(axis=1)
    auc = np.trapezoid(conc, t_grid, axis=1)
    return RenalCLGrid(r_grid=r_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc)


def test_renal_prior_median_one_clipped_to_range():
    rng = np.random.default_rng(0)
    r = RenalCLPrior(cv=1.0, r_min=0.2, r_max=5.0).sample(20000, rng)
    assert abs(np.median(r) - 1.0) < 0.05
    assert r.min() >= 0.2 and r.max() <= 5.0


def test_renal_grid_conc_at_interpolates_and_guards_horizon():
    g = _toy_grid()
    # at r=1.0, t=1.0 -> exactly 4.0
    assert np.isclose(g.conc_at(np.array([1.0]), 1.0)[0], 4.0)
    import pytest
    with pytest.raises(ValueError):
        g.conc_at(np.array([1.0]), 99.0)  # beyond horizon


def test_renal_forward_interpolates_cmax_auc_monotone():
    fwd = RenalCLForward(_toy_grid())
    state = fwd(np.array([0.5, 2.0]))
    # lower r -> higher cmax and auc
    assert state["cmax"][0] > state["cmax"][1]
    assert state["auc"][0] > state["auc"][1]
    assert callable(state["conc_at"])
    assert state["conc_at"](1.0).shape == (2,)


ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def _iv_regimen():
    from sisyphus.regimen.types import DosingRegimen
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_build_renal_cl_grid_shape_and_horizon():
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    g = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=5, r_range=(0.5, 2.0))
    assert g.cmax.shape == g.auc.shape == (5,)
    assert np.all(g.r_grid[1:] > g.r_grid[:-1])
    # horizon spans last_dose (4*8=32h) + max(interval 8, 24) = 56h
    assert g.t_grid[-1] >= 56.0 - 1e-6
    import pytest
    with pytest.raises(ValueError):
        g.conc_at(np.array([1.0]), 999.0)


def test_build_renal_cl_grid_lower_r_higher_exposure():
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    g = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=5, r_range=(0.5, 2.0))
    # ascending r -> descending steady-state AUC (more renal CL -> less exposure)
    assert g.auc[0] > g.auc[-1]


def test_build_renal_cl_grid_faithful_to_solve_regimen():
    # The grid's per-r curve must equal a direct solve_regimen at that r.
    from sisyphus.mipd.grid import _build_grid_engine
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.regimen.solver import solve_regimen

    reg = _iv_regimen()
    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=3, r_range=(1.0, 1.0))  # single r=1
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        ATENOLOL, reg.events[0].dose_mg, "iv", 1.0, "rodgers_rowland"
    )
    params = ResolvedParams(realized_graph, drug.realize_means())
    sim = solve_regimen(compiled, params, reg, t_total_h=float(g.t_grid[-1]))
    direct = np.interp(g.t_grid, sim.time_h, sim.concentrations[obs_node])
    assert np.allclose(g.conc[0], direct, rtol=1e-6, atol=1e-9)


def test_sir_posterior_renal_trough_moves_r_correctly():
    # Higher trough than the r=1 curve predicts -> patient clears SLOWER -> r < 1.
    from sisyphus.mipd.clgrid import MeasuredConc
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        sir_posterior_renal,
    )
    grid = _toy_grid()  # at r=1.0, conc_at(t=2.0) == 2.0
    fwd = RenalCLForward(grid)
    rng = np.random.default_rng(0)
    # measured trough HIGHER than the r=1 prediction (4.0 vs 2.0) -> slower CL -> r<1
    high = sir_posterior_renal(
        RenalCLPrior(cv=1.0, r_min=0.5, r_max=2.0), fwd,
        [MeasuredConc(value=4.0, t=2.0, cv=0.1)], n_samples=20000, rng=rng,
    )
    assert high.renal_scale.point < 1.0
    assert high.cmax.point > grid.cmax[1]  # slower CL -> higher exposure than r=1
    assert high.n_eff > 100


def test_sir_posterior_renal_iv_has_degenerate_f():
    from sisyphus.mipd.clgrid import MeasuredConc
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        sir_posterior_renal,
    )
    post = sir_posterior_renal(
        RenalCLPrior(r_min=0.5, r_max=2.0), RenalCLForward(_toy_grid()),
        [MeasuredConc(value=3.0, t=1.0, cv=0.2)], n_samples=5000,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(post.f.samples, 1.0)  # F == 1 for IV
    assert post.renal_scale is not None
