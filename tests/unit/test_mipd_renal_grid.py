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
    auc = np.trapz(conc, t_grid, axis=1)
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
