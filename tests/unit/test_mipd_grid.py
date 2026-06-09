"""Faithfulness tests for the engine CL-grid builder.

build_cl_grid solves the engine at a grid of clint scales by reproducing the
predict() engine path with enzyme_affinity scaled. The load-bearing test: at
clint-scale 1.0 the grid must reproduce predict()'s engine Cmax/AUC exactly,
proving the reproduction is faithful.
"""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import CLGrid
from sisyphus.mipd.grid import build_cl_grid
from sisyphus.pipeline.predict import predict

MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
DOSE = 7.5


def test_build_cl_grid_returns_well_formed_grid():
    grid = build_cl_grid(MIDAZOLAM, DOSE, n_grid=9, s_range=(0.1, 10.0))
    assert isinstance(grid, CLGrid)
    assert grid.s_grid.shape == (9,)
    assert grid.conc.shape == (9, grid.t_grid.size)
    assert grid.cmax.shape == grid.auc.shape == grid.f_engine.shape == (9,)
    assert np.all(grid.s_grid[1:] > grid.s_grid[:-1])  # ascending
    assert np.all(grid.f_engine > 0) and np.all(grid.f_engine <= 1.0 + 1e-9)


def test_cl_grid_at_unit_scale_reproduces_predict_engine_pk():
    grid = build_cl_grid(MIDAZOLAM, DOSE, n_grid=13, s_range=(0.1, 10.0))
    i = int(np.argmin(np.abs(np.log(grid.s_grid))))
    assert grid.s_grid[i] == pytest.approx(1.0, rel=1e-9)  # geomspace includes 1.0
    ap = predict(MIDAZOLAM, DOSE)
    assert grid.cmax[i] == pytest.approx(ap.engine_pk.cmax.mean, rel=0.03)
    assert grid.auc[i] == pytest.approx(ap.engine_pk.auc_0t.mean, rel=0.03)


def test_cl_grid_higher_clint_lowers_exposure():
    """For a high-extraction drug, more clint -> more first-pass -> lower Cmax/AUC."""
    grid = build_cl_grid(MIDAZOLAM, DOSE, n_grid=9, s_range=(0.1, 10.0))
    assert grid.cmax[0] > grid.cmax[-1]  # low clint -> high Cmax
    assert grid.auc[0] > grid.auc[-1]


def test_predict_posterior_cl_latent_with_measured_conc_is_well_formed():
    from sisyphus.mipd.api import predict_posterior
    from sisyphus.mipd.clgrid import MeasuredConc

    post = predict_posterior(
        MIDAZOLAM, DOSE, [MeasuredConc(0.02, t=2.0, cv=0.25)], n_grid=9, seed=0
    )
    assert post.cl_scale is not None  # the 2-latent path ran
    assert post.meta_cmax is not None  # product posterior attached
    assert post.cmax_90ci is not None  # calibrated PI attached
    assert post.cmax.point > 0


def test_predict_posterior_cl_latent_no_obs_returns_prior_centered_posterior():
    from sisyphus.mipd.api import predict_posterior

    post = predict_posterior(MIDAZOLAM, DOSE, cl_latent=True, n_grid=9, seed=0)
    assert post.cl_scale is not None
    # no observations -> posterior == prior, clint-scale centered on 1.0
    assert post.cl_scale.point == pytest.approx(1.0, abs=0.4)
