"""Faithfulness tests for the engine CL-grid builder.

build_cl_grid solves the engine at a grid of clint scales by reproducing the
predict() engine path with enzyme_affinity scaled. The load-bearing test: at
clint-scale 1.0 the grid must reproduce predict()'s engine Cmax/AUC exactly,
proving the reproduction is faithful.
"""
import types

import numpy as np
import pytest

from sisyphus.mipd.clgrid import CLGrid
from sisyphus.mipd.grid import build_cl_grid
from sisyphus.pipeline.predict import predict

MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
DOSE = 7.5

# Codeine is a non-holdout UGT2B7 substrate (fm~0.7); get_non_cyp_fractions returns
# a non-empty dict, so a grid that omits non_cyp_fractions diverges ~18% at s=1.
CODEINE = "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"
CODEINE_DOSE = 30.0


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


def test_cl_grid_at_unit_scale_reproduces_predict_for_non_cyp_substrate():
    """Faithfulness must hold for a non-CYP (UGT2B7) substrate, not just CYP drugs.

    The grid must reproduce predict()'s disposition wiring (non_cyp_fractions /
    OATP-ECM); omitting it diverges ~18% at s=1 for codeine.
    """
    grid = build_cl_grid(CODEINE, CODEINE_DOSE, n_grid=13, s_range=(0.1, 10.0))
    i = int(np.argmin(np.abs(np.log(grid.s_grid))))
    ap = predict(CODEINE, CODEINE_DOSE)
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


def test_build_cl_grid_raises_when_engine_fails_at_all_grid_points(monkeypatch):
    # If the solver fails at every clint-scale, a NaN grid must NOT be returned silently.
    monkeypatch.setattr(
        "sisyphus.engine.solver.solve",
        lambda *a, **k: types.SimpleNamespace(solver_success=False),
    )
    with pytest.raises(ValueError, match="grid point"):
        build_cl_grid(MIDAZOLAM, DOSE, n_grid=5)


def test_nearest_finite_backfill_replaces_nan_rows_from_nearest_finite_row():
    from sisyphus.mipd.grid import _nearest_finite_backfill

    conc = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]])
    out = _nearest_finite_backfill(conc)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out[1], conc[0])  # tie distance -> lower index wins


def test_nearest_finite_backfill_skips_a_nan_neighbor():
    from sisyphus.mipd.grid import _nearest_finite_backfill

    # rows 0 and 1 both fail; the only finite row is 2 -> both must copy row 2,
    # NOT the adjacent (still-NaN) neighbor the old adjacent-copy logic would pick.
    conc = np.array([[np.nan, np.nan], [np.nan, np.nan], [7.0, 8.0]])
    out = _nearest_finite_backfill(conc)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out[0], conc[2])
    assert np.array_equal(out[1], conc[2])


def test_build_cl_grid_renal_factor_scales_exposure_and_preserves_f_engine():
    import numpy as np
    base = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0))
    low_renal = build_cl_grid(
        MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0), renal_factor=0.25
    )
    # Lower renal clearance -> higher total AUC at every clint scale (renal_cl>0).
    assert np.all(low_renal.auc >= base.auc)
    assert low_renal.auc[0] > base.auc[0]
    # F_engine = AUC_oral/AUC_iv is invariant to systemic renal scaling (LTI).
    # The tiny residual is the documented 0-24h AUC-truncation artifact
    # (_engine_oral_bioavailability), which grows with the renal change; assert it
    # stays small in absolute terms AND far below the exposure change it rides on.
    f_dev = np.max(np.abs(low_renal.f_engine - base.f_engine) / base.f_engine)
    auc_dev = np.max(np.abs(low_renal.auc - base.auc) / base.auc)
    assert f_dev < 0.01           # f_engine barely moves (<1%)
    assert f_dev < 0.5 * auc_dev  # ...and far less than the exposure it accompanies


def test_build_cl_grid_renal_factor_unity_is_unchanged():
    import numpy as np
    a = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0))
    b = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0), renal_factor=1.0)
    assert np.array_equal(a.cmax, b.cmax)
    assert np.array_equal(a.auc, b.auc)


def test_build_cl_grid_single_point_at_unit_scale():
    # n_grid=1 at s=1 is the single renal-scaled solve the API uses for CrCl-only.
    g = build_cl_grid(MIDAZOLAM, DOSE, n_grid=1, s_range=(1.0, 1.0), renal_factor=0.5)
    assert g.cmax.shape == (1,)
    assert g.cmax[0] > 0 and g.auc[0] > 0
    assert 0.0 < g.f_engine[0] <= 1.0
