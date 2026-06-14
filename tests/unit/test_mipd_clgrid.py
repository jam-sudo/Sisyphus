"""Tests for the CL-grid surrogate forward + CL latent + MeasuredConc (pure numpy).

A CL (clint-scale) latent changes the curve *shape* (not a vertical scale), so the
forward precomputes the engine response over a clint-scale grid and interpolates.
These tests exercise the 2-latent (F, CL) mechanism against a *synthetic* grid with
known analytics — no engine — so they are fast and exact.
"""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import (
    CLGrid,
    CLGridForward,
    CLPrior,
    MeasuredConc,
    sir_posterior_2d,
)
from sisyphus.mipd.core import FPrior, MeasuredAUC, PosteriorPK

# Synthetic grid: f_engine = 0.5 (const), cmax(s) = 1/s, auc(s) = 10/s,
# conc(t; s) = cmax(s) * exp(-s * t)  (peaks at t=0 == cmax, decays faster for high s).
_S = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
_T = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
_CMAX = 1.0 / _S
_AUC = 10.0 / _S
_FENG = np.full_like(_S, 0.5)
_CONC = np.array([_CMAX[g] * np.exp(-_S[g] * _T) for g in range(len(_S))])
GRID = CLGrid(s_grid=_S, t_grid=_T, conc=_CONC, cmax=_CMAX, auc=_AUC, f_engine=_FENG)


def test_grid_conc_at_returns_curve_value_at_grid_point():
    # at s=1.0 (grid point), conc at t=0 is cmax(1)=1.0; at t=1 it is exp(-1).
    assert GRID.conc_at(np.array([1.0]), 0.0)[0] == pytest.approx(1.0, rel=1e-6)
    assert GRID.conc_at(np.array([1.0]), 1.0)[0] == pytest.approx(np.exp(-1.0), rel=1e-3)


def test_forward_at_f_engine_reproduces_grid_cmax_auc():
    fwd = CLGridForward(GRID)
    state = fwd(np.array([0.5]), np.array([1.0]))  # F == f_engine, s == 1 -> scale 1
    assert state["cmax"][0] == pytest.approx(1.0, rel=1e-6)
    assert state["auc"][0] == pytest.approx(10.0, rel=1e-6)
    assert state["conc_at"](0.0)[0] == pytest.approx(1.0, rel=1e-6)


def test_forward_F_is_vertical_scale_and_CL_sets_shape():
    fwd = CLGridForward(GRID)
    # doubling F doubles exposure (vertical); cmax(s) at s=2 is 0.5 (shape via CL).
    assert fwd(np.array([1.0]), np.array([1.0]))["cmax"][0] == pytest.approx(2.0, rel=1e-6)
    assert fwd(np.array([0.5]), np.array([2.0]))["cmax"][0] == pytest.approx(0.5, rel=1e-6)
    assert fwd(np.array([0.5]), np.array([2.0]))["auc"][0] == pytest.approx(5.0, rel=1e-6)


def test_measured_conc_loglik_peaks_where_forward_conc_matches():
    fwd = CLGridForward(GRID)
    # three (F,s) candidates; pick the one whose conc(t=1) matches the obs.
    f = np.array([0.5, 0.5, 0.5])
    s = np.array([1.0, 2.0, 4.0])
    state = fwd(f, s)
    target = state["conc_at"](1.0)[1]  # the s=2 candidate's conc at t=1
    ll = MeasuredConc(target, t=1.0, cv=0.05).log_likelihood(state)
    assert np.argmax(ll) == 1


def test_clprior_samples_in_grid_range_centered_at_one():
    s = CLPrior(cv=1.0, s_min=_S[0], s_max=_S[-1]).sample(20000, np.random.default_rng(0))
    assert s.min() >= _S[0]
    assert s.max() <= _S[-1]
    assert np.median(s) == pytest.approx(1.0, abs=0.15)


def test_sir_2d_with_measured_auc_concentrates_cl_scale():
    rng = np.random.default_rng(0)
    # auc = (F/0.5)*10/s; with F~0.5, obs AUC=5 -> s≈2.
    post = sir_posterior_2d(
        FPrior(0.5, 0.5),
        CLPrior(cv=1.0, s_min=_S[0], s_max=_S[-1]),
        CLGridForward(GRID),
        [MeasuredAUC(5.0, cv=0.05)],
        n_samples=40000,
        rng=rng,
    )
    assert isinstance(post, PosteriorPK)
    assert post.cl_scale is not None
    assert post.cl_scale.point == pytest.approx(2.0, rel=0.20)
    assert post.auc.point == pytest.approx(5.0, rel=0.20)


def test_conc_at_raises_when_time_is_beyond_the_grid():
    # the synthetic grid maxes at t=12.0 h; a later trough must not edge-clamp silently.
    with pytest.raises(ValueError, match="outside the engine grid"):
        GRID.conc_at(np.array([1.0]), 20.0)


def test_measured_conc_raises_when_time_is_beyond_the_grid():
    fwd = CLGridForward(GRID)
    state = fwd(np.array([0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="outside the engine grid"):
        MeasuredConc(0.01, t=20.0).log_likelihood(state)


def test_conc_at_allows_time_at_the_grid_boundary():
    # t exactly at the last grid node is in-bounds (no spurious raise).
    assert np.isfinite(GRID.conc_at(np.array([1.0]), 12.0)[0])


def test_sir_2d_with_measured_conc_constrains_the_posterior():
    rng = np.random.default_rng(0)
    fwd = CLGridForward(GRID)
    # truth at (F=0.5, s=2): conc(t=2) = cmax(2)*exp(-2*2) = 0.5*exp(-4).
    truth = 0.5 * np.exp(-4.0)
    post = sir_posterior_2d(
        FPrior(0.5, 0.5),
        CLPrior(cv=1.0, s_min=_S[0], s_max=_S[-1]),
        fwd,
        [MeasuredConc(truth, t=2.0, cv=0.05)],
        n_samples=40000,
        rng=rng,
    )
    # the conc anchor should pull cl_scale toward 2 (and away from the prior median 1).
    assert post.cl_scale.point == pytest.approx(2.0, rel=0.35)
