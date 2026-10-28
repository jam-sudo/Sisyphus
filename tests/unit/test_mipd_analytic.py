"""Known-parameter checks for the opt-in structural MIPD forward and posterior."""
import numpy as np
import pytest

from sisyphus.mipd.analytic import (
    AnalyticRegimenForward,
    StructuralPrior,
    engine_structural_prior,
    sir_structural_posterior,
)
from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.core import MeasuredF
from sisyphus.mipd.covariates import Covariates
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"


def _scalar_conc(forward, t, cl=5.0, v=25.0, f=1.0, ka=None):
    absorption = np.array([ka]) if ka is not None else None
    return float(forward.concentration(
        t, np.array([cl]), np.array([v]), np.array([f]), absorption
    )[0])


def test_iv_bolus_and_infusion_forward_are_linear_in_dose():
    bolus = DosingRegimen.iv_infusion(100, 0, 12, 1)
    infusion = DosingRegimen.iv_infusion(100, 1, 12, 1)
    assert _scalar_conc(AnalyticRegimenForward(bolus), 0) == pytest.approx(4.0)
    at_end = _scalar_conc(AnalyticRegimenForward(infusion), 1)
    expected = 100 / 5 * (1 - np.exp(-5 / 25))
    assert at_end == pytest.approx(expected)
    doubled = DosingRegimen.iv_infusion(200, 1, 12, 1)
    assert _scalar_conc(AnalyticRegimenForward(doubled), 1) == pytest.approx(2 * at_end)


def test_structural_posterior_recovers_iv_cl_and_v_from_peak_and_trough():
    regimen = DosingRegimen.iv_infusion(100, 0, 8, 10)
    forward = AnalyticRegimenForward(regimen)
    last = regimen.last_dose_time_h
    peak = MeasuredConc(_scalar_conc(forward, last + 0.1), last + 0.1, cv=0.05)
    trough = MeasuredConc(_scalar_conc(forward, last + 8), last + 8, cv=0.05)
    prior = StructuralPrior(cl_l_h=8, volume_l=40, renal_cl_l_h=0.5)
    both = sir_structural_posterior(
        prior, regimen, (peak, trough), n_samples=20000, rng=np.random.default_rng(0)
    )
    trough_only = sir_structural_posterior(
        prior, regimen, (trough,), n_samples=20000, rng=np.random.default_rng(0)
    )
    assert both.total_clearance_l_h.point == pytest.approx(5.0, rel=0.12)
    assert both.volume_l.point == pytest.approx(25.0, rel=0.15)
    assert both.cl_scale.point < 1.0  # IV non-renal clearance is the freed CL axis
    assert trough_only.volume_l.point == pytest.approx(40.0, rel=0.2)
    assert both.cmax_90ci is None


def test_oral_absorption_rate_updates_from_curve_shape():
    regimen = DosingRegimen.oral_repeated(100, 12, 8)
    forward = AnalyticRegimenForward(regimen)
    last = regimen.last_dose_time_h
    observations = [
        MeasuredConc(
            _scalar_conc(forward, last + phase, cl=5, v=25, f=0.7, ka=0.6),
            last + phase, cv=0.1,
        ) for phase in (1, 3, 8, 12)
    ] + [MeasuredF(0.7, cv=0.05)]
    post = sir_structural_posterior(
        StructuralPrior(6, 30, f=0.6, ka_per_h=1.0), regimen, observations,
        n_samples=30000, rng=np.random.default_rng(0),
    )
    assert post.absorption_rate_per_h.point == pytest.approx(0.6, rel=0.4)
    assert post.total_clearance_l_h.point == pytest.approx(5.0, rel=0.2)


def test_structural_zero_observations_are_exact_prior_prediction():
    regimen = DosingRegimen.iv_infusion(100, 0, 12, 3)
    prior = StructuralPrior(5, 25)
    post = sir_structural_posterior(prior, regimen, (), n_samples=100)
    forward = AnalyticRegimenForward(regimen)
    state = forward.state(np.array([5.0]), np.array([25.0]), np.array([1.0]))
    assert post.cmax.point == state["cmax"][0]
    assert post.auc.point == pytest.approx(state["auc"][0], rel=1e-14)
    assert post.volume_l.ci90 == (25.0, 25.0)


def test_crcl_cv_adds_independent_lognormal_latent():
    regimen = DosingRegimen.iv_infusion(100, 0, 12, 3)
    prior = StructuralPrior(5, 25, renal_cl_l_h=5)
    obs = [MeasuredConc(1.0, 36.0)]
    off = sir_structural_posterior(prior, regimen, obs, n_samples=3000)
    on = sir_structural_posterior(prior, regimen, obs, n_samples=3000, crcl_cv=0.3)
    assert off.crcl_scale is None
    assert on.crcl_scale is not None
    assert on.crcl_scale.ci90[1] > on.crcl_scale.ci90[0]


def test_engine_structural_prior_and_no_observation_tdm_invariant():
    regimen = DosingRegimen.iv_infusion(50, 0.5, 8, 5)
    prior, _ = engine_structural_prior(ATENOLOL, regimen)
    assert prior.cl_l_h > 0 and prior.volume_l > 0
    base = predict_tdm(ATENOLOL, regimen, (), n_grid=3, n_samples=500, seed=0)
    structural = predict_tdm(
        ATENOLOL, regimen, (), structural_latents=True, n_grid=3, n_samples=500, seed=0
    )
    assert np.array_equal(base.cmax.samples, structural.cmax.samples)
    assert np.array_equal(base.auc.samples, structural.auc.samples)


def test_measured_crcl_cv_enables_latent_in_tdm():
    regimen = DosingRegimen.iv_infusion(50, 0.5, 8, 5)
    post = predict_tdm(
        ATENOLOL, regimen, [MeasuredConc(1.0, 36.0)],
        covariates=Covariates(crcl_ml_min=80, crcl_cv=0.2), n_samples=500,
    )
    assert post.crcl_scale is not None
    assert post.crcl_scale.ci90[1] > post.crcl_scale.ci90[0]
