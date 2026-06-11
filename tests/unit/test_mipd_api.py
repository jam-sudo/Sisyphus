"""Integration tests for the MIPD API and amortizer (call the real engine).

The load-bearing test is ``test_..._reproduces_routing_cmax``: conditioning the
engine-as-prior on a MeasuredF observation must reproduce the empirically
validated measured-F routing Cmax (Gate 0b/0c) — i.e. the SIR core generalizes
the shipped, validated mechanism into a full posterior.
"""
import dataclasses

import numpy as np
import pytest

import sisyphus.pipeline.predict as pp
from sisyphus.mipd import MeasuredF, PosteriorPK
from sisyphus.mipd.amortizer import NeuralAmortizer, SIRAmortizer
from sisyphus.mipd.api import predict_posterior
from sisyphus.mipd.core import APrioriPK
from sisyphus.mipd.covariates import Covariates
from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
DOSE = 7.5
ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def test_predict_posterior_no_observations_matches_apriori_engine_cmax():
    post = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    assert isinstance(post, PosteriorPK)
    ap_cmax = predict(MIDAZOLAM, DOSE).engine_pk.cmax.mean
    assert post.cmax.point == pytest.approx(ap_cmax, rel=0.20)


def test_predict_posterior_measured_F_reproduces_routing_cmax():
    f = 0.30
    post = predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(f, cv=0.10)], seed=0)
    routed = predict(
        MIDAZOLAM, DOSE, measured_adme=MeasuredADMEInput(f_bioavail=f)
    ).engine_pk.cmax.mean
    assert post.cmax.point == pytest.approx(routed, rel=0.20)


def test_predict_posterior_observation_narrows_cmax_interval():
    wide = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    tight = predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.30, cv=0.10)], seed=0)
    wlo, whi = wide.cmax.ci90
    tlo, thi = tight.cmax.ci90
    assert (thi - tlo) < (whi - wlo)


def test_sir_amortizer_reproduces_sir_core_posterior():
    ap = APrioriPK(cmax0=1.0, auc0=10.0, f_engine=0.5)
    post = SIRAmortizer(prior_cv=1.0, n_samples=20000).posterior(
        ap, [MeasuredF(0.2, cv=0.05)], rng=np.random.default_rng(0)
    )
    assert post.f.point == pytest.approx(0.2, abs=0.03)
    assert post.cmax.point == pytest.approx(0.4, rel=0.15)


def test_neural_amortizer_raises_informative_error_without_torch():
    with pytest.raises(NotImplementedError, match="torch"):
        NeuralAmortizer()


def test_f_only_path_calls_predict_once_with_compute_f_engine(monkeypatch):
    # The F-only path must read engine_f off the single a-priori predict() call —
    # no second 'probe' predict() (the removed warning-regex hack, review #10).
    calls = []
    real = pp.predict

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(pp, "predict", spy)
    predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.3)], seed=0)
    assert len(calls) == 1
    assert calls[0].get("compute_f_engine") is True


def test_f_only_path_raises_when_engine_f_unavailable(monkeypatch):
    # If the engine F-reference solve is unavailable (engine_f None), the F-only
    # path must refuse rather than fabricate an F_engine (relocated review #2 guard).
    good = predict(MIDAZOLAM, DOSE, compute_f_engine=True)
    broken = dataclasses.replace(good, engine_f=None)
    monkeypatch.setattr(pp, "predict", lambda *a, **k: broken)
    with pytest.raises(ValueError, match="F-reference solve unavailable"):
        predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.3)], seed=0)


def test_predict_posterior_rejects_non_oral_route_on_f_only_path():
    with pytest.raises(ValueError, match="oral"):
        predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.3)], route="iv", seed=0)


def test_predict_posterior_rejects_non_oral_route_on_cl_grid_path():
    with pytest.raises(ValueError, match="oral"):
        predict_posterior(MIDAZOLAM, DOSE, cl_latent=True, route="iv", n_grid=5, seed=0)


def test_covariate_none_is_bit_identical_to_no_covariate():
    a = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    b = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(), seed=0)
    assert np.array_equal(a.cmax.samples, b.cmax.samples)


def test_crcl_only_uses_single_solve_not_2latent_grid():
    post = predict_posterior(
        MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=25), seed=0
    )
    assert post.cl_scale is None  # clint latent NOT freed (no MeasuredConc)
    assert post.cmax.point > 0


def test_renal_impairment_raises_exposure_for_renal_drug():
    healthy = predict_posterior(ATENOLOL, DOSE, covariates=Covariates(crcl_ml_min=120), seed=0)
    impaired = predict_posterior(ATENOLOL, DOSE, covariates=Covariates(crcl_ml_min=20), seed=0)
    assert impaired.auc.point > healthy.auc.point
    # Document the expected magnitude (engine attributes a modest renal fraction to
    # atenolol): a 6x CrCl drop raises AUC by >3%. Guards a near-off renal regression.
    assert impaired.auc.point / healthy.auc.point > 1.03


def test_extreme_crcl_emits_warning_normal_does_not():
    low = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=3), seed=0)
    assert any("crcl" in w.lower() for w in low.warnings)
    normal = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=90), seed=0)
    assert normal.warnings == ()


def test_renal_individualization_larger_for_high_renal_fraction_drug():
    """The CrCl effect on AUC is larger for a renally-cleared drug (atenolol)
    than for a hepatically-cleared one (midazolam), since renal_cl = 7.5*fup is a
    larger fraction of total CL when fup is high and metabolism is low."""
    def auc_ratio(smiles):
        hi = predict_posterior(smiles, DOSE, covariates=Covariates(crcl_ml_min=120), seed=0)
        lo = predict_posterior(smiles, DOSE, covariates=Covariates(crcl_ml_min=20), seed=0)
        return lo.auc.point / hi.auc.point

    assert auc_ratio(ATENOLOL) > auc_ratio(MIDAZOLAM)
    assert auc_ratio(ATENOLOL) > 1.0  # impairment raises exposure for the renal drug


def test_predict_posterior_weight_age_routes_through_individualized_solve():
    # weight (no obs, no CrCl) must reach the engine via the individualized solve,
    # not the reference F-only path; the clint latent stays fixed.
    post = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=10, age_years=2), seed=0)
    assert post.cl_scale is None
    assert post.cmax.point > 0


def test_predict_posterior_lower_weight_higher_cmax():
    # lower body weight -> smaller distribution volume -> higher Cmax (the clean oral
    # weight signal; AUC is confounded by first-pass F for high-extraction drugs).
    ref = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    light = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=40), seed=0)
    assert light.cmax.point > ref.cmax.point


def test_predict_posterior_empty_covariates_bit_identical():
    a = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    b = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(), seed=0)
    assert np.array_equal(a.cmax.samples, b.cmax.samples)


def test_predict_posterior_extreme_weight_warns():
    post = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=1.0), seed=0)
    assert any("weight" in w.lower() for w in post.warnings)
