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
from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
DOSE = 7.5


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
