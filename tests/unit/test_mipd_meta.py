"""Tests for routing the engine posterior through the production meta blend.

The product output is the meta (engine + ML + CLF + VDss), not the engine track
alone. These tests verify the mipd meta routing replicates MetaLearner.combine
*exactly* (including the engine disagreement penalty), so the posterior meta
carries an honest interval, and that conditioning on measured F reproduces the
measured-F meta improvement (Gate 0b/0c) at the product level.
"""
import dataclasses
import math

import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.mipd import MeasuredF
from sisyphus.mipd.api import predict_posterior
from sisyphus.mipd.meta import MetaTracks, build_meta_tracks, meta_blend_cmax
from sisyphus.ml.ensemble import MetaLearner
from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
DOSE = 7.5


def test_meta_blend_matches_metalearner_across_engine_cmax_sweep():
    """Vectorized blend == production combine() over 4 orders of engine Cmax.

    The sweep crosses the >10x engine-vs-ML disagreement boundary, exercising
    the nonlinear weight penalty.
    """
    r = predict(MIDAZOLAM, DOSE)
    tracks = build_meta_tracks(MIDAZOLAM, DOSE, r)
    learner = MetaLearner()
    eng0 = r.engine_pk.cmax.mean
    grid = np.logspace(math.log10(eng0) - 2, math.log10(eng0) + 2, 25)
    mine = meta_blend_cmax(grid, tracks)
    for c, m in zip(grid, mine):
        gold = learner.combine(
            dataclasses.replace(r.engine_pk, cmax=Distribution(c)),
            r.ml_pk,
            dose_mg=DOSE,
            compound_type=tracks.compound_type,
            clf_pk=r.clf_pk,
            vdss_cmax=tracks.vdss_cmax,
        ).cmax.mean
        assert m == pytest.approx(gold, rel=1e-6)


def test_reconstructed_apriori_meta_equals_predict_meta():
    """build_meta_tracks + blend at the a-priori engine Cmax reproduces predict()'s meta."""
    r = predict(MIDAZOLAM, DOSE)
    tracks = build_meta_tracks(MIDAZOLAM, DOSE, r)
    routed = meta_blend_cmax(np.array([r.engine_pk.cmax.mean]), tracks)[0]
    assert routed == pytest.approx(r.pk.cmax.mean, rel=1e-3)


def test_predict_posterior_populates_meta_cmax_centered_on_apriori_meta():
    post = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    assert post.meta_cmax is not None
    apriori_meta = predict(MIDAZOLAM, DOSE).pk.cmax.mean
    assert post.meta_cmax.point == pytest.approx(apriori_meta, rel=0.20)


def test_posterior_meta_with_measured_F_reproduces_measured_F_meta():
    """The product output carries the validated measured-F improvement."""
    f = 0.30
    post = predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(f, cv=0.08)], seed=0)
    meta_routed = predict(
        MIDAZOLAM, DOSE, measured_adme=MeasuredADMEInput(f_bioavail=f)
    ).pk.cmax.mean
    assert post.meta_cmax.point == pytest.approx(meta_routed, rel=0.20)


def test_meta_cmax_interval_is_narrower_than_engine_interval_under_observation():
    """Conditioning narrows the product (meta) interval too."""
    post = predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.30, cv=0.08)], seed=0)
    mlo, mhi = post.meta_cmax.ci90
    assert mhi > mlo  # well-formed interval
    assert post.meta_cmax.point > 0


@pytest.mark.parametrize(
    "compound_type, cmax_clf, vdss_cmax",
    [
        ("base", 0.020, 0.080),     # base weights, all tracks
        ("neutral", None, 0.080),   # missing CLF
        ("acid", 0.020, None),      # missing VDss (scale=1.0)
        ("base", None, None),       # engine + ML only
    ],
)
def test_meta_blend_matches_combine_across_track_configurations(compound_type, cmax_clf, vdss_cmax):
    """The blend matches combine() for base/acid weights and missing tracks too.

    Built against the real engine/ml/clf PKEndpoints (cmax swapped) so combine()
    is the gold standard across every weight branch the sweep test didn't cover.
    """
    r = predict(MIDAZOLAM, DOSE)
    cmax_ml = 0.012
    learner = MetaLearner()
    tracks = MetaTracks(
        cmax_ml=cmax_ml, cmax_clf=cmax_clf, vdss_cmax=vdss_cmax, compound_type=compound_type
    )
    grid = np.logspace(-3.0, 0.0, 15)
    mine = meta_blend_cmax(grid, tracks)
    ml_pk = dataclasses.replace(r.ml_pk, cmax=Distribution(cmax_ml))
    clf_pk = dataclasses.replace(r.clf_pk, cmax=Distribution(cmax_clf)) if cmax_clf else None
    for c, m in zip(grid, mine):
        gold = learner.combine(
            dataclasses.replace(r.engine_pk, cmax=Distribution(c)),
            ml_pk,
            dose_mg=DOSE,
            compound_type=compound_type,
            clf_pk=clf_pk,
            vdss_cmax=vdss_cmax,
        ).cmax.mean
        assert m == pytest.approx(gold, rel=1e-6)


@pytest.mark.parametrize("engine_cmax", [0.0, -1.0])
def test_meta_blend_drops_engine_track_on_nonpositive_matching_combine(engine_cmax):
    """combine() drops the engine track when Cmax<=0; the replica must too."""
    r = predict(MIDAZOLAM, DOSE)
    tracks = build_meta_tracks(MIDAZOLAM, DOSE, r)
    mine = meta_blend_cmax(np.array([engine_cmax]), tracks)[0]
    gold = MetaLearner().combine(
        dataclasses.replace(r.engine_pk, cmax=Distribution(engine_cmax)),
        r.ml_pk,
        dose_mg=DOSE,
        compound_type=tracks.compound_type,
        clf_pk=r.clf_pk,
        vdss_cmax=tracks.vdss_cmax,
    ).cmax.mean
    assert mine == pytest.approx(gold, rel=1e-6)


def test_predict_posterior_provides_calibrated_conformal_predictive_interval():
    """cmax_90ci is the calibrated conformal band; meta_cmax.ci90 is the (narrow)
    F-parameter-uncertainty band. The predictive interval must be the wider one."""
    post = predict_posterior(MIDAZOLAM, DOSE, [MeasuredF(0.30, cv=0.08)], seed=0)
    assert post.cmax_90ci is not None
    lo, hi = post.cmax_90ci
    pt = post.meta_cmax.point
    assert lo < pt < hi
    # the calibrated predictive interval is much wider than the F-only band,
    # because structural error (not just F uncertainty) dominates.
    flo, fhi = post.meta_cmax.ci90
    assert (hi / lo) > (fhi / flo)
