"""Route the engine posterior through the production meta blend.

The product output is the meta-learner's Cmax (engine + ML + CLF + VDss), not the
engine track alone. The non-engine tracks are F-independent, so they are fixed
across the posterior; only the engine track varies with the bioavailability
latent. ``meta_blend_cmax`` replicates ``MetaLearner.combine``'s Cmax blend
(including the engine-vs-ML disagreement penalty) vectorized over posterior
engine-Cmax samples, yielding a posterior over the *meta* Cmax with an honest
interval. A consistency test pins it to the production ``combine`` exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.ml.ensemble import (
    _DISAGREEMENT_THRESHOLD_LOG10,
    _W_VDSS,
    MetaLearner,
)

# Must match pipeline.predict._BW_KG_VDSS (VDss analytical Cmax = dose/(VDss*BW)).
_BW_KG_VDSS = 70.0


@dataclass(frozen=True)
class MetaTracks:
    """The F-independent meta tracks (fixed across the posterior)."""

    cmax_ml: float | None
    cmax_clf: float | None
    vdss_cmax: float | None
    compound_type: str


def build_meta_tracks(smiles: str, dose_mg: float, apriori_result) -> MetaTracks:
    """Assemble the non-engine meta tracks from an a-priori ``predict`` result.

    ml and clf Cmax come straight off the result; vdss Cmax and compound_type are
    recomputed from the (F-independent) chemistry/ADME layer so the blend can be
    reconstructed without modifying the predict contract. This is exact when the
    a-priori predict used no measured-ADME overrides (vdss/peff); the
    ``test_reconstructed_apriori_meta_equals_predict_meta`` consistency test
    guards against ``_BW_KG_VDSS`` / formula drift from ``pipeline.predict``.
    """
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile

    cmax_ml = apriori_result.ml_pk.cmax.mean if apriori_result.ml_pk is not None else None
    cmax_clf = apriori_result.clf_pk.cmax.mean if apriori_result.clf_pk is not None else None

    profile = compute_profile(smiles)
    vdss_cmax: float | None
    try:
        adme = predict_adme(profile)
        vdss_cmax = dose_mg / (adme.vdss.mean * _BW_KG_VDSS) if adme.vdss.mean > 0 else None
    except Exception:
        vdss_cmax = None

    return MetaTracks(
        cmax_ml=cmax_ml,
        cmax_clf=cmax_clf,
        vdss_cmax=vdss_cmax,
        compound_type=profile.compound_type,
    )


def meta_blend_cmax(
    engine_cmax: np.ndarray | float,
    tracks: MetaTracks,
    learner: MetaLearner | None = None,
) -> np.ndarray:
    """Vectorized replica of ``MetaLearner.combine`` Cmax over engine samples.

    Faithful for any engine_cmax, including non-positive values (combine() drops
    the engine track when Cmax<=0), so it is a drop-in for the production blend.
    """
    learner = learner or MetaLearner()
    engine_cmax = np.asarray(engine_cmax, dtype=float)
    log_eng = np.log10(np.maximum(engine_cmax, 1e-10))

    is_base = tracks.compound_type == "base"
    w_eng = learner.w_engine_base if is_base else learner.w_engine_other
    w_ml = learner.w_ml_base if is_base else learner.w_ml_other
    w_clf = learner.w_clf_base if is_base else learner.w_clf_other

    vdss_avail = tracks.vdss_cmax is not None and tracks.vdss_cmax > 0
    scale = (1.0 - _W_VDSS) if vdss_avail else 1.0

    # Fixed (engine-independent) contribution to the log-Cmax weighted mean.
    fixed_num = 0.0
    fixed_w = 0.0
    has_ml = tracks.cmax_ml is not None and tracks.cmax_ml > 0
    log_ml = math.log10(max(tracks.cmax_ml, 1e-10)) if has_ml else None
    if has_ml:
        fixed_num += (w_ml * scale) * log_ml
        fixed_w += w_ml * scale
    if tracks.cmax_clf is not None and tracks.cmax_clf > 0:
        fixed_num += (w_clf * scale) * math.log10(max(tracks.cmax_clf, 1e-10))
        fixed_w += w_clf * scale
    if vdss_avail:
        fixed_num += _W_VDSS * math.log10(max(tracks.vdss_cmax, 1e-10))
        fixed_w += _W_VDSS

    # Engine weight with the >10x engine-vs-ML disagreement penalty (vectorized).
    w_eng_eff = np.full_like(log_eng, w_eng * scale)
    if has_ml:
        disagree = np.abs(log_eng - log_ml)
        penalty = np.where(
            disagree > _DISAGREEMENT_THRESHOLD_LOG10,
            _DISAGREEMENT_THRESHOLD_LOG10 / np.maximum(disagree, 1e-12),
            1.0,
        )
        w_eng_eff = w_eng_eff * penalty

    # combine() gates the engine track on Cmax > 0 (ensemble.py:128): a
    # non-positive engine Cmax drops the engine track entirely (and its
    # disagreement penalty). Mirror that instead of clamping log to -10.
    w_eng_eff = np.where(np.asarray(engine_cmax) > 0, w_eng_eff, 0.0)

    num = w_eng_eff * log_eng + fixed_num
    den = w_eng_eff + fixed_w
    # den == 0 only when the engine is dropped and no fixed track exists ->
    # combine() returns 0.0 (no tracks); reproduce that.
    meta_log = num / np.where(den > 0, den, 1.0)
    return np.where(den > 0, np.power(10.0, meta_log), 0.0)
