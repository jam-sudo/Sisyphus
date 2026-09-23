"""Pipeline exposes the legacy development-residual interval honestly.

It is set even at n_mc_samples=0 and is multiplicative around Meta, but it is
not labeled as independent split conformal. Parameter MC remains a separate
field when requested.
"""

import json
import pathlib

from sisyphus.pipeline.predict import predict

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
_ART = pathlib.Path("data/validation/development_residual_interval.json")


def _q90_meta():
    return json.loads(_ART.read_text())["tracks"]["meta"]["0.1"]


def test_development_residual_90ci_set_without_mc():
    res = predict(_CAFFEINE, 100.0, "oral")
    assert res.cmax_90ci is not None
    cmax = res.pk.cmax.mean
    factor = 10 ** _q90_meta()
    lo, hi = res.cmax_90ci
    assert abs(lo - cmax / factor) <= 1e-6 * cmax
    assert abs(hi - cmax * factor) <= 1e-6 * cmax
    # The interval brackets the point estimate
    assert lo < cmax < hi
    assert res.cmax_prediction.interval_source == "development_empirical_residual"
    assert res.cmax_prediction.residual_interval_90 == res.cmax_90ci
    assert res.cmax_prediction.parameter_interval_90 is None


def test_parameter_mc_is_not_overwritten_by_residual_band():
    res = predict(_CAFFEINE, 100.0, "oral", n_mc_samples=12)
    cmax = res.cmax_prediction
    assert cmax.residual_interval_90 is not None
    assert cmax.parameter_interval_90 is not None
    assert cmax.residual_interval_90 != cmax.parameter_interval_90
    assert cmax.interval_90 == cmax.residual_interval_90
