"""Pipeline exposes the legacy development-residual interval honestly.

It is set even at n_mc_samples=0 and is multiplicative around Meta, but it is
not labeled as independent split conformal. Parameter MC remains a separate
field when requested.
"""

import json
import pathlib
from dataclasses import replace
from importlib import import_module

import numpy as np

from sisyphus.core import SimResult
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


def test_residual_band_is_omitted_for_nondefault_partition_method():
    result = predict(_CAFFEINE, 100.0, "oral", kp_method="berezhkovskiy")
    assert result.cmax_prediction.residual_interval_90 is None
    assert result.cmax_prediction.interval_source is None


def test_residual_band_is_omitted_after_engine_fallback(monkeypatch):
    import sisyphus.engine.solver as solver

    monkeypatch.setattr(
        solver,
        "solve",
        lambda *args, **kwargs: SimResult(
            time_h=np.array([0.0]), concentrations={}, amounts={},
            mass_balance_error=0.0, solver_success=False,
        ),
    )
    result = predict(_CAFFEINE, 100.0)
    assert result.engine_pk is None
    assert result.cmax_prediction.residual_interval_90 is None
    assert result.cmax_prediction.interval_source is None


def test_residual_band_is_omitted_for_licensed_profile(monkeypatch):
    pipeline = import_module("sisyphus.pipeline.predict")
    monkeypatch.setattr(
        pipeline, "_RESOURCES", replace(pipeline._RESOURCES, profile="licensed_research")
    )
    result = pipeline.predict(_CAFFEINE, 100.0)
    assert result.cmax_prediction.residual_interval_90 is None
    assert result.cmax_prediction.interval_source is None
