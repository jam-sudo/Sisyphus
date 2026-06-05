"""Pipeline wires the calibrated split-conformal interval into cmax_90ci.

The user-facing 90% PI is now the train-calibrated conformal interval
(holdout-validated coverage ~0.95 at nominal 0.90), not the parameter-only MC
interval (~0.30 coverage). It is set even at n_mc_samples=0 (a deterministic
transform of the meta point estimate) and is multiplicative: meta /÷ 10**q90.
"""

import json
import pathlib

from sisyphus.pipeline.predict import predict

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
_ART = pathlib.Path("data/validation/conformal_calibration.json")


def _q90_meta():
    return json.loads(_ART.read_text())["tracks"]["meta"]["0.1"]


def test_conformal_90ci_set_without_mc():
    """Default predict (n_mc_samples=0) populates cmax_90ci from the conformal q90."""
    res = predict(_CAFFEINE, 100.0, "oral")
    assert res.cmax_90ci is not None
    cmax = res.pk.cmax.mean
    factor = 10 ** _q90_meta()
    lo, hi = res.cmax_90ci
    assert abs(lo - cmax / factor) <= 1e-6 * cmax
    assert abs(hi - cmax * factor) <= 1e-6 * cmax
    # The interval brackets the point estimate
    assert lo < cmax < hi
