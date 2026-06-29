import numpy as np
import pytest
from sisyphus.validation.pod_metrics import (
    geomean_span, implied_threshold, invert_pod, loo_predict_pod,
)


def test_geomean_span_ratio():
    assert geomean_span([2.0, 4.0, 8.0]) == pytest.approx(4.0)  # 8/2


def test_geomean_span_needs_two_positive():
    assert np.isnan(geomean_span([5.0]))
    assert np.isnan(geomean_span([0.0, -1.0]))


def test_implied_threshold_interpolates_on_dose():
    doses = [10.0, 100.0, 1000.0]
    hazards = [1.0, 5.0, 50.0]
    # at dose 100 -> hazard 5 exactly
    assert implied_threshold(doses, hazards, 100.0) == pytest.approx(5.0)
    # linear-in-dose interp between 10 and 100
    assert implied_threshold(doses, hazards, 55.0) == pytest.approx(3.0)


def test_invert_pod_is_inverse_of_threshold():
    doses = [10.0, 100.0, 1000.0]
    hazards = [1.0, 5.0, 50.0]
    # hazard 5 -> dose 100
    assert invert_pod(doses, hazards, 5.0) == pytest.approx(100.0)
    # clamps below/above the swept range
    assert invert_pod(doses, hazards, 0.5) == pytest.approx(10.0)
    assert invert_pod(doses, hazards, 99.0) == pytest.approx(1000.0)


def test_loo_predict_pod_uses_geomean_of_others():
    # three drugs with identical curves and identical observed PoD -> perfect LOO
    curve_d = [10.0, 100.0, 1000.0]
    curve_h = [1.0, 10.0, 100.0]
    per = {
        n: {"h_star": 10.0, "doses": curve_d, "hazards": curve_h, "pod_obs": 100.0}
        for n in ("a", "b", "c")
    }
    out = loo_predict_pod(per)
    assert out["a"]["h_star_used"] == pytest.approx(10.0)
    assert out["a"]["pod_pred"] == pytest.approx(100.0)
    assert out["a"]["fold_error"] == pytest.approx(1.0)
