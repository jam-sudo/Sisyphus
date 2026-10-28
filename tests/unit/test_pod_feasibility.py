import numpy as np
import pytest

from sisyphus.validation.pod_metrics import (
    attribution_pass,
    auc_null_hazard,
    cluster_pass,
    exposure_pass,
    geomean_span,
    implied_threshold,
    invert_pod,
    loo_predict_pod,
    negative_pass,
    predict_pass,
    zone_pass,
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


def test_auc_null_hazard_is_clint_times_auc():
    time = [0.0, 1.0, 2.0]
    c_u = [0.0, 2.0, 0.0]          # trapezoid AUC = 2.0
    # (vmax/km) * AUC = (10/5) * 2 = 4
    assert auc_null_hazard(c_u, time, vmax_bio=10.0, km_bio=5.0) == pytest.approx(4.0)


def test_cluster_pass_band():
    ok, span = cluster_pass([3.0, 9.0, 12.0])      # span 4.0 <= 10
    assert ok and span == pytest.approx(4.0)
    ok2, span2 = cluster_pass([1.0, 50.0])          # span 50 > 10
    assert not ok2 and span2 == pytest.approx(50.0)


def test_attribution_pass_zonal_must_not_be_worse():
    assert attribution_pass(zonal_span=4.0, null_span=6.0) is True
    assert attribution_pass(zonal_span=8.0, null_span=6.0) is False


def test_negative_control_must_sit_below_toxicant_band():
    assert negative_pass(neg_hazard=0.5, toxicant_h_stars=[10.0, 20.0], margin=2.0) is True
    assert negative_pass(neg_hazard=8.0, toxicant_h_stars=[10.0, 20.0], margin=2.0) is False


def test_zone_pass_pericentral_is_outlet_half():
    # n_sub=10: indices 0..9, 0=periportal(inlet), 9=pericentral(outlet)
    assert zone_pass(9, "pericentral", 10) is True
    assert zone_pass(1, "pericentral", 10) is False
    assert zone_pass(0, "periportal", 10) is True
    assert zone_pass(9, "periportal", 10) is False


def test_exposure_pass_fold_band():
    assert exposure_pass(pred=2.0, obs=1.0, fold=3.0) is True   # 2x
    assert exposure_pass(pred=5.0, obs=1.0, fold=3.0) is False  # 5x


def test_predict_pass_counts_within_bracket_or_band():
    loo = {"a": {"pod_pred": 100.0}, "b": {"pod_pred": 100.0}, "c": {"pod_pred": 9000.0}}
    brackets = {"a": (50.0, 150.0), "b": (80.0, 400.0), "c": (50.0, 200.0)}
    ok, n_ok = predict_pass(loo, brackets, fold_band=5.0)  # a,b in bracket; c far out
    assert n_ok == 2 and ok is True   # N-1 of 3
