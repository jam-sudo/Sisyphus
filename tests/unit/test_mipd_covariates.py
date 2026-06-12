"""Tests for mipd.covariates (Covariates) and mipd.core.ci_floor."""
import logging

import pytest

from sisyphus.mipd.core import ci_floor
from sisyphus.mipd.covariates import Covariates, cockcroft_gault


def test_ci_floor_off_by_default_fraction():
    assert ci_floor((0.9, 1.1), 1.0, 0.0) == (0.9, 1.1)


def test_ci_floor_widens_overtight_interval():
    # half-width 0.05 < 0.2*1.0 -> widen to +/- 0.2 around the mean
    assert ci_floor((0.95, 1.05), 1.0, 0.2) == (0.8, 1.2)


def test_ci_floor_leaves_wide_interval_unchanged():
    assert ci_floor((0.5, 1.5), 1.0, 0.2) == (0.5, 1.5)


def test_ci_floor_none_passthrough():
    assert ci_floor(None, 1.0, 0.2) is None


def test_ci_floor_nonpositive_mean_passthrough():
    assert ci_floor((0.9, 1.1), 0.0, 0.2) == (0.9, 1.1)


def test_covariates_renal_factor_unity_at_reference():
    assert Covariates(crcl_ml_min=125.0).renal_factor() == 1.0


def test_covariates_renal_factor_scales_linearly():
    assert Covariates(crcl_ml_min=62.5).renal_factor() == 0.5


def test_covariates_empty_is_no_op():
    assert Covariates().renal_factor() == 1.0
    assert Covariates(crcl_ml_min=None).renal_factor() == 1.0


def test_covariates_rejects_nonpositive_crcl():
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=0.0)
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=-5.0)


def test_covariates_weight_age_fields_and_validation():
    c = Covariates(body_weight_kg=10.0, age_years=2.0)
    assert c.body_weight_kg == 10.0 and c.age_years == 2.0
    with pytest.raises(ValueError):
        Covariates(body_weight_kg=0.0)
    with pytest.raises(ValueError):
        Covariates(age_years=-1.0)


def test_covariates_has_physiology():
    assert Covariates().has_physiology() is False
    assert Covariates(crcl_ml_min=50).has_physiology() is False  # CrCl is not physiology
    assert Covariates(body_weight_kg=10).has_physiology() is True
    assert Covariates(age_years=80).has_physiology() is True


def test_covariates_renal_factor_unaffected_by_weight_age():
    # renal is CrCl-only — weight/age never change renal_factor
    assert Covariates(body_weight_kg=10, age_years=80).renal_factor() == 1.0
    assert Covariates(crcl_ml_min=62.5, body_weight_kg=10, age_years=80).renal_factor() == 0.5


def test_covariates_warnings_flags_extremes():
    assert Covariates().warnings() == ()
    assert Covariates(crcl_ml_min=90, body_weight_kg=70, age_years=30).warnings() == ()
    assert any("crcl" in w.lower() for w in Covariates(crcl_ml_min=3).warnings())
    assert any("weight" in w.lower() for w in Covariates(body_weight_kg=1.0).warnings())
    assert any("age" in w.lower() for w in Covariates(age_years=120).warnings())


def test_cockcroft_gault_male_cancelling_anchor():
    # weight 72, scr 1.0 cancels the 72 denominator -> CrCl == 140 - age
    assert cockcroft_gault(60, 72.0, 1.0, "male") == pytest.approx(80.0)


def test_cockcroft_gault_female_factor():
    assert cockcroft_gault(60, 72.0, 1.0, "female") == pytest.approx(68.0)  # 80 * 0.85


def test_cockcroft_gault_non_cancelling_anchor():
    # full arithmetic: (140-40)*80 / (72*1.2) = 8000 / 86.4 = 92.5926
    assert cockcroft_gault(40, 80.0, 1.2, "male") == pytest.approx(92.5926, rel=1e-4)


def test_cockcroft_gault_sex_case_insensitive():
    assert cockcroft_gault(60, 72.0, 1.0, "Male") == pytest.approx(80.0)
    assert cockcroft_gault(60, 72.0, 1.0, "FEMALE") == pytest.approx(68.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"age_years": 60, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "x"}, "sex"),
        ({"age_years": 0, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "age_years"),
        ({"age_years": 140, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "age_years"),
        ({"age_years": 60, "weight_kg": 0, "scr_mg_dl": 1.0, "sex": "male"}, "weight_kg"),
        ({"age_years": 60, "weight_kg": 72, "scr_mg_dl": 0, "sex": "male"}, "scr_mg_dl"),
        ({"age_years": float("nan"), "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "finite"),
        ({"age_years": 60, "weight_kg": float("inf"), "scr_mg_dl": 1.0, "sex": "male"}, "finite"),
    ],
)
def test_cockcroft_gault_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cockcroft_gault(**kwargs)


def test_cockcroft_gault_pediatric_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cockcroft_gault(10, 30.0, 1.0, "male")
    assert "not validated" in caplog.text.lower()


def test_cockcroft_gault_low_scr_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cockcroft_gault(60, 72.0, 0.4, "male")
    assert "overestimate" in caplog.text.lower()


def test_cockcroft_gault_rejects_non_string_sex():
    with pytest.raises(ValueError, match="sex"):
        cockcroft_gault(60, 72.0, 1.0, None)


def test_cockcroft_gault_threshold_boundaries_do_not_warn(caplog):
    # age exactly 18 and SCr exactly 0.6 are NOT below the thresholds (< not <=) -> no advisory.
    with caplog.at_level(logging.WARNING):
        cockcroft_gault(18, 72.0, 0.6, "male")
    assert "not validated" not in caplog.text.lower()
    assert "overestimate" not in caplog.text.lower()


def test_from_cockcroft_gault_is_renal_only():
    cov = Covariates.from_cockcroft_gault(60, 72.0, 1.0, "male")
    assert cov.crcl_ml_min == pytest.approx(80.0)
    assert cov.body_weight_kg is None   # renal-only: weight/age are estimate inputs, not stored
    assert cov.age_years is None
    assert cov.has_physiology() is False  # so no generate_physiology rebuild is triggered


def test_from_cockcroft_gault_feeds_renal_factor():
    cov = Covariates.from_cockcroft_gault(60, 72.0, 1.0, "male")
    assert cov.renal_factor() == pytest.approx(80.0 / 125.0)


def test_from_cockcroft_gault_propagates_validation():
    with pytest.raises(ValueError, match="sex"):
        Covariates.from_cockcroft_gault(60, 72.0, 1.0, "unknown")
