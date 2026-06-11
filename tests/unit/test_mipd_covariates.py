"""Tests for mipd.covariates (Covariates) and mipd.core.ci_floor."""
from sisyphus.mipd.core import ci_floor


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
    from sisyphus.mipd.covariates import Covariates
    assert Covariates(crcl_ml_min=125.0).renal_factor() == 1.0


def test_covariates_renal_factor_scales_linearly():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates(crcl_ml_min=62.5).renal_factor() == 0.5


def test_covariates_empty_is_no_op():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().renal_factor() == 1.0
    assert Covariates(crcl_ml_min=None).renal_factor() == 1.0


def test_covariates_rejects_nonpositive_crcl():
    import pytest

    from sisyphus.mipd.covariates import Covariates
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=0.0)
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=-5.0)


def test_covariates_weight_age_fields_and_validation():
    import pytest

    from sisyphus.mipd.covariates import Covariates
    c = Covariates(body_weight_kg=10.0, age_years=2.0)
    assert c.body_weight_kg == 10.0 and c.age_years == 2.0
    with pytest.raises(ValueError):
        Covariates(body_weight_kg=0.0)
    with pytest.raises(ValueError):
        Covariates(age_years=-1.0)


def test_covariates_has_physiology():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().has_physiology() is False
    assert Covariates(crcl_ml_min=50).has_physiology() is False  # CrCl is not physiology
    assert Covariates(body_weight_kg=10).has_physiology() is True
    assert Covariates(age_years=80).has_physiology() is True


def test_covariates_renal_factor_unaffected_by_weight_age():
    from sisyphus.mipd.covariates import Covariates
    # renal is CrCl-only — weight/age never change renal_factor
    assert Covariates(body_weight_kg=10, age_years=80).renal_factor() == 1.0
    assert Covariates(crcl_ml_min=62.5, body_weight_kg=10, age_years=80).renal_factor() == 0.5


def test_covariates_warnings_flags_extremes():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().warnings() == ()
    assert Covariates(crcl_ml_min=90, body_weight_kg=70, age_years=30).warnings() == ()
    assert any("crcl" in w.lower() for w in Covariates(crcl_ml_min=3).warnings())
    assert any("weight" in w.lower() for w in Covariates(body_weight_kg=1.0).warnings())
    assert any("age" in w.lower() for w in Covariates(age_years=120).warnings())
