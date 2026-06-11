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
