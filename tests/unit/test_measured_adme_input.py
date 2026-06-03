"""Unit tests for MeasuredADMEInput construction and validation."""
import pytest

from sisyphus.predict.adme import MeasuredADMEInput


def test_valid_fup_clint_pair_constructs():
    m = MeasuredADMEInput(fup=0.20, clint=13.0)
    assert m.fup == 0.20
    assert m.clint == 13.0
    assert m.fup_cv == 0.15
    assert m.clint_cv == 0.20


def test_all_none_constructs():
    m = MeasuredADMEInput()
    assert m.fup is None and m.clint is None


def test_peff_only_is_allowed():
    # Only fup+clint are an atomic pair; peff may be supplied alone.
    m = MeasuredADMEInput(peff=5.0)
    assert m.peff == 5.0


def test_fup_without_clint_raises():
    with pytest.raises(ValueError, match="supplied together"):
        MeasuredADMEInput(fup=0.20)


def test_clint_without_fup_raises():
    with pytest.raises(ValueError, match="supplied together"):
        MeasuredADMEInput(clint=13.0)


def test_nonpositive_value_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        MeasuredADMEInput(fup=0.0, clint=13.0)


def test_cv_below_floor_raises():
    with pytest.raises(ValueError, match="< 0.10"):
        MeasuredADMEInput(fup=0.20, clint=13.0, fup_cv=0.05)
