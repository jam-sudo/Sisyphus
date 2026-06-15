# tests/unit/test_pgx_metrics.py
from __future__ import annotations

import pytest

from sisyphus.validation.pgx_metrics import (
    a_emp,
    analytical_fold,
    fm_agreement,
    fm_invivo,
)


def test_analytical_fold_pm_is_one_over_one_minus_fm():
    assert analytical_fold(fm=0.9, activity=0.0) == pytest.approx(10.0)
    assert analytical_fold(fm=0.5, activity=0.0) == pytest.approx(2.0)


def test_analytical_fold_partial_activity():
    assert analytical_fold(fm=1.0, activity=0.5) == pytest.approx(2.0)


def test_fm_invivo_inverts_pm_fold():
    assert fm_invivo(10.0) == pytest.approx(0.9)
    assert fm_invivo(2.0) == pytest.approx(0.5)


def test_a_emp_recovers_activity():
    fold = analytical_fold(fm=0.8, activity=0.3)
    assert a_emp(obs_fold=fold, fm=0.8) == pytest.approx(0.3, abs=1e-9)


def test_fm_agreement_within_tolerance():
    fm_vitro = [0.90, 0.78, 0.82]
    fm_vivo = [0.88, 0.75, 0.846]
    out = fm_agreement(fm_vitro, fm_vivo, tol=0.15)
    assert out["n"] == 3
    assert out["frac_within_tol"] == pytest.approx(1.0)
    assert 0.7 <= out["slope"] <= 1.3
