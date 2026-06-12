"""Unit + integration tests for mipd.dosing (dose recommendation / target attainment)."""
import numpy as np  # noqa: F401  (used by later-task tests)
import pytest

from sisyphus.mipd.clgrid import MeasuredConc  # noqa: F401  (used by later-task tests)
from sisyphus.mipd.covariates import Covariates  # noqa: F401  (used by later-task tests)
from sisyphus.mipd.dosing import (
    CandidateEval,  # noqa: F401  (used by later-task tests)
    Constraint,
    DoseRecommendation,  # noqa: F401  (used by later-task tests)
    DoseTarget,
    recommend_dose,  # noqa: F401  (used by later-task tests)
)
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally cleared


def _iv_regimen():
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_constraint_rejects_unknown_quantity():
    with pytest.raises(ValueError, match="quantity"):
        Constraint(quantity="halflife", low=1.0)


def test_constraint_rejects_no_bound():
    with pytest.raises(ValueError, match="at least one"):
        Constraint(quantity="trough")


def test_constraint_rejects_low_above_high():
    with pytest.raises(ValueError, match="low"):
        Constraint(quantity="trough", low=5.0, high=2.0)


def test_constraint_rejects_negative_low():
    with pytest.raises(ValueError, match="low must be >= 0"):
        Constraint(quantity="trough", low=-1.0)


def test_constraint_rejects_nonpositive_high():
    with pytest.raises(ValueError, match="high must be > 0"):
        Constraint(quantity="cmax", high=0.0)


def test_constraint_accepts_one_sided():
    c = Constraint(quantity="cmax", high=10.0)
    assert c.low is None and c.high == 10.0


def test_dose_target_rejects_empty():
    with pytest.raises(ValueError, match="at least one constraint"):
        DoseTarget(constraints=())
