"""Unit + integration tests for mipd.dosing (dose recommendation / target attainment)."""
import math

import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc  # noqa: F401  (used by later-task tests)
from sisyphus.mipd.covariates import Covariates  # noqa: F401  (used by later-task tests)
from sisyphus.mipd.dosing import (
    CandidateEval,  # noqa: F401  (used by later-task tests)
    Constraint,
    DoseRecommendation,  # noqa: F401  (used by later-task tests)
    DoseTarget,
    _attainment,
    _center_m,
    _interval_reference,
    _max_overlap_region,
    _sample_m_intervals,
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


def test_sample_m_intervals_two_sided_window():
    # q_ref = 1.0 for all; trough window [2, 4] -> m must be in [2, 4].
    q_ref = {"trough": np.array([1.0, 1.0, 1.0])}
    target = DoseTarget((Constraint("trough", low=2.0, high=4.0),))
    m_lo, m_hi = _sample_m_intervals(q_ref, target)
    assert np.allclose(m_lo, 2.0)
    assert np.allclose(m_hi, 4.0)


def test_sample_m_intervals_unbounded_sides():
    q_ref = {"cmax": np.array([2.0, 2.0])}
    # ceiling only -> m_lo == 0, m_hi == high/q
    lo, hi = _sample_m_intervals(q_ref, DoseTarget((Constraint("cmax", high=10.0),)))
    assert np.allclose(lo, 0.0) and np.allclose(hi, 5.0)
    # floor only -> m_lo == low/q, m_hi == inf
    lo, hi = _sample_m_intervals(q_ref, DoseTarget((Constraint("cmax", low=4.0),)))
    assert np.allclose(lo, 2.0) and np.all(np.isinf(hi))


def test_max_overlap_region_picks_densest_segment():
    # intervals: [0,2], [1,3], [1,4]  -> max overlap (3) on [1, 2]
    m_lo = np.array([0.0, 1.0, 1.0])
    m_hi = np.array([2.0, 3.0, 4.0])
    a, b, count = _max_overlap_region(m_lo, m_hi)
    assert count == 3
    assert a == pytest.approx(1.0) and b == pytest.approx(2.0)


def test_max_overlap_region_all_infeasible_returns_zero():
    # every sample interval empty (m_lo > m_hi) -> no attainable dose multiplier
    a, b, count = _max_overlap_region(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    assert (a, b, count) == (0.0, 0.0, 0)


def test_max_overlap_region_single_feasible_interval():
    a, b, count = _max_overlap_region(np.array([2.0]), np.array([4.0]))
    assert count == 1 and a == pytest.approx(2.0) and b == pytest.approx(4.0)


def test_attainment_counts_covering_intervals():
    m_lo = np.array([0.0, 1.0, 1.0])
    m_hi = np.array([2.0, 3.0, 4.0])
    assert _attainment(1.5, m_lo, m_hi) == pytest.approx(1.0)   # all three cover 1.5
    assert _attainment(3.5, m_lo, m_hi) == pytest.approx(1.0 / 3.0)  # only [1,4]


def test_center_m_rules():
    assert _center_m(2.0, 8.0) == pytest.approx(4.0)        # bounded -> geometric mid sqrt(16)
    assert _center_m(3.0, math.inf) == pytest.approx(3.0)   # only floors -> smallest (a)
    assert _center_m(0.0, 5.0) == pytest.approx(5.0)        # only ceilings -> largest (b)


def test_center_m_floor_only_nonbinding_keeps_current_dose():
    # b == inf and a == 0 (floor non-binding for all samples) -> keep current dose (1.0)
    assert _center_m(0.0, math.inf) == pytest.approx(1.0)


def test_lti_exactness_grid_scales_linearly_with_dose():
    # The whole layer rests on the engine being linear in dose. Doubling the dose must
    # exactly double steady-state Cmax and AUC at every renal scale. A future saturable
    # nonlinearity would fail HERE.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g1 = build_renal_cl_grid(
        ATENOLOL, DosingRegimen.iv_infusion(50.0, 0.5, 8.0, 5), n_grid=5
    )
    g2 = build_renal_cl_grid(
        ATENOLOL, DosingRegimen.iv_infusion(100.0, 0.5, 8.0, 5), n_grid=5
    )
    assert np.allclose(g2.cmax, 2.0 * g1.cmax, rtol=1e-6)
    assert np.allclose(g2.auc, 2.0 * g1.auc, rtol=1e-6)


def test_interval_reference_returns_quantities_at_reference_dose():
    reg = _iv_regimen()
    r = np.array([1.0, 1.0, 1.0])
    q_ref, d_ref = _interval_reference(
        ATENOLOL, reg, 8.0, r, renal_factor=1.0, body_weight_kg=None,
        age_years=None, n_grid=5, kp_method="rodgers_rowland",
    )
    assert d_ref == 50.0
    assert set(q_ref) == {"trough", "cmax", "auc24"}
    for key in q_ref:
        assert q_ref[key].shape == (3,)
        assert np.all(q_ref[key] > 0)
    # trough must equal the grid's own conc at the end of the final interval at r=1.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=5)
    expect = float(g.conc_at(np.array([1.0]), reg.last_dose_time_h + 8.0)[0])
    assert q_ref["trough"][0] == pytest.approx(expect, rel=1e-9)
