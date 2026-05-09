"""Unit tests for ECM generalization test classifier.

Tests pure classifier logic per spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.
"""  # noqa: E501

from __future__ import annotations

import math

import pytest

from sisyphus.validation.oatp_generalization import (
    DrugOutcome,
    Mode,
    classify_aggregate,
    classify_drug,
)

# ---------- per-drug classification ----------

def test_pass_when_both_conditions_met():
    """Drug passes iff 90% PI contains obs AND |log10 FE| <= 0.48."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=1.2,  # FE = 1.2, log10 FE = 0.079, passes FE gate
        pi_low=0.8,
        pi_high=1.5,  # contains obs=1.0
    )
    assert out.passed is True
    assert out.log10_fe == pytest.approx(math.log10(1.2))


def test_fail_when_point_estimate_out_of_fe_gate():
    """FE = 3.5 (log10 = 0.544 > 0.48) → fail, even if PI contains obs."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=3.5,
        pi_low=0.5,
        pi_high=10.0,  # contains obs
    )
    assert out.passed is False


def test_fail_when_pi_does_not_contain_obs():
    """PI must contain observed."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=0.5,
        pi_low=0.3,
        pi_high=0.7,  # does NOT contain obs=1.0
    )
    assert out.passed is False


def test_fe_gate_boundary_inclusive():
    """|log10 FE| = 0.48 exactly passes the gate (FE = 3.02 approx)."""
    fe_boundary = 10 ** 0.48
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=fe_boundary,
        pi_low=0.1,
        pi_high=100.0,
    )
    assert out.passed is True


# ---------- aggregate mode classification ----------

def _make(obs: float, pred: float, pi: tuple[float, float], name: str = "d") -> DrugOutcome:
    return classify_drug(name, obs, pred, pi[0], pi[1])


def test_mode_A_when_all_pass():
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "a"),
        _make(1.0, 0.9, (0.5, 2.0), "b"),
        _make(1.0, 1.2, (0.5, 2.0), "c"),
    ]
    assert classify_aggregate(outcomes) == Mode.A


def test_mode_B_two_fail_same_direction_large_magnitude():
    """2/3 fail, both over-predict, median log10 FE > 0.5 → Mode B."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 5.0, (2.0, 12.0), "over1"),  # log10 FE = 0.699
        _make(1.0, 6.0, (3.0, 15.0), "over2"),  # log10 FE = 0.778
    ]
    assert classify_aggregate(outcomes) == Mode.B


def test_mode_B_all_fail_same_direction():
    """3/3 fail, same direction → Mode B (systematic), not Mode D."""
    outcomes = [
        _make(1.0, 5.0, (2.0, 12.0), "o1"),
        _make(1.0, 6.0, (3.0, 15.0), "o2"),
        _make(1.0, 4.0, (1.5, 10.0), "o3"),
    ]
    assert classify_aggregate(outcomes) == Mode.B


def test_mode_C_mixed_direction_failures():
    """2/3 fail, one over + one under → Mode C."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 5.0, (2.0, 12.0), "over"),
        _make(1.0, 0.2, (0.05, 0.5), "under"),
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_C_single_failure():
    """Single failure regardless of magnitude → Mode C."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass1"),
        _make(1.0, 0.9, (0.5, 2.0), "pass2"),
        _make(1.0, 5.0, (2.0, 12.0), "fail"),
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_C_two_same_direction_small_magnitude():
    """2/3 fail, same direction, but median |log10 FE| ≤ 0.5 → Mode C (not B).

    Both failures have |log10 FE| just above the 0.48 pass gate but the median
    is below 0.5 — below the Mode B magnitude threshold.
    """
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 3.05, (1.2, 7.0), "f1"),  # log10 FE ~ 0.484 (fails gate, below 0.5)
        _make(1.0, 3.10, (1.2, 7.0), "f2"),  # log10 FE ~ 0.491
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_D_all_fail_mixed_direction():
    """3/3 fail, mixed directions → Mode D."""
    outcomes = [
        _make(1.0, 5.0, (2.0, 12.0), "over"),
        _make(1.0, 6.0, (3.0, 15.0), "over2"),
        _make(1.0, 0.15, (0.05, 0.4), "under"),
    ]
    assert classify_aggregate(outcomes) == Mode.D


def test_precedence_A_over_B():
    """All-pass never reaches Mode B check."""
    outcomes = [
        _make(1.0, 1.0, (0.5, 2.0), "a"),
        _make(1.0, 1.0, (0.5, 2.0), "b"),
        _make(1.0, 1.0, (0.5, 2.0), "c"),
    ]
    assert classify_aggregate(outcomes) == Mode.A
