"""Unit tests for the zonation weight profile.
Spec: 2026-06-17-liver-zonation-phase0-design.md §3.1."""
from __future__ import annotations

import math

import pytest

from sisyphus.validation.pgx_metrics import plugflow_E_linear, zonation_weights


def test_weights_sum_to_one():
    for direction in ("pericentral", "periportal", "uniform"):
        w = zonation_weights(10, 3.0, direction)
        assert math.isclose(sum(w), 1.0, rel_tol=1e-12)


def test_uniform_when_ratio_one():
    assert zonation_weights(5, 1.0, "pericentral") == pytest.approx([0.2] * 5)


def test_pericentral_increases_toward_outlet():
    w = zonation_weights(5, 3.0, "pericentral")
    assert all(w[i] < w[i + 1] for i in range(4))         # increasing toward tank N (outlet)
    assert math.isclose(w[-1] / w[0], 3.0, rel_tol=1e-9)  # ratio = w_max/w_min


def test_periportal_is_pericentral_reversed():
    assert zonation_weights(6, 2.5, "periportal") == pytest.approx(
        zonation_weights(6, 2.5, "pericentral")[::-1]
    )


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        zonation_weights(5, 0.5, "pericentral")   # ratio < 1
    with pytest.raises(ValueError):
        zonation_weights(5, 2.0, "sideways")      # bad direction


def test_plugflow_E_linear_matches_hand_value():
    # E = 1 - exp(-fu*CLint/Q); fu=0.3, CLint=90, Q=90 -> 1-exp(-0.3)=0.259
    assert plugflow_E_linear(0.3, 90.0, 90.0) == pytest.approx(1 - math.exp(-0.3), rel=1e-9)
