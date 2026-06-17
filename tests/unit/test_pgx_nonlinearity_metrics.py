"""Unit tests for the genotype-nonlinearity pure metrics.
Spec: 2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md §5.2, §6."""
from __future__ import annotations

import pytest

from sisyphus.validation.pgx_metrics import (
    box_robustness_pass,
    delta_beta,
    km_uM_to_unbound_mgL,
    loglog_beta,
)


def test_km_conversion_worked_example():
    # propafenone: 5.3 µM × fu_mic 0.5 × MW 341.4 / 1000 = 0.90471 mg/L (spike value)
    assert km_uM_to_unbound_mgL(5.3, 341.4, 0.5) == pytest.approx(0.90471, rel=1e-4)


def test_km_conversion_rejects_bad_input():
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(-1.0, 300.0, 0.5)
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(5.0, 300.0, 0.0)
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(5.0, 300.0, 1.5)


def test_loglog_beta_proportional_is_one():
    assert loglog_beta([100.0, 200.0, 400.0], [10.0, 20.0, 40.0]) == pytest.approx(1.0)


def test_loglog_beta_supraproportional_gt_one():
    assert loglog_beta([100.0, 200.0], [10.0, 40.0]) == pytest.approx(2.0)


def test_loglog_beta_needs_two_points():
    with pytest.raises(ValueError):
        loglog_beta([100.0], [10.0])


def test_delta_beta_sign():
    assert delta_beta(1.6, 1.0) == pytest.approx(0.6)
    assert delta_beta(1.0, 1.6) == pytest.approx(-0.6)


def test_box_robustness_pass_requires_every_corner():
    assert box_robustness_pass([0.2, 0.15, 0.30, 0.12], threshold=0.10) is True
    assert box_robustness_pass([0.2, 0.05, 0.30, 0.12], threshold=0.10) is False
    with pytest.raises(ValueError):
        box_robustness_pass([], threshold=0.10)
