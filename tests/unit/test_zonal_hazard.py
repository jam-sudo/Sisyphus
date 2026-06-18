"""Unit tests for the zonal reactive-metabolite hazard post-processor.
Spec: 2026-06-18-zonal-reactive-metabolite-hazard-design.md §3.2."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.validation.pgx_metrics import mm_rate, zonal_hazard


def test_mm_rate_basic():
    assert mm_rate(1.0, 10.0, 1.0) == pytest.approx(5.0)   # c=Km -> half Vmax
    assert mm_rate(0.0, 10.0, 1.0) == pytest.approx(0.0)


def test_zonal_hazard_threshold_zero_below_capacity():
    # constant C_u; formation MM(c)=Vmax_bio*c/(Km+c) < detox capacity => hazard 0
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]                # one zone, c=1
    # MM(1; vmax=4, km=1) = 2.0; detox capacity 3.0 > 2.0 -> no excess
    h = zonal_hazard(c, [4.0], 1.0, [3.0], time)
    assert h[0] == pytest.approx(0.0)


def test_zonal_hazard_positive_above_capacity():
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]
    # MM(1; vmax=8, km=1) = 4.0; detox 1.0 -> excess 3.0 over T=10 -> 30
    h = zonal_hazard(c, [8.0], 1.0, [1.0], time)
    assert h[0] == pytest.approx(30.0, rel=1e-6)


def test_zonal_hazard_monotonic_decreasing_in_detox():
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]
    h_lo = zonal_hazard(c, [8.0], 1.0, [1.0], time)[0]
    h_hi = zonal_hazard(c, [8.0], 1.0, [3.0], time)[0]
    assert h_hi < h_lo


def test_zonal_hazard_per_zone_shape():
    time = np.linspace(0.0, 5.0, 51)
    c = [np.full_like(time, 2.0), np.full_like(time, 0.5)]   # 2 zones
    h = zonal_hazard(c, [10.0, 10.0], 1.0, [2.0, 2.0], time)
    assert len(h) == 2 and h[0] > h[1]      # higher-conc zone has more hazard
