"""V3: verify compute_endpoints() windowed Cmax extraction."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import SimResult
from sisyphus.pk.endpoints import compute_endpoints


def _build_sim_result() -> SimResult:
    # IV bolus-like profile: max at t=0, monotonic decline.
    time = np.linspace(0.0, 24.0, 500)
    conc = 5.405 * np.exp(-0.1 * time)  # decays from 5.405 at t=0
    conc[time < 0.083] = 5.405  # plateau at the very-early t
    return SimResult(
        time_h=time,
        concentrations={"venous_blood": conc},
        amounts={"venous_blood": conc * 3.7},
        mass_balance_error=0.0,
        solver_success=True,
    )


def test_compute_endpoints_default_picks_t0():
    result = _build_sim_result()
    pk = compute_endpoints(result)
    assert pk.cmax.mean == pytest.approx(5.405, rel=1e-3)
    assert pk.tmax.mean == pytest.approx(0.0, abs=1e-3)


def test_compute_endpoints_windowed_skips_t0():
    result = _build_sim_result()
    pk = compute_endpoints(result, t_min_h=5.0 / 60.0)
    # At t = 0.083h, conc = 5.405 * exp(-0.1 * 0.083) ≈ 5.360
    assert pk.cmax.mean < 5.405
    assert pk.cmax.mean == pytest.approx(5.405 * np.exp(-0.1 * 5.0 / 60.0), rel=1e-2)
    assert pk.tmax.mean >= 5.0 / 60.0


def test_compute_endpoints_t_min_h_zero_is_backward_compatible():
    result = _build_sim_result()
    pk_default = compute_endpoints(result)
    pk_zero = compute_endpoints(result, t_min_h=0.0)
    assert pk_default.cmax.mean == pytest.approx(pk_zero.cmax.mean)
    assert pk_default.tmax.mean == pytest.approx(pk_zero.tmax.mean)
