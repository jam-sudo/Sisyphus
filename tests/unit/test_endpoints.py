import numpy as np
import pytest

from sisyphus.core import Distribution, SimResult
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.pk.nca import auc_trapezoidal, terminal_half_life


class TestNCA:
    def test_auc_triangle(self):
        time = np.array([0, 1, 2], dtype=float)
        conc = np.array([0, 10, 0], dtype=float)
        assert auc_trapezoidal(time, conc) == pytest.approx(10.0)

    def test_auc_rectangle(self):
        time = np.array([0, 1, 2, 3], dtype=float)
        conc = np.array([5, 5, 5, 5], dtype=float)
        assert auc_trapezoidal(time, conc) == pytest.approx(15.0)

    def test_terminal_half_life_exponential(self):
        """Known exponential decay: C = 10 * exp(-0.1*t), t½ = ln(2)/0.1 = 6.93"""
        time = np.linspace(0, 50, 100)
        conc = 10.0 * np.exp(-0.1 * time)
        # Cmax is at t=0, so all points after 0 are terminal
        t_half = terminal_half_life(time, conc)
        assert t_half is not None
        assert t_half == pytest.approx(np.log(2) / 0.1, rel=0.01)

    def test_terminal_half_life_too_few_points(self):
        time = np.array([0, 1], dtype=float)
        conc = np.array([10, 5], dtype=float)
        assert terminal_half_life(time, conc) is None

    def test_terminal_half_life_none_if_positive_slope(self):
        """Rising concentration after Cmax should return None."""
        time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        # Cmax at index 0, then strictly increasing — slope > 0
        conc = np.array([10, 11, 12, 13, 14, 15, 16], dtype=float)
        assert terminal_half_life(time, conc) is None

    def test_auc_increasing(self):
        """AUC over a ramp: integral of t from 0 to 3 = 4.5"""
        time = np.array([0, 1, 2, 3], dtype=float)
        conc = np.array([0, 1, 2, 3], dtype=float)
        assert auc_trapezoidal(time, conc) == pytest.approx(4.5)


class TestEndpoints:
    def test_cmax_tmax(self):
        time = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        conc = np.array([0, 5, 10, 7, 4, 2], dtype=float)
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc * 3.7},
            mass_balance_error=0.0,
            solver_success=True,
        )
        pk = compute_endpoints(result)
        assert pk.cmax.mean == pytest.approx(10.0)
        assert pk.tmax.mean == pytest.approx(2.0)

    def test_auc_computed(self):
        time = np.array([0, 1, 2, 3], dtype=float)
        conc = np.array([0, 10, 10, 0], dtype=float)
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc},
            mass_balance_error=0.0,
            solver_success=True,
        )
        pk = compute_endpoints(result)
        assert pk.auc_0t.mean == pytest.approx(20.0)

    def test_endpoints_returns_distributions(self):
        """All mandatory fields are Distribution instances with cv=0."""
        time = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        conc = np.array([0, 5, 10, 7, 4, 2], dtype=float)
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc},
            mass_balance_error=0.0,
            solver_success=True,
        )
        pk = compute_endpoints(result)
        assert isinstance(pk.cmax, Distribution)
        assert isinstance(pk.tmax, Distribution)
        assert isinstance(pk.auc_0t, Distribution)
        assert pk.cmax.cv == 0.0
        assert pk.tmax.cv == 0.0
        assert pk.auc_0t.cv == 0.0

    def test_t_half_estimated(self):
        """When terminal decline is present, t_half should be a Distribution."""
        time = np.linspace(0, 24, 200)
        # Absorb then eliminate: peak around t=2
        conc = 10.0 * (np.exp(-0.1 * time) - np.exp(-2.0 * time))
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc},
            mass_balance_error=0.0,
            solver_success=True,
        )
        pk = compute_endpoints(result)
        assert pk.t_half is not None
        assert isinstance(pk.t_half, Distribution)
        assert pk.t_half.mean > 0.0

    def test_custom_observation_node(self):
        """compute_endpoints respects the observation_node argument."""
        time = np.array([0, 1, 2, 3], dtype=float)
        conc_vb = np.array([0, 1, 2, 1], dtype=float)
        conc_liver = np.array([0, 3, 6, 3], dtype=float)
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc_vb, "liver": conc_liver},
            amounts={"venous_blood": conc_vb, "liver": conc_liver},
            mass_balance_error=0.0,
            solver_success=True,
        )
        pk = compute_endpoints(result, observation_node="liver")
        assert pk.cmax.mean == pytest.approx(6.0)
        assert pk.tmax.mean == pytest.approx(2.0)
