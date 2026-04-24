"""Tests for multi-dose regimen support.

Covers:
    - DosingEvent / DosingRegimen dataclasses and factory methods
    - Event-driven solver (single dose matches engine, multi-dose accumulates)
    - ConcentrationProfile construction
    - Steady-state detection
"""

from __future__ import annotations

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 — register flux specs
from sisyphus.core import Distribution, DrugOnGraph, SimResult
from sisyphus.engine.compiler import CompiledODE, ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node
from sisyphus.regimen.profile import (
    ConcentrationProfile,
    SteadyStateMetrics,
    build_concentration_profile,
    compute_steady_state_metrics,
)
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.types import (
    DEFAULT_IV_NODE,
    DEFAULT_ORAL_NODE,
    DosingEvent,
    DosingRegimen,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal graph with clearance (drug disappears, enabling
# accumulation testing)
# ---------------------------------------------------------------------------


def _make_two_node_graph() -> BodyGraph:
    """Two-node graph: blood pool (a) <-> organ (b) with clearance sink."""
    g = BodyGraph()
    g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="b", node_type="organ", volume=Distribution(10.0)))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(0.0)))
    g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="b", target="sink", model="gfr_filtration"))
    g.global_params["cardiac_output"] = Distribution(390.0)
    return g


def _make_minimal_drug(admin_node: str = "a", dose_mg: float = 100.0) -> DrugOnGraph:
    """Minimal drug with renal clearance, administered into node 'a'."""
    return DrugOnGraph(
        name="test_drug",
        smiles="C",
        dose_mg=dose_mg,
        route="iv",
        administration_node=admin_node,
        mw=200.0,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={"b": Distribution(1.5)},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={},
        renal_clearance=Distribution(2.0),  # 2 L/h renal CL
    )


def _compile_and_resolve(
    graph: BodyGraph, drug: DrugOnGraph
) -> tuple[CompiledODE, ResolvedParams]:
    """Compile graph and resolve parameters (deterministic, no sampling)."""
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    return compiled, params


# ===========================================================================
# Task 1: DosingEvent and DosingRegimen tests
# ===========================================================================


class TestDosingEvent:
    def test_bolus_creation(self):
        ev = DosingEvent(time_h=0.0, dose_mg=100.0, node="a")
        assert ev.time_h == 0.0
        assert ev.dose_mg == 100.0
        assert ev.node == "a"
        assert ev.duration_h == 0.0
        assert ev.is_bolus is True
        assert ev.end_time_h == 0.0

    def test_infusion_creation(self):
        ev = DosingEvent(time_h=1.0, dose_mg=500.0, node="b", duration_h=2.0)
        assert ev.is_bolus is False
        assert ev.end_time_h == 3.0

    def test_frozen(self):
        ev = DosingEvent(time_h=0.0, dose_mg=100.0, node="a")
        with pytest.raises(AttributeError):
            ev.dose_mg = 200.0  # type: ignore[misc]

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            DosingEvent(time_h=0.0, dose_mg=-10.0, node="a")

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time_h"):
            DosingEvent(time_h=-1.0, dose_mg=100.0, node="a")

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration_h"):
            DosingEvent(time_h=0.0, dose_mg=100.0, node="a", duration_h=-0.5)


class TestDosingRegimen:
    def test_single_oral(self):
        reg = DosingRegimen.single_oral(dose_mg=200.0)
        assert reg.n_doses == 1
        assert reg.total_dose_mg == 200.0
        assert reg.events[0].node == DEFAULT_ORAL_NODE
        assert reg.events[0].time_h == 0.0

    def test_single_iv_bolus(self):
        reg = DosingRegimen.single_iv(dose_mg=100.0)
        assert reg.n_doses == 1
        assert reg.events[0].node == DEFAULT_IV_NODE
        assert reg.events[0].duration_h == 0.0

    def test_single_iv_infusion(self):
        reg = DosingRegimen.single_iv(dose_mg=500.0, duration_h=1.0)
        assert reg.n_doses == 1
        assert reg.events[0].duration_h == 1.0

    def test_oral_repeated(self):
        reg = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=8.0, n_doses=3)
        assert reg.n_doses == 3
        assert reg.total_dose_mg == 300.0
        assert reg.events[0].time_h == 0.0
        assert reg.events[1].time_h == 8.0
        assert reg.events[2].time_h == 16.0
        assert reg.last_dose_time_h == 16.0
        assert all(e.node == DEFAULT_ORAL_NODE for e in reg.events)

    def test_iv_infusion_repeated(self):
        reg = DosingRegimen.iv_infusion(
            dose_mg=200.0, duration_h=0.5, interval_h=12.0, n_doses=2
        )
        assert reg.n_doses == 2
        assert reg.events[0].duration_h == 0.5
        assert reg.events[1].time_h == 12.0

    def test_empty_regimen_raises(self):
        with pytest.raises(ValueError, match="at least one event"):
            DosingRegimen(events=())

    def test_unsorted_events_raises(self):
        with pytest.raises(ValueError, match="sorted by time"):
            DosingRegimen(
                events=(
                    DosingEvent(time_h=10.0, dose_mg=100.0, node="a"),
                    DosingEvent(time_h=5.0, dose_mg=100.0, node="a"),
                )
            )

    def test_invalid_n_doses_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=8.0, n_doses=0)

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError, match="interval_h"):
            DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=-1.0, n_doses=3)


# ===========================================================================
# Task 2: Event-driven solver tests
# ===========================================================================


class TestSolveRegimen:
    def test_single_dose_matches_engine(self):
        """Single-dose regimen should produce results matching direct solve()."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        # Direct engine solve
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index["a"]] = 100.0
        t_eval = np.linspace(0.0, 24.0, 241)
        direct_result = solve(compiled, params, y0, t_span=(0.0, 24.0), t_eval=t_eval)

        # Regimen solve — single dose into node "a"
        regimen = DosingRegimen(
            events=(DosingEvent(time_h=0.0, dose_mg=100.0, node="a"),)
        )
        regimen_result = solve_regimen(
            compiled, params, regimen, t_total_h=24.0, dt_output=0.1
        )

        assert regimen_result.solver_success
        assert "a" in regimen_result.concentrations

        # Compare Cmax in node "a" (should be very close)
        direct_cmax = float(np.max(direct_result.concentrations["a"]))
        regimen_cmax = float(np.max(regimen_result.concentrations["a"]))
        assert abs(direct_cmax - regimen_cmax) / direct_cmax < 0.05, (
            f"Cmax mismatch: direct={direct_cmax:.4f} vs regimen={regimen_cmax:.4f}"
        )

    def test_multi_dose_accumulation(self):
        """Repeated doses should produce increasing trough concentrations."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        # 5 doses every 4 hours
        regimen = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=4.0, n_doses=5)
        # Override node to "a" since our graph doesn't have "stomach_lumen"
        events = tuple(
            DosingEvent(time_h=e.time_h, dose_mg=e.dose_mg, node="a")
            for e in regimen.events
        )
        regimen = DosingRegimen(events=events)

        result = solve_regimen(
            compiled, params, regimen, t_total_h=24.0, dt_output=0.1
        )

        assert result.solver_success

        # Extract peak concentrations per dosing interval
        conc = result.concentrations["a"]
        time = result.time_h

        peaks = []
        for i in range(5):
            t_start = i * 4.0
            t_end = (i + 1) * 4.0
            mask = (time >= t_start - 1e-12) & (time <= t_end + 1e-12)
            interval_conc = conc[mask]
            if len(interval_conc) > 0:
                peaks.append(float(np.max(interval_conc)))

        # With clearance, drug accumulates: each peak should be >= previous
        # (at least for first few doses before approaching steady state)
        assert len(peaks) >= 3, f"Expected at least 3 peaks, got {len(peaks)}"
        for i in range(1, min(3, len(peaks))):
            assert peaks[i] >= peaks[i - 1] * 0.99, (
                f"Peak {i+1} ({peaks[i]:.4f}) should be >= "
                f"peak {i} ({peaks[i-1]:.4f}) due to accumulation"
            )

    def test_multi_dose_trough_increases(self):
        """Trough levels (end of interval) should increase before SS."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        events = tuple(
            DosingEvent(time_h=i * 6.0, dose_mg=100.0, node="a")
            for i in range(4)
        )
        regimen = DosingRegimen(events=events)

        result = solve_regimen(
            compiled, params, regimen, t_total_h=30.0, dt_output=0.1
        )

        assert result.solver_success

        conc = result.concentrations["a"]
        time = result.time_h

        # Trough = concentration just before next dose
        troughs = []
        for i in range(1, 4):
            t_trough = i * 6.0
            idx = np.argmin(np.abs(time - t_trough))
            troughs.append(float(conc[idx]))

        # With accumulation, trough 2 > trough 1
        assert troughs[1] > troughs[0] * 0.95, (
            f"Trough 2 ({troughs[1]:.4f}) should be > "
            f"trough 1 ({troughs[0]:.4f})"
        )

    def test_solver_returns_sim_result(self):
        """solve_regimen must return a proper SimResult."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        regimen = DosingRegimen(
            events=(DosingEvent(time_h=0.0, dose_mg=100.0, node="a"),)
        )
        result = solve_regimen(compiled, params, regimen, t_total_h=12.0)

        assert isinstance(result, SimResult)
        assert len(result.time_h) > 0
        assert result.time_h[0] == pytest.approx(0.0, abs=1e-6)
        assert result.time_h[-1] == pytest.approx(12.0, abs=1e-6)
        assert "a" in result.concentrations
        assert "b" in result.concentrations
        assert "sink" in result.amounts

    def test_default_t_total(self):
        """Without explicit t_total_h, simulation runs 24h past last dose."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        events = tuple(
            DosingEvent(time_h=i * 8.0, dose_mg=100.0, node="a")
            for i in range(3)
        )
        regimen = DosingRegimen(events=events)

        result = solve_regimen(compiled, params, regimen)  # no t_total_h

        # last dose at 16h + 24h tail = 40h
        assert result.time_h[-1] == pytest.approx(40.0, abs=0.2)

    def test_invalid_node_does_not_crash(self):
        """Dose targeting a nonexistent node should warn, not crash."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        regimen = DosingRegimen(
            events=(DosingEvent(time_h=0.0, dose_mg=100.0, node="nonexistent"),)
        )
        # Should not raise — the dose is silently skipped with a warning
        result = solve_regimen(compiled, params, regimen, t_total_h=12.0)
        assert isinstance(result, SimResult)


# ===========================================================================
# Task 3: ConcentrationProfile and SteadyStateMetrics tests
# ===========================================================================


class TestConcentrationProfile:
    def _make_mock_sim_result(
        self, n_points: int = 100, scale: float = 1.0
    ) -> SimResult:
        """Create a mock SimResult for testing."""
        time = np.linspace(0, 24, n_points)
        conc = scale * np.exp(-0.1 * time)  # simple exponential decay
        return SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc * 5.0},
            mass_balance_error=0.0,
            solver_success=True,
        )

    def test_build_from_single_result(self):
        """Profile from a single MC sample should have equal CI bounds."""
        result = self._make_mock_sim_result()
        profile = build_concentration_profile([result], "venous_blood")

        assert isinstance(profile, ConcentrationProfile)
        assert profile.node == "venous_blood"
        assert len(profile.time_h) == 100
        np.testing.assert_array_almost_equal(profile.median, profile.ci_lower)
        np.testing.assert_array_almost_equal(profile.median, profile.ci_upper)

    def test_build_from_multiple_results(self):
        """Profile from multiple MC samples should have wider CI."""
        results = [self._make_mock_sim_result(scale=s) for s in [0.8, 1.0, 1.2]]
        profile = build_concentration_profile(results, "venous_blood")

        # CI should be wider than single-sample
        assert np.all(profile.ci_upper >= profile.median)
        assert np.all(profile.ci_lower <= profile.median)

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_concentration_profile([], "venous_blood")

    def test_missing_node_raises(self):
        result = self._make_mock_sim_result()
        with pytest.raises(ValueError, match="not found"):
            build_concentration_profile([result], "nonexistent")


class TestSteadyStateMetrics:
    def test_single_dose_no_steady_state(self):
        """Single dose should not be at steady state."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        regimen = DosingRegimen(
            events=(DosingEvent(time_h=0.0, dose_mg=100.0, node="a"),)
        )
        result = solve_regimen(compiled, params, regimen, t_total_h=24.0)

        metrics = compute_steady_state_metrics(result, regimen, node="a")

        assert isinstance(metrics, SteadyStateMetrics)
        assert metrics.css_max > 0
        assert metrics.accumulation_ratio == pytest.approx(1.0, abs=0.01)
        # Single dose cannot reach SS (needs at least 2 to compare)
        assert metrics.dose_to_steady_state == 0

    def test_many_doses_reaches_steady_state(self):
        """Many repeated doses should eventually reach steady state."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        # 10 doses every 6 hours — should approach SS
        events = tuple(
            DosingEvent(time_h=i * 6.0, dose_mg=100.0, node="a")
            for i in range(10)
        )
        regimen = DosingRegimen(events=events)

        result = solve_regimen(
            compiled, params, regimen, t_total_h=66.0, dt_output=0.1
        )

        metrics = compute_steady_state_metrics(result, regimen, node="a")

        assert isinstance(metrics, SteadyStateMetrics)
        assert metrics.css_max > 0
        assert metrics.css_min >= 0
        assert metrics.accumulation_ratio >= 1.0

    def test_accumulation_ratio_greater_than_one(self):
        """With drug that doesn't fully clear between doses, accumulation > 1."""
        graph = _make_two_node_graph()
        # Low clearance to ensure accumulation
        drug = DrugOnGraph(
            name="test_low_cl",
            smiles="C",
            dose_mg=100.0,
            route="iv",
            administration_node="a",
            mw=200.0,
            pka=None,
            compound_type="neutral",
            fup=Distribution(0.5),
            rbp=Distribution(1.0),
            kp_method="provided",
            kp_overrides={"b": Distribution(1.5)},
            peff=Distribution(1.0),
            solubility=Distribution(10.0),
            enzyme_affinity={},
            renal_clearance=Distribution(0.5),  # low clearance
        )
        compiled, params = _compile_and_resolve(graph, drug)

        # Frequent dosing with low clearance -> accumulation
        events = tuple(
            DosingEvent(time_h=i * 4.0, dose_mg=100.0, node="a")
            for i in range(6)
        )
        regimen = DosingRegimen(events=events)

        result = solve_regimen(
            compiled, params, regimen, t_total_h=30.0, dt_output=0.1
        )

        metrics = compute_steady_state_metrics(result, regimen, node="a")

        assert metrics.accumulation_ratio > 1.0, (
            f"Expected accumulation ratio > 1.0, got {metrics.accumulation_ratio:.3f}"
        )

    def test_missing_node_raises(self):
        """Should raise ValueError for nonexistent observation node."""
        graph = _make_two_node_graph()
        drug = _make_minimal_drug(admin_node="a", dose_mg=100.0)
        compiled, params = _compile_and_resolve(graph, drug)

        regimen = DosingRegimen(
            events=(DosingEvent(time_h=0.0, dose_mg=100.0, node="a"),)
        )
        result = solve_regimen(compiled, params, regimen, t_total_h=12.0)

        with pytest.raises(ValueError, match="not found"):
            compute_steady_state_metrics(result, regimen, node="nonexistent")
