"""Integration test: 2-species mass balance under prodrug routing.

Synthetic test: parent dosed at gut_wall, converts to active in
venous_blood_active at rate k=1/h, active eliminated at rate k_el=0.5/h,
MW ratio=1, yield=1.

Verifies cumulative active produced and parent dissipated balance via
analytical solution and mass-conservation invariant.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _build_synthetic_graph(k_conversion=1.0, k_elimination=0.5, yield_=1.0):
    """3-node, 2-edge minimal graph for analytical comparison.

    gut_wall (parent storage) → venous_blood_active (active) → metabolized_gut (sink)
    Vd_active=50L, CL_active=k_el×Vd=25 L/h
    """
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ",
                    volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(50.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=k_conversion, cv=0.0),
        conversion_yield=Distribution(mean=yield_, cv=0.0),
        mw_parent=100.0, mw_active=100.0,
    ))
    cl_active = k_elimination * 50.0  # so CL/Vd = k_el
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=cl_active, cv=0.0),
        vd_l=Distribution(mean=50.0, cv=0.0),
    ))
    return g


def _make_minimal_drug(active=None, obs_species="active"):
    from tests.unit.test_active_metabolite import _minimal_drug
    if active is None:
        active = ActiveMetabolite(
            name="X", mw=100.0,
            fup=Distribution(0.5), CL_per_h=Distribution(25.0),
            Vd_L=Distribution(50.0), conversion_rate_per_h=Distribution(1.0),
            conversion_site="gut_wall",
            conversion_yield_fraction=Distribution(1.0),
        )
    return _minimal_drug(active=active, obs_species=obs_species)


def test_synthetic_two_species_matches_analytical_cascade():
    """Compiled ODE for 2-species prodrug should match analytical solution."""
    g = _build_synthetic_graph(k_conversion=1.0, k_elimination=0.5, yield_=1.0)
    drug = _make_minimal_drug()

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    # Initial condition: 100 mg parent at gut_wall
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["gut_wall"]] = 100.0

    t_span = (0.0, 24.0)
    t_eval = np.linspace(0.0, 24.0, 200)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    assert sol.success

    parent_idx = compiled.state_index["gut_wall"]
    active_idx = compiled.state_index["venous_blood_active"]
    sink_idx = compiled.state_index["metabolized_gut"]

    # Mass balance at all time points: parent + active + sink ≈ 100
    total = sol.y[parent_idx] + sol.y[active_idx] + sol.y[sink_idx]
    np.testing.assert_allclose(total, 100.0, rtol=1e-4,
                               err_msg="Mass balance violated at one or more time points")

    # Also check final time explicitly
    final_total = sol.y[parent_idx, -1] + sol.y[active_idx, -1] + sol.y[sink_idx, -1]
    assert final_total == pytest.approx(100.0, rel=1e-4), (
        f"Mass balance violated: total={final_total}")

    # Compare to analytical 1st-order cascade:
    # A_parent(t) = 100 * exp(-k*t)
    # A_active(t) = (k * 100 / (k_el - k)) * (exp(-k*t) - exp(-k_el*t))
    k = 1.0
    k_el = 0.5
    expected_parent = 100.0 * np.exp(-k * sol.t)
    expected_active = (k * 100.0 / (k_el - k)) * (
        np.exp(-k * sol.t) - np.exp(-k_el * sol.t)
    )
    np.testing.assert_allclose(sol.y[parent_idx], expected_parent, rtol=1e-3)
    np.testing.assert_allclose(sol.y[active_idx], expected_active, rtol=1e-3)


def test_yield_below_one_reduces_active_proportionally():
    """yield=0.5, no elimination → final active accumulates to ~50% of parent dose."""
    g = _build_synthetic_graph(k_conversion=1.0, k_elimination=0.0, yield_=0.5)
    # CL=0 for the elimination edge; Vd must be > 0 to avoid division by zero guard
    am = ActiveMetabolite(
        name="Y", mw=100.0, fup=Distribution(0.5),
        CL_per_h=Distribution(0.0), Vd_L=Distribution(50.0),
        conversion_rate_per_h=Distribution(1.0), conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.5),
    )
    from tests.unit.test_active_metabolite import _minimal_drug
    drug = _minimal_drug(active=am, obs_species="active")

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["gut_wall"]] = 100.0

    sol = solve_ivp(rhs, (0.0, 100.0), y0,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    assert sol.success

    # After many parent half-lives: parent fully converted (100 mg total mass leaves);
    # active accumulates 100 × 0.5 yield = 50 mg.
    active_final = sol.y[compiled.state_index["venous_blood_active"], -1]
    assert active_final == pytest.approx(50.0, rel=1e-3)


def test_zero_yield_no_active_produced():
    """yield=0 → parent depleted, but no active accumulates."""
    g = _build_synthetic_graph(k_conversion=1.0, k_elimination=0.0, yield_=0.0)
    am = ActiveMetabolite(
        name="Z", mw=100.0, fup=Distribution(0.5),
        CL_per_h=Distribution(0.0), Vd_L=Distribution(50.0),
        conversion_rate_per_h=Distribution(1.0), conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.0),
    )
    from tests.unit.test_active_metabolite import _minimal_drug
    drug = _minimal_drug(active=am, obs_species="active")

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["gut_wall"]] = 100.0
    sol = solve_ivp(rhs, (0.0, 100.0), y0, method="LSODA",
                    rtol=1e-9, atol=1e-12)
    assert sol.success

    parent_final = sol.y[compiled.state_index["gut_wall"], -1]
    active_final = sol.y[compiled.state_index["venous_blood_active"], -1]
    assert parent_final == pytest.approx(0.0, abs=1e-6)
    assert active_final == pytest.approx(0.0, abs=1e-6)
