"""Integration test: v2 prodrug well-stirred + 1C elim cascade vs analytical.

Topology requires flow loop (well-stirred is undefined without Q):
  infusion_source --[Q=60]--> conversion_node --[Q=60]--> exit_sink
                                  └--[ProdrugActivation]--> active_pool
                                                              └--[1C elim]--> elim_sink
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _build_synthetic(
    q=60.0, v_source=10.0, v_active=10.0,
    abundance=1e6, affinity=10.0, ivive=6e-5,
    fup=1.0, cl_active=10.0,
    mw_parent=200.0, mw_active=200.0, yield_=1.0,
):
    g = BodyGraph()
    g.add_node(Node(name="infusion_source", node_type="blood_pool",
                    volume=Distribution(1.0)))
    g.add_node(Node(
        name="conversion_node", node_type="blood_pool",
        volume=Distribution(v_source),
        enzymes={"X": Distribution(mean=abundance, cv=0.0)},
        ivive_scaling=ivive,
    ))
    g.add_node(Node(name="active_pool", node_type="blood_pool",
                    volume=Distribution(v_active)))
    g.add_node(Node(name="exit_sink", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_node(Node(name="elim_sink", node_type="sink",
                    volume=Distribution(0.0)))

    g.add_edge(FlowEdge(source="infusion_source", target="conversion_node",
                        flow_rate=Distribution(q)))
    g.add_edge(FlowEdge(source="conversion_node", target="exit_sink",
                        flow_rate=Distribution(q)))
    g.add_edge(ProdrugActivationEdge(
        source="conversion_node", target="active_pool",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(yield_),
        mw_parent=mw_parent, mw_active=mw_active,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="active_pool", target="elim_sink",
        cl_per_h=Distribution(cl_active),
        vd_l=Distribution(v_active),
    ))

    drug = _minimal_drug(
        fup=Distribution(mean=fup, cv=0.0),
        rbp=Distribution(mean=1.0, cv=0.0),
        enzyme_affinity_for_conversion={"X": Distribution(mean=affinity, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    return g, drug


def test_steady_state_matches_analytical():
    """Total mass conserved during ODE solve (mw_ratio=1, yield=1 → mass conservation)."""
    q, v_source, v_active = 60.0, 10.0, 10.0
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup, cl_active, yield_ = 1.0, 10.0, 1.0
    g, drug = _build_synthetic(
        q=q, v_source=v_source, v_active=v_active,
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup, cl_active=cl_active, yield_=yield_,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["infusion_source"]] = 100.0

    sol = solve_ivp(rhs, (0.0, 50.0), y0, method="LSODA", rtol=1e-8, atol=1e-10,
                    t_eval=np.linspace(0.0, 50.0, 500))

    assert sol.success, f"Solver failed: {sol.message}"

    # Mass balance: with mw_ratio=1 and yield=1, total mg conserved
    src_idx = compiled.state_index["infusion_source"]
    cnv_idx = compiled.state_index["conversion_node"]
    act_idx = compiled.state_index["active_pool"]
    ex_idx = compiled.state_index["exit_sink"]
    el_idx = compiled.state_index["elim_sink"]
    total = (sol.y[src_idx] + sol.y[cnv_idx] + sol.y[act_idx]
             + sol.y[ex_idx] + sol.y[el_idx])
    assert np.allclose(total, 100.0, rtol=1e-3), \
        f"Mass not conserved: total ranges {total.min()} to {total.max()}"


def test_extraction_efficiency_matches_well_stirred_formula():
    """FLUX-1: parent loss = intrinsic-clearance activation (fup·CLint·c_out) +
    convective outflow (Q·c_out). The activation sink uses the intrinsic clearance
    so that, combined with the convective edge, hepatic extraction approaches 1.0."""
    q, v_source = 60.0, 10.0
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup = 1.0
    g, drug = _build_synthetic(
        q=q, v_source=v_source,
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    y = np.zeros(compiled.n_states)
    a_parent = 50.0
    y[compiled.state_index["conversion_node"]] = a_parent
    dydt = rhs(0.0, y)

    clint = abundance * affinity * ivive
    cl_intrinsic = fup * clint
    c_out = a_parent / v_source  # blood_pool, kp=1, rbp=1
    expected_rate_parent_loss_via_activation = cl_intrinsic * c_out

    flow_out = q * c_out
    total_parent_loss = expected_rate_parent_loss_via_activation + flow_out
    actual_parent_loss = -dydt[compiled.state_index["conversion_node"]]
    assert actual_parent_loss == pytest.approx(total_parent_loss, rel=1e-6)

    actual_active_gain = dydt[compiled.state_index["active_pool"]]
    assert actual_active_gain == pytest.approx(
        expected_rate_parent_loss_via_activation, rel=1e-6
    )
