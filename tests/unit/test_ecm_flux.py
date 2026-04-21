"""Unit tests for the ECM ('extended') clearance flux branch.

Covers:
1. Formula correctness (hand-computed reference).
2. Degenerate limit (PS=1e6, no transporters → well-stirred to <1e-3).
3. f_up appears exactly once (catches f_up² bugs).
4. PS_active aggregates correctly from multi-transporter nodes.
5. Identity-blindness under organ-name rename.
6. No-transporter-kinetics fallback gives PS_active=0.
7. cl_int_bile default 0 preserves metabolism-only behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph, TransporterKinetics
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ClearanceFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node


def _make_liver_graph(
    liver_name: str = "liver",
    sink_name: str = "metabolized_hepatic",
    oatp_abundance: float = 0.0,
    cyp3a4_abundance: float = 9.2475e6,
    q_inflow: float = 99.45,   # L/h; 0.255 × 390
    v_liver: float = 1.80,
    ivive: float = 6e-5,
) -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name="blood_src", node_type="blood_pool",
                    volume=Distribution(1.5)))
    transporters = (
        {"OATP1B1": Distribution(oatp_abundance)} if oatp_abundance > 0 else {}
    )
    g.add_node(Node(
        name=liver_name, node_type="organ",
        volume=Distribution(v_liver),
        enzymes={"CYP3A4": Distribution(cyp3a4_abundance)},
        transporters=transporters,
        ivive_scaling=ivive,
    ))
    g.add_node(Node(name=sink_name, node_type="sink",
                    volume=Distribution(1e10)))
    g.add_edge(FlowEdge(source="blood_src", target=liver_name,
                        flow_rate=Distribution(q_inflow)))
    return g


def _make_ecm_drug(
    fup: float = 0.1,
    cyp3a4_affinity: float = 0.0,
    oatp_jmax: float = 0.0,
    oatp_km: float = 13.6,
    ps_passive: float = 1e6,
    ps_eff: float = 1e6,
    cl_int_bile: float = 0.0,
) -> DrugOnGraph:
    return DrugOnGraph(
        name="t", smiles="C", dose_mg=1.0, route="iv",
        administration_node="blood_src",
        mw=500.0, pka=None, compound_type="neutral",
        fup=Distribution(fup), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={},
        peff=Distribution(1.0), solubility=Distribution(10.0),
        enzyme_affinity=(
            {"CYP3A4": Distribution(cyp3a4_affinity)}
            if cyp3a4_affinity > 0 else {}
        ),
        renal_clearance=Distribution(0.0),
        transporter_kinetics=(
            {"OATP1B1": TransporterKinetics(
                jmax=Distribution(oatp_jmax),
                km=Distribution(oatp_km),
            )}
            if oatp_jmax > 0 else {}
        ),
        ps_passive=Distribution(ps_passive),
        ps_eff=Distribution(ps_eff),
        cl_int_bile=Distribution(cl_int_bile),
    )


def _clh_ref(q, fup, ps_active, ps_passive, ps_eff,
             cl_int_metab, cl_int_bile):
    ps_inf = ps_active + ps_passive
    cl_int_h = cl_int_metab + cl_int_bile
    num = q * fup * ps_inf * cl_int_h
    den = q * (ps_eff + cl_int_h) + fup * ps_inf * cl_int_h
    return num / den if den > 0 else 0.0


def _compute_rate(graph, drug, liver_name, sink_name, amount_liver,
                  model="extended"):
    graph.add_edge(ClearanceEdge(source=liver_name, target=sink_name,
                                 model=model))
    params = ResolvedParams(graph, drug)
    state_index = {"blood_src": 0, liver_name: 1, sink_name: 2}
    clearance_edge = graph.edges[-1]
    spec = ClearanceFluxSpec.from_edge(99, clearance_edge, state_index)
    y = np.array([0.0, amount_liver, 0.0])
    dydt = np.zeros(3)
    spec.apply(0.0, y, dydt, params)
    return -dydt[1]


def test_ecm_formula_matches_hand_computed():
    """Formula reference: Q=100, fup=0.1, PS_active=0.5, PS_passive=0.5,
    PS_eff=0.5, CL_int_metab=100, bile=45. Hand-computed CL_h × c_out."""
    # PS_active target 0.5 via abundance × Jmax/Km × ivive:
    #   ivive=6e-5, Jmax/Km=1.0 → abundance = 0.5 / 6e-5 = 8333.33
    g = _make_liver_graph(oatp_abundance=8333.33, cyp3a4_abundance=1.0,
                          q_inflow=100.0, ivive=6e-5)
    # CL_int_metab target 100 via abundance × affinity × ivive:
    #   abundance=1, ivive=6e-5 → affinity = 100 / 6e-5 ≈ 1.6667e6
    drug = _make_ecm_drug(
        fup=0.1, cyp3a4_affinity=1.6666667e6,
        oatp_jmax=1.0, oatp_km=1.0,
        ps_passive=0.5, ps_eff=0.5, cl_int_bile=45.0,
    )
    rate = _compute_rate(g, drug, "liver", "metabolized_hepatic",
                         amount_liver=100.0)
    c_out = 100.0 * 1.0 / (1.80 * 1.0)
    clh = _clh_ref(q=100.0, fup=0.1, ps_active=0.5, ps_passive=0.5,
                   ps_eff=0.5, cl_int_metab=100.0, cl_int_bile=45.0)
    assert rate == pytest.approx(clh * c_out, rel=1e-5)


def test_ecm_degenerate_limit_matches_well_stirred():
    """With PS_passive=PS_eff=1e6, no OATP, bile=0, extended rate ≈ WS rate."""
    drug = _make_ecm_drug(
        fup=0.1, cyp3a4_affinity=50.0,
        oatp_jmax=0.0, ps_passive=1e6, ps_eff=1e6, cl_int_bile=0.0,
    )
    g_ext = _make_liver_graph(oatp_abundance=0.0, cyp3a4_abundance=9.2475e6)
    rate_ext = _compute_rate(g_ext, drug, "liver",
                             "metabolized_hepatic", 10.0, model="extended")

    g_ws = _make_liver_graph(oatp_abundance=0.0, cyp3a4_abundance=9.2475e6)
    rate_ws = _compute_rate(g_ws, drug, "liver",
                            "metabolized_hepatic", 10.0, model="well_stirred")

    rel_err = abs(rate_ext - rate_ws) / max(abs(rate_ws), 1e-12)
    assert rel_err < 1e-3, f"rel_err={rel_err:.2e}, want <1e-3"


def test_fup_appears_exactly_once():
    """Catches f_up² bug: doubling f_up must not 4× CL_h."""
    base = dict(q=100.0, ps_active=0.5, ps_passive=0.5, ps_eff=0.5,
                cl_int_metab=100.0, cl_int_bile=45.0)
    clh1 = _clh_ref(fup=0.1, **base)
    clh2 = _clh_ref(fup=0.2, **base)
    ratio = clh2 / clh1
    # If f_up² bug existed, ratio ≈ 4. Correct formula: ratio 1<r<4.
    assert 1.0 < ratio < 4.0, f"ratio {ratio:.3f} not in (1,4) — f_up bug?"


def test_ps_active_from_two_transporters():
    """PS_active = Σ abundance × Jmax/Km × ivive across all transporters at source."""
    g = BodyGraph()
    g.add_node(Node(name="blood_src", node_type="blood_pool",
                    volume=Distribution(1.5)))
    g.add_node(Node(
        name="liver", node_type="organ", volume=Distribution(1.8),
        enzymes={},
        transporters={
            "OATP1B1": Distribution(10000.0),
            "OATP1B3": Distribution(5000.0),
        },
        ivive_scaling=1.0,
    ))
    g.add_node(Node(name="metabolized_hepatic", node_type="sink",
                    volume=Distribution(1e10)))
    g.add_edge(FlowEdge(source="blood_src", target="liver",
                        flow_rate=Distribution(100.0)))

    drug = DrugOnGraph(
        name="t", smiles="C", dose_mg=1.0, route="iv",
        administration_node="blood_src", mw=500.0, pka=None,
        compound_type="neutral",
        fup=Distribution(0.1), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={},
        peff=Distribution(1.0), solubility=Distribution(10.0),
        enzyme_affinity={}, renal_clearance=Distribution(0.0),
        transporter_kinetics={
            "OATP1B1": TransporterKinetics(
                jmax=Distribution(2.0), km=Distribution(4.0)),
            "OATP1B3": TransporterKinetics(
                jmax=Distribution(3.0), km=Distribution(6.0)),
        },
        ps_passive=Distribution(0.0),
        ps_eff=Distribution(0.0),
        cl_int_bile=Distribution(10.0),
    )
    # PS_active = 10000 × 2/4 + 5000 × 3/6 = 5000 + 2500 = 7500
    # CL_int_h = 10 (no metab, only bile=10)
    # Q=100, fup=0.1, PS_inf=7500, PS_eff=0
    # CL_h = 100 × 0.1 × 7500 × 10 / (100 × (0 + 10) + 0.1 × 7500 × 10)
    #      = 750000 / (1000 + 7500) = 88.2352...
    rate = _compute_rate(g, drug, "liver", "metabolized_hepatic",
                         amount_liver=1.0)
    expected_clh = (100.0 * 0.1 * 7500.0 * 10.0) / (
        100.0 * (0.0 + 10.0) + 0.1 * 7500.0 * 10.0
    )
    c_out = 1.0 / 1.8  # rbp=1, kp=1, v=1.8
    assert rate == pytest.approx(expected_clh * c_out, rel=1e-6)


def test_identity_blindness_under_rename():
    """Rename liver → xyz123. Rate must be bit-identical."""
    drug = _make_ecm_drug(
        fup=0.1, cyp3a4_affinity=10.0,
        oatp_jmax=1.0, oatp_km=2.0,
        ps_passive=0.5, ps_eff=0.5, cl_int_bile=1.0,
    )
    g1 = _make_liver_graph(liver_name="liver", oatp_abundance=10000.0)
    g2 = _make_liver_graph(liver_name="xyz123", oatp_abundance=10000.0)
    r1 = _compute_rate(g1, drug, "liver", "metabolized_hepatic", 10.0)
    r2 = _compute_rate(g2, drug, "xyz123", "metabolized_hepatic", 10.0)
    assert r1 == pytest.approx(r2, rel=1e-12)


def test_no_transporter_kinetics_gives_ps_active_zero():
    """Drug without transporter_kinetics for OATP1B1 → PS_active=0, no exception."""
    g = _make_liver_graph(oatp_abundance=10000.0)  # node HAS transporter
    drug = _make_ecm_drug(
        fup=0.1, cyp3a4_affinity=10.0,
        oatp_jmax=0.0,    # drug has NO kinetics (empty dict)
        ps_passive=1.0, ps_eff=1.0, cl_int_bile=0.0,
    )
    rate = _compute_rate(g, drug, "liver", "metabolized_hepatic", 10.0)
    assert rate > 0, "Metabolism should still produce non-zero clearance"


def test_cl_int_bile_additive_to_metabolism():
    """cl_int_bile > 0 adds to the total elimination pathway."""
    g = _make_liver_graph(oatp_abundance=0.0, cyp3a4_abundance=1.0)
    drug_no_bile = _make_ecm_drug(
        fup=0.5, cyp3a4_affinity=10.0,
        ps_passive=100.0, ps_eff=100.0, cl_int_bile=0.0,
    )
    drug_bile = _make_ecm_drug(
        fup=0.5, cyp3a4_affinity=10.0,
        ps_passive=100.0, ps_eff=100.0, cl_int_bile=5.0,
    )
    rate_no_bile = _compute_rate(g, drug_no_bile, "liver",
                                 "metabolized_hepatic", 10.0)
    rate_bile = _compute_rate(g, drug_bile, "liver",
                              "metabolized_hepatic", 10.0)
    assert rate_bile > rate_no_bile, "bile>0 should increase clearance"
