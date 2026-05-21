"""Engine integration tests for B-11 fu_correction_liver gating (Task 6).

Verifies the correction multiplies fup at flagged nodes ONLY in
ClearanceFluxSpec well_stirred + parallel_tube branches. Uses a
synthetic minimal BodyGraph (NOT predict()) so the test exercises
exactly the code paths Task 6 modifies, independent of which
production drug routes through which flux type.

Identity-blind random-rename invariance is verified in Task 8.
ProdrugActivationFluxSpec gating is verified in Task 7. ECM
(extended) branch is out of B-11 scope.
"""

from __future__ import annotations

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ClearanceFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node

# ---------------------------------------------------------------------------
# Helpers — mirrors the _make_drug pattern in tests/unit/test_flux.py
# ---------------------------------------------------------------------------


def _make_drug(**overrides) -> DrugOnGraph:
    """Minimal DrugOnGraph for synthetic-graph flux tests."""
    defaults = dict(
        name="test",
        smiles="C",
        dose_mg=100.0,
        route="oral",
        administration_node="a",
        mw=100.0,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.1),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        particle_radius_um=25.0,
        ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


def _make_clearance_graph(
    *,
    model: str,
    fu_correction_applicable: float,
    abundance: float = 10.0,
    flow_rate: float = 20.0,
) -> tuple[BodyGraph, ClearanceFluxSpec, dict[str, int]]:
    """Build a 3-node graph: blood -> liver (CL edge to sink).

    `blood -> liver` FlowEdge supplies total_inflow at the liver. The
    `liver -> sink` ClearanceEdge is the spec under test. The clint
    contribution is small relative to Q so CL is in the linear regime
    (CL ≈ fup × CLint), which keeps the cross-test ratio clean.
    """
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(
        Node(
            name="liver",
            node_type="organ",
            volume=Distribution(1.5),
            enzymes={"CYP3A4": Distribution(abundance)},
            ivive_scaling=0.01,
            fu_correction_applicable=fu_correction_applicable,
        )
    )
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="blood", target="liver", flow_rate=Distribution(flow_rate)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model=model))

    # state_index assigned alphabetically: blood=0, liver=1, sink=2
    state_index = {"blood": 0, "liver": 1, "sink": 2}
    spec = ClearanceFluxSpec.from_edge(1, g.edges[1], state_index)
    return g, spec, state_index


def _eval_clearance_rate(
    *,
    model: str,
    fu_correction_applicable: float,
    fu_correction_liver: float,
    affinity: float = 5.0,
) -> float:
    """Return the absolute mass-loss rate at the liver source node."""
    g, spec, _ = _make_clearance_graph(
        model=model, fu_correction_applicable=fu_correction_applicable
    )
    drug = _make_drug(
        enzyme_affinity={"CYP3A4": Distribution(affinity)},
        fu_correction_liver=Distribution(mean=fu_correction_liver, cv=0.0),
        kp_overrides={"liver": Distribution(1.0)},  # explicit Kp=1 to keep c_out simple
    )
    params = ResolvedParams(g, drug)

    # liver carries 100mg of drug; blood and sink empty.
    y = np.array([0.0, 100.0, 0.0])
    dydt = np.zeros(3)
    spec.apply(0.0, y, dydt, params)

    # Mass conservation: liver loses what sink gains.
    assert abs(dydt[1] + dydt[2]) < 1e-12
    return float(-dydt[1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_well_stirred_fu_correction_amplifies_clearance_when_flagged():
    """well_stirred branch: with the node flagged, fu_correction_liver=5
    increases the effective fup five-fold inside the WS formula, which
    raises hepatic CL and thus the clearance rate.

    Numeric check matches the analytical well-stirred formula:
        CL = (Q * fup_eff * CLint) / (Q + fup_eff * CLint)
    with Q=20, CLint = abundance*affinity*ivive = 10*5*0.01 = 0.5,
    fup_base = 0.1, c_out = A * rbp / (V * kp) = 100 * 1 / (1.5 * 1) = 66.667.
    """
    rate_baseline = _eval_clearance_rate(
        model="well_stirred",
        fu_correction_applicable=1.0,
        fu_correction_liver=1.0,
    )
    rate_corrected = _eval_clearance_rate(
        model="well_stirred",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )

    # Analytical expected values
    q, clint, fup = 20.0, 0.5, 0.1
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected_baseline = (q * fup * clint) / (q + fup * clint) * c_out
    expected_corrected = (q * 5 * fup * clint) / (q + 5 * fup * clint) * c_out

    assert abs(rate_baseline - expected_baseline) < 1e-10
    assert abs(rate_corrected - expected_corrected) < 1e-10

    # Sanity: fu_corr=5 strictly raises CL (the gate's purpose).
    assert rate_corrected > rate_baseline
    # And in the near-linear regime (fup*CLint << Q), the ratio approaches 5.
    assert 4.9 < rate_corrected / rate_baseline < 5.0


def test_parallel_tube_fu_correction_amplifies_clearance_when_flagged():
    """parallel_tube branch: same gating behavior with the PT formula
        CL = Q * (1 - exp(-fup_eff * CLint / Q))
    Verifies the Task 6 patch is applied to the PT branch too.
    """
    rate_baseline = _eval_clearance_rate(
        model="parallel_tube",
        fu_correction_applicable=1.0,
        fu_correction_liver=1.0,
    )
    rate_corrected = _eval_clearance_rate(
        model="parallel_tube",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )

    q, clint, fup = 20.0, 0.5, 0.1
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected_baseline = q * (1.0 - np.exp(-fup * clint / q)) * c_out
    expected_corrected = q * (1.0 - np.exp(-5 * fup * clint / q)) * c_out

    assert abs(rate_baseline - expected_baseline) < 1e-10
    assert abs(rate_corrected - expected_corrected) < 1e-10

    assert rate_corrected > rate_baseline
    # PT in linear regime also ~5× (1 - exp(-5x) / (1 - exp(-x)) → 5 as x → 0).
    assert 4.9 < rate_corrected / rate_baseline < 5.0


def test_well_stirred_no_effect_when_node_flag_off():
    """Gate must be off: with fu_correction_applicable=0.0, the
    fu_correction_liver value is ignored — CL is identical whether
    fu_correction_liver=1.0 or 5.0. This is the load-bearing invariant
    of B-11: non-hepatic nodes (and unflagged hepatic nodes) must see
    zero behavior change."""
    rate_at_one = _eval_clearance_rate(
        model="well_stirred",
        fu_correction_applicable=0.0,
        fu_correction_liver=1.0,
    )
    rate_at_five = _eval_clearance_rate(
        model="well_stirred",
        fu_correction_applicable=0.0,
        fu_correction_liver=5.0,
    )

    # Bit-identical: gate off means the multiplication line is skipped entirely.
    assert rate_at_one == rate_at_five

    # And rate_at_one must equal the un-corrected analytical CL
    # (i.e., fu_correction_liver has truly no effect).
    q, clint, fup = 20.0, 0.5, 0.1
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected = (q * fup * clint) / (q + fup * clint) * c_out
    assert abs(rate_at_one - expected) < 1e-10
