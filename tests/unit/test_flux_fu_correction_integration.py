"""Engine integration tests for B-11 fu_correction_liver gating (Task 6).

Verifies the correction multiplies fup at flagged nodes ONLY in
ClearanceFluxSpec well_stirred + parallel_tube branches. Uses a
synthetic minimal BodyGraph (NOT predict()) so the test exercises
exactly the code paths Task 6 modifies, independent of which
production drug routes through which flux type.

Identity-blind random-rename invariance is verified in Task 8.
ProdrugActivationFluxSpec gating is verified in Task 7. The ECM
(extended) clearance model intentionally does NOT apply the correction:
it models concentrative hepatic uptake explicitly (PS_active/PS_inf/
PS_eff), so multiplying fup by the WS-style fu_inc/fu_plasma factor
would double-count that uptake. That decision is pinned by
``test_extended_clearance_ignores_fu_correction_by_design`` and the
silent-no-op trap (the production liver clears via ``extended``) is
closed by a runtime warning from ``pipeline.predict`` — see
``test_predict_warns_when_fu_correction_dropped_at_extended_node``
(2026-06-05 contract fix).
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

    # FLUX-1: intrinsic clearance CL_int = fup_eff * CLint applied to c_out
    # (flow limitation now emerges from the separate convective outflow edge).
    clint, fup = 0.5, 0.1
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected_baseline = fup * clint * c_out
    expected_corrected = 5 * fup * clint * c_out

    assert abs(rate_baseline - expected_baseline) < 1e-10
    assert abs(rate_corrected - expected_corrected) < 1e-10

    # Sanity: fu_corr=5 strictly raises CL (the gate's purpose).
    assert rate_corrected > rate_baseline
    # Intrinsic clearance is exactly linear in fup_eff, so the ratio is exactly 5.
    assert abs(rate_corrected / rate_baseline - 5.0) < 1e-9


def test_parallel_tube_fu_correction_amplifies_clearance_when_flagged():
    """parallel_tube branch: same fu_correction gating as well_stirred.

    Post-FLUX-1 the parallel_tube branch collapses to the intrinsic
    clearance ``fup_eff * CLint`` applied to ``c_out`` — a single
    well-mixed compartment cannot represent the axial gradient a true
    parallel-tube ``CL = Q*(1 - exp(-fup_eff*CLint/Q))`` requires — so the
    assertion below pins that collapsed form. This still verifies the
    Task 6 fu correction is applied to the PT branch.
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

    # FLUX-1: parallel_tube collapses to the intrinsic clearance fup_eff*CLint
    # in a single well-mixed compartment (true PT needs an axial gradient).
    clint, fup = 0.5, 0.1
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected_baseline = fup * clint * c_out
    expected_corrected = 5 * fup * clint * c_out

    assert abs(rate_baseline - expected_baseline) < 1e-10
    assert abs(rate_corrected - expected_corrected) < 1e-10

    assert rate_corrected > rate_baseline
    # Intrinsic clearance is exactly linear in fup_eff, so the ratio is exactly 5.
    assert abs(rate_corrected / rate_baseline - 5.0) < 1e-9


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
    clint, fup = 0.5, 0.1  # FLUX-1 intrinsic clearance (q no longer wraps CL)
    c_out = 100.0 * 1.0 / (1.5 * 1.0)
    expected = fup * clint * c_out
    assert abs(rate_at_one - expected) < 1e-10


def _eval_clearance_rate_with_names(
    *,
    model: str,
    blood_name: str,
    liver_name: str,
    sink_name: str,
    fu_correction_applicable: float,
    fu_correction_liver: float,
    abundance: float = 10.0,
    flow_rate: float = 20.0,
    affinity: float = 5.0,
) -> float:
    """Build a 3-node graph with arbitrary node names and return the
    absolute mass-loss rate at the (flagged) liver-equivalent source.

    Mirrors `_eval_clearance_rate` exactly but parametrises every node
    name so the rename test can vary only the strings.
    """
    g = BodyGraph()
    g.add_node(Node(name=blood_name, node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(
        Node(
            name=liver_name,
            node_type="organ",
            volume=Distribution(1.5),
            enzymes={"CYP3A4": Distribution(abundance)},
            ivive_scaling=0.01,
            fu_correction_applicable=fu_correction_applicable,
        )
    )
    g.add_node(Node(name=sink_name, node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source=blood_name, target=liver_name, flow_rate=Distribution(flow_rate)))
    g.add_edge(ClearanceEdge(source=liver_name, target=sink_name, model=model))

    # State indices assigned alphabetically by node name (BodyGraph dict
    # preserves insertion order; we mirror the synthetic-graph pattern in
    # the helpers above which orders blood=0, liver=1, sink=2). To stay
    # robust to arbitrary names, derive the order from g.nodes insertion.
    node_names = list(g.nodes.keys())
    state_index = {name: i for i, name in enumerate(node_names)}
    liver_idx = state_index[liver_name]
    sink_idx = state_index[sink_name]
    spec = ClearanceFluxSpec.from_edge(1, g.edges[1], state_index)

    drug = _make_drug(
        enzyme_affinity={"CYP3A4": Distribution(affinity)},
        fu_correction_liver=Distribution(mean=fu_correction_liver, cv=0.0),
        kp_overrides={liver_name: Distribution(1.0)},
    )
    params = ResolvedParams(g, drug)

    y = np.zeros(3)
    y[liver_idx] = 100.0
    dydt = np.zeros(3)
    spec.apply(0.0, y, dydt, params)

    assert abs(dydt[liver_idx] + dydt[sink_idx]) < 1e-12
    return float(-dydt[liver_idx])


def test_identity_blind_random_rename_invariant():
    """B-11 invariant: ClearanceFlux apply at a flagged node produces
    bit-identical rates after every node in the BodyGraph is renamed to
    a random string (as long as the fu_correction_applicable flag travels
    with the renamed liver, which the synthetic-graph builder ensures).

    Mirrors tests/regression/test_prodrug_v2_identity_blind.py — but
    operates on the engine flux directly (no predict()) because the
    Task 6 patch lives entirely inside ClearanceFluxSpec.apply and the
    rate computation is the load-bearing invariant.

    Verifies that the engine reads node_param("fu_correction_applicable")
    from the per-node flag rather than string-matching the literal
    "liver". Any `if node_name == "liver"` branch inside the engine would
    break this test.
    """
    # Canonical names — the original engineering vocabulary.
    rate_canonical_corr = _eval_clearance_rate_with_names(
        model="well_stirred",
        blood_name="blood",
        liver_name="liver",
        sink_name="sink",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )

    # Random rename — every node string is replaced with an unrelated tag.
    rate_renamed_corr = _eval_clearance_rate_with_names(
        model="well_stirred",
        blood_name="Z1Q9K",
        liver_name="Q4M2X",
        sink_name="P5N7T",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )

    assert rate_canonical_corr == rate_renamed_corr, (
        f"Identity-blind violated (well_stirred, corrected): "
        f"canonical={rate_canonical_corr!r}, renamed={rate_renamed_corr!r}"
    )

    # And the gate-off case: same equality must hold whether or not the
    # flag is on, but renaming with the flag off must also be bit-identical.
    rate_canonical_off = _eval_clearance_rate_with_names(
        model="well_stirred",
        blood_name="blood",
        liver_name="liver",
        sink_name="sink",
        fu_correction_applicable=0.0,
        fu_correction_liver=5.0,
    )
    rate_renamed_off = _eval_clearance_rate_with_names(
        model="well_stirred",
        blood_name="Z1Q9K",
        liver_name="Q4M2X",
        sink_name="P5N7T",
        fu_correction_applicable=0.0,
        fu_correction_liver=5.0,
    )
    assert rate_canonical_off == rate_renamed_off, (
        f"Identity-blind violated (well_stirred, gate off): "
        f"canonical={rate_canonical_off!r}, renamed={rate_renamed_off!r}"
    )

    # Sanity: the corrected vs uncorrected paths must differ (otherwise
    # the test could be trivially satisfied by the gate never firing).
    assert rate_canonical_corr != rate_canonical_off

    # Parallel-tube branch: same invariance must hold.
    rate_pt_canonical = _eval_clearance_rate_with_names(
        model="parallel_tube",
        blood_name="blood",
        liver_name="liver",
        sink_name="sink",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )
    rate_pt_renamed = _eval_clearance_rate_with_names(
        model="parallel_tube",
        blood_name="K8B3Y",
        liver_name="X7R2L",
        sink_name="M3J9V",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )
    assert rate_pt_canonical == rate_pt_renamed, (
        f"Identity-blind violated (parallel_tube): "
        f"canonical={rate_pt_canonical!r}, renamed={rate_pt_renamed!r}"
    )


def test_prodrug_flux_applies_correction_at_flagged_node_via_predict(monkeypatch):
    """End-to-end: clopidogrel routes through ProdrugActivationFlux at the
    liver (flagged) node. fu_correction_liver > 1.0 must lower predicted parent
    Cmax. Verified via predict() because clopidogrel is a registered prodrug
    so its production path IS the one Task 7 modifies."""
    from sisyphus.pipeline.predict import predict

    # Patch at the source module so the function-local import in ivive.py
    # re-binds on every predict() call.
    clopidogrel = "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"  # non-stereo

    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=5.0, cv=0.0),
        raising=False,
    )
    high = predict(clopidogrel, dose_mg=300.0, route="oral")
    cmax_high = float(high.engine_pk.cmax.mean)

    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=1.0, cv=0.0),
        raising=False,
    )
    low = predict(clopidogrel, dose_mg=300.0, route="oral")
    cmax_low = float(low.engine_pk.cmax.mean)

    assert cmax_high < cmax_low, (
        f"ProdrugActivationFlux at liver must respect fu_correction_liver; "
        f"fu_corr=5: {cmax_high:.4f}, fu_corr=1: {cmax_low:.4f}"
    )


def test_extended_clearance_ignores_fu_correction_by_design():
    """The extended (ECM) clearance model intentionally does NOT apply
    fu_correction_liver, unlike well_stirred / parallel_tube.

    fu_correction_liver is fu_inc/fu_plasma — an empirical lumped correction
    for albumin-facilitated / transporter-mediated *concentrative hepatic
    uptake* (B-11 spec §2). The ECM models exactly that uptake mechanistically
    (PS_active/PS_inf/PS_eff), so multiplying fup by the WS-style factor on top
    would double-count. This test PINS that decision: a flagged node whose
    clearance edge is ``extended`` produces a bit-identical rate whether
    fu_correction_liver is 1.0 or 5.0. The PRODUCTION liver clearance edge is
    ``extended`` (reference_man.yaml), so this is the behavior on the real
    headline path. The matching contract guard — a runtime warning when a
    non-1.0 value is curated — lives in pipeline.predict.
    """
    rate_baseline = _eval_clearance_rate(
        model="extended",
        fu_correction_applicable=1.0,
        fu_correction_liver=1.0,
    )
    rate_corrected = _eval_clearance_rate(
        model="extended",
        fu_correction_applicable=1.0,
        fu_correction_liver=5.0,
    )

    # Sanity: the ECM sink is active (non-zero), else the equality is vacuous.
    assert rate_baseline > 0.0
    # fu_correction has NO effect on the extended branch — by design.
    assert rate_baseline == rate_corrected, (
        f"extended ECM must ignore fu_correction_liver (would double-count the "
        f"explicit PS uptake); baseline={rate_baseline!r}, "
        f"corrected={rate_corrected!r}"
    )


def test_predict_warns_when_fu_correction_dropped_at_extended_node(monkeypatch):
    """pipeline.predict surfaces a warning when a non-identity
    fu_correction_liver is curated for a drug whose flagged hepatic node clears
    via the extended ECM (which drops the correction by design).

    Closes the B-11 'silent no-op' trap: the production liver clearance edge is
    ``extended``, so without this warning a future curated value would vanish
    unnoticed and still pass every test. Patches the source-module lookup (the
    function-local import in ivive.py re-binds on every predict() call).
    """
    from sisyphus.pipeline.predict import predict

    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=2.0, cv=0.0),
        raising=False,
    )
    result = predict(caffeine, dose_mg=100.0, route="oral")

    assert any(
        "fu_correction_liver" in w and "not applied" in w for w in result.warnings
    ), f"expected dropped-fu_correction warning; got {result.warnings!r}"


def test_predict_no_fu_correction_warning_at_default(monkeypatch):
    """Control: with the default fu_correction_liver=1.0 (every shipped registry
    value today), NO warning is emitted — so the production headline path is
    bit-identical (the guard fires only on a curated non-1.0 value)."""
    from sisyphus.pipeline.predict import predict

    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=1.0, cv=0.0),
        raising=False,
    )
    result = predict(caffeine, dose_mg=100.0, route="oral")

    assert not any(
        "fu_correction_liver" in w and "not applied" in w for w in result.warnings
    ), f"no warning expected at default 1.0; got {result.warnings!r}"
