"""RBP-2: blood:plasma concentration-basis consistency across flux types.

Single convention (spec 2026-06-04-rbp-concentration-basis-design.md):
a node's ``A/V`` is **whole-blood** concentration (blood-pool volumes are
whole-blood volumes). Therefore:

- convective FLOW carries blood: blood-pool source -> ``A/V`` (no RBP);
  tissue source -> ``A*RBP/(V*Kp)`` (unchanged).
- the metabolic/renal CLEARANCE sink acts on unbound PLASMA:
  ``rate = (fup or renal_cl-equiv) * A/(V*Kp)`` (no RBP) -> the emergent
  hepatic extraction is ``E = fup*CLint/(Q*RBP + fup*CLint) =
  fu_b*CLint/(Q + fu_b*CLint)`` with blood unbound ``fu_b = fup/RBP``
  (Pang & Han 2019, the canonical well-stirred form).
- DIFFUSION already forms the blood unbound conc (``fup*c_vasc/RBP``) -> unchanged.

These tests use a synthetic RBP != 1 (no production drug currently realises
RBP != 1 — the RBP model resets out-of-band predictions to 1.0 — so the whole
fix is a bit-identical no-op on the holdout and the fix is pure correctness).
"""

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ClearanceFluxSpec, FlowFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node


def _make_drug(**overrides) -> DrugOnGraph:
    defaults = dict(
        name="test", smiles="C", dose_mg=100.0, route="oral",
        administration_node="a", mw=100.0, pka=None, compound_type="neutral",
        fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={},
        peff=Distribution(1.0), solubility=Distribution(10.0),
        enzyme_affinity={}, renal_clearance=Distribution(0.0),
        particle_radius_um=25.0, ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


def _flow_flux(source_type: str, rbp: float, kp: float = 1.0) -> float:
    """Return -dydt[source] for a single flow edge from *source_type* node."""
    g = BodyGraph()
    g.add_node(Node(name="src", node_type=source_type, volume=Distribution(2.0)))
    g.add_node(Node(name="dst", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_edge(FlowEdge(source="src", target="dst", flow_rate=Distribution(10.0)))
    drug = _make_drug(rbp=Distribution(rbp), kp_overrides={"src": Distribution(kp)})
    params = ResolvedParams(g, drug)
    state_index = {"dst": 0, "src": 1}
    spec = FlowFluxSpec.from_edge(0, g.edges[0], state_index)
    y = np.array([0.0, 100.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    return -dydt[1]


class TestFlowConcentrationBasis:
    def test_blood_pool_source_flow_is_rbp_independent(self):
        """Convection of a blood pool carries whole blood A/V — no RBP factor."""
        f1 = _flow_flux("blood_pool", rbp=1.0)
        f2 = _flow_flux("blood_pool", rbp=2.0)
        # blood-pool A/V is already blood; flux = q*(A/V) = 10*(100/2) = 500
        assert abs(f1 - 500.0) < 1e-9
        assert abs(f2 - f1) < 1e-9, "blood-pool flow must not scale with RBP"

    def test_tissue_source_flow_keeps_rbp(self):
        """Tissue convective outlet is blood = A*RBP/(V*Kp) — RBP factor retained."""
        f1 = _flow_flux("organ", rbp=1.0, kp=1.0)
        f2 = _flow_flux("organ", rbp=2.0, kp=1.0)
        assert abs(f1 - 500.0) < 1e-9  # 10*(100*1/(2*1))
        assert abs(f2 - 2.0 * f1) < 1e-9, "tissue flow outlet must scale with RBP"


def _well_stirred_rate(rbp: float) -> float:
    g = BodyGraph()
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.5),
                    enzymes={"CYP3A4": Distribution(100.0)}, ivive_scaling=0.01))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))
    drug = _make_drug(kp_overrides={"liver": Distribution(2.0)},
                      enzyme_affinity={"CYP3A4": Distribution(50.0)},
                      rbp=Distribution(rbp))
    params = ResolvedParams(g, drug)
    state_index = {"liver": 0, "sink": 1}
    spec = ClearanceFluxSpec.from_edge(0, g.edges[0], state_index)
    y = np.array([50.0, 0.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    return dydt[1]  # rate into sink


def _gfr_rate(rbp: float) -> float:
    g = BodyGraph()
    g.add_node(Node(name="kidney", node_type="organ", volume=Distribution(0.3)))
    g.add_node(Node(name="urine", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="kidney", target="urine", model="gfr_filtration"))
    drug = _make_drug(renal_clearance=Distribution(7.2),
                      kp_overrides={"kidney": Distribution(1.0)}, rbp=Distribution(rbp))
    params = ResolvedParams(g, drug)
    state_index = {"kidney": 0, "urine": 1}
    spec = ClearanceFluxSpec.from_edge(0, g.edges[0], state_index)
    y = np.array([30.0, 0.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    return dydt[1]


class TestClearanceConcentrationBasis:
    def test_well_stirred_sink_drives_off_plasma(self):
        """Metabolic sink uses unbound plasma fup*A/(V*Kp) — RBP-independent at fixed A."""
        r1 = _well_stirred_rate(1.0)
        r2 = _well_stirred_rate(2.0)
        # fup*CLint*A/(V*Kp) = 0.5*(100*50*0.01)*50/(1.5*2) = 25*16.667 = 416.67
        assert abs(r1 - 416.6666666667) < 1e-6
        assert abs(r2 - r1) < 1e-9, "metabolic sink must not scale with RBP (plasma basis)"

    def test_gfr_sink_drives_off_plasma(self):
        """GFR filtration uses plasma A/(V*Kp) — RBP-independent at fixed A."""
        r1 = _gfr_rate(1.0)
        r2 = _gfr_rate(2.0)
        assert abs(r1 - 720.0) < 1e-9  # 7.2 * 30/(0.3*1)
        assert abs(r2 - r1) < 1e-9, "renal filtration must not scale with RBP (plasma basis)"
