"""FLUX-1 regression: perfusion-compartment extraction must approach 1.0, not 0.5.

A clearing organ modeled as a perfusion compartment carries an explicit
convective outflow ``Q·c_out`` (FlowEdge) alongside its ClearanceEdge. The
clearance sink must therefore apply the *intrinsic* (flow-unlimited) clearance:
the emergent steady-state extraction is then

    E = clearance_flux / (clearance_flux + convective_flux)
      = fup·CLint / (Q + fup·CLint)   →  1.0  as fup·CLint → ∞

The pre-fix bug applied the *whole-organ* CL_h (which already embeds Q) to
``c_out``, double-counting the flow term and capping E at 0.5:

    E_bug = fup·CLint / (Q + 2·fup·CLint)  →  0.5

Both the ``well_stirred`` (gut) and ``extended``/ECM (production liver) paths
are covered. See docs/superpowers/specs/2026-06-03-flux1-extraction-double-count-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ClearanceFluxSpec, FlowFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node

_Q = 99.45  # L/h, liver inflow (0.255 × 390)
_IVIVE = 6e-5
_ABUNDANCE = 9.2475e6
_FUP = 0.1


def _drug(affinity: float) -> DrugOnGraph:
    return DrugOnGraph(
        name="t", smiles="C", dose_mg=1.0, route="iv",
        administration_node="blood_src", mw=500.0, pka=None,
        compound_type="neutral", fup=Distribution(_FUP), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={}, peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={"CYP3A4": Distribution(affinity)} if affinity > 0 else {},
        renal_clearance=Distribution(0.0), transporter_kinetics={},
        ps_passive=Distribution(1e6), ps_eff=Distribution(1e6),
        cl_int_bile=Distribution(0.0),
    )


def _realized_extraction(model: str, affinity: float) -> float:
    """Realized steady-state extraction = clearance / (clearance + convective).

    Both fluxes share the same ``c_out``, so their ratio is the realized E
    independent of volume/Kp/concentration.
    """
    g = BodyGraph()
    g.add_node(Node(name="blood_src", node_type="blood_pool", volume=Distribution(1.5)))
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.80),
                    enzymes={"CYP3A4": Distribution(_ABUNDANCE)}, ivive_scaling=_IVIVE))
    g.add_node(Node(name="venous", node_type="blood_pool", volume=Distribution(1.5)))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1e10)))
    g.add_edge(FlowEdge(source="blood_src", target="liver", flow_rate=Distribution(_Q)))
    g.add_edge(FlowEdge(source="liver", target="venous", flow_rate=Distribution(_Q)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model=model))
    params = ResolvedParams(g, _drug(affinity))
    si = {"blood_src": 0, "liver": 1, "venous": 2, "sink": 3}
    flow_out = FlowFluxSpec.from_edge(1, g.edges[1], si)
    clr = ClearanceFluxSpec.from_edge(2, g.edges[2], si)
    y = np.array([0.0, 1.0, 0.0, 0.0])
    d_conv = np.zeros(4)
    flow_out.apply(0.0, y, d_conv, params)
    d_clr = np.zeros(4)
    clr.apply(0.0, y, d_clr, params)
    conv, clear = -d_conv[1], -d_clr[1]
    return clear / (clear + conv) if (clear + conv) > 0 else 0.0


@pytest.mark.parametrize("model", ["well_stirred", "extended"])
def test_high_clint_extraction_approaches_one(model):
    # affinity=100 → fup·CLint ≈ 5548: canonical E = 0.982. Bug caps at 0.495.
    E = _realized_extraction(model, affinity=100.0)
    assert E > 0.90, f"{model}: extraction {E:.4f} capped below 0.90 (flow double-count)"


@pytest.mark.parametrize("model", ["well_stirred", "extended"])
def test_extraction_matches_canonical_well_stirred(model):
    # Across a CLint sweep, realized E must equal x/(Q+x), not x/(Q+2x).
    for affinity in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        x = _FUP * _ABUNDANCE * affinity * _IVIVE  # fup·CLint
        E = _realized_extraction(model, affinity)
        assert E == pytest.approx(x / (_Q + x), rel=2e-3), (
            f"{model} affinity={affinity}: E={E:.4f} != canonical {x/(_Q+x):.4f}"
        )


def test_low_clint_regime_unchanged():
    # At fup·CLint << Q the two formulas coincide; fix must not perturb here.
    E = _realized_extraction("well_stirred", affinity=1e-4)
    x = _FUP * _ABUNDANCE * 1e-4 * _IVIVE
    assert E == pytest.approx(x / (_Q + x), rel=1e-3)
