"""Tests for ProdrugActivationFluxSpec."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.engine.flux import FLUX_REGISTRY
from sisyphus.graph.types import ProdrugActivationEdge


def test_prodrug_activation_registered():
    """ProdrugActivationFluxSpec is registered for 'prodrug_activation' edge type."""
    assert "prodrug_activation" in FLUX_REGISTRY


def test_prodrug_activation_apply_math():
    """apply() depletes parent at src; produces active at tgt with MW × yield scaling."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0),
        conversion_yield=Distribution(mean=0.85),
        mw_parent=237.26, mw_active=241.25,
    )
    state_index = {"gut_wall": 0, "venous_blood_active": 1}
    spec = ProdrugActivationFluxSpec.from_edge(0, edge, state_index)

    # Mock ResolvedParams: returns conversion_rate=12 and conversion_yield=0.85
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "conversion_rate": 12.0, "conversion_yield": 0.85}[p]

    y = np.array([10.0, 0.0])  # 10 mg parent at gut_wall
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)

    # Parent flux: k × A = 12 × 10 = 120 mg_parent/h
    # Active flux: 120 × (241.25/237.26) × 0.85 ≈ 103.72 mg_active/h
    expected_parent_loss = 12.0 * 10.0
    expected_active_gain = expected_parent_loss * (241.25 / 237.26) * 0.85
    assert dydt[0] == pytest.approx(-expected_parent_loss)
    assert dydt[1] == pytest.approx(expected_active_gain)


def test_prodrug_activation_zero_yield_no_active():
    """yield=0 → src still depleted, but tgt gets nothing."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="src", target="tgt",
        conversion_rate=Distribution(mean=5.0),
        conversion_yield=Distribution(mean=0.0),
        mw_parent=100.0, mw_active=100.0,
    )
    spec = ProdrugActivationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "conversion_rate": 5.0, "conversion_yield": 0.0}[p]
    y = np.array([4.0, 0.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    assert dydt[0] == pytest.approx(-20.0)  # src depleted
    assert dydt[1] == 0.0                   # tgt unchanged


def test_prodrug_activation_invalid_mw_raises():
    """from_edge raises if mw_parent is non-positive (division would fail)."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="a", target="b",
        conversion_rate=Distribution(1.0), conversion_yield=Distribution(1.0),
        mw_parent=0.0, mw_active=100.0,  # invalid
    )
    with pytest.raises(ValueError, match="mw_parent must be positive"):
        ProdrugActivationFluxSpec.from_edge(0, edge, {"a": 0, "b": 1})


def test_prodrug_activation_zero_amount_no_flux():
    """A=0 → no flux."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="src", target="tgt",
        conversion_rate=Distribution(mean=5.0),
        conversion_yield=Distribution(mean=1.0),
        mw_parent=100.0, mw_active=100.0,
    )
    spec = ProdrugActivationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "conversion_rate": 5.0, "conversion_yield": 1.0}[p]
    y = np.zeros(2)
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    assert dydt[0] == 0.0
    assert dydt[1] == 0.0


from sisyphus.graph.types import OneCompartmentEliminationEdge


def test_one_compartment_elimination_registered():
    assert "one_compartment_elimination" in FLUX_REGISTRY


def test_one_compartment_elimination_apply_math():
    """apply() removes mass at rate (CL/Vd) × A from source, adds to target."""
    from sisyphus.engine.flux import OneCompartmentEliminationFluxSpec

    edge = OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0),
        vd_l=Distribution(mean=150.0),
    )
    state_index = {"venous_blood_active": 0, "metabolized_gut": 1}
    spec = OneCompartmentEliminationFluxSpec.from_edge(0, edge, state_index)

    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "cl_per_h": 40.0, "vd_l": 150.0}[p]

    y = np.array([30.0, 0.0])  # 30 mg active
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)

    # rate = (CL/Vd) × A = (40/150) × 30 = 8.0 mg/h
    expected_rate = (40.0 / 150.0) * 30.0
    assert dydt[0] == pytest.approx(-expected_rate)
    assert dydt[1] == pytest.approx(expected_rate)


def test_one_compartment_elimination_zero_amount():
    """A=0 → no flux."""
    from sisyphus.engine.flux import OneCompartmentEliminationFluxSpec

    edge = OneCompartmentEliminationEdge(
        source="src", target="tgt",
        cl_per_h=Distribution(mean=10.0), vd_l=Distribution(mean=50.0),
    )
    spec = OneCompartmentEliminationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {"cl_per_h": 10.0, "vd_l": 50.0}[p]
    y = np.zeros(2)
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    assert dydt[0] == 0.0
    assert dydt[1] == 0.0


def test_one_compartment_elimination_zero_vd_safe():
    """vd=0 (degenerate) → no flux instead of division-by-zero crash."""
    from sisyphus.engine.flux import OneCompartmentEliminationFluxSpec

    edge = OneCompartmentEliminationEdge(
        source="src", target="tgt",
        cl_per_h=Distribution(mean=10.0), vd_l=Distribution(mean=0.0),
    )
    spec = OneCompartmentEliminationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {"cl_per_h": 10.0, "vd_l": 0.0}[p]
    y = np.array([30.0, 0.0])
    dydt = np.zeros(2)
    # Should not raise ZeroDivisionError
    spec.apply(0.0, y, dydt, params)
    # Behavior: no flux when vd is non-positive
    assert dydt[0] == 0.0
    assert dydt[1] == 0.0
