"""Tests for OneCompartmentEliminationFluxSpec.

v1 ProdrugActivationFluxSpec kinetic-rate tests removed (architecture
replaced by well-stirred extraction in v2; see test_prodrug_v2_flux.py).
We keep the FLUX_REGISTRY membership check for prodrug_activation since
the registration string is identical across v1/v2.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.engine.flux import FLUX_REGISTRY
from sisyphus.graph.types import OneCompartmentEliminationEdge


def test_prodrug_activation_registered():
    """ProdrugActivationFluxSpec is registered for 'prodrug_activation' edge type."""
    assert "prodrug_activation" in FLUX_REGISTRY


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
