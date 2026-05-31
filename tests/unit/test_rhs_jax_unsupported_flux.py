"""Audit follow-up: the JAX RHS must not silently drop unsupported flux types.

``ProdrugActivationFluxSpec`` and ``OneCompartmentEliminationFluxSpec`` have no
branch in ``make_jax_rhs``'s isinstance chain. Before the
``_unsupported_flux_specs`` guard they fell through the build loop silently, so a
graph with a prodrug-activation or active-metabolite edge would have that flux
omitted from the JAX RHS with no error (the SciPy backend handles them
correctly). These tests pin a loud failure instead of silent corruption.

The guard is pure Python (no JAX import) so it runs even though JAX is optional
and absent from ``requirements-lock.txt``.
"""
from __future__ import annotations

from sisyphus.engine.flux import (
    FlowFluxSpec,
    OneCompartmentEliminationFluxSpec,
    ProdrugActivationFluxSpec,
)
from sisyphus.engine.rhs_jax import _unsupported_flux_specs


def _prodrug() -> ProdrugActivationFluxSpec:
    return ProdrugActivationFluxSpec(
        0, 0, 1, "parent", "active", frozenset({"CYP3A4"}), 1.0
    )


def _one_compartment() -> OneCompartmentEliminationFluxSpec:
    return OneCompartmentEliminationFluxSpec(0, 0, 1, "active", "sink")


def test_flags_prodrug_activation():
    spec = _prodrug()
    assert _unsupported_flux_specs([spec]) == [spec]


def test_flags_one_compartment_elimination():
    spec = _one_compartment()
    assert _unsupported_flux_specs([spec]) == [spec]


def test_supported_flux_not_flagged():
    flow = FlowFluxSpec(0, 0, 1, "a", "b")
    assert _unsupported_flux_specs([flow]) == []


def test_mixed_returns_only_unsupported():
    flow = FlowFluxSpec(0, 0, 1, "a", "b")
    prod = _prodrug()
    one = _one_compartment()
    out = _unsupported_flux_specs([flow, prod, one])
    assert flow not in out
    assert prod in out
    assert one in out
