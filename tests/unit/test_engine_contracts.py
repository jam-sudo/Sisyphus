"""Unit tests for engine-level contract validation (WS-2)."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.engine.contracts import (
    assert_fu_correction_honored,
    flagged_nodes_without_honoring_flux,
)
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    ClearanceEdge,
    Node,
    ProdrugActivationEdge,
)


def _node(name: str, flagged: bool) -> Node:
    return Node(
        name=name,
        node_type="organ",
        volume=Distribution(1.0),
        ivive_scaling=1.0e-4,
        fu_correction_applicable=1.0 if flagged else 0.0,
    )


def _graph_extended_only_flagged() -> BodyGraph:
    g = BodyGraph()
    g.add_node(_node("liver", flagged=True))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="extended"))
    return g


def _graph_extended_plus_prodrug_flagged() -> BodyGraph:
    g = _graph_extended_only_flagged()
    g.add_node(Node(name="active", node_type="organ", volume=Distribution(1.0)))
    g.add_edge(
        ProdrugActivationEdge(
            source="liver",
            target="active",
            enzyme_tags=frozenset({"CYP3A4"}),
            conversion_yield=Distribution(1.0),
            mw_parent=300.0,
            mw_active=280.0,
        )
    )
    return g


def test_flagged_extended_only_is_an_offender():
    g = _graph_extended_only_flagged()
    assert flagged_nodes_without_honoring_flux(g) == ["liver"]


def test_prodrug_coexistence_is_not_an_offender():
    g = _graph_extended_plus_prodrug_flagged()
    assert flagged_nodes_without_honoring_flux(g) == []


def test_unflagged_extended_is_not_an_offender():
    g = BodyGraph()
    g.add_node(_node("liver", flagged=False))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="extended"))
    assert flagged_nodes_without_honoring_flux(g) == []


def test_assert_raises_on_nonidentity_total_drop():
    g = _graph_extended_only_flagged()
    with pytest.raises(ValueError, match="entirely dropped"):
        assert_fu_correction_honored(g, fu_correction_liver_mean=1.4)


def test_assert_noop_on_identity_value():
    g = _graph_extended_only_flagged()
    assert_fu_correction_honored(g, fu_correction_liver_mean=1.0)  # no raise


def test_assert_noop_when_prodrug_honors_it():
    g = _graph_extended_plus_prodrug_flagged()
    assert_fu_correction_honored(g, fu_correction_liver_mean=1.4)  # no raise


def test_pipeline_raises_when_curated_value_would_be_dropped(monkeypatch):
    """A non-1.0 fu_correction_liver on the production (extended) liver raises."""
    import sisyphus.predict.hepatic_fu_correction as hfc
    from sisyphus.core import Distribution
    from sisyphus.pipeline.predict import predict

    # Force a non-identity hepatic fu_correction for any SMILES.
    monkeypatch.setattr(
        hfc, "lookup_hepatic_fu_correction", lambda smiles: Distribution(1.4, cv=0.0)
    )
    with pytest.raises(ValueError, match="entirely dropped"):
        predict("CCO", dose_mg=100.0, route="oral")


def test_non_contract_engine_valueerror_degrades_to_ml_fallback(monkeypatch):
    """A generic ValueError from the engine (e.g. compile) must NOT hard-abort
    predict(); it degrades to the ML-only fallback (the WS-2 narrow re-raise only
    propagates the fu_correction contract violation)."""
    import sisyphus.engine.compiler as comp
    from sisyphus.pipeline.predict import predict

    def _boom(self, graph):
        raise ValueError("synthetic compile failure")

    monkeypatch.setattr(comp.ODECompiler, "compile", _boom)
    result = predict("CCO", dose_mg=100.0, route="oral")
    assert result is not None  # degraded, did not raise
    assert any("Engine failed" in w or "failed" in w.lower() for w in result.warnings)
