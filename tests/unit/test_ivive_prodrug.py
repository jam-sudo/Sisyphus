"""Tests for build_drug_on_graph registry integration.

v1 lookup-shape tests (2-tuple, conversion_rate_per_h schema) removed in
T18 (2026-04-28). v2 returns a 3-tuple ``(am, obs, affinities)``; v2
schema requires ``affinity_source`` and ``yield_source`` enums. See
``tests/unit/test_prodrug_v2_registry.py`` for the full v2 contract.
"""
from __future__ import annotations


def test_registry_lookup_returns_none_for_arbitrary_smiles(tmp_path):
    """Sanity check: lookup_active_metabolite is the integration mechanism."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = tmp_path / "registry.json"
    p.write_text("{}")
    assert lookup_active_metabolite("CCO", registry_path=p) is None


def test_build_drug_on_graph_no_registry_match_keeps_defaults(tmp_path):
    """Sanity: SMILES not in registry → registry returns None."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = tmp_path / "registry.json"
    p.write_text("{}")
    assert lookup_active_metabolite("c1ccccc1", registry_path=p) is None
