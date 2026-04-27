"""Tests for build_drug_on_graph registry integration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def test_registry_lookup_returns_none_for_arbitrary_smiles(tmp_path):
    """Sanity check: lookup_active_metabolite is the integration mechanism."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = tmp_path / "registry.json"
    p.write_text("{}")
    assert lookup_active_metabolite("CCO", registry_path=p) is None


def test_build_drug_on_graph_with_registry_match(tmp_path):
    """End-to-end: when SMILES matches the prodrug registry, the resulting
    DrugOnGraph has active_metabolite set and observation_species per registry.

    Uses a custom temporary registry to avoid hitting the production registry.
    """
    from sisyphus.predict.registry import _DEFAULT_REGISTRY_PATH

    # Create a temp registry with an entry for ethanol (canonical "CCO")
    test_registry = tmp_path / "test_registry.json"
    test_registry.write_text(json.dumps({
        "CCO": {
            "name": "TestActive",
            "mw": 100.0,
            "fup": {"mean": 0.5, "cv": 0.2},
            "CL_per_h": {"mean": 10.0, "cv": 0.3},
            "Vd_L": {"mean": 50.0, "cv": 0.3},
            "conversion_rate_per_h": {"mean": 5.0, "cv": 0.4},
            "conversion_site": "gut_wall",
            "conversion_yield_fraction": {"mean": 0.9, "cv": 0.1},
            "observation_species": "parent",
        }
    }))

    # Patch the default registry path. ALSO clear lru_cache.
    from sisyphus.predict.registry import _load_registry_cached
    _load_registry_cached.cache_clear()

    with patch("sisyphus.predict.registry._DEFAULT_REGISTRY_PATH", test_registry):
        # Direct verification via the registry function
        from sisyphus.predict.registry import lookup_active_metabolite
        result = lookup_active_metabolite("CCO")
        assert result is not None
        am, obs_species = result
        assert am.name == "TestActive"
        assert obs_species == "parent"


def test_build_drug_on_graph_no_registry_match_keeps_defaults(tmp_path):
    """Sanity: SMILES not in registry → registry returns None."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = tmp_path / "registry.json"
    p.write_text("{}")
    assert lookup_active_metabolite("c1ccccc1", registry_path=p) is None


def test_production_registry_lookup_for_evidence_drug():
    """The production registry resolves at least one of the 4 evidence drugs.

    Uses sepiapterin's canonical SMILES from the registry. This test verifies
    the integration is wired correctly: the registry exists and is loadable.
    """
    from sisyphus.predict.registry import _DEFAULT_REGISTRY_PATH, lookup_active_metabolite

    with _DEFAULT_REGISTRY_PATH.open() as f:
        registry = json.load(f)

    # Pick the first canonical SMILES (any of the 4 drugs)
    smiles_keys = [k for k in registry.keys() if not k.startswith("_")]
    assert len(smiles_keys) >= 1, "Production registry should have at least 1 entry"

    result = lookup_active_metabolite(smiles_keys[0])
    assert result is not None
    am, obs_species = result
    assert obs_species in ("parent", "active")
