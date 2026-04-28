"""Unit tests for v2 prodrug registry schema + loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisyphus.core import ActiveMetabolite
from sisyphus.predict.registry import lookup_active_metabolite


def _write_registry(tmp_path: Path, entries: dict) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(entries))
    return p


def _v2_entry(**overrides) -> dict:
    base = {
        "name": "BH4",
        "mw": 241.25,
        "fup": {"mean": 0.23, "cv": 0.3},
        "CL_per_h": {"mean": 40.0, "cv": 0.35},
        "Vd_L": {"mean": 150.0, "cv": 0.3},
        "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
        "yield_source": "literature",
        "observation_species": "parent",
        "enzyme_affinity_for_conversion": {
            "SPR": {"mean": 50.0, "cv": 0.5}
        },
        "affinity_source": "literature",
    }
    base.update(overrides)
    return base


def test_lookup_returns_three_tuple(tmp_path):
    smiles = "C"
    canonical = "C"
    reg = _write_registry(tmp_path, {canonical: _v2_entry()})
    result = lookup_active_metabolite(smiles, registry_path=reg)
    assert result is not None
    assert len(result) == 3
    am, obs, affinities = result
    assert isinstance(am, ActiveMetabolite)
    assert obs == "parent"
    assert "SPR" in affinities
    assert affinities["SPR"].mean == 50.0
    assert affinities["SPR"].cv == 0.5


def test_lookup_returns_none_for_unknown_smiles(tmp_path):
    reg = _write_registry(tmp_path, {})
    assert lookup_active_metabolite("CCO", registry_path=reg) is None


def test_loader_rejects_infrastructure_only(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(affinity_source="infrastructure_only")})
    with pytest.raises(ValueError, match="affinity_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_unknown_affinity_source(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(affinity_source="bogus")})
    with pytest.raises(ValueError, match="affinity_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_empty_enzyme_affinity_for_conversion(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(enzyme_affinity_for_conversion={})})
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_negative_vd(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(Vd_L={"mean": -1.0, "cv": 0.0})})
    with pytest.raises(ValueError, match="Vd"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_unknown_yield_source(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(yield_source="bogus")})
    with pytest.raises(ValueError, match="yield_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_strips_citation_keys_from_distribution(tmp_path):
    """Distribution loader must ignore extra 'citation' keys in affinity entries."""
    entry = _v2_entry(
        enzyme_affinity_for_conversion={
            "SPR": {"mean": 50.0, "cv": 0.5, "citation": "Park 2008"}
        }
    )
    reg = _write_registry(tmp_path, {"C": entry})
    result = lookup_active_metabolite("C", registry_path=reg)
    assert result is not None
    _, _, affinities = result
    assert affinities["SPR"].mean == 50.0
