"""Tests for prodrug registry loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_registry(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(content))
    return p


def test_lookup_returns_none_for_missing_smiles(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {})
    assert lookup_active_metabolite("CCO", registry_path=p) is None


def test_lookup_returns_active_metabolite_for_match(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    # "CCO" is canonical SMILES for ethanol — use as test key
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "TestActive",
            "mw": 241.25,
            "fup": {"mean": 0.23, "cv": 0.3},
            "CL_per_h": {"mean": 40.0, "cv": 0.35},
            "Vd_L": {"mean": 150.0, "cv": 0.3},
            "conversion_rate_per_h": {"mean": 12.0, "cv": 0.4},
            "conversion_site": "gut_wall",
            "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
            "observation_species": "active",
        }
    })
    result = lookup_active_metabolite("CCO", registry_path=p)
    assert result is not None
    am, obs_species = result
    assert am.name == "TestActive"
    assert obs_species == "active"


def test_lookup_default_observation_species_is_active(tmp_path):
    """If observation_species not specified, default to 'active'."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
        }
    })
    result = lookup_active_metabolite("CCO", registry_path=p)
    assert result is not None
    _, obs_species = result
    assert obs_species == "active"


def test_lookup_negative_rate_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": -1.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
        }
    })
    with pytest.raises(ValueError, match="conversion_rate must be positive"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_yield_out_of_range_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.5},
        }
    })
    with pytest.raises(ValueError, match="conversion_yield must be in"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_invalid_observation_species_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
            "observation_species": "middle",
        }
    })
    with pytest.raises(ValueError, match="observation_species must be"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_canonical_smiles_match(tmp_path):
    """SMILES is canonicalized via RDKit before lookup. 'OCC' canonicalizes to 'CCO'."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {"CCO": {
        "name": "X", "mw": 100.0,
        "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
        "Vd_L": {"mean": 50.0},
        "conversion_rate_per_h": {"mean": 5.0},
        "conversion_site": "venous_blood",
        "conversion_yield_fraction": {"mean": 1.0},
    }})
    result = lookup_active_metabolite("OCC", registry_path=p)
    assert result is not None  # canonicalized matches


def test_lookup_invalid_smiles_returns_none(tmp_path):
    """Non-parseable SMILES → None (not a crash)."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {})
    assert lookup_active_metabolite("not-a-smiles!", registry_path=p) is None


def test_lookup_missing_required_field_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            # missing fup, CL_per_h, etc.
        }
    })
    with pytest.raises(ValueError, match="missing field"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_actual_registry_loads_4_entries():
    """Verify the production registry contains 4 valid entries."""
    from sisyphus.predict.registry import _DEFAULT_REGISTRY_PATH
    import json
    with _DEFAULT_REGISTRY_PATH.open() as f:
        registry = json.load(f)
    # Filter out comment-style keys (those starting with _)
    entries = {k: v for k, v in registry.items() if not k.startswith("_")}
    assert len(entries) == 4, f"Expected 4 prodrug entries, got {len(entries)}"
    expected_names = {"BH4", "GS-441524", "tebipenem", "R406"}
    actual_names = {v["name"] for v in entries.values()}
    assert actual_names == expected_names
