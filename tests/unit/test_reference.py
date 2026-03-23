"""Unit tests for validation/reference.py — reference data loading."""

import json
from pathlib import Path

import pytest

from sisyphus.validation.reference import DrugReference, load_reference


def _make_ref_data() -> dict:
    """Create minimal clinical_pk.json content."""
    return {
        "metadata": {"n_drugs": 3},
        "drugs": {
            "caffeine": {
                "name": "caffeine",
                "tier": "platinum",
                "dose_mg": 100.0,
                "route": "oral",
                "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "pk_params": {
                    "cmax_mg_L": 1.5,
                    "auc_mg_h_L": 10.0,
                },
            },
            "no_smiles_drug": {
                "name": "no_smiles_drug",
                "tier": "silver",
                "dose_mg": 50.0,
                "route": "oral",
                "pk_params": {"cmax_mg_L": 2.0},
            },
            "no_cmax_drug": {
                "name": "no_cmax_drug",
                "tier": "silver",
                "dose_mg": 50.0,
                "route": "oral",
                "smiles": "CC",
                "pk_params": {},
            },
        },
    }


class TestLoadReference:
    def test_loads_valid_drugs(self, tmp_path: Path):
        ref_path = tmp_path / "clinical_pk.json"
        holdout_path = tmp_path / "holdout.json"
        ref_path.write_text(json.dumps(_make_ref_data()))
        holdout_path.write_text(json.dumps({"train": ["caffeine"], "holdout": []}))

        results = load_reference(ref_path)
        # Only caffeine should load (no_smiles_drug lacks smiles, no_cmax_drug lacks cmax)
        assert len(results) == 1
        assert results[0].name == "caffeine"
        assert results[0].smiles == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        assert results[0].dose_mg == 100.0
        assert results[0].cmax_obs == 1.5
        assert results[0].auc_obs == 10.0

    def test_skips_drugs_without_smiles(self, tmp_path: Path):
        ref_path = tmp_path / "clinical_pk.json"
        holdout_path = tmp_path / "holdout.json"
        data = {
            "drugs": {
                "no_smiles": {
                    "dose_mg": 50.0,
                    "route": "oral",
                    "pk_params": {"cmax_mg_L": 2.0},
                }
            }
        }
        ref_path.write_text(json.dumps(data))
        holdout_path.write_text(json.dumps({"train": [], "holdout": []}))
        results = load_reference(ref_path)
        assert len(results) == 0

    def test_drug_reference_frozen(self):
        dr = DrugReference(
            name="test",
            smiles="C",
            dose_mg=10.0,
            route="oral",
            cmax_obs=1.0,
            auc_obs=None,
            in_holdout=False,
            in_training=True,
        )
        with pytest.raises(AttributeError):
            dr.name = "changed"  # type: ignore[misc]

    def test_loads_from_default_path(self):
        """Smoke test: loading from the actual data directory should work."""
        results = load_reference()
        assert len(results) > 0
        # Every loaded record should have smiles and positive cmax
        for r in results:
            assert r.smiles
            assert r.cmax_obs > 0
