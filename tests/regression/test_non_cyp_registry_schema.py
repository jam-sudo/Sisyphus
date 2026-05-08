"""Schema regression for nat2_substrates.json + ugt1a1_substrates.json.

Five gates per registry plus cross-cutting holdout-disjoint check.
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_NAT2_PATH = _REPO_ROOT / "data" / "enzymes" / "nat2_substrates.json"
_UGT1A1_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a1_substrates.json"
_YAML_PATH = _REPO_ROOT / "data" / "physiology" / "reference_man.yaml"
_HOLDOUT_PATH = _REPO_ROOT / "data" / "reference" / "holdout.json"


_EXPECTED_NAT2 = frozenset({"isoniazid", "hydralazine", "procainamide"})
_EXPECTED_UGT1A1 = frozenset({"raltegravir", "atazanavir", "dolutegravir"})


def _load(path):
    return json.loads(path.read_text())


def test_nat2_seed_pinned():
    data = _load(_NAT2_PATH)
    actual = {s["drug"] for s in data["substrates"]}
    assert actual == _EXPECTED_NAT2, (
        f"NAT2 seed drift: expected {_EXPECTED_NAT2}, got {actual}. "
        f"Update _EXPECTED_NAT2 with explicit decision (literature mf, "
        f"holdout check, integration test)."
    )


def test_ugt1a1_seed_pinned():
    data = _load(_UGT1A1_PATH)
    actual = {s["drug"] for s in data["substrates"]}
    assert actual == _EXPECTED_UGT1A1, (
        f"UGT1A1 seed drift: expected {_EXPECTED_UGT1A1}, got {actual}."
    )


def test_nat2_inchikey_matches_smiles():
    data = _load(_NAT2_PATH)
    for s in data["substrates"]:
        m = Chem.MolFromSmiles(s["smiles"])
        assert m is not None, f"{s['drug']}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(m)
        assert derived == s["inchikey"], (
            f"{s['drug']}: registered InChIKey {s['inchikey']} != "
            f"RDKit-derived {derived}"
        )


def test_ugt1a1_inchikey_matches_smiles():
    data = _load(_UGT1A1_PATH)
    for s in data["substrates"]:
        m = Chem.MolFromSmiles(s["smiles"])
        assert m is not None, f"{s['drug']}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(m)
        assert derived == s["inchikey"]


def test_nat2_metabolic_fraction_in_range():
    data = _load(_NAT2_PATH)
    for s in data["substrates"]:
        mf = s["metabolic_fraction"]
        assert 0.0 <= mf <= 1.0, f"{s['drug']}: mf={mf} not in [0, 1]"


def test_ugt1a1_metabolic_fraction_in_range():
    data = _load(_UGT1A1_PATH)
    for s in data["substrates"]:
        mf = s["metabolic_fraction"]
        assert 0.0 <= mf <= 1.0, f"{s['drug']}: mf={mf} not in [0, 1]"


def test_yaml_has_nat2_and_ugt1a1_in_liver_enzymes():
    """Schema gate: both registries' target enzymes must be in liver.enzymes."""
    text = _YAML_PATH.read_text()
    # Cheap substring check — full YAML parse not necessary for presence.
    assert "NAT2:" in text and "UGT1A1:" in text, (
        f"Expected NAT2: and UGT1A1: in {_YAML_PATH}; otherwise apply_phenotype_to_graph "
        f"will warn 'tag not found' for these genes."
    )


def test_no_registry_drug_in_holdout():
    """Cross-cutting: registry must not include any 107-holdout drug.

    If you intentionally add a holdout drug, run scripts/run_engine_benchmark.py,
    diff against data/training/4track_holdout_predictions.json, and update this
    test only after confirming Meta AAFE is bit-identical (or document the diff
    in experiment-log.md).
    """
    holdout_data = _load(_HOLDOUT_PATH)
    # Handle both legacy list format and dict format with "holdout" key
    if isinstance(holdout_data, dict):
        holdout_list = holdout_data.get("holdout", [])
    elif isinstance(holdout_data, list):
        holdout_list = holdout_data
    else:
        holdout_list = []

    holdout_set = {d.lower() for d in holdout_list if isinstance(d, str)}

    for path in (_NAT2_PATH, _UGT1A1_PATH):
        data = _load(path)
        for s in data["substrates"]:
            assert s["drug"].lower() not in holdout_set, (
                f"{s['drug']} appears in holdout. Either remove from "
                f"{path.name} OR run holdout regen + invariance check + "
                f"update this test gate."
            )
