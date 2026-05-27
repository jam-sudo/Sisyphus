"""Schema gates for ugt2b7_substrates.json + ugt1a9_substrates.json.

Pattern: parallel to tests/regression/test_oatp_registry_schema.py.

Four gates:
  1. Schema completeness: every entry has drug, smiles, inchikey,
     metabolic_fraction, literature (non-empty), notes.
  2. InChIKey matches RDKit canonicalization of registered SMILES.
  3. metabolic_fraction in (0, 1].
  4. No InChIKey appears in two or more of
     {nat2, ugt1a1, ugt2b7, ugt1a9} simultaneously.
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_REGISTRY_PATHS = {
    "nat2":   _REPO_ROOT / "data" / "enzymes" / "nat2_substrates.json",
    "ugt1a1": _REPO_ROOT / "data" / "enzymes" / "ugt1a1_substrates.json",
    "ugt2b7": _REPO_ROOT / "data" / "enzymes" / "ugt2b7_substrates.json",
    "ugt1a9": _REPO_ROOT / "data" / "enzymes" / "ugt1a9_substrates.json",
}

_REQUIRED_FIELDS = {"drug", "smiles", "inchikey", "metabolic_fraction", "literature", "notes"}


def _load(name: str) -> dict:
    return json.loads(_REGISTRY_PATHS[name].read_text())


def test_ugt2b7_registry_exists():
    assert _REGISTRY_PATHS["ugt2b7"].exists(), "ugt2b7_substrates.json missing"


def test_ugt1a9_registry_exists():
    assert _REGISTRY_PATHS["ugt1a9"].exists(), "ugt1a9_substrates.json missing"


def test_schema_completeness():
    for name in ("ugt2b7", "ugt1a9"):
        data = _load(name)
        assert "substrates" in data, f"{name}: missing 'substrates' array"
        assert len(data["substrates"]) >= 1, f"{name}: at least 1 substrate required"
        for entry in data["substrates"]:
            missing = _REQUIRED_FIELDS - set(entry.keys())
            assert not missing, f"{name}/{entry.get('drug', '<unknown>')}: missing fields {missing}"
            assert entry["literature"], f"{name}/{entry['drug']}: empty literature list"


def test_inchikey_matches_smiles():
    for name in ("ugt2b7", "ugt1a9"):
        for entry in _load(name)["substrates"]:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"{name}/{entry['drug']}: SMILES parse failed"
            derived = Chem.MolToInchiKey(mol)
            assert derived == entry["inchikey"], (
                f"{name}/{entry['drug']}: registered InChIKey {entry['inchikey']!r} "
                f"does not match RDKit-derived {derived!r}"
            )


def test_metabolic_fraction_range():
    for name in ("ugt2b7", "ugt1a9"):
        for entry in _load(name)["substrates"]:
            fm = entry["metabolic_fraction"]
            assert 0.0 < fm <= 1.0, f"{name}/{entry['drug']}: fm={fm} not in (0, 1]"


def test_no_cross_registry_duplicates():
    """No InChIKey appears in two or more of NAT2/UGT1A1/UGT2B7/UGT1A9."""
    seen: dict[str, str] = {}
    for name in ("nat2", "ugt1a1", "ugt2b7", "ugt1a9"):
        for entry in _load(name)["substrates"]:
            ikey = entry["inchikey"]
            if ikey in seen:
                raise AssertionError(
                    f"Duplicate InChIKey {ikey} ({entry['drug']}) appears in "
                    f"both {seen[ikey]!r} and {name!r} registries. "
                    f"Approach 1 single-registry-per-drug invariant violated."
                )
            seen[ikey] = name
