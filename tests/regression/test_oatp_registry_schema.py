"""Schema gates for oatp1b1.json + cyp_clearance_overrides.json pairing.

Prevents the pitavastatin-class double-counting bug from recurring: every
drug flagged ecm_applicable=true MUST have a matching metabolic_fraction
entry in cyp_clearance_overrides.json, otherwise auto-activating ECM in
predict() triple-counts hepatic clearance (verified empirically:
pitavastatin Cmax direction-flips 2.22x over -> 2.12x under at unchanged
fold-error magnitude).

Three gates:
  1. seed list pinned: catches silent flag flips during review
  2. InChIKey matches RDKit canonicalization of registered SMILES: catches
     SMILES drift like the issue #25 connectivity error case
  3. metabolic_fraction paired: catches missing override entries that
     would triple-count hepatic clearance
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_OATP1B1_PATH = _REPO_ROOT / "data" / "transporters" / "oatp1b1.json"
_OVERRIDES_PATH = _REPO_ROOT / "data" / "transporters" / "cyp_clearance_overrides.json"


# Pin the v0.3 seed list. If a drug is added/removed here, this test
# must fail loud so the spec change requires an explicit decision.
# 2026-05-04 (v0.3.1): pitavastatin promoted with metabolic_fraction=0
# (Niemi 2009 PM/EM ~3x, OATP-rate-limited, parallel pravastatin justification).
_EXPECTED_ECM_APPLICABLE = frozenset({"pravastatin", "pitavastatin"})


def _load_oatp() -> dict:
    return json.loads(_OATP1B1_PATH.read_text())


def _load_overrides() -> dict:
    return json.loads(_OVERRIDES_PATH.read_text())


def test_seed_list_pinned():
    """Only the expected drugs are flagged ecm_applicable=true."""
    data = _load_oatp()
    actual_true = {
        name for name, entry in data["drugs"].items()
        if entry.get("ecm_applicable") is True
    }
    assert actual_true == _EXPECTED_ECM_APPLICABLE, (
        f"ecm_applicable=true set drift: expected {_EXPECTED_ECM_APPLICABLE}, "
        f"got {actual_true}. If intentional, update _EXPECTED_ECM_APPLICABLE "
        f"and ensure cyp_clearance_overrides.json has paired metabolic_fraction "
        f"entries for the new drugs."
    )


def test_inchikey_matches_smiles():
    """For each ecm_applicable=true entry, the registered InChIKey must
    match the RDKit canonicalization of the registered SMILES."""
    data = _load_oatp()
    for name, entry in data["drugs"].items():
        if entry.get("ecm_applicable") is not True:
            continue
        smiles = entry["smiles"]
        registered = entry["inchikey"]
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(mol)
        assert derived == registered, (
            f"{name}: registered InChIKey {registered} does not match "
            f"RDKit-derived {derived} from SMILES. Either correct the "
            f"SMILES or update the InChIKey field."
        )


def test_metabolic_fraction_paired():
    """Every ecm_applicable=true drug has a paired entry in
    cyp_clearance_overrides.json — otherwise auto-activating ECM
    triple-counts hepatic clearance (pitavastatin double-counting bug
    prevention)."""
    oatp = _load_oatp()
    overrides = _load_overrides()
    override_inchikeys = {entry["inchikey"] for entry in overrides["overrides"]}

    for name, entry in oatp["drugs"].items():
        if entry.get("ecm_applicable") is not True:
            continue
        ikey = entry["inchikey"]
        assert ikey in override_inchikeys, (
            f"{name} (InChIKey {ikey}) is flagged ecm_applicable=true "
            f"but has no metabolic_fraction entry in "
            f"cyp_clearance_overrides.json. This will triple-count "
            f"hepatic clearance when predict() auto-activates ECM. "
            f"Add a literature-justified metabolic_fraction entry "
            f"OR remove the ecm_applicable flag."
        )
        matched = next(e for e in overrides["overrides"] if e["inchikey"] == ikey)
        assert "metabolic_fraction" in matched, (
            f"{name} (InChIKey {ikey}) override entry exists but is missing "
            f"the metabolic_fraction field — runtime will default to 1.0, "
            f"triple-counting hepatic clearance."
        )
