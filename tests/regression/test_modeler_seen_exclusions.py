"""Previously exposed identities must never enter the blinded final test."""

import json
from pathlib import Path

from scripts.audit_external_holdout_manifest import EXCLUSION, _norm_name, _repository_exclusions

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data/validation/frdb_modeler_seen_identities_2026-09-23.json"


def test_modeler_seen_names_and_structures_are_excluded():
    records = json.loads(LEDGER.read_text())["candidates"]
    assert len(records) == 28
    structures, names = _repository_exclusions(ROOT / "nonexistent-v1-manifest.json")
    source = str(LEDGER.relative_to(ROOT))
    for record in records:
        assert source in names.get(_norm_name(record["name"]), set())
        for structure in record["structures"]:
            assert source in structures.get(EXCLUSION.ik14(structure["smiles"]), set())


def test_p2_source_review_identities_are_excluded_by_name_alias_and_structure():
    ledger = ROOT / "data/validation/p2_modeler_seen_identities_2026-09-24.json"
    records = json.loads(ledger.read_text())["candidates"]
    assert len(records) == 2
    structures, names = _repository_exclusions(ROOT / "nonexistent-v1-manifest.json")
    source = str(ledger.relative_to(ROOT))
    for record in records:
        assert source in names.get(_norm_name(record["name"]), set())
        for alias in record["aliases"]:
            assert source in names.get(_norm_name(alias["name"]), set())
        assert EXCLUSION.ik14(record["smiles"]) == record["ik14"]
        assert source in structures.get(record["ik14"], set())
