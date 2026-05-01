"""Structural validation of v3 prodrug activation registry per spec §6.2.

Verifies each registry entry has required v3 fields (citation,
doctrine_path, disposition_state, etc.) and that conditional fields
(ceiling_rationale, interpretation_decision) are present per disposition.

This is structural validation only — no value comparison (avoids tautology
with implementation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY_PATH = Path("data/sbi/prodrug_activation_registry.json")
ALLOWED_DISPOSITIONS = {"literature_applied", "interpretation_resolved", "ceiling_accepted"}
REQUIRED_FIELDS = {
    "citation",
    "doctrine_path",
    "disposition_state",
    "source_dbs_searched",
    "n_candidates_reviewed",
}


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def registry() -> dict:
    return _load_registry()


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.exists(), f"{REGISTRY_PATH} not found"


def test_each_entry_has_v3_metadata_block(registry: dict) -> None:
    """Each prodrug entry must have a `v3_metadata` dict with required fields."""
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict):
            continue  # skip non-entry top-level keys
        meta = entry.get("v3_metadata")
        assert meta is not None, f"{drug_name}: v3_metadata block missing"
        assert isinstance(meta, dict), f"{drug_name}: v3_metadata must be dict"
        missing = REQUIRED_FIELDS - set(meta.keys())
        assert not missing, f"{drug_name}: v3_metadata missing fields {missing}"


def test_disposition_state_valid(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        ds = entry["v3_metadata"]["disposition_state"]
        assert ds in ALLOWED_DISPOSITIONS, f"{drug_name}: invalid disposition_state {ds!r}"


def test_citation_required_for_non_ceiling(registry: dict) -> None:
    """citation must be non-empty string when disposition is literature_applied
    or interpretation_resolved; may be null only for ceiling_accepted."""
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        ds = meta["disposition_state"]
        citation = meta.get("citation")
        if ds in ("literature_applied", "interpretation_resolved"):
            assert isinstance(citation, str) and citation.strip(), (
                f"{drug_name}: disposition {ds} requires non-empty citation"
            )
        elif ds == "ceiling_accepted":
            # citation may be null OR non-empty (e.g., V/F found but F primary not)
            if citation is not None:
                assert isinstance(citation, str), f"{drug_name}: citation must be str or null"


def test_ceiling_rationale_required(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        if meta["disposition_state"] == "ceiling_accepted":
            rationale = meta.get("ceiling_rationale", "").strip()
            assert rationale, f"{drug_name}: ceiling_accepted requires non-empty ceiling_rationale"


def test_interpretation_decision_required(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        if meta["disposition_state"] == "interpretation_resolved":
            decision = meta.get("interpretation_decision", "").strip()
            assert decision, (
                f"{drug_name}: interpretation_resolved requires non-empty interpretation_decision"
            )


def test_n_candidates_reviewed_int(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        n = entry["v3_metadata"]["n_candidates_reviewed"]
        assert isinstance(n, int) and n >= 1, (
            f"{drug_name}: n_candidates_reviewed must be int ≥ 1, got {n!r}"
        )


def test_source_dbs_searched_is_list(registry: dict) -> None:
    allowed_dbs = {"PubMed", "GoogleScholar", "FDA", "EMA", "ChEMBL", "DrugBank", "bioRxiv"}
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        dbs = entry["v3_metadata"]["source_dbs_searched"]
        assert isinstance(dbs, list) and dbs, (
            f"{drug_name}: source_dbs_searched must be non-empty list"
        )
        unknown = set(dbs) - allowed_dbs
        assert not unknown, f"{drug_name}: unknown source dbs {unknown}; allowed = {allowed_dbs}"
