"""Schema regression over the production hepatic_fu_correction registry (B-11).

Catches violations as Phase B adds curated entries. Phase A end state
is an empty overrides list -- these tests pass trivially until then.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "transporters" / "hepatic_fu_correction.json"
)

_LITERATURE_REQUIRED_DISPOSITIONS = frozenset({
    "literature_applied", "class_extrapolated",
})

_VALID_DISPOSITIONS = frozenset({
    "literature_applied",
    "class_extrapolated",
    "ceiling_accepted",
    "not_applicable",
})


def _load_overrides() -> list[dict]:
    if not _REGISTRY_PATH.exists():
        return []
    return json.loads(_REGISTRY_PATH.read_text()).get("overrides", [])


def test_literature_applied_requires_citation():
    """literature_applied and class_extrapolated entries must have a non-empty
    ``literature`` array (each item a citation string)."""
    violations = []
    for entry in _load_overrides():
        if entry.get("disposition") in _LITERATURE_REQUIRED_DISPOSITIONS:
            lit = entry.get("literature") or []
            if not lit:
                violations.append(entry.get("drug"))
    assert not violations, (
        f"literature_applied/class_extrapolated entries with no citation: "
        f"{violations}. Anti-fudge: every value above 1.0 must trace to a paper."
    )


def test_disposition_in_allowed_set():
    bad = []
    for entry in _load_overrides():
        if entry.get("disposition") not in _VALID_DISPOSITIONS:
            bad.append((entry.get("drug"), entry.get("disposition")))
    assert not bad, (
        f"hepatic_fu_correction entries with invalid disposition: {bad}. "
        f"Allowed: {sorted(_VALID_DISPOSITIONS)}"
    )
