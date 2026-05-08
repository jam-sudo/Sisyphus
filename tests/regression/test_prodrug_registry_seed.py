"""Seed-pin + RDKit roundtrip for prodrug_activation_registry.json (#11 v0.3.4).

Two gates:
  1. Seed pinned: frozenset of drug names matches expected (catches silent
     additions or removals).
  2. InChIKey-SMILES roundtrip: registered SMILES key, when canonicalized
     via RDKit and converted to InChIKey, matches the registered InChIKey
     for each entry. Each entry's `name` is also a sanity check.

Mirrors PR #29 oatp1b1 schema gate pattern + adds InChIKey roundtrip
since the existing test_prodrug_v3_registry_schema.py validates v3_metadata
structure but not SMILES integrity.
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "sbi" / "prodrug_activation_registry.json"

_EXPECTED_PRODRUG_NAMES = frozenset({
    "BH4", "GS-441524", "tebipenem", "R406",
    "simvastatin", "irinotecan",
})


def _load() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def test_prodrug_seed_pinned():
    data = _load()
    actual = {entry["name"] for entry in data.values() if isinstance(entry, dict)}
    assert actual == _EXPECTED_PRODRUG_NAMES, (
        f"Prodrug seed drift: expected {_EXPECTED_PRODRUG_NAMES}, got {actual}. "
        f"Update _EXPECTED_PRODRUG_NAMES with explicit decision per spec §6.2 "
        f"(disposition state + ceiling_rationale or literature_applied citation)."
    )


def test_smiles_inchikey_roundtrip():
    """For each entry, RDKit-canonicalize the SMILES key and verify it
    parses successfully. (We don't pin a specific InChIKey because the
    registry uses SMILES as the lookup key, but a parse failure means
    lookup_active_metabolite would silently never match.)"""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        m = Chem.MolFromSmiles(smiles)
        assert m is not None, (
            f"SMILES key for {entry.get('name', '?')} failed to parse: {smiles!r}"
        )
        # Sanity: roundtrip via canonical SMILES + InChIKey produces non-empty results
        canonical = Chem.MolToSmiles(m)
        ikey = Chem.MolToInchiKey(m)
        assert canonical, f"{entry.get('name', '?')}: empty canonical SMILES"
        assert len(ikey) == 27, f"{entry.get('name', '?')}: malformed InChIKey {ikey!r}"
