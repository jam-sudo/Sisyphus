"""Schema regression for per-enzyme yield rule (B-04).

Asserts:
  1. Every entry in the production registry satisfies the all-or-nothing
     per-enzyme yield rule (spec §5.4): if any enzyme declares 'yield',
     all enzymes must.
  2. Every per-enzyme 'yield' field has a mean in [0, 1].

Mirrors the test_oatp_registry_schema.py pattern (paired-registry gate).
"""
from __future__ import annotations

import json
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "sbi" / "prodrug_activation_registry.json"


def _load() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def test_all_or_nothing_per_enzyme_yield():
    """For each registry entry, either zero or all enzymes declare 'yield'."""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        affinities = entry.get("enzyme_affinity_for_conversion", {})
        if not affinities:
            continue
        n_with_yield = sum(
            1 for tag, dist in affinities.items()
            if isinstance(dist, dict) and "yield" in dist
        )
        if n_with_yield == 0:
            # All fall back to entry-level conversion_yield_fraction. Valid.
            continue
        assert n_with_yield == len(affinities), (
            f"prodrug entry {entry.get('name', smiles)!r}: mixed per-enzyme "
            f"yield declaration. {n_with_yield}/{len(affinities)} enzymes "
            f"declare 'yield'. Must be all or none (spec §5.4)."
        )


def test_per_enzyme_yield_in_unit_interval():
    """Each per-enzyme yield 'mean' is in [0, 1]."""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        for tag, dist in entry.get("enzyme_affinity_for_conversion", {}).items():
            if not isinstance(dist, dict):
                continue
            y = dist.get("yield")
            if y is None:
                continue
            assert isinstance(y, dict) and "mean" in y, (
                f"{entry.get('name', smiles)!r} enzyme {tag!r}: 'yield' "
                f"must be a dict with 'mean', got {y!r}"
            )
            assert 0.0 <= float(y["mean"]) <= 1.0, (
                f"{entry.get('name', smiles)!r} enzyme {tag!r}: yield mean "
                f"{y['mean']} out of [0, 1]"
            )
