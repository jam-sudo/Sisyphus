"""Regression test: 107-holdout drugs do not match the prodrug registry.

The 4 evidence prodrugs (sepiapterin, remdesivir, tebipenem_pivoxil,
fostamatinib) are in the N50 secondary holdout cycle 2026Q2, NOT in the
107-holdout. Verifying this prevents accidental contamination of the
107-holdout AAFE headline (2.695) by prodrug routing changes.

Holdout structure: ``holdout.json`` stores drug names in ``data["holdout"]``
(a list of 107 strings). SMILES are resolved from ``clinical_pk.json``
``data["drugs"][name]["smiles"]``.

Slow part (full benchmark MC=1000 → AAFE within 1% of 2.695) is CI-only
and not run in this fast suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_holdout_drugs_have_no_prodrug_registry_match():
    """None of the 107 holdout drugs should match the prodrug registry."""
    from sisyphus.predict.registry import lookup_active_metabolite

    repo_root = _repo_root()

    # Load holdout drug names (107-item list of strings)
    holdout_path = repo_root / "data" / "reference" / "holdout.json"
    with holdout_path.open() as f:
        holdout_data = json.load(f)
    holdout_names = holdout_data["holdout"]
    assert len(holdout_names) == 107, (
        f"Expected 107 holdout drugs, got {len(holdout_names)}. "
        "holdout.json may have changed — verify the split is intact."
    )

    # Resolve SMILES from clinical_pk.json
    cpk_path = repo_root / "data" / "reference" / "clinical_pk.json"
    with cpk_path.open() as f:
        cpk = json.load(f)
    drugs_ref = cpk["drugs"]

    smiles_list: list[tuple[str, str]] = []
    missing_smiles: list[str] = []
    for name in holdout_names:
        entry = drugs_ref.get(name, {})
        smi = entry.get("smiles") or entry.get("SMILES") or entry.get("canonical_smiles")
        if smi:
            smiles_list.append((name, smi))
        else:
            missing_smiles.append(name)

    assert len(smiles_list) > 0, (
        "No SMILES resolved from holdout drugs via clinical_pk.json — check file structure."
    )
    if missing_smiles:
        pytest.fail(
            f"The following holdout drugs have no SMILES in clinical_pk.json: {missing_smiles}. "
            "The test cannot verify them — resolve the missing SMILES or update the reference."
        )

    # Check each holdout SMILES against the prodrug registry
    matches: list[str] = []
    for name, smi in smiles_list:
        result = lookup_active_metabolite(smi)
        if result is not None:
            matches.append(name)

    assert matches == [], (
        f"107-holdout drugs unexpectedly match the prodrug registry: {matches}. "
        "Prodrug routing changes would contaminate the 107-holdout AAFE headline (2.695). "
        "Move these drugs out of the holdout or remove them from the registry."
    )
