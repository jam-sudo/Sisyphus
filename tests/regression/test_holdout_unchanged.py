"""Regression test: 107-holdout drugs in prodrug registry obey scoring doctrine.

Prior rule (pre-B-03): no 107-holdout drug may appear in the prodrug
registry at all.

B-03 doctrine (2026-05-20): a 107-holdout drug may appear in the registry
ONLY IF it observes ``observation_species="parent"`` AND has
``metabolic_fraction=0`` in ``cyp_clearance_overrides.json``. The first
condition keeps the scored species apples-to-apples with the holdout
reference; the second prevents double-counting of parent hepatic
consumption between the explicit ProdrugActivationEdges and the default
XGBoost-derived enzyme_affinity path.

An ``observation_species="active"`` registry entry for a holdout drug
would inject a 5-20x species-mismatch fold error into the headline AAFE
and is still forbidden.

Holdout structure: ``holdout.json`` stores drug names in ``data["holdout"]``
(a list of 107 strings). SMILES are resolved from ``clinical_pk.json``
``data["drugs"][name]["smiles"]``.

Slow part (full benchmark MC=1000) is CI-only and not run in this
fast suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_holdout_smiles() -> list[tuple[str, str]]:
    """Return [(name, smiles)] for every 107-holdout drug; fail on gaps."""
    repo_root = _repo_root()

    holdout_path = repo_root / "data" / "reference" / "holdout.json"
    with holdout_path.open() as f:
        holdout_data = json.load(f)
    holdout_names = holdout_data["holdout"]
    assert len(holdout_names) == 107, (
        f"Expected 107 holdout drugs, got {len(holdout_names)}. "
        "holdout.json may have changed — verify the split is intact."
    )

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
    return smiles_list


def test_holdout_drugs_in_registry_observe_parent_only():
    """A holdout drug in the registry must use ``observation_species='parent'``.

    ``observation_species='active'`` would silently change the scored
    species (parent Cmax vs active-metabolite Cmax) and inject a
    species-mismatch fold error into the headline AAFE.
    """
    from sisyphus.predict.registry import lookup_active_metabolite

    violations: list[tuple[str, str]] = []
    for name, smi in _resolve_holdout_smiles():
        result = lookup_active_metabolite(smi)
        if result is None:
            continue
        _, observation_species, _, _ = result
        if observation_species != "parent":
            violations.append((name, observation_species))

    assert violations == [], (
        f"107-holdout drugs registered with observation_species != 'parent': "
        f"{violations}. Active-species observation injects a species mismatch into the "
        f"headline AAFE; either remove the registry entry or set observation_species='parent'."
    )


def test_holdout_drugs_in_registry_have_metabolic_fraction_zero():
    """A holdout drug in the registry must zero its default XGBoost CL path.

    Without ``metabolic_fraction=0`` in ``cyp_clearance_overrides.json``,
    the explicit ProdrugActivationEdges plus the default XGBoost-derived
    enzyme_affinity path would double-count parent hepatic consumption.
    """
    from sisyphus.predict.cyp_clearance_overrides import lookup_metabolic_fraction
    from sisyphus.predict.registry import lookup_active_metabolite

    violations: list[tuple[str, float]] = []
    for name, smi in _resolve_holdout_smiles():
        if lookup_active_metabolite(smi) is None:
            continue
        mf = lookup_metabolic_fraction(smi)
        if mf != 0.0:
            violations.append((name, mf))

    assert violations == [], (
        f"107-holdout drugs in the prodrug registry without metabolic_fraction=0: "
        f"{violations}. Without this override the default XGBoost CL path runs in "
        f"parallel with the explicit ProdrugActivationEdges, double-counting parent "
        f"hepatic consumption. Add an entry to data/transporters/cyp_clearance_overrides.json."
    )
