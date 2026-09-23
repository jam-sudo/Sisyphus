"""Unit tests for scripts/build_n50_exclusion.py — the InChIKey-14 exclusion tool.

The 2026Q2 N50 curation used a name-based inventory and missed synonym/salt
variants, letting drugs already in training pass as "never-touched". These tests
pin the mechanism that catches them: exclusion keyed on the InChIKey-14
connectivity block, which is representation-independent.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/build_n50_exclusion.py"


@pytest.fixture(scope="module")
def excl_module():
    spec = importlib.util.spec_from_file_location("build_n50_exclusion", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ik14_is_representation_independent(excl_module):
    """Two valid SMILES writings of benzene share one InChIKey-14."""
    k1 = excl_module.ik14("c1ccccc1")
    k2 = excl_module.ik14("C1=CC=CC=C1")
    assert k1 and k1 == k2


def test_ik14_bad_smiles_returns_none(excl_module):
    assert excl_module.ik14("not_a_smiles") is None
    assert excl_module.ik14("") is None
    assert excl_module.ik14(None) is None


def test_ik14_equates_rifampin_and_rifampicin(excl_module):
    """The exact contamination the name-based inventory missed: N50 'rifampin'
    is the same molecule as MMPK-training 'rifampicin'. InChIKey-14 equates them
    where a name string cannot."""
    n50 = json.loads((ROOT / "data/reference/holdout_n50.json").read_text())
    rif_n50 = n50["drugs"]["rifampin"]["smiles"]

    smi_train = None
    with (ROOT / "data/training/mmpk_expanded_full.csv").open() as f:
        for row in csv.DictReader(f):
            if row.get("name", "").strip().lower() == "rifampicin":
                smi_train = row.get("canon_smiles")
                break
    assert smi_train, "rifampicin not found in MMPK training corpus"

    k_n50 = excl_module.ik14(rif_n50)
    k_train = excl_module.ik14(smi_train)
    assert k_n50 and k_n50 == k_train


def test_distinct_molecules_differ(excl_module):
    """Sanity: unrelated molecules do not collide on IK14."""
    assert excl_module.ik14("CCO") != excl_module.ik14("c1ccccc1")


def test_ik14_strips_hydrochloride_counterion(excl_module):
    """Free base and HCl salt must collide in the exclusion inventory."""
    free_base = "CN1CCC(C2=CC=CC=C2)CC1"
    hydrochloride = "CN1CCC(C2=CC=CC=C2)CC1.Cl"
    assert excl_module.ik14(free_base) == excl_module.ik14(hydrochloride)


def test_drugbank_indexes_salt_stripped_parent_even_with_precomputed_key(
    excl_module, tmp_path
):
    salt = "CC(=O)[O-].CC(=O)[O-].[Ca+2]"
    stored = excl_module.Chem.MolToInchiKey(excl_module.Chem.MolFromSmiles(salt))[:14]
    parent = excl_module.ik14("CC(=O)[O-]")
    assert stored != parent
    path = tmp_path / excl_module.DRUGBANK
    path.parent.mkdir(parents=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "canonical_smiles", "inchikey_14"])
        writer.writeheader()
        writer.writerow({
            "name": "calcium acetate", "canonical_smiles": salt, "inchikey_14": stored,
        })
    db = excl_module._ingest_drugbank(tmp_path)
    assert db[parent] == db[stored] == "calcium acetate"


def test_audit_fails_closed_for_missing_sources(excl_module, tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"drugs": {"test": {"smiles": "CCO"}}}))
    with pytest.raises(FileNotFoundError, match="N50 exclusion sources missing"):
        excl_module.audit(tmp_path, candidate)


def test_audit_rejects_unparseable_candidate(excl_module, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(excl_module, "_require_sources", lambda root: None)
    monkeypatch.setattr(excl_module, "_ingest_hard", lambda root: {})
    monkeypatch.setattr(excl_module, "_ingest_drugbank", lambda root: {})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"drugs": {"bad": {"smiles": "not_a_smiles"}}}))
    assert excl_module.audit(tmp_path, candidate) == 1
    assert "clean of every" not in capsys.readouterr().out
    candidate.write_text(json.dumps({"drugs": {}}))
    with pytest.raises(ValueError, match="non-empty drugs"):
        excl_module.audit(tmp_path, candidate)
