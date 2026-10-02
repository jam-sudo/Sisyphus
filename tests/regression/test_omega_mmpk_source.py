"""The recovered Omega snapshot supports the documented exclusion count."""

import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from scripts import check_prospective_eligibility as prospective
from sisyphus.validation.identity import ik14

ROOT = Path(__file__).resolve().parents[2]


def test_omega_cmax_source_matches_recorded_holdout_exclusion_count():
    rows = list(csv.DictReader((ROOT / "data/training/omega_mmpk_clean.csv").open()))
    holdout = json.loads((ROOT / "data/reference/holdout.json").read_text())["holdout"]
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    names = {name.casefold() for name in holdout}
    molecules = [clinical.get(name) or clinical.get(name.replace(" ", "_")) for name in holdout]
    smiles = {
        Chem.MolToSmiles(mol, isomericSmiles=True)
        for entry in molecules if entry and entry.get("smiles")
        if (mol := Chem.MolFromSmiles(entry["smiles"])) is not None
    }
    iks = {key for entry in molecules if entry and (key := ik14(entry.get("smiles")))}
    excluded = 0
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
        excluded += (
            row["name"].casefold() in names
            or canonical in smiles
            or ik14(row["smiles"]) in iks
        )
    assert (len(rows), excluded, len(rows) - excluded) == (1128, 100, 1028)


def test_prospective_gate_requires_omega_cmax_source(monkeypatch, tmp_path):
    monkeypatch.setattr(prospective, "TRAINING", tmp_path)
    (tmp_path / "clf_training.csv").write_text("name,smiles\n")
    (tmp_path / "mmpk_sisyphus_holdout_exclusions.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="omega_mmpk_clean.csv"):
        prospective.build_index()
