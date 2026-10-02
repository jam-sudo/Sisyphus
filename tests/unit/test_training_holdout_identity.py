"""Salt forms must not bypass holdout exclusion in training entry points."""

from __future__ import annotations

import csv
import json

from rdkit import Chem

from sisyphus.validation.identity import ik14

_CLOPIDOGREL_SALT = "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1.O=S(=O)(O)O"


def test_pka_checks_normalized_structure_despite_precomputed_salt_key():
    from scripts.train_pka_model import is_holdout

    raw_key = Chem.MolToInchiKey(Chem.MolFromSmiles(_CLOPIDOGREL_SALT))[:14]
    parent_key = ik14(_CLOPIDOGREL_SALT)
    assert raw_key != parent_key
    keys = {"canonical_smiles": set(), "inchikey_prefixes": {parent_key}, "names": set()}
    assert is_holdout(_CLOPIDOGREL_SALT, "clopidogrel bisulfate", raw_key, keys)


def test_sbi_pool_excludes_salt_variant_even_with_false_flag(tmp_path, monkeypatch):
    from scripts import sbi_select_train_drug_set as select

    pool = tmp_path / "pool.csv"
    with pool.open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["name", "canon_smiles", "dose_mg", "in_holdout"]
        )
        writer.writeheader()
        writer.writerow({
            "name": "clopidogrel bisulfate", "canon_smiles": _CLOPIDOGREL_SALT,
            "dose_mg": "100", "in_holdout": "False",
        })
    monkeypatch.setattr(select, "POOL_CSV", pool)
    assert select.load_pool() == []


def test_deconvolution_excludes_salt_variant_even_with_false_flag(tmp_path, monkeypatch):
    from scripts import run_deconvolution as deconvolution

    (tmp_path / "mmpk_sisyphus_holdout_exclusions.json").write_text(json.dumps({}))
    with (tmp_path / "mmpk_expanded_full.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "name", "canon_smiles", "dose_mg", "cmax_mg_L", "in_holdout",
        ])
        writer.writeheader()
        writer.writerow({
            "name": "clopidogrel bisulfate", "canon_smiles": _CLOPIDOGREL_SALT,
            "dose_mg": "100", "cmax_mg_L": "0.3", "in_holdout": "False",
        })
    monkeypatch.setattr(deconvolution, "_TRAINING_DIR", tmp_path)
    assert deconvolution.load_mmpk_data() == []


def test_ml_cmax_excludes_salt_variant_even_with_false_flag(tmp_path, monkeypatch):
    from scripts import ml_cmax_improvement as ml

    reference = tmp_path / "data/reference"
    reference.mkdir(parents=True)
    (reference / "holdout.json").write_text(json.dumps({"holdout": ["clopidogrel"]}))
    (reference / "clinical_pk.json").write_text(json.dumps({
        "drugs": {"clopidogrel": {"smiles": _CLOPIDOGREL_SALT.split(".")[0]}},
    }))
    training = tmp_path / "data/training"
    training.mkdir(parents=True)
    with (training / "mmpk_expanded_full.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "name", "canon_smiles", "dose_mg", "cmax_mg_L",
            "log_cmax_per_dose", "in_holdout",
        ])
        writer.writeheader()
        writer.writerow({
            "name": "clopidogrel bisulfate", "canon_smiles": _CLOPIDOGREL_SALT,
            "dose_mg": "100", "cmax_mg_L": "0.3",
            "log_cmax_per_dose": "-2.52", "in_holdout": "False",
        })
    monkeypatch.setattr(ml, "ROOT", tmp_path)
    assert ml.load_mmpk_data().empty


def test_clint_docking_excludes_salt_variant(tmp_path, monkeypatch):
    from scripts import train_clint_docking as docking

    source = tmp_path / "clearance.tab"
    source.write_text(
        "Drug_ID\tDrug\tY\n"
        f"salt_variant\t{_CLOPIDOGREL_SALT}\t1.0\n"
    )
    monkeypatch.setattr(docking, "HEP_AZ_PATH", source)
    assert docking.load_tdc_data().empty
