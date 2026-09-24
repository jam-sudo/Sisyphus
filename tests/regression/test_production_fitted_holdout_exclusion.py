"""A corrected development structure must not remain in any active fitted snapshot."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from sisyphus.validation.holdout_contract import _PRODUCTION_FITTED_MODELS, sha256_file
from sisyphus.validation.identity import ik14
from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def test_active_fitted_rows_exclude_current_development_compounds() -> None:
    holdout = [row for row in load_reference() if row.in_holdout]
    assert len(holdout) == 107
    holdout_keys = {ik14(row.smiles) for row in holdout}
    holdout_names = {_name_key(row.name) for row in holdout}
    assert None not in holdout_keys

    for model in _PRODUCTION_FITTED_MODELS:
        metadata = json.loads((ROOT / model).read_text())
        fitted = ROOT / metadata["trained_on"]["dataset_path"]
        assert sha256_file(fitted) == metadata["trained_on"]["sha256"], model
        with fitted.open(newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t" if fitted.suffix == ".tab" else ",")
            smiles_column = (
                "canonical_smiles" if "canonical_smiles" in rows.fieldnames else "smiles"
            )
            assert smiles_column in rows.fieldnames, fitted
            for row in rows:
                key = ik14(row[smiles_column])
                assert key is not None, (fitted, row)
                assert key not in holdout_keys, (model, row)
                for name in (row.get("name"), row.get("drug_id")):
                    if name:
                        assert _name_key(name) not in holdout_names, (model, row)
