#!/usr/bin/env python3
"""Build the P0 candidate manifest from FRDB metadata without reading Cmax values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from audit_external_holdout_manifest import EXCLUSION, _norm_name, _repository_exclusions
from rdkit import Chem

SOURCE_SHA = "647b80d9cdac4a0517ce649570517dc1aff40545bdc84617ec9f1a56405f9c7a"
MODEL_SHA = "618106b53b0c9ce3c5b8a8fe62c5adf8f02b2308"
DOSE_FACTORS = {"mg": 1.0, "μg": 0.001, "g": 1000.0}
CMAX_UNITS = {"ng/mL", "μg/mL", "μg/L", "mg/L", "pg/mL", "ng/L", "mg/mL"}
FIXED = {
    "pk_age_group": "ADULT",
    "pk_health_status": "HEALTHY",
    "pk_food_status": "FASTED",
    "pk_routes": "Oral",
    "pk_experiment_type": "SINGLE",
    "pk_analyte_tissue": "PLASMA",
    "pkappcombo": "false",
}
SAFE_FIELDS = (
    "id",
    "compound_id",
    "pk_analyte_smiles",
    "pk_dose_value",
    "pk_dose_units",
    "pk_cmax_units",
    "pk_source_type",
    "pk_source_uri",
    *FIXED,
)


def build(source: Path, output: Path) -> dict:
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("FRDB snapshot hash mismatch")
    structures, names = _repository_exclusions(output)
    by_id: dict[str, list[dict]] = defaultdict(list)
    with ZipFile(source) as archive:
        with archive.open("frdb/frdb-drugs.tsv") as handle:
            drugs = {
                row["compound_id"]: row["compound_name"]
                for row in csv.DictReader(
                    io.TextIOWrapper(handle, encoding="utf-8"), delimiter="\t"
                )
            }
        with archive.open("frdb/frdb-pk.tsv") as handle:
            for raw in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"), delimiter="\t"):
                row = {key: raw[key] for key in SAFE_FIELDS}  # no Cmax value enters the manifest
                if any(row[key] != value for key, value in FIXED.items()):
                    continue
                if (
                    row["pk_dose_units"] not in DOSE_FACTORS
                    or row["pk_cmax_units"] not in CMAX_UNITS
                ):
                    continue
                try:
                    dose = float(row["pk_dose_value"]) * DOSE_FACTORS[row["pk_dose_units"]]
                except ValueError:
                    continue
                if not math.isfinite(dose) or dose <= 0:
                    continue
                smiles = row["pk_analyte_smiles"]
                key = EXCLUSION.ik14(smiles)
                name = drugs.get(row["compound_id"], "")
                if not key or not name or key in structures or _norm_name(name) in names:
                    continue
                mol = EXCLUSION._largest_organic_fragment(Chem.MolFromSmiles(smiles))
                row["smiles"] = Chem.MolToSmiles(mol, canonical=True)
                row["ik14"] = key
                row["dose_mg"] = dose
                by_id[row["compound_id"]].append(row)

    candidates = []
    seen_keys = set()
    order = sorted(
        by_id,
        key=lambda cid: (hashlib.sha256(f"sisyphus-p0-2026-09-23:{cid}".encode()).hexdigest(), cid),
    )
    exclusions = {"multiple_analyte_keys": 0, "duplicate_connectivity": 0}
    for cid in order:
        rows = sorted(by_id[cid], key=lambda row: int(row["id"]))
        keys = {row["ik14"] for row in rows}
        if len(keys) != 1:
            exclusions["multiple_analyte_keys"] += 1
            continue
        key = next(iter(keys))
        if key in seen_keys:
            exclusions["duplicate_connectivity"] += 1
            continue
        seen_keys.add(key)
        candidates.append(
            {
                "candidate_id": cid,
                "name": drugs[cid],
                "smiles": rows[0]["smiles"],
                "ik14": key,
                "arms": [
                    {
                        "arm_id": f"frdb_{row['id']}",
                        "dose_mg": row["dose_mg"],
                        "cmax_unit": row["pk_cmax_units"],
                        "source_type": row["pk_source_type"],
                        "source_uri": row["pk_source_uri"],
                    }
                    for row in rows
                ],
            }
        )
    manifest = {
        "protocol": "docs/validation/self_run_pilot_p0.md",
        "model_git_sha": MODEL_SHA,
        "source_sha256": SOURCE_SHA,
        "candidates": candidates,
        "metadata_exclusions": exclusions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "candidates": len(candidates),
        "arms": sum(len(c["arms"]) for c in candidates),
        **exclusions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), sort_keys=True))
