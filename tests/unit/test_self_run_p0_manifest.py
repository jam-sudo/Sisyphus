"""A Cmax label in the source must never reach the label-free pilot manifest."""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_self_run_p0_manifest as pilot  # noqa: E402


def test_manifest_excludes_outcome_values(tmp_path, monkeypatch):
    source = tmp_path / "frdb.zip"
    row = {
        **pilot.FIXED,
        "id": "1",
        "compound_id": "candidate_1",
        "pk_analyte_smiles": "CCO",
        "pk_dose_value": "100",
        "pk_dose_units": "mg",
        "pk_cmax_units": "ng/mL",
        "pk_source_type": "paper",
        "pk_source_uri": "https://example.org/source",
        "pk_cmax_value": "SECRET_LABEL_12345",
    }
    drugs = io.StringIO()
    writer = csv.DictWriter(drugs, fieldnames=["compound_id", "compound_name"], delimiter="\t")
    writer.writeheader()
    writer.writerow({"compound_id": "candidate_1", "compound_name": "Example"})
    pk = io.StringIO()
    writer = csv.DictWriter(pk, fieldnames=list(row), delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
    with ZipFile(source, "w") as archive:
        archive.writestr("frdb/frdb-drugs.tsv", drugs.getvalue())
        archive.writestr("frdb/frdb-pk.tsv", pk.getvalue())
    monkeypatch.setattr(pilot, "SOURCE_SHA", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(pilot, "_repository_exclusions", lambda _: ({}, {}))

    output = tmp_path / "manifest.json"
    assert pilot.build(source, output)["candidates"] == 1
    assert "SECRET_LABEL_12345" not in output.read_text()
    assert json.loads(output.read_text())["candidates"][0]["arms"][0]["dose_mg"] == 100
