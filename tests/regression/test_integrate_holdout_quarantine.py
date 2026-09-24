"""Re-integrating archived source rows must not undo source-adjudicated exclusions."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts import integrate_holdout


def test_legacy_fda_builder_cannot_restore_superseded_rows():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/reference/fda_extraction_results.json"
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, str(root / "scripts/build_final_fda_results.py")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0 or "Refusing to overwrite newer source-adjudicated" in result.stderr
    assert path.read_bytes() == before


def test_integration_preserves_quarantined_references(tmp_path, monkeypatch):
    source = integrate_holdout.DATA_REF
    for name in (
        "holdout.json", "clinical_pk.json", "osp_observed.json", "curated_pk_data.json",
        "manual_pk_curation.json", "fda_extraction_results.json",
    ):
        shutil.copyfile(source / name, tmp_path / name)
    monkeypatch.setattr(integrate_holdout, "DATA_REF", tmp_path)

    integrate_holdout.main()

    data = json.loads((tmp_path / "clinical_pk_v2.json").read_text())
    assert data["metadata"]["holdout_with_cmax"] == 75
    for name in ("acamprosate", "cimetidine", "mefenamic acid", "atovaquone", "leflunomide",
                 "lopinavir", "penicillamine", "pilocarpine", "prasugrel",
                 "sirolimus", "venlafaxine"):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
