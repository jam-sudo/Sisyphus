"""Re-integrating archived source rows must not undo source-adjudicated exclusions."""

import json
import shutil

from scripts import integrate_holdout


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
    assert data["metadata"]["holdout_with_cmax"] == 73
    for name in (
        "acamprosate", "alvimopan", "cimetidine", "mefenamic acid", "atovaquone", "leflunomide",
        "lopinavir", "penicillamine", "pilocarpine", "prasugrel", "sirolimus", "venlafaxine",
    ):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]


def test_integration_does_not_restore_uncertain_cmax(tmp_path, monkeypatch):
    source = integrate_holdout.DATA_REF
    for name in ("holdout.json", "clinical_pk.json", "curated_pk_data.json", "manual_pk_curation.json"):
        shutil.copyfile(source / name, tmp_path / name)
    clinical_path = tmp_path / "clinical_pk.json"
    clinical = json.loads(clinical_path.read_text())
    for name in ("ramelteon", "paroxetine"):
        clinical["drugs"][name]["pk_params"].pop("cmax_mg_L")
    clinical_path.write_text(json.dumps(clinical))
    monkeypatch.setattr(integrate_holdout, "DATA_REF", tmp_path)

    integrate_holdout.main()

    result = json.loads((tmp_path / "clinical_pk_v2.json").read_text())
    for name in ("ramelteon", "paroxetine"):
        assert "cmax_mg_L" not in result["drugs"][name]["pk_params"]
