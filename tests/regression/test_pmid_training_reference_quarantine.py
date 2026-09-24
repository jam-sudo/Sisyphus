"""Unsupported PMID-linked arms stay out; re-sourced metformin stays in."""

import json
from pathlib import Path

import pytest

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/clinical_pk.json"


def test_pmid_training_arms_are_quarantined():
    data = json.loads(REFERENCE.read_text())
    for name in ("theophylline", "verapamil"):
        row = data["drugs"][name]
        assert row["tier"] == "unverified"
        assert not row["pk_params"]
        assert "ct_curve" not in row
    refs = {row.name: row for row in load_reference(REFERENCE)}
    assert not {"theophylline", "verapamil"} & {
        row.name for row in refs.values()
    }
    metformin = data["drugs"]["metformin"]
    assert metformin["tier"] == "gold"
    assert metformin["route"] == "oral"
    assert "DailyMed" in metformin["source"]
    assert "ct_curve" not in metformin
    assert refs["metformin"].dose_mg == pytest.approx(500 * 129.167 / 165.63, rel=1e-5)
    assert refs["metformin"].cmax_obs == pytest.approx(1.03)
    assert data["metadata"]["n_with_cmax"] == sum(
        bool(row.get("pk_params", {}).get("cmax_mg_L"))
        for row in data["drugs"].values()
    )
