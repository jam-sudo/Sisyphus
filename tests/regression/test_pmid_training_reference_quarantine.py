"""Unsupported PMID-linked arms stay out; re-sourced metformin stays in."""

import json
from pathlib import Path

import pytest

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/clinical_pk.json"


def test_pmid_training_arms_are_quarantined():
    data = json.loads(REFERENCE.read_text())
    for name in (
        "acetaminophen", "amoxicillin", "diazepam", "metoprolol",
        "midazolam", "theophylline", "verapamil", "atenolol", "caffeine",
    ):
        row = data["drugs"][name]
        assert row["tier"] == "unverified"
        assert not row["pk_params"]
        assert "ct_curve" not in row
    refs = {row.name: row for row in load_reference(REFERENCE)}
    assert not {
        "acetaminophen", "amoxicillin", "diazepam", "metoprolol",
        "midazolam", "theophylline", "verapamil", "atenolol", "caffeine",
    } & {
        row.name for row in refs.values()
    }
    metformin = data["drugs"]["metformin"]
    assert metformin["tier"] == "gold"
    assert metformin["route"] == "oral"
    assert "DailyMed" in metformin["source"]
    assert "ct_curve" not in metformin
    assert refs["metformin"].dose_mg == pytest.approx(500 * 129.167 / 165.63, rel=1e-5)
    assert refs["metformin"].cmax_obs == pytest.approx(1.03)
    omeprazole = data["drugs"]["omeprazole"]
    assert omeprazole["tier"] == "gold"
    assert "Losec" in omeprazole["source"]
    assert omeprazole["dose_mg"] == 20.0
    assert omeprazole["pk_params"] == {"cmax_mg_L": 0.311, "auc_mg_h_L": 0.567}
    assert "ct_curve" not in omeprazole
    assert refs["omeprazole"].cmax_obs == pytest.approx(0.311)
    assert data["metadata"]["n_with_cmax"] == sum(
        bool(row.get("pk_params", {}).get("cmax_mg_L"))
        for row in data["drugs"].values()
    )


def test_residual_led_training_reference_audit():
    data = json.loads(REFERENCE.read_text())
    refs = {row.name: row for row in load_reference(REFERENCE)}
    for name in ("lanthanum carbonate", "cefpodoxime proxetil", "serdexmethylphenidate"):
        row = data["drugs"][name]
        assert row["tier"] == "unverified"
        assert row["pk_params"] == {}
        assert "ct_curve" not in row
        assert name not in refs
    for name, dose, cmax in (("primaquine", 30, 0.127), ("flutamide", 250, 0.0252)):
        row = data["drugs"][name]
        assert row["pk_params"] == {"cmax_mg_L": cmax}
        assert "ct_curve" not in row
        assert refs[name].dose_mg == pytest.approx(dose)
        assert refs[name].cmax_obs == pytest.approx(cmax)
