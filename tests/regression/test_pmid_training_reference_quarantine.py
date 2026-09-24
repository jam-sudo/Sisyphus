"""Unsupported PMID-linked arms stay out; re-sourced metformin stays in."""

import csv
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


def test_audited_training_reference_arms():
    data = json.loads(REFERENCE.read_text())
    refs = {row.name: row for row in load_reference(REFERENCE)}
    for name in (
        "lanthanum carbonate", "cefpodoxime proxetil", "serdexmethylphenidate",
        "carglumic acid", "belzutifan", "pazopanib",
        "benzhydrocodone", "dimethyl", "guanfacine er", "naproxen", "oseltamivir",
        "atazanavir", "butalbital", "cefixime", "efavirenz",
        "paricalcitol", "vorasidenib", "entecavir",
    ):
        row = data["drugs"][name]
        assert row["tier"] == "unverified"
        assert row["pk_params"] == {}
        assert "ct_curve" not in row
        assert name not in refs
    for name, dose, cmax in (
        ("primaquine", 30, 0.127), ("flutamide", 250, 0.0252),
        ("carisoprodol", 350, 1.8), ("atorvastatin", 40, 0.01705),
        ("naproxen oral", 500, 64.3),
        ("clarithromycin", 500, 1.77),
        ("hydroxychloroquine", 155, 0.0503),
        ("isotretinoin", 80, 0.301),
        ("tranexamic acid", 1300, 13.83),
        ("aspirin", 500, 4.4),
        ("metaxalone", 400, 0.983),
        ("pregabalin", 300, 7.42008),
        ("sertraline", 50, 0.01139),
        ("tramadol", 87.852, 0.308),
        ("cyclobenzaprine", 8.829433, 0.007),
        ("desloratadine", 5, 0.0020581),
        ("entacapone", 200, 1.2),
        ("fluconazole", 400, 6.72),
        ("gabapentin", 300, 3.22369),
        ("glycopyrrolate", 2, 0.000318),
    ):
        row = data["drugs"][name]
        assert row["pk_params"]["cmax_mg_L"] == pytest.approx(cmax)
        assert "ct_curve" not in row
        assert refs[name].dose_mg == pytest.approx(dose)
        assert refs[name].cmax_obs == pytest.approx(cmax)
    for name in ("pregabalin", "sertraline"):
        assert "@" in data["drugs"][name]["smiles"]
    assert data["drugs"]["glycopyrrolate"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.00181)
    assert "auc_mg_h_L" not in data["drugs"]["fluconazole"]["pk_params"]


def test_spurious_sertraline_training_duplicate_removed():
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "sertraline"
                and row["dose_mg"] == "50.0"
                and row["cmax_mg_L"] == "0.165"
                for row in csv.DictReader(handle)
            )
