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
        "amphetamine", "felodipine",
    ):
        row = data["drugs"][name]
        assert row["tier"] == "unverified"
        assert not row["pk_params"]
        assert "ct_curve" not in row
    refs = {row.name: row for row in load_reference(REFERENCE)}
    assert not {
        "acetaminophen", "amoxicillin", "diazepam", "metoprolol",
        "midazolam", "theophylline", "verapamil", "atenolol", "caffeine",
        "amphetamine", "felodipine",
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
        ("abacavir", 600, 4.26),
        ("pantoprazole", 40, 2.5),
        ("propranolol", 80, 0.0495),
        ("lofexidine", 0.36, 0.00082),
        ("indapamide", 5, 0.04779),
        ("fluoxetine", 20, 0.0132),
        ("ezetimibe", 10, 0.00348),
        ("ibuprofen", 400, 32.92),
        ("metoclopramide", 10, 0.028),
        ("furosemide", 40, 1.10971),
        ("terbinafine", 250, 1.0),
        ("nifedipine", 10, 0.0789),
        ("pitavastatin", 2, 0.10609),
    ):
        row = data["drugs"][name]
        assert row["pk_params"]["cmax_mg_L"] == pytest.approx(cmax)
        assert "ct_curve" not in row
        assert refs[name].dose_mg == pytest.approx(dose)
        assert refs[name].cmax_obs == pytest.approx(cmax)
    for name in ("pregabalin", "sertraline"):
        assert "@" in data["drugs"][name]["smiles"]
    assert data["drugs"]["abacavir"]["smiles"].count("[C@@H]") == 2
    assert data["drugs"]["cyclobenzaprine"]["tier"] == "silver"
    assert "thalf_h" not in data["drugs"]["entacapone"]["pk_params"]
    assert data["drugs"]["abacavir"]["pk_params"]["thalf_h"] == pytest.approx(1.54)
    assert data["drugs"]["pantoprazole"]["pk_params"]["thalf_h"] == pytest.approx(1)
    assert "bioavailability_pct" not in data["drugs"]["propranolol"]["pk_params"]
    assert data["drugs"]["glycopyrrolate"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.00181)
    assert data["drugs"]["lofexidine"]["pk_params"]["bioavailability_pct"] == pytest.approx(72)
    assert data["drugs"]["indapamide"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.91952)
    assert data["drugs"]["ezetimibe"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.06862)
    assert data["drugs"]["ibuprofen"]["pk_params"]["auc_mg_h_L"] == pytest.approx(117.38)
    assert data["drugs"]["metoclopramide"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.268)
    assert data["drugs"]["furosemide"]["pk_params"]["auc_mg_h_L"] == pytest.approx(2.609)
    assert "thalf_h" not in data["drugs"]["terbinafine"]["pk_params"]
    assert "auc_mg_h_L" not in data["drugs"]["terbinafine"]["pk_params"]
    assert "thalf_h" not in data["drugs"]["metoclopramide"]["pk_params"]
    assert "@" in data["drugs"]["ezetimibe"]["smiles"]
    assert "/C=C/" in data["drugs"]["terbinafine"]["smiles"]
    assert "auc_mg_h_L" not in data["drugs"]["fluconazole"]["pk_params"]
    assert "thalf_h" not in data["drugs"]["nifedipine"]["pk_params"]
    assert data["drugs"]["pitavastatin"]["pk_params"]["auc_mg_h_L"] == pytest.approx(0.32125)
    assert data["drugs"]["pitavastatin"]["pk_params"]["thalf_h"] == pytest.approx(9.52)


def test_spurious_sertraline_training_duplicate_removed():
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "sertraline"
                and row["dose_mg"] == "50.0"
                and row["cmax_mg_L"] == "0.165"
                for row in csv.DictReader(handle)
            )
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "abacavir"
                and row["dose_mg"] == "600.0"
                and row["cmax_mg_L"] == "3.67"
                for row in csv.DictReader(handle)
            )
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "ezetimibe"
                and row["dose_mg"] == "10.0"
                and row["cmax_mg_L"] == "0.0034"
                for row in csv.DictReader(handle)
            )
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "terbinafine"
                and row["dose_mg"] == "250.0"
                and row["cmax_mg_L"] == "1.0"
                for row in csv.DictReader(handle)
            )
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(
                row["name"] == "amphetamine"
                and row["dose_mg"] == "18.8"
                and row["cmax_mg_L"] == "0.0449"
                for row in csv.DictReader(handle)
            )
