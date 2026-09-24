"""Scored development labels must be observed parent-drug arms."""

import json
from pathlib import Path

import pytest

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]


def test_adjudicated_holdout_arms_match_scored_cache():
    data = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    refs = {row.name: row for row in load_reference() if row.in_holdout}
    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    assert cache["n_holdout"] == data["metadata"]["holdout_with_cmax"] == len(refs) == 85
    assert {row["name"] for row in cache["drugs"]} == set(refs)
    for row in cache["drugs"]:
        assert row["obs"] == refs[row["name"]].cmax_obs
        assert not any(word in data["drugs"][row["name"]]["source"].lower()
                       for word in ("estimated", "simulated"))

    for name in (
        "abiraterone", "atovaquone", "clozapine", "darolutamide",
        "darunavir ethanolate", "glasdegib", "itraconazole", "leflunomide",
        "pomalidomide", "ranolazine", "sirolimus", "sonidegib", "tamsulosin",
        "vilazodone",
    ):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert name not in refs
    for name in (
        "adefovir dipivoxil", "fesoterodine", "molnupiravir", "prasugrel",
        "tenofovir disoproxil", "valacyclovir", "valganciclovir",
    ):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert name not in refs
    assert refs["clopidogrel"].dose_mg == 300.0
    assert refs["clopidogrel"].cmax_obs == 0.0145
    assert refs["levocetirizine"].dose_mg == 5.0
    assert refs["levocetirizine"].cmax_obs == 0.27
    assert refs["methylphenidate"].dose_mg == 20.0
    assert refs["methylphenidate"].cmax_obs == 0.0091
    assert refs["paroxetine"].dose_mg == 25.0
    assert refs["paroxetine"].cmax_obs == 0.0055
    assert refs["nilotinib"].dose_mg == 200.0
    assert refs["nilotinib"].cmax_obs == 0.615
    assert refs["norethindrone"].dose_mg == 0.35
    assert refs["norethindrone"].cmax_obs == 0.004817
    assert refs["carbamazepine"].dose_mg == 200.0
    assert refs["carbamazepine"].cmax_obs == 1.9
    assert refs["zonisamide"].dose_mg == 300.0
    assert refs["zonisamide"].cmax_obs == 3.42
    assert refs["oxybutynin"].dose_mg == 5.0
    assert refs["oxybutynin"].cmax_obs == 0.0082
    assert refs["oxybutynin"].auc_obs is None
    assert refs["dasatinib"].dose_mg == 100.0
    assert refs["dasatinib"].cmax_obs == 0.2246
    assert refs["dasatinib"].auc_obs is None
    assert refs["cetirizine"].dose_mg == pytest.approx(10 * 388.89 / 461.82, rel=1e-6)
    assert refs["cetirizine"].cmax_obs == pytest.approx(0.266)
    assert refs["cetirizine"].auc_obs == pytest.approx(2.526)
    assert refs["febuxostat"].dose_mg == 40.0
    assert refs["febuxostat"].cmax_obs == pytest.approx(1.82)
    assert refs["febuxostat"].auc_obs == pytest.approx(4.61)
    assert "ct_curve" not in data["drugs"]["febuxostat"]
    assert refs["clomipramine"].dose_mg == pytest.approx(50 * 314.852 / 351.31, rel=1e-6)
    for name, dose, cmax in (
        ("alosetron", 1, 0.005),
        ("azacitidine", 300, 0.145),
        ("tamoxifen", 20, 0.04),
    ):
        assert refs[name].dose_mg == dose
        assert refs[name].cmax_obs == cmax
    for name in ("alosetron", "azacitidine", "clomipramine", "tamoxifen"):
        assert "ct_curve" not in data["drugs"][name]
        assert "thalf_h" not in data["drugs"][name]["pk_params"]
    assert data["drugs"]["clonidine"]["tier"] == "unverified"
    assert data["drugs"]["clonidine"]["dose_mg"] == 0.087
    assert not data["drugs"]["clonidine"]["pk_params"]
    assert "clonidine" not in refs
    assert refs["pindolol"].cmax_obs == 0.0331
    assert "thalf_h" not in data["drugs"]["pindolol"]["pk_params"]
    assert refs["sumatriptan"].dose_mg == 25.0
    assert refs["sumatriptan"].cmax_obs == 0.018
    assert refs["bexagliflozin"].dose_mg == 20.0
    assert refs["bexagliflozin"].cmax_obs == 0.134
    for name in ("clonidine", "pindolol", "sumatriptan", "bexagliflozin"):
        assert "ct_curve" not in data["drugs"][name]
    assert data["drugs"]["indomethacin"]["tier"] == "silver"
    assert refs["indomethacin"].cmax_obs == 1.54
    assert refs["ketoconazole"].cmax_obs == 3.5
    assert refs["levofloxacin"].dose_mg == 500.0
    assert refs["levofloxacin"].cmax_obs == 5.1
    assert refs["metronidazole"].cmax_obs == 13.0
    for name in ("indomethacin", "ketoconazole", "levofloxacin", "metronidazole"):
        assert "ct_curve" not in data["drugs"][name]
