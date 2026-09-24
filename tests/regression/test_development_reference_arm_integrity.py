"""Scored development labels must be observed parent-drug arms."""

import json
from pathlib import Path

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]


def test_adjudicated_holdout_arms_match_scored_cache():
    data = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    refs = {row.name: row for row in load_reference() if row.in_holdout}
    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    assert cache["n_holdout"] == data["metadata"]["holdout_with_cmax"] == len(refs) == 94
    assert {row["name"] for row in cache["drugs"]} == set(refs)
    for row in cache["drugs"]:
        assert row["obs"] == refs[row["name"]].cmax_obs
        assert not any(word in data["drugs"][row["name"]]["source"].lower()
                       for word in ("estimated", "simulated"))

    for name in (
        "abiraterone", "atovaquone", "darunavir ethanolate", "leflunomide",
        "sirolimus", "tamsulosin",
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
