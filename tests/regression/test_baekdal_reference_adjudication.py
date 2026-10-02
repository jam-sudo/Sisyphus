"""Keep the Bækdal original-arm labels separate from unsupported PK-DB values."""

import json
from pathlib import Path

import pytest

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/clinical_pk.json"


def test_baekdal_digoxin_arm_and_warfarin_quarantine():
    data = json.loads(REFERENCE.read_text())
    digoxin = data["drugs"]["digoxin"]
    warfarin = data["drugs"]["warfarin"]
    assert digoxin["dose_mg"] == pytest.approx(0.5)
    assert digoxin["pk_params"]["cmax_mg_L"] == pytest.approx(3.11 / 1000)
    assert "supplementary table 2c" in digoxin["source"]
    assert "ct_curve" not in digoxin
    assert warfarin["tier"] == "unverified"
    assert "cmax_mg_L" not in warfarin["pk_params"]
    assert "ct_curve" not in warfarin

    refs = {row.name: row for row in load_reference(REFERENCE)}
    assert refs["digoxin"].in_holdout
    assert refs["digoxin"].cmax_obs == pytest.approx(0.00311)
    assert "warfarin" not in refs
    assert data["metadata"]["n_with_cmax"] == sum(
        bool(row.get("pk_params", {}).get("cmax_mg_L"))
        for row in data["drugs"].values()
    )
    assert data["metadata"]["holdout_with_cmax"] == sum(row.in_holdout for row in refs.values())

    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    row = next(row for row in cache["drugs"] if row["name"] == "digoxin")
    assert row["obs"] == refs["digoxin"].cmax_obs
