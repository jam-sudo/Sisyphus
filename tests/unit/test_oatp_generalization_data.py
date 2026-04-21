"""Schema and value-envelope tests for the OATP generalization observation file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "validation" / "oatp_generalization_drugs.json"

_EXPECTED_DRUGS = {"bosentan", "valsartan", "repaglinide"}
_VERIFIED_DRUGS = {"valsartan"}
_BLOCKED_DRUGS = {"bosentan", "repaglinide"}

_CMAX_ENVELOPES = {
    "bosentan": (0.5, 3.0),   # plan original per spec 9115e63; ee24164 correction (8-20) was invalid
    "valsartan": (0.3, 6.0),
    "repaglinide": (0.020, 0.080),
}

_DOSE_ENVELOPES = {
    "bosentan": (100.0, 250.0),
    "valsartan": (20.0, 160.0),
    "repaglinide": (2.0, 2.0),
}


def _load() -> dict:
    assert _DATA_FILE.exists(), f"{_DATA_FILE} does not exist"
    with _DATA_FILE.open() as f:
        return json.load(f)


def test_top_level_schema():
    data = _load()
    assert "drugs" in data
    assert isinstance(data["drugs"], dict)
    assert set(data["drugs"].keys()) == _EXPECTED_DRUGS


def test_top_level_status_fields():
    data = _load()
    assert "n_verified" in data, "top-level n_verified missing"
    assert "n_blocked" in data, "top-level n_blocked missing"
    assert "outcome_status" in data, "top-level outcome_status missing"


def test_n_verified_and_n_blocked():
    data = _load()
    assert data["n_verified"] == 1, f"expected n_verified=1, got {data['n_verified']}"
    assert data["n_blocked"] == 2, f"expected n_blocked=2, got {data['n_blocked']}"


def test_each_drug_has_status():
    data = _load()
    for drug, entry in data["drugs"].items():
        assert "status" in entry, f"{drug} missing 'status' field"
        assert entry["status"] in ("VERIFIED", "BLOCKED"), (
            f"{drug} status must be VERIFIED or BLOCKED, got {entry['status']!r}"
        )


def test_blocked_drugs_have_reason():
    data = _load()
    for drug in _BLOCKED_DRUGS:
        entry = data["drugs"][drug]
        assert entry.get("status") == "BLOCKED", f"{drug} expected BLOCKED status"
        assert entry.get("blocked_reason"), f"{drug} missing non-empty blocked_reason"


@pytest.mark.parametrize("drug", sorted(_VERIFIED_DRUGS))
def test_per_drug_required_fields(drug: str):
    data = _load()
    entry = data["drugs"][drug]
    for field in ("dose_mg", "observed_cmax_mg_l", "administration", "smiles", "source"):
        assert field in entry, f"{drug} missing field {field}"


@pytest.mark.parametrize("drug", sorted(_VERIFIED_DRUGS))
def test_dose_within_envelope(drug: str):
    data = _load()
    dose = float(data["drugs"][drug]["dose_mg"])
    lo, hi = _DOSE_ENVELOPES[drug]
    assert lo <= dose <= hi, f"{drug} dose {dose} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_VERIFIED_DRUGS))
def test_cmax_within_envelope(drug: str):
    data = _load()
    cmax = float(data["drugs"][drug]["observed_cmax_mg_l"])
    lo, hi = _CMAX_ENVELOPES[drug]
    assert lo <= cmax <= hi, f"{drug} cmax {cmax} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_VERIFIED_DRUGS))
def test_administration_is_iv(drug: str):
    data = _load()
    admin = data["drugs"][drug]["administration"]
    assert admin.startswith("iv_"), f"{drug} admin {admin!r} must be iv_bolus or iv_infusion_Xmin"


@pytest.mark.parametrize("drug", sorted(_VERIFIED_DRUGS))
def test_source_has_doi(drug: str):
    data = _load()
    src = data["drugs"][drug]["source"]
    assert "doi" in src.lower() or "10." in src, f"{drug} source missing DOI: {src!r}"


def test_valsartan_individual_values_consistency():
    """Individual subject values must reproduce reported mean, SD, and N.

    Catches transcription errors in observed_cmax_mg_l — the value the engine run
    compares against.
    """
    import statistics

    data = _load()
    entry = data["drugs"]["valsartan"]
    individuals = entry["individual_cmax_mg_l"]

    assert len(individuals) == entry["patient_n"]
    assert abs(statistics.mean(individuals) - entry["observed_cmax_mg_l"]) < 0.01
    assert abs(statistics.stdev(individuals) - entry["observed_cmax_sd_mg_l"]) < 0.02
