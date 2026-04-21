"""Schema and value-envelope tests for the OATP generalization observation file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "validation" / "oatp_generalization_drugs.json"

_EXPECTED_DRUGS = {"bosentan", "valsartan", "repaglinide"}

_CMAX_ENVELOPES = {
    "bosentan": (8.0, 20.0),   # plan had (0.5, 3.0) but 250mg IV dose/Vss=250/18=13.9mg/L; corrected
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


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_per_drug_required_fields(drug: str):
    data = _load()
    entry = data["drugs"][drug]
    for field in ("dose_mg", "observed_cmax_mg_l", "administration", "smiles", "source"):
        assert field in entry, f"{drug} missing field {field}"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_dose_within_envelope(drug: str):
    data = _load()
    dose = float(data["drugs"][drug]["dose_mg"])
    lo, hi = _DOSE_ENVELOPES[drug]
    assert lo <= dose <= hi, f"{drug} dose {dose} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_cmax_within_envelope(drug: str):
    data = _load()
    cmax = float(data["drugs"][drug]["observed_cmax_mg_l"])
    lo, hi = _CMAX_ENVELOPES[drug]
    assert lo <= cmax <= hi, f"{drug} cmax {cmax} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_administration_is_iv(drug: str):
    data = _load()
    admin = data["drugs"][drug]["administration"]
    assert admin.startswith("iv_"), f"{drug} admin {admin!r} must be iv_bolus or iv_infusion_Xmin"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_source_has_doi(drug: str):
    data = _load()
    src = data["drugs"][drug]["source"]
    assert "doi" in src.lower() or "10." in src, f"{drug} source missing DOI: {src!r}"
