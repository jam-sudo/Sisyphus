"""Schema and value-envelope tests for the OATP generalization observation file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "validation" / "oatp_generalization_drugs.json"

_EXPECTED_DRUGS = {"valsartan", "glimepiride"}
_VERIFIED_DRUGS = {"valsartan", "glimepiride"}
_BLOCKED_DRUGS: set[str] = set()

_CMAX_ENVELOPES = {
    "valsartan": (0.3, 6.0),
    "glimepiride": (0.20, 0.30),  # observed 0.243 mg/L (Badian 1994 via PMC11768776 Table 3)
}

_DOSE_ENVELOPES = {
    "valsartan": (20.0, 160.0),
    "glimepiride": (1.0, 1.0),
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
    assert data["n_verified"] == 2, f"expected n_verified=2, got {data['n_verified']}"
    assert data["n_blocked"] == 0, f"expected n_blocked=0, got {data['n_blocked']}"


def test_each_drug_has_status():
    data = _load()
    for drug, entry in data["drugs"].items():
        assert "status" in entry, f"{drug} missing 'status' field"
        assert entry["status"] in ("VERIFIED", "BLOCKED"), (
            f"{drug} status must be VERIFIED or BLOCKED, got {entry['status']!r}"
        )


@pytest.mark.skipif(not _BLOCKED_DRUGS, reason="no blocked drugs in v2 substrate set")
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
def test_source_has_doi_or_pmid(drug: str):
    data = _load()
    src = data["drugs"][drug]["source"]
    has_doi = "doi" in src.lower() or "10." in src
    has_pmid = "pmid" in src.lower() or "pmc" in src.lower()
    assert has_doi or has_pmid, f"{drug} source missing DOI or PMID: {src!r}"


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


def test_glimepiride_smiles_formula():
    """Glimepiride SMILES must parse to C24H34N4O5S (PubChem CID 3476)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        pytest.skip("rdkit not installed")

    data = _load()
    smiles = data["drugs"]["glimepiride"]["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"glimepiride SMILES failed to parse: {smiles!r}"
    formula = rdMolDescriptors.CalcMolFormula(mol)
    assert formula == "C24H34N4O5S", f"glimepiride formula {formula!r} != expected C24H34N4O5S"


def test_dropped_drugs_archived():
    """Dropped drugs from amendment v2 must be listed in dropped_under_amendment_v2."""
    data = _load()
    assert "dropped_under_amendment_v2" in data, "top-level dropped_under_amendment_v2 key missing"
    dropped = data["dropped_under_amendment_v2"]
    assert isinstance(dropped, dict), "dropped_under_amendment_v2 must be a dict with drugs and reason"
    assert "drugs" in dropped, "dropped_under_amendment_v2 must have a 'drugs' list"
    assert "bosentan" in dropped["drugs"], "bosentan must be in dropped_under_amendment_v2"
    assert "repaglinide" in dropped["drugs"], "repaglinide must be in dropped_under_amendment_v2"
    assert "reason" in dropped, "dropped_under_amendment_v2 must have a 'reason' field"
