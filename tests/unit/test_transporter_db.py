"""Unit tests for the OATP1B1 transporter database loader."""

from __future__ import annotations

import pytest

from sisyphus.core import TransporterKinetics
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics


def test_load_pravastatin():
    kinetics = load_oatp1b1_kinetics("pravastatin")
    assert kinetics is not None
    assert "OATP1B1" in kinetics
    tk = kinetics["OATP1B1"]
    assert isinstance(tk, TransporterKinetics)
    assert tk.jmax.mean == pytest.approx(228.0)
    assert tk.jmax.cv == pytest.approx(0.30)
    assert tk.km.mean == pytest.approx(13.6)
    assert tk.km.cv == pytest.approx(0.25)


def test_load_unknown_drug_returns_none():
    assert load_oatp1b1_kinetics("aspirin") is None


def test_load_is_case_insensitive():
    assert load_oatp1b1_kinetics("Pravastatin") is not None
    assert load_oatp1b1_kinetics("PRAVASTATIN") is not None


def test_build_drug_on_graph_carries_transporter_kinetics():
    """Passing transporter_kinetics to build_drug_on_graph populates the field."""
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    pravastatin_smiles = (
        "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
        "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
    )
    profile = compute_profile(pravastatin_smiles)
    adme = predict_adme(profile)
    kinetics = load_oatp1b1_kinetics("pravastatin")

    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        transporter_kinetics=kinetics,
    )
    assert "OATP1B1" in drug.transporter_kinetics
    assert drug.transporter_kinetics["OATP1B1"].jmax.mean == pytest.approx(228.0)


def test_build_drug_on_graph_without_transporter_is_empty():
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    profile = compute_profile("CCO")  # ethanol
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=10.0, route="oral")
    assert drug.transporter_kinetics == {}


def test_load_hepatic_ecm_params_pravastatin():
    """load_hepatic_ecm_params returns PS/biliary Distributions for curated drugs."""
    from sisyphus.predict.transporter_db import load_hepatic_ecm_params
    params = load_hepatic_ecm_params("pravastatin")
    assert params is not None
    assert "ps_passive" in params
    assert "ps_eff" in params
    assert "cl_int_bile" in params
    assert 0.0 < params["ps_passive"].mean < 100.0
    assert 0.0 < params["ps_eff"].mean < 100.0
    assert params["cl_int_bile"].mean >= 0.0


def test_load_hepatic_ecm_params_unknown_drug_returns_none():
    from sisyphus.predict.transporter_db import load_hepatic_ecm_params
    assert load_hepatic_ecm_params("unknowndrug_xyz") is None


def test_load_hepatic_ecm_params_case_insensitive():
    from sisyphus.predict.transporter_db import load_hepatic_ecm_params
    p_lower = load_hepatic_ecm_params("pravastatin")
    p_upper = load_hepatic_ecm_params("PRAVASTATIN")
    assert p_lower is not None and p_upper is not None
    assert p_lower["ps_passive"].mean == p_upper["ps_passive"].mean


import json as _json
import pathlib as _pathlib

import pytest as _pytest

# Pre-registration envelopes per spec amendment v2 (6e7ce0a), 2026-04-21.
# Substrate set: valsartan + glimepiride (N=2).
# Bosentan and repaglinide dropped from test set; archived in
# dropped_under_amendment_v2 key of oatp1b1.json.
_GENERALIZATION_DRUGS_EXPECTED = {
    # valsartan: Km verified (Niemi 2009 PMC2765590 Table 2 citing Yamashiro 2006).
    # Jmax not found in any open-access source (searched Shitara 2013, Karlgren 2012,
    # Kalliokoski 2010, Maeda 2006, Poirier 2009, PBPK reviews, PMC). Case B: BLOCKED.
    "valsartan": {"km_range_uM": (1.0, 1.8), "jmax_range": (30.0, 80.0)},
    # glimepiride: Km 10.02 µM (Chen 2018 PMID 29498478, HEK293-OATP1B1);
    # Jmax 155 pmol/min/mg (Yang 2018 PMC6054689 Fig 5, OATP1B1*1a HEK293T).
    "glimepiride": {"km_range_uM": (8.0, 13.0), "jmax_range": (120.0, 200.0)},
}

# Literature extraction outcome 2026-04-21 (amendment v2):
# glimepiride: Km=10.02 µM (Chen 2018 PMID 29498478), Jmax=155 pmol/min/mg
#   (Yang 2018 PMC6054689 Fig 5, OATP1B1*1a). VERIFIED.
# valsartan: Km=1.39 µM VERIFIED (Niemi 2009 PMC2765590). Jmax BLOCKED — paywalled
#   in Yamashiro 2006 DMD (DOI 10.1124/dmd.106.009365); not found in any open-access
#   secondary source after exhaustive search. Remains in blocked_drugs. Case B.
_VERIFIED_KINETICS_DRUGS: set[str] = {"glimepiride"}
_BLOCKED_KINETICS_DRUGS: set[str] = {"valsartan"}

_OATP1B1_FILE = (
    _pathlib.Path(__file__).resolve().parents[2]
    / "data" / "transporters" / "oatp1b1.json"
)


def _load_oatp1b1_json() -> dict:
    with _OATP1B1_FILE.open() as f:
        return _json.load(f)


# ── Tests for VERIFIED kinetics drugs (empty set — all blocked) ──────────────
# The parametrize over an empty set produces 0 test cases (harmless; acts as
# a placeholder so that when future drugs are verified, the test structure works).

@_pytest.mark.parametrize("drug", sorted(_VERIFIED_KINETICS_DRUGS))
def test_generalization_drug_has_entry(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    assert kinetics is not None, f"{drug} missing from oatp1b1.json drugs key"
    assert "OATP1B1" in kinetics


@_pytest.mark.parametrize("drug", sorted(_VERIFIED_KINETICS_DRUGS))
def test_generalization_drug_km_in_envelope(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    km = kinetics["OATP1B1"].km.mean
    lo, hi = _GENERALIZATION_DRUGS_EXPECTED[drug]["km_range_uM"]
    assert lo <= km <= hi, f"{drug} Km {km} outside [{lo}, {hi}] uM"


@_pytest.mark.parametrize("drug", sorted(_VERIFIED_KINETICS_DRUGS))
def test_generalization_drug_jmax_in_envelope(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    jmax = kinetics["OATP1B1"].jmax.mean
    lo, hi = _GENERALIZATION_DRUGS_EXPECTED[drug]["jmax_range"]
    assert lo <= jmax <= hi, f"{drug} Jmax {jmax} outside [{lo}, {hi}] pmol/min/mg"


@_pytest.mark.parametrize("drug", sorted(_VERIFIED_KINETICS_DRUGS))
def test_generalization_drug_cv_widened(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    assert kinetics["OATP1B1"].jmax.cv >= 0.30, f"{drug} Jmax CV must be >= 0.30"
    assert kinetics["OATP1B1"].km.cv >= 0.25, f"{drug} Km CV must be >= 0.25"


# ── Tests for BLOCKED drugs ───────────────────────────────────────────────────

@_pytest.mark.parametrize("drug", sorted(_BLOCKED_KINETICS_DRUGS))
def test_blocked_drug_returns_none_from_loader(drug: str):
    """Blocked drugs are absent from 'drugs' key → loader returns None."""
    kinetics = load_oatp1b1_kinetics(drug)
    assert kinetics is None, (
        f"{drug} should return None (it is in blocked_drugs, not drugs). "
        f"If Jmax was verified and the entry was promoted to drugs, update "
        f"_VERIFIED_KINETICS_DRUGS and remove from _BLOCKED_KINETICS_DRUGS."
    )


@_pytest.mark.parametrize("drug", sorted(_BLOCKED_KINETICS_DRUGS))
def test_blocked_drug_documented_in_json(drug: str):
    """Blocked drugs must appear in 'blocked_drugs' key with status and reason."""
    data = _load_oatp1b1_json()
    assert "blocked_drugs" in data, "oatp1b1.json missing 'blocked_drugs' top-level key"
    assert drug in data["blocked_drugs"], f"{drug} missing from blocked_drugs"
    entry = data["blocked_drugs"][drug]
    assert entry.get("status") == "BLOCKED", f"{drug} blocked_drugs entry missing status=BLOCKED"
    assert "blocked_reason" in entry, f"{drug} missing blocked_reason"
    assert "sources_attempted" in entry, f"{drug} missing sources_attempted"
    assert len(entry["sources_attempted"]) >= 3, (
        f"{drug} sources_attempted must list >= 3 sources tried"
    )


@_pytest.mark.parametrize("drug", sorted(_BLOCKED_KINETICS_DRUGS))
def test_blocked_drug_plan_envelope_recorded(drug: str):
    """Blocked drugs must record the pre-registered plan envelope for auditability."""
    data = _load_oatp1b1_json()
    entry = data["blocked_drugs"][drug]
    assert "plan_expected_envelope" in entry, f"{drug} missing plan_expected_envelope"
    env = entry["plan_expected_envelope"]
    assert "km_range_uM" in env and "jmax_range" in env, (
        f"{drug} plan_expected_envelope must contain km_range_uM and jmax_range"
    )
