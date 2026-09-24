"""Executable external-holdout eligibility, provenance, and quota contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sisyphus.validation.holdout_contract as contract
from sisyphus.validation.holdout_contract import (
    _PRODUCTION_FITTED_MODELS,
    is_primary_eligible,
    primary_ineligibility_reasons,
    sha256_file,
    source_record_hash,
    validate_payload,
    validate_source_quotas,
    verify_training_membership,
)

ROOT = Path(__file__).resolve().parents[2]


def _label_arm() -> dict:
    return {
        "arm_id": "a1",
        "dose_mg": 10.0,
        "route": "oral",
        "dosage_form": "tablet",
        "release_type": "IR",
        "food_state": "fasted",
        "postdose_fast_h": 4.0,
        "salt_form": None,
        "dose_basis": "parent_active_moiety",
        "dose_basis_evidence": "Source table reports 10 mg of parent drug.",
        "analyte": "parent",
        "matrix": "plasma",
        "dose_regimen": "single",
        "population": {"age_group": "adult", "health_status": "healthy"},
        "co_medications": [],
        "cmax_statistic": "arithmetic_mean",
        "study_n": 12,
        "source": {
            "category": "regulatory",
            "agency": "FDA",
            "source_family": "FDA",
            "source_date": "2025-01-01",
            "citation": "review",
            "url_or_doi": "https://example.test/review",
            "table_or_page": "p. 10",
        },
        "verified_by": ["curator-a", "curator-b"],
    }


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("route", "iv", "route_not_oral"),
        ("release_type", "ER", "release_not_ir"),
        ("food_state", "fed", "not_fasted"),
        ("postdose_fast_h", 2.0, "postdose_fast_under_4h"),
        ("postdose_fast_h", None, "postdose_fast_under_4h"),
        ("dose_basis", "unknown", "dose_basis_unverified"),
        ("dose_basis_evidence", "", "dose_basis_evidence_missing"),
        ("analyte", "active_metabolite", "not_parent_analyte"),
        ("matrix", "serum", "matrix_not_plasma"),
        ("dose_regimen", "multiple_steady_state", "not_single_dose"),
        ("co_medications", ["ritonavir"], "co_medication_present"),
    ],
)
def test_primary_eligibility_is_derived(field, value, reason):
    arm = _label_arm()
    arm[field] = value
    assert not is_primary_eligible(arm)
    assert reason in primary_ineligibility_reasons(arm)


def test_source_record_hash_binds_metadata_but_not_outcome():
    arm = _label_arm()
    first = source_record_hash(arm)
    arm["observed_cmax_mg_l"] = 9.9
    assert source_record_hash(arm) == first
    arm["food_state"] = "fed"
    assert source_record_hash(arm) != first
    arm["food_state"] = "fasted"
    arm["verified_by"] = ["curator-a", "curator-c"]
    assert source_record_hash(arm) != first
    arm["verified_by"] = ["curator-a", "curator-b"]
    arm["dose_basis_evidence"] = "Different source table"
    assert source_record_hash(arm) != first
    arm["dose_basis_evidence"] = "Source table reports 10 mg of parent drug."
    arm["cmax_statistic"] = "geometric_mean"
    assert source_record_hash(arm) != first


def test_source_quotas_are_compound_weighted_and_enforced():
    compounds = []
    agencies = ["FDA", "EMA", "PMDA", "HC"]
    for i in range(10):
        regulatory = i < 7
        compounds.append(
            {
                "candidate_id": f"c{i}",
                "arms": [
                    {
                        "primary_eligible": True,
                        "source_category": "regulatory" if regulatory else "peer_reviewed",
                        "source_agency": agencies[i % 4] if regulatory else None,
                    }
                ],
            }
        )
    metrics = validate_source_quotas({"compounds": compounds})
    assert metrics["regulatory_fraction"] == pytest.approx(0.7)

    bad = copy.deepcopy(compounds)
    for compound in bad[:7]:
        compound["arms"][0]["source_agency"] = "FDA"
    with pytest.raises(ValueError, match="above 0.30"):
        validate_source_quotas({"compounds": bad})


def test_source_plan_schema_enforces_acquisition_and_three_way_allocation():
    zero = "0" * 64
    plan = {
        "protocol": "external_holdout_v1",
        "cycle_id": "cycle-1",
        "fixed_before_prediction": True,
        "inventory_n": 900,
        "inventory_path": "inventory.json",
        "inventory_sha256": zero,
        "verified_n": 550,
        "verified_shortlist_path": "verified.json",
        "verified_shortlist_sha256": zero,
        "calibration_n": 140,
        "final_test_n": 260,
        "reserve_n": 150,
        "allocation_path": "allocation.json",
        "allocation_sha256": zero,
        "exclusion_flow_path": "exclusion.json",
        "exclusion_flow_sha256": zero,
        "label_content_sha256": zero,
        "source_windows": [
            {"source_family": "FDA", "start_date": "2020-01-01", "end_date": "2026-01-01"},
            {"source_family": "EMA", "start_date": "2020-01-01", "end_date": "2026-01-01"},
        ],
        "curators": ["curator-a", "curator-b"],
    }
    validate_payload(plan, "external_holdout_v1_source_plan.schema.json")
    for nonfinite in (float("nan"), float("inf")):
        plan["inventory_n"] = nonfinite
        with pytest.raises(ValueError, match="non-finite"):
            validate_payload(plan, "external_holdout_v1_source_plan.schema.json")
    plan["inventory_n"] = 899
    with pytest.raises(ValueError, match="minimum of 900"):
        validate_payload(plan, "external_holdout_v1_source_plan.schema.json")


def test_public_training_membership_sources_are_complete_and_hash_pinned():
    membership = json.loads(
        (ROOT / "data/validation/training_membership_sources_v1.json").read_text()
    )
    expected = {
        "data/ppbr_az.tab",
        "data/training/fup_tdc_public_clean.csv",
        "data/caco2_wang.tab",
        "data/training/peff_tdc_public_clean.csv",
        "data/training/omega_mmpk_clean.csv",
        "data/training/cmax_omega_public_clean.csv",
        "data/training/mmpk_expanded_full.csv",
        "data/training/mmpk_expanded_v2.csv",
        "data/training/mmpk_pbpk_features.csv",
        "data/training/clf_training.csv",
        "data/training/bioavailability_v1.csv",
        "data/training/clint_expanded_v2.csv",
        "data/training/clint_merged_v3_biogen.csv",
        "data/training/vdss_v2_training.csv",
        "data/training/clearance_hepatocyte_az.tab",
    }
    assert membership["profile"] == "public"
    assert {row["path"] for row in membership["sources"]} == expected
    for row in membership["sources"]:
        assert sha256_file(ROOT / row["path"]) == row["sha256"]


def test_clf_and_vdf_manifests_pin_their_co_committed_training_source():
    source = "data/training/clf_training.csv"
    digest = sha256_file(ROOT / source)
    for model in ("xgboost_clf", "xgboost_vdf"):
        metadata = json.loads((ROOT / f"models/direct_pk/{model}.meta.json").read_text())
        assert metadata["trained_on"] == {"dataset_path": source, "sha256": digest}
        assert metadata["artifact_sha256"] == sha256_file(
            ROOT / f"models/direct_pk/{model}.json"
        )


def test_training_membership_refuses_missing_or_modified_source(tmp_path, monkeypatch):
    inventory_path = tmp_path / "data/validation/training_membership_sources_v1.json"
    inventory_path.parent.mkdir(parents=True)
    source_path = tmp_path / "data/training/example.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("original")
    inventory_path.write_text(json.dumps({
        "profile": "public",
        "sources": [{"path": "data/training/example.csv", "sha256": sha256_file(source_path)}],
    }))
    for model_path in _PRODUCTION_FITTED_MODELS:
        metadata_path = tmp_path / model_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({"trained_on": {
            "dataset_path": "data/training/example.csv",
            "sha256": sha256_file(source_path),
        }}))
    fup_artifact = tmp_path / "models/adme/xgboost_fup_v2.json"
    fup_artifact.write_text("placeholder")
    freeze = {
        "training_membership_path": "data/validation/training_membership_sources_v1.json",
        "training_membership_sha256": sha256_file(inventory_path),
    }
    assert verify_training_membership(
        tmp_path, freeze, {"data/training/example.csv"}
    ) == sha256_file(inventory_path)
    cmax_meta = tmp_path / _PRODUCTION_FITTED_MODELS[0]
    cmax_meta.write_text(json.dumps({"trained_on": {
        "dataset_path": "mmpk_clean.csv (Omega)", "sha256": "unknown_legacy",
    }}))
    with pytest.raises(ValueError, match="Unverifiable production model training source"):
        verify_training_membership(tmp_path, freeze, {"data/training/example.csv"})
    cmax_meta.write_text(json.dumps({"trained_on": {
        "dataset_path": "data/training/example.csv", "sha256": "0" * 64,
    }}))
    with pytest.raises(ValueError, match="Production model training source SHA256 mismatch"):
        verify_training_membership(tmp_path, freeze, {"data/training/example.csv"})
    cmax_meta.write_text(json.dumps({"trained_on": {
        "dataset_path": "data/training/example.csv", "sha256": sha256_file(source_path),
    }}))
    fup_artifact.write_bytes((ROOT / "models/adme/xgboost_fup_v2.json").read_bytes())
    monkeypatch.setattr(contract, "_DRUGBANK_FUP_ARTIFACT_SHA256", sha256_file(fup_artifact))
    with pytest.raises(ValueError, match="DrugBank targets outside the public profile"):
        verify_training_membership(tmp_path, freeze, {"data/training/example.csv"})
    fup_artifact.write_text("placeholder")
    source_path.write_text("modified")
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        verify_training_membership(tmp_path, freeze, {"data/training/example.csv"})
    source_path.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        verify_training_membership(tmp_path, freeze, {"data/training/example.csv"})
