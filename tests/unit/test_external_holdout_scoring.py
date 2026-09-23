"""Tests for the pre-registered external-holdout scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisyphus.validation.holdout_contract import (
    sha256_file,
    source_record_hash,
    validate_payload,
    verify_source_plan,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def _module():
    path = ROOT / "scripts" / "score_external_holdout.py"
    spec = importlib.util.spec_from_file_location("score_external_holdout", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_weights_compounds_not_arms():
    scorer = _module()
    rows = [
        {
            "candidate_id": "a",
            "primary_eligible": True,
            "observed_cmax_mg_l": 1.0,
            "meta_cmax_mg_l": 1.0,
            "ml_cmax_mg_l": 2.0,
        },
        {
            "candidate_id": "a",
            "primary_eligible": True,
            "observed_cmax_mg_l": 2.0,
            "meta_cmax_mg_l": 2.0,
            "ml_cmax_mg_l": 4.0,
        },
        {
            "candidate_id": "b",
            "primary_eligible": True,
            "observed_cmax_mg_l": 1.0,
            "meta_cmax_mg_l": 4.0,
            "ml_cmax_mg_l": 4.0,
        },
    ]
    result = scorer.score(rows, seed=7, n_boot=1000)
    assert result["n_compounds"] == 2
    assert result["n_arms"] == 3
    assert result["meta_aafe"] == pytest.approx(2.0)
    assert result["ml_aafe"] == pytest.approx(2.0 * 2**0.5)
    assert result["meta_ml_aafe_ratio"] == pytest.approx(2**-0.5)


def test_score_rejects_nonpositive_observation():
    scorer = _module()
    rows = [
        {
            "candidate_id": "a",
            "primary_eligible": True,
            "observed_cmax_mg_l": 0.0,
            "meta_cmax_mg_l": 1.0,
            "ml_cmax_mg_l": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="Non-positive"):
        scorer.score(rows, seed=7, n_boot=10)


def test_manifest_validation_rejects_arm_input_drift():
    scorer = _module()
    manifest = {
        "n_target": 1,
        "compounds": [
            {
                "candidate_id": "a",
                "arms": [
                    {
                        "arm_id": "arm1",
                        "dose_mg": 10.0,
                        "route": "oral",
                        "primary_eligible": True,
                    }
                ],
            }
        ],
    }
    row = {
        "candidate_id": "a",
        "arm_id": "arm1",
        "dose_mg": 20.0,
        "route": "oral",
        "primary_eligible": True,
        "observed_cmax_mg_l": 1.0,
        "meta_cmax_mg_l": 1.0,
        "ml_cmax_mg_l": 1.0,
        "execution_status": "ok",
        "interval_source": "development_empirical_residual",
        "source_record_hash": "0" * 64,
        "source": {"category": "regulatory", "agency": "FDA"},
    }
    with pytest.raises(ValueError, match="Dose mismatch"):
        scorer.validate_results_against_manifest([row], manifest)


def test_label_join_is_exact_and_keeps_outcome_separate():
    scorer = _module()
    predictions = [
        {
            "candidate_id": "a",
            "arm_id": "arm1",
            "dose_mg": 10.0,
            "route": "oral",
            "primary_eligible": True,
            "meta_cmax_mg_l": 1.2,
            "ml_cmax_mg_l": 1.4,
        }
    ]
    labels = {
        "records": [
            {
                "candidate_id": "a",
                "arms": [
                    {
                        "arm_id": "arm1",
                        "dose_mg": 10.0,
                        "route": "oral",
                        "observed_cmax_mg_l": 1.0,
                    }
                ],
            }
        ]
    }
    joined = scorer.join_predictions_and_labels(predictions, labels)
    assert "observed_cmax_mg_l" not in predictions[0]
    assert joined[0]["observed_cmax_mg_l"] == 1.0


def _synthetic_contracts(n: int = 120):
    zero = "0" * 64
    agencies = ["FDA", "EMA", "PMDA", "HealthCanada"]
    compounds = []
    predictions = []
    records = []
    for i in range(n):
        regulatory = i < 84
        agency = agencies[i % len(agencies)] if regulatory else None
        category = "regulatory" if regulatory else "peer_reviewed"
        label = {
            "arm_id": "arm1",
            "dose_mg": 10.0,
            "route": "oral",
            "dosage_form": "tablet",
            "release_type": "IR",
            "food_state": "fasted",
            "salt_form": None,
            "analyte": "parent",
            "matrix": "plasma",
            "dose_regimen": "single",
            "population": {"age_group": "adult", "health_status": "healthy"},
            "co_medications": [],
            "observed_cmax_mg_l": 1.0,
            "study_n": 12,
            "source": {
                "category": category,
                "agency": agency,
                "source_family": agency or category,
                "source_date": "2025-01-01",
                "citation": f"source {i}",
                "url_or_doi": f"https://example.test/{i}",
                "table_or_page": "p. 1",
            },
            "verified_by": ["curator-a", "curator-b"],
        }
        label["source_record_hash"] = source_record_hash(label)
        compounds.append(
            {
                "candidate_id": f"c{i}",
                "name": f"compound-{i}",
                "smiles": "C" * (i + 1),
                "arms": [
                    {
                        "arm_id": "arm1",
                        "dose_mg": 10.0,
                        "route": "oral",
                        "primary_eligible": True,
                        "source_category": category,
                        "source_agency": agency,
                        "source_record_hash": label["source_record_hash"],
                    }
                ],
            }
        )
        predictions.append(
            {
                "candidate_id": f"c{i}",
                "arm_id": "arm1",
                "dose_mg": 10.0,
                "route": "oral",
                "primary_eligible": True,
                "meta_cmax_mg_l": 1.1,
                "ml_cmax_mg_l": 1.3,
                "meta_pi90_low_mg_l": 0.1,
                "meta_pi90_high_mg_l": 10.0,
                "interval_source": "development_empirical_residual",
                "execution_status": "ok",
            }
        )
        records.append({"candidate_id": f"c{i}", "arms": [label]})

    manifest = {
        "protocol": "docs/validation/external_holdout_v1_protocol.md",
        "cycle_id": "synthetic-v1",
        "n_target": n,
        "labels_blinded": True,
        "source_plan_path": "source-plan.json",
        "source_plan_sha256": zero,
        "exclusion_union_sha256": zero,
        "freeze": {
            "git_sha": "0" * 40,
            "source_tree_sha256": zero,
            "artifact_inventory_sha256": zero,
            "training_membership_path": "training.json",
            "training_membership_sha256": zero,
            "feature_schema_path": "features.json",
            "feature_schema_sha256": zero,
            "solver_settings_path": "solver.json",
            "solver_settings_sha256": zero,
            "dependency_lock_sha256": zero,
            "container_digest": "sha256:" + zero,
            "random_seed": 7,
            "resource_profile": "public",
        },
        "compounds": compounds,
    }
    payload = {
        "cycle_id": "synthetic-v1",
        "manifest_sha256": zero,
        "audit_report_sha256": zero,
        "git_sha": "0" * 40,
        "source_tree_sha256": zero,
        "dependency_lock_sha256": zero,
        "artifact_inventory_sha256": zero,
        "training_membership_sha256": zero,
        "feature_schema_sha256": zero,
        "solver_settings_sha256": zero,
        "container_digest": "sha256:" + zero,
        "artifact_provenance": {"resource_profile": "public"},
        "rows": predictions,
    }
    labels = {"cycle_id": "synthetic-v1", "manifest_sha256": zero, "records": records}
    return manifest, payload, labels


def test_full_synthetic_holdout_contract_and_scoring():
    scorer = _module()
    manifest, payload, labels = _synthetic_contracts()
    validate_payload(manifest, "external_holdout_v1_manifest.schema.json")
    validate_payload(payload, "external_holdout_v1_predictions.schema.json")
    validate_payload(labels, "external_holdout_v1_labels.schema.json")
    joined = scorer.join_predictions_and_labels(payload["rows"], labels)
    scorer.validate_results_against_manifest(joined, manifest)
    result = scorer.score(joined, seed=7, n_boot=100)
    assert result["n_compounds"] == 120


def test_scorer_rejects_manifest_eligibility_forgery():
    scorer = _module()
    manifest, payload, labels = _synthetic_contracts()
    labels["records"][0]["arms"][0]["food_state"] = "fed"
    forged = labels["records"][0]["arms"][0]
    forged["source_record_hash"] = source_record_hash(forged)
    manifest["compounds"][0]["arms"][0]["source_record_hash"] = forged[
        "source_record_hash"
    ]
    joined = scorer.join_predictions_and_labels(payload["rows"], labels)
    with pytest.raises(ValueError, match="Derived eligibility mismatch"):
        scorer.validate_results_against_manifest(joined, manifest)


def test_scorer_rejects_clinical_source_outside_frozen_window():
    scorer = _module()
    manifest, predictions, labels = _synthetic_contracts()
    arm = labels["records"][0]["arms"][0]
    arm["source"]["source_date"] = "2019-01-01"
    arm["source_record_hash"] = source_record_hash(arm)
    manifest["compounds"][0]["arms"][0]["source_record_hash"] = arm["source_record_hash"]
    joined = scorer.join_predictions_and_labels(predictions["rows"], labels)
    windows = [
        {"source_family": family, "start_date": "2020-01-01", "end_date": "2026-01-01"}
        for family in ("FDA", "EMA", "PMDA", "HealthCanada", "peer_reviewed")
    ]
    with pytest.raises(ValueError, match="outside frozen source window"):
        scorer.validate_results_against_manifest(joined, manifest, windows)


def test_scorer_rejects_verifier_absent_from_source_plan():
    scorer = _module()
    manifest, predictions, labels = _synthetic_contracts()
    arm = labels["records"][0]["arms"][0]
    arm["verified_by"] = ["curator-a", "stranger"]
    arm["source_record_hash"] = source_record_hash(arm)
    manifest["compounds"][0]["arms"][0]["source_record_hash"] = arm["source_record_hash"]
    joined = scorer.join_predictions_and_labels(predictions["rows"], labels)
    with pytest.raises(ValueError, match="Unregistered verifier"):
        scorer.validate_results_against_manifest(
            joined, manifest, curators=["curator-a", "curator-b"]
        )


def test_cli_uses_frozen_seed_and_bootstrap_count(tmp_path, monkeypatch):
    scorer = _module()
    manifest, predictions, labels = _synthetic_contracts()
    inventory = [
        {
            "candidate_id": f"c{i}", "name": f"compound-{i}",
            "source_family": "FDA", "source_date": "2025-01-01", "source_ref": f"nda-{i}",
        }
        for i in range(900)
    ]
    verified = [
        {"candidate_id": f"c{i}", "name": f"compound-{i}", "smiles": "C" * (i + 1)}
        for i in range(550)
    ]
    allocation = {
        "calibration": [f"c{i}" for i in range(120, 260)],
        "final_test": [f"c{i}" for i in range(120)],
        "reserve": [f"c{i}" for i in range(260, 550)],
    }
    flow = [
        {
            "candidate_id": f"c{i}",
            "decision": "verified" if i < 550 else "excluded",
            "reason": "source ineligible" if i >= 550 else "",
        }
        for i in range(900)
    ]
    plan = {
        "protocol": "external_holdout_v1", "cycle_id": "synthetic-v1",
        "fixed_before_prediction": True, "inventory_n": 900, "verified_n": 550,
        "calibration_n": 140, "final_test_n": 120, "reserve_n": 290,
        "source_windows": [
            {"source_family": "FDA", "start_date": "2020-01-01", "end_date": "2026-01-01"},
            {"source_family": "EMA", "start_date": "2020-01-01", "end_date": "2026-01-01"},
            {"source_family": "PMDA", "start_date": "2020-01-01", "end_date": "2026-01-01"},
            {"source_family": "HealthCanada", "start_date": "2020-01-01", "end_date": "2026-01-01"},
            {
                "source_family": "peer_reviewed",
                "start_date": "2020-01-01",
                "end_date": "2026-01-01",
            },
        ],
        "curators": ["curator-a", "curator-b"],
    }
    for stem, contents in (
        ("inventory", inventory), ("verified_shortlist", verified),
        ("allocation", allocation), ("exclusion_flow", flow),
    ):
        path = tmp_path / f"{stem}.json"
        path.write_text(json.dumps(contents))
        plan[f"{stem}_path"] = path.name
        plan[f"{stem}_sha256"] = sha256_file(path)
    plan_path = tmp_path / "source-plan.json"
    plan_path.write_text(json.dumps(plan))
    manifest["source_plan_sha256"] = sha256_file(plan_path)
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.json"
    output_path = tmp_path / "score.json"
    manifest_path.write_text(json.dumps(manifest))
    verify_source_plan(manifest_path, manifest, lambda smiles: smiles)
    allocation["calibration"][0] = "c0"
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(allocation))
    plan["allocation_sha256"] = sha256_file(allocation_path)
    plan_path.write_text(json.dumps(plan))
    manifest["source_plan_sha256"] = sha256_file(plan_path)
    with pytest.raises(ValueError, match="partition"):
        verify_source_plan(manifest_path, manifest)
    allocation["calibration"][0] = "c120"
    allocation_path.write_text(json.dumps(allocation))
    plan["allocation_sha256"] = sha256_file(allocation_path)
    plan_path.write_text(json.dumps(plan))
    manifest["source_plan_sha256"] = sha256_file(plan_path)
    verified[0]["smiles"] = verified[1]["smiles"]
    verified_path = tmp_path / "verified_shortlist.json"
    verified_path.write_text(json.dumps(verified))
    plan["verified_shortlist_sha256"] = sha256_file(verified_path)
    plan_path.write_text(json.dumps(plan))
    manifest["source_plan_sha256"] = sha256_file(plan_path)
    with pytest.raises(ValueError, match="share a salt/stereo family"):
        verify_source_plan(manifest_path, manifest, lambda smiles: smiles)
    verified[0]["smiles"] = "C"
    verified_path.write_text(json.dumps(verified))
    plan["verified_shortlist_sha256"] = sha256_file(verified_path)
    plan_path.write_text(json.dumps(plan))
    manifest["source_plan_sha256"] = sha256_file(plan_path)
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    predictions["manifest_sha256"] = manifest_sha
    labels["manifest_sha256"] = manifest_sha
    predictions_path.write_text(json.dumps(predictions))
    labels_path.write_text(json.dumps(labels))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_external_holdout.py", str(predictions_path),
            "--labels", str(labels_path),
            "--manifest", str(manifest_path),
            "--manifest-sha256", manifest_sha,
            "--predictions-sha256", hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            "--labels-sha256", hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "--out", str(output_path),
        ],
    )
    scorer.main()
    report = json.loads(output_path.read_text())
    assert (report["seed"], report["n_bootstrap"]) == (7, 100000)
