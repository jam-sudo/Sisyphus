"""Tests for the pre-registered external-holdout scorer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sisyphus.validation.holdout_contract import (
    source_record_hash,
    validate_payload,
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
