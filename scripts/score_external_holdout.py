#!/usr/bin/env python3
"""One-time compound-cluster scoring for a frozen external holdout.

Frozen predictions and the separately controlled label store are joined by exact
candidate/arm key only at scoring time. This scorer performs no model fitting or
routing and compares Meta with direct ML using the pre-registered paired estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from sisyphus.validation.holdout_contract import (
    is_primary_eligible,
    label_content_sha256,
    source_record_hash,
    validate_payload,
    validate_source_quotas,
    verify_source_plan,
)


def _compound_errors(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pred = float(row[key])
        obs = float(row["observed_cmax_mg_l"])
        if pred <= 0 or obs <= 0:
            raise ValueError(f"Non-positive prediction/observation in {row['candidate_id']}")
        grouped[row["candidate_id"]].append(abs(np.log(pred / obs)))
    return np.asarray([np.mean(grouped[cid]) for cid in sorted(grouped)], dtype=float)


def _compound_signed_errors(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pred = float(row[key])
        obs = float(row["observed_cmax_mg_l"])
        grouped[row["candidate_id"]].append(np.log(pred / obs))
    return np.asarray([np.mean(grouped[cid]) for cid in sorted(grouped)], dtype=float)


def _paired_bootstrap(
    meta: np.ndarray,
    ml: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster bootstrap without allocating a 100000 x N index matrix."""
    meta_boot = np.empty(n_boot, dtype=float)
    ml_boot = np.empty(n_boot, dtype=float)
    ratio_boot = np.empty(n_boot, dtype=float)
    paired = meta - ml
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        idx = rng.integers(0, len(meta), size=(stop - start, len(meta)))
        meta_boot[start:stop] = np.exp(meta[idx].mean(axis=1))
        ml_boot[start:stop] = np.exp(ml[idx].mean(axis=1))
        ratio_boot[start:stop] = np.exp(paired[idx].mean(axis=1))
    return meta_boot, ml_boot, ratio_boot


def validate_results_against_manifest(
    rows: list[dict],
    manifest: dict,
    source_windows: list[dict] | None = None,
    curators: list[str] | None = None,
) -> None:
    """Require an exact candidate/arm/input match to the frozen manifest."""
    expected: dict[tuple[str, str], dict] = {}
    for compound in manifest.get("compounds", []):
        cid = str(compound.get("candidate_id", ""))
        for arm in compound.get("arms", []):
            key = (cid, str(arm.get("arm_id", "")))
            if key in expected:
                raise ValueError(f"Duplicate manifest arm: {key}")
            expected[key] = arm

    actual: dict[tuple[str, str], dict] = {}
    required = {
        "candidate_id",
        "arm_id",
        "dose_mg",
        "route",
        "primary_eligible",
        "observed_cmax_mg_l",
        "meta_cmax_mg_l",
        "ml_cmax_mg_l",
        "execution_status",
        "interval_source",
        "source_record_hash",
        "source",
    }
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Result row missing {missing}")
        key = (str(row["candidate_id"]), str(row["arm_id"]))
        if key in actual:
            raise ValueError(f"Duplicate result arm: {key}")
        actual[key] = row

    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"Result/manifest arm mismatch: missing={missing}, extra={extra}")
    for key, arm in expected.items():
        row = actual[key]
        if not np.isclose(float(row["dose_mg"]), float(arm["dose_mg"]), rtol=0, atol=1e-12):
            raise ValueError(f"Dose mismatch for {key}")
        if row["route"] != arm["route"]:
            raise ValueError(f"Route mismatch for {key}")
        if row["primary_eligible"] is not arm["primary_eligible"]:
            raise ValueError(f"Eligibility mismatch for {key}")
        derived_eligible = is_primary_eligible(row)
        if derived_eligible is not arm["primary_eligible"]:
            raise ValueError(
                f"Derived eligibility mismatch for {key}: "
                f"derived={derived_eligible}, manifest={arm['primary_eligible']}"
            )
        computed_source_hash = source_record_hash(row)
        if row["source_record_hash"] != computed_source_hash:
            raise ValueError(f"Label source_record_hash is invalid for {key}")
        if arm["source_record_hash"] != computed_source_hash:
            raise ValueError(f"Manifest/label source_record_hash mismatch for {key}")
        if curators is not None and not set(row["verified_by"]) <= set(curators):
            raise ValueError(f"Unregistered verifier for {key}")
        source = row["source"]
        if source["category"] != arm["source_category"]:
            raise ValueError(f"Source category mismatch for {key}")
        if source.get("agency") != arm.get("source_agency"):
            raise ValueError(f"Source agency mismatch for {key}")
        if source_windows is not None:
            expected_family = (
                source["agency"] if source["category"] == "regulatory" else source["category"]
            )
            source_date = date.fromisoformat(source["source_date"])
            if source["source_family"] != expected_family or not any(
                window["source_family"] == expected_family
                and date.fromisoformat(window["start_date"]) <= source_date
                <= date.fromisoformat(window["end_date"])
                for window in source_windows
            ):
                raise ValueError(f"Clinical source is outside frozen source window: {key}")
        if arm["primary_eligible"]:
            if row["execution_status"] != "ok":
                raise ValueError(f"Primary prediction execution failed for {key}")
            if row["interval_source"] != "development_empirical_residual":
                raise ValueError(f"Unexpected primary interval source for {key}")
            lo = row.get("meta_pi90_low_mg_l")
            hi = row.get("meta_pi90_high_mg_l")
            if lo is None or hi is None or not (0 < float(lo) < float(hi)):
                raise ValueError(f"Invalid or missing primary residual interval for {key}")

    expected_primary = int(manifest["n_target"])
    actual_primary = len(
        {
            str(row["candidate_id"])
            for row in rows
            if row["primary_eligible"] is True
        }
    )
    if actual_primary != expected_primary:
        raise ValueError(
            f"Primary compound count {actual_primary} != frozen n_target {expected_primary}"
        )


def join_predictions_and_labels(prediction_rows: list[dict], labels: dict) -> list[dict]:
    """Attach observed Cmax after enforcing a one-to-one blinded arm join."""
    label_arms: dict[tuple[str, str], dict] = {}
    for record in labels.get("records", []):
        cid = str(record.get("candidate_id", ""))
        for arm in record.get("arms", []):
            key = (cid, str(arm.get("arm_id", "")))
            if key in label_arms:
                raise ValueError(f"Duplicate label arm: {key}")
            label_arms[key] = arm

    prediction_keys = {
        (str(row.get("candidate_id", "")), str(row.get("arm_id", "")))
        for row in prediction_rows
    }
    if prediction_keys != label_arms.keys():
        missing = sorted(prediction_keys - label_arms.keys())
        extra = sorted(label_arms.keys() - prediction_keys)
        raise ValueError(f"Prediction/label arm mismatch: missing={missing}, extra={extra}")

    joined: list[dict] = []
    for row in prediction_rows:
        key = (str(row["candidate_id"]), str(row["arm_id"]))
        label = label_arms[key]
        if not np.isclose(float(row["dose_mg"]), float(label["dose_mg"]), rtol=0, atol=1e-12):
            raise ValueError(f"Prediction/label dose mismatch for {key}")
        if row["route"] != label["route"]:
            raise ValueError(f"Prediction/label route mismatch for {key}")
        joined.append({**row, **label})
    return joined


def statistic_sensitivity(rows: list[dict], seed: int, n_boot: int) -> dict:
    """Describe same-statistic compound subsets without changing the primary gate."""
    by_compound: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_compound[row["candidate_id"]].append(row)
    groups = {kind: [] for kind in (
        "arithmetic_mean", "geometric_mean", "geometric_lsmean", "median"
    )}
    mixed = 0
    for compound_rows in by_compound.values():
        types = {row["cmax_statistic"] for row in compound_rows}
        if len(types) == 1:
            groups[next(iter(types))].extend(compound_rows)
        else:
            mixed += 1
    result = {}
    for kind, subset in groups.items():
        n = len({row["candidate_id"] for row in subset})
        entry = {"n_compounds": n, "n_arms": len(subset), "score": None}
        if n >= 20:
            meta = _compound_errors(subset, "meta_cmax_mg_l")
            ml = _compound_errors(subset, "ml_cmax_mg_l")
            _, _, ratios = _paired_bootstrap(meta, ml, np.random.default_rng(seed), n_boot)
            entry["score"] = {
                "meta_aafe": float(np.exp(meta.mean())),
                "ml_aafe": float(np.exp(ml.mean())),
                "meta_ml_aafe_ratio": float(np.exp((meta - ml).mean())),
                "paired_ratio_95_ci": [float(v) for v in np.percentile(ratios, [2.5, 97.5])],
            }
        result[kind] = entry
    return {"groups": result, "mixed_statistic_compounds": mixed}


def score(rows: list[dict], seed: int, n_boot: int) -> dict:
    primary = [row for row in rows if row.get("primary_eligible") is True]
    if not primary:
        raise ValueError("No primary-eligible rows")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    meta = _compound_errors(primary, "meta_cmax_mg_l")
    ml = _compound_errors(primary, "ml_cmax_mg_l")
    if len(meta) != len(ml):
        raise ValueError("Meta and ML compound sets differ")
    rng = np.random.default_rng(seed)
    meta_boot, ml_boot, ratio_boot = _paired_bootstrap(meta, ml, rng, n_boot)
    meta_ci = tuple(float(v) for v in np.percentile(meta_boot, [2.5, 97.5]))
    ml_ci = tuple(float(v) for v in np.percentile(ml_boot, [2.5, 97.5]))
    ratio_ci = tuple(float(v) for v in np.percentile(ratio_boot, [2.5, 97.5]))

    signed = _compound_signed_errors(primary, "meta_cmax_mg_l")
    symmetric_fold = np.exp(meta)
    ratio = float(np.exp((meta - ml).mean()))
    bias = float(np.exp(signed.mean()))
    pct_2fold = float(100 * np.mean(symmetric_fold <= 2))
    p90_fold = float(np.percentile(symmetric_fold, 90))

    interval_rows = [
        row
        for row in primary
        if row.get("meta_pi90_low_mg_l") is not None
        and row.get("meta_pi90_high_mg_l") is not None
    ]
    interval_coverage = None
    interval_coverage_ci = None
    interval_factor_median = None
    if len(interval_rows) == len(primary):
        by_compound_coverage: dict[str, list[float]] = defaultdict(list)
        by_compound_width: dict[str, list[float]] = defaultdict(list)
        for row in interval_rows:
            lo = float(row["meta_pi90_low_mg_l"])
            hi = float(row["meta_pi90_high_mg_l"])
            obs_value = float(row["observed_cmax_mg_l"])
            if lo <= 0 or hi <= lo:
                raise ValueError(f"Invalid PI bounds in {row['candidate_id']}")
            by_compound_coverage[row["candidate_id"]].append(float(lo <= obs_value <= hi))
            by_compound_width[row["candidate_id"]].append(float(np.sqrt(hi / lo)))
        compound_coverage = np.asarray(
            [np.mean(v) for v in by_compound_coverage.values()], dtype=float
        )
        interval_coverage = float(np.mean(compound_coverage))
        coverage_boot = np.empty(n_boot, dtype=float)
        for start in range(0, n_boot, 5000):
            stop = min(start + 5000, n_boot)
            idx = rng.integers(
                0,
                len(compound_coverage),
                size=(stop - start, len(compound_coverage)),
            )
            coverage_boot[start:stop] = compound_coverage[idx].mean(axis=1)
        interval_coverage_ci = tuple(
            float(v) for v in np.percentile(coverage_boot, [2.5, 97.5])
        )
        interval_factor_median = float(
            np.median([np.mean(v) for v in by_compound_width.values()])
        )

    ratio_limit = 0.85 if len(meta) == 120 else 0.90
    superiority = bool(ratio <= ratio_limit and ratio_ci[1] < 1.0)
    release_gate = bool(
        superiority
        and 0.8 <= bias <= 1.25
        and pct_2fold >= 50.0
        and p90_fold < 6.0
        and interval_coverage is not None
        and 0.85 <= interval_coverage <= 0.975
        and interval_factor_median is not None
        and interval_factor_median <= 13.0
    )
    return {
        "n_compounds": int(len(meta)),
        "n_arms": len(primary),
        "meta_aafe": float(np.exp(meta.mean())),
        "meta_bootstrap_95_ci": meta_ci,
        "ml_aafe": float(np.exp(ml.mean())),
        "ml_bootstrap_95_ci": ml_ci,
        "meta_ml_aafe_ratio": ratio,
        "meta_ml_ratio_limit": ratio_limit,
        "paired_ratio_95_ci": ratio_ci,
        "meta_geometric_bias": bias,
        "meta_pct_within_2fold": pct_2fold,
        "meta_p90_fold": p90_fold,
        "meta_pi90_coverage": interval_coverage,
        "meta_pi90_coverage_95_ci": interval_coverage_ci,
        "meta_pi90_median_multiplicative_factor": interval_factor_median,
        "meta_superiority_gate": superiority,
        "production_release_gate": release_gate,
        "cmax_statistic_sensitivity": statistic_sensitivity(primary, seed, n_boot),
        "seed": seed,
        "n_bootstrap": n_boot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--labels-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    actual_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if actual_sha != args.manifest_sha256:
        raise ValueError(
            f"Manifest hash mismatch: expected {args.manifest_sha256}, got {actual_sha}"
        )
    actual_predictions_sha = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    if actual_predictions_sha != args.predictions_sha256:
        raise ValueError("Predictions SHA256 does not match the committed value")
    actual_labels_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    if actual_labels_sha != args.labels_sha256:
        raise ValueError("Labels SHA256 does not match the custodian value")
    manifest = json.loads(args.manifest.read_text())
    payload = json.loads(args.predictions.read_text())
    labels = json.loads(args.labels.read_text())
    validate_payload(manifest, "external_holdout_v1_manifest.schema.json")
    source_plan = verify_source_plan(args.manifest, manifest)
    validate_source_quotas(manifest)
    validate_payload(payload, "external_holdout_v1_predictions.schema.json")
    validate_payload(labels, "external_holdout_v1_labels.schema.json")
    if label_content_sha256(labels) != source_plan["label_content_sha256"]:
        raise ValueError("Blinded label content differs from the pre-prediction commitment")
    if payload.get("manifest_sha256") != actual_sha:
        raise ValueError("predictions manifest_sha256 is missing or does not match")
    if payload.get("cycle_id") != manifest.get("cycle_id"):
        raise ValueError("predictions cycle_id does not match manifest")
    if labels.get("manifest_sha256") != actual_sha:
        raise ValueError("labels manifest_sha256 is missing or does not match")
    if labels["predictions_sha256"] != actual_predictions_sha:
        raise ValueError("Custodian prediction commitment does not match frozen predictions")
    if labels.get("cycle_id") != manifest.get("cycle_id"):
        raise ValueError("labels cycle_id does not match manifest")
    freeze = manifest["freeze"]
    freeze_echoes = {
        "git_sha": "git_sha",
        "source_tree_sha256": "source_tree_sha256",
        "dependency_lock_sha256": "dependency_lock_sha256",
        "artifact_inventory_sha256": "artifact_inventory_sha256",
        "training_membership_sha256": "training_membership_sha256",
        "feature_schema_sha256": "feature_schema_sha256",
        "solver_settings_sha256": "solver_settings_sha256",
        "container_digest": "container_digest",
    }
    for payload_key, freeze_key in freeze_echoes.items():
        if payload.get(payload_key) != freeze.get(freeze_key):
            raise ValueError(f"Prediction freeze field mismatch: {payload_key}")
    prediction_rows = payload.get("rows")
    if not isinstance(prediction_rows, list):
        raise ValueError("predictions.rows must be a list")
    rows = join_predictions_and_labels(prediction_rows, labels)
    validate_results_against_manifest(
        rows, manifest, source_plan["source_windows"], source_plan["curators"]
    )
    report = score(rows, freeze["random_seed"], 100000)
    report["manifest_sha256"] = actual_sha
    report["label_content_sha256"] = source_plan["label_content_sha256"]
    report["predictions_sha256"] = actual_predictions_sha
    report["labels_sha256"] = actual_labels_sha
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
