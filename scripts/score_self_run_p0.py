#!/usr/bin/env python3
"""Score the committed, source-adjudicated P0 pilot exactly once."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from score_external_holdout import (
    _compound_errors,
    _compound_signed_errors,
    _paired_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "e304a1e57275576f3de65b515a9392ab74194529fc68f0924574d1298a44ab3c"
PREDICTIONS_SHA = "c54702d9f7d9b9721b0d01aeaeedcbf55381d0ad8effb744d1612f7ecb3b7674"
ADJUDICATION_SHA = "cb964fc5a30ad140df54a9d175082f154038312c2d8337cbdc87c5c142673812"
MODEL_SHA = "618106b53b0c9ce3c5b8a8fe62c5adf8f02b2308"
UNITS_TO_MG_L = {"ng/mL": 0.001, "μg/mL": 1.0, "μg/L": 0.001, "pg/mL": 0.000001}


def _load(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")
    return json.loads(raw)


def _statistic_type(label: str) -> str:
    value = label.lower()
    if value.startswith(("geometric least", "geometric lsmean")):
        return "geometric_lsmean"
    if value.startswith("geometric mean"):
        return "geometric"
    if value.startswith("median"):
        return "median"
    if value.startswith(("arithmetic", "mean")):
        return "arithmetic"
    raise ValueError(f"Unclassified source Cmax statistic: {label!r}")


def joined_rows(manifest: dict, predictions: dict, adjudication: dict) -> list[dict]:
    if predictions["manifest_sha256"] != MANIFEST_SHA or predictions["model_git_sha"] != MODEL_SHA:
        raise ValueError("Prediction freeze metadata mismatch")
    frozen = {
        (c["candidate_id"], a["arm_id"]): a for c in manifest["candidates"] for a in c["arms"]
    }
    predicted = {}
    for row in predictions["rows"]:
        key = row["candidate_id"], row["arm_id"]
        if key in predicted:
            raise ValueError(f"Duplicate prediction: {key}")
        predicted[key] = row
    if predicted.keys() != frozen.keys():
        raise ValueError("Prediction arms differ from frozen manifest")
    if len(adjudication["candidates"]) != len(manifest["candidates"]):
        raise ValueError("Adjudication candidate count mismatch")

    joined = []
    for i, (candidate, ruling) in enumerate(
        zip(manifest["candidates"], adjudication["candidates"]), 1
    ):
        cid = candidate["candidate_id"]
        if ruling["position"] != i or ruling["candidate_id"] != cid:
            raise ValueError(f"Adjudication order mismatch at {i}")
        arm_ids = {a["arm_id"] for a in candidate["arms"]}
        assigned = (
            ruling["included_arm_ids"] + ruling["excluded_arm_ids"] + ruling["unresolved_arm_ids"]
        )
        if len(assigned) != len(set(assigned)) or set(assigned) != arm_ids:
            raise ValueError(f"Adjudication arm coverage mismatch: {cid}")
        if ruling["disposition"] == "include":
            if not ruling["included_arm_ids"] or ruling["unresolved_arm_ids"]:
                raise ValueError(f"Partial inclusion: {cid}")
        elif ruling["included_arm_ids"]:
            raise ValueError(f"Non-included candidate has included arms: {cid}")
        evidence = {a["arm_id"]: a for a in ruling["included_arm_evidence"]}
        if set(evidence) != set(ruling["included_arm_ids"]):
            raise ValueError(f"Included evidence mismatch: {cid}")
        for arm_id in ruling["included_arm_ids"]:
            key = cid, arm_id
            observed = evidence[arm_id]
            pred = predicted[key]
            if pred["status"] != "ok":
                raise ValueError(f"Failed included prediction: {key}")
            if observed["dose_mg"] != frozen[key]["dose_mg"]:
                raise ValueError(f"Dose mismatch: {key}")
            expected = (
                float(observed["observed_cmax"]) * UNITS_TO_MG_L[observed["observed_cmax_unit"]]
            )
            if not np.isclose(expected, observed["observed_cmax_mg_l"], rtol=0, atol=1e-12):
                raise ValueError(f"Observed unit conversion mismatch: {key}")
            if min(expected, pred["meta_cmax_mg_l"], pred["ml_cmax_mg_l"]) <= 0:
                raise ValueError(f"Non-positive Cmax: {key}")
            joined.append(
                {
                    **pred,
                    "observed_cmax_mg_l": expected,
                    "source_statistic_type": _statistic_type(observed["cmax_statistic"]),
                    "first_pass_verification": observed["first_pass_verification"],
                }
            )
    return joined


def score(rows: list[dict], *, seed: int = 20260923, n_boot: int = 100000) -> dict:
    if not rows or n_boot < 1:
        raise ValueError("Empty cohort or bootstrap")
    meta = _compound_errors(rows, "meta_cmax_mg_l")
    ml = _compound_errors(rows, "ml_cmax_mg_l")
    signed = _compound_signed_errors(rows, "meta_cmax_mg_l")
    rng = np.random.default_rng(seed)
    meta_boot, ml_boot, ratio_boot = _paired_bootstrap(meta, ml, rng, n_boot)

    def ci(values):
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    return {
        "n_compounds": len(meta),
        "n_arms": len(rows),
        "meta_aafe": float(np.exp(meta.mean())),
        "meta_bootstrap_95_ci": ci(meta_boot),
        "ml_aafe": float(np.exp(ml.mean())),
        "ml_bootstrap_95_ci": ci(ml_boot),
        "meta_ml_aafe_ratio": float(np.exp((meta - ml).mean())),
        "paired_ratio_95_ci": ci(ratio_boot),
        "meta_pct_compounds_within_2fold": float(100 * np.mean(np.exp(meta) <= 2)),
        "meta_geometric_bias": float(np.exp(signed.mean())),
        "seed": seed,
        "n_bootstrap": n_boot,
    }


def sensitivities(rows: list[dict]) -> dict:
    by_compound: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_compound[row["candidate_id"]].append(row)
    by_statistic: dict[str, list[dict]] = defaultdict(list)
    mixed_compounds = 0
    strict_verified = []
    for compound_rows in by_compound.values():
        types = {row["source_statistic_type"] for row in compound_rows}
        if len(types) == 1:
            by_statistic[next(iter(types))].extend(compound_rows)
        else:
            mixed_compounds += 1
        if all(row["first_pass_verification"] == "verified" for row in compound_rows):
            strict_verified.extend(compound_rows)
    statistic_groups = {}
    for kind in ("arithmetic", "geometric", "geometric_lsmean", "median"):
        group = by_statistic[kind]
        n = len({row["candidate_id"] for row in group})
        statistic_groups[kind] = {
            "n_compounds": n,
            "n_arms": len(group),
            "score": score(group) if n >= 4 else None,
        }
    return {
        "same_statistic": {
            "rule": (
                "Score only compounds whose every included arm has the same source Cmax "
                "statistic; groups with fewer than four compounds report counts only."
            ),
            "groups": statistic_groups,
            "mixed_statistic_compounds": mixed_compounds,
        },
        "all_scored_arms_first_pass_verified": score(strict_verified),
    }


def main() -> None:
    manifest = _load(ROOT / "data/validation/self_run_p0_candidates.json", MANIFEST_SHA)
    predictions = _load(ROOT / "data/validation/self_run_p0_predictions.json", PREDICTIONS_SHA)
    adjudication = _load(
        ROOT / "data/validation/self_run_p0_source_adjudication.json", ADJUDICATION_SHA
    )
    rows = joined_rows(manifest, predictions, adjudication)
    report = score(rows)
    report["sensitivity_analyses"] = sensitivities(rows)
    report.update(
        {
            "protocol": "docs/validation/self_run_pilot_p0.md",
            "source_adjudication_commit": "d9c4e4c",
            "manifest_sha256": MANIFEST_SHA,
            "predictions_sha256": PREDICTIONS_SHA,
            "source_adjudication_sha256": ADJUDICATION_SHA,
            "interpretation": (
                "AI-assisted diagnostic P0; not independent External Holdout V1 "
                "or a clinical release gate."
            ),
        }
    )
    out = ROOT / "data/validation/self_run_p0_results.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
