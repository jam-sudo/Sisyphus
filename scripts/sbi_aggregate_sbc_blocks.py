#!/usr/bin/env python3
"""Reassemble Slurm SBC blocks and apply the original global gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from sisyphus.sbi.importance import MAX_REJECTION_ATTEMPT_COUNT
from sisyphus.sbi.joint_smc import SMC_MIN_UNIQUE_INITIAL_ANCESTOR_COUNT


def aggregate_blocks(paths: list[Path], flow: str, expected_pairs: int) -> dict:
    """Combine rank and coverage counts without changing the SBC test rules."""
    if not paths:
        raise ValueError("No SBC blocks provided")
    grouped: dict[tuple[str, ...], list[tuple[dict, dict]]] = defaultdict(list)
    first = None
    for path in paths:
        block = json.loads(path.read_text())
        if block.get("sampler") != "corrected":
            raise ValueError(f"Block {path} did not use the corrected sampler")
        if first is None:
            first = block
        else:
            for field in ("n_calibration", "n_posterior", "levels", "ks_threshold",
                          "coverage_tol"):
                if block[field] != first[field]:
                    raise ValueError(f"Block {path} disagrees on {field}")
        entries = block["drugs"] if flow in ("multi_drug", "multi_observation") \
            else block["pairs"]
        for entry in entries:
            key = (entry["name"],) if "name" in entry else (
                entry["population"], entry["drug"]
            )
            grouped[key].append((block, entry))
    if len(grouped) != expected_pairs:
        raise ValueError(f"Found {len(grouped)} pairs, expected {expected_pairs}")
    assert first is not None
    output = {key: first[key] for key in (
        "posterior", "validation_set", "n_calibration", "n_posterior", "levels",
        "seed", "ks_threshold", "coverage_tol", "populations",
    ) if key in first}
    if "validation_set" in output:
        output["validation_set"] = "data/sbi/validation_drug_set.json"
    output["sampler"] = "corrected"
    key_name = "drugs" if flow in ("multi_drug", "multi_observation") else "pairs"
    output[key_name] = []
    output["flow"] = flow
    for key, parts in sorted(grouped.items()):
        available = max((entry.get("n_available", 0) for _, entry in parts), default=0)
        intervals = sorted((block["case_start"], min(block["case_end"], available))
                           for block, _ in parts)
        cursor = 0
        for start, end in intervals:
            if start != cursor or end <= start:
                raise ValueError(f"Missing or overlapping block for {key}: {intervals}")
            cursor = end
        if cursor != available or available == 0:
            raise ValueError(f"Incomplete SBC blocks for {key}: {cursor}/{available}")
        low_ancestor_parts = [
            entry for _, entry in parts
            if any(diagnostics.get("unique_initial_ancestors", 0)
                   < SMC_MIN_UNIQUE_INITIAL_ANCESTOR_COUNT
                   for diagnostics in entry.get("joint_smc_diagnostics", []))
        ]
        if any(not entry.get("ranks") for _, entry in parts) or low_ancestor_parts:
            row = {
                "n_valid": available,
                "ks_pvalues": [],
                "coverage": {},
                "coverage_max_deviation": None,
                "pass": None,
                "sampling_error": (
                    "Joint SMC lineage below floor" if low_ancestor_parts else
                    "At least one calibration block had low ESS or failed"
                ),
                "sampling_errors": [entry["sampling_error"] for _, entry in parts
                                    if entry.get("sampling_error")]
                + (["Joint SMC lineage below floor"] if low_ancestor_parts else []),
                "sampling_status": (
                    "joint_smc_low_ancestor" if low_ancestor_parts else
                    "uncomputable_by_exact_fallback"
                    if any("Exact rejection yielded" in entry.get("sampling_error", "")
                           for _, entry in parts) else "incomplete"
                ),
                "sampling_error_blocks": sum(
                    bool(entry.get("sampling_error") or entry in low_ancestor_parts)
                    for _, entry in parts
                ),
                "rejection_fallback_cases": sum(
                    entry.get("rejection_fallback_cases", 0) for _, entry in parts
                ),
                "forced_exact_rejection_cases": sum(
                    entry.get("forced_exact_rejection_cases", 0) for _, entry in parts
                ),
                "joint_smc_fallback_cases": sum(
                    entry.get("joint_smc_fallback_cases", 0) for _, entry in parts
                ),
                "joint_smc_diagnostics": [
                    diagnostics for _, entry in parts
                    for diagnostics in entry.get("joint_smc_diagnostics", [])
                ],
                "rejection_attempt_counts": [
                    count for _, entry in parts
                    for count in entry.get("rejection_attempt_counts", [])
                ],
                "max_rejection_attempts": max(
                    (entry.get("max_rejection_attempts", MAX_REJECTION_ATTEMPT_COUNT)
                     for _, entry in parts),
                    default=0,
                ),
            }
            if len(key) == 1:
                row["name"] = key[0]
            else:
                row["population"], row["drug"] = key
            output[key_name].append(row)
            continue
        ess_values = [ess for _, entry in parts for ess in entry.get("correction_ess", [])]
        proposal_counts = [count for _, entry in parts
                           for count in entry.get("proposal_counts", [])]
        ranks = np.concatenate([np.asarray(entry["ranks"]) for _, entry in
                                sorted(parts, key=lambda part: part[0]["case_start"])])
        if len(ranks) != available:
            raise ValueError(f"Rank count mismatch for {key}")
        pvalues = [float(stats.kstest(ranks[:, d] / first["n_posterior"], "uniform").pvalue)
                   for d in range(3)]
        coverage = {}
        max_deviation = 0.0
        for level in first["levels"]:
            counts = np.sum([entry["coverage_counts"][str(level)] for _, entry in parts],
                            axis=0)
            empirical = counts / available
            if flow == "continuous":
                deviation = abs(float(empirical.mean()) - level / 100)
            else:
                deviation = float(np.max(np.abs(empirical - level / 100)))
            coverage[str(level)] = empirical.tolist()
            max_deviation = max(max_deviation, deviation)
        passed = all(p > first["ks_threshold"] for p in pvalues) \
            and max_deviation <= first["coverage_tol"]
        row = {
            "n_valid": available,
            "ks_pvalues": pvalues,
            "coverage": coverage,
            "coverage_max_deviation": max_deviation,
            "pass": passed,
            "sampling_error_blocks": 0,
            "rejection_fallback_cases": sum(
                entry.get("rejection_fallback_cases", 0) for _, entry in parts
            ),
            "forced_exact_rejection_cases": sum(
                entry.get("forced_exact_rejection_cases", 0) for _, entry in parts
            ),
            "joint_smc_fallback_cases": sum(
                entry.get("joint_smc_fallback_cases", 0) for _, entry in parts
            ),
            "joint_smc_diagnostics": [
                diagnostics for _, entry in parts
                for diagnostics in entry.get("joint_smc_diagnostics", [])
            ],
            "rejection_attempt_counts": [count for _, entry in parts
                                         for count in entry.get("rejection_attempt_counts", [])],
            "max_rejection_attempts": max(
                (entry.get("max_rejection_attempts", MAX_REJECTION_ATTEMPT_COUNT)
                 for _, entry in parts),
                default=0,
            ),
            "median_correction_ess": float(np.median(ess_values)) if ess_values else None,
            "max_proposals_used": max(proposal_counts, default=0),
        }
        if len(key) == 1:
            row["name"] = key[0]
        else:
            row["population"], row["drug"] = key
        if row["joint_smc_fallback_cases"]:
            row["sampling_status"] = "joint_smc_fallback"
        output[key_name].append(row)
    output["pass_count"] = sum(row["pass"] is True for row in output[key_name])
    output["incomplete_count"] = sum(row["pass"] is None for row in output[key_name])
    output["failed_count"] = sum(row["pass"] is False for row in output[key_name])
    output["rejection_fallback_cases"] = sum(
        row.get("rejection_fallback_cases", 0) for row in output[key_name]
    )
    output["forced_exact_rejection_cases"] = sum(
        row.get("forced_exact_rejection_cases", 0) for row in output[key_name]
    )
    output["joint_smc_fallback_cases"] = sum(
        row.get("joint_smc_fallback_cases", 0) for row in output[key_name]
    )
    output["sampling_error_blocks"] = sum(
        row["sampling_error_blocks"] for row in output[key_name]
    )
    output["total_pairs"] = expected_pairs
    output["gate_passed"] = (
        False if output["failed_count"] else
        None if output["incomplete_count"] else True
    )
    return output


def _holm_rejection_count(pvalues: list[float], alpha: float) -> int:
    """Count sequential Holm-Bonferroni rejections in one flow family."""
    ordered = sorted(pvalues)
    count = 0
    for index, pvalue in enumerate(ordered):
        if pvalue > alpha / (len(ordered) - index):
            break
        count += 1
    return count


def main() -> None:
    """Aggregate a completed per-flow block directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=("multi_drug", "multi_observation",
                                           "hierarchical", "continuous"), required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_blocks(sorted(args.blocks.glob("*.json")), args.flow,
                              args.expected_pairs)
    null_path = Path(__file__).resolve().parent.parent / "data/validation/sbi_global_gate_null.json"
    null_result = json.loads(null_path.read_text())
    result["calibrated_null_global_pass_probability"] = null_result[args.flow][
        "pass_probability"
    ]
    result["calibrated_null_ci_95"] = null_result[args.flow]["ci_95"]
    result["calibrated_null_assumptions"] = null_result["assumptions"]
    exact_null_path = (Path(__file__).resolve().parent.parent
                       / "data/validation/sbi_global_gate_exact_reference_null.json")
    exact_null = json.loads(exact_null_path.read_text())
    result["exact_reference_null_global_pass_probability"] = exact_null[args.flow][
        "pass_probability"
    ]
    result["exact_reference_null_ci_95"] = exact_null[args.flow]["ci_95"]
    result["exact_reference_null_assumptions"] = exact_null["assumptions"]
    policy_path = (Path(__file__).resolve().parent.parent
                   / "data/validation/sbi_global_gate_policy.json")
    policy = json.loads(policy_path.read_text())
    entries = result["drugs"] if args.flow in ("multi_drug", "multi_observation") \
        else result["pairs"]
    result["uncomputable_pairs"] = [
        entry["name"] if "name" in entry else f"{entry['population']}/{entry['drug']}"
        for entry in entries if entry["pass"] is None
    ]
    evaluable = [entry for entry in entries if entry["pass"] is not None]
    pvalues = [pvalue for entry in evaluable for pvalue in entry["ks_pvalues"]]
    missing_ks = 3 * (len(entries) - len(evaluable))
    alpha = policy["holm_gate"]["family_wise_alpha"]
    coverage_ok = all(
        entry["coverage_max_deviation"] <= policy["holm_gate"]["coverage_tolerance_fraction"]
        for entry in evaluable
    )
    rejected = _holm_rejection_count(pvalues + [1.0] * missing_ks, alpha)
    result["holm_global_gate"] = {
        "passed": (False if rejected or not coverage_ok else
                   None if missing_ks else True),
        "ks_rejections": rejected,
        "coverage_passed": coverage_ok,
        "family_wise_alpha": alpha,
        "predeclared_at_utc": policy["predeclared_at_utc"],
        "definition": policy["holm_gate"]["definition"],
        "calibrated_null_pass_probability": null_result[args.flow]["holm"][
            "pass_probability"
        ],
        "calibrated_null_ci_95": null_result[args.flow]["holm"]["ci_95"],
        "calibrated_null_assumptions": null_result["assumptions"],
    }
    size_pvalues = pvalues[:]
    size_owners = [index for index, entry in enumerate(entries) if entry["pass"] is not None
                   for _ in entry["ks_pvalues"]]
    coverage_test_count = 0
    for row_index, entry in enumerate(entries):
        if entry["pass"] is None:
            continue
        for level in result["levels"]:
            counts = np.rint(
                np.asarray(entry["coverage"][str(level)]) * entry["n_valid"]
            ).astype(int)
            if args.flow == "continuous":
                counts = np.array([int(counts.sum())])
                n_trials = 3 * entry["n_valid"]
            else:
                n_trials = entry["n_valid"]
            coverage_pvalues = [
                float(stats.binomtest(int(count), n_trials, level / 100).pvalue)
                for count in counts
            ]
            size_pvalues.extend(coverage_pvalues)
            size_owners.extend([row_index] * len(coverage_pvalues))
            coverage_test_count += len(counts)
    missing_coverage = (len(entries) - len(evaluable)) * len(result["levels"]) * (
        1 if args.flow == "continuous" else 3
    )
    size_pvalues.extend([1.0] * (missing_ks + missing_coverage))
    size_owners.extend([-1] * (missing_ks + missing_coverage))
    coverage_test_count += missing_coverage
    size_alpha = policy["size_calibrated_gate"]["family_wise_alpha"]
    size_rejected = _holm_rejection_count(size_pvalues, size_alpha)
    for entry in entries:
        entry["size_calibrated_rejections"] = 0
    for test_index in sorted(range(len(size_pvalues)), key=size_pvalues.__getitem__)[
        :size_rejected
    ]:
        if size_owners[test_index] >= 0:
            entries[size_owners[test_index]]["size_calibrated_rejections"] += 1
    result["size_calibrated_global_gate"] = {
        "passed": None if missing_ks else size_rejected == 0,
        "status": ("INCOMPLETE" if missing_ks else
                   "FAIL" if size_rejected else "PASS"),
        "uncomputable_pairs": result["uncomputable_pairs"],
        "rejections": size_rejected,
        "ks_test_count": len(pvalues) + missing_ks,
        "coverage_test_count": coverage_test_count,
        "family_wise_alpha": size_alpha,
        "predeclared_at_utc": policy["size_calibrated_gate"]["predeclared_at_utc"],
        "definition": policy["size_calibrated_gate"]["definition"],
        "calibrated_null_pass_probability": null_result[args.flow]["size_calibrated"][
            "pass_probability"
        ],
        "calibrated_null_ci_95": null_result[args.flow]["size_calibrated"]["ci_95"],
        "calibrated_null_assumptions": null_result["assumptions"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
