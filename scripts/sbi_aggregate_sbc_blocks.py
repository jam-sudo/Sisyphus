#!/usr/bin/env python3
"""Reassemble Slurm SBC blocks and apply the original global gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


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
        "seed", "ks_threshold", "coverage_tol",
    ) if key in first}
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
        if any(not entry.get("ranks") for _, entry in parts):
            row = {
                "n_valid": available,
                "ks_pvalues": [0.0] * 3,
                "coverage": {},
                "coverage_max_deviation": 1.0,
                "pass": False,
                "sampling_error": "At least one calibration block had low ESS or failed",
                "rejection_fallback_cases": sum(
                    entry.get("rejection_fallback_cases", 0) for _, entry in parts
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
            "rejection_fallback_cases": sum(
                entry.get("rejection_fallback_cases", 0) for _, entry in parts
            ),
            "median_correction_ess": float(np.median(ess_values)) if ess_values else None,
            "max_proposals_used": max(proposal_counts, default=0),
        }
        if len(key) == 1:
            row["name"] = key[0]
        else:
            row["population"], row["drug"] = key
        output[key_name].append(row)
    output["pass_count"] = sum(row["pass"] for row in output[key_name])
    output["rejection_fallback_cases"] = sum(
        row.get("rejection_fallback_cases", 0) for row in output[key_name]
    )
    output["total_pairs"] = expected_pairs
    output["gate_passed"] = output["pass_count"] == expected_pairs
    return output


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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
