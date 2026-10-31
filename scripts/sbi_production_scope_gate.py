#!/usr/bin/env python3
"""Apply the predeclared production-scope SBC gate to one flow artifact."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sbi_aggregate_sbc_blocks import _holm_rejection_count  # noqa: E402
from scripts.sbi_global_gate_null import production_scope_probability  # noqa: E402

# Source: data/validation/sbi_production_scope_gate_policy.json.
DEFAULT_NULL_REPETITION_COUNT = 10000


def production_scope_gate(
    aggregate: dict,
    flow: str,
    *,
    repetitions: int = DEFAULT_NULL_REPETITION_COUNT,
) -> dict:
    """Test exactly the pairs eligible for production SBI routing."""
    if flow not in ("multi_drug", "multi_observation", "hierarchical", "continuous"):
        raise ValueError(f"Unknown flow: {flow}")
    policy_path = (Path(__file__).resolve().parents[1]
                   / "data/validation/sbi_production_scope_gate_policy.json")
    policy = json.loads(policy_path.read_text())
    rows = aggregate["drugs"] if flow in ("multi_drug", "multi_observation") \
        else aggregate["pairs"]

    def label(row: dict) -> str:
        return row["name"] if "name" in row else f"{row['population']}/{row['drug']}"

    behavior_eligible = [row["pass"] is not None
                         and not row.get("rejection_fallback_cases", 0)
                         and not row.get("joint_smc_fallback_cases", 0)
                         and not row.get("sampling_error_blocks", 0) for row in rows]
    selected = [row for row, eligible in zip(rows, behavior_eligible, strict=True)
                if eligible and row.get("size_calibrated_rejections") == 0]
    pvalues = [p for row in selected for p in row["ks_pvalues"]]
    for row in selected:
        n_trials = row["n_valid"]
        for level in aggregate["levels"]:
            counts = np.rint(np.asarray(row["coverage"][str(level)]) * n_trials).astype(int)
            if flow == "continuous":
                counts = np.array([int(counts.sum())])
                coverage_trials = 3 * n_trials
            else:
                coverage_trials = n_trials
            pvalues.extend(
                float(binomtest(int(count), coverage_trials, level / 100).pvalue)
                for count in counts
            )
    alpha = policy["family_wise_alpha"]
    rejected = _holm_rejection_count(pvalues, alpha) if pvalues else 0
    null = production_scope_probability(
        behavior_eligible, [row["pass"] is None for row in rows],
        aggregate["n_calibration"], continuous_coverage=flow == "continuous",
        repetitions=repetitions, seed=42,
    )
    return {
        "status": ("INCOMPLETE" if not selected else "FAIL" if rejected else "PASS"),
        "passed": None if not selected else rejected == 0,
        "selected_count": len(selected),
        "selected_pairs": [label(row) for row in selected],
        "excluded_pairs": [label(row) for row in rows if row not in selected],
        "test_count": len(pvalues),
        "rejections": rejected,
        "family_wise_alpha": alpha,
        "predeclared_at_utc": policy["predeclared_at_utc"],
        "amended_at_utc": policy["amended_at_utc"],
        "calibrated_null_pass_probability": null["pass_probability"],
        "calibrated_null_ci_95": null["ci_95"],
        "calibrated_null_nonempty_probability": null["nonempty_probability"],
        "calibrated_null_repetitions": repetitions,
        "calibrated_null_assumptions": aggregate["calibrated_null_assumptions"],
    }


def main() -> None:
    """Write the production-scope result into an aggregated SBC artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True,
                        choices=("multi_drug", "multi_observation", "hierarchical",
                                 "continuous"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_NULL_REPETITION_COUNT)
    args = parser.parse_args()
    aggregate = json.loads(args.artifact.read_text())
    aggregate["production_scope_size_calibrated_gate"] = production_scope_gate(
        aggregate, args.flow, repetitions=args.repetitions,
    )
    args.artifact.write_text(json.dumps(aggregate, indent=2))
    logging.info("Wrote production-scope gate to %s", args.artifact)


if __name__ == "__main__":
    main()
