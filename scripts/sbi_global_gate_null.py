#!/usr/bin/env python3
"""Simulate global SBC-gate pass rates for exact posterior ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, kstwo

# Source: the unchanged SBC scripts' default posterior draw count.
POSTERIOR_DRAW_COUNT = 500
# Source: the unchanged SBC scripts' default coverage levels.
COVERAGE_LEVELS_PERCENT = (50, 80, 90, 95)
# Source: the unchanged SBC scripts' KS threshold.
KS_THRESHOLD_P = 0.01
# Source: the unchanged SBC scripts' coverage tolerance.
COVERAGE_TOLERANCE_FRACTION = 0.10


def gate_probability(
    n_pairs: int,
    n_calibration: int,
    *,
    continuous_coverage: bool = False,
    repetitions: int = 10000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Draw exact discrete-uniform SBC ranks and apply the existing global gate.

    Rank draws are independent across parameters and pairs. This is an exact
    calibrated null for marginal ranks, with the four coverage levels sharing
    each rank just as in a real SBC run.
    """
    if n_pairs < 1 or n_calibration < 1 or repetitions < 1:
        raise ValueError("Pair, calibration, and repetition counts must be positive")
    rng = np.random.default_rng(seed)
    passed = 0
    for _ in range(repetitions):
        ranks = rng.integers(0, POSTERIOR_DRAW_COUNT + 1, size=(n_pairs, n_calibration, 3))
        sorted_ranks = np.sort(ranks / POSTERIOR_DRAW_COUNT, axis=1)
        grid = np.arange(1, n_calibration + 1)[None, :, None] / n_calibration
        d_stat = np.maximum(
            grid - sorted_ranks,
            sorted_ranks - (grid - 1 / n_calibration),
        ).max(axis=1)
        if not np.all(kstwo.sf(d_stat, n_calibration) > KS_THRESHOLD_P):
            continue
        coverage_ok = True
        for level in COVERAGE_LEVELS_PERCENT:
            tail = (1 - level / 100) / 2
            inside = ((ranks >= tail * POSTERIOR_DRAW_COUNT)
                      & (ranks <= (1 - tail) * POSTERIOR_DRAW_COUNT))
            empirical = inside.mean(axis=1)
            if continuous_coverage:
                empirical = empirical.mean(axis=1)
            if np.any(np.abs(empirical - level / 100) > COVERAGE_TOLERANCE_FRACTION):
                coverage_ok = False
                break
        passed += coverage_ok
    ci = binomtest(passed, repetitions).proportion_ci(confidence_level=0.95)
    return {
        "pairs": n_pairs,
        "simultaneous_ks_tests": 3 * n_pairs,
        "n_calibration": n_calibration,
        "repetitions": repetitions,
        "passed": passed,
        "pass_probability": passed / repetitions,
        "ci_95": [ci.low, ci.high],
    }


def main() -> None:
    """Write null probabilities for the three existing SBC gate shapes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {
        "multi_drug": gate_probability(13, 100, repetitions=args.repetitions, seed=args.seed),
        "hierarchical": gate_probability(26, 100, repetitions=args.repetitions, seed=args.seed + 1),
        "continuous": gate_probability(
            52, 50, continuous_coverage=True,
            repetitions=args.repetitions, seed=args.seed + 2,
        ),
        "assumptions": "Exact uniform marginal ranks; independent parameters and pairs",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
