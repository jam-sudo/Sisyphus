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


def _inside_quantile_interval(
    ranks: np.ndarray,
    level: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply NumPy's interpolated posterior quantile rule to exact ranks."""
    tail = (1 - level / 100) / 2
    lower_pos = (POSTERIOR_DRAW_COUNT - 1) * tail
    upper_pos = (POSTERIOR_DRAW_COUNT - 1) * (1 - tail)
    lower_edge = int(np.floor(lower_pos)) + 1
    upper_edge = int(np.floor(upper_pos)) + 1
    above_lower = ((ranks > lower_edge)
                   | ((ranks == lower_edge) & (rng.random(ranks.shape)
                                               >= lower_pos % 1)))
    below_upper = ((ranks < upper_edge)
                   | ((ranks == upper_edge) & (rng.random(ranks.shape)
                                               <= upper_pos % 1)))
    return above_lower & below_upper


def gate_probability(
    n_pairs: int,
    n_calibration: int,
    *,
    continuous_coverage: bool = False,
    repetitions: int = 10000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Draw exact discrete-uniform SBC ranks and apply the existing global gate.

    By posterior exchangeability, the truth's rank among 500 exact posterior
    draws is discrete-uniform on 0..500. Rank draws here are independent across
    parameters and pairs; the four coverage levels share each rank.
    """
    if n_pairs < 1 or n_calibration < 1 or repetitions < 1:
        raise ValueError("Pair, calibration, and repetition counts must be positive")
    rng = np.random.default_rng(seed)
    passed = 0
    holm_passed = 0
    for _ in range(repetitions):
        ranks = rng.integers(0, POSTERIOR_DRAW_COUNT + 1, size=(n_pairs, n_calibration, 3))
        sorted_ranks = np.sort(ranks / POSTERIOR_DRAW_COUNT, axis=1)
        grid = np.arange(1, n_calibration + 1)[None, :, None] / n_calibration
        d_stat = np.maximum(
            grid - sorted_ranks,
            sorted_ranks - (grid - 1 / n_calibration),
        ).max(axis=1)
        pvalues = kstwo.sf(d_stat, n_calibration)
        original_ks_ok = bool(np.all(pvalues > KS_THRESHOLD_P))
        # Holm makes no rejection exactly when the smallest p exceeds alpha / m.
        if np.min(pvalues) <= KS_THRESHOLD_P / pvalues.size:
            continue
        coverage_ok = True
        for level in COVERAGE_LEVELS_PERCENT:
            inside = _inside_quantile_interval(ranks, level, rng)
            empirical = inside.mean(axis=1)
            if continuous_coverage:
                empirical = empirical.mean(axis=1)
            if np.any(np.abs(empirical - level / 100) > COVERAGE_TOLERANCE_FRACTION):
                coverage_ok = False
                break
        if coverage_ok:
            holm_passed += 1
            passed += original_ks_ok
    ci = binomtest(passed, repetitions).proportion_ci(confidence_level=0.95)
    holm_ci = binomtest(holm_passed, repetitions).proportion_ci(confidence_level=0.95)
    return {
        "pairs": n_pairs,
        "simultaneous_ks_tests": 3 * n_pairs,
        "n_calibration": n_calibration,
        "repetitions": repetitions,
        "passed": passed,
        "pass_probability": passed / repetitions,
        "ci_95": [ci.low, ci.high],
        "holm": {
            "passed": holm_passed,
            "pass_probability": holm_passed / repetitions,
            "ci_95": [holm_ci.low, holm_ci.high],
        },
    }


def main() -> None:
    """Write null probabilities for the three existing SBC gate shapes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    single_gate = gate_probability(13, 100, repetitions=args.repetitions, seed=args.seed)
    results = {
        "multi_drug": single_gate,
        "multi_observation": single_gate.copy(),
        "hierarchical": gate_probability(26, 100, repetitions=args.repetitions, seed=args.seed + 1),
        "continuous": gate_probability(
            52, 50, continuous_coverage=True,
            repetitions=args.repetitions, seed=args.seed + 2,
        ),
        "assumptions": (
            "Exact uniform marginal ranks; independent parameters and pairs; "
            "uniform location within posterior quantile gaps"
        ),
        "method": "Exact calibrated-rank sampling by posterior exchangeability",
        "gate": {
            "posterior_draws": POSTERIOR_DRAW_COUNT,
            "coverage_levels_percent": COVERAGE_LEVELS_PERCENT,
            "ks_threshold_p": KS_THRESHOLD_P,
            "coverage_tolerance_fraction": COVERAGE_TOLERANCE_FRACTION,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
