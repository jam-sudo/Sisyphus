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
# Source: predeclared size-calibrated policy in data/validation/sbi_global_gate_policy.json.
SIZE_CALIBRATED_FWER_ALPHA_P = 0.01
# Source: data/validation/sbi_global_gate_exact_reference_policy.json.
EXACT_REFERENCE_RANK_BATCH_COUNT = 256
# Source: data/validation/sbi_global_gate_exact_reference_policy.json.
EXACT_REFERENCE_RANK_COUNT = 20000


def _exact_reference_ranks(
    draws: np.ndarray, count: int, rng: np.random.Generator,
) -> np.ndarray:
    """Use exchangeable exact posterior draws to retain parameter rank dependence."""
    if draws.ndim != 2 or draws.shape[1] != 3 or len(draws) < 2 or count < 1:
        raise ValueError("Exact reference draws must have shape (N>=2, 3)")
    ranks = np.empty((count, 3), dtype=np.int16)
    for start in range(0, count, EXACT_REFERENCE_RANK_BATCH_COUNT):
        end = min(start + EXACT_REFERENCE_RANK_BATCH_COUNT, count)
        values = draws[rng.integers(
            0, len(draws), size=(end - start, POSTERIOR_DRAW_COUNT + 1),
        )]
        tie_breakers = rng.random(values.shape)
        truth = values[:, :1, :]
        ranks[start:end] = np.sum(
            (values[:, 1:, :] < truth)
            | ((values[:, 1:, :] == truth)
               & (tie_breakers[:, 1:, :] < tie_breakers[:, :1, :])),
            axis=1,
        )
    return ranks


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
    rank_pool: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Draw exact discrete-uniform SBC ranks and apply the existing global gate.

    By posterior exchangeability, the truth's rank among 500 exact posterior
    draws is discrete-uniform on 0..500. Rank draws here are independent across
    parameters and pairs; the four coverage levels share each rank.
    """
    if n_pairs < 1 or n_calibration < 1 or repetitions < 1:
        raise ValueError("Pair, calibration, and repetition counts must be positive")
    if rank_pool is not None and (
        rank_pool.ndim != 2 or rank_pool.shape[1] != 3 or len(rank_pool) < 1
        or np.any((rank_pool < 0) | (rank_pool > POSTERIOR_DRAW_COUNT))
    ):
        raise ValueError("Rank pool must have shape (N>=1, 3) and ranks in 0..500")
    rng = np.random.default_rng(seed)
    size_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    coverage_draws = n_calibration * (3 if continuous_coverage else 1)
    binomial_pvalues = {
        level: np.array([
            binomtest(count, coverage_draws, level / 100).pvalue
            for count in range(coverage_draws + 1)
        ]) for level in COVERAGE_LEVELS_PERCENT
    }
    passed = 0
    holm_passed = 0
    size_passed = 0
    size_test_count = n_pairs * (7 if continuous_coverage else 15)
    for _ in range(repetitions):
        ranks = (rng.integers(0, POSTERIOR_DRAW_COUNT + 1,
                              size=(n_pairs, n_calibration, 3))
                 if rank_pool is None else
                 rank_pool[rng.integers(0, len(rank_pool), size=(n_pairs, n_calibration))])
        sorted_ranks = np.sort(ranks / POSTERIOR_DRAW_COUNT, axis=1)
        grid = np.arange(1, n_calibration + 1)[None, :, None] / n_calibration
        d_stat = np.maximum(
            grid - sorted_ranks,
            sorted_ranks - (grid - 1 / n_calibration),
        ).max(axis=1)
        pvalues = kstwo.sf(d_stat, n_calibration)
        size_pvalues = [pvalues.ravel()]
        for level in COVERAGE_LEVELS_PERCENT:
            inside = _inside_quantile_interval(ranks, level, size_rng)
            counts = (inside.sum(axis=(1, 2)) if continuous_coverage
                      else inside.sum(axis=1))
            size_pvalues.append(binomial_pvalues[level][counts].ravel())
        size_passed += int(
            min(float(np.min(values)) for values in size_pvalues)
            > SIZE_CALIBRATED_FWER_ALPHA_P / size_test_count
        )
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
    size_ci = binomtest(size_passed, repetitions).proportion_ci(confidence_level=0.95)
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
        "size_calibrated": {
            "simultaneous_tests": size_test_count,
            "passed": size_passed,
            "pass_probability": size_passed / repetitions,
            "ci_95": [size_ci.low, size_ci.high],
        },
    }


def production_scope_probability(
    behavior_eligible: list[bool],
    incomplete: list[bool],
    n_calibration: int,
    *,
    continuous_coverage: bool = False,
    repetitions: int = 10000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Simulate exact ranks through flow-wide route screening and the routed gate."""
    n_pairs = len(behavior_eligible)
    if n_pairs < 1 or len(incomplete) != n_pairs or n_calibration < 1 or repetitions < 1:
        raise ValueError("Masks, calibration count, and repetitions must be valid")
    eligible = np.asarray(behavior_eligible, dtype=bool)
    missing = np.asarray(incomplete, dtype=bool)
    rng = np.random.default_rng(seed)
    coverage_draws = n_calibration * (3 if continuous_coverage else 1)
    binomial_pvalues = {
        level: np.array([
            binomtest(count, coverage_draws, level / 100).pvalue
            for count in range(coverage_draws + 1)
        ]) for level in COVERAGE_LEVELS_PERCENT
    }
    tests_per_pair = 7 if continuous_coverage else 15
    passed = nonempty = 0
    for _ in range(repetitions):
        ranks = rng.integers(0, POSTERIOR_DRAW_COUNT + 1,
                             size=(n_pairs, n_calibration, 3))
        sorted_ranks = np.sort(ranks / POSTERIOR_DRAW_COUNT, axis=1)
        grid = np.arange(1, n_calibration + 1)[None, :, None] / n_calibration
        d_stat = np.maximum(grid - sorted_ranks,
                            sorted_ranks - (grid - 1 / n_calibration)).max(axis=1)
        p_by_pair = [kstwo.sf(d_stat, n_calibration)]
        for level in COVERAGE_LEVELS_PERCENT:
            inside = _inside_quantile_interval(ranks, level, rng)
            counts = (inside.sum(axis=(1, 2)) if continuous_coverage
                      else inside.sum(axis=1))
            coverage_p = binomial_pvalues[level][counts]
            p_by_pair.append(coverage_p[:, None] if continuous_coverage else coverage_p)
        pvalues = np.concatenate(p_by_pair, axis=1)
        pvalues[missing] = 1.0
        flat = pvalues.ravel()
        rejected_rows = np.zeros(n_pairs, dtype=bool)
        for index, test_index in enumerate(np.argsort(flat)):
            if flat[test_index] > SIZE_CALIBRATED_FWER_ALPHA_P / (flat.size - index):
                break
            rejected_rows[test_index // tests_per_pair] = True
        selected = eligible & ~rejected_rows
        if np.any(selected):
            nonempty += 1
            scoped = pvalues[selected].ravel()
            passed += int(np.min(scoped) > SIZE_CALIBRATED_FWER_ALPHA_P / scoped.size)
    ci = binomtest(passed, repetitions).proportion_ci(confidence_level=0.95)
    return {
        "pass_probability": passed / repetitions,
        "passed": passed,
        "ci_95": [ci.low, ci.high],
        "nonempty_probability": nonempty / repetitions,
        "repetitions": repetitions,
        "tests_per_pair": tests_per_pair,
    }


def main() -> None:
    """Write null probabilities for the three existing SBC gate shapes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-reference", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank_pool = None
    if args.exact_reference:
        reference_rng = np.random.default_rng(args.seed)
        rank_pool = np.concatenate([
            _exact_reference_ranks(
                np.load(path)["theta"], EXACT_REFERENCE_RANK_COUNT, reference_rng,
            ) for path in args.exact_reference
        ])
    single_gate = gate_probability(
        13, 100, repetitions=args.repetitions, seed=args.seed, rank_pool=rank_pool,
    )
    results = {
        "multi_drug": single_gate,
        "multi_observation": single_gate.copy(),
        "hierarchical": gate_probability(
            26, 100, repetitions=args.repetitions, seed=args.seed + 1, rank_pool=rank_pool,
        ),
        "continuous": gate_probability(
            52, 50, continuous_coverage=True,
            repetitions=args.repetitions, seed=args.seed + 2, rank_pool=rank_pool,
        ),
        "assumptions": (
            "Exact exchangeable posterior draws from the supplied synthetic contexts; "
            "parameter dependence retained within each rank triple; pairs and calibration "
            "cases sampled independently; uniform location within posterior quantile gaps"
            if rank_pool is not None else
            "Exact uniform marginal ranks; independent parameters and pairs; "
            "uniform location within posterior quantile gaps"
        ),
        "method": ("Exact rejection posterior reference rank resampling"
                   if rank_pool is not None else
                   "Exact calibrated-rank sampling by posterior exchangeability"),
        "exact_reference_files": [path.name for path in args.exact_reference],
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
