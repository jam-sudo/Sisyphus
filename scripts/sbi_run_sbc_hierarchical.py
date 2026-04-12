#!/usr/bin/env python3
"""Per-(population, drug) SBC gate for the hierarchical conditional amortizer.

Runs simulation-based calibration independently for each (population, drug)
pair in the validation set. For each pair:

    1. Build a fresh EngineSimulator with the population's physiology YAML.
    2. Extract population-independent 12D drug features (adult graph).
    3. Construct the population one-hot vector.
    4. Draw n_calibration thetas from the prior, simulate x_obs per draw.
    5. Pack observation = [log10_cmax, drug_features_std, pop_onehot].
    6. Sample posterior, compute rank + coverage.
    7. KS uniformity test + coverage deviation.

Gate (per pair):
    - All KS p-values > 0.01
    - All coverage levels within 10pp of nominal

Usage::

    python scripts/sbi_run_sbc_hierarchical.py \\
        --posterior models/sbi/hierarchical_mini_nsf.pt \\
        --validation-set data/sbi/validation_drug_set.json \\
        --n-calibration 100 --n-posterior 500 \\
        --out data/validation/sbi_sbc_hierarchical_mini.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_sbc_hier")

from sisyphus.sbi.amortizer import load_result  # noqa: E402
from sisyphus.sbi.multi_drug import (  # noqa: E402
    DrugSpec,
    extract_drug_features,
    load_drug_specs_from_json,
    load_populations,
    population_onehot,
)
from sisyphus.sbi.priors import build_box_prior  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator  # noqa: E402


def _coverage_and_ranks(
    thetas_true: np.ndarray,
    posterior,
    xs_packed: np.ndarray,
    levels: tuple[int, ...],
    n_posterior: int = 500,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Per-dim ranks and coverage counts for one (pop, drug)."""
    import torch

    n_cal, d_theta = thetas_true.shape
    ranks = np.zeros((n_cal, d_theta), dtype=np.int64)
    cov = {level: np.zeros(d_theta, dtype=np.int64) for level in levels}
    for i in range(n_cal):
        x_t = torch.as_tensor(xs_packed[i:i + 1], dtype=torch.float32)
        samples = posterior.sample((n_posterior,), x=x_t, show_progress_bars=False)
        s = samples.detach().cpu().numpy()
        for d in range(d_theta):
            ranks[i, d] = int(np.sum(s[:, d] < thetas_true[i, d]))
        for level in levels:
            lo_q = (100 - level) / 2 / 100.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(s, lo_q, axis=0)
            hi = np.quantile(s, hi_q, axis=0)
            inside = (thetas_true[i] >= lo) & (thetas_true[i] <= hi)
            cov[level] += inside.astype(np.int64)
    return ranks, cov


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--validation-set", type=Path,
                   default=ROOT / "data" / "sbi" / "validation_drug_set.json")
    p.add_argument("--n-calibration", type=int, default=100)
    p.add_argument("--n-posterior", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", type=str, default="50,80,90,95")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ks-threshold", type=float, default=0.01)
    p.add_argument("--coverage-tol", type=float, default=0.10)
    args = p.parse_args()

    levels = tuple(int(x) for x in args.levels.split(","))

    log.info("Loading posterior: %s", args.posterior)
    result = load_result(args.posterior)
    import torch
    aux_path = args.posterior.with_suffix(".aux.pt")
    if aux_path.exists():
        aux = torch.load(aux_path, weights_only=False)
        feat_mean = np.asarray(aux["feat_mean"], dtype=np.float64)
        feat_std = np.asarray(aux["feat_std"], dtype=np.float64)
        pop_order = list(aux.get("population_order", ["adult", "pediatric_5y"]))
        log.info("  loaded standardizer + pop order: %s", pop_order)
    else:
        feat_mean = None
        feat_std = None
        pop_order = ["adult", "pediatric_5y"]
        log.warning("  no aux file — using defaults")

    prior = build_box_prior()
    populations = load_populations()
    specs = load_drug_specs_from_json(args.validation_set)
    log.info("Loaded %d validation drugs, %d populations", len(specs), len(pop_order))

    from scipy import stats as scistats

    with open(args.validation_set) as f:
        val_data = json.load(f)
    val_meta = {d["name"]: d for d in val_data["drugs"]}

    out: dict = {
        "posterior": str(args.posterior),
        "validation_set": str(args.validation_set),
        "populations": pop_order,
        "n_calibration": args.n_calibration,
        "n_posterior": args.n_posterior,
        "levels": list(levels),
        "seed": args.seed,
        "ks_threshold": args.ks_threshold,
        "coverage_tol": args.coverage_tol,
        "pairs": [],
    }

    pass_count = 0
    total_pairs = len(specs) * len(pop_order)
    pair_idx = 0
    t_all = time.time()

    for pop_name in pop_order:
        pop_spec = populations[pop_name]
        pop_oh = population_onehot(pop_name, pop_order)

        for spec in specs:
            pair_idx += 1
            t0 = time.time()
            log.info(
                "[sbc-hier] [%d/%d] %s / %s",
                pair_idx, total_pairs, pop_name, spec.name,
            )

            # Build simulator for this (pop, drug) pair
            sim = EngineSimulator.for_drug(
                smiles=spec.smiles,
                dose_mg=spec.dose_mg,
                route=spec.route,
                physiology_yaml=pop_spec.physiology_yaml,
            )

            # Drug features from adult graph (population-independent)
            adult_sim = EngineSimulator.for_drug(
                smiles=spec.smiles,
                dose_mg=spec.dose_mg,
                route=spec.route,
            )
            logp_hint = val_meta.get(spec.name, {}).get("logp")
            feats = extract_drug_features(
                adult_sim,
                logp=float(logp_hint) if logp_hint is not None else None,
            )

            # Standardize features
            if feat_mean is not None:
                feats_std = (feats - feat_mean) / feat_std
            else:
                feats_std = feats

            rng = np.random.default_rng(
                args.seed + pair_idx * 1000
            )
            thetas_true = prior.sample_numpy(args.n_calibration, rng)

            xs = np.zeros((args.n_calibration, 1), dtype=np.float64)
            for j in range(args.n_calibration):
                val = sim.simulate_single(
                    thetas_true[j],
                    seed=args.seed + 7777 * pair_idx + j,
                )
                xs[j, 0] = val if np.isfinite(val) else -20.0

            # Pack: [log10_cmax, drug_features_std, pop_onehot]
            xs_packed = np.concatenate([
                xs,
                np.tile(feats_std[None, :], (args.n_calibration, 1)),
                np.tile(pop_oh[None, :], (args.n_calibration, 1)),
            ], axis=1).astype(np.float64)

            ranks, cov_counts = _coverage_and_ranks(
                thetas_true, result.posterior, xs_packed, levels=levels,
                n_posterior=args.n_posterior,
            )

            ks_stats = np.zeros(3)
            ks_p = np.zeros(3)
            for d in range(3):
                normalized = ranks[:, d] / max(args.n_posterior, 1)
                res = scistats.kstest(normalized, "uniform")
                ks_stats[d] = float(res.statistic)
                ks_p[d] = float(res.pvalue)

            coverage = {
                str(level): (
                    cov_counts[level].astype(np.float64) / args.n_calibration
                ).tolist()
                for level in levels
            }

            ks_ok = bool(np.all(ks_p > args.ks_threshold))
            cov_ok = True
            cov_max_dev = 0.0
            for level in levels:
                emp = np.array(coverage[str(level)])
                dev = float(np.max(np.abs(emp - level / 100.0)))
                cov_max_dev = max(cov_max_dev, dev)
                if dev > args.coverage_tol:
                    cov_ok = False
            pair_pass = ks_ok and cov_ok
            if pair_pass:
                pass_count += 1

            elapsed = time.time() - t0
            out["pairs"].append({
                "population": pop_name,
                "drug": spec.name,
                "dose_mg": spec.dose_mg,
                "ks_stats": ks_stats.tolist(),
                "ks_pvalues": ks_p.tolist(),
                "coverage": coverage,
                "coverage_max_deviation": float(cov_max_dev),
                "pass": pair_pass,
                "elapsed_s": float(elapsed),
            })
            log.info(
                "  KS_p=[%.3f %.3f %.3f]  cov_max_dev=%.3f  pass=%s  (%.1fs)",
                ks_p[0], ks_p[1], ks_p[2], cov_max_dev, pair_pass, elapsed,
            )

    total_elapsed = time.time() - t_all
    out["total_elapsed_s"] = float(total_elapsed)
    out["pass_count"] = int(pass_count)
    out["total_pairs"] = int(total_pairs)
    out["gate_passed"] = bool(pass_count == total_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", args.out)
    log.info(
        "SUMMARY: %d/%d (pop, drug) pairs pass gate  (total %.1fs)",
        pass_count, total_pairs, total_elapsed,
    )

    if pass_count < total_pairs:
        log.warning("GATE NOT FULLY PASSED")
    else:
        log.info("GATE PASSED")


if __name__ == "__main__":
    main()
