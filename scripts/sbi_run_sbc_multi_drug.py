#!/usr/bin/env python3
"""Per-drug SBC gate for the multi-drug conditional amortizer.

Runs simulation-based calibration independently for each holdout
validation drug. For each drug:

    1. Build a fresh EngineSimulator (full scipy engine).
    2. Extract the nominal 12-D drug feature vector.
    3. Draw n_calibration thetas from the prior, simulate x_obs per draw.
    4. For each draw, construct the packed observation and sample the
       posterior. Compute rank of the true theta within posterior samples
       and coverage at nominal CI levels.
    5. KS uniformity test + coverage deviation.

Gate:
    - All KS p-values > 0.01
    - All coverage levels within 10pp of nominal

Aggregates per-drug results into ``data/validation/sbi_sbc_multi_drug.json``.

Usage::

    python scripts/sbi_run_sbc_multi_drug.py \\
        --posterior models/sbi/multi_drug_mini_nsf.pt \\
        --validation-set data/sbi/validation_drug_set.json \\
        --n-calibration 100 --n-posterior 500 \\
        --out data/validation/sbi_sbc_multi_drug_mini.json
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
log = logging.getLogger("sbi_sbc_md")

from sisyphus.sbi.amortizer import load_result  # noqa: E402
from sisyphus.sbi.multi_drug import (  # noqa: E402
    DrugSpec,
    MultiDrugSimulator,
    extract_drug_features,
    load_drug_specs_from_json,
)
from sisyphus.sbi.priors import build_box_prior  # noqa: E402


def _coverage_and_ranks(
    thetas_true: np.ndarray,
    posterior,
    xs_packed: np.ndarray,
    levels: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Per-dim ranks and coverage counts for one drug."""
    import torch

    n_cal, d_theta = thetas_true.shape
    ranks = np.zeros((n_cal, d_theta), dtype=np.int64)
    cov = {level: np.zeros(d_theta, dtype=np.int64) for level in levels}
    for i in range(n_cal):
        x_t = torch.as_tensor(xs_packed[i : i + 1], dtype=torch.float32)
        samples = posterior.sample((500,), x=x_t, show_progress_bars=False)
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
        log.info("  loaded feature standardizer from %s", aux_path.name)
    else:
        feat_mean = None
        feat_std = None
        log.warning("  no aux standardizer file — assuming features were not scaled")

    prior = build_box_prior()

    specs = load_drug_specs_from_json(args.validation_set)
    log.info("Loaded %d validation drugs", len(specs))

    from scipy import stats as scistats
    import json as _json

    out: dict = {
        "posterior": str(args.posterior),
        "validation_set": str(args.validation_set),
        "n_calibration": args.n_calibration,
        "n_posterior": args.n_posterior,
        "levels": list(levels),
        "seed": args.seed,
        "ks_threshold": args.ks_threshold,
        "coverage_tol": args.coverage_tol,
        "drugs": [],
    }

    pass_count = 0
    t_all = time.time()

    with open(args.validation_set) as f:
        val_data = json.load(f)
    val_meta = {d["name"]: d for d in val_data["drugs"]}

    for i, spec in enumerate(specs):
        t0 = time.time()
        log.info("[sbc-md] [%d/%d] %s", i + 1, len(specs), spec.name)
        sim_bundle = MultiDrugSimulator.for_single(spec)
        sim = sim_bundle.simulators[spec.name]
        logp_hint = val_meta.get(spec.name, {}).get("logp")
        feats = extract_drug_features(sim, logp=float(logp_hint) if logp_hint is not None else None)

        # Standardize features same way as training
        if feat_mean is not None:
            feats_std = (feats - feat_mean) / feat_std
        else:
            feats_std = feats

        rng = np.random.default_rng(args.seed + i * 1000)
        thetas_true = prior.sample_numpy(args.n_calibration, rng)

        xs = np.zeros((args.n_calibration, 1), dtype=np.float64)
        for j in range(args.n_calibration):
            val = sim.simulate_single(thetas_true[j], seed=args.seed + 7777 * (i + 1) + j)
            xs[j, 0] = val if np.isfinite(val) else -20.0
        xs_packed = np.concatenate(
            [xs, np.tile(feats_std[None, :], (args.n_calibration, 1))],
            axis=1,
        ).astype(np.float64)

        ranks, cov_counts = _coverage_and_ranks(
            thetas_true, result.posterior, xs_packed, levels=levels
        )

        ks_stats = np.zeros(3)
        ks_p = np.zeros(3)
        for d in range(3):
            normalized = ranks[:, d] / max(args.n_posterior, 1)
            res = scistats.kstest(normalized, "uniform")
            ks_stats[d] = float(res.statistic)
            ks_p[d] = float(res.pvalue)

        coverage = {
            level: (cov_counts[level].astype(np.float64) / args.n_calibration).tolist()
            for level in levels
        }
        # Gate check
        ks_ok = bool(np.all(ks_p > args.ks_threshold))
        cov_ok = True
        cov_max_dev = 0.0
        for level in levels:
            emp = np.array(coverage[level])
            dev = float(np.max(np.abs(emp - level / 100.0)))
            cov_max_dev = max(cov_max_dev, dev)
            if dev > args.coverage_tol:
                cov_ok = False
        drug_pass = ks_ok and cov_ok
        if drug_pass:
            pass_count += 1

        elapsed = time.time() - t0
        out["drugs"].append({
            "name": spec.name,
            "dose_mg": spec.dose_mg,
            "ks_stats": ks_stats.tolist(),
            "ks_pvalues": ks_p.tolist(),
            "coverage": coverage,
            "coverage_max_deviation": float(cov_max_dev),
            "drug_features": feats.tolist(),
            "pass": drug_pass,
            "elapsed_s": float(elapsed),
        })
        log.info(
            "  KS_p=[%.3f %.3f %.3f]  cov_max_dev=%.3f  pass=%s  (%.1fs)",
            ks_p[0], ks_p[1], ks_p[2], cov_max_dev, drug_pass, elapsed,
        )

    total_elapsed = time.time() - t_all
    out["total_elapsed_s"] = float(total_elapsed)
    out["pass_count"] = int(pass_count)
    out["n_drugs"] = int(len(specs))
    out["gate_passed"] = bool(pass_count == len(specs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        _json.dump(out, f, indent=2)
    log.info("Wrote %s", args.out)
    log.info(
        "SUMMARY: %d/%d drugs pass gate  (total %.1fs)",
        pass_count, len(specs), total_elapsed,
    )

    if pass_count < len(specs):
        log.warning("GATE NOT FULLY PASSED — investigate before full run")
    else:
        log.info("GATE PASSED")


if __name__ == "__main__":
    main()
