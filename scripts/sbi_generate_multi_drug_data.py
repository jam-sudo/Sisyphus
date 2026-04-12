#!/usr/bin/env python3
"""Generate conditional (theta, log10_cmax, drug_features) training pairs
for the multi-drug SBI amortizer (Track A, Phase 2.0).

Each drug in ``data/sbi/train_drug_set.json`` is assigned an independent
block of thetas sampled from the prior. Every block runs under its own
``EngineSimulator``, parallelised across drugs via ``multiprocessing``.

The mini mode is a kill-switch gate: fewer drugs × fewer thetas per drug,
designed to fail fast if the architecture is broken. The full mode is the
production 50 drugs × 1000 thetas run.

Usage::

    python scripts/sbi_generate_multi_drug_data.py --mode mini
    python scripts/sbi_generate_multi_drug_data.py --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_gen_md")

TRAIN_JSON = ROOT / "data" / "sbi" / "train_drug_set.json"


def _worker(args: tuple) -> dict:
    """One worker — build a simulator, sweep a theta block, return block data."""
    (name, smiles, dose_mg, route, thetas, base_seed, logp_hint) = args

    # Late imports so this works under multiprocessing spawn.
    from sisyphus.predict.chemistry import compute_profile  # noqa: F401
    from sisyphus.sbi.multi_drug import DrugSpec, MultiDrugSimulator, extract_drug_features
    from sisyphus.sbi.simulator import EngineSimulator

    t0 = time.time()
    sim = EngineSimulator.for_drug(smiles=smiles, dose_mg=dose_mg, route=route)
    feats = extract_drug_features(sim, logp=logp_hint)
    n = len(thetas)
    log10_cmax = np.full((n, 1), np.nan, dtype=np.float64)
    for i in range(n):
        val = sim.simulate_single(thetas[i], seed=base_seed + i)
        log10_cmax[i, 0] = val if np.isfinite(val) else np.nan
    elapsed = time.time() - t0
    n_valid = int(np.isfinite(log10_cmax[:, 0]).sum())
    return {
        "name": name,
        "thetas": thetas.astype(np.float64),
        "log10_cmax": log10_cmax,
        "features": feats,
        "elapsed": elapsed,
        "n_valid": n_valid,
        "n_requested": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("mini", "full"), default="mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-drugs", type=int, default=20)
    p.add_argument("--mini-theta-per-drug", type=int, default=500)
    p.add_argument("--full-theta-per-drug", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from sisyphus.sbi.priors import build_box_prior

    prior = build_box_prior()

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)
    all_drugs = train_data["drugs"]
    print(f"[gen-md] loaded {len(all_drugs)} drugs from {TRAIN_JSON.name}")

    if args.mode == "mini":
        drugs = all_drugs[: args.mini_drugs]
        theta_per = args.mini_theta_per_drug
        if args.out is None:
            args.out = ROOT / "data" / "sbi" / "multi_drug_mini.npz"
    else:
        drugs = all_drugs
        theta_per = args.full_theta_per_drug
        if args.out is None:
            args.out = ROOT / "data" / "sbi" / "multi_drug_train.npz"

    total = len(drugs) * theta_per
    print(f"[gen-md] mode={args.mode}  drugs={len(drugs)}  "
          f"theta/drug={theta_per}  total_sims={total}")
    print(f"[gen-md] workers={args.workers}  out={args.out}")

    # Per-drug theta blocks use base_seed + drug_index * 2**16 so blocks are disjoint.
    rng_master = np.random.default_rng(args.seed)
    jobs = []
    for di, d in enumerate(drugs):
        block_rng = np.random.default_rng(args.seed + 1_000_000 * (di + 1))
        thetas = prior.sample_numpy(theta_per, block_rng)
        base_seed = args.seed + (di + 1) * 2**16
        jobs.append(
            (
                d["name"],
                d["smiles"],
                float(d["dose_mg"]),
                d.get("route", "oral"),
                thetas,
                int(base_seed),
                float(d.get("logp", 2.0)),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = mp.get_context("spawn")
    results: list[dict] = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            elapsed = time.time() - t0
            done_frac = (i + 1) / len(jobs)
            print(
                f"[gen-md] [{i+1:>3d}/{len(jobs)}] {res['name']:<30s}  "
                f"valid={res['n_valid']}/{res['n_requested']}  "
                f"drug_dt={res['elapsed']:5.1f}s  "
                f"total={elapsed:6.1f}s  "
                f"eta={elapsed/done_frac - elapsed:6.1f}s",
                flush=True,
            )
            results.append(res)

    total_elapsed = time.time() - t0
    print(f"[gen-md] wall time: {total_elapsed:.1f}s  ({total/total_elapsed:.1f} sim/s)")

    # Stack outputs (sort by drug name for reproducibility)
    results.sort(key=lambda r: r["name"])
    theta_rows = []
    x_rows = []
    feat_rows = []
    name_rows = []
    per_drug_meta = []
    for r in results:
        theta = r["thetas"]
        x = r["log10_cmax"]
        feats = r["features"]
        mask = np.isfinite(x[:, 0])
        theta_keep = theta[mask]
        x_keep = x[mask]
        theta_rows.append(theta_keep)
        x_rows.append(x_keep)
        feat_rows.append(np.tile(feats[None, :], (int(mask.sum()), 1)))
        name_rows.extend([r["name"]] * int(mask.sum()))
        per_drug_meta.append({
            "name": r["name"],
            "n_requested": int(r["n_requested"]),
            "n_valid": int(r["n_valid"]),
            "elapsed_s": float(r["elapsed"]),
            "features": feats.tolist(),
        })

    theta_all = np.concatenate(theta_rows, axis=0).astype(np.float32)
    x_all = np.concatenate(x_rows, axis=0).astype(np.float32)
    feat_all = np.concatenate(feat_rows, axis=0).astype(np.float32)
    names_all = np.array(name_rows)

    np.savez(
        args.out,
        theta=theta_all,
        x=x_all,
        drug_features=feat_all,
        drug_names=names_all,
        prior_low=prior.low.astype(np.float32),
        prior_high=prior.high.astype(np.float32),
        theta_names=np.array(prior.names),
        seed=np.int64(args.seed),
        mode=np.array(args.mode),
    )
    print(
        f"[gen-md] saved: theta={theta_all.shape}, x={x_all.shape}, "
        f"drug_features={feat_all.shape}, names={names_all.shape}"
    )
    meta_path = args.out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "seed": args.seed,
                "n_drugs": len(drugs),
                "theta_per_drug": theta_per,
                "total_requested": total,
                "wall_time_s": total_elapsed,
                "workers": args.workers,
                "per_drug": per_drug_meta,
            },
            f,
            indent=2,
        )
    print(f"[gen-md] wrote {args.out} + {meta_path.name}")


if __name__ == "__main__":
    main()
