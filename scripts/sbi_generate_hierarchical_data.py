#!/usr/bin/env python3
"""Generate hierarchical (theta, log10_cmax, drug_features, pop_onehot)
training pairs for the population-aware SBI amortizer (Track C1).

For each (population, drug) pair, generates a block of thetas from the prior
and simulates log10(Cmax) using the population-specific graph. Drug features
are always extracted from the adult reference graph (population-independent).

Modes:
    mini — quick gate: fewer drugs x fewer thetas x 2 populations
    full — production run: all drugs x 1000 thetas x 2 populations

Usage::

    python scripts/sbi_generate_hierarchical_data.py --mode mini
    python scripts/sbi_generate_hierarchical_data.py --mode full
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
log = logging.getLogger("sbi_gen_hier")

TRAIN_JSON = ROOT / "data" / "sbi" / "train_drug_set.json"
POPULATIONS_JSON = ROOT / "data" / "sbi" / "populations.json"


def _worker(args: tuple) -> dict:
    """One worker: build a simulator for (pop, drug), sweep theta block."""
    (pop_name, physiology_yaml, drug_name, smiles, dose_mg, route,
     thetas, base_seed, logp_hint) = args

    from sisyphus.sbi.multi_drug import extract_drug_features
    from sisyphus.sbi.simulator import EngineSimulator

    t0 = time.time()
    sim = EngineSimulator.for_drug(
        smiles=smiles,
        dose_mg=dose_mg,
        route=route,
        physiology_yaml=Path(physiology_yaml),
    )

    # Features always from adult graph — but we only compute once per drug.
    # The main process handles feature extraction; worker just simulates.
    n = len(thetas)
    log10_cmax = np.full((n, 1), np.nan, dtype=np.float64)
    for i in range(n):
        val = sim.simulate_single(thetas[i], seed=base_seed + i)
        log10_cmax[i, 0] = val if np.isfinite(val) else np.nan
    elapsed = time.time() - t0
    n_valid = int(np.isfinite(log10_cmax[:, 0]).sum())
    return {
        "pop_name": pop_name,
        "drug_name": drug_name,
        "thetas": thetas.astype(np.float64),
        "log10_cmax": log10_cmax,
        "elapsed": elapsed,
        "n_valid": n_valid,
        "n_requested": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("mini", "full"), default="mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-drugs", type=int, default=10)
    p.add_argument("--mini-theta-per-drug", type=int, default=200)
    p.add_argument("--full-theta-per-drug", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from sisyphus.sbi.multi_drug import (
        extract_drug_features,
        load_populations,
        population_onehot,
    )
    from sisyphus.sbi.priors import build_box_prior
    from sisyphus.sbi.simulator import EngineSimulator

    prior = build_box_prior()
    populations = load_populations(POPULATIONS_JSON)
    pop_names = sorted(populations.keys())
    n_pop = len(pop_names)
    print(f"[gen-hier] populations: {pop_names}")

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)
    all_drugs = train_data["drugs"]

    if args.mode == "mini":
        drugs = all_drugs[: args.mini_drugs]
        theta_per = args.mini_theta_per_drug
        if args.out is None:
            args.out = ROOT / "data" / "sbi" / "hierarchical_mini.npz"
    else:
        drugs = all_drugs
        theta_per = args.full_theta_per_drug
        if args.out is None:
            args.out = ROOT / "data" / "sbi" / "hierarchical_train.npz"

    total = len(drugs) * theta_per * n_pop
    print(f"[gen-hier] mode={args.mode}  drugs={len(drugs)}  pops={n_pop}  "
          f"theta/pair={theta_per}  total_sims={total}")

    # Pre-compute drug features from adult graph (population-independent)
    print("[gen-hier] extracting population-independent drug features (adult graph)...")
    drug_features: dict[str, np.ndarray] = {}
    for d in drugs:
        sim_adult = EngineSimulator.for_drug(
            smiles=d["smiles"],
            dose_mg=float(d["dose_mg"]),
            route=d.get("route", "oral"),
        )
        drug_features[d["name"]] = extract_drug_features(
            sim_adult, logp=float(d.get("logp", 2.0))
        )
    print(f"[gen-hier] extracted features for {len(drug_features)} drugs")

    # Build job list: one job per (population, drug)
    jobs = []
    for pi, pop_name in enumerate(pop_names):
        pop_spec = populations[pop_name]
        for di, d in enumerate(drugs):
            block_rng = np.random.default_rng(
                args.seed + 1_000_000 * (pi * len(drugs) + di + 1)
            )
            thetas = prior.sample_numpy(theta_per, block_rng)
            base_seed = args.seed + (pi * len(drugs) + di + 1) * 2**16
            jobs.append((
                pop_name,
                str(pop_spec.physiology_yaml),
                d["name"],
                d["smiles"],
                float(d["dose_mg"]),
                d.get("route", "oral"),
                thetas,
                int(base_seed),
                float(d.get("logp", 2.0)),
            ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = mp.get_context("spawn")
    results: list[dict] = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            elapsed = time.time() - t0
            done_frac = (i + 1) / len(jobs)
            print(
                f"[gen-hier] [{i+1:>3d}/{len(jobs)}] "
                f"{res['pop_name']}/{res['drug_name']:<25s}  "
                f"valid={res['n_valid']}/{res['n_requested']}  "
                f"drug_dt={res['elapsed']:5.1f}s  "
                f"total={elapsed:6.1f}s  "
                f"eta={elapsed/done_frac - elapsed:6.1f}s",
                flush=True,
            )
            results.append(res)

    total_elapsed = time.time() - t0

    # Stack outputs
    theta_rows = []
    x_rows = []
    feat_rows = []
    pop_rows = []
    name_rows = []
    pop_name_rows = []
    per_pair_meta = []

    for res in sorted(results, key=lambda r: (r["pop_name"], r["drug_name"])):
        theta = res["thetas"]
        x = res["log10_cmax"]
        mask = np.isfinite(x[:, 0])
        n_valid = int(mask.sum())
        if n_valid == 0:
            continue
        theta_rows.append(theta[mask])
        x_rows.append(x[mask])
        feat_rows.append(
            np.tile(drug_features[res["drug_name"]][None, :], (n_valid, 1))
        )
        oh = population_onehot(res["pop_name"], pop_names)
        pop_rows.append(np.tile(oh[None, :], (n_valid, 1)))
        name_rows.extend([res["drug_name"]] * n_valid)
        pop_name_rows.extend([res["pop_name"]] * n_valid)
        per_pair_meta.append({
            "pop_name": res["pop_name"],
            "drug_name": res["drug_name"],
            "n_requested": int(res["n_requested"]),
            "n_valid": int(res["n_valid"]),
            "elapsed_s": float(res["elapsed"]),
        })

    theta_all = np.concatenate(theta_rows, axis=0).astype(np.float32)
    x_all = np.concatenate(x_rows, axis=0).astype(np.float32)
    feat_all = np.concatenate(feat_rows, axis=0).astype(np.float32)
    pop_all = np.concatenate(pop_rows, axis=0).astype(np.float32)

    np.savez(
        args.out,
        theta=theta_all,
        x=x_all,
        drug_features=feat_all,
        pop_onehot=pop_all,
        drug_names=np.array(name_rows),
        pop_names=np.array(pop_name_rows),
        population_order=np.array(pop_names),
        prior_low=prior.low.astype(np.float32),
        prior_high=prior.high.astype(np.float32),
        theta_names=np.array(prior.names),
        seed=np.int64(args.seed),
        mode=np.array(args.mode),
    )
    print(
        f"[gen-hier] saved: theta={theta_all.shape}, x={x_all.shape}, "
        f"drug_features={feat_all.shape}, pop_onehot={pop_all.shape}"
    )
    print(f"[gen-hier] wall time: {total_elapsed:.1f}s  ({total/total_elapsed:.1f} sim/s)")

    meta_path = args.out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "seed": args.seed,
            "n_drugs": len(drugs),
            "n_populations": n_pop,
            "population_order": pop_names,
            "theta_per_pair": theta_per,
            "total_requested": total,
            "wall_time_s": total_elapsed,
            "workers": args.workers,
            "per_pair": per_pair_meta,
        }, f, indent=2)
    print(f"[gen-hier] wrote {args.out} + {meta_path.name}")


if __name__ == "__main__":
    main()
