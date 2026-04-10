#!/usr/bin/env python3
"""Generate (theta, x) training pairs for SBI amortizer.

Runs the full scipy engine simulator on N theta samples drawn from the
box-uniform prior and saves the results to ``data/sbi/morphine_train.npz``.

Usage::

    python scripts/sbi_generate_training_data.py --n 5000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_gen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.sbi.priors import build_box_prior  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator  # noqa: E402

# Default: morphine (well-characterized holdout drug with existing IBIS benchmark)
DEFAULT_DRUG = "morphine"
DEFAULT_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
DEFAULT_DOSE_MG = 30.0
DEFAULT_ROUTE = "oral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drug-name", default=DEFAULT_DRUG)
    parser.add_argument("--smiles", default=DEFAULT_SMILES)
    parser.add_argument("--dose-mg", type=float, default=DEFAULT_DOSE_MG)
    parser.add_argument("--route", default=DEFAULT_ROUTE, choices=("oral", "iv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to data/sbi/{drug_name}_train.npz",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / "data" / "sbi" / f"{args.drug_name}_train.npz"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sbi-gen] Building simulator for {args.drug_name} ({args.n} samples)...")
    sim = EngineSimulator.for_drug(args.smiles, args.dose_mg, route=args.route)

    prior = build_box_prior()
    rng = np.random.default_rng(args.seed)
    thetas = prior.sample_numpy(args.n, rng).astype(np.float64)

    print(f"[sbi-gen] Prior: {prior.names}")
    print(f"[sbi-gen]  low : {prior.low}")
    print(f"[sbi-gen]  high: {prior.high}")

    t0 = time.time()

    def _progress(n_done: int) -> None:
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        remaining = (args.n - n_done) / rate if rate > 0 else 0.0
        print(
            f"[sbi-gen] {n_done:>5d}/{args.n} done "
            f"({100 * n_done / args.n:5.1f}%)  "
            f"{rate:5.1f} sim/s  "
            f"eta {remaining/60:4.1f} min",
            flush=True,
        )

    log10_cmax = sim.simulate_batch(thetas, seed=args.seed + 1, progress=_progress)
    elapsed = time.time() - t0

    n_valid = int(np.sum(np.isfinite(log10_cmax)))
    n_fail = int(args.n - n_valid)
    print(
        f"[sbi-gen] DONE. {n_valid}/{args.n} valid ({100 * n_valid / args.n:.1f}%), "
        f"{n_fail} failures, {elapsed:.1f}s total ({args.n/elapsed:.1f} sim/s)"
    )

    # Drop failures for training
    finite = np.isfinite(log10_cmax[:, 0])
    thetas_ok = thetas[finite]
    x_ok = log10_cmax[finite]
    print(
        f"[sbi-gen] Saved shapes: theta={thetas_ok.shape}, x={x_ok.shape}, "
        f"x range=[{float(x_ok.min()):.3f}, {float(x_ok.max()):.3f}]"
    )

    np.savez(
        args.out,
        theta=thetas_ok.astype(np.float32),
        x=x_ok.astype(np.float32),
        theta_all=thetas.astype(np.float32),
        x_all=log10_cmax.astype(np.float32),
        prior_low=prior.low.astype(np.float32),
        prior_high=prior.high.astype(np.float32),
        theta_names=np.array(prior.names),
        drug_name=np.array(args.drug_name),
        smiles=np.array(args.smiles),
        dose_mg=np.float32(args.dose_mg),
        route=np.array(args.route),
        seed=np.int64(args.seed),
        n_requested=np.int64(args.n),
        n_valid=np.int64(n_valid),
    )
    print(f"[sbi-gen] Wrote {args.out}")


if __name__ == "__main__":
    main()
