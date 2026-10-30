#!/usr/bin/env python3
"""Exact joint-posterior rejection reference for one synthetic Cmax observation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sisyphus.sbi.multi_drug import MultiDrugSimulator, load_drug_specs_from_json
from sisyphus.sbi.priors import build_box_prior

# Source: repository layout; this script resides in scripts/.
ROOT_PATH = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


def main() -> None:
    """Sample theta exactly by accepting joint prior (theta, nuisance) draws."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument("--n-accepted", type=int, default=200)
    parser.add_argument("--max-proposals", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_accepted < 1 or args.max_proposals < args.n_accepted:
        raise ValueError("Invalid rejection budget")
    spec = next(s for s in load_drug_specs_from_json(
        ROOT_PATH / "data/sbi/validation_drug_set.json"
    ) if s.name == args.drug)
    sim = MultiDrugSimulator.for_single(spec).simulators[spec.name]
    rng = np.random.default_rng(args.seed)
    prior = build_box_prior()
    true_theta = prior.sample_numpy(1, rng)[0]
    observed = sim.simulate_single(true_theta, int(rng.integers(0, 2**31 - 1)))
    if not np.isfinite(observed):
        raise RuntimeError("Synthetic observation solve failed")
    accepted: list[list[float]] = []
    proposals = 0
    while len(accepted) < args.n_accepted and proposals < args.max_proposals:
        theta = prior.sample_numpy(1, rng)[0]
        predicted = sim.simulate_single(
            theta, int(rng.integers(0, 2**31 - 1)), add_noise=False,
        )
        proposals += 1
        if np.isfinite(predicted) and rng.random() < np.exp(
            -0.5 * ((observed - predicted) / sim.obs_sigma_log10) ** 2
        ):
            accepted.append(theta.tolist())
        if proposals % 1000 == 0:
            log.info("%s: %d accepted / %d proposals", args.drug, len(accepted), proposals)
    if len(accepted) < args.n_accepted:
        raise RuntimeError(f"Only {len(accepted)} accepted in {proposals} proposals")
    samples = np.asarray(accepted)
    payload = {
        "drug": args.drug,
        "seed": args.seed,
        "true_theta": true_theta.tolist(),
        "observation_log10_cmax": float(observed),
        "n_accepted": len(accepted),
        "n_proposals": proposals,
        "acceptance_rate": len(accepted) / proposals,
        "theta_mean": samples.mean(axis=0).tolist(),
        "theta_sd": samples.std(axis=0, ddof=1).tolist(),
        "method": "Exact joint-prior rejection using Gaussian density divided by its maximum",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
