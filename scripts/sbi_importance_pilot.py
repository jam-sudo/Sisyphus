#!/usr/bin/env python3
"""Measure correction ESS and nuisance-estimator noise on one validation drug."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sisyphus.sbi.amortizer import load_result
from sisyphus.sbi.multi_drug import (
    MultiDrugSimulator,
    load_drug_specs_from_json,
)
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW, build_box_prior

log = logging.getLogger(__name__)
# Source: repository layout; this script resides in scripts/.
ROOT_PATH = Path(__file__).resolve().parent.parent


def main() -> None:
    """Write a reproducible budget comparison for one synthetic observation."""
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument(
        "--posterior", type=Path, default=ROOT_PATH / "models/sbi/multi_drug_nsf.pt",
    )
    parser.add_argument("--n-proposals", type=int, default=64)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_proposals < 2 or args.k_max < 2:
        raise ValueError("Pilot needs at least two proposals and nuisance draws")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    specs = load_drug_specs_from_json(ROOT_PATH / "data/sbi/validation_drug_set.json")
    spec = next(spec for spec in specs if spec.name == args.drug)
    bundle = MultiDrugSimulator.for_single(spec)
    sim = bundle.simulators[spec.name]
    model = load_result(args.posterior)
    aux = torch.load(args.posterior.with_suffix(".aux.pt"), weights_only=False)
    features = bundle.get_features(spec.name)
    features = (features - np.asarray(aux["feat_mean"])) / np.asarray(aux["feat_std"])
    true_theta = build_box_prior().sample_numpy(1, rng)[0]
    observed = sim.simulate_single(true_theta, int(rng.integers(0, 2**31 - 1)))
    x = np.r_[observed, features].astype(np.float32)
    x_t = torch.as_tensor(x[None, :])
    n_prior = max(1, int(round(args.n_proposals * 0.05)))
    prior_fraction = n_prior / args.n_proposals
    with torch.no_grad():
        flow = model.posterior.sample(
            (args.n_proposals - n_prior,), x=x_t, show_progress_bars=False,
            reject_outside_prior=False,
        ).detach().cpu().numpy()
        prior = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n_prior, 3))
        theta = np.concatenate([flow, prior])
        log_q_flow = model.posterior.log_prob(
            torch.as_tensor(theta, dtype=torch.float32), x=x_t, norm_posterior=False,
        ).detach().cpu().numpy()
    log_prior = -np.log(np.prod(PRIOR_HIGH - PRIOR_LOW))
    log_prior_theta = np.where(
        np.all((theta >= PRIOR_LOW) & (theta <= PRIOR_HIGH), axis=1), log_prior, -np.inf,
    )
    log_q = np.logaddexp(
        np.log1p(-prior_fraction) + log_q_flow, np.log(prior_fraction) + log_prior,
    )
    sigma = sim.obs_sigma_log10
    likelihood = np.full((args.n_proposals, args.k_max), -np.inf)
    for i, row in enumerate(theta):
        for k in range(args.k_max):
            predicted = sim.simulate_single(
                row, int(rng.integers(0, 2**31 - 1)), add_noise=False,
            )
            if np.isfinite(predicted):
                likelihood[i, k] = (
                    -0.5 * ((observed - predicted) / sigma) ** 2
                    - np.log(sigma) - 0.5 * np.log(2 * np.pi)
                )
        if (i + 1) % 16 == 0:
            log.info("%s: %d/%d proposals", args.drug, i + 1, args.n_proposals)

    budgets = []
    for k in (1, 2, 4, 8):
        if k > args.k_max:
            continue
        log_lik = logsumexp(likelihood[:, :k], axis=1) - np.log(k)
        log_w = log_lik + log_prior_theta - log_q
        weights = np.exp(log_w - logsumexp(log_w))
        budgets.append({
            "proposals": args.n_proposals,
            "nuisance_draws": k,
            "ode_solves": args.n_proposals * k,
            "ess": float(1.0 / np.sum(weights**2)),
            "theta_mean": np.average(theta, axis=0, weights=weights).tolist(),
        })
    payload = {
        "drug": args.drug,
        "seed": args.seed,
        "defensive_prior_fraction": prior_fraction,
        "true_theta": true_theta.tolist(),
        "observation_log10_cmax": float(observed),
        "budgets": budgets,
        "reference": "largest proposal and nuisance budget in this pilot; not exact",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
