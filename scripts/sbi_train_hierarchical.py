#!/usr/bin/env python3
"""Train a hierarchical conditional NPE from hierarchical_*.npz data.

The amortizer learns ``p(theta | x_obs)`` where ``x_obs`` is the
concatenation of:
    [log10(Cmax), drug_features (12D), pop_onehot (N_pop D)]

so the density estimator conditions on the observed Cmax, the nominal ADME
profile of the drug, AND the population class. One network serves all drugs
in the training set across all populations.

Usage::

    python scripts/sbi_train_hierarchical.py \\
        --data data/sbi/hierarchical_mini.npz \\
        --out  models/sbi/hierarchical_mini_nsf.pt \\
        --density-estimator nsf --hidden-features 64 --num-transforms 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_train_hier")

from sisyphus.sbi.amortizer import save_result, train_npe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--density-estimator", default="nsf", choices=("maf", "nsf"))
    p.add_argument("--hidden-features", type=int, default=64)
    p.add_argument("--num-transforms", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--stop-after-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-hidden", type=int, default=None,
                   help="If set, prepend an MLP embedding_net with this hidden size")
    args = p.parse_args()

    log.info("Loading %s", args.data)
    data = np.load(args.data, allow_pickle=True)
    theta = data["theta"].astype(np.float32)
    log10_cmax = data["x"].astype(np.float32)
    drug_features = data["drug_features"].astype(np.float32)
    pop_onehot = data["pop_onehot"].astype(np.float32)
    prior_low = data["prior_low"].astype(np.float32)
    prior_high = data["prior_high"].astype(np.float32)
    population_order = list(data["population_order"])

    log.info(
        "  theta=%s  log10_cmax=%s  drug_features=%s  pop_onehot=%s",
        theta.shape, log10_cmax.shape, drug_features.shape, pop_onehot.shape,
    )
    log.info("  populations: %s", population_order)
    log.info(
        "  log10_cmax range=[%.3f, %.3f]",
        float(log10_cmax.min()), float(log10_cmax.max()),
    )

    # Pack observation: (N, 1 + 12 + n_pop) = (N, 15) for 2 populations
    n_drug_feat = drug_features.shape[1]
    n_pop = pop_onehot.shape[1]
    x_full = np.concatenate(
        [log10_cmax, drug_features, pop_onehot], axis=1
    ).astype(np.float32)
    log.info("  x_full (packed) shape=%s  (1 + %d + %d)", x_full.shape, n_drug_feat, n_pop)

    # Standardize drug-feature columns only (log10_cmax is already log-scale;
    # pop one-hot is binary and should not be standardized).
    feat_mean = drug_features.mean(axis=0)
    feat_std = drug_features.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    x_full[:, 1:1 + n_drug_feat] = (drug_features - feat_mean) / feat_std

    t0 = time.time()
    result = train_npe(
        theta=theta,
        x=x_full,
        prior_low=prior_low,
        prior_high=prior_high,
        density_estimator=args.density_estimator,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after_epochs,
        seed=args.seed,
        embedding_net_hidden=args.embedding_hidden,
    )
    elapsed = time.time() - t0
    log.info("Training done in %.1fs", elapsed)
    if result.training_losses:
        log.info("  last train loss=%.4f", result.training_losses[-1])
    if result.validation_losses:
        log.info("  last valid loss=%.4f", result.validation_losses[-1])

    # Persist feature standardizer + population metadata
    save_result(result, args.out)
    import torch
    aux_path = args.out.with_suffix(".aux.pt")
    torch.save({
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_features": int(n_drug_feat),
        "n_pop": int(n_pop),
        "population_order": population_order,
        "hierarchical": True,
    }, aux_path)
    log.info("Wrote %s and %s", args.out, aux_path)


if __name__ == "__main__":
    main()
