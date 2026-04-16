"""Train the continuous hierarchical NPE.

Usage:
    python scripts/sbi_train_continuous_hierarchical.py \
        --data data/sbi/continuous_train.npz \
        --out models/sbi/continuous_hierarchical_nsf.pt \
        --epochs 100
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
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train continuous hierarchical NPE")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/sbi/continuous_hierarchical_nsf.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--transforms", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    theta_all = data["theta"]
    x_all = data["x"]
    feat_all = data["drug_features"]
    bw_all = data["bw"]
    age_all = data["age"]

    finite = np.isfinite(x_all[:, 0])
    theta_all = theta_all[finite]
    x_all = x_all[finite]
    feat_all = feat_all[finite]
    bw_all = bw_all[finite]
    age_all = age_all[finite]
    logger.info("Training data: %d rows (%d dropped)", theta_all.shape[0], (~finite).sum())

    log_bw_norm = np.log10(bw_all / 70.0)
    log_age_norm = np.log10(np.clip(age_all, 0.5, None))

    feat_extended = np.column_stack([feat_all, log_bw_norm, log_age_norm])
    feat_mean = feat_extended.mean(axis=0)
    feat_std = feat_extended.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    feat_norm = (feat_extended - feat_mean) / feat_std

    obs = np.column_stack([x_all, feat_norm])  # (N, 15)

    import torch

    theta_t = torch.as_tensor(theta_all, dtype=torch.float32)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)

    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW

    prior = BoxUniform(
        low=torch.tensor(PRIOR_LOW, dtype=torch.float32),
        high=torch.tensor(PRIOR_HIGH, dtype=torch.float32),
    )

    inference = SNPE(prior=prior, density_estimator="nsf")
    inference.append_simulations(theta_t, obs_t)

    t0 = time.time()
    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        max_num_epochs=args.epochs,
        show_train_summary=True,
    )
    logger.info("Training took %.1f min", (time.time() - t0) / 60)

    posterior = inference.build_posterior(density_estimator)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "posterior": posterior,
            "prior_low": list(PRIOR_LOW),
            "prior_high": list(PRIOR_HIGH),
        },
        out_path,
    )

    aux_path = out_path.with_suffix(".aux.pt")
    torch.save(
        {
            "feat_mean": feat_mean.tolist(),
            "feat_std": feat_std.tolist(),
            "continuous_hierarchical": True,
            "n_feat_dims": 14,
        },
        aux_path,
    )
    logger.info("Saved %s + %s", out_path, aux_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
