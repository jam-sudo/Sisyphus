#!/usr/bin/env python3
"""Train an amortized posterior (NPE) from pre-generated simulation data.

Usage::

    python scripts/sbi_train_amortizer.py \\
        --data data/sbi/morphine_train.npz \\
        --out  models/sbi/morphine_posterior.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_train")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.sbi.amortizer import save_result, train_npe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "sbi" / "morphine_train.npz"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "models" / "sbi" / "morphine_posterior.pt"
    )
    parser.add_argument("--density-estimator", default="maf", choices=("maf", "nsf"))
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading training data: %s", args.data)
    data = np.load(args.data, allow_pickle=True)
    theta = data["theta"]
    x = data["x"]
    prior_low = data["prior_low"]
    prior_high = data["prior_high"]
    names = tuple(str(n) for n in data["theta_names"])

    log.info("  theta shape=%s names=%s", theta.shape, names)
    log.info("  x shape=%s range=[%.3f, %.3f]", x.shape, float(x.min()), float(x.max()))
    log.info("  prior_low=%s high=%s", prior_low, prior_high)

    t0 = time.time()
    result = train_npe(
        theta=theta,
        x=x,
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
    )
    elapsed = time.time() - t0
    log.info("Training done in %.1fs", elapsed)
    log.info(
        "  train losses: %d epochs (last=%s)",
        len(result.training_losses),
        result.training_losses[-1] if result.training_losses else "n/a",
    )
    log.info(
        "  valid losses: %d epochs (last=%s)",
        len(result.validation_losses),
        result.validation_losses[-1] if result.validation_losses else "n/a",
    )

    save_result(result, args.out)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
