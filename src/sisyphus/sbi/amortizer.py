"""Amortized posterior training via Neural Posterior Estimation (NPE).

Thin wrapper over the ``sbi`` library's SNPE so the rest of the codebase
doesn't need to import torch directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NPETrainingResult:
    posterior: Any  # sbi DirectPosterior
    density_estimator: Any  # neural network
    training_losses: list[float]
    validation_losses: list[float]
    n_train_samples: int
    prior_low: np.ndarray
    prior_high: np.ndarray

    def sample(self, x_obs: np.ndarray, n: int = 5000) -> np.ndarray:
        """Draw n samples from posterior conditioned on observed x."""
        import torch

        x_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t[None, :]
        samples = self.posterior.sample((n,), x=x_t, show_progress_bars=False)
        return samples.detach().cpu().numpy()


def train_npe(
    theta: np.ndarray,
    x: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    density_estimator: str = "maf",
    hidden_features: int = 50,
    num_transforms: int = 5,
    training_batch_size: int = 256,
    learning_rate: float = 5e-4,
    max_num_epochs: int = 200,
    validation_fraction: float = 0.15,
    stop_after_epochs: int = 20,
    seed: int = 42,
) -> NPETrainingResult:
    """Train a Neural Posterior Estimator on (theta, x) pairs.

    Args:
        theta: (N, d_theta) float32 array
        x:     (N, d_x) float32 array
        prior_low/high: bounds for BoxUniform prior (same as data generation)
        density_estimator: 'maf' (Masked Autoregressive Flow) or 'nsf'
        ... sbi training hyperparameters
    """
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    np.random.seed(seed)

    prior_torch = BoxUniform(
        low=torch.as_tensor(prior_low, dtype=torch.float32),
        high=torch.as_tensor(prior_high, dtype=torch.float32),
    )

    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)

    logger.info(
        "Training NPE on N=%d, theta_dim=%d, x_dim=%d, estimator=%s",
        theta_t.shape[0],
        theta_t.shape[1],
        x_t.shape[1],
        density_estimator,
    )

    inference = SNPE(
        prior=prior_torch,
        density_estimator=density_estimator,
        show_progress_bars=True,
    )
    inference.append_simulations(theta_t, x_t)

    density_net = inference.train(
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        max_num_epochs=max_num_epochs,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
    )

    posterior = inference.build_posterior(density_net)

    # sbi exposes training history via inference._summary
    summary = getattr(inference, "_summary", {}) or {}
    train_losses = list(summary.get("training_loss", [])) or []
    valid_losses = list(summary.get("validation_loss", [])) or []

    return NPETrainingResult(
        posterior=posterior,
        density_estimator=density_net,
        training_losses=[float(x) for x in train_losses],
        validation_losses=[float(x) for x in valid_losses],
        n_train_samples=int(theta_t.shape[0]),
        prior_low=np.asarray(prior_low, dtype=np.float64).copy(),
        prior_high=np.asarray(prior_high, dtype=np.float64).copy(),
    )


def save_result(result: NPETrainingResult, path: Path) -> None:
    """Save posterior + metadata. Uses torch for the density estimator."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "density_estimator": result.density_estimator,
        "training_losses": result.training_losses,
        "validation_losses": result.validation_losses,
        "n_train_samples": result.n_train_samples,
        "prior_low": result.prior_low.tolist(),
        "prior_high": result.prior_high.tolist(),
    }
    torch.save(payload, path)


def load_result(path: Path) -> NPETrainingResult:
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    payload = torch.load(path, weights_only=False)
    prior_low = np.asarray(payload["prior_low"], dtype=np.float64)
    prior_high = np.asarray(payload["prior_high"], dtype=np.float64)
    prior_torch = BoxUniform(
        low=torch.as_tensor(prior_low, dtype=torch.float32),
        high=torch.as_tensor(prior_high, dtype=torch.float32),
    )
    inference = SNPE(prior=prior_torch)
    posterior = inference.build_posterior(payload["density_estimator"])
    return NPETrainingResult(
        posterior=posterior,
        density_estimator=payload["density_estimator"],
        training_losses=payload.get("training_losses", []),
        validation_losses=payload.get("validation_losses", []),
        n_train_samples=int(payload.get("n_train_samples", 0)),
        prior_low=prior_low,
        prior_high=prior_high,
    )
