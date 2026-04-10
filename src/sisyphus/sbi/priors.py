"""Prior distributions for SBI over ADME parameters.

Keep priors simple and interpretable for the POC: independent Box-Uniform
priors over log-scale shifts (CLint, Peff) and an absolute value (fup).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# POC theta coordinate system:
#   theta[0] = log10(CLint_actual / CLint_nominal)   ∈ [-1.0, +1.0]  (10x range)
#   theta[1] = fup                                    ∈ [ 0.01, 1.0]  (physical range)
#   theta[2] = log10(Peff_actual / Peff_nominal)      ∈ [-0.5, +0.5]  (3x range)
#
# Note: fup is absolute, not a multiplier, because fup ∈ (0, 1] is a hard
# physical constraint that multiplicative priors don't respect cleanly.
PRIOR_LOW = np.array([-1.0, 0.01, -0.5], dtype=np.float64)
PRIOR_HIGH = np.array([1.0, 1.00, 0.5], dtype=np.float64)
THETA_NAMES = ("log10_clint_shift", "fup", "log10_peff_shift")
N_THETA = 3


@dataclass(frozen=True)
class PriorSpec:
    low: np.ndarray
    high: np.ndarray
    names: tuple[str, ...]

    @property
    def n_dim(self) -> int:
        return len(self.low)

    def as_torch(self) -> "torch.distributions.Distribution":
        from sbi.utils import BoxUniform

        return BoxUniform(
            low=torch.as_tensor(self.low, dtype=torch.float32),
            high=torch.as_tensor(self.high, dtype=torch.float32),
        )

    def sample_numpy(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.n_dim))


def build_box_prior() -> PriorSpec:
    """POC box-uniform prior over (log10_clint_shift, fup, log10_peff_shift)."""
    return PriorSpec(low=PRIOR_LOW.copy(), high=PRIOR_HIGH.copy(), names=THETA_NAMES)
