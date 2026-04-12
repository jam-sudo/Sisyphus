"""Prior distributions for SBI over ADME parameters.

Keep priors simple and interpretable for the POC: independent Box-Uniform
priors over log-scale shifts (CLint, Peff) and logit-transformed fup.

Phase 2.0.5 change (2026-04-12): theta[1] was reparametrized from
absolute fup ∈ [0.01, 1.0] to logit(fup) ∈ [−4.595, 4.595]. This gives
uniform coverage in logit space, preventing the prior from concentrating
mass on high-fup drugs and underrepresenting low-fup acids/statins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# --- logit / sigmoid helpers ---
def _logit(p: float) -> float:
    """log(p / (1 - p)), clipped to avoid ±inf."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """1 / (1 + exp(-x)), numerically stable."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# Theta coordinate system (Phase 2.0.5):
#   theta[0] = log10(CLint_actual / CLint_nominal)   ∈ [-1.0, +1.0]  (10x range)
#   theta[1] = logit(fup)                             ∈ [logit(0.01), logit(0.99)]
#              ≈ [-4.595, +4.595]
#              Uniform in logit space → non-uniform in fup space, better
#              resolution at low fup where acids/statins cluster.
#   theta[2] = log10(Peff_actual / Peff_nominal)      ∈ [-0.5, +0.5]  (3x range)
#
# To recover physical fup from theta[1]: fup = sigmoid(theta[1]).
_LOGIT_FUP_LOW = _logit(0.01)    # ≈ -4.595
_LOGIT_FUP_HIGH = _logit(0.99)   # ≈ +4.595

PRIOR_LOW = np.array([-1.0, _LOGIT_FUP_LOW, -0.5], dtype=np.float64)
PRIOR_HIGH = np.array([1.0, _LOGIT_FUP_HIGH, 0.5], dtype=np.float64)
THETA_NAMES = ("log10_clint_shift", "logit_fup", "log10_peff_shift")
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
    """Box-uniform prior over (log10_clint_shift, logit_fup, log10_peff_shift).

    Phase 2.0.5: theta[1] is logit(fup), not absolute fup.
    Use ``sigmoid(theta[1])`` to recover fup in [0, 1].
    """
    return PriorSpec(low=PRIOR_LOW.copy(), high=PRIOR_HIGH.copy(), names=THETA_NAMES)
