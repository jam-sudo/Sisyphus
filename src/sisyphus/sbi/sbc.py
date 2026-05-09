"""Simulation-Based Calibration diagnostics.

SBC is the rigorous validity check for amortized Bayesian inference.
If the amortizer is well-calibrated, then drawing

    theta* ~ prior, x* = simulator(theta*), posterior_samples ~ p(theta | x*)

the rank of theta* within posterior_samples should be uniformly distributed
on [0, n_posterior_samples]. Non-uniformity signals miscalibration.

We report two summaries:
  1. Per-dimension rank histograms (uniformity check via KS test)
  2. Empirical coverage at nominal credible-interval levels (50/80/90/95%)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SBCResult:
    """Output of a SBC run."""

    ranks: np.ndarray  # (n_calibration, d_theta) integer ranks in [0, n_posterior]
    n_posterior_samples: int
    ks_pvalues: np.ndarray  # (d_theta,) KS test p-values vs Uniform
    ks_stats: np.ndarray  # (d_theta,) KS statistics
    coverage: dict[int, np.ndarray]  # nominal level (%) → empirical coverage per dim
    n_calibration: int

    def passes(self, ks_threshold: float = 0.01, coverage_tol: float = 0.1) -> bool:
        """Heuristic gate: all KS p-values above threshold and coverage close.

        ks_threshold: minimum p-value (failure means significant non-uniformity)
        coverage_tol: max absolute deviation from nominal for any level/dim
        """
        if np.any(self.ks_pvalues < ks_threshold):
            return False
        for nominal_pct, emp in self.coverage.items():
            nominal = nominal_pct / 100.0
            if np.any(np.abs(emp - nominal) > coverage_tol):
                return False
        return True

    def summary_str(self) -> str:
        lines = [
            f"SBC summary (n_calibration={self.n_calibration}, n_posterior={self.n_posterior_samples}):",  # noqa: E501
            "  KS test p-values (want > 0.01 for uniformity):",
        ]
        for i, p in enumerate(self.ks_pvalues):
            lines.append(f"    dim {i}: KS={self.ks_stats[i]:.4f}, p={p:.4f}")
        lines.append("  Empirical coverage at nominal levels:")
        for level in sorted(self.coverage):
            emp = self.coverage[level]
            emp_str = ", ".join(f"{v:.3f}" for v in emp)
            lines.append(f"    {level}%: [{emp_str}]  (nominal {level/100:.3f})")
        return "\n".join(lines)


def run_sbc(
    posterior,
    simulator_fn,
    prior_sample_fn,
    n_calibration: int = 500,
    n_posterior_samples: int = 500,
    levels: tuple[int, ...] = (50, 80, 90, 95),
    seed: int = 0,
) -> SBCResult:
    """Run simulation-based calibration.

    Args:
        posterior: trained sbi DirectPosterior (or any object with .sample((n,), x=...))
        simulator_fn: callable theta (d_theta,) -> x (d_x,)   (single theta)
        prior_sample_fn: callable n -> (n, d_theta) numpy
        n_calibration: number of (theta*, x*) calibration draws
        n_posterior_samples: posterior samples per calibration draw
        levels: nominal CI levels to check (%)
    """
    import torch

    rng = np.random.default_rng(seed)  # noqa: F841
    thetas_true = prior_sample_fn(n_calibration)  # (N, d_theta)
    d_theta = thetas_true.shape[1]

    ranks = np.zeros((n_calibration, d_theta), dtype=np.int64)
    coverage_counts: dict[int, np.ndarray] = {
        level: np.zeros(d_theta, dtype=np.int64) for level in levels
    }

    xs = np.zeros((n_calibration, 1), dtype=np.float64)
    for i in range(n_calibration):
        xi = simulator_fn(thetas_true[i])
        xs[i, 0] = xi if np.isfinite(xi) else -20.0

    for i in range(n_calibration):
        x_t = torch.as_tensor(xs[i : i + 1], dtype=torch.float32)
        samples = posterior.sample(
            (n_posterior_samples,), x=x_t, show_progress_bars=False
        )
        samples_np = samples.detach().cpu().numpy()  # (n_post, d_theta)
        # Rank of true theta within posterior samples (per dimension)
        for d in range(d_theta):
            rank = int(np.sum(samples_np[:, d] < thetas_true[i, d]))
            ranks[i, d] = rank
        # Coverage: does true lie within central nominal CI?
        for level in levels:
            lo_q = (100 - level) / 2 / 100.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(samples_np, lo_q, axis=0)
            hi = np.quantile(samples_np, hi_q, axis=0)
            inside = (thetas_true[i] >= lo) & (thetas_true[i] <= hi)
            coverage_counts[level] += inside.astype(np.int64)

    # KS uniformity test per dimension (on normalized ranks in [0, 1])
    ks_stats = np.zeros(d_theta)
    ks_pvalues = np.zeros(d_theta)
    for d in range(d_theta):
        normalized = ranks[:, d] / max(n_posterior_samples, 1)
        res = stats.kstest(normalized, "uniform")
        ks_stats[d] = float(res.statistic)
        ks_pvalues[d] = float(res.pvalue)

    coverage = {
        level: coverage_counts[level].astype(np.float64) / n_calibration
        for level in levels
    }

    return SBCResult(
        ranks=ranks,
        n_posterior_samples=n_posterior_samples,
        ks_pvalues=ks_pvalues,
        ks_stats=ks_stats,
        coverage=coverage,
        n_calibration=n_calibration,
    )
