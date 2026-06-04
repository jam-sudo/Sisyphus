"""Split-conformal prediction intervals on log-Cmax (spec 2026-06-04-conformal-...).

Conformal gives valid marginal coverage regardless of base-model structural error
(the bias is absorbed into a wider interval) — the fix for the 29.9%@90% MC
under-coverage. These tests pin the finite-sample guarantee and the multiplicative
log-space interval.
"""

import numpy as np

from sisyphus.validation.conformal import (
    conformal_interval,
    conformal_quantile,
    empirical_coverage,
    nonconformity_scores,
)


class TestConformalQuantile:
    def test_finite_sample_rank(self):
        """q is the ceil((n+1)(1-alpha))/n empirical quantile (1-indexed rank)."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # n=10
        # alpha=0.1 -> rank = ceil(11*0.9) = ceil(9.9) = 10 -> scores[10-1] = 1.0
        assert conformal_quantile(scores, alpha=0.1) == 1.0
        # alpha=0.2 -> rank = ceil(11*0.8) = ceil(8.8) = 9 -> scores[8] = 0.9
        assert conformal_quantile(scores, alpha=0.2) == 0.9

    def test_unbounded_when_too_few_calibration_points(self):
        """If the required rank exceeds n, the interval is unbounded (inf)."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # n=5
        # alpha=0.1 -> rank = ceil(6*0.9) = ceil(5.4) = 6 > 5 -> inf
        assert conformal_quantile(scores, alpha=0.1) == np.inf

    def test_unsorted_input_ok(self):
        scores = np.array([1.0, 0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4, 0.6])
        assert conformal_quantile(scores, alpha=0.1) == 1.0


class TestConformalInterval:
    def test_multiplicative_log_space(self):
        lo, hi = conformal_interval(pred=2.0, q=np.log10(8.2))
        assert abs(lo - 2.0 / 8.2) < 1e-9
        assert abs(hi - 2.0 * 8.2) < 1e-9

    def test_infinite_q_gives_unbounded(self):
        lo, hi = conformal_interval(pred=2.0, q=np.inf)
        assert lo == 0.0 and hi == np.inf


class TestNonconformity:
    def test_absolute_log_fold(self):
        s = nonconformity_scores(pred=[10.0, 1.0], obs=[1.0, 10.0])
        assert np.allclose(s, [1.0, 1.0])  # |log10(10/1)| = 1


class TestMarginalCoverageGuarantee:
    def test_split_conformal_achieves_nominal_coverage(self):
        """On a fresh exchangeable test set, split-conformal coverage >= 1-alpha.

        This is the property that fixes the 30%-at-90% under-coverage: it holds
        even when the base 'model' has large, structural (non-Gaussian) error.
        """
        rng = np.random.default_rng(0)
        n = 4000
        # Heavy-tailed, biased log-residuals (structural-error analogue, not just noise)
        resid = rng.standard_t(df=3, size=n) * 0.4 + 0.2  # biased + heavy tail
        obs = np.full(n, 1.0)
        pred = obs * (10.0 ** resid)  # pred is biased & dispersed vs obs
        cal = slice(0, 2000)
        test = slice(2000, n)
        scores = nonconformity_scores(pred[cal], obs[cal])
        q = conformal_quantile(scores, alpha=0.1)
        cov = empirical_coverage(pred[test], obs[test], q)
        # Marginal guarantee: >= 0.90 (allow a small finite-sample slack downward)
        assert cov >= 0.88, f"conformal 90% coverage was {cov:.3f}, expected ~0.90"
        # And it should not be absurdly over-conservative
        assert cov <= 0.97, f"coverage {cov:.3f} unexpectedly conservative"
