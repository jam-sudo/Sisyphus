"""Unit tests for validation/metrics.py and validation/reference.py."""

import numpy as np
import pytest

from sisyphus.validation.metrics import aafe, pct_within_n_fold, pi_coverage


class TestAAFE:
    def test_perfect_prediction(self):
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        assert aafe(pred, obs) == pytest.approx(1.0)

    def test_2fold_overprediction(self):
        pred = np.array([2.0, 4.0, 6.0])
        obs = np.array([1.0, 2.0, 3.0])
        assert aafe(pred, obs) == pytest.approx(2.0)

    def test_2fold_underprediction(self):
        pred = np.array([0.5, 1.0, 1.5])
        obs = np.array([1.0, 2.0, 3.0])
        assert aafe(pred, obs) == pytest.approx(2.0)

    def test_mixed_directions(self):
        """Over and under predictions combine geometrically."""
        pred = np.array([2.0, 0.5])
        obs = np.array([1.0, 1.0])
        # |log10(2)| = 0.301, |log10(0.5)| = 0.301, mean = 0.301
        assert aafe(pred, obs) == pytest.approx(2.0)

    def test_zero_predictions_skipped(self):
        pred = np.array([0.0, 2.0, 4.0])
        obs = np.array([1.0, 2.0, 4.0])
        # Only indices 1, 2 valid
        assert aafe(pred, obs) == pytest.approx(1.0)

    def test_all_zeros_returns_inf(self):
        pred = np.array([0.0, 0.0])
        obs = np.array([1.0, 2.0])
        assert aafe(pred, obs) == float("inf")

    def test_empty_returns_inf(self):
        pred = np.array([])
        obs = np.array([])
        assert aafe(pred, obs) == float("inf")


class TestPctWithinNFold:
    def test_all_within_2fold(self):
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        assert pct_within_n_fold(pred, obs) == pytest.approx(100.0)

    def test_mixed_within_2fold(self):
        pred = np.array([1.0, 3.0, 10.0])
        obs = np.array([1.0, 2.0, 3.0])
        # 1/1=1 (within), 3/2=1.5 (within), 10/3=3.33 (outside)
        assert pct_within_n_fold(pred, obs) == pytest.approx(66.67, abs=1.0)

    def test_none_within(self):
        pred = np.array([100.0])
        obs = np.array([1.0])
        assert pct_within_n_fold(pred, obs) == pytest.approx(0.0)

    def test_custom_fold(self):
        pred = np.array([1.0, 3.0, 10.0])
        obs = np.array([1.0, 1.0, 1.0])
        # n=3: 1/1=1 (in), 3/1=3 (in), 10/1=10 (out)
        assert pct_within_n_fold(pred, obs, n=3.0) == pytest.approx(66.67, abs=1.0)

    def test_empty_returns_zero(self):
        pred = np.array([])
        obs = np.array([])
        assert pct_within_n_fold(pred, obs) == 0.0


class TestPICoverage:
    def test_all_covered(self):
        obs = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        assert pi_coverage(obs, lower, upper) == pytest.approx(1.0)

    def test_partial_coverage(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.5, 1.5, 2.5, 10.0])
        upper = np.array([1.5, 2.5, 3.5, 12.0])
        assert pi_coverage(obs, lower, upper) == pytest.approx(0.75)

    def test_none_covered(self):
        obs = np.array([100.0, 200.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert pi_coverage(obs, lower, upper) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        obs = np.array([])
        lower = np.array([])
        upper = np.array([])
        assert pi_coverage(obs, lower, upper) == 0.0
