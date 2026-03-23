import numpy as np
import pytest

from sisyphus.core import Distribution


class TestDistribution:
    def test_deterministic_returns_mean(self):
        d = Distribution(mean=5.0, cv=0.0)
        rng = np.random.default_rng(42)
        assert d.sample(rng) == 5.0

    def test_lognormal_positive(self):
        d = Distribution(mean=5.0, cv=0.3, dist_type="lognormal")
        rng = np.random.default_rng(42)
        samples = [d.sample(rng) for _ in range(1000)]
        assert all(s > 0 for s in samples)
        assert abs(np.mean(samples) - 5.0) / 5.0 < 0.1  # within 10% of mean

    def test_normal_sampling(self):
        d = Distribution(mean=100.0, cv=0.1, dist_type="normal")
        rng = np.random.default_rng(42)
        samples = [d.sample(rng) for _ in range(1000)]
        assert abs(np.mean(samples) - 100.0) < 5.0
        assert abs(np.std(samples) - 10.0) < 3.0

    def test_std_property(self):
        d = Distribution(mean=100.0, cv=0.1)
        assert d.std == pytest.approx(10.0)

    def test_frozen(self):
        d = Distribution(mean=1.0)
        with pytest.raises(AttributeError):
            d.mean = 2.0

    def test_negative_cv_raises(self):
        with pytest.raises(ValueError):
            Distribution(mean=1.0, cv=-0.1)

    def test_invalid_dist_type_raises(self):
        with pytest.raises(ValueError):
            Distribution(mean=1.0, dist_type="banana")

    def test_uniform_sampling(self):
        d = Distribution(mean=10.0, cv=0.2, dist_type="uniform")
        rng = np.random.default_rng(42)
        samples = [d.sample(rng) for _ in range(1000)]
        assert all(s > 0 for s in samples)  # mean=10, cv=0.2, std=2, range ~6.5-13.5
        assert abs(np.mean(samples) - 10.0) < 1.0

    def test_zero_mean_deterministic(self):
        d = Distribution(mean=0.0, cv=0.0)
        rng = np.random.default_rng(42)
        assert d.sample(rng) == 0.0
