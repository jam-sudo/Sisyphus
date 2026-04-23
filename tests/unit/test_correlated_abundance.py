"""Unit tests for Distribution.correlation_group and correlated sampling."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.graph.builder import _parse_distribution
from sisyphus.physiology.correlation_registry import (
    CorrelationSpec,
    _REGISTRY,
    get,
    load_from_json,
    register,
)


class TestDistributionCorrelationGroup:
    """Tests for the correlation_group field added in Task 1."""

    def test_default_correlation_group_is_none(self) -> None:
        d = Distribution(mean=100.0, cv=0.1)
        assert d.correlation_group is None

    def test_correlation_group_can_be_set(self) -> None:
        d = Distribution(mean=100.0, cv=0.1, correlation_group="liver_achour2021")
        assert d.correlation_group == "liver_achour2021"

    def test_correlation_group_difference_breaks_equality(self) -> None:
        """Distributions with different groups are not equal (frozen dataclass __eq__)."""
        a = Distribution(mean=100.0, cv=0.1)
        b = Distribution(mean=100.0, cv=0.1, correlation_group="g1")
        assert a != b


class TestParseDistribution:
    """YAML → Distribution parsing, incl. correlation_group."""

    def test_bare_scalar_produces_none_group(self) -> None:
        d = _parse_distribution(9247500)
        assert d.mean == 9247500
        assert d.cv == 0.0
        assert d.correlation_group is None

    def test_dict_without_group_defaults_none(self) -> None:
        d = _parse_distribution({"mean": 9247500, "cv": 0.763})
        assert d.mean == 9247500
        assert d.cv == 0.763
        assert d.correlation_group is None

    def test_dict_with_group_stored(self) -> None:
        d = _parse_distribution(
            {"mean": 9247500, "cv": 0.763, "correlation_group": "liver_achour2021"}
        )
        assert d.correlation_group == "liver_achour2021"

    def test_float_scalar(self) -> None:
        d = _parse_distribution(3.14)
        assert d.mean == 3.14
        assert d.cv == 0.0
        assert d.correlation_group is None


class TestRegistry:
    """Tests for the correlation_registry module."""

    def setup_method(self) -> None:
        _REGISTRY.clear()

    def test_register_and_get(self) -> None:
        spec = CorrelationSpec(
            members=("A", "B"),
            log_corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            cvs=np.array([0.3, 0.4]),
        )
        register("test_group", spec)
        got = get("test_group")
        assert got is spec

    def test_get_missing_returns_none(self) -> None:
        assert get("nonexistent") is None

    def test_load_from_json_achour(self, tmp_path: pathlib.Path) -> None:
        j = {
            "name": "tg",
            "members": ["A", "B"],
            "cv": [0.3, 0.4],
            "log_corr_matrix": [[1.0, 0.5], [0.5, 1.0]],
        }
        p = tmp_path / "tg.json"
        p.write_text(json.dumps(j))
        load_from_json(p)
        got = get("tg")
        assert got is not None
        assert got.members == ("A", "B")
        assert np.allclose(got.log_corr_matrix, [[1.0, 0.5], [0.5, 1.0]])
        assert np.allclose(got.cvs, [0.3, 0.4])

    def test_load_real_achour_file_populates_registry(self) -> None:
        """The committed data/physiology/achour2021_correlation.json loads
        and registers under its declared name."""
        p = pathlib.Path(__file__).resolve().parents[2] / "data" / "physiology" / "achour2021_correlation.json"
        load_from_json(p)
        got = get("liver_achour2021")
        assert got is not None
        assert len(got.members) >= 5


from sisyphus.physiology.correlation_registry import sample_correlated


class TestSampleCorrelated:
    """Gates B and C: marginal CV + joint correlation fidelity."""

    def test_marginals_match_cv_independent(self) -> None:
        """With identity correlation, each marginal matches its own lognormal CV."""
        rng = np.random.default_rng(42)
        means = np.array([100.0, 50.0, 10.0])
        cvs = np.array([0.5, 0.3, 0.1])
        log_corr = np.eye(3)
        samples = np.array(
            [sample_correlated(means, cvs, log_corr, rng) for _ in range(10_000)]
        )
        emp_mean = samples.mean(axis=0)
        emp_cv = samples.std(axis=0, ddof=1) / emp_mean
        # Gate B tolerances: ±1% mean, ±5% relative CV
        assert np.allclose(emp_mean, means, rtol=0.02)
        for ec, c in zip(emp_cv, cvs):
            assert abs(ec - c) / c < 0.05

    def test_recovers_log_corr_matrix(self) -> None:
        """Empirical log-space correlation matches the input matrix (Gate C)."""
        rng = np.random.default_rng(1234)
        means = np.array([100.0, 50.0, 10.0])
        cvs = np.array([0.5, 0.5, 0.5])
        target = np.array(
            [[1.0, 0.6, 0.3],
             [0.6, 1.0, 0.2],
             [0.3, 0.2, 1.0]]
        )
        samples = np.array(
            [sample_correlated(means, cvs, target, rng) for _ in range(20_000)]
        )
        emp_log_corr = np.corrcoef(np.log(samples), rowvar=False)
        # Gate C tolerance: ±0.05 off-diagonal
        assert np.allclose(emp_log_corr, target, atol=0.05)

    def test_all_samples_positive(self) -> None:
        rng = np.random.default_rng(7)
        means = np.array([100.0, 50.0])
        cvs = np.array([1.0, 0.8])
        log_corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        for _ in range(1000):
            s = sample_correlated(means, cvs, log_corr, rng)
            assert (s > 0).all()

    def test_degenerate_identity_matches_independent(self) -> None:
        """log_corr=I produces samples with ~zero empirical cross-correlation."""
        rng = np.random.default_rng(99)
        means = np.array([100.0, 100.0])
        cvs = np.array([0.5, 0.5])
        samples = np.array(
            [sample_correlated(means, cvs, np.eye(2), rng) for _ in range(10_000)]
        )
        emp_corr = np.corrcoef(np.log(samples), rowvar=False)[0, 1]
        assert abs(emp_corr) < 0.05

    def test_healthy_proxy_gate_Bprime(self) -> None:
        """Gate B': 0.5× CV configuration still reproduces marginals."""
        rng = np.random.default_rng(2026)
        means = np.array([100.0, 50.0])
        cvs = np.array([0.763, 0.484]) * 0.5  # healthy proxy
        log_corr = np.eye(2)
        samples = np.array(
            [sample_correlated(means, cvs, log_corr, rng) for _ in range(10_000)]
        )
        emp_cv = samples.std(axis=0, ddof=1) / samples.mean(axis=0)
        for ec, c in zip(emp_cv, cvs):
            assert abs(ec - c) / c < 0.05
