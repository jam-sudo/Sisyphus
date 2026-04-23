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
