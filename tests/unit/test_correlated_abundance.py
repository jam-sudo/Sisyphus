"""Unit tests for Distribution.correlation_group and correlated sampling."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.graph.builder import _parse_distribution


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
