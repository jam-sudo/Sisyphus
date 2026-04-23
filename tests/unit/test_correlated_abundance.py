"""Unit tests for Distribution.correlation_group and correlated sampling."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution


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
