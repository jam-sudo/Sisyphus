"""Tests for ActiveMetabolite dataclass."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


def test_active_metabolite_minimal_construction():
    """Required fields produce a valid frozen ActiveMetabolite."""
    am = ActiveMetabolite(
        name="BH4",
        mw=241.25,
        fup=Distribution(mean=0.23, cv=0.3),
        CL_per_h=Distribution(mean=40.0, cv=0.35),
        Vd_L=Distribution(mean=150.0, cv=0.3),
        conversion_rate_per_h=Distribution(mean=12.0, cv=0.4),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(mean=0.85, cv=0.1),
    )
    assert am.name == "BH4"
    assert am.mw == 241.25
    assert am.conversion_site == "gut_wall"


def test_active_metabolite_is_frozen():
    """ActiveMetabolite must be frozen (immutable)."""
    am = ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )
    with pytest.raises((AttributeError, Exception)):
        am.name = "different"  # type: ignore[misc]
