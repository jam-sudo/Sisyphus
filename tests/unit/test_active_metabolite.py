"""Tests for ActiveMetabolite dataclass."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


@pytest.fixture
def bh4_active():
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(mean=0.23, cv=0.3),
        CL_per_h=Distribution(mean=40.0, cv=0.35),
        Vd_L=Distribution(mean=150.0, cv=0.3),
        conversion_rate_per_h=Distribution(mean=12.0, cv=0.4),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(mean=0.85, cv=0.1),
    )


def test_active_metabolite_minimal_construction(bh4_active):
    """Required fields produce a valid frozen ActiveMetabolite."""
    assert bh4_active.name == "BH4"
    assert bh4_active.mw == 241.25
    assert bh4_active.conversion_site == "gut_wall"


def test_active_metabolite_is_frozen(bh4_active):
    """ActiveMetabolite must be frozen (immutable)."""
    with pytest.raises(AttributeError):
        bh4_active.name = "different"  # type: ignore[misc]
