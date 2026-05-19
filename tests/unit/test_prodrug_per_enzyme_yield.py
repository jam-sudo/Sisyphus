"""Unit tests for per-enzyme prodrug yield (B-04).

See docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
"""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


def _minimal_active(**overrides) -> ActiveMetabolite:
    base = dict(
        name="A",
        mw=200.0,
        fup=Distribution(0.5),
        CL_per_h=Distribution(10.0),
        Vd_L=Distribution(20.0),
        conversion_rate_per_h=Distribution(0.0),
        conversion_site="",
        conversion_yield_fraction=Distribution(1.0),
    )
    base.update(overrides)
    return ActiveMetabolite(**base)


class TestActiveMetaboliteEnzymeYields:
    def test_default_is_empty_dict(self):
        am = _minimal_active()
        assert am.enzyme_yields == {}

    def test_can_set_per_enzyme_yields(self):
        yields = {
            "CES1": Distribution(mean=0.0, cv=0.0),
            "CYP2C19": Distribution(mean=1.0, cv=0.30),
        }
        am = _minimal_active(enzyme_yields=yields)
        assert am.enzyme_yields == yields
