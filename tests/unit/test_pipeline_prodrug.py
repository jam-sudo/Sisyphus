"""Tests for pipeline-level prodrug routing helpers."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


def _bh4():
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )


def _drug_with(active=None, obs_species="parent"):
    """Reuse the shared helper from test_active_metabolite."""
    from tests.unit.test_active_metabolite import _minimal_drug
    return _minimal_drug(active=active, obs_species=obs_species)


def test_resolve_observation_node_active():
    """observation_species='active' + active_metabolite set -> '{obs}_active'."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with(active=_bh4(), obs_species="active")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood_active"


def test_resolve_observation_node_parent_default():
    """observation_species='parent' (default) -> base node."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with(active=None, obs_species="parent")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood"


def test_resolve_observation_node_parent_override_with_active():
    """observation_species='parent' (override) + active_metabolite -> still base."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with(active=_bh4(), obs_species="parent")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood"


def test_adjust_ad_prodrug_with_active_upgrades():
    """PRODRUG flag + active_metabolite set -> in_domain=True + warning."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with(active=_bh4(), obs_species="active")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=["PRODRUG"])
    assert in_domain is True
    assert any("Prodrug" in w for w in warnings)


def test_adjust_ad_prodrug_no_active_remains_out():
    """PRODRUG flag + no active_metabolite -> in_domain=False."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with(active=None, obs_species="parent")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=["PRODRUG"])
    assert in_domain is False
    assert warnings == []


def test_adjust_ad_no_prodrug_with_active_warns_non_structural():
    """No PRODRUG flag + active_metabolite set -> in_domain=True + non-structural warning."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with(active=_bh4(), obs_species="active")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=[])
    assert in_domain is True
    assert any("non-structural" in w.lower() or "without structural" in w.lower()
               for w in warnings)


def test_adjust_ad_no_flags_clean_drug():
    """No flags, no active -> in_domain=True, no warnings."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with(active=None, obs_species="parent")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=[])
    assert in_domain is True
    assert warnings == []


def test_adjust_ad_with_other_flags_keeps_them():
    """Other AD flags besides PRODRUG are preserved in domain judgment."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with(active=_bh4(), obs_species="active")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=["PRODRUG", "HIGH_MW"])
    # HIGH_MW remains -> in_domain should still be False
    assert in_domain is False
