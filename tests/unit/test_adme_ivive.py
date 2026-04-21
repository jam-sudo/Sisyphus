"""Tests for ADME prediction and IVIVE translation."""

from __future__ import annotations

from sisyphus.core import Distribution
from sisyphus.predict.chemistry import compute_profile

# ── Midazolam SMILES ──────────────────────────────────────────────
_MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2"
_BENZENE_SMILES = "c1ccccc1"
_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
_DIPHENHYDRAMINE_SMILES = "O(CCN(C)C)C(c1ccccc1)c1ccccc1"  # strong base, pKa ~9.8


# ═══════════════════════════════════════════════════════════════════
#  ADME prediction
# ═══════════════════════════════════════════════════════════════════


class TestADME:
    def test_predict_midazolam(self):
        """Midazolam ADME predictions should be in reasonable ranges."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_MIDAZOLAM_SMILES)
        adme = predict_adme(profile)
        assert 0.001 <= adme.fup.mean <= 1.0
        assert adme.clint.mean > 0
        assert adme.peff.mean > 0
        assert adme.rbp.mean > 0

    def test_all_distributions(self):
        """All ADME outputs should be Distributions with cv > 0."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_BENZENE_SMILES)
        adme = predict_adme(profile)
        assert isinstance(adme.fup, Distribution)
        assert isinstance(adme.clint, Distribution)
        assert isinstance(adme.peff, Distribution)
        assert isinstance(adme.solubility, Distribution)
        assert isinstance(adme.vdss, Distribution)
        assert isinstance(adme.rbp, Distribution)
        assert adme.fup.cv > 0
        assert adme.clint.cv > 0

    def test_fup_clamped(self):
        """fup should be clamped to [0.001, 1.0]."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_BENZENE_SMILES)
        adme = predict_adme(profile)
        assert 0.001 <= adme.fup.mean <= 1.0

    def test_peff_from_logp(self):
        """Peff heuristic should give positive values."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_ASPIRIN_SMILES)
        adme = predict_adme(profile)
        assert 0.1 <= adme.peff.mean <= 50.0

    def test_solubility_from_logp(self):
        """Solubility heuristic should give positive values."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_ASPIRIN_SMILES)
        adme = predict_adme(profile)
        assert 0.001 <= adme.solubility.mean <= 100.0

    def test_vdss_positive(self):
        """VDss should be positive."""
        from sisyphus.predict.adme import predict_adme

        profile = compute_profile(_MIDAZOLAM_SMILES)
        adme = predict_adme(profile)
        assert adme.vdss.mean > 0


# ═══════════════════════════════════════════════════════════════════
#  IVIVE / Kp
# ═══════════════════════════════════════════════════════════════════


class TestIVIVE:
    def test_build_drug_on_graph(self):
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_MIDAZOLAM_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=2.0)
        assert drug.dose_mg == 2.0
        assert drug.route == "oral"
        assert len(drug.enzyme_affinity) > 0
        assert len(drug.kp_overrides) > 0
        assert drug.fup.mean == adme.fup.mean

    def test_drug_on_graph_has_kp_for_tissues(self):
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_BENZENE_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=10.0)
        # Should have Kp for major organs
        assert "liver" in drug.kp_overrides
        assert "brain" in drug.kp_overrides
        assert drug.kp_overrides["liver"].mean > 0

    def test_enzyme_affinity_positive(self):
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_ASPIRIN_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=500.0)
        for tag, dist in drug.enzyme_affinity.items():
            assert dist.mean >= 0, f"Negative affinity for {tag}"

    def test_iv_route(self):
        """IV route should set administration_node to venous_blood."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_BENZENE_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=10.0, route="iv")
        assert drug.administration_node == "venous_blood"
        assert drug.route == "iv"

    def test_oral_route(self):
        """Oral route should set administration_node to stomach_lumen."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_BENZENE_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=10.0, route="oral")
        assert drug.administration_node == "stomach_lumen"
        assert drug.route == "oral"

    def test_kp_values_reasonable(self):
        """Kp values should be in physiological range."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_MIDAZOLAM_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=2.0)
        for name, kp in drug.kp_overrides.items():
            assert 0.01 <= kp.mean <= 200.0, f"Kp for {name} = {kp.mean} out of range"

    def test_renal_clearance_non_negative(self):
        """Renal clearance should be non-negative."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_MIDAZOLAM_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=2.0)
        assert drug.renal_clearance.mean >= 0

    def test_compound_type_preserved(self):
        """compound_type should pass through from MolecularProfile."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile(_ASPIRIN_SMILES)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose_mg=500.0)
        assert drug.compound_type == profile.compound_type

    def test_default_kp_method_is_rodgers_rowland(self):
        """Default kp_method should be 'rodgers_rowland'.

        Berezhkovskiy was tested (2026-03-26) but reverted — broke error
        cancellation, Meta AAFE worsened 2.058→2.067.
        """
        import inspect

        from sisyphus.predict.ivive import build_drug_on_graph

        sig = inspect.signature(build_drug_on_graph)
        default = sig.parameters["kp_method"].default
        assert default == "rodgers_rowland"

    def test_base_increases_cyp2d6_fraction(self):
        """Bases should have higher CYP2D6 fraction than neutrals."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        # Diphenhydramine is a strong base (tertiary amine, pKa ~9.8)
        profile_base = compute_profile(_DIPHENHYDRAMINE_SMILES)
        adme_base = predict_adme(profile_base)
        drug_base = build_drug_on_graph(profile_base, adme_base, dose_mg=2.0)

        # Benzene is neutral
        profile_neutral = compute_profile(_BENZENE_SMILES)
        adme_neutral = predict_adme(profile_neutral)
        drug_neutral = build_drug_on_graph(profile_neutral, adme_neutral, dose_mg=2.0)

        # If both have CYP2D6, base should have higher fraction relative to total
        if "CYP2D6" in drug_base.enzyme_affinity and "CYP2D6" in drug_neutral.enzyme_affinity:
            # The fraction allocated to CYP2D6 should be higher for bases
            base_total = sum(d.mean for d in drug_base.enzyme_affinity.values())
            neutral_total = sum(d.mean for d in drug_neutral.enzyme_affinity.values())
            if base_total > 0 and neutral_total > 0:
                base_frac = drug_base.enzyme_affinity["CYP2D6"].mean / base_total
                neutral_frac = drug_neutral.enzyme_affinity["CYP2D6"].mean / neutral_total
                assert base_frac > neutral_frac


# ═══════════════════════════════════════════════════════════════════
#  Berezhkovskiy correction
# ═══════════════════════════════════════════════════════════════════


class TestBerezhkovskiyCorrection:
    def test_bz_reduces_kp_for_low_fup(self):
        """BZ correction halves Kp for highly bound drugs (fup=0.01)."""
        from sisyphus.predict.ivive import _apply_bz_correction

        kp_rr = 100.0
        kp_bz = _apply_bz_correction(kp_rr, fup=0.01)
        assert kp_bz < kp_rr
        assert abs(kp_bz - 50.25) < 1.0

    def test_bz_minimal_for_high_fup(self):
        """BZ correction still reduces Kp even at high fup."""
        from sisyphus.predict.ivive import _apply_bz_correction

        kp_rr = 5.0
        kp_bz = _apply_bz_correction(kp_rr, fup=0.9)
        assert kp_bz < kp_rr

    def test_bz_kp_one_unchanged(self):
        """Kp=1.0 is a fixed point of BZ correction."""
        from sisyphus.predict.ivive import _apply_bz_correction

        assert _apply_bz_correction(1.0, 0.5) == 1.0


# ═══════════════════════════════════════════════════════════════════
#  ECM parameters
# ═══════════════════════════════════════════════════════════════════


class TestECMParameters:
    def test_build_drug_on_graph_applies_hepatic_ecm_params(self):
        """When hepatic_ecm_params kwarg is provided, ECM fields override defaults."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.ivive import build_drug_on_graph

        profile = compute_profile("CCO")
        adme = predict_adme(profile)

        # Without kwarg → defaults
        drug_default = build_drug_on_graph(profile, adme, dose_mg=100.0, route="iv")
        assert drug_default.ps_passive.mean == 1e6
        assert drug_default.cl_int_bile.mean == 0.0

        # With kwarg → overridden
        custom = {
            "ps_passive": Distribution(0.8, cv=0.4),
            "ps_eff": Distribution(0.8, cv=0.4),
            "cl_int_bile": Distribution(45.0, cv=0.5),
        }
        drug_custom = build_drug_on_graph(
            profile, adme, dose_mg=100.0, route="iv",
            hepatic_ecm_params=custom,
        )
        assert drug_custom.ps_passive.mean == 0.8
        assert drug_custom.ps_eff.mean == 0.8
        assert drug_custom.cl_int_bile.mean == 45.0
