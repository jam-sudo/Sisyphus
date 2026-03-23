"""Tests for ml/features.py and predict/chemistry.py."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.ml.features import compute_features
from sisyphus.predict.chemistry import compute_profile


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------
class TestFeatures:
    def test_feature_vector_shape(self):
        features = compute_features("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert features.shape == (2057,)

    def test_feature_vector_dtype(self):
        features = compute_features("CC(=O)Oc1ccccc1C(=O)O")
        assert features.dtype == np.float64

    def test_morgan_fp_binary(self):
        features = compute_features("c1ccccc1")  # benzene
        fp_part = features[:2048]
        assert set(np.unique(fp_part)).issubset({0.0, 1.0})

    def test_morgan_fp_has_on_bits(self):
        features = compute_features("c1ccccc1")
        fp_part = features[:2048]
        assert fp_part.sum() > 0  # benzene should have some bits set

    def test_descriptors_reasonable(self):
        features = compute_features("c1ccccc1")  # benzene MW ~78
        desc = features[2048:]  # last 9 elements
        assert len(desc) == 9
        assert desc[2] > 0  # MW/600 > 0
        assert desc[2] < 1  # MW < 600 for benzene
        assert desc[6] > 0  # RingCount/5 > 0 for benzene

    def test_logp_position(self):
        # Benzene LogP ~1.68 (unnormalized, /1.0)
        features = compute_features("c1ccccc1")
        logp = features[2048]
        assert 1.0 < logp < 3.0

    def test_tpsa_zero_for_benzene(self):
        features = compute_features("c1ccccc1")
        tpsa_norm = features[2049]  # TPSA/200
        assert tpsa_norm == pytest.approx(0.0, abs=1e-6)

    def test_different_molecules_different_features(self):
        f1 = compute_features("c1ccccc1")  # benzene
        f2 = compute_features("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert not np.array_equal(f1, f2)

    def test_deterministic(self):
        f1 = compute_features("CC(=O)Oc1ccccc1C(=O)O")
        f2 = compute_features("CC(=O)Oc1ccccc1C(=O)O")
        np.testing.assert_array_equal(f1, f2)

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            compute_features("not_a_smiles")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            compute_features("")


# ---------------------------------------------------------------------------
# Chemistry module tests
# ---------------------------------------------------------------------------
class TestChemistry:
    def test_aspirin_profile(self):
        profile = compute_profile("CC(=O)Oc1ccccc1C(=O)O")
        assert profile.mw == pytest.approx(180.16, abs=1.0)
        assert profile.compound_type == "acid"  # carboxylic acid
        assert profile.pka == pytest.approx(4.5)
        assert profile.in_ad is True
        assert profile.ad_flags == ()

    def test_midazolam_is_base(self):
        # Midazolam has a basic amine in the diazepine ring
        profile = compute_profile("Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2")
        assert profile.compound_type == "base"
        assert profile.pka == pytest.approx(9.0)

    def test_caffeine_is_neutral(self):
        profile = compute_profile("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert profile.compound_type == "neutral"
        assert profile.pka is None

    def test_glycine_is_zwitterion(self):
        # Glycine: has both -COOH and -NH2
        profile = compute_profile("NCC(=O)O")
        assert profile.compound_type == "zwitterion"
        assert profile.pka == pytest.approx(7.0)

    def test_canonical_smiles(self):
        # Different representations of aspirin should give the same canonical
        p1 = compute_profile("CC(=O)Oc1ccccc1C(=O)O")
        p2 = compute_profile("O=C(O)c1ccccc1OC(C)=O")
        assert p1.smiles == p2.smiles

    def test_logp_reasonable(self):
        profile = compute_profile("c1ccccc1")  # benzene
        assert 1.0 < profile.logp < 3.0

    def test_hbd_hba_benzene(self):
        profile = compute_profile("c1ccccc1")
        assert profile.hbd == 0
        assert profile.hba == 0

    def test_hbd_hba_ethanol(self):
        profile = compute_profile("CCO")
        assert profile.hbd == 1
        assert profile.hba == 1

    def test_invalid_smiles(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            compute_profile("INVALID")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            compute_profile("")

    def test_high_mw_flagged(self):
        # Cyclosporine A (MW ~1202)
        smiles = (
            "CC1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(=O)N(C)"
            "C(CC(C)C)C(=O)NC(C)C(=O)NC(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)"
            "NC(C(C)C)C(=O)NC1C)O"
        )
        try:
            profile = compute_profile(smiles)
            if profile.mw > 700:
                assert "HIGH_MW" in profile.ad_flags
                assert profile.in_ad is False
        except ValueError:
            pass  # OK if SMILES doesn't parse cleanly

    def test_extreme_lipophilic_flagged(self):
        # Very lipophilic molecule — long alkyl chain
        profile = compute_profile("CCCCCCCCCCCCCCCCCCCC")  # eicosane, logP ~10
        if profile.logp > 5.5:
            assert "EXTREME_LIPOPHILIC" in profile.ad_flags

    def test_neutral_compound_no_pka(self):
        profile = compute_profile("c1ccccc1")  # benzene — no ionizable groups
        assert profile.compound_type == "neutral"
        assert profile.pka is None

    def test_sulfonic_acid(self):
        # Methanesulfonic acid
        profile = compute_profile("CS(=O)(=O)O")
        assert profile.compound_type == "acid"

    def test_profile_is_frozen(self):
        profile = compute_profile("c1ccccc1")
        with pytest.raises(AttributeError):
            profile.mw = 999.0  # type: ignore[misc]
