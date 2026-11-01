"""Tests for the pH-7.4 ionization block in ``descriptors.compute_features_v2``."""

from __future__ import annotations

import numpy as np

from sisyphus.descriptors import (
    IONIZATION_FEATURE_NAMES,
    N_IONIZATION_FEATURES,
    compute_features,
    compute_features_v2,
)

_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
_IMIPRAMINE = "CN(C)CCCN1c2ccccc2CCc2ccccc21"
_CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
_CETIRIZINE = "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1"
_DICLOFENAC = "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"
_LOSARTAN = "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nnn[nH]1"


def _block(smiles: str) -> dict[str, float]:
    vec = compute_features_v2(smiles)[-N_IONIZATION_FEATURES:]
    return dict(zip(IONIZATION_FEATURE_NAMES, vec.tolist()))


class TestShape:
    def test_length_is_v1_plus_block(self):
        assert len(compute_features_v2(_ASPIRIN)) == 2057 + N_IONIZATION_FEATURES

    def test_prefix_is_bit_identical_to_v1(self):
        v1 = compute_features(_IMIPRAMINE)
        v2 = compute_features_v2(_IMIPRAMINE)
        assert np.array_equal(v1, v2[: len(v1)])

    def test_dtype(self):
        assert compute_features_v2(_ASPIRIN).dtype == np.float64

    def test_names_match_width(self):
        assert len(IONIZATION_FEATURE_NAMES) == N_IONIZATION_FEATURES


class TestChargeClass:
    def test_carboxylic_acid_is_net_anion(self):
        b = _block(_IBUPROFEN)
        assert b["n_cooh"] == 1
        assert b["has_strong_acid"] == 1
        assert b["net_anion"] == 1
        assert b["net_cation"] == 0
        assert b["is_zwitterion"] == 0

    def test_aliphatic_amine_is_net_cation(self):
        b = _block(_IMIPRAMINE)
        assert b["n_aliph_amine"] == 1
        assert b["has_strong_base"] == 1
        assert b["net_cation"] == 1
        assert b["net_anion"] == 0

    def test_acid_plus_aliphatic_amine_is_zwitterion(self):
        b = _block(_CETIRIZINE)
        assert b["has_strong_acid"] == 1
        assert b["has_strong_base"] == 1
        assert b["is_zwitterion"] == 1
        assert b["net_anion"] == 0
        assert b["net_cation"] == 0

    def test_nonionizable(self):
        b = _block(_CAFFEINE)
        assert b["has_strong_acid"] == 0
        assert b["has_strong_base"] == 0
        assert b["is_nonionizable"] == 1

    def test_classes_are_mutually_exclusive_and_exhaustive(self):
        for smiles in (_ASPIRIN, _IBUPROFEN, _IMIPRAMINE, _CAFFEINE,
                       _CETIRIZINE, _DICLOFENAC, _LOSARTAN):
            b = _block(smiles)
            total = (b["net_anion"] + b["net_cation"]
                     + b["is_zwitterion"] + b["is_nonionizable"])
            assert total == 1, smiles


class TestArylAmineIsNotBasic:
    """An aryl amine is un-protonated at pH 7.4 and must not read as basic.

    Aniline pKaH 4.6, diphenylamine pKaH 0.8 vs aliphatic amine pKaH 9-11
    (Perrin 1965).  Diclofenac's only nitrogen is a diarylamine, so it is a
    carboxylic acid, not a zwitterion.
    """

    def test_diclofenac_is_an_acid_not_a_zwitterion(self):
        b = _block(_DICLOFENAC)
        assert b["n_cooh"] == 1
        assert b["n_aryl_amine"] == 1
        assert b["has_strong_base"] == 0
        assert b["net_anion"] == 1
        assert b["is_zwitterion"] == 0

    def test_aryl_amine_counted_separately_from_aliphatic(self):
        # Imipramine has both: one aliphatic (basic) and one aryl N.
        b = _block(_IMIPRAMINE)
        assert b["n_aliph_amine"] == 1
        assert b["n_aryl_amine"] == 1


class TestAcidicGroupsBeyondCarboxyl:
    def test_tetrazole_is_a_strong_acid(self):
        b = _block(_LOSARTAN)
        assert b["n_tetrazole"] == 1
        assert b["has_strong_acid"] == 1
        assert b["net_anion"] == 1
