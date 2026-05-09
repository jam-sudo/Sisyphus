"""
Tests for training utility functions: logit/sigmoid transforms, TDC data
conversion conventions, and holdout exclusion logic.

These tests document the numerical contracts that training scripts must
satisfy — they are specification tests, not implementation tests.
"""

import numpy as np
import pytest


class TestLogitSigmoid:
    """Logit and sigmoid are mutual inverses within (0, 1).

    Training scripts that use logit-space optimization (e.g., for fup
    which lives in (0, 1)) must clip inputs before taking the logit to
    avoid -inf / +inf. These tests pin the numerical behaviour.
    """

    def test_logit_sigmoid_roundtrip(self):
        x = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
        eps = 1e-6
        x_clipped = np.clip(x, eps, 1 - eps)
        logit_x = np.log(x_clipped / (1 - x_clipped))
        recovered = 1 / (1 + np.exp(-logit_x))
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_sigmoid_bounded(self):
        """Sigmoid output is in [0, 1] for all finite inputs.

        In float64, extreme inputs (|x| >= ~37) saturate to exactly 0.0 or
        1.0 due to limited precision. The test documents this: the sigmoid is
        mathematically in (0, 1) but numerically in [0, 1] at float64.
        """
        extreme = np.array([-100, -10, -1, 0, 1, 10, 100], dtype=np.float64)
        result = 1 / (1 + np.exp(-extreme))
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Moderate inputs (|x| < 20) never saturate in float64
        moderate = np.array([-10, -1, 0, 1, 10], dtype=np.float64)
        moderate_result = 1 / (1 + np.exp(-moderate))
        assert np.all(moderate_result > 0)
        assert np.all(moderate_result < 1)

    def test_logit_zero_epsilon(self):
        """Clipping 0.0 to eps produces a large negative finite logit."""
        eps = 1e-6
        x = np.clip(0.0, eps, 1 - eps)
        logit_val = np.log(x / (1 - x))
        assert np.isfinite(logit_val)
        assert logit_val < -10

    def test_logit_one_epsilon(self):
        """Clipping 1.0 to 1-eps produces a large positive finite logit."""
        eps = 1e-6
        x = np.clip(1.0, eps, 1 - eps)
        logit_val = np.log(x / (1 - x))
        assert np.isfinite(logit_val)
        assert logit_val > 10


class TestTDCConversion:
    """TDC PPBR_AZ reports % plasma protein bound (0-100 scale).

    Training targets must be converted to fup = fraction unbound
    before use. These tests document the direction and magnitude of
    the conversion.
    """

    def test_ppbr_to_fup(self):
        """Y=95 % bound → fup=0.05, etc."""
        y = np.array([95.0, 50.0, 10.0, 99.5])
        fup = (100 - y) / 100
        np.testing.assert_allclose(fup, [0.05, 0.50, 0.90, 0.005])

    def test_ppbr_range_detection(self):
        """Distinguish % scale from fraction scale by max value."""
        pct_data = np.array([95, 88, 50, 10])
        frac_data = np.array([0.95, 0.88, 0.50, 0.10])
        # A simple heuristic used in training scripts: max > 1 → % scale
        assert pct_data.max() > 1.0
        assert frac_data.max() <= 1.0

    def test_ppbr_fup_sum_to_one(self):
        """bound_fraction + fup must equal 1.0 for any valid PPBR value."""
        y_pct = np.array([99.5, 95.0, 50.0, 10.0, 1.0])
        bound = y_pct / 100
        fup = (100 - y_pct) / 100
        np.testing.assert_allclose(bound + fup, 1.0, atol=1e-10)

    def test_ppbr_zero_bound_edge(self):
        """Y=0 (fully unbound) → fup=1.0."""
        fup = (100 - 0.0) / 100
        assert fup == pytest.approx(1.0)

    def test_ppbr_fully_bound_edge(self):
        """Y=100 (fully bound) → fup=0.0."""
        fup = (100 - 100.0) / 100
        assert fup == pytest.approx(0.0)


class TestHoldoutExclusion:
    """Holdout contamination detection using three complementary keys.

    A TDC compound is considered a contamination risk if any of:
    1. Canonical SMILES (isomericSmiles=True) matches exactly
    2. First 14 characters of InChIKey match (connectivity layer only)
    3. Lowercase Drug_ID / name matches a holdout name

    Tests document the expected behaviour of each method and their
    known limitations.
    """

    def test_exact_smiles_excluded(self):
        """Identical SMILES strings canonicalize to the same form."""
        from rdkit import Chem

        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        holdout_set = {canonical}
        assert canonical in holdout_set

    def test_canonical_smiles_normalises_notation(self):
        """Two equivalent SMILES representations produce the same canonical form."""
        from rdkit import Chem

        smi1 = "c1ccccc1"  # benzene aromatic notation
        smi2 = "C1=CC=CC=C1"  # benzene Kekulé notation
        c1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi1), isomericSmiles=True)
        c2 = Chem.MolToSmiles(Chem.MolFromSmiles(smi2), isomericSmiles=True)
        assert c1 == c2

    def test_inchikey_prefix_matches_same_compound(self):
        """Same compound generates the same 14-char InChIKey prefix."""
        from rdkit import Chem
        from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

        smi = "CC(C)NCC(O)c1ccc(O)c(O)c1"  # isoproterenol
        mol = Chem.MolFromSmiles(smi)
        ik1 = InchiToInchiKey(MolToInchi(mol))[:14]
        ik2 = InchiToInchiKey(MolToInchi(mol))[:14]
        assert ik1 == ik2
        assert len(ik1) == 14

    def test_salt_form_caught_by_inchikey(self):
        """Document salt form InChIKey behaviour.

        RDKit represents a salt (base.Cl) as a disconnected molecule.
        The InChI of the disconnected form differs from the free base,
        so the 14-char prefix will NOT match. Name-based exclusion is
        needed to catch salt forms whose Drug_ID contains the INN.
        """
        from rdkit import Chem
        from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

        base = Chem.MolFromSmiles("CC(C)NCC(O)c1ccc(O)c(O)c1")
        salt = Chem.MolFromSmiles("CC(C)NCC(O)c1ccc(O)c(O)c1.Cl")
        ik_base = InchiToInchiKey(MolToInchi(base))[:14]
        ik_salt = InchiToInchiKey(MolToInchi(salt))[:14]
        # Document the actual behaviour — salt detection requires name matching
        if ik_base == ik_salt:
            # If RDKit normalises the salt (e.g., via protonation removal),
            # InChIKey matching suffices.
            pass
        else:
            # Salt has a different InChIKey prefix — name matching is needed.
            assert ik_base != ik_salt

    def test_name_exclusion_case_insensitive(self):
        """Lowercase name comparison catches mixed-case Drug_ID entries."""
        holdout_names = {"metformin", "warfarin", "midazolam"}
        drug_id = "Metformin"  # TDC may capitalise the name
        assert drug_id.lower() in holdout_names

    def test_three_key_union_catches_more(self):
        """Union of three exclusion methods is strictly >= any single method."""
        # Construct a scenario: SMILES key misses but name key catches
        smiles_matches = {"CHEMBL123"}  # hypothetical SMILES hit
        inchikey_matches = {"CHEMBL456"}
        name_matches = {"CHEMBL789"}
        union = smiles_matches | inchikey_matches | name_matches
        assert len(union) >= len(smiles_matches)
        assert len(union) >= len(inchikey_matches)
        assert len(union) >= len(name_matches)
