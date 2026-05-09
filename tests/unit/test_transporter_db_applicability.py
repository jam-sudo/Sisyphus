"""Unit tests for OATP ECM applicability lookup."""
from __future__ import annotations

from sisyphus.predict.transporter_db import is_oatp_ecm_applicable

_PRAVA_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_FLUVA_SMILES = (
    "CC(C)N1C2=CC=CC=C2C(=C1/C=C/[C@H](O)C[C@H](O)CC(=O)O)"
    "C3=CC=C(F)C=C3"
)
_MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2"


def test_pravastatin_is_applicable():
    """Pravastatin is the canonical OATP-rate-limited substrate (Niemi 2009 PM/EM ~2.6x)."""
    assert is_oatp_ecm_applicable(_PRAVA_SMILES) is True


def test_fluvastatin_not_applicable():
    """Fluvastatin is CYP2C9-dominant (Niemi 2009 PM/EM ~1.0x)."""
    assert is_oatp_ecm_applicable(_FLUVA_SMILES) is False


def test_non_oatp_drug_not_applicable():
    """Midazolam is not in the OATP1B1 registry at all."""
    assert is_oatp_ecm_applicable(_MIDAZOLAM_SMILES) is False


def test_invalid_smiles_returns_false():
    """Bad SMILES → False (no exception, fail-safe)."""
    assert is_oatp_ecm_applicable("not_a_valid_smiles") is False


def test_empty_smiles_returns_false():
    """Empty SMILES → False (no exception)."""
    assert is_oatp_ecm_applicable("") is False


from sisyphus.predict.transporter_db import (  # noqa: E402
    load_hepatic_ecm_params_for_smiles,
    load_oatp1b1_kinetics_for_smiles,
)


def test_load_oatp1b1_kinetics_for_smiles_pravastatin():
    """SMILES-keyed loader returns the same kinetics as the name-keyed one."""
    kin = load_oatp1b1_kinetics_for_smiles(_PRAVA_SMILES)
    assert kin is not None
    assert "OATP1B1" in kin
    assert kin["OATP1B1"].jmax.mean == 228.0
    assert kin["OATP1B1"].km.mean == 13.6


def test_load_oatp1b1_kinetics_for_smiles_unknown_returns_none():
    """Unregistered SMILES → None (caller falls through to no-ECM path)."""
    assert load_oatp1b1_kinetics_for_smiles(_MIDAZOLAM_SMILES) is None


def test_load_hepatic_ecm_params_for_smiles_pravastatin():
    """SMILES-keyed ECM loader returns the registered params."""
    ecm = load_hepatic_ecm_params_for_smiles(_PRAVA_SMILES)
    assert ecm is not None
    assert ecm["ps_passive"].mean == 0.8
    assert ecm["ps_eff"].mean == 0.8
    assert ecm["cl_int_bile"].mean == 45.0


def test_load_hepatic_ecm_params_for_smiles_unknown_returns_none():
    assert load_hepatic_ecm_params_for_smiles(_MIDAZOLAM_SMILES) is None


def test_load_for_invalid_smiles_returns_none():
    assert load_oatp1b1_kinetics_for_smiles("not_smiles") is None
    assert load_hepatic_ecm_params_for_smiles("not_smiles") is None
