"""Unit tests for OATP ECM applicability lookup."""
from __future__ import annotations

import pytest

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
