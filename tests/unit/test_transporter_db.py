"""Unit tests for the OATP1B1 transporter database loader."""

from __future__ import annotations

import pytest

from sisyphus.core import TransporterKinetics
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics


def test_load_pravastatin():
    kinetics = load_oatp1b1_kinetics("pravastatin")
    assert kinetics is not None
    assert "OATP1B1" in kinetics
    tk = kinetics["OATP1B1"]
    assert isinstance(tk, TransporterKinetics)
    assert tk.jmax.mean == pytest.approx(228.0)
    assert tk.jmax.cv == pytest.approx(0.30)
    assert tk.km.mean == pytest.approx(13.6)
    assert tk.km.cv == pytest.approx(0.25)


def test_load_unknown_drug_returns_none():
    assert load_oatp1b1_kinetics("aspirin") is None


def test_load_is_case_insensitive():
    assert load_oatp1b1_kinetics("Pravastatin") is not None
    assert load_oatp1b1_kinetics("PRAVASTATIN") is not None
