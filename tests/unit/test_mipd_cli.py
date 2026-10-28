"""MIPD package and CLI entry point checks."""
import sys

from sisyphus import cli
from sisyphus.mipd import Covariates, predict_posterior, predict_tdm


def test_public_exports():
    assert callable(predict_posterior)
    assert callable(predict_tdm)
    assert Covariates().renal_factor() == 1.0


def test_mipd_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "sisyphus", "mipd", "--smiles", "CCO", "--dose", "100",
        "--measured-f", "0.5", "--n-samples", "100",
    ])
    cli.main()
    assert "Engine posterior Cmax:" in capsys.readouterr().out
