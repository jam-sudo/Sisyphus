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


def test_mipd_cli_repeated_iv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "sisyphus", "mipd", "--smiles", "CCO", "--dose", "100",
        "--route", "iv", "--interval", "12", "--doses", "2",
        "--obs", "24:0.1", "--n-samples", "100",
    ])
    cli.main()
    assert "Posterior F: 1.0000" in capsys.readouterr().out


def test_mipd_cli_structural_forward_search(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "sisyphus", "mipd", "--smiles", "CCO", "--dose", "100",
        "--route", "iv", "--interval", "12", "--doses", "3",
        "--crcl", "80", "--crcl-cv", "0.2",
        "--target-peak", "0.01:100", "--candidate-doses", "50", "100",
        "--candidate-intervals", "12", "24", "--loading-target", "1",
        "--n-samples", "100",
    ])
    cli.main()
    out = capsys.readouterr().out
    assert "Recommended maintenance:" in out
    assert "Loading dose:" in out


def test_mipd_cli_population_gfr_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "sisyphus", "mipd", "--smiles", "CCO", "--dose", "100",
        "--body-weight", "10", "--pma-weeks", "47.7", "--n-samples", "100",
    ])
    cli.main()
    assert "population-estimate" in capsys.readouterr().err
