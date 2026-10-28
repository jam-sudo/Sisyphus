"""Every documented CLI command must render help without formatting errors."""

import sys

import pytest

from sisyphus.cli import main


@pytest.mark.parametrize(
    "command", ("predict", "simulate", "tdm", "ddi", "pkpd", "dose-adjust", "benchmark")
)
def test_subcommand_help(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["sisyphus", command, "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "usage: sisyphus" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "options", "expected"),
    (
        (
            "ddi",
            ["--tdi-enzyme", "CYP3A4", "--kinact", "0.2", "--ki-tdi", "1",
             "--kdeg", "0.1", "--concentration", "1"],
            "inhibition (user-supplied TDI)",
        ),
        ("pkpd", ["--ke0", "0.5", "--ec50", "0.05"], "Peak effect:"),
    ),
)
def test_scenario_cli_smoke(monkeypatch, capsys, command, options, expected):
    monkeypatch.setattr(
        sys, "argv", ["sisyphus", command, "--smiles", "CCO", "--dose", "100", *options]
    )
    main()
    assert expected in capsys.readouterr().out
