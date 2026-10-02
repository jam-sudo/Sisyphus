"""Every documented CLI command must render help without formatting errors."""

import sys

import pytest

from sisyphus.cli import main


@pytest.mark.parametrize(
    "command", ("predict", "simulate", "tdm", "ddi", "dose-adjust", "benchmark")
)
def test_subcommand_help(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["sisyphus", command, "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "usage: sisyphus" in capsys.readouterr().out
