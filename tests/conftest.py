"""Shared pytest options for the test suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="use --run-slow to run"))
