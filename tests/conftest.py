"""Shared test fixtures and configuration."""

import importlib.metadata

import pytest


@pytest.fixture
def package_name() -> str:
    """Return the package name (snake_case for imports)."""
    return __package__.split(".")[0] if __package__ else "familiar_connect"


@pytest.fixture
def cli_name() -> str:
    """Return the CLI command name (kebab-case)."""
    # Dynamically get from package metadata if possible
    try:
        return importlib.metadata.metadata("familiar-connect")["Name"]
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return "familiar-connect"
