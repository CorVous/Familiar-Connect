"""Tests for version management and version command."""

import re
import subprocess  # noqa: S404
import sys

import python_template
from python_template import __version__


def test_version_import() -> None:
    """Test that __version__ can be imported from package."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    # Should start with X.Y.Z (may have additional dev/git info)
    # Examples: "0.1.0", "0.1.1.dev0+g47aa3f55f.d20260327"

    # Match semantic version at start (X.Y.Z)
    pattern = r"^\d+\.\d+\.\d+"
    assert re.match(pattern, __version__), (
        f"Version {__version__} doesn't start with X.Y.Z"
    )


def test_version_subcommand() -> None:
    """Test version subcommand displays version."""
    package_name = python_template.__package__ or "python_template"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", package_name, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_version_flag() -> None:
    """Test --version flag displays version."""
    package_name = python_template.__package__ or "python_template"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", package_name, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
