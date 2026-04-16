"""Integration tests for CLI functionality."""

import re
import subprocess  # noqa: S404
import sys

import pytest

# Get package name dynamically - import from the actual package
import familiar_connect

_PACKAGE_NAME: str = familiar_connect.__package__ or "familiar_connect"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_cli_help() -> None:
    """Test that CLI help works."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    # CLI name should appear in help output


def test_cli_version_flag() -> None:
    """Test --version flag."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # Version should start with semantic version format

    assert re.search(r"\d+\.\d+\.\d+", result.stdout), "Version not found in output"


def test_cli_version_subcommand() -> None:
    """Test version subcommand."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # Version should start with semantic version format

    assert re.search(r"\d+\.\d+\.\d+", result.stdout), "Version not found in output"


def test_cli_verbose_flag() -> None:
    """Test that -v flag enables INFO logging."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "-v", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # version subcommand emits its INFO log; styled formatter suppresses
    # the level prefix, so check for the [Version] tag instead.
    combined = _strip(result.stdout) + _strip(result.stderr)
    assert "Version" in combined


def test_cli_very_verbose_flag() -> None:
    """Test that -vv flag enables DEBUG logging."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "-vv", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    # DEBUG logging should show more details
    combined = _strip(result.stdout) + _strip(result.stderr)
    assert "DEBUG:" in combined


def test_cli_no_subcommand() -> None:
    """Test that running without subcommand shows help."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_invalid_subcommand() -> None:
    """Test that invalid subcommand shows error."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "invalid"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid" in result.stderr.lower()


def test_version_subcommand_help() -> None:
    """Test that version subcommand has help."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "version", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "version" in result.stdout.lower()


def test_library_import_no_cli_execution() -> None:
    """Test that importing package doesn't execute CLI code."""
    # This test runs in subprocess to ensure clean import
    code = """
import sys
import io

# Capture stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Import package
from familiar_connect import __version__

# Get captured output
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()

# Restore
sys.stdout = old_stdout
sys.stderr = old_stderr

# Check no CLI output
assert not stdout_value, f"Unexpected stdout: {stdout_value}"
assert not stderr_value, f"Unexpected stderr: {stderr_value}"
# Version should be a non-empty string
assert __version__ and isinstance(__version__, str)
print("SUCCESS")
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "SUCCESS" in result.stdout


@pytest.mark.skip(reason="Example command not registered - template only")
def test_example_subcommand() -> None:
    """Test example subcommand (when registered)."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", _PACKAGE_NAME, "example", "World"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Hello, World!" in result.stdout


@pytest.mark.skip(reason="Example command not registered - template only")
def test_example_subcommand_with_greeting() -> None:
    """Test example subcommand with custom greeting."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            _PACKAGE_NAME,
            "example",
            "World",
            "--greeting",
            "Hi",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Hi, World!" in result.stdout
