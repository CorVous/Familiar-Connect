"""Tests for CLI parser and entry point."""

import argparse
import subprocess  # noqa: S404
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from familiar_connect import __version__
from familiar_connect.cli import create_parser, main, setup_logging


def test_create_parser_exists() -> None:
    """Test that create_parser function exists and returns ArgumentParser."""
    parser = create_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_has_verbose_argument() -> None:
    """Test that parser has -v/--verbose argument."""
    parser = create_parser()

    # Test -v flag
    args = parser.parse_args(["-v", "version"])
    assert hasattr(args, "verbose")

    # Test --verbose flag
    args = parser.parse_args(["--verbose", "version"])
    assert hasattr(args, "verbose")


def test_parser_verbose_count_based() -> None:
    """Test that -v can be repeated for increased verbosity."""
    parser = create_parser()

    # No -v flag
    args = parser.parse_args(["version"])
    assert args.verbose == 0

    # Single -v
    args = parser.parse_args(["-v", "version"])
    assert args.verbose >= 1

    # Double -vv
    args = parser.parse_args(["-vv", "version"])
    assert args.verbose >= 2


def test_parser_has_version_flag() -> None:
    """Test that parser has --version flag."""
    parser = create_parser()

    # Capture version output
    with patch("sys.stdout", new=StringIO()) as fake_out:
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0
        output = fake_out.getvalue()
        assert __version__ in output


def test_parser_no_subcommand_shows_help() -> None:
    """Test that running with no subcommand shows help message."""
    result = subprocess.run(
        [sys.executable, "-m", "familiar_connect"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Should show help (exit code 0) or error asking for subcommand
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()


def test_main_function_exists() -> None:
    """Test that main() entry point function exists."""
    assert callable(main)


def test_setup_logging_exists() -> None:
    """Test that setup_logging function exists."""
    assert callable(setup_logging)
