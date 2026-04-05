"""Tests for logging configuration."""

import logging

import pytest

from python_template.cli import setup_logging


def test_setup_logging_default() -> None:
    """Test that default logging level is WARNING."""
    setup_logging(verbose=0)

    # Get root logger level
    assert logging.root.level == logging.WARNING


def test_setup_logging_verbose_once() -> None:
    """Test that -v sets logging to INFO."""
    setup_logging(verbose=1)

    assert logging.root.level == logging.INFO


def test_setup_logging_verbose_twice() -> None:
    """Test that -vv sets logging to DEBUG."""
    setup_logging(verbose=2)

    assert logging.root.level == logging.DEBUG


def test_setup_logging_verbose_many() -> None:
    """Test that -vvv+ stays at DEBUG."""
    setup_logging(verbose=5)

    assert logging.root.level == logging.DEBUG


def test_setup_logging_explicit_level_debug() -> None:
    """Test that explicit DEBUG level works."""
    setup_logging(verbose=0, level="DEBUG")

    assert logging.root.level == logging.DEBUG


def test_setup_logging_explicit_level_info() -> None:
    """Test that explicit INFO level works."""
    setup_logging(verbose=0, level="INFO")

    assert logging.root.level == logging.INFO


def test_setup_logging_explicit_level_warning() -> None:
    """Test that explicit WARNING level works."""
    setup_logging(verbose=0, level="WARNING")

    assert logging.root.level == logging.WARNING


def test_setup_logging_explicit_level_error() -> None:
    """Test that explicit ERROR level works."""
    setup_logging(verbose=0, level="ERROR")

    assert logging.root.level == logging.ERROR


def test_setup_logging_explicit_level_critical() -> None:
    """Test that explicit CRITICAL level works."""
    setup_logging(verbose=0, level="CRITICAL")

    assert logging.root.level == logging.CRITICAL


def test_setup_logging_explicit_level_case_insensitive() -> None:
    """Test that log level is case-insensitive."""
    setup_logging(verbose=0, level="debug")
    assert logging.root.level == logging.DEBUG

    setup_logging(verbose=0, level="InFo")
    assert logging.root.level == logging.INFO


def test_setup_logging_invalid_level() -> None:
    """Test that invalid log level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(verbose=0, level="INVALID")


def test_setup_logging_explicit_overrides_verbose() -> None:
    """Test that explicit level overrides verbose count."""
    # Even with verbose=2 (DEBUG), explicit WARNING should win
    setup_logging(verbose=2, level="WARNING")

    assert logging.root.level == logging.WARNING
