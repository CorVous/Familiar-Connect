"""Tests for logging configuration."""

from __future__ import annotations

import logging
import re
import sys
from types import TracebackType

import pytest

from familiar_connect import log_style as ls
from familiar_connect.cli import setup_logging
from familiar_connect.log_style import StyledFormatter

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_ExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _record(
    level: int,
    msg: str,
    exc_info: _ExcInfo = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="t",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )


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


# ---------------------------------------------------------------------------
# StyledFormatter — tag repaint + level label placement
# ---------------------------------------------------------------------------


def test_formatter_info_no_prefix() -> None:
    msg = f"{ls.tag('Hi', ls.G)} body"
    out = StyledFormatter().format(_record(logging.INFO, msg))
    assert _strip(out) == "[Hi] body"


def test_formatter_debug_keeps_prefix() -> None:
    out = StyledFormatter().format(_record(logging.DEBUG, "plain"))
    assert _strip(out) == "DEBUG: plain"


def test_formatter_warning_moves_label_after_tag_yellow() -> None:
    msg = f"{ls.tag('Filter', ls.M)} body"
    out = StyledFormatter().format(_record(logging.WARNING, msg))
    assert _strip(out) == "[Filter] WARNING body"
    # inner tag text + label both wear yellow
    assert f"{ls.Y}Filter" in out
    assert f"{ls.Y}WARNING" in out


def test_formatter_error_moves_label_after_tag_red() -> None:
    msg = f"{ls.tag('Content', ls.M)} body"
    out = StyledFormatter().format(_record(logging.ERROR, msg))
    assert _strip(out) == "[Content] ERROR body"
    assert f"{ls.R}Content" in out
    assert f"{ls.R}ERROR" in out


def test_formatter_critical_uses_red() -> None:
    msg = f"{ls.tag('Boom', ls.M)} fatal"
    out = StyledFormatter().format(_record(logging.CRITICAL, msg))
    assert _strip(out) == "[Boom] CRITICAL fatal"
    assert f"{ls.R}CRITICAL" in out


def test_formatter_untagged_warning_falls_back_to_prefix() -> None:
    out = StyledFormatter().format(_record(logging.WARNING, "no tag here"))
    assert _strip(out) == "WARNING: no tag here"


def test_formatter_untagged_error_falls_back_to_prefix() -> None:
    out = StyledFormatter().format(_record(logging.ERROR, "bare error"))
    assert _strip(out) == "ERROR: bare error"


def _capture_exc_info(exc: BaseException) -> _ExcInfo:
    """Raise + catch to populate ``sys.exc_info`` with a real traceback."""
    try:
        raise exc  # noqa: TRY301 — fixture deliberately raises to capture traceback
    except BaseException:  # noqa: BLE001 — test fixture, re-captured below
        return sys.exc_info()


def test_formatter_appends_exception_traceback() -> None:
    """``_logger.exception`` relies on this — exc_info traceback must follow message."""
    exc_info = _capture_exc_info(RuntimeError("boom"))
    out = StyledFormatter().format(_record(logging.ERROR, "Command failed", exc_info))
    stripped = _strip(out)
    assert "ERROR: Command failed" in stripped
    assert "Traceback (most recent call last):" in stripped
    assert "RuntimeError: boom" in stripped


def test_formatter_appends_exception_traceback_with_tag() -> None:
    exc_info = _capture_exc_info(ValueError("kaboom"))
    msg = f"{ls.tag('Boot', ls.M)} startup failed"
    out = StyledFormatter().format(_record(logging.ERROR, msg, exc_info))
    stripped = _strip(out)
    assert stripped.startswith("[Boot] ERROR startup failed")
    assert "Traceback (most recent call last):" in stripped
    assert "ValueError: kaboom" in stripped


def test_formatter_info_does_not_get_double_traceback() -> None:
    """Cached exc_text must not be rendered twice on subsequent format calls."""
    exc_info = _capture_exc_info(RuntimeError("once"))
    record = _record(logging.ERROR, "Command failed", exc_info)
    fmt = StyledFormatter()
    first = _strip(fmt.format(record))
    second = _strip(fmt.format(record))
    assert first.count("RuntimeError: once") == 1
    assert second.count("RuntimeError: once") == 1
