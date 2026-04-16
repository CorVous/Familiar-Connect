"""Styled console log primitives.

Each call site composes its own log string from these primitives —
emoji, label, and colour choices live next to the log they describe.

Call ``init()`` once at startup (done inside ``setup_logging``).
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from colorama import Fore, Style
from colorama import init as _colorama_init


def init(strip: bool = False) -> None:  # noqa: FBT001, FBT002
    """Initialize colorama — call once at process start."""
    _colorama_init(strip=strip, autoreset=False)


# ---------------------------------------------------------------------------
# Public colour constants
# ---------------------------------------------------------------------------

# colorama Fore.X are class-level ints that the AnsiCodes constructor rewrites
# into ANSI escape strings at instantiation; cast so static checkers see str.
W = cast("str", Fore.WHITE)
C = cast("str", Fore.CYAN)
G = cast("str", Fore.GREEN)
Y = cast("str", Fore.YELLOW)
B = cast("str", Fore.BLUE)
M = cast("str", Fore.MAGENTA)
R = cast("str", Fore.RED)
LG = cast("str", Fore.LIGHTGREEN_EX)
LY = cast("str", Fore.LIGHTYELLOW_EX)
LC = cast("str", Fore.LIGHTCYAN_EX)
LM = cast("str", Fore.LIGHTMAGENTA_EX)
LB = cast("str", Fore.LIGHTBLUE_EX)
LW = cast("str", Fore.LIGHTWHITE_EX)
RS = cast("str", Style.RESET_ALL)


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------


def tag(text: str, color: str) -> str:
    """Bracketed label. Brackets always white; inner text takes ``color``."""
    return f"{W}[{color}{text}{W}]{RS}"


def kv(key: str, val: str, *, kc: str = W, vc: str = W) -> str:
    """``key=value`` chunk with separate colours for key and value."""
    return f"{kc}{key}={RS}{vc}{val}{RS}"


def word(text: str, color: str) -> str:
    """Single coloured word."""
    return f"{color}{text}{RS}"


def trunc(text: str, limit: int = 200) -> str:
    """Truncate with ``…`` ellipsis if longer than *limit*."""
    return f"{text[:limit]}{'…' if len(text) > limit else ''}"


# ---------------------------------------------------------------------------
# Custom logging formatter
# ---------------------------------------------------------------------------


class StyledFormatter(logging.Formatter):
    """Suppress INFO prefix; colour WARNING/ERROR/DEBUG level labels."""

    _LEVEL_PREFIX: ClassVar[dict[int, str]] = {
        logging.DEBUG: f"{LW}DEBUG{RS}: ",
        logging.WARNING: f"{Y}WARNING{RS}: ",
        logging.ERROR: f"{R}ERROR{RS}: ",
        logging.CRITICAL: f"{R}CRITICAL{RS}: ",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        prefix = self._LEVEL_PREFIX.get(record.levelno, "")
        return f"{prefix}{msg}"
