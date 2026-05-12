"""Styled console log primitives.

Each call site composes its own log string from these primitives —
emoji, label, and colour choices live next to the log they describe.

Call ``init()`` once at startup (done inside ``setup_logging``).
"""

from __future__ import annotations

import logging
import re
from typing import cast

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


# matches a leading `tag(text, color)` render: W[ COLOR text W] RS
_TAG_RE = re.compile(r"^\x1b\[\d+m\[\x1b\[\d+m([^\x1b]+)\x1b\[\d+m\]\x1b\[0m")


class StyledFormatter(logging.Formatter):
    """Repaint leading tag + append level label for WARNING/ERROR; DEBUG prefix kept.

    Preserves stdlib behavior of appending exception/stack traces — without
    this, `_logger.exception(...)` would silently drop the traceback.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        lvl = record.levelno
        if lvl >= logging.WARNING:
            color = R if lvl >= logging.ERROR else Y
            label = logging.getLevelName(lvl)
            new_msg, n = _TAG_RE.subn(
                lambda m: f"{W}[{color}{m.group(1)}{W}]{RS} {color}{label}{RS}",
                msg,
                count=1,
            )
            out = new_msg if n else f"{color}{label}{RS}: {msg}"
        elif lvl == logging.DEBUG:
            out = f"{LW}DEBUG{RS}: {msg}"
        else:
            out = msg
        # mirror logging.Formatter: append exc_info / stack_info if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not out.endswith("\n"):
                out += "\n"
            out += record.exc_text
        if record.stack_info:
            if not out.endswith("\n"):
                out += "\n"
            out += self.formatStack(record.stack_info)
        return out
