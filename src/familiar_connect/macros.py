"""Minimal SillyTavern macro substitution.

Supported macros (deliberate subset — unknown macros pass through unchanged):
  {{char}}         — character name
  {{user}}         — user name (default: "User")
  {{trim}}         — strip leading/trailing whitespace from the whole string
  {{scenario}}     — character scenario
  {{personality}}  — character personality
  {{description}}  — character description
  {{// ... }}      — comment, removed entirely

Unsupported macros ({{getvar::...}}, {{random:...}}, conditionals, etc.)
are left as-is so callers can see they were not resolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MacroContext:
    """Values to substitute into macro placeholders."""

    char: str = ""
    user: str = "User"
    scenario: str = ""
    personality: str = ""
    description: str = ""


# Matches {{// any comment text }}
_COMMENT_RE = re.compile(r"\{\{//[^}]*\}\}")

# Matches any {{macro}} or {{macro::arg}} token
_MACRO_RE = re.compile(r"\{\{([^}]+)\}\}")

_SIMPLE: dict[str, str] = {}  # populated per-call from context


def substitute(text: str, ctx: MacroContext) -> str:
    """Resolve macros in *text* using the values in *ctx*.

    Processing order:
    1. Remove comments (``{{// ... }}``)
    2. Replace known simple macros
    3. Handle ``{{trim}}`` — strip the whole string
    4. Leave unknown macros untouched
    """
    # 1. Strip comments
    text = _COMMENT_RE.sub("", text)

    simple = {
        "char": ctx.char,
        "user": ctx.user,
        "scenario": ctx.scenario,
        "personality": ctx.personality,
        "description": ctx.description,
    }

    trim_requested = False

    def _replace(m: re.Match[str]) -> str:
        nonlocal trim_requested
        key = m.group(1).strip()
        if key == "trim":
            trim_requested = True
            return ""
        if key in simple:
            return simple[key]
        # Unknown macro — pass through
        return m.group(0)

    # 2 & 3. Replace all macros in one pass
    text = _MACRO_RE.sub(_replace, text)

    # 4. Apply trim if requested
    if trim_requested:
        text = text.strip()

    return text
