"""Minimal SillyTavern macro substitution.

Supported macros (deliberate subset — unknown macros pass through unchanged):
  {{char}}         — character name
  {{user}}         — user name (default: "User")
  {{trim}}         — strip whitespace from whole string
  {{scenario}}     — character scenario
  {{personality}}  — character personality
  {{description}} — character description
  {{// ... }}      — comment, removed entirely

Unsupported macros ({{getvar::...}}, {{random:...}}, conditionals, etc.)
left as-is so callers see unresolved tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MacroContext:
    """Values substituted into macro placeholders."""

    char: str = ""
    user: str = "User"
    scenario: str = ""
    personality: str = ""
    description: str = ""


# matches {{// any comment text }}
_COMMENT_RE = re.compile(r"\{\{//[^}]*\}\}")

# matches any {{macro}} or {{macro::arg}} token
_MACRO_RE = re.compile(r"\{\{([^}]+)\}\}")

_SIMPLE: dict[str, str] = {}  # populated per-call from context


def substitute(text: str, ctx: MacroContext) -> str:
    """Resolve macros in *text* using *ctx*.

    Order:
    1. strip comments (``{{// ... }}``)
    2. replace known simple macros
    3. handle ``{{trim}}`` — strip whole string
    4. leave unknown macros untouched
    """
    # 1. strip comments
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
        # unknown macro — pass through
        return m.group(0)

    # 2 & 3. replace all macros in one pass
    text = _MACRO_RE.sub(_replace, text)

    # 4. apply trim if requested
    if trim_requested:
        text = text.strip()

    return text
