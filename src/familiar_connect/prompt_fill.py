"""Crash-safe ``{placeholder}`` fill for config-sourced prompt templates.

Config-sourced (per-familiar overridable) prompt text is filled with
dynamic values (``{self_name}`` etc.) at build time. ``str.format``
would raise on a stray ``{``/``}``, an unknown token, or a missing
expected one — a phrasing override must never crash a pass. This fills
by literal substitution: only the listed ``{key}`` tokens are replaced;
everything else (stray braces, unknown tokens) passes through verbatim.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\{(\w+)\}")


def fill_placeholders(template: str, /, **values: object) -> str:
    """Replace each ``{key}`` in *template* with ``str(value)``.

    Unlisted ``{...}`` tokens and stray braces pass through unchanged —
    never raises (graceful degrade for config overrides).

    Single pass / no re-expansion: each ``{key}`` token is filled exactly
    once over the original template — an injected value containing another
    key's token is left literal, never re-scanned. Order-independent.
    """

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    return _TOKEN.sub(_sub, template)
