"""Crash-safe ``{placeholder}`` fill for config-sourced prompt templates.

Config-sourced (per-familiar overridable) prompt text is filled with
dynamic values (``{self_name}`` etc.) at build time. ``str.format``
would raise on a stray ``{``/``}``, an unknown token, or a missing
expected one — a phrasing override must never crash a pass. This fills
by literal substitution: only the listed ``{key}`` tokens are replaced;
everything else (stray braces, unknown tokens) passes through verbatim.
"""

from __future__ import annotations


def fill_placeholders(template: str, /, **values: object) -> str:
    """Replace each ``{key}`` in *template* with ``str(value)``.

    Unlisted ``{...}`` tokens and stray braces pass through unchanged —
    never raises (graceful degrade for config overrides).
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out
