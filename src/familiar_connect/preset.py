"""SillyTavern preset loader and system-prompt assembler.

Loads a SillyTavern preset JSON and assembles a system prompt string by
walking the ``prompt_order`` and substituting character-card fields for
marker entries.

Only the fields needed for the text-chat prototype are handled.  Unknown
macros (``{{getvar::...}}``, ``{{setvar::...}}``, etc.) pass through from
``macros.substitute`` unchanged, which is fine since they are not rendered
in the final prompt.

``chatHistory`` is always skipped — the caller is responsible for providing
conversation history as discrete ``Message`` objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.macros import MacroContext, substitute

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.character import CharacterCard

PathLike = str | Path

# Identifiers that are "markers" for well-known character-card sections.
# Values are callables (card) -> str.
_MARKER_FIELDS: dict[str, Callable[[CharacterCard], str]] = {
    "charDescription": lambda c: c.description,
    "charPersonality": lambda c: c.personality,
    "scenario": lambda c: c.scenario,
    "personaDescription": lambda _: "",
    "dialogueExamples": lambda c: c.mes_example,
    "worldInfoBefore": lambda _: "",
    "worldInfoAfter": lambda _: "",
    "enhanceDefinitions": lambda _: "",
    # chatHistory is always skipped (handled by the caller as messages).
}

_SKIP_IDENTIFIERS = {"chatHistory"}


class PresetError(Exception):
    """Raised when a preset file cannot be loaded or parsed."""


def load_preset(source: dict | PathLike) -> dict:
    """Load a SillyTavern preset from a file path or an already-parsed dict.

    :param source: Path to a ``.json`` file (str or Path), or a dict.
    :raises PresetError: If the file cannot be read or parsed.
    :return: The parsed preset dict.
    """
    if isinstance(source, dict):
        return source

    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read preset file: {source}"
        raise PresetError(msg) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"Preset JSON is invalid: {exc}"
        raise PresetError(msg) from exc


def assemble_prompt(
    preset: dict,
    card: CharacterCard,
    character_id: int = 100001,
) -> str:
    r"""Assemble a system prompt from a SillyTavern preset and a character card.

    Walks ``prompt_order`` for *character_id* (falling back to the first entry
    if the id is not found), skipping disabled entries and ``chatHistory``.
    Marker entries are substituted with the corresponding ``CharacterCard``
    field; other entries have their ``content`` run through macro substitution.
    Empty sections are omitted; non-empty sections are joined with ``\n\n``.

    :param preset: Parsed preset dict (from :func:`load_preset`).
    :param card: The character card to fill marker slots from.
    :param character_id: Which ``prompt_order`` entry to use (default 100001).
    :return: Assembled system prompt string.
    """
    prompts_by_id: dict[str, dict] = {
        p["identifier"]: p for p in preset.get("prompts", [])
    }

    order_entry = _find_order(preset, character_id)
    if order_entry is None:
        return ""

    macro_ctx = MacroContext(
        char=card.name,
        scenario=card.scenario,
        personality=card.personality,
        description=card.description,
    )

    sections: list[str] = []

    for item in order_entry:
        ident = item.get("identifier", "")
        if not item.get("enabled", True):
            continue
        if ident in _SKIP_IDENTIFIERS:
            continue

        text = _resolve_entry(ident, prompts_by_id, card, macro_ctx)
        if text:
            sections.append(text)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_order(preset: dict, character_id: int) -> list[dict] | None:
    """Return the order list for *character_id*, or the first order if not found."""
    prompt_order: list[dict] = preset.get("prompt_order", [])
    if not prompt_order:
        return None
    for entry in prompt_order:
        if entry.get("character_id") == character_id:
            return entry.get("order", [])
    # Fall back to the first available order.
    return prompt_order[0].get("order", [])


def _resolve_entry(
    ident: str,
    prompts_by_id: dict[str, dict],
    card: CharacterCard,
    macro_ctx: MacroContext,
) -> str:
    """Return the resolved text for a single prompt entry, or '' to skip."""
    if ident in _SKIP_IDENTIFIERS:
        return ""

    # Marker with a known card field
    if ident in _MARKER_FIELDS:
        value: str = _MARKER_FIELDS[ident](card)
        return substitute(value, macro_ctx).strip()

    # Custom prompt or standard non-marker prompt
    prompt = prompts_by_id.get(ident)
    if prompt is None:
        return ""

    # Markers whose field we don't know → skip
    if prompt.get("marker"):
        return ""

    content: str = prompt.get("content", "")
    return substitute(content, macro_ctx).strip()
