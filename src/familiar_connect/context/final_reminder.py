"""Tail block appended to every system prompt.

Restates the current time (so the model doesn't drift on long-lived
caches) and enumerates the literal sentinels the responder honours:
``<silent>``, ``[@DisplayName]``, and ``[↩ <message_id>]``. Voice
channels only see ``<silent>`` — the others have no meaning without
text routing.

Also rendered a *second time* by the responders as a trailing
``system`` message after recent history, with
``include_mode_instruction=True`` so the per-mode operating
directive (e.g. "You are speaking aloud…") sits at the very tail
of the context window — recency-biased models are less likely to
ignore it there.
"""

from __future__ import annotations

from datetime import UTC, datetime

# Per-viewer-mode operating directive surfaced in the trailing
# reminder. Intentionally duplicates the strings the
# ``OperatingModeLayer`` is configured with in
# ``commands/run.py``: the head copy primes the system prompt;
# this tail copy combats recency bias on long contexts. Keep the
# wording in sync if you edit one.
_MODE_INSTRUCTIONS: dict[str, str] = {
    "voice": (
        "You are speaking aloud. Keep replies short "
        "(one or two sentences). Avoid markdown."
    ),
    "text": (
        "You are chatting in a text channel. Markdown and multi-line replies are fine."
    ),
}


def _fmt_when(now: datetime) -> str:
    """Render datetime as ``YYYY-MM-DD H:MMpm UTC`` (no leading zero)."""
    aware = now.astimezone(UTC)
    clock = aware.strftime("%I:%M%p").lstrip("0")
    return f"{aware.strftime('%Y-%m-%d')} {clock} UTC"


def build_final_reminder(
    *,
    viewer_mode: str,
    now: datetime | None = None,
    include_mode_instruction: bool = False,
) -> str:
    """Render the closing reminder block.

    Always lists ``<silent>``. Text channels add the ping + reply
    sentinels — those rely on per-message routing the voice path
    has no equivalent for. ``include_mode_instruction`` appends the
    per-mode operating directive — used by the trailing-system-message
    copy so the directive lands at the tail of the context window.
    """
    when = _fmt_when(now or datetime.now(tz=UTC))
    lines = [
        "---",
        "",
        f"It is now: {when}",
        "",
        "Special input:",
        "",
        "* `<silent>` - do nothing",
    ]
    if viewer_mode == "text":
        lines.extend([
            "",
            "* `[@DisplayName]` - ping user",
            "* `[↩ <message_id>]` - reply to message",
        ])
    if include_mode_instruction:
        instruction = _MODE_INSTRUCTIONS.get(viewer_mode)
        if instruction:
            lines.extend(["", instruction])
    return "\n".join(lines)
