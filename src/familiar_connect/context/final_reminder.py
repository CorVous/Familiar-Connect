"""Tail block appended to every system prompt.

Restates the current time (so the model doesn't drift on long-lived
caches) and enumerates the literal sentinels the responder honours:
``<silent>``, ``[@DisplayName]``, and ``[↩ <message_id>]``. Voice
channels only see ``<silent>`` — the others have no meaning without
text routing.
"""

from __future__ import annotations

from datetime import UTC, datetime


def _fmt_when(now: datetime) -> str:
    """Render datetime as ``YYYY-MM-DD H:MMpm UTC`` (no leading zero)."""
    aware = now.astimezone(UTC)
    clock = aware.strftime("%I:%M%p").lstrip("0")
    return f"{aware.strftime('%Y-%m-%d')} {clock} UTC"


def build_final_reminder(
    *,
    viewer_mode: str,
    now: datetime | None = None,
) -> str:
    """Render the closing reminder block.

    Always lists ``<silent>``. Text channels add the ping + reply
    sentinels — those rely on per-message routing the voice path
    has no equivalent for.
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
    return "\n".join(lines)
