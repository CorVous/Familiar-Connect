"""Tail block appended to every system prompt.

Restates current time (so model doesn't drift on long-lived caches)
and enumerates text-channel sentinels: ``[@DisplayName]``,
``[↩ <message_id>]``.

Rendered a *second time* by the responders as a trailing
``system`` message after recent history, with
``include_mode_instruction=True`` so the per-mode operating
directive (e.g. "You are speaking aloud…") sits at the tail of the
context window — recency-biased models less likely to ignore it
there.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

# Per-viewer-mode operating directive in the trailing reminder.
# Intentionally duplicates strings ``OperatingModeLayer`` is
# configured with in ``commands/run.py``: head copy primes system
# prompt; tail copy combats recency bias on long contexts. keep
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


def _fmt_when(now: datetime, display_tz: str = "UTC") -> str:
    """Render as ``YYYY-MM-DD H:MMpm TZ`` in *display_tz* (no leading zero on hour).

    *display_tz* must be a valid IANA name; validated at config load.
    """
    aware = now.astimezone(ZoneInfo(display_tz))
    clock = aware.strftime("%I:%M%p").lstrip("0")
    return f"{aware.strftime('%Y-%m-%d')} {clock} {aware.strftime('%Z')}"


def build_final_reminder(
    *,
    viewer_mode: str,
    now: datetime | None = None,
    display_tz: str = "UTC",
    include_time: bool = True,
    include_mode_instruction: bool = False,
    tools_enabled: bool = False,
    post_history_instructions: str | None = None,
    focus_channel_id: int | None = None,
    unread_digest: dict[int, int] | None = None,
    channel_names: dict[int, str] | None = None,
) -> str:
    """Render closing reminder block.

    Text channels list ping + reply sentinels.
    ``include_time=False`` omits the timestamp — use on head system
    messages to keep the cache prefix stable. ``display_tz`` (IANA
    name) sets the zone the ``It is now:`` line renders in.
    ``include_mode_instruction`` appends the per-mode operating
    directive — used by trailing-system-message copy so
    directive lands at tail of context window. ``tools_enabled``
    (voice only) appends a short instruction targeting the
    empty-content tool_call failure mode — nudges model to speak
    before invoking a tool so user doesn't hear silence mid-turn.
    ``focus_channel_id`` appends a directive naming the active
    channel and the shift_focus tool. ``unread_digest`` renders a
    compact unreads summary (channels with count > 0 only).
    ``post_history_instructions`` (per-familiar etiquette) appended
    last — deepest, most recency-biased slot. Blank/None omits it.
    """
    lines = ["---"]
    if include_time:
        when = _fmt_when(now or datetime.now(tz=UTC), display_tz)
        lines.extend(["", f"It is now: {when}"])
    if viewer_mode == "text":
        lines.extend([
            "",
            "Special input:",
            "",
            "* `[@DisplayName]` - ping user",
            "* `[↩ <message_id>]` - reply to message",
        ])
    if include_mode_instruction:
        instruction = _MODE_INSTRUCTIONS.get(viewer_mode)
        if instruction:
            lines.extend(["", instruction])
    if tools_enabled and viewer_mode == "voice":
        lines.extend([
            "",
            (
                "Always speak at least a brief acknowledgement before "
                "calling a tool. Never reply with a tool call alone."
            ),
        ])
    if focus_channel_id is not None or unread_digest:
        names = channel_names or {}

        def _ch(cid: int) -> str:
            n = names.get(cid)
            return f"#{n}" if n else f"#{cid}"

        focus_part = (
            f"Your attention is currently on {_ch(focus_channel_id)}."
            if focus_channel_id is not None
            else ""
        )
        active = (
            [(cid, cnt) for cid, cnt in unread_digest.items() if cnt > 0]
            if unread_digest
            else []
        )
        if active:
            ch_list = ", ".join(
                _ch(cid) + (f" ({cnt})" if cnt > 1 else "") for cid, cnt in active
            )
            total = sum(c for _, c in active)
            verb = "is" if total == 1 else "are"
            noun = "a new message" if total == 1 else "new messages"
            unread_part = (
                f"There {verb} {noun} in {ch_list} "
                "— use shift_focus if it pulls your attention."
            )
        else:
            unread_part = ""

        block = " ".join(p for p in [focus_part, unread_part] if p)
        if block:
            lines.extend(["", block])
    if post_history_instructions and post_history_instructions.strip():
        lines.extend(["", post_history_instructions.strip()])
    return "\n".join(lines)
