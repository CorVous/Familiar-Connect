"""Render a :class:`PipelineOutput` into a list of chat messages.

Single source of truth for chat layout. Four rules live here:

- System prompt layers: everything except ``recent_history`` and
  ``depth_inject``
- Recent history: discrete ``Message`` objects from ``HistoryStore``
  (the ``recent_history`` Contribution is prose for budgeter
  accounting only)
- Depth-inject: inserted at configurable position-from-end (SillyTavern
  ``@D N`` convention)
- Speaker prefixes: user turns get ``Speaker: `` prefix because
  OpenRouter drops the OpenAI ``name`` field for non-OpenAI models;
  assistant turns are NOT prefixed
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from familiar_connect.config import ChannelMode
from familiar_connect.context.types import Layer
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.context.pipeline import PipelineOutput
    from familiar_connect.history.store import HistoryStore

_SECTION_SEPARATOR = "\n\n"

_DEFAULT_HISTORY_WINDOW = 20
"""How many recent turns to pull from the store when the caller doesn't say."""

_SYSTEM_PROMPT_LAYER_ORDER: tuple[Layer, ...] = (
    Layer.core,
    Layer.character,
    Layer.content,
    Layer.history_summary,
    Layer.author_note,
)
"""Natural top-to-bottom order of layers inside the system prompt.

Layers not listed here are deliberately excluded:
:class:`Layer.recent_history` renders as discrete messages;
:class:`Layer.depth_inject` renders mid-chat.
"""

_SPEAKER_PREFIX_PREAMBLE = (
    "User messages in this conversation are prefixed with the speaker's "
    'display name followed by a colon — for example, "Alice: hello". '
    "Different prefixes mean different people. When replying, address "
    "users by name when it's natural; do NOT prefix your own replies with "
    "your name — just write the reply directly."
)
"""Fixed preamble prepended to every system prompt.

This is a property of the rendering contract itself, not something the
character or mode can tune. It tells the LLM how to interpret the
``Speaker: `` prefix the renderer attaches to user turns, and
explicitly instructs it not to echo the prefix back in its own
replies. Kept as a module-level constant (rather than a provider) so
the invariant moves with the code that enforces it."""


def _prefix_user_content(speaker: str | None, content: str) -> str:
    """Prepend ``Speaker: `` to user content when a speaker is known.

    The prefix is redundant with :attr:`Message.name` for backends
    that honour the OpenAI ``name`` field, but it's the only way
    non-OpenAI models routed through OpenRouter ever see the
    speaker — see the module docstring for the rationale. A
    ``None`` speaker (system-generated events, sanitised-to-empty
    display names) renders bare rather than producing literal
    ``None: content``.
    """
    if not speaker:
        return content
    return f"{speaker}: {content}"


def _format_timestamp(ts: datetime, tz_name: str) -> str:
    """Render *ts* as ``[HH:MM]`` in the given IANA timezone."""
    tz = ZoneInfo(tz_name)
    local = ts.astimezone(tz)
    return f"[{local.strftime('%H:%M')}]"


_GAP_THRESHOLD = timedelta(seconds=30)

_FULL_RP_GAP_THRESHOLD = timedelta(minutes=5)
"""Minimum gap between full_rp turns before a cross-context breadcrumb
is considered. Longer than the voice gap threshold because RP scenes
have natural typing pauses that shouldn't trigger breadcrumbs."""


def _format_gap(delta: timedelta) -> str | None:
    """Return a human-readable gap hint, or ``None`` if under threshold.

    Returns phrases like ``(about 2 minutes later)``.
    """
    if delta < _GAP_THRESHOLD:
        return None
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"(about {seconds} seconds later)"
    minutes = seconds // 60
    if minutes < 60:
        return f"(about {minutes} minute{'s' if minutes != 1 else ''} later)"
    hours = minutes // 60
    if hours < 24:
        return f"(about {hours} hour{'s' if hours != 1 else ''} later)"
    days = hours // 24
    return f"(about {days} day{'s' if days != 1 else ''} later)"


def _collect_gap_breadcrumbs(
    cross_summaries: dict[int, str],
    cross_timestamps: dict[int, datetime],
    *,
    gap_start: datetime,
    gap_end: datetime,
) -> str:
    """Return combined breadcrumb text for cross-channel activity during a gap.

    Only includes summaries whose latest activity timestamp falls
    strictly within ``(gap_start, gap_end)``. Returns an empty string
    when no channels qualify.
    """
    parts: list[str] = []
    for channel_id, summary in cross_summaries.items():
        ts = cross_timestamps.get(channel_id)
        if ts is not None and gap_start < ts < gap_end:
            parts.append(summary)
    return " ".join(parts)


def assemble_chat_messages(
    output: PipelineOutput,
    *,
    store: HistoryStore,
    history_window_size: int = _DEFAULT_HISTORY_WINDOW,
    depth_inject_position: int = 0,
    depth_inject_role: str = "system",
    mode: ChannelMode | None = None,
    display_tz: str = "UTC",
) -> list[Message]:
    """Return a list of :class:`Message` ready to hand to the LLM client."""
    request = output.request
    by_layer = output.budget.by_layer

    # 1. system prompt: every layer except recent_history and depth_inject
    system_prompt = _build_system_prompt(by_layer)
    messages: list[Message] = [Message(role="system", content=system_prompt)]

    # 2. recent history as discrete messages; user turns get Speaker:
    # prefix, assistant turns left bare. Per-mode formatting applies
    # timestamps or gap hints.
    turns = store.recent(
        familiar_id=request.familiar_id,
        channel_id=request.channel_id,
        limit=history_window_size,
        mode=mode,
    )

    # pre-fetch cross-context data for full_rp breadcrumbs
    cross_summaries: dict[int, str] = {}
    cross_timestamps: dict[int, datetime] = {}
    if mode is ChannelMode.full_rp:
        others = store.distinct_other_channels(
            familiar_id=request.familiar_id,
            exclude_channel_id=request.channel_id,
        )
        for info in others:
            cached = store.get_cross_context(
                familiar_id=request.familiar_id,
                viewer_mode="full_rp",
                source_channel_id=info.channel_id,
            )
            if cached is not None:
                cross_summaries[info.channel_id] = cached.summary_text
                cross_timestamps[info.channel_id] = info.latest_timestamp

    prev_ts: datetime | None = None
    for turn in turns:
        content = turn.content

        # full_rp gap breadcrumbs: when there's a gap >= 5 minutes AND
        # another channel had activity during that gap, insert a
        # role=system breadcrumb before the resuming turn.
        if mode is ChannelMode.full_rp and prev_ts is not None:
            gap = turn.timestamp - prev_ts
            if gap >= _FULL_RP_GAP_THRESHOLD:
                breadcrumb = _collect_gap_breadcrumbs(
                    cross_summaries,
                    cross_timestamps,
                    gap_start=prev_ts,
                    gap_end=turn.timestamp,
                )
                if breadcrumb:
                    messages.append(
                        Message(role="system", content=breadcrumb),
                    )

        # per-mode formatting on user turns
        if turn.role == "user":
            if mode is ChannelMode.text_conversation_rp:
                ts_prefix = _format_timestamp(turn.timestamp, display_tz)
                content = _prefix_user_content(
                    turn.speaker,
                    turn.content,
                )
                content = f"{ts_prefix} {content}"
            else:
                content = _prefix_user_content(turn.speaker, turn.content)

        # imitate_voice gap hints: prefix the *next* turn's content
        # when the gap from the previous turn exceeds the threshold.
        if mode is ChannelMode.imitate_voice and prev_ts is not None:
            gap = turn.timestamp - prev_ts
            hint = _format_gap(gap)
            if hint:
                content = f"{hint} {content}"

        prev_ts = turn.timestamp

        if turn.role == "user":
            messages.append(
                Message(
                    role="user",
                    content=content,
                    name=turn.speaker,
                ),
            )
        else:
            messages.append(
                Message(role=turn.role, content=content, name=None),
            )

    # 3. interruption-context note — inserted as a system message
    # immediately before the final user turn. Populated only by the
    # voice long-interruption path (Step 8); absent for every normal
    # turn.
    interruption_note = (request.interruption_context or "").strip()
    if interruption_note:
        messages.append(
            Message(role="system", content=interruption_note),
        )

    # 4. pending user turns — falls back to single trigger utterance
    # when none provided (voice path, tests, etc.)
    if request.pending_turns:
        messages.extend(
            Message(
                role="user",
                content=_prefix_user_content(pt.speaker, pt.text),
                name=pt.speaker,
            )
            for pt in request.pending_turns
        )
    else:
        messages.append(
            Message(
                role="user",
                content=_prefix_user_content(request.speaker, request.utterance),
                name=request.speaker,
            ),
        )

    # 5. depth-inject at position-from-end; values larger than the
    # chat buffer clamp to just after the system prompt
    depth_text = by_layer.get(Layer.depth_inject, "").strip()
    if depth_text:
        distance_from_end = max(1, depth_inject_position)
        insert_at = max(1, len(messages) - distance_from_end)
        messages.insert(
            insert_at,
            Message(role=depth_inject_role, content=depth_text),
        )

    return messages


def _build_system_prompt(by_layer: dict[Layer, str]) -> str:
    # speaker-prefix preamble always first so the LLM reads the
    # rendering convention before any character or mode content
    sections: list[str] = [_SPEAKER_PREFIX_PREAMBLE]
    for layer in _SYSTEM_PROMPT_LAYER_ORDER:
        text = by_layer.get(layer, "").strip()
        if text:
            sections.append(text)
    return _SECTION_SEPARATOR.join(sections)
