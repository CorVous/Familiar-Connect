"""Per-guild response-state tracking for voice interruption handling.

This module holds the :class:`ResponseTracker` — a small state machine
that knows whether the familiar is currently idle, generating a reply,
or speaking a reply on a voice channel. It is the prerequisite for the
interruption detector in the same module (added in a later step) and
for cancellable LLM / mid-playback yield logic in ``bot.py``.

Today the tracker is **observational only**: transitions are logged at
INFO but no interrupt dispatch happens yet. The fields are shaped for
the scenarios documented in
``docs/roadmap/interruption-flow.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    import discord

    from familiar_connect.tts import WordTimestamp


_logger = logging.getLogger(__name__)


class ResponseState(Enum):
    """Where the familiar sits in the voice-reply lifecycle."""

    IDLE = "IDLE"
    """No reply in flight. Interruption detection is disabled."""

    GENERATING = "GENERATING"
    """An LLM call is in progress. TTS hasn't started."""

    SPEAKING = "SPEAKING"
    """TTS audio is playing via ``vc.play()``."""


@dataclass
class ResponseTracker:
    """Per-guild snapshot of an in-flight voice response.

    One tracker exists per voice-connected guild. The tracker owns:

    - the current :class:`ResponseState`,
    - the cancellable :class:`asyncio.Task` running the LLM call,
    - the complete reply text (once generation finishes),
    - Cartesia's per-word timestamps (for word-boundary splits),
    - the playback start time (``time.monotonic()``) for elapsed-time
      math when an interruption arrives during ``SPEAKING``,
    - a reference to the Discord :class:`VoiceClient` so interrupt
      handlers can call ``vc.stop()`` without threading it through every
      helper,
    - flags needed by the tolerance system: ``is_unsolicited`` (was the
      reply started as an interjection/lull?) and ``mood_modifier``
      (ephemeral per-response drift cached at generation time).

    The tracker is pure state — behaviour lives in
    :func:`transition`, ``bot.py`` (which drives transitions), and the
    upcoming ``InterruptionDetector``.
    """

    guild_id: int
    state: ResponseState = ResponseState.IDLE
    generation_task: asyncio.Task[str] | None = None
    response_text: str | None = None
    timestamps: list[WordTimestamp] = field(default_factory=list)
    playback_start_time: float | None = None
    vc: discord.VoiceClient | None = None
    is_unsolicited: bool = False
    mood_modifier: float = 0.0

    def transition(self, new_state: ResponseState) -> None:
        """Move to *new_state* and log the transition at INFO.

        Same-state transitions are silently ignored. This matters for
        the voice lull path: the voice pipeline transitions to
        ``GENERATING`` before the side-model YES/NO eval, and on YES
        :func:`_run_voice_response` also transitions to ``GENERATING``
        — making the call a no-op keeps the log honest instead of
        showing a spurious ``GENERATING→GENERATING`` line.

        The method is deliberately permissive — it does not enforce a
        strict state graph because the real constraints (must cancel the
        task before leaving ``GENERATING``, must ``vc.stop()`` before
        leaving ``SPEAKING``) live in the callers. Enforcing here would
        duplicate that logic and make tests harder to write. Instead,
        the log line makes unexpected sequences easy to spot.
        """
        old = self.state
        if old is new_state:
            return
        self.state = new_state
        _logger.info(
            "tracker guild=%s state: %s→%s (unsolicited=%s)",
            self.guild_id,
            old.value,
            new_state.value,
            self.is_unsolicited,
        )
        if new_state is ResponseState.IDLE:
            # Reset per-response scratch so the next turn starts clean.
            self.generation_task = None
            self.response_text = None
            self.timestamps = []
            self.playback_start_time = None
            self.is_unsolicited = False
            self.mood_modifier = 0.0


class ResponseTrackerRegistry:
    """Per-guild lookup for :class:`ResponseTracker` instances.

    A thin wrapper over a dict so call sites don't have to care about
    lazy creation. The interruption detector and ``bot.py`` both
    resolve a tracker the same way: ``registry.get(guild_id)``.
    """

    def __init__(self) -> None:
        self._by_guild: dict[int, ResponseTracker] = {}

    def get(self, guild_id: int) -> ResponseTracker:
        """Return the tracker for *guild_id*, creating one on first use."""
        tracker = self._by_guild.get(guild_id)
        if tracker is None:
            tracker = ResponseTracker(guild_id=guild_id)
            self._by_guild[guild_id] = tracker
        return tracker

    def drop(self, guild_id: int) -> None:
        """Remove *guild_id*'s tracker (called on voice disconnect)."""
        self._by_guild.pop(guild_id, None)

    def snapshot(self) -> dict[int, ResponseTracker]:
        """Return a shallow copy of the guild→tracker map (for tests)."""
        return dict(self._by_guild)
