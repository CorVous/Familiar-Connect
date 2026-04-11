"""Voice interruption detection and response state management.

Implements the interruption flow described in
``future-features/interruption-flow.md``. The module contains:

- :class:`ResponseState` — the three phases of a voice response.
- :class:`InterruptionKind` — short vs long classification.
- :class:`ResponseTracker` — per-guild state machine holding the
  generation task, response text, word timestamps, and playback timing.
- :class:`InterruptionDetector` — watches Deepgram VAD events during
  non-IDLE states and classifies/dispatches interruptions.
- :func:`classify_interruption` — pure classifier for duration → kind.
- :func:`should_keep_talking` — RNG toll check with mood modifier.
- :func:`split_at_elapsed` — word-position tracking for partial resume.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.tts import WordTimestamp

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResponseState(Enum):
    """The three phases of a voice response lifecycle."""

    IDLE = "idle"
    """Not generating or speaking. Interruption detection is off."""

    GENERATING = "generating"
    """LLM call is in-flight. TTS has not started."""

    SPEAKING = "speaking"
    """TTS audio is playing via the Discord voice client."""


class InterruptionKind(Enum):
    """Classification of an interruption by duration."""

    short = "short"
    long = "long"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def classify_interruption(
    duration_s: float,
    *,
    min_interruption_s: float,
    short_long_boundary_s: float,
) -> InterruptionKind | None:
    """Classify a speech interruption by duration.

    :param duration_s: How long the interrupting user spoke.
    :param min_interruption_s: Minimum duration to count as an interruption.
    :param short_long_boundary_s: Threshold separating short from long.
    :returns: ``None`` if below minimum (back-channel noise),
        :attr:`InterruptionKind.short`, or :attr:`InterruptionKind.long`.
    """
    if duration_s < min_interruption_s:
        return None
    if duration_s < short_long_boundary_s:
        return InterruptionKind.short
    return InterruptionKind.long


def should_keep_talking(
    interrupt_tolerance: float,
    mood_modifier: float = 0.0,
    *,
    rng: random.Random | None = None,
) -> bool:
    """Check whether the familiar should keep talking through an interruption.

    :param interrupt_tolerance: Base tolerance from config (0.0-1.0).
    :param mood_modifier: Ephemeral adjustment from mood evaluator.
    :param rng: Optional seeded RNG for deterministic testing.
    :returns: ``True`` if the familiar keeps talking.
    """
    effective = max(0.0, min(1.0, interrupt_tolerance + mood_modifier))
    r = rng if rng is not None else random.Random()  # noqa: S311
    return r.random() < effective


def split_at_elapsed(
    timestamps: list[WordTimestamp],
    elapsed_ms: float,
) -> tuple[str, str]:
    """Split word timestamps at a playback position.

    :param timestamps: Ordered word-level timestamps from TTS.
    :param elapsed_ms: How many milliseconds of audio were played.
    :returns: ``(partial_delivered, remaining_text)`` — the words
        already spoken and the words still to be spoken.
    """
    if not timestamps:
        return ("", "")

    last_spoken_idx = -1
    for i, ts in enumerate(timestamps):
        if ts.end_ms <= elapsed_ms:
            last_spoken_idx = i

    delivered = " ".join(ts.word for ts in timestamps[: last_spoken_idx + 1])
    remaining = " ".join(ts.word for ts in timestamps[last_spoken_idx + 1 :])
    return (delivered, remaining)


# ---------------------------------------------------------------------------
# ResponseTracker
# ---------------------------------------------------------------------------


@dataclass
class ResponseTracker:
    """Per-guild state machine for the voice response lifecycle.

    Tracks which phase the familiar is in (idle, generating, speaking),
    holds references to the in-flight generation task, the completed
    response text, TTS word timestamps, and playback timing. Used by the
    interruption handlers in ``bot.py`` to decide what to do when a user
    speaks.
    """

    state: ResponseState = ResponseState.IDLE
    generation_task: asyncio.Task[str] | None = None
    response_text: str | None = None
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    playback_start_time: float | None = None
    silence_event: asyncio.Event = field(default_factory=asyncio.Event)
    mood_modifier: float = 0.0
    """Cached mood drift modifier, set at generation start."""

    def start_generating(self, task: asyncio.Task[str]) -> None:
        """Transition IDLE → GENERATING with the given task."""
        if self.state is not ResponseState.IDLE:
            msg = f"Cannot start generating from state {self.state.value}"
            raise RuntimeError(msg)
        self.state = ResponseState.GENERATING
        self.generation_task = task
        self.silence_event.clear()

    def generation_complete(self, text: str) -> None:
        """Record the completed response text (still in GENERATING state)."""
        self.response_text = text

    def start_speaking(
        self,
        word_timestamps: list[WordTimestamp] | None = None,
    ) -> None:
        """Transition GENERATING → SPEAKING."""
        if self.state is not ResponseState.GENERATING:
            msg = f"Cannot start speaking from state {self.state.value}"
            raise RuntimeError(msg)
        self.state = ResponseState.SPEAKING
        self.word_timestamps = word_timestamps or []
        self.playback_start_time = time.monotonic()

    def stop_speaking(self) -> float:
        """Stop playback and return elapsed milliseconds.

        :returns: Milliseconds since playback started.
        :raises RuntimeError: If not currently speaking.
        """
        if self.state is not ResponseState.SPEAKING:
            msg = f"Cannot stop speaking from state {self.state.value}"
            raise RuntimeError(msg)
        if self.playback_start_time is None:
            return 0.0
        return (time.monotonic() - self.playback_start_time) * 1000

    def reset(self) -> None:
        """Return to IDLE and clear all transient state."""
        self.state = ResponseState.IDLE
        self.generation_task = None
        self.response_text = None
        self.word_timestamps = []
        self.playback_start_time = None
        self.silence_event.clear()
        self.mood_modifier = 0.0


# ---------------------------------------------------------------------------
# InterruptionEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterruptionEvent:
    """An interruption that has been detected and classified."""

    kind: InterruptionKind
    duration_s: float
    transcript: str
    interrupter_ids: frozenset[int]


# ---------------------------------------------------------------------------
# InterruptionDetector
# ---------------------------------------------------------------------------


class InterruptionDetector:
    """Watches VAD events and dispatches interruption handlers.

    Created once per guild, receives ``on_speech_started`` and
    ``on_utterance_end`` calls from the voice pipeline. Classifies
    interruptions and dispatches to the appropriate handler based on
    the current :class:`ResponseState`.

    :param tracker: The guild's :class:`ResponseTracker`.
    :param min_interruption_s: From :class:`CharacterConfig`.
    :param short_long_boundary_s: From :class:`CharacterConfig`.
    :param on_short_during_generating: Async callback for short
        interruptions while generating.
    :param on_long_during_generating: Async callback for long
        interruptions while generating.
    :param on_short_during_speaking: Async callback for short
        interruptions while speaking.
    :param on_long_during_speaking: Async callback for long
        interruptions while speaking.
    """

    def __init__(
        self,
        *,
        tracker: ResponseTracker,
        min_interruption_s: float,
        short_long_boundary_s: float,
        on_short_during_generating: Callable[[InterruptionEvent], Awaitable[None]],
        on_long_during_generating: Callable[[InterruptionEvent], Awaitable[None]],
        on_short_during_speaking: Callable[[InterruptionEvent], Awaitable[None]],
        on_long_during_speaking: Callable[[InterruptionEvent], Awaitable[None]],
    ) -> None:
        self._tracker = tracker
        self._min_interruption_s = min_interruption_s
        self._short_long_boundary_s = short_long_boundary_s
        self._on_short_during_generating = on_short_during_generating
        self._on_long_during_generating = on_long_during_generating
        self._on_short_during_speaking = on_short_during_speaking
        self._on_long_during_speaking = on_long_during_speaking

        # Interruption accumulation state
        self._speech_start_time: float | None = None
        self._interrupter_ids: set[int] = set()

    def on_speech_started(self, user_id: int, timestamp: float) -> None:
        """Record the start of user speech during a non-IDLE state.

        :param user_id: The speaking user's id.
        :param timestamp: ``time.monotonic()`` value when speech started.
        """
        if self._tracker.state is ResponseState.IDLE:
            return
        if self._speech_start_time is None:
            self._speech_start_time = timestamp
        self._interrupter_ids.add(user_id)

    async def on_utterance_end(
        self,
        user_id: int,
        transcript: str,
        timestamp: float,
    ) -> InterruptionEvent | None:
        """Process the end of user speech and dispatch if it qualifies.

        :param user_id: The speaking user's id.
        :param transcript: What the user said.
        :param timestamp: ``time.monotonic()`` value when speech ended.
        :returns: The :class:`InterruptionEvent` if dispatched, else ``None``.
        """
        if self._tracker.state is ResponseState.IDLE:
            self._reset_accumulation()
            return None

        if self._speech_start_time is None:
            return None

        self._interrupter_ids.add(user_id)
        duration_s = timestamp - self._speech_start_time
        kind = classify_interruption(
            duration_s,
            min_interruption_s=self._min_interruption_s,
            short_long_boundary_s=self._short_long_boundary_s,
        )

        if kind is None:
            self._reset_accumulation()
            return None

        event = InterruptionEvent(
            kind=kind,
            duration_s=duration_s,
            transcript=transcript,
            interrupter_ids=frozenset(self._interrupter_ids),
        )
        self._reset_accumulation()

        state = self._tracker.state
        if state is ResponseState.GENERATING:
            if kind is InterruptionKind.short:
                await self._on_short_during_generating(event)
            else:
                await self._on_long_during_generating(event)
        elif state is ResponseState.SPEAKING:
            if kind is InterruptionKind.short:
                await self._on_short_during_speaking(event)
            else:
                await self._on_long_during_speaking(event)

        return event

    def _reset_accumulation(self) -> None:
        """Clear the in-progress interruption tracking state."""
        self._speech_start_time = None
        self._interrupter_ids.clear()
