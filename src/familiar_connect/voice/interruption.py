"""Voice interruption detection and response state management.

Implements the interruption flow described in
``future-features/interruption-flow.md``. The module contains:

- :class:`ResponseState` — the three phases of a voice response.
- :class:`InterruptionKind` — short vs long classification.
- :class:`ResponseTracker` — per-guild state machine holding the
  generation task, response text, word timestamps, and playback timing.
- :class:`InterruptionDetector` — watches Deepgram VAD events during
  non-IDLE states.  Uses a two-phase model: *moment 1* (toll check
  after ``min_interruption_s``) and *moment 2* (short/long dispatch
  after everyone stops talking + lull).
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


def _idle_event_factory() -> asyncio.Event:
    """Create an :class:`asyncio.Event` that starts **set** (IDLE)."""
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class ResponseTracker:
    """Per-guild state machine for the voice response lifecycle.

    Tracks which phase the familiar is in (idle, generating, speaking),
    holds references to the in-flight generation task, the completed
    response text, TTS word timestamps, and playback timing. Used by the
    interruption handlers in ``bot.py`` to decide what to do when a user
    speaks.

    :attr:`idle_event` is **set** whenever the tracker is IDLE and
    **cleared** when it transitions to GENERATING.  Concurrent handlers
    can ``await idle_event.wait()`` instead of polling — they wake up
    deterministically when :meth:`reset` runs.
    """

    state: ResponseState = ResponseState.IDLE
    generation_task: asyncio.Task[object] | None = None
    response_text: str | None = None
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    playback_start_time: float | None = None
    silence_event: asyncio.Event = field(default_factory=asyncio.Event)
    idle_event: asyncio.Event = field(default_factory=_idle_event_factory)
    """Set when IDLE, cleared when GENERATING. Use for waiting."""
    mood_modifier: float = 0.0
    """Cached mood drift modifier, set at generation start."""

    def start_generating(self, task: asyncio.Task[object]) -> None:
        """Transition IDLE → GENERATING with the given task."""
        if self.state is not ResponseState.IDLE:
            msg = f"Cannot start generating from state {self.state.value}"
            raise RuntimeError(msg)
        self.state = ResponseState.GENERATING
        self.generation_task = task
        self.silence_event.clear()
        self.idle_event.clear()

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
        self.idle_event.set()


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
    ``on_utterance_end`` calls from the voice pipeline.  Uses a
    **two-phase** model:

    *Moment 1* — fires ``min_interruption_s`` after the first
    ``SpeechStarted`` while the familiar is non-IDLE.  Calls
    ``on_interrupt_start`` (the toll check).  If it returns ``True``
    (the familiar yields), tracking continues.  If ``False`` (keeps
    talking), the interruption is discarded.

    *Moment 2* — fires after **all** speakers stop and a
    ``lull_timeout_s`` silence window elapses.  The total duration
    (first ``SpeechStarted`` to last ``UtteranceEnd``) is classified
    as short or long, and the matching handler is dispatched.

    :param tracker: The guild's :class:`ResponseTracker`.
    :param min_interruption_s: Wall-clock seconds before moment 1
        fires.
    :param short_long_boundary_s: Duration threshold separating
        short from long interruptions at moment 2.
    :param lull_timeout_s: Seconds of silence (no active speakers)
        before moment 2 fires.
    :param on_interrupt_start: Moment-1 callback.  Returns ``True``
        if the familiar yields.
    :param on_short_during_generating: Moment-2 handler for short
        interruptions that began while generating.
    :param on_long_during_generating: Moment-2 handler for long
        interruptions that began while generating.
    :param on_short_during_speaking: Moment-2 handler for short
        interruptions that began while speaking.
    :param on_long_during_speaking: Moment-2 handler for long
        interruptions that began while speaking.
    """

    def __init__(
        self,
        *,
        tracker: ResponseTracker,
        min_interruption_s: float,
        short_long_boundary_s: float,
        lull_timeout_s: float,
        on_interrupt_start: Callable[[], Awaitable[bool]],
        on_short_during_generating: Callable[[InterruptionEvent], Awaitable[None]],
        on_long_during_generating: Callable[[InterruptionEvent], Awaitable[None]],
        on_short_during_speaking: Callable[[InterruptionEvent], Awaitable[None]],
        on_long_during_speaking: Callable[[InterruptionEvent], Awaitable[None]],
    ) -> None:
        self._tracker = tracker
        self._min_interruption_s = min_interruption_s
        self._short_long_boundary_s = short_long_boundary_s
        self._lull_timeout_s = lull_timeout_s
        self._on_interrupt_start = on_interrupt_start
        self._on_short_during_generating = on_short_during_generating
        self._on_long_during_generating = on_long_during_generating
        self._on_short_during_speaking = on_short_during_speaking
        self._on_long_during_speaking = on_long_during_speaking

        # Interruption accumulation state
        self._speech_start_time: float | None = None
        self._last_utterance_end_time: float | None = None
        self._interrupter_ids: set[int] = set()
        self._active_speakers: set[int] = set()

        # Phase tracking
        self._threshold_passed: bool = False
        self._moment1_in_progress: bool = False
        self._yielded: bool = False
        self._state_at_interrupt: ResponseState | None = None

        # Async timer tasks
        self._threshold_task: asyncio.Task[None] | None = None
        self._lull_task: asyncio.Task[None] | None = None
        self._deferred_moment1: asyncio.Task[None] | None = None

    # ----- public API -----

    def on_speech_started(self, user_id: int, timestamp: float) -> None:
        """Record the start of user speech during a non-IDLE state.

        :param user_id: The speaking user's id.
        :param timestamp: ``time.monotonic()`` value when speech started.
        """
        if self._tracker.state is ResponseState.IDLE:
            _logger.debug("VAD SpeechStarted user=%d ignored (tracker IDLE)", user_id)
            return
        _logger.debug(
            "VAD SpeechStarted user=%d state=%s active_speakers=%s",
            user_id,
            self._tracker.state.value,
            self._active_speakers,
        )
        self._active_speakers.add(user_id)
        self._interrupter_ids.add(user_id)
        # Cancel pending lull — someone is talking again.
        if self._lull_task is not None:
            self._lull_task.cancel()
            self._lull_task = None
        if self._speech_start_time is None:
            self._speech_start_time = timestamp
            _logger.info(
                "Interruption tracking started (user=%d, state=%s)",
                user_id,
                self._tracker.state.value,
            )
            self._threshold_task = asyncio.create_task(self._threshold_check())
        elif self._threshold_passed and not self._yielded:
            # Threshold already passed but nobody was speaking when it
            # fired.  Now someone started again — fire moment 1.
            self._deferred_moment1 = asyncio.create_task(self._fire_moment1())

    async def on_utterance_end(
        self,
        user_id: int,
        transcript: str,  # noqa: ARG002
        timestamp: float,
    ) -> None:
        """Record the end of user speech and start the lull timer if all silent.

        :param user_id: The speaking user's id.
        :param transcript: What the user said (unused — real transcripts
            arrive via the transcription pipeline).
        :param timestamp: ``time.monotonic()`` value when speech ended.
        """
        if self._tracker.state is ResponseState.IDLE:
            _logger.debug("VAD UtteranceEnd user=%d ignored (tracker IDLE)", user_id)
            self._reset_accumulation()
            return

        if self._speech_start_time is None:
            return

        self._interrupter_ids.add(user_id)
        self._last_utterance_end_time = timestamp
        self._active_speakers.discard(user_id)
        _logger.debug(
            "VAD UtteranceEnd user=%d remaining_speakers=%s",
            user_id,
            self._active_speakers,
        )

        if not self._active_speakers:
            _logger.debug("All speakers silent — starting lull timer")
            self._start_lull_timer()

    # ----- internal timers -----

    async def _threshold_check(self) -> None:
        """Wait ``min_interruption_s`` then fire moment 1 if appropriate."""
        await asyncio.sleep(self._min_interruption_s)
        self._threshold_task = None
        self._threshold_passed = True
        if self._tracker.state is ResponseState.IDLE or self._speech_start_time is None:
            _logger.debug("Threshold passed but tracker now IDLE — skipping moment 1")
            return
        if self._active_speakers:
            _logger.info(
                "Moment 1: threshold passed, active speakers=%s — firing toll check",
                self._active_speakers,
            )
            await self._fire_moment1()
        else:
            _logger.debug(
                "Threshold passed but no active speakers — deferring moment 1"
            )
        # If nobody is speaking, _fire_moment1 deferred to next
        # on_speech_started (via _threshold_passed flag).

    async def _fire_moment1(self) -> None:
        """Run the toll-check callback and either yield or discard."""
        if self._moment1_in_progress or self._yielded:
            return
        self._moment1_in_progress = True
        self._state_at_interrupt = self._tracker.state
        yielded = await self._on_interrupt_start()
        self._moment1_in_progress = False
        # Accumulation may have been reset while we were awaiting.
        if self._speech_start_time is None:
            return
        if yielded:
            self._yielded = True
        else:
            self._reset_accumulation()

    def _start_lull_timer(self) -> None:
        if self._lull_task is not None:
            self._lull_task.cancel()
        self._lull_task = asyncio.create_task(self._lull_expired())

    async def _lull_expired(self) -> None:
        """Moment 2: classify the interruption and dispatch."""
        await asyncio.sleep(self._lull_timeout_s)
        self._lull_task = None

        if not self._yielded:
            # Moment 1 never fired or the familiar didn't yield.
            _logger.debug("Lull expired but familiar did not yield — resetting")
            self._reset_accumulation()
            return

        if self._speech_start_time is None or self._last_utterance_end_time is None:
            self._reset_accumulation()
            return

        duration_s = self._last_utterance_end_time - self._speech_start_time
        kind = classify_interruption(
            duration_s,
            min_interruption_s=self._min_interruption_s,
            short_long_boundary_s=self._short_long_boundary_s,
        )

        if kind is None:
            self._reset_accumulation()
            return

        event = InterruptionEvent(
            kind=kind,
            duration_s=duration_s,
            transcript="",
            interrupter_ids=frozenset(self._interrupter_ids),
        )
        state = self._state_at_interrupt
        self._reset_accumulation()

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

    def _reset_accumulation(self) -> None:
        """Clear the in-progress interruption tracking state."""
        self._speech_start_time = None
        self._last_utterance_end_time = None
        self._interrupter_ids.clear()
        self._active_speakers.clear()
        self._threshold_passed = False
        self._moment1_in_progress = False
        self._yielded = False
        self._state_at_interrupt = None
        if self._threshold_task is not None:
            self._threshold_task.cancel()
            self._threshold_task = None
        if self._lull_task is not None:
            self._lull_task.cancel()
            self._lull_task = None
