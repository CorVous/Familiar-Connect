"""Per-guild response-state tracking and interruption detection.

- :class:`ResponseTracker` — IDLE / GENERATING / SPEAKING state machine
- :class:`InterruptionDetector` — classifies user speech bursts
- :func:`split_at_elapsed` — word-boundary split for mid-playback yield

See ``docs/roadmap/interruption-flow.md`` for design.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    import discord

    from familiar_connect.chattiness import BufferedMessage
    from familiar_connect.tts import WordTimestamp


_logger = logging.getLogger(__name__)


UNSOLICITED_TOLERANCE_BIAS = 0.35
"""Bias added for unsolicited interjections. With ``average`` base
(0.30) yields 0.65 effective — 65% push-through probability."""


def _make_set_event() -> asyncio.Event:
    """Make a pre-set ``asyncio.Event`` so unblocked awaits pass."""
    ev = asyncio.Event()
    ev.set()
    return ev


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
    """Per-guild in-flight voice response state.

    Pure state — behaviour lives in callers (``bot.py``,
    :class:`InterruptionDetector`).
    """

    guild_id: int
    state: ResponseState = ResponseState.IDLE
    generation_task: asyncio.Task[Any] | None = None
    response_text: str | None = None
    timestamps: list[WordTimestamp] = field(default_factory=list)
    playback_start_time: float | None = None
    vc: discord.VoiceClient | None = None
    is_unsolicited: bool = False
    mood_modifier: float = 0.0
    # step 8: pause-then-commit for interruption during GENERATING
    delivery_gate: asyncio.Event = field(default_factory=_make_set_event)
    """Cleared at min-crossed (GENERATING); set on short-finalize or IDLE.
    Awaited by ``_run_voice_response`` before TTS kicks in."""
    cancel_committed: bool = False
    """Set by ``_commit_long``; read post-gate to abort regen path."""
    pending_buffer: list[BufferedMessage] = field(default_factory=list)
    """Original user buffer stashed at GENERATING entry so the dispatcher
    can replay it on the regen path (nothing is in history yet)."""
    # observer installed by InterruptionDetector for burst lifecycle
    on_state_change: Callable[[ResponseState], None] | None = None

    def transition(self, new_state: ResponseState) -> None:
        """Move to *new_state*; same-state transitions silently ignored."""
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
            # reset per-response scratch
            self.generation_task = None
            self.response_text = None
            self.timestamps = []
            self.playback_start_time = None
            self.is_unsolicited = False
            self.mood_modifier = 0.0
            # step 8: release any pending gate + clear commit flag + buffer
            self.delivery_gate.set()
            self.cancel_committed = False
            self.pending_buffer = []
        elif new_state is ResponseState.SPEAKING:
            # stamp playback start for elapsed-time math on yield
            self.playback_start_time = time.monotonic()
        callback = self.on_state_change
        if callback is not None:
            callback(new_state)

    def compute_effective_tolerance(self, base: float) -> float:
        """Clamp ``base + mood + unsolicited bias`` to ``[0, 1]``."""
        bias = UNSOLICITED_TOLERANCE_BIAS if self.is_unsolicited else 0.0
        total = base + self.mood_modifier + bias
        return max(0.0, min(1.0, total))

    def should_keep_talking(
        self,
        base: float,
        *,
        rng: Callable[[], float] | None = None,
    ) -> bool:
        """Roll against effective tolerance; True = push through."""
        tolerance = self.compute_effective_tolerance(base)
        roll = (rng if rng is not None else random.random)()  # noqa: S311
        keep = roll < tolerance
        bias = UNSOLICITED_TOLERANCE_BIAS if self.is_unsolicited else 0.0
        _logger.info(
            "toll: base=%.2f mood=%+.2f unsolicited=%+.2f "
            "effective=%.2f roll=%.2f → %s",
            base,
            self.mood_modifier,
            bias,
            tolerance,
            roll,
            "keep_talking" if keep else "yield",
        )
        return keep


def split_at_elapsed(
    timestamps: list[WordTimestamp],
    elapsed_ms: float,
) -> tuple[list[WordTimestamp], list[WordTimestamp]]:
    """Split *timestamps* into ``(delivered, remaining)`` at *elapsed_ms*.

    Word is delivered once its ``start_ms`` has passed.
    """
    for i, ts in enumerate(timestamps):
        if ts.start_ms >= elapsed_ms:
            return timestamps[:i], timestamps[i:]
    return list(timestamps), []


class ResponseTrackerRegistry:
    """Per-guild :class:`ResponseTracker` lookup; lazy-creates on first use."""

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


class InterruptionClass(Enum):
    """Burst classification by duration against two thresholds."""

    discarded = "discarded"
    """Below ``min_interruption_s`` — back-channel noise."""

    short = "short"
    """Between ``min`` and ``short_long_boundary_s``."""

    long = "long"
    """At or above ``short_long_boundary_s``."""


class _Cancelable(Protocol):
    """Minimal handle protocol: anything with a ``cancel()`` method."""

    def cancel(self) -> None: ...


class InterruptionDetector:
    """Classify user speech bursts against current response state.

    Observes :class:`VoiceLullMonitor` voice-activity events and
    :class:`ResponseTracker` state transitions. Burst begins when
    user is speaking + tracker is not IDLE; finalized after
    ``lull_timeout_s`` silence. Currently detect-only (INFO log).
    """

    def __init__(
        self,
        *,
        tracker_registry: ResponseTrackerRegistry,
        guild_id: int,
        min_interruption_s: float,
        short_long_boundary_s: float,
        lull_timeout_s: float,
        base_tolerance: float,
        clock: Callable[[], float] | None = None,
        scheduler: Callable[[float, Callable[[], None]], _Cancelable] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._tracker_registry = tracker_registry
        self._guild_id = guild_id
        self._min_interruption_s = min_interruption_s
        self._short_long_boundary_s = short_long_boundary_s
        self._lull_timeout_s = lull_timeout_s
        self._base_tolerance = base_tolerance
        self._clock = clock if clock is not None else time.monotonic
        self._scheduler = scheduler
        self._rng = rng

        # users currently speaking (tracked even during IDLE)
        self._speaking: set[int] = set()
        self._burst_started_at: float | None = None
        self._burst_last_ended_at: float | None = None
        # user who opened the burst
        self._burst_starter_id: int | None = None
        # latest non-IDLE state during burst (resolved at log time)
        self._burst_latest_state: ResponseState | None = None
        # pending finalize timer; armed on channel-wide silence
        self._lull_handle: _Cancelable | None = None
        # pending min-threshold timer; fires at min_interruption_s
        self._min_handle: _Cancelable | None = None
        # latch so min-crossed log fires at most once per burst
        self._min_logged: bool = False
        # step 8: transcript finals captured during active burst
        self._burst_transcript: list[str] = []
        # step 8: boundary timer; fires at short_long_boundary_s
        self._long_handle: _Cancelable | None = None
        # step 8: latch so long commit fires at most once per burst
        self._long_committed: bool = False
        # step 8: regen dispatcher (injected by bot.py)
        self._dispatch: Callable[..., Any] | None = None

        # observe tracker transitions to start/abort bursts
        tracker = tracker_registry.get(guild_id)
        tracker.on_state_change = self.on_tracker_state_change

    def set_dispatch(self, dispatch: Callable[..., Any]) -> None:
        """Install regen dispatcher invoked by ``_commit_long``."""
        self._dispatch = dispatch

    def on_transcript(self, user_id: int, text: str) -> None:
        """Append a final transcript to the burst accumulator if active.

        Called by voice_pipeline fan-out alongside the response handler.
        Ignored when no burst is in flight (tracker IDLE or post-finalize).
        """
        del user_id
        if self._burst_started_at is None:
            return
        stripped = text.strip()
        if stripped:
            self._burst_transcript.append(stripped)

    def on_voice_activity(
        self,
        user_id: int,
        event: VoiceActivityEvent,
    ) -> None:
        """Consume per-user speaking transition from :class:`VoiceLullMonitor`."""
        tracker = self._tracker_registry.get(self._guild_id)
        if event is VoiceActivityEvent.started:
            # track globally so IDLE→GENERATING can retroactively start burst
            self._speaking.add(user_id)
            if tracker.state is ResponseState.IDLE:
                return
            # new speech cancels pending finalize; burst keeps growing
            self._cancel_lull()
            if self._burst_started_at is None:
                self._start_burst(user_id)
            # fresh speech may push accumulated duration past min
            self._maybe_log_min_crossed()
            return

        # ended
        self._speaking.discard(user_id)
        if self._burst_started_at is None:
            return
        if self._speaking:
            return
        # all users quiet — arm lull timer to finalize burst
        self._burst_last_ended_at = self._clock()
        self._arm_lull()
        # utterance end may itself cross min threshold
        self._maybe_log_min_crossed()

    def on_tracker_state_change(self, new_state: ResponseState) -> None:
        """Handle tracker transition; may start or abort a burst."""
        if new_state is ResponseState.IDLE:
            if self._burst_started_at is None:
                return
            # preserve burst if familiar was speaking (natural end-of-turn)
            if self._burst_latest_state is ResponseState.SPEAKING:
                return
            self._abort_burst()
            return
        # non-IDLE: record state; retroactively start burst if users speaking
        self._burst_latest_state = new_state
        if self._burst_started_at is None and self._speaking:
            starter = next(iter(self._speaking))
            self._start_burst(starter)
            self._maybe_log_min_crossed()

    def _start_burst(self, starter_id: int) -> None:
        """Begin a new burst attributed to *starter_id*."""
        self._burst_started_at = self._clock()
        self._burst_starter_id = starter_id
        self._min_logged = False
        self._burst_transcript = []
        self._long_committed = False
        tracker = self._tracker_registry.get(self._guild_id)
        # capture tracker's non-IDLE state for log-time resolution
        if tracker.state is not ResponseState.IDLE:
            self._burst_latest_state = tracker.state
        # arm min-crossed and long-boundary timers
        self._arm_min_timer()
        self._arm_long_timer()

    def _abort_burst(self) -> None:
        """Cancel timers and reset burst without classifying."""
        self._cancel_lull()
        self._cancel_min_timer()
        self._cancel_long_timer()
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_starter_id = None
        self._burst_latest_state = None
        self._min_logged = False
        self._burst_transcript = []
        self._long_committed = False

    def _schedule(self, delay: float, callback: Callable[[], None]) -> _Cancelable:
        scheduler = self._scheduler
        if scheduler is None:
            loop = asyncio.get_event_loop()
            scheduler = loop.call_later
        return scheduler(delay, callback)

    def _arm_lull(self) -> None:
        self._cancel_lull()
        self._lull_handle = self._schedule(self._lull_timeout_s, self._finalize_burst)

    def _cancel_lull(self) -> None:
        if self._lull_handle is not None:
            self._lull_handle.cancel()
            self._lull_handle = None

    def _arm_min_timer(self) -> None:
        self._cancel_min_timer()
        self._min_handle = self._schedule(
            self._min_interruption_s,
            self._on_min_crossed,
        )

    def _cancel_min_timer(self) -> None:
        if self._min_handle is not None:
            self._min_handle.cancel()
            self._min_handle = None

    def _on_min_crossed(self) -> None:
        """Timer callback: re-evaluate min crossing."""
        self._min_handle = None
        self._maybe_log_min_crossed()

    def _arm_long_timer(self) -> None:
        self._cancel_long_timer()
        self._long_handle = self._schedule(
            self._short_long_boundary_s,
            self._on_long_boundary_crossed,
        )

    def _cancel_long_timer(self) -> None:
        if self._long_handle is not None:
            self._long_handle.cancel()
            self._long_handle = None

    def _on_long_boundary_crossed(self) -> None:
        """Timer callback: long boundary reached mid-burst."""
        self._long_handle = None
        # gate the commit on active burst + GENERATING + not-yet-committed
        if self._burst_started_at is None or self._long_committed:
            return
        state = self._current_tracker_state()
        if state is not ResponseState.GENERATING:
            # SPEAKING case is handled by Steps 11/12; ignore here.
            return
        self._commit_long()

    def _maybe_log_min_crossed(self) -> None:
        """Log once per burst when accumulated duration crosses ``min``."""
        if (
            self._min_logged
            or self._burst_started_at is None
            or self._burst_starter_id is None
        ):
            return
        if self._speaking:
            effective = self._clock() - self._burst_started_at
        elif self._burst_last_ended_at is not None:
            effective = self._burst_last_ended_at - self._burst_started_at
        else:
            return
        if effective < self._min_interruption_s:
            return
        state = self._current_tracker_state()
        if state is None:
            # tracker returned to IDLE; burst about to be aborted
            return
        _logger.info(
            "interruption: min threshold crossed by user=%s during %s",
            self._burst_starter_id,
            state.value,
        )
        self._min_logged = True
        tracker = self._tracker_registry.get(self._guild_id)
        # step 8: pause TTS/play path until burst finalizes
        if state is ResponseState.GENERATING:
            tracker.delivery_gate.clear()
        # roll tolerance when burst crosses min during SPEAKING
        if state is ResponseState.SPEAKING:
            tracker.should_keep_talking(self._base_tolerance, rng=self._rng)

    def _current_tracker_state(self) -> ResponseState | None:
        """Latest non-IDLE tracker state during active burst."""
        return self._burst_latest_state

    def _commit_long(self) -> None:
        """Cancel generation + dispatch regen. Idempotent per burst."""
        if self._long_committed:
            return
        self._long_committed = True
        starter_id = self._burst_starter_id
        tracker = self._tracker_registry.get(self._guild_id)
        # cancel any live generation task
        task = tracker.generation_task
        if task is not None and not task.done():
            task.cancel()
        # commit: set flag, release gate so awaiter sees cancel_committed
        tracker.cancel_committed = True
        tracker.delivery_gate.set()
        _logger.info(
            "dispatch: long@GENERATING → cancel+regen speaker=%s",
            starter_id,
        )
        dispatch = self._dispatch
        if dispatch is None or starter_id is None:
            return
        transcript = " ".join(self._burst_transcript).strip()
        original_buffer = list(tracker.pending_buffer)
        try:
            dispatch(
                speaker_user_id=starter_id,
                transcript=transcript,
                original_buffer=original_buffer,
                tracker=tracker,
            )
        except Exception:
            _logger.exception("interruption regen dispatch failed")

    def _finalize_burst(self) -> None:
        """Classify + log accumulated burst (lull timer callback)."""
        self._lull_handle = None
        self._cancel_min_timer()
        self._cancel_long_timer()
        started_at = self._burst_started_at
        last_ended_at = self._burst_last_ended_at
        starter_id = self._burst_starter_id
        state = self._current_tracker_state()
        already_committed = self._long_committed
        # reset regardless; next burst starts clean
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_starter_id = None
        self._burst_latest_state = None
        self._min_logged = False
        self._burst_transcript = []
        self._long_committed = False
        if started_at is None or last_ended_at is None or starter_id is None:
            return
        if state is None:
            # defensive; _start_burst always populates _burst_latest_state
            return
        # effective duration: burst start to last speech, excluding lull
        duration = last_ended_at - started_at

        classification = self._classify(duration)
        _logger.info(
            "interruption: %s (%.2fs) by user=%s during %s",
            classification.value,
            duration,
            starter_id,
            state.value,
        )
        # step 8: release gate on short-finalize during GENERATING so bot
        # delivers the original response. long-finalize: if we already
        # committed mid-burst, don't re-dispatch; gate is already released.
        if state is ResponseState.GENERATING and not already_committed:
            tracker = self._tracker_registry.get(self._guild_id)
            tracker.delivery_gate.set()

    def _classify(self, duration: float) -> InterruptionClass:
        if duration < self._min_interruption_s:
            return InterruptionClass.discarded
        if duration < self._short_long_boundary_s:
            return InterruptionClass.short
        return InterruptionClass.long
