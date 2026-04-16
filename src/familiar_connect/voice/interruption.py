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

from familiar_connect import log_style as ls
from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    import discord

    from familiar_connect.tts import WordTimestamp


_logger = logging.getLogger(__name__)


UNSOLICITED_TOLERANCE_BIAS = 0.35
"""Bias added for unsolicited interjections. With ``average`` base
(0.30) yields 0.65 effective — 65% push-through probability."""


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
    interruption_elapsed_ms: float | None = None
    vc: discord.VoiceClient | None = None
    is_unsolicited: bool = False
    mood_modifier: float = 0.0
    # observer installed by InterruptionDetector for burst lifecycle
    on_state_change: Callable[[ResponseState], None] | None = None
    # Step 12 — long-SPEAKING-yield scratch. Set by InterruptionDetector
    # at Moment 1 (yield) and finalization; awaited + consumed by
    # _run_voice_response. (``interruption_elapsed_ms`` declared above.)
    interrupt_event: asyncio.Event | None = None
    interrupt_classification: InterruptionClass | None = None
    interrupt_transcript: str = ""
    interrupt_starter_name: str = ""
    # Step 9 — transcripts captured from short@GENERATING bursts. Stashed
    # here so _run_voice_response can flush them after the original
    # buffer write (preserves chronology: buffer → interrupter →
    # assistant reply). Each entry is (resolved_name, transcript).
    pending_interrupter_turns: list[tuple[str, str]] = field(default_factory=list)
    # Step 11 bug-fix — set by InterruptionDetector when a short@SPEAKING
    # yield+resume task is in flight. _deliver_to_monitor checks this flag
    # and skips the lull if set, preventing the concurrent voice endpointing
    # lull from transitioning IDLE→GENERATING and causing the resume task to
    # bail (the race that silences the resumed audio). Cleared by
    # _on_short_yield_resume and on IDLE transition.
    short_yield_pending: bool = False

    def transition(self, new_state: ResponseState) -> None:
        """Move to *new_state*; same-state transitions silently ignored."""
        old = self.state
        if old is new_state:
            return
        self.state = new_state
        state_str = f"{ls.LC}{old.value}{ls.RS}→{ls.LC}{new_state.value}{ls.RS}"
        _logger.info(
            f"{ls.tag('🔄 State', ls.C)} "
            f"{ls.kv('guild', str(self.guild_id))} "
            f"{ls.W}state={ls.RS}{state_str} "
            f"{ls.kv('unsolicited', str(self.is_unsolicited), vc=ls.LW)}"
        )
        if new_state is ResponseState.IDLE:
            # reset per-response scratch
            self.generation_task = None
            self.response_text = None
            self.timestamps = []
            self.playback_start_time = None
            self.interruption_elapsed_ms = None
            self.is_unsolicited = False
            self.mood_modifier = 0.0
            # Step 12 yield scratch
            self.interruption_elapsed_ms = None
            self.interrupt_event = None
            self.interrupt_classification = None
            self.interrupt_transcript = ""
            self.interrupt_starter_name = ""
            # Step 9 — clear any leftover interrupter turns not yet
            # flushed (e.g., response discarded before delivery).
            self.pending_interrupter_turns = []
            self.short_yield_pending = False
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
        outcome = "keep_talking" if keep else "yield"
        out_color = ls.G if keep else ls.Y
        _logger.info(
            f"{ls.tag('🎲 Toll', ls.C)} "
            f"{ls.kv('base', f'{base:.2f}', vc=ls.LW)} "
            f"{ls.kv('mood', f'{self.mood_modifier:+.2f}', vc=ls.LW)} "
            f"{ls.kv('unsolicited', f'{bias:+.2f}', vc=ls.LW)} "
            f"{ls.kv('effective', f'{tolerance:.2f}', vc=ls.LW)} "
            f"{ls.kv('roll', f'{roll:.2f}', vc=ls.LW)} "
            f"{ls.word('→', ls.W)} "
            f"{ls.word(outcome, out_color)}"
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
        on_long_during_generating: Callable[[int, str], None] | None = None,
        on_long_boundary_crossed: Callable[[int, str], None] | None = None,
        on_short_yield_resume: (
            Callable[[list[WordTimestamp]], Coroutine[Any, Any, None]] | None
        ) = None,
        on_push_through_transcript: Callable[[int, str], None] | None = None,
        name_resolver: Callable[[int], str] | None = None,
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
        # Sync callback invoked when a burst finalizes as ``long`` during
        # ``GENERATING``. Called with ``(starter_id, transcript)``. The
        # caller (bot.py) wraps the async cancel+regen work in
        # ``asyncio.create_task`` so the detector stays synchronous.
        self._on_long_during_generating = on_long_during_generating
        # Sync callback invoked immediately when burst crosses
        # ``short_long_boundary_s`` during ``GENERATING`` — before the lull
        # fires. Used by bot.py to cancel the in-flight LLM task early.
        self._on_long_boundary_crossed = on_long_boundary_crossed
        # Step 11 dispatch callbacks for short@SPEAKING:
        # - ``on_short_yield_resume``: async callback invoked at finalize
        #   time with the remaining-word timestamps when the familiar
        #   yielded mid-playback. Caller re-synthesises the remainder.
        # - ``on_push_through_transcript``: sync callback with the
        #   interrupter's transcript when tolerance chose push-through.
        self._on_short_yield_resume = on_short_yield_resume
        self._on_push_through_transcript = on_push_through_transcript
        # Step 12: resolver maps user_id → display name for the long@SPEAKING
        # regen interruption_context ("Alice interrupted you"). bot.py
        # supplies ``lambda uid: user_names.get(uid, f"User-{uid}")``.
        self._name_resolver = name_resolver

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
        # Pending long-boundary timer. Armed at burst start for
        # ``short_long_boundary_s`` seconds; fires ``on_long_boundary_crossed``
        # the instant accumulated duration crosses the threshold while GENERATING.
        self._long_handle: _Cancelable | None = None
        # Latch so the boundary-crossed callback fires at most once per burst.
        self._long_fired: bool = False
        # Deepgram finals accumulated during the active burst. Cleared at
        # each new burst start and at finalize. Passed to the dispatch
        # callback so the regen call can include the user's words.
        self._burst_transcript: str = ""
        # Timestamps of the remaining (not-yet-played) words captured at
        # stop-time on a SPEAKING yield. Step 11 dispatch consumes these
        # to re-synthesise the unspoken tail after the lull confirms short.
        self._remaining_timestamps: list[WordTimestamp] = []
        # True when _maybe_log_min_crossed stopped the vc this burst.
        self._did_yield: bool = False
        # Delivery gate: cleared at burst start, set at finalize/abort.
        # _run_voice_response awaits this before transitioning to SPEAKING
        # so _burst_latest_state stays GENERATING at _finalize_burst time
        # even when TTS synthesis spans the lull window.
        self._delivery_gate: asyncio.Event = asyncio.Event()
        self._delivery_gate.set()  # no burst initially → gate open
        self._last_classification: InterruptionClass | None = None

        # observe tracker transitions to start/abort bursts
        tracker = tracker_registry.get(guild_id)
        tracker.on_state_change = self.on_tracker_state_change

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
            self._maybe_dispatch_long_cancel()
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
        self._maybe_dispatch_long_cancel()

    def on_tracker_state_change(self, new_state: ResponseState) -> None:
        """Handle tracker transition; may start or abort a burst."""
        if new_state is ResponseState.IDLE:
            if self._burst_started_at is None:
                return
            # preserve burst if familiar was speaking (natural end-of-turn)
            if self._burst_latest_state is ResponseState.SPEAKING:
                return
            # Early-cancel path: LLM task cancelled while user is still
            # talking. Keep the burst alive so the lull can finalize it
            # and fire on_long_during_generating → regen.
            if self._speaking:
                return
            self._abort_burst()
            return
        # non-IDLE: record state; retroactively start burst if users speaking
        self._burst_latest_state = new_state
        if self._burst_started_at is None and self._speaking:
            starter = next(iter(self._speaking))
            self._start_burst(starter)
            self._maybe_log_min_crossed()

    def on_transcript(self, user_id: int, text: str) -> None:  # noqa: ARG002
        """Accumulate a Deepgram final into the burst transcript.

        Wire from the voice pipeline's ``_route_transcript_to_monitor``
        alongside the existing :class:`~familiar_connect.voice_lull.VoiceLullMonitor`
        call so the detector captures what the interrupting user said.
        Ignored when no burst is active. ``user_id`` is unused (all
        users in a multi-user burst contribute to the same transcript).
        """
        if self._burst_started_at is None:
            return
        if self._burst_transcript:
            self._burst_transcript += " " + text
        else:
            self._burst_transcript = text

    async def wait_for_lull(self) -> InterruptionClass | None:
        """Await active burst finalization; return classification or None.

        Returns ``None`` immediately if no burst is active (gate already
        open). Captures the gate reference before awaiting so a new burst
        starting mid-TTS gets a fresh closed gate — the caller awaits
        THAT burst rather than the already-finalized one.
        """
        gate = self._delivery_gate
        if gate.is_set():
            return None
        await gate.wait()
        return self._last_classification

    def _start_burst(self, starter_id: int) -> None:
        """Begin a new burst attributed to *starter_id*."""
        self._burst_started_at = self._clock()
        self._burst_starter_id = starter_id
        self._min_logged = False
        self._long_fired = False
        self._burst_transcript = ""
        # Fresh event per burst so regen's wait_for_lull() captures the
        # new (closed) gate rather than the stale (open) one from the
        # previous burst, preventing false long-classification returns.
        self._delivery_gate = asyncio.Event()
        tracker = self._tracker_registry.get(self._guild_id)
        # capture tracker's non-IDLE state for log-time resolution
        if tracker.state is not ResponseState.IDLE:
            self._burst_latest_state = tracker.state
        # arm min-crossed timer
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
        self._long_fired = False
        self._burst_transcript = ""
        self._remaining_timestamps = []
        self._did_yield = False
        self._last_classification = None
        self._delivery_gate.set()

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

    def _arm_long_timer(self) -> None:
        self._cancel_long_timer()
        self._long_handle = self._schedule(
            self._short_long_boundary_s,
            self._on_long_crossed,
        )

    def _cancel_long_timer(self) -> None:
        if self._long_handle is not None:
            self._long_handle.cancel()
            self._long_handle = None

    def _on_long_crossed(self) -> None:
        """Timer callback: re-evaluate whether the long boundary was crossed."""
        self._long_handle = None
        self._maybe_dispatch_long_cancel()

    def _maybe_dispatch_long_cancel(self) -> None:
        """Cancel generation immediately if burst ≥ boundary while GENERATING.

        Parallel to ``_maybe_log_min_crossed``; latch ``_long_fired`` so
        the callback fires at most once per burst.
        """
        if (
            self._long_fired
            or self._burst_started_at is None
            or self._burst_starter_id is None
            or self._on_long_boundary_crossed is None
        ):
            return
        if self._speaking:
            effective = self._clock() - self._burst_started_at
        elif self._burst_last_ended_at is not None:
            effective = self._burst_last_ended_at - self._burst_started_at
        else:
            return
        if effective < self._short_long_boundary_s:
            return
        tracker = self._tracker_registry.get(self._guild_id)
        if tracker.state is not ResponseState.GENERATING:
            return
        self._long_fired = True
        _logger.info(
            f"{ls.tag('⚠️ Interrupt', ls.LY)} "
            f"{ls.kv('user', str(self._burst_starter_id), vc=ls.LC)} "
            f"{ls.kv('type', 'long_boundary_early', vc=ls.LY)}"
        )
        self._on_long_boundary_crossed(self._burst_starter_id, self._burst_transcript)

    def _on_min_crossed(self) -> None:
        """Timer callback: re-evaluate min crossing."""
        self._min_handle = None
        self._maybe_log_min_crossed()

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
        speaker = (
            self._name_resolver(self._burst_starter_id)
            if self._name_resolver is not None
            else str(self._burst_starter_id)
        )
        _logger.info(
            f"{ls.tag('⚠️ Interrupt', ls.LY)} "
            f"{ls.kv('speaker', speaker, vc=ls.LC)} "
            f"{ls.kv('during', state.value, vc=ls.LW)} "
            f"{ls.kv('type', 'min_threshold', vc=ls.LY)}"
        )
        self._min_logged = True
        # Moment 1: burst crossed ``min`` while the familiar is speaking.
        # Roll tolerance; on yield: stop playback, capture elapsed + remaining
        # timestamps (Step 11 resume), set starter_name + interrupt_event
        # (Step 12 regen). Single yield latch drives both dispatch paths.
        if state is ResponseState.SPEAKING:
            tracker = self._tracker_registry.get(self._guild_id)
            keep = tracker.should_keep_talking(self._base_tolerance, rng=self._rng)
            if not keep:
                if tracker.vc is not None and tracker.vc.is_playing():
                    tracker.vc.stop()
                if tracker.playback_start_time is not None:
                    elapsed_ms = (self._clock() - tracker.playback_start_time) * 1000
                    tracker.interruption_elapsed_ms = elapsed_ms
                    delivered, remaining_ts = split_at_elapsed(
                        tracker.timestamps, elapsed_ms
                    )
                    self._remaining_timestamps = remaining_ts
                    _logger.info(
                        f"{ls.tag('Yield Split', ls.Y)} "
                        f"{ls.kv('elapsed', f'{elapsed_ms:.0f}ms', vc=ls.LW)} "
                        f"{ls.kv('words', str(len(tracker.timestamps)), vc=ls.LW)} "
                        f"{ls.kv('delivered', str(len(delivered)), vc=ls.LW)} "
                        f"{ls.kv('remaining', str(len(remaining_ts)), vc=ls.LW)}"
                    )
                starter = self._burst_starter_id
                tracker.interrupt_starter_name = (
                    self._name_resolver(starter)
                    if self._name_resolver is not None and starter is not None
                    else f"User {starter}"
                )
                tracker.interrupt_event = asyncio.Event()
                self._did_yield = True

    def _current_tracker_state(self) -> ResponseState | None:
        """Latest non-IDLE tracker state during active burst."""
        return self._burst_latest_state

    def _finalize_burst(self) -> None:
        """Classify + log accumulated burst (lull timer callback)."""
        self._lull_handle = None
        self._cancel_min_timer()
        self._cancel_long_timer()
        started_at = self._burst_started_at
        last_ended_at = self._burst_last_ended_at
        starter_id = self._burst_starter_id
        state = self._current_tracker_state()
        transcript = self._burst_transcript
        remaining = self._remaining_timestamps
        did_yield = self._did_yield
        # Reset regardless; a new burst starts from a clean slate.
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_starter_id = None
        self._burst_latest_state = None
        self._min_logged = False
        self._long_fired = False
        self._burst_transcript = ""
        self._remaining_timestamps = []
        self._did_yield = False
        if started_at is None or last_ended_at is None or starter_id is None:
            return
        if state is None:
            # defensive; _start_burst always populates _burst_latest_state
            return
        # effective duration: burst start to last speech, excluding lull
        duration = last_ended_at - started_at

        classification = self._classify(duration)
        self._last_classification = classification
        self._delivery_gate.set()
        speaker = (
            self._name_resolver(starter_id)
            if self._name_resolver is not None
            else str(starter_id)
        )
        _logger.info(
            f"{ls.tag('⚠️ Interrupt', ls.Y)} "
            f"{ls.kv('type', classification.value, vc=ls.LY)} "
            f"{ls.kv('duration', f'{duration:.2f}s', vc=ls.LW)} "
            f"{ls.kv('speaker', speaker, vc=ls.LC)} "
            f"{ls.kv('during', state.value, vc=ls.LW)}"
        )
        # Step 8 dispatch: long burst during GENERATING → cancel + regen.
        if (
            classification is InterruptionClass.long
            and state is ResponseState.GENERATING
            and self._on_long_during_generating is not None
        ):
            _logger.info(
                f"{ls.tag('🔁 Cancel+Regen', ls.Y)} "
                f"{ls.kv('speaker', str(starter_id), vc=ls.LC)} "
                f"{ls.kv('trigger', 'long@GENERATING', vc=ls.LY)}"
            )
            self._on_long_during_generating(starter_id, transcript)
        # Step 11 dispatch: short interruption during SPEAKING.
        if (
            classification is InterruptionClass.short
            and state is ResponseState.SPEAKING
        ):
            outcome = "yield+resume" if did_yield else "push-through"
            _logger.info(
                f"{ls.tag('✋ Yield', ls.Y)} "
                f"{ls.kv('speaker', str(starter_id), vc=ls.LC)} "
                f"{ls.kv('outcome', outcome, vc=ls.LY)}"
            )
            if did_yield:
                cb = self._on_short_yield_resume
                if remaining:
                    # Set before scheduling so _deliver_to_monitor (which may
                    # fire in the same event-loop window) sees the flag and
                    # suppresses the concurrent voice endpointing lull.
                    t = self._tracker_registry.get(self._guild_id)
                    t.short_yield_pending = True
                    if cb is not None:
                        asyncio.create_task(cb(remaining))  # noqa: RUF006
            else:
                cb2 = self._on_push_through_transcript
                if cb2 is not None and transcript:
                    cb2(starter_id, transcript)
        # Step 11b dispatch: long interruption during SPEAKING with push-through.
        # Familiar kept talking (did_yield=False) but user spoke for a long time.
        # Forward transcript to push-through callback so it lands in history.
        if (
            classification is InterruptionClass.long
            and state is ResponseState.SPEAKING
            and not did_yield
        ):
            push_speaker = (
                self._name_resolver(starter_id)
                if self._name_resolver is not None
                else str(starter_id)
            )
            _logger.info(
                f"{ls.tag('➡️ Push Through', ls.C)} "
                f"{ls.kv('speaker', push_speaker, vc=ls.LC)} "
                f"{ls.kv('trigger', 'long@SPEAKING', vc=ls.LW)}"
            )
            cb3 = self._on_push_through_transcript
            if cb3 is not None and transcript:
                cb3(starter_id, transcript)
        # Step 9 dispatch: short interruption during GENERATING.
        # Familiar keeps generating; delivery gate already opens on
        # finalize so playback proceeds naturally. Stash the interrupter
        # transcript on the tracker so _run_voice_response can flush it
        # to history after the original buffer write (preserves chronology).
        if (
            classification is InterruptionClass.short
            and state is ResponseState.GENERATING
        ):
            _logger.info(
                f"{ls.tag('⏳ Polite Wait', ls.C)} "
                f"{ls.kv('speaker', str(starter_id), vc=ls.LC)} "
                f"{ls.kv('trigger', 'short@GENERATING', vc=ls.LW)}"
            )
            if transcript.strip():
                tracker = self._tracker_registry.get(self._guild_id)
                resolved = (
                    self._name_resolver(starter_id)
                    if self._name_resolver is not None
                    else f"User-{starter_id}"
                )
                tracker.pending_interrupter_turns.append((resolved, transcript))
        # Step 12: if yield was decided at Moment 1, deliver classification
        # and transcript to the tracker so _run_voice_response can dispatch.
        # Fires for both short and long yields — consumer picks based on
        # tracker.interrupt_classification.
        if did_yield:
            tracker = self._tracker_registry.get(self._guild_id)
            tracker.interrupt_classification = classification
            tracker.interrupt_transcript = transcript
            if tracker.interrupt_event is not None:
                tracker.interrupt_event.set()

    def _classify(self, duration: float) -> InterruptionClass:
        if duration < self._min_interruption_s:
            return InterruptionClass.discarded
        if duration < self._short_long_boundary_s:
            return InterruptionClass.short
        return InterruptionClass.long
