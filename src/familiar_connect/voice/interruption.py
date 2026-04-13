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

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    import discord

    from familiar_connect.tts import WordTimestamp


_logger = logging.getLogger(__name__)


UNSOLICITED_TOLERANCE_BIAS = 0.35
"""Added to :meth:`ResponseTracker.compute_effective_tolerance` when
the response was started as an unsolicited interjection (chattiness /
voice-lull path). Unsolicited remarks should tend to push through
interruptions — if the familiar spoke up on its own it's more
committed to finishing the thought than if it was directly addressed.
The value is deliberately large: combined with ``average`` tolerance
(0.30) it yields ``0.65`` effective, i.e. a 65%% probability of
pushing through, while leaving room for ``very_meek`` + negative mood
to still yield cleanly."""


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
    generation_task: asyncio.Task[Any] | None = None
    response_text: str | None = None
    timestamps: list[WordTimestamp] = field(default_factory=list)
    playback_start_time: float | None = None
    interruption_elapsed_ms: float | None = None
    vc: discord.VoiceClient | None = None
    is_unsolicited: bool = False
    mood_modifier: float = 0.0
    # Optional observer invoked after every non-no-op transition.
    # Installed by :class:`InterruptionDetector` so it can retroactively
    # start bursts (pre-existing speech carrying into GENERATING) and
    # abort bursts when the familiar returns to IDLE.
    on_state_change: Callable[[ResponseState], None] | None = None

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
            self.interruption_elapsed_ms = None
            self.is_unsolicited = False
            self.mood_modifier = 0.0
        elif new_state is ResponseState.SPEAKING:
            # Stamp the wall clock at the moment audio begins so
            # Step 11's yield path can compute elapsed_ms without
            # threading the start time through every caller.
            self.playback_start_time = time.monotonic()
        callback = self.on_state_change
        if callback is not None:
            callback(new_state)

    def compute_effective_tolerance(self, base: float) -> float:
        """Combine *base* tolerance with mood + unsolicited biases.

        :param base: The familiar's configured
            :attr:`~familiar_connect.config.InterruptTolerance.base_probability`.
        :returns: ``clamp(base + mood_modifier + unsolicited_bias, 0,
            1)``. ``unsolicited_bias`` is
            :data:`UNSOLICITED_TOLERANCE_BIAS` when
            :attr:`is_unsolicited` is true, else ``0.0``.
        """
        bias = UNSOLICITED_TOLERANCE_BIAS if self.is_unsolicited else 0.0
        total = base + self.mood_modifier + bias
        return max(0.0, min(1.0, total))

    def should_keep_talking(
        self,
        base: float,
        *,
        rng: Callable[[], float] | None = None,
    ) -> bool:
        """Roll against the effective tolerance; log the outcome.

        Called by the interruption detector the moment a burst
        crosses ``min_interruption_s`` while the familiar is
        ``SPEAKING``. A ``True`` result means push through — keep
        playing; ``False`` means yield — stop playback and let the
        user finish. Dispatch on the result lands in Step 11; this
        method is intentionally logged even when no-one acts on it so
        operators can tune ``interrupt_tolerance`` from voice-session
        logs alone.

        :param base: See :meth:`compute_effective_tolerance`.
        :param rng: Optional zero-argument callable returning a float
            in ``[0, 1)``. Defaults to :func:`random.random`. Injected
            in tests for deterministic rolls.
        """
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
    """Partition *timestamps* into delivered and remaining at *elapsed_ms*.

    Called when the familiar yields mid-playback: we need to know
    which words already came out of the speaker (history + context
    for a regenerated reply) and which are still pending (re-synthesis
    on resume). A word is treated as **delivered** the moment its
    ``start_ms`` has passed — even if playback was cut partway
    through — so resumption begins on the next clean word boundary
    and never re-plays a half-spoken word.

    :param timestamps: Per-word windows from
        :class:`familiar_connect.tts.TTSResult`. Assumed to be in
        playback order; no sort is performed.
    :param elapsed_ms: Milliseconds since audio began (``(now -
        tracker.playback_start_time) * 1000``).
    :returns: ``(delivered, remaining)`` — two new lists whose
        concatenation equals the input.
    """
    for i, ts in enumerate(timestamps):
        if ts.start_ms >= elapsed_ms:
            return timestamps[:i], timestamps[i:]
    return list(timestamps), []


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


class InterruptionClass(Enum):
    """How a user speech burst during a voice response is classified.

    The detector measures the burst duration from the first
    ``VoiceActivityEvent.started`` to the moment all users go silent
    (last ``VoiceActivityEvent.ended``) and bins it against two
    thresholds from :class:`~familiar_connect.config.CharacterConfig`.
    """

    discarded = "discarded"
    """Burst was shorter than ``min_interruption_s`` — back-channel
    noise ("mm-hm", laughter), not a real interruption."""

    short = "short"
    """Burst was ≥ ``min_interruption_s`` but < ``short_long_boundary_s``
    — a polite pause that only yields the floor briefly."""

    long = "long"
    """Burst was ≥ ``short_long_boundary_s`` — a full interruption
    that should cancel/regenerate the in-flight response."""


class _Cancelable(Protocol):
    """Minimal handle protocol: anything with a ``cancel()`` method."""

    def cancel(self) -> None: ...


class InterruptionDetector:
    """Classify user speech bursts against the current response state.

    Subscribes to :class:`~familiar_connect.voice_lull.VoiceLullMonitor`'s
    per-user voice-activity stream (Discord audio frames, no Deepgram
    VAD), and also observes :class:`ResponseTracker` state transitions
    so bursts line up with the familiar's actual lifecycle rather than
    just the event stream.

    Burst lifecycle:

    - A burst begins the first moment both conditions hold: (a) at
      least one user is speaking, and (b) the tracker is not
      :attr:`ResponseState.IDLE`. That means speech carrying over from
      ``IDLE`` into ``GENERATING`` retroactively starts a burst at the
      moment of the state transition.
    - Sub-utterance pauses shorter than ``lull_timeout_s`` keep rolling
      into the same burst.
    - The tracker's current state is resolved at *log time*, so a
      burst that begins in ``GENERATING`` but continues after the
      familiar starts speaking is reported as a ``SPEAKING``
      interruption.
    - When the tracker returns to ``IDLE`` mid-burst (the familiar
      stopped speaking / gave up generating), the burst is aborted and
      all timers cancelled — no classification is emitted.

    Today this is **detect-only**: each classified burst produces a
    single INFO log line. Later steps will dispatch on short/long
    classifications via ``ResponseTracker``.

    :param tracker_registry: Shared registry so the detector can read
        the current state of the guild's response at event time.
    :param guild_id: The voice-connected guild this detector is scoped
        to. One detector per voice-connected guild (paired 1:1 with a
        :class:`~familiar_connect.voice_lull.VoiceLullMonitor`).
    :param min_interruption_s: Bursts shorter than this are
        :attr:`InterruptionClass.discarded`.
    :param short_long_boundary_s: Bursts at or above this are
        :attr:`InterruptionClass.long`; between the two thresholds
        they are :attr:`InterruptionClass.short`.
    :param lull_timeout_s: Channel-wide silence required after the
        last ``ended`` event before the burst is finalized. Matches
        :attr:`~familiar_connect.config.CharacterConfig.voice_lull_timeout`.
    :param base_tolerance: The familiar's configured
        :attr:`~familiar_connect.config.InterruptTolerance.base_probability`.
        Used to roll ``should_keep_talking`` when the burst crosses
        ``min`` during ``SPEAKING``.
    :param clock: Optional monotonic clock override, for tests.
        Defaults to :func:`time.monotonic`.
    :param scheduler: Optional scheduling hook. Given ``(delay_s, cb)``
        it must schedule ``cb`` after ``delay_s`` and return an object
        with a ``.cancel()`` method. Defaults to
        ``asyncio.get_event_loop().call_later``. Injected in unit tests
        to drive the lull timer synchronously.
    :param rng: Optional zero-argument callable returning ``[0, 1)``,
        used for the tolerance roll. Defaults to
        :func:`random.random`. Injected in unit tests for determinism.
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
        on_short_yield_resume: (
            Callable[[list[WordTimestamp]], Coroutine[Any, Any, None]] | None
        ) = None,
        on_push_through_transcript: Callable[[int, str], None] | None = None,
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
        self._on_short_yield_resume = on_short_yield_resume
        self._on_push_through_transcript = on_push_through_transcript

        # Users currently speaking, tracked regardless of tracker
        # state. This lets us retroactively start a burst when the
        # tracker leaves IDLE while someone is already talking.
        self._speaking: set[int] = set()
        self._burst_started_at: float | None = None
        self._burst_last_ended_at: float | None = None
        # The user whose speech opened the burst — included in every
        # interruption log so operators can see who did the interrupting.
        self._burst_starter_id: int | None = None
        # The latest non-IDLE state observed during the burst. Updated
        # at burst start and on every non-IDLE tracker transition; not
        # cleared on SPEAKING→IDLE so the classification log still
        # reports ``during SPEAKING`` for bursts that outlive playback.
        self._burst_latest_state: ResponseState | None = None
        # Pending finalize timer. Armed when everyone goes quiet,
        # cancelled on any new ``started`` event within the same lull.
        self._lull_handle: _Cancelable | None = None
        # Pending min-threshold timer. Armed at burst start for
        # ``min_interruption_s`` seconds; together with re-evaluations
        # on every ``started`` / ``ended`` event, guarantees the
        # min-crossed log fires the instant the burst's accumulated
        # wall-clock duration (first ``started`` → now) crosses
        # ``min_interruption_s``. Cancelled on burst finalize.
        self._min_handle: _Cancelable | None = None
        # Latch so the min-crossed log fires at most once per burst.
        self._min_logged: bool = False
        # Transcript text accumulated from Deepgram finals during the burst.
        # Used by the push-through path to write the interrupter's words to
        # history at finalize time.
        self._burst_transcript: str = ""
        # Timestamps of the remaining (not-yet-played) words captured at
        # stop-time on a yield. Stored here so finalize can dispatch even
        # after the tracker has transitioned to IDLE and cleared its own
        # timestamps field.
        self._remaining_timestamps: list[WordTimestamp] = []  # type: ignore[type-arg]
        # True when _maybe_log_min_crossed stopped the vc this burst.
        self._did_yield: bool = False

        # Register as an observer on the guild's tracker so state
        # transitions can retroactively start / abort bursts.
        tracker = tracker_registry.get(guild_id)
        tracker.on_state_change = self.on_tracker_state_change

    def on_voice_activity(
        self,
        user_id: int,
        event: VoiceActivityEvent,
    ) -> None:
        """Consume a per-user speaking transition.

        Wire as the ``on_voice_activity`` callback on the guild's
        :class:`~familiar_connect.voice_lull.VoiceLullMonitor`.
        """
        tracker = self._tracker_registry.get(self._guild_id)
        if event is VoiceActivityEvent.started:
            # Track speaking state globally (even during IDLE) so that
            # a later IDLE → GENERATING transition can retroactively
            # start a burst for speech that's already in progress.
            self._speaking.add(user_id)
            if tracker.state is ResponseState.IDLE:
                return
            # Any new speech cancels a pending finalize; we're still
            # within the same lull window so the burst keeps growing.
            self._cancel_lull()
            if self._burst_started_at is None:
                self._start_burst(user_id)
            # A fresh ``started`` may push the accumulated duration
            # (``now - burst_started_at``) past ``min`` if the timer
            # already fired during a silent gap without logging.
            self._maybe_log_min_crossed()
            return

        # ended
        self._speaking.discard(user_id)
        if self._burst_started_at is None:
            # Nothing to close: either we're between bursts, or the
            # speaker went quiet while the tracker was IDLE.
            return
        if self._speaking:
            return
        # All users quiet — arm the lull timer. The burst isn't
        # finalized until ``lull_timeout_s`` passes without anyone
        # speaking again; short inter-utterance pauses (< lull_timeout)
        # keep accumulating into the same classification.
        self._burst_last_ended_at = self._clock()
        self._arm_lull()
        # The update to ``burst_last_ended_at`` may itself cross ``min``
        # even if we're now silent — e.g., the timer fired during an
        # earlier gap and stayed silent, and this utterance carried
        # the burst past the threshold.
        self._maybe_log_min_crossed()

    def on_tracker_state_change(self, new_state: ResponseState) -> None:
        """React to a :class:`ResponseTracker` state transition.

        Wired by ``__init__`` as the tracker's ``on_state_change``
        callback. Handles the lifecycle edges the event stream alone
        can't see:

        - Leaving ``IDLE`` while users are already speaking —
          retroactively starts a burst timed from this transition
          (speech pre-dating the transition doesn't count against the
          ``min`` threshold, but its continuation does).
        - ``SPEAKING → IDLE``: the familiar finished its turn cleanly.
          Keep the burst alive so the interruption still classifies;
          the latest non-IDLE state (``SPEAKING``) is preserved and
          used at log time.
        - ``GENERATING → IDLE``: the familiar abandoned generation
          (e.g., interjection eval said NO). There's nothing left to
          interrupt, so the burst is aborted and all timers cancelled
          with no classification emitted.
        """
        if new_state is ResponseState.IDLE:
            if self._burst_started_at is None:
                return
            # Preserve the burst only when the familiar was actually
            # speaking — a natural end-of-turn shouldn't erase the
            # interruption record.
            if self._burst_latest_state is ResponseState.SPEAKING:
                return
            self._abort_burst()
            return
        # Non-IDLE: record the new state and, if users were already
        # speaking through the transition, start a burst now. Pick any
        # active speaker as the starter (in practice this is the user
        # whose speech carried across).
        self._burst_latest_state = new_state
        if self._burst_started_at is None and self._speaking:
            starter = next(iter(self._speaking))
            self._start_burst(starter)
            self._maybe_log_min_crossed()

    def on_transcript(self, _user_id: int, text: str) -> None:
        """Accumulate Deepgram final text during an active burst.

        Wire alongside :meth:`on_voice_activity` from the voice pipeline's
        transcript handler so the interrupter's words are available at
        :meth:`_finalize_burst` time. Text is ignored when no burst is
        in progress (i.e., between turns or during ``IDLE``).
        """
        if self._burst_started_at is None or not text:
            return
        self._burst_transcript = (
            (self._burst_transcript + " " + text).strip()
            if self._burst_transcript
            else text
        )

    def _start_burst(self, starter_id: int) -> None:
        """Begin a new burst attributed to *starter_id*."""
        self._burst_started_at = self._clock()
        self._burst_starter_id = starter_id
        self._min_logged = False
        tracker = self._tracker_registry.get(self._guild_id)
        # Capture the tracker's non-IDLE state for log-time resolution.
        # Direct ``tracker.state = ...`` assignments in tests bypass
        # the observer, so reading here keeps them working.
        if tracker.state is not ResponseState.IDLE:
            self._burst_latest_state = tracker.state
        # Arm the min-crossed timer. Fires in real time at
        # ``min_interruption_s`` after burst start — the earliest
        # moment the accumulated duration can cross the threshold.
        self._arm_min_timer()

    def _abort_burst(self) -> None:
        """Cancel timers and reset burst scratch without classifying."""
        self._cancel_lull()
        self._cancel_min_timer()
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_starter_id = None
        self._burst_latest_state = None
        self._min_logged = False
        self._burst_transcript = ""
        self._remaining_timestamps = []
        self._did_yield = False

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
        """Timer callback: re-evaluate whether the min has been crossed."""
        self._min_handle = None
        self._maybe_log_min_crossed()

    def _maybe_log_min_crossed(self) -> None:
        """Log once per burst the moment accumulated duration ≥ ``min``.

        "Accumulated duration" is wall-clock from burst start to the
        present:

        - while someone is speaking, ``now - burst_started_at``;
        - during an intra-lull silent gap, ``last_ended_at -
          burst_started_at``.

        Called from the min timer and from both ``started`` / ``ended``
        event handlers so we catch every possible crossing moment —
        continuous speech (timer), restart after a below-min gap
        (``started``), and utterance-end that pushes us over the line
        (``ended``).

        The tracker state is resolved *at log time*, so a burst that
        began in ``GENERATING`` and crosses ``min`` only after the
        familiar transitions to ``SPEAKING`` reports the ``SPEAKING``
        state.
        """
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
            # Tracker returned to IDLE before we could log — the burst
            # is about to be aborted via on_tracker_state_change.
            return
        _logger.info(
            "interruption: min threshold crossed by user=%s during %s",
            self._burst_starter_id,
            state.value,
        )
        self._min_logged = True
        # Moment 1: burst crossed ``min`` while speaking. Roll to decide
        # yield vs. push-through. On yield: stop playback immediately and
        # capture the remaining timestamps so finalize can re-synth them
        # even after the tracker has returned to IDLE.
        if state is ResponseState.SPEAKING:
            tracker = self._tracker_registry.get(self._guild_id)
            keep = tracker.should_keep_talking(self._base_tolerance, rng=self._rng)
            if not keep:
                if tracker.vc is not None and tracker.vc.is_playing():
                    tracker.vc.stop()
                if tracker.playback_start_time is not None:
                    elapsed_ms = (self._clock() - tracker.playback_start_time) * 1000
                    tracker.interruption_elapsed_ms = elapsed_ms
                    _, self._remaining_timestamps = split_at_elapsed(
                        tracker.timestamps, elapsed_ms
                    )
                self._did_yield = True

    def _current_tracker_state(self) -> ResponseState | None:
        """State to report in interruption logs.

        Returns the latest non-IDLE tracker state observed during the
        active burst. This lets a burst that spans ``GENERATING →
        SPEAKING`` report as ``SPEAKING``, and a burst that began in
        ``SPEAKING`` and continues past playback-end (``SPEAKING →
        IDLE``) still classify as ``SPEAKING``.
        """
        return self._burst_latest_state

    def _finalize_burst(self) -> None:
        """Classify + log the accumulated burst. Invoked by the lull timer."""
        self._lull_handle = None
        self._cancel_min_timer()
        started_at = self._burst_started_at
        last_ended_at = self._burst_last_ended_at
        starter_id = self._burst_starter_id
        state = self._current_tracker_state()
        # Snapshot Step 11 fields before reset.
        burst_transcript = self._burst_transcript
        remaining = self._remaining_timestamps
        did_yield = self._did_yield
        # Reset regardless; a new burst starts from a clean slate.
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_starter_id = None
        self._burst_latest_state = None
        self._min_logged = False
        self._burst_transcript = ""
        self._remaining_timestamps = []
        self._did_yield = False
        if started_at is None or last_ended_at is None or starter_id is None:
            return
        if state is None:
            # Shouldn't happen — _start_burst always populates
            # _burst_latest_state — but stay defensive.
            return
        # Effective speech duration — burst start to last speech,
        # excluding trailing lull silence. Sub-utterance gaps are
        # included (they're part of the same conversational turn).
        duration = last_ended_at - started_at

        classification = self._classify(duration)
        _logger.info(
            "interruption: %s (%.2fs) by user=%s during %s",
            classification.value,
            duration,
            starter_id,
            state.value,
        )
        # Step 11 dispatch: short interruption during SPEAKING.
        if (
            classification is InterruptionClass.short
            and state is ResponseState.SPEAKING
        ):
            _logger.info(
                "dispatch: short@SPEAKING → %s speaker=%s",
                "yield+resume" if did_yield else "push-through",
                starter_id,
            )
            if did_yield:
                cb = self._on_short_yield_resume
                if cb is not None and remaining:
                    asyncio.create_task(cb(remaining))  # noqa: RUF006
            else:
                cb2 = self._on_push_through_transcript
                if cb2 is not None and burst_transcript:
                    cb2(starter_id, burst_transcript)

    def _classify(self, duration: float) -> InterruptionClass:
        if duration < self._min_interruption_s:
            return InterruptionClass.discarded
        if duration < self._short_long_boundary_s:
            return InterruptionClass.short
        return InterruptionClass.long
