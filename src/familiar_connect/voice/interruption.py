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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Callable

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
    VAD). A *burst* begins on the first ``started`` event received
    while the :class:`ResponseTracker` is not :attr:`ResponseState.IDLE`
    and stays open across sub-utterance gaps within the same lull —
    finalization waits for ``lull_timeout_s`` of channel-wide silence,
    matching the voice-lull boundary the rest of the pipeline uses. If
    new speech arrives before the timer fires, aggregation continues.

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
    :param clock: Optional monotonic clock override, for tests.
        Defaults to :func:`time.monotonic`.
    :param scheduler: Optional scheduling hook. Given ``(delay_s, cb)``
        it must schedule ``cb`` after ``delay_s`` and return an object
        with a ``.cancel()`` method. Defaults to
        ``asyncio.get_event_loop().call_later``. Injected in unit tests
        to drive the lull timer synchronously.
    """

    def __init__(
        self,
        *,
        tracker_registry: ResponseTrackerRegistry,
        guild_id: int,
        min_interruption_s: float,
        short_long_boundary_s: float,
        lull_timeout_s: float,
        clock: Callable[[], float] | None = None,
        scheduler: Callable[[float, Callable[[], None]], _Cancelable] | None = None,
    ) -> None:
        self._tracker_registry = tracker_registry
        self._guild_id = guild_id
        self._min_interruption_s = min_interruption_s
        self._short_long_boundary_s = short_long_boundary_s
        self._lull_timeout_s = lull_timeout_s
        self._clock = clock if clock is not None else time.monotonic
        self._scheduler = scheduler

        # Users currently speaking *within the active burst*. Users who
        # start speaking while the tracker is IDLE are never added, so
        # their ``ended`` events don't spuriously close a burst.
        self._speaking: set[int] = set()
        self._burst_started_at: float | None = None
        self._burst_last_ended_at: float | None = None
        # The ResponseState observed when the burst began. Used for the
        # classification log so mid-burst transitions don't change the
        # reported state.
        self._burst_state: ResponseState | None = None
        # Pending finalize timer. Armed when everyone goes quiet,
        # cancelled on any new ``started`` event within the same lull.
        self._lull_handle: _Cancelable | None = None

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
            if tracker.state is ResponseState.IDLE:
                # Detector dormant — nothing to classify.
                return
            # Any new speech cancels a pending finalize; we're still
            # within the same lull window so the burst keeps growing.
            self._cancel_lull()
            if self._burst_started_at is None:
                self._burst_started_at = self._clock()
                self._burst_state = tracker.state
            self._speaking.add(user_id)
            return

        # ended
        if user_id not in self._speaking:
            # User never joined the active burst (they started during
            # IDLE, or we're between bursts). Ignore.
            return
        self._speaking.discard(user_id)
        if self._speaking:
            return
        # All users quiet — arm the lull timer. The burst isn't
        # finalized until ``lull_timeout_s`` passes without anyone
        # speaking again; short inter-utterance pauses (< lull_timeout)
        # keep accumulating into the same classification.
        self._burst_last_ended_at = self._clock()
        self._arm_lull()

    def _arm_lull(self) -> None:
        self._cancel_lull()
        scheduler = self._scheduler
        if scheduler is None:
            loop = asyncio.get_event_loop()
            scheduler = loop.call_later
        self._lull_handle = scheduler(self._lull_timeout_s, self._finalize_burst)

    def _cancel_lull(self) -> None:
        if self._lull_handle is not None:
            self._lull_handle.cancel()
            self._lull_handle = None

    def _finalize_burst(self) -> None:
        """Classify + log the accumulated burst. Invoked by the lull timer."""
        self._lull_handle = None
        started_at = self._burst_started_at
        last_ended_at = self._burst_last_ended_at
        observed = self._burst_state
        # Reset regardless; a new burst starts from a clean slate.
        self._burst_started_at = None
        self._burst_last_ended_at = None
        self._burst_state = None
        if started_at is None or last_ended_at is None or observed is None:
            return
        # Effective speech duration — first speech to last speech,
        # excluding trailing lull silence. Sub-utterance gaps are
        # included (they're part of the same conversational turn).
        duration = last_ended_at - started_at

        classification = self._classify(duration)
        # Separate info log for the min-threshold crossing so operators
        # can see "this burst is no longer back-channel noise" as a
        # distinct event from the short/long classification. Only fires
        # when the burst survives the noise floor.
        if duration >= self._min_interruption_s:
            _logger.info(
                "interruption: min threshold crossed (%.2fs, min=%.2fs) during %s",
                duration,
                self._min_interruption_s,
                observed.value,
            )
        _logger.info(
            "interruption: %s (%.2fs) during %s",
            classification.value,
            duration,
            observed.value,
        )

    def _classify(self, duration: float) -> InterruptionClass:
        if duration < self._min_interruption_s:
            return InterruptionClass.discarded
        if duration < self._short_long_boundary_s:
            return InterruptionClass.short
        return InterruptionClass.long
