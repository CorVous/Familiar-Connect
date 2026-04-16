"""Tests for the voice-response state machine scaffold (Steps 3 + 5)."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from familiar_connect.tts import WordTimestamp
from familiar_connect.voice.interruption import (
    UNSOLICITED_TOLERANCE_BIAS,
    InterruptionClass,
    InterruptionDetector,
    ResponseState,
    ResponseTracker,
    ResponseTrackerRegistry,
    split_at_elapsed,
)
from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Callable

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    """Strip ANSI escape codes for substring assertions on styled log lines."""
    return _ANSI_RE.sub("", text)


class TestResponseStateEnum:
    def test_idle_generating_speaking(self) -> None:
        """All three states exist with their expected string values."""
        assert ResponseState.IDLE.value == "IDLE"
        assert ResponseState.GENERATING.value == "GENERATING"
        assert ResponseState.SPEAKING.value == "SPEAKING"


class TestResponseTrackerDefaults:
    def test_defaults(self) -> None:
        """Fresh tracker starts idle with all scratch fields at zero/None/empty."""
        t = ResponseTracker(guild_id=42)
        assert t.state is ResponseState.IDLE
        assert t.generation_task is None
        assert t.response_text is None
        assert t.timestamps == []
        assert t.is_unsolicited is False
        assert t.mood_modifier == 0.0  # noqa: RUF069


class TestResponseTrackerTransition:
    def test_transition_updates_state(self) -> None:
        t = ResponseTracker(guild_id=1)
        t.transition(ResponseState.GENERATING)
        assert t.state is ResponseState.GENERATING

    def test_transition_to_idle_clears_scratch(self) -> None:
        t = ResponseTracker(guild_id=1)
        t.state = ResponseState.SPEAKING
        t.response_text = "hello"
        t.playback_start_time = 123.0
        t.is_unsolicited = True
        t.mood_modifier = 0.3

        t.transition(ResponseState.IDLE)

        assert t.state is ResponseState.IDLE
        assert t.response_text is None
        assert t.playback_start_time is None
        assert t.is_unsolicited is False
        assert t.mood_modifier == 0.0  # noqa: RUF069

    def test_transition_from_idle_to_generating_preserves_unsolicited_flag(
        self,
    ) -> None:
        # The caller sets is_unsolicited before transitioning; transition
        # itself must not touch it on non-IDLE moves.
        t = ResponseTracker(guild_id=1)
        t.is_unsolicited = True
        t.transition(ResponseState.GENERATING)
        assert t.is_unsolicited is True

    def test_transition_logs_at_info(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=7)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            t.transition(ResponseState.GENERATING)
        assert any(
            "guild=7" in _strip(rec.message)
            and "IDLE→GENERATING" in _strip(rec.message)
            for rec in caplog.records
        )

    def test_same_state_transition_is_silent_noop(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # YES path on voice lull: voice pipeline marks GENERATING for
        # the eval, then _run_voice_response also marks GENERATING.
        # The second call must not log a spurious GENERATING→GENERATING.
        t = ResponseTracker(guild_id=1)
        t.transition(ResponseState.GENERATING)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            t.transition(ResponseState.GENERATING)
        assert not any(
            "GENERATING→GENERATING" in _strip(rec.message) for rec in caplog.records
        )
        assert t.state is ResponseState.GENERATING

    def test_same_state_idle_does_not_clear_scratch(self) -> None:
        # An idempotent IDLE transition shouldn't wipe scratch fields
        # (it was already idle — there's no per-response state to reset).
        t = ResponseTracker(guild_id=1)
        t.response_text = "preserved"
        t.transition(ResponseState.IDLE)
        assert t.response_text == "preserved"

    def test_transition_log_includes_unsolicited_flag(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=7)
        t.is_unsolicited = True
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            t.transition(ResponseState.GENERATING)
        assert any("unsolicited=True" in _strip(rec.message) for rec in caplog.records)


class TestResponseTrackerRegistry:
    def test_get_creates_on_first_use(self) -> None:
        reg = ResponseTrackerRegistry()
        t = reg.get(100)
        assert isinstance(t, ResponseTracker)
        assert t.guild_id == 100
        assert t.state is ResponseState.IDLE

    def test_get_returns_same_instance(self) -> None:
        reg = ResponseTrackerRegistry()
        a = reg.get(100)
        b = reg.get(100)
        assert a is b

    def test_different_guilds_get_different_trackers(self) -> None:
        reg = ResponseTrackerRegistry()
        a = reg.get(100)
        b = reg.get(200)
        assert a is not b
        assert a.guild_id == 100
        assert b.guild_id == 200

    def test_drop_removes_tracker(self) -> None:
        reg = ResponseTrackerRegistry()
        a = reg.get(100)
        reg.drop(100)
        b = reg.get(100)
        assert a is not b

    def test_drop_missing_is_noop(self) -> None:
        reg = ResponseTrackerRegistry()
        reg.drop(999)  # Must not raise.

    def test_snapshot_returns_copy(self) -> None:
        reg = ResponseTrackerRegistry()
        reg.get(1)
        reg.get(2)
        snap = reg.snapshot()
        assert set(snap.keys()) == {1, 2}
        # Mutating the snapshot must not affect the registry.
        snap.clear()
        assert set(reg.snapshot().keys()) == {1, 2}


class _FakeClock:
    """Monotonic clock stub driven by tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, delta: float) -> None:
        self.now += delta

    def __call__(self) -> float:
        return self.now


class _FakeHandle:
    def __init__(
        self,
        scheduler: _FakeScheduler,
        callback: Callable[[], None],
    ) -> None:
        self.scheduler = scheduler
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeScheduler:
    """Synchronous scheduler stub for the detector's timers.

    Holds a list of pending callbacks (the detector may have both a
    min-threshold timer and a lull-finalize timer armed simultaneously).
    Tests fire specific callbacks via :meth:`fire_for` or the next
    live one via :meth:`fire_next`.
    """

    def __init__(self) -> None:
        self.pending: list[_FakeHandle] = []

    def __call__(
        self,
        delay: float,
        callback: Callable[[], None],
    ) -> _FakeHandle:
        del delay
        handle = _FakeHandle(self, callback)
        self.pending.append(handle)
        return handle

    def _live(self) -> list[_FakeHandle]:
        return [h for h in self.pending if not h.cancelled]

    def fire_for(self, callback: Callable[[], None]) -> None:
        """Fire the pending callback with identity ``callback``."""
        for handle in self._live():
            if handle.callback == callback:
                handle.cancelled = True
                handle.callback()
                return
        msg = f"no pending callback matching {callback!r}"
        raise RuntimeError(msg)

    def fire_next(self) -> None:
        """Fire the oldest live pending callback."""
        live = self._live()
        if not live:
            msg = "no pending callbacks"
            raise RuntimeError(msg)
        handle = live[0]
        handle.cancelled = True
        handle.callback()

    @property
    def live_count(self) -> int:
        return len(self._live())

    @property
    def is_armed(self) -> bool:
        return self.live_count > 0

    def has_pending(self, callback: Callable[[], None]) -> bool:
        return any(h.callback == callback for h in self._live())


def _make_detector(
    *,
    guild_id: int = 1,
    min_s: float = 1.5,
    boundary_s: float = 4.0,
    lull_s: float = 5.0,
    base_tolerance: float = 0.30,
    rng: Callable[[], float] | None = None,
    on_long_during_generating: Callable[[int, str], None] | None = None,
    on_long_boundary_crossed: Callable[[int, str], None] | None = None,
    name_resolver: Callable[[int], str] | None = None,
) -> tuple[InterruptionDetector, ResponseTrackerRegistry, _FakeClock, _FakeScheduler]:
    registry = ResponseTrackerRegistry()
    clock = _FakeClock()
    scheduler = _FakeScheduler()
    # A deterministic default roll keeps existing tests (which don't
    # care about yield-vs-keep) from gaining a random log line.
    if rng is None:
        rng = lambda: 0.99  # noqa: E731 — inline for test clarity
    detector = InterruptionDetector(
        tracker_registry=registry,
        guild_id=guild_id,
        min_interruption_s=min_s,
        short_long_boundary_s=boundary_s,
        lull_timeout_s=lull_s,
        base_tolerance=base_tolerance,
        clock=clock,
        scheduler=scheduler,
        rng=rng,
        on_long_during_generating=on_long_during_generating,
        on_long_boundary_crossed=on_long_boundary_crossed,
        name_resolver=name_resolver,
    )
    return detector, registry, clock, scheduler


class TestInterruptionDetectorIdle:
    def test_idle_state_ignores_activity(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, _registry, clock, scheduler = _make_detector()
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(5.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
        assert not scheduler.is_armed
        assert not any("Interrupt" in _strip(rec.message) for rec in caplog.records)


class TestInterruptionDetectorClassification:
    def test_below_min_is_discarded(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(0.5)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)  # lull timer expires
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "discarded" in msgs[0]
        assert "SPEAKING" in msgs[0]

    def test_between_min_and_boundary_is_short(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "short" in msgs[0]
        assert "SPEAKING" in msgs[0]

    def test_at_or_above_boundary_is_long(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(5.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "long" in msgs[0]
        assert "GENERATING" in msgs[0]


class TestInterruptionDetectorMultiUser:
    def test_burst_measures_from_first_start_to_last_end(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.SPEAKING

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(1.0)
            detector.on_voice_activity(43, VoiceActivityEvent.started)
            clock.advance(0.5)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            # User 43 still speaking — no lull arming yet.
            assert not scheduler.has_pending(detector._finalize_burst)
            msgs_during = [
                r.message for r in caplog.records if "interruption:" in r.message
            ]
            assert msgs_during == []
            clock.advance(1.5)
            detector.on_voice_activity(43, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)

        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        # Total duration: 1.0 + 0.5 + 1.5 = 3.0s → short.
        assert "short" in msgs[0]


class TestInterruptionDetectorLullAggregation:
    def test_sub_utterance_gap_within_lull_aggregates(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The hallmark behavior: "I was just—" <0.5s pause> "—thinking"
        # is one interruption, not two. Both sub-utterances roll into
        # the same classification because the gap is shorter than the
        # lull timeout.
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, lull_s=5.0
        )
        registry.get(1).state = ResponseState.SPEAKING

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(1.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            # Lull timer now armed — intra-lull pause begins.
            assert scheduler.has_pending(detector._finalize_burst)
            clock.advance(0.5)  # short gap, well under lull_timeout_s
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            # New speech cancels the lull timer.
            assert not scheduler.has_pending(detector._finalize_burst)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)  # now the full lull expires

        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        # Single classification covering both sub-utterances.
        # Duration: 1.0 + 0.5 + 2.0 = 3.5s → short.
        assert len(msgs) == 1
        assert "short" in msgs[0]

    def test_new_speech_after_lull_fires_starts_fresh_burst(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, lull_s=5.0
        )
        registry.get(1).state = ResponseState.SPEAKING

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)  # first burst classified
            clock.advance(10.0)
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(5.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)  # second burst classified

        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 2
        assert "short" in msgs[0]
        assert "long" in msgs[1]


class TestInterruptionDetectorStateUpgrade:
    def test_generating_to_speaking_midburst_is_reported_as_speaking(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A burst that begins in GENERATING but continues through the
        # familiar's SPEAKING transition should be classified as a
        # SPEAKING interruption — the state is resolved at log time.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.transition(ResponseState.GENERATING)

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(1.0)
            tracker.transition(ResponseState.SPEAKING)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)

        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "during=SPEAKING" in msgs[0]
        assert "GENERATING" not in msgs[0]


class TestInterruptionDetectorStarterUser:
    def test_classification_log_includes_starter_user_id(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "speaker=42" in msgs[0]

    def test_min_crossed_log_includes_starter_user_id(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            scheduler.fire_for(detector._on_min_crossed)
        crossed = [
            _strip(r.message)
            for r in caplog.records
            if "min_threshold" in _strip(r.message)
        ]
        assert len(crossed) == 1
        assert "speaker=42" in crossed[0]

    def test_starter_is_first_speaker_even_with_multiple_users(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(0.5)
            detector.on_voice_activity(43, VoiceActivityEvent.started)
            clock.advance(1.5)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            detector.on_voice_activity(43, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "speaker=42" in msgs[0]
        assert "speaker=43" not in msgs[0]


class TestInterruptionDetectorPreExistingSpeech:
    def test_speech_carrying_into_generating_starts_burst(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # User begins speaking while the tracker is IDLE. When the
        # tracker transitions to GENERATING (e.g., a voice-lull
        # interject kicks in), the detector retroactively starts a
        # burst so the speech counts as a GENERATING interruption.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        # User 42 starts talking during IDLE — detector sees the event
        # but doesn't open a burst yet.
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert not scheduler.has_pending(detector._finalize_burst)
        # Tracker transitions to GENERATING while user is still speaking.
        tracker.transition(ResponseState.GENERATING)
        clock.advance(2.0)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "during=GENERATING" in msgs[0]
        assert "speaker=42" in msgs[0]

    def test_no_burst_started_when_idle_transition_has_no_speakers(
        self,
    ) -> None:
        # IDLE → GENERATING with nobody talking: don't spuriously
        # create a burst.
        detector, registry, _clock, scheduler = _make_detector()
        registry.get(1).transition(ResponseState.GENERATING)
        assert not scheduler.has_pending(detector._finalize_burst)
        assert not scheduler.has_pending(detector._on_min_crossed)


class TestInterruptionDetectorIdleTransition:
    def test_speaking_to_idle_preserves_burst_and_lull_timer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The familiar finishing playback mid-interruption must not
        # erase the interruption — the lull timer keeps running and
        # the burst classifies normally, labelled "during SPEAKING".
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.transition(ResponseState.SPEAKING)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        assert scheduler.has_pending(detector._finalize_burst)
        tracker.transition(ResponseState.IDLE)
        # Lull timer survives the transition.
        assert scheduler.has_pending(detector._finalize_burst)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "during=SPEAKING" in msgs[0]
        assert "speaker=42" in msgs[0]

    def test_generating_to_idle_aborts_burst(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Familiar abandons generation (e.g., interjection eval said
        # NO). Nothing to interrupt — abort and emit no classification.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.transition(ResponseState.GENERATING)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        assert scheduler.has_pending(detector._finalize_burst)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            tracker.transition(ResponseState.IDLE)
        assert not scheduler.has_pending(detector._finalize_burst)
        assert not scheduler.has_pending(detector._on_min_crossed)
        assert not any("Interrupt" in _strip(r.message) for r in caplog.records)

    def test_generating_to_idle_while_speaking_keeps_burst(
        self,
    ) -> None:
        # Early-cancel path: LLM task cancelled (GENERATING→IDLE) while the
        # user is still talking. Burst must stay alive so the lull can fire
        # _finalize_burst → on_long_during_generating → regen.
        detector, registry, _clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        tracker = registry.get(1)
        tracker.transition(ResponseState.GENERATING)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert scheduler.has_pending(detector._on_min_crossed)
        tracker.transition(ResponseState.IDLE)
        # Burst kept alive — timers must still be pending.
        assert detector._burst_started_at is not None
        assert scheduler.has_pending(detector._on_min_crossed)


class TestInterruptionDetectorMinThresholdLog:
    def test_burst_ending_before_min_does_not_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Burst ends before the min timer fires — the pending timer
        # gets cancelled at finalize; no log.
        detector, registry, _clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        assert not any("min_threshold" in _strip(r.message) for r in caplog.records)

    def test_still_speaking_at_min_fires_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            # Simulate the 2.0s of real time the scheduler would
            # actually have waited before firing the timer.
            clock.advance(2.0)
            scheduler.fire_for(detector._on_min_crossed)
        crossed = [
            _strip(r.message)
            for r in caplog.records
            if "min_threshold" in _strip(r.message)
        ]
        assert len(crossed) == 1
        assert "during=SPEAKING" in crossed[0]
        # No duration should be emitted — the log is about the
        # crossing event itself, not a measured duration.
        assert "s)" not in crossed[0]
        assert "min=" not in crossed[0]

    def test_accumulated_duration_crosses_min_on_new_speech(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Sub-utterances: first utterance ends below ``min`` and the
        # timer fires during the silent gap without logging. When the
        # user starts speaking again past the 2s wall-clock mark, the
        # accumulated duration has crossed and we log.
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)  # t=0
            clock.advance(0.5)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)  # t=0.5
            clock.advance(1.5)  # t=2.0 — timer would fire here
            scheduler.fire_for(detector._on_min_crossed)
            # Still silent, last_ended-started=0.5 < 2 → no log yet.
            assert not any("min_threshold" in _strip(r.message) for r in caplog.records)
            clock.advance(0.5)  # t=2.5
            # New speech: now-started = 2.5 ≥ 2 → log fires.
            detector.on_voice_activity(42, VoiceActivityEvent.started)
        crossed = [
            _strip(r.message)
            for r in caplog.records
            if "min_threshold" in _strip(r.message)
        ]
        assert len(crossed) == 1

    def test_accumulated_duration_crosses_min_on_ended(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Timer fires during a below-min silent gap; the next utterance
        # is the one that pushes the burst past ``min`` — the ``ended``
        # event (which sets ``last_ended_at``) triggers the log.
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)  # t=0
            clock.advance(0.5)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)  # t=0.5
            scheduler.fire_for(detector._on_min_crossed)  # silent, no log
            clock.advance(1.0)  # t=1.5
            detector.on_voice_activity(42, VoiceActivityEvent.started)  # 1.5<2
            assert not any("min_threshold" in _strip(r.message) for r in caplog.records)
            clock.advance(1.0)  # t=2.5
            # ``last_ended_at - started_at`` = 2.5 ≥ 2 → log fires.
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
        crossed = [
            _strip(r.message)
            for r in caplog.records
            if "min_threshold" in _strip(r.message)
        ]
        assert len(crossed) == 1

    def test_min_crossed_logs_at_most_once_per_burst(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(3.0)
            # Timer fires: speaking, 3 ≥ 2 → log.
            scheduler.fire_for(detector._on_min_crossed)
            clock.advance(1.0)
            # Subsequent ``ended`` would also satisfy the threshold,
            # but the latch prevents a second log.
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
        crossed = [
            _strip(r.message)
            for r in caplog.records
            if "min_threshold" in _strip(r.message)
        ]
        assert len(crossed) == 1

    def test_min_fires_while_silent_does_not_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Speaker went quiet before the min timer expired — at fire
        # time there's no active speech so the burst hasn't crossed.
        detector, registry, _clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            # Min timer is still pending (lull arming doesn't cancel
            # it); fire it directly.
            scheduler.fire_for(detector._on_min_crossed)
        assert not any("min_threshold" in _strip(r.message) for r in caplog.records)

    def test_finalize_cancels_pending_min_timer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # If the lull finalizes the burst before the min timer has
        # fired, the min timer is cancelled — it should not log later.
        detector, registry, _clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.SPEAKING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            scheduler.fire_for(detector._finalize_burst)
        assert not scheduler.has_pending(detector._on_min_crossed)
        assert not any("min_threshold" in _strip(r.message) for r in caplog.records)


class TestInterruptionClassEnum:
    def test_three_classes(self) -> None:
        assert {c.value for c in InterruptionClass} == {
            "discarded",
            "short",
            "long",
        }


class TestResponseTrackerTolerance:
    def test_effective_tolerance_is_base_when_solicited_and_no_mood(self) -> None:
        t = ResponseTracker(guild_id=1)
        assert t.compute_effective_tolerance(0.30) == 0.30  # noqa: RUF069

    def test_effective_tolerance_adds_unsolicited_bias(self) -> None:
        t = ResponseTracker(guild_id=1, is_unsolicited=True)
        assert t.compute_effective_tolerance(0.30) == pytest.approx(
            0.30 + UNSOLICITED_TOLERANCE_BIAS
        )

    def test_effective_tolerance_adds_mood_modifier(self) -> None:
        t = ResponseTracker(guild_id=1, mood_modifier=0.15)
        assert t.compute_effective_tolerance(0.30) == pytest.approx(0.45)

    def test_effective_tolerance_subtracts_negative_mood(self) -> None:
        t = ResponseTracker(guild_id=1, mood_modifier=-0.20)
        assert t.compute_effective_tolerance(0.30) == pytest.approx(0.10)

    def test_effective_tolerance_clamps_above_one(self) -> None:
        t = ResponseTracker(guild_id=1, is_unsolicited=True, mood_modifier=0.50)
        # 0.60 + 0.35 + 0.50 = 1.45 → clamped to 1.0
        assert t.compute_effective_tolerance(0.60) == 1.0  # noqa: RUF069

    def test_effective_tolerance_clamps_below_zero(self) -> None:
        t = ResponseTracker(guild_id=1, mood_modifier=-0.50)
        # 0.10 + 0 + -0.50 = -0.40 → clamped to 0.0
        assert t.compute_effective_tolerance(0.10) == 0.0  # noqa: RUF069

    def test_should_keep_talking_true_when_roll_below_tolerance(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=1)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            keep = t.should_keep_talking(0.30, rng=lambda: 0.1)
        assert keep is True
        assert any("→ keep_talking" in _strip(r.message) for r in caplog.records)

    def test_should_keep_talking_false_when_roll_above_tolerance(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=1)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            keep = t.should_keep_talking(0.30, rng=lambda: 0.9)
        assert keep is False
        assert any("→ yield" in _strip(r.message) for r in caplog.records)

    def test_should_keep_talking_log_includes_all_components(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=1, is_unsolicited=True, mood_modifier=0.10)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            t.should_keep_talking(0.30, rng=lambda: 0.50)
        msgs = [
            _strip(r.message) for r in caplog.records if "Toll" in _strip(r.message)
        ]
        assert len(msgs) == 1
        assert "base=0.30" in msgs[0]
        assert "mood=+0.10" in msgs[0]
        assert "unsolicited=+0.35" in msgs[0]
        assert "effective=0.75" in msgs[0]
        assert "roll=0.50" in msgs[0]


class TestInterruptionDetectorToleranceRoll:
    def test_roll_fires_at_min_crossing_during_speaking(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0, base_tolerance=0.30, rng=lambda: 0.99
        )
        registry.get(1).state = ResponseState.SPEAKING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            scheduler.fire_for(detector._on_min_crossed)
        toll = [
            _strip(r.message) for r in caplog.records if "Toll" in _strip(r.message)
        ]
        assert len(toll) == 1
        assert "→ yield" in toll[0]

    def test_roll_does_not_fire_during_generating(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Moment 1 roll is only meaningful during SPEAKING — during
        # GENERATING the dispatch is long=cancel-and-regen /
        # short=pause-and-deliver, not a tolerance roll.
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            scheduler.fire_for(detector._on_min_crossed)
        assert not any("Toll" in _strip(r.message) for r in caplog.records)

    def test_roll_uses_tracker_unsolicited_flag(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # With unsolicited=True and base=0.30, effective = 0.65 so a
        # roll of 0.50 still keeps talking; without the bias, the same
        # roll would yield.
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0, base_tolerance=0.30, rng=lambda: 0.50
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.is_unsolicited = True
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            scheduler.fire_for(detector._on_min_crossed)
        toll = [
            _strip(r.message) for r in caplog.records if "Toll" in _strip(r.message)
        ]
        assert len(toll) == 1
        assert "→ keep_talking" in toll[0]
        assert "unsolicited=+0.35" in toll[0]


class TestSplitAtElapsed:
    """``split_at_elapsed`` partitions the word-timestamp list.

    Returns ``(delivered, remaining)`` around the caller-supplied
    elapsed playback time. Used by Step 11 to resume from a word
    boundary on short-interruption yield and by Step 12 to record
    only the delivered portion in history on long-interruption yield.
    """

    def test_empty_list_returns_two_empty_lists(self) -> None:
        delivered, remaining = split_at_elapsed([], 500.0)
        assert delivered == []
        assert remaining == []

    def test_before_first_word_returns_all_remaining(self) -> None:
        ts = [
            WordTimestamp("hello", 100.0, 400.0),
            WordTimestamp("world", 500.0, 800.0),
        ]
        delivered, remaining = split_at_elapsed(ts, 50.0)
        assert delivered == []
        assert remaining == ts

    def test_after_last_word_returns_all_delivered(self) -> None:
        ts = [
            WordTimestamp("hello", 100.0, 400.0),
            WordTimestamp("world", 500.0, 800.0),
        ]
        delivered, remaining = split_at_elapsed(ts, 900.0)
        assert delivered == ts
        assert remaining == []

    def test_between_words_splits_at_boundary(self) -> None:
        # Elapsed lands in the 400-500ms gap between words; "hello" is
        # already delivered, "world" is still pending.
        ts = [
            WordTimestamp("hello", 100.0, 400.0),
            WordTimestamp("world", 500.0, 800.0),
            WordTimestamp("!", 850.0, 900.0),
        ]
        delivered, remaining = split_at_elapsed(ts, 450.0)
        assert [w.word for w in delivered] == ["hello"]
        assert [w.word for w in remaining] == ["world", "!"]

    def test_mid_word_keeps_in_flight_word_in_delivered(self) -> None:
        # Elapsed falls inside "hello". The word has already begun
        # playing; treat it as delivered so resumption starts with the
        # next word (no stutter from re-synthesising a half-played word).
        ts = [
            WordTimestamp("hello", 100.0, 400.0),
            WordTimestamp("world", 500.0, 800.0),
        ]
        delivered, remaining = split_at_elapsed(ts, 250.0)
        assert [w.word for w in delivered] == ["hello"]
        assert [w.word for w in remaining] == ["world"]

    def test_exactly_at_word_start_keeps_word_in_remaining(self) -> None:
        # The new word has not started yet at its own start_ms — treat
        # it as still pending so resumption plays it in full.
        ts = [
            WordTimestamp("hello", 100.0, 400.0),
            WordTimestamp("world", 500.0, 800.0),
        ]
        delivered, remaining = split_at_elapsed(ts, 500.0)
        assert [w.word for w in delivered] == ["hello"]
        assert [w.word for w in remaining] == ["world"]


class TestSpeakingTransitionStampsPlaybackStart:
    """``transition(SPEAKING)`` records ``time.monotonic()`` once.

    Step 11 computes ``elapsed_ms = now - playback_start_time`` on
    yield; that only works if the stamp is recorded at exactly the
    point the audio starts playing.
    """

    def test_idle_to_speaking_sets_playback_start_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "familiar_connect.voice.interruption.time.monotonic",
            lambda: 42.5,
        )
        t = ResponseTracker(guild_id=1)
        t.state = ResponseState.GENERATING
        t.transition(ResponseState.SPEAKING)
        assert t.playback_start_time == 42.5  # noqa: RUF069

    def test_transition_to_non_speaking_does_not_stamp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "familiar_connect.voice.interruption.time.monotonic",
            lambda: 99.0,
        )
        t = ResponseTracker(guild_id=1)
        t.transition(ResponseState.GENERATING)
        assert t.playback_start_time is None

    def test_speaking_to_idle_clears_playback_start_time(self) -> None:
        # Already covered by TestResponseTrackerTransition but pinned
        # here alongside the set-on-SPEAKING invariant so the full
        # lifecycle lives in one place.
        t = ResponseTracker(guild_id=1)
        t.state = ResponseState.SPEAKING
        t.playback_start_time = 100.0
        t.transition(ResponseState.IDLE)
        assert t.playback_start_time is None


# ---------------------------------------------------------------------------
# Step 8 — transcript accumulation + long@GENERATING dispatch
# ---------------------------------------------------------------------------


class TestInterruptionDetectorTranscriptCapture:
    """``on_transcript`` accumulates text only during an active burst."""

    def test_on_transcript_ignored_outside_burst(self) -> None:
        # No burst → transcript silently dropped.
        detector, _registry, _clock, _scheduler = _make_detector()
        detector.on_transcript(42, "hello")
        assert not detector._burst_transcript

    def test_transcript_accumulated_during_burst(self) -> None:
        detector, registry, _clock, _scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hello world")
        assert detector._burst_transcript == "hello world"

    def test_multiple_transcripts_joined_with_space(self) -> None:
        detector, registry, _clock, _scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hello")
        detector.on_transcript(42, "world")
        assert detector._burst_transcript == "hello world"

    def test_transcript_cleared_at_new_burst(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        # First burst
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "first")
        clock.advance(1.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        # Finalize the first burst (lull fires)
        scheduler.fire_for(detector._finalize_burst)
        # Second burst starts fresh
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert not detector._burst_transcript

    def test_transcript_accepted_from_any_user_in_burst(self) -> None:
        # Multi-user: second user's transcript also accumulated.
        detector, registry, _clock, _scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hey")
        detector.on_transcript(99, "listen")
        assert detector._burst_transcript == "hey listen"


class TestInterruptionDetectorLongDispatch:
    """long@GENERATING fires the ``on_long_during_generating`` callback."""

    def test_long_during_generating_calls_callback(self) -> None:
        calls: list[tuple[int, str]] = []

        def callback(starter_id: int, transcript: str) -> None:
            calls.append((starter_id, transcript))

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_during_generating=callback
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hey listen to me")
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert calls == [(42, "hey listen to me")]

    def test_dispatch_receives_full_accumulated_transcript(self) -> None:
        received: list[str] = []

        def callback(_starter_id: int, transcript: str) -> None:
            received.append(transcript)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_during_generating=callback
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "part one")
        detector.on_transcript(42, "part two")
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert received == ["part one part two"]

    def test_long_during_speaking_does_not_call_generating_callback(self) -> None:
        calls: list[tuple[int, str]] = []

        def callback(starter_id: int, transcript: str) -> None:
            calls.append((starter_id, transcript))

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_during_generating=callback
        )
        registry.get(1).state = ResponseState.SPEAKING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert calls == []

    def test_short_during_generating_does_not_call_callback(self) -> None:
        calls: list[tuple[int, str]] = []

        def callback(starter_id: int, transcript: str) -> None:
            calls.append((starter_id, transcript))

        # boundary_s=30.0 so a 2s burst is short, not long
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=30.0, on_long_during_generating=callback
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert calls == []

    def test_no_callback_set_does_not_raise(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Default (no callback) must not raise even on a long burst.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(5.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        msgs = [
            _strip(r.message)
            for r in caplog.records
            if "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
        ]
        assert any("long" in m and "GENERATING" in m for m in msgs)

    def test_dispatch_log_emitted_when_callback_set(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def callback(starter_id: int, transcript: str) -> None:
            pass

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_during_generating=callback
        )
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(5.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        assert any("long@GENERATING" in _strip(r.message) for r in caplog.records)


class TestInterruptionDetectorDeliveryGate:
    """Delivery gate: cleared at burst start, set at finalize/abort."""

    @pytest.mark.asyncio
    async def test_gate_open_when_no_burst(self) -> None:
        detector, _, _, _ = _make_detector()
        result = await detector.wait_for_lull()
        assert result is None

    def test_gate_closed_at_burst_start(self) -> None:
        detector, registry, _, _ = _make_detector()
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert not detector._delivery_gate.is_set()

    @pytest.mark.asyncio
    async def test_gate_opens_after_abort(self) -> None:
        detector, registry, _, _ = _make_detector()
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert not detector._delivery_gate.is_set()
        detector._abort_burst()
        result = await detector.wait_for_lull()
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_opens_and_returns_long(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        # Create task BEFORE finalize so it suspends at gate.wait() (gate closed).
        wait_task = asyncio.create_task(detector.wait_for_lull())
        await asyncio.sleep(0)  # yield; task reaches gate.wait()
        scheduler.fire_for(detector._finalize_burst)
        result = await wait_task
        assert result is InterruptionClass.long

    @pytest.mark.asyncio
    async def test_gate_opens_and_returns_short(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        # Create task BEFORE finalize so it suspends at gate.wait() (gate closed).
        wait_task = asyncio.create_task(detector.wait_for_lull())
        await asyncio.sleep(0)  # yield; task reaches gate.wait()
        scheduler.fire_for(detector._finalize_burst)
        result = await wait_task
        assert result is InterruptionClass.short

    @pytest.mark.asyncio
    async def test_gate_returns_none_after_burst_finalized(self) -> None:
        # After finalize the gate is already open; wait_for_lull() must return
        # None immediately — not the previous burst's classification.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        result = await detector.wait_for_lull()  # gate already open
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_clears_between_bursts(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        # First burst finalizes → gate opens.
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert detector._delivery_gate.is_set()
        # Second burst starts → gate re-closes.
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert not detector._delivery_gate.is_set()


class TestInterruptionDetectorLongBoundaryCrossed:
    """Early cancel: on_long_boundary_crossed fires before lull when GENERATING."""

    def test_callback_fires_when_timer_crosses_boundary(self) -> None:
        fired: list[int] = []

        def cb(user_id: int, transcript: str) -> None:  # noqa: ARG001
            fired.append(user_id)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_boundary_crossed=cb
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(4.0)
        scheduler.fire_for(detector._on_long_crossed)
        assert fired == [42]

    def test_callback_not_fired_if_not_generating(self) -> None:
        fired: list[int] = []

        def cb(user_id: int, transcript: str) -> None:  # noqa: ARG001
            fired.append(user_id)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_boundary_crossed=cb
        )
        registry.get(1).state = ResponseState.SPEAKING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(4.0)
        scheduler.fire_for(detector._on_long_crossed)
        assert fired == []

    def test_callback_fires_at_most_once_per_burst(self) -> None:
        fired: list[int] = []

        def cb(user_id: int, transcript: str) -> None:  # noqa: ARG001
            fired.append(user_id)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_boundary_crossed=cb
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(4.0)
        scheduler.fire_for(detector._on_long_crossed)
        assert fired == [42]
        # Calling again must be a no-op.
        detector._maybe_dispatch_long_cancel()
        assert fired == [42]

    def test_long_timer_cancelled_at_finalize(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert detector._long_handle is not None
        clock.advance(5.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert detector._long_handle is None

    def test_long_timer_cancelled_at_abort(self) -> None:
        detector, registry, _clock, _scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert detector._long_handle is not None
        detector._abort_burst()
        assert detector._long_handle is None

    def test_callback_fires_on_restart_after_gap(self) -> None:
        # Timer fires mid-gap when effective < boundary → no fire.
        # User restarts; effective now ≥ boundary → fires on started event.
        fired: list[int] = []

        def cb(user_id: int, transcript: str) -> None:  # noqa: ARG001
            fired.append(user_id)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5, boundary_s=4.0, on_long_boundary_crossed=cb
        )
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(3.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        # Long timer fires; effective = 3.0 < 4.0 → no dispatch.
        scheduler.fire_for(detector._on_long_crossed)
        assert fired == []
        # User starts again; clock now at 3.0+gap. Simulate gap then restart.
        clock.advance(2.0)  # clock = 5.0; effective = 5.0 ≥ 4.0
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert fired == [42]


# ---------------------------------------------------------------------------
# Step 11 — Short interruption during SPEAKING (yield + push-through)
# ---------------------------------------------------------------------------


class _FakeVC:
    """Minimal VoiceClient stand-in for Step 11 dispatch tests."""

    def __init__(self, *, playing: bool = True) -> None:
        self.playing = playing
        self.stopped = False

    def is_playing(self) -> bool:
        return self.playing

    def stop(self) -> None:
        self.stopped = True
        self.playing = False


def _make_detector_with_callbacks(
    *,
    guild_id: int = 1,
    min_s: float = 1.5,
    boundary_s: float = 4.0,
    lull_s: float = 5.0,
    base_tolerance: float = 0.30,
    rng: Callable[[], float] | None = None,
    on_short_yield_resume: Callable | None = None,
    on_push_through_transcript: Callable | None = None,
) -> tuple[InterruptionDetector, ResponseTrackerRegistry, _FakeClock, _FakeScheduler]:
    registry = ResponseTrackerRegistry()
    clock = _FakeClock()
    scheduler = _FakeScheduler()
    if rng is None:
        rng = lambda: 0.99  # noqa: E731
    detector = InterruptionDetector(
        tracker_registry=registry,
        guild_id=guild_id,
        min_interruption_s=min_s,
        short_long_boundary_s=boundary_s,
        lull_timeout_s=lull_s,
        base_tolerance=base_tolerance,
        clock=clock,
        scheduler=scheduler,
        rng=rng,
        on_short_yield_resume=on_short_yield_resume,
        on_push_through_transcript=on_push_through_transcript,
    )
    return detector, registry, clock, scheduler


class TestStep11ShortSpeakingDispatch:
    """Step 11: short interruption during SPEAKING drives yield or push-through."""

    # ------------------------------------------------------------------
    # Moment 1: min crossed → vc.stop() / no-op
    # ------------------------------------------------------------------

    def test_short_speaking_yield_stops_vc(self) -> None:
        # roll=0.99 > base=0.10 → keep=False → yield → vc.stop()
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.10, rng=lambda: 0.99
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        assert fake_vc.stopped is True

    def test_short_speaking_push_through_does_not_stop_vc(self) -> None:
        # roll=0.01 < base=0.99 → keep=True → push-through → vc untouched
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.99, rng=lambda: 0.01
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        assert fake_vc.stopped is False

    def test_very_meek_tolerance_yields(self) -> None:
        # base=0.10, roll=0.99 → 0.99 ≥ 0.10 → keep=False → yield
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.10, rng=lambda: 0.99
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        assert fake_vc.stopped is True

    def test_very_stubborn_tolerance_keeps_talking(self) -> None:
        # base=0.60, roll=0.01 → 0.01 < 0.60 → keep=True → push-through
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.60, rng=lambda: 0.01
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        assert fake_vc.stopped is False

    def test_unsolicited_bias_shifts_roll_to_keep(self) -> None:
        # base=0.30, is_unsolicited=True → effective=0.65.
        # roll=0.50: 0.50 < 0.65 → keep (would yield at 0.30 without bias).
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.30, rng=lambda: 0.50
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]
        tracker.is_unsolicited = True

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        assert fake_vc.stopped is False  # bias pushed past roll → keep

    def test_interruption_elapsed_ms_stored_on_tracker_on_yield(self) -> None:
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5, boundary_s=4.0, base_tolerance=0.10, rng=lambda: 0.99
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = 10.0
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]
        clock.now = 11.5  # 1.5s elapsed

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        # elapsed = (now - playback_start) * 1000 = 1.5s * 1000 = 1500ms
        assert tracker.interruption_elapsed_ms is not None
        assert tracker.interruption_elapsed_ms == pytest.approx(
            (clock.now - 10.0) * 1000, abs=1.0
        )

    def test_interruption_elapsed_ms_cleared_on_idle(self) -> None:
        t = ResponseTracker(guild_id=1)
        t.state = ResponseState.SPEAKING
        t.interruption_elapsed_ms = 1200.0
        t.transition(ResponseState.IDLE)
        assert t.interruption_elapsed_ms is None

    # ------------------------------------------------------------------
    # Lull confirmed → dispatch resume or push-through
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_short_speaking_yield_resumes_from_word_boundary(self) -> None:
        # 3 words; elapsed_ms = 1500ms lands between word 1 (end 300ms)
        # and word 2 (start 1600ms) → remaining = ["world", "bye"].
        resumed: list[list[WordTimestamp]] = []

        async def _capture_resume(remaining: list[WordTimestamp]) -> None:  # noqa: RUF029
            resumed.append(remaining)

        fake_vc = _FakeVC()
        ts = [
            WordTimestamp("hello", 0.0, 300.0),
            WordTimestamp("world", 1600.0, 2000.0),
            WordTimestamp("bye", 2100.0, 2400.0),
        ]
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.10,
            rng=lambda: 0.99,
            on_short_yield_resume=_capture_resume,
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = 0.0  # clock starts at 0.0
        tracker.timestamps = ts
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        # Burst starts at clock=0.0; advance 1.5s then fire min timer.
        # elapsed_ms = (1.5 - 0.0) * 1000 = 1500ms → after "hello"
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        # Burst ends, lull fires as short (1.5s < 4.0s boundary)
        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        await asyncio.sleep(0)  # yield to let the created task run

        assert len(resumed) == 1
        assert [w.word for w in resumed[0]] == ["world", "bye"]

    def test_short_speaking_push_through_transcript_appended(self) -> None:
        pushed: list[tuple[int, str]] = []

        def _capture_push(user_id: int, transcript: str) -> None:
            pushed.append((user_id, transcript))

        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.99,  # very stubborn — keeps talking
            rng=lambda: 0.01,
            on_push_through_transcript=_capture_push,
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        # Burst starts at clock=0.0; transcript arrives during burst
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hey wait")
        clock.advance(1.5)  # min crossed (effective = 1.5 >= 1.5)
        scheduler.fire_for(detector._on_min_crossed)  # keep=True → no stop

        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert pushed == [(42, "hey wait")]

    def test_yield_dispatch_only_on_short_not_long(self) -> None:
        # Yield at min, burst ends up long → on_short_yield_resume NOT called
        resumed: list[object] = []

        async def _capture_resume(remaining: list[WordTimestamp]) -> None:  # noqa: RUF029
            resumed.append(remaining)

        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.10,
            rng=lambda: 0.99,
            on_short_yield_resume=_capture_resume,
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        # Min crossed → yield
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        # Burst continues → long (4.0s total ≥ boundary 4.0)
        clock.advance(2.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert resumed == []

    def test_on_transcript_ignored_before_burst(self) -> None:
        pushed: list[tuple[int, str]] = []

        def _capture_push(user_id: int, transcript: str) -> None:
            pushed.append((user_id, transcript))

        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.99,
            rng=lambda: 0.01,
            on_push_through_transcript=_capture_push,
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        fake_vc = _FakeVC()
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        # transcript before burst starts
        detector.on_transcript(42, "pre-burst noise")

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        # only the empty transcript from during the burst
        assert pushed == []

    def test_short_speaking_yield_sets_pending_flag(self) -> None:
        # After short@SPEAKING yield+resume finalize, tracker.short_yield_pending
        # must be True so _deliver_to_monitor suppresses the concurrent lull.
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.10,
            rng=lambda: 0.99,  # roll > tolerance → yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = 0.0
        tracker.timestamps = [
            WordTimestamp("hello", 0.0, 300.0),
            WordTimestamp("world", 1600.0, 2000.0),
        ]
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)  # yield; remaining=["world"]

        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.short_yield_pending is True

    def test_short_speaking_push_through_does_not_set_pending_flag(self) -> None:
        # push-through: familiar keeps talking → no resume pending.
        fake_vc = _FakeVC()
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.99,  # very stubborn → keep
            rng=lambda: 0.01,
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = fake_vc  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)

        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.short_yield_pending is False


class TestLongSpeakingYieldDispatch:
    """Step 12: long interruption during SPEAKING triggers yield dispatch.

    At Moment 1 (min crossed, SPEAKING, yield decision):
    - ``vc.stop()`` is called
    - ``tracker.interruption_elapsed_ms`` is set
    - ``tracker.interrupt_event`` is created

    At finalize (long classification):
    - ``tracker.interrupt_event`` is set
    - ``tracker.interrupt_classification`` is ``InterruptionClass.long``
    - ``tracker.interrupt_transcript`` carries the accumulated text
    """

    def test_yield_stops_vc_and_records_elapsed(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0,
            boundary_s=30.0,
            rng=lambda: 0.99,  # roll > tolerance → yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now  # playback started at t=0
        vc_mock = MagicMock()
        tracker.vc = vc_mock

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)  # crosses min_s
        scheduler.fire_for(detector._on_min_crossed)

        vc_mock.stop.assert_called_once()
        assert tracker.interruption_elapsed_ms is not None
        assert tracker.interruption_elapsed_ms == pytest.approx(2000.0)  # 2s in ms
        assert tracker.interrupt_event is not None

    def test_yield_long_finalize_sets_event_and_classification(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0,
            boundary_s=5.0,
            rng=lambda: 0.99,  # yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)  # yield dispatch
        clock.advance(4.0)  # total duration 6s ≥ boundary_s=5 → long
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.interrupt_event is not None
        assert tracker.interrupt_event.is_set()
        assert tracker.interrupt_classification is InterruptionClass.long

    def test_yield_long_finalize_captures_transcript(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0,
            boundary_s=5.0,
            rng=lambda: 0.99,  # yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "tell me")
        detector.on_transcript(42, "more please")
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(4.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.interrupt_transcript == "tell me more please"

    def test_yield_short_finalize_sets_short_classification(self) -> None:
        # burst between min and boundary → short classification
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0,
            boundary_s=30.0,
            rng=lambda: 0.99,  # yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(3.0)  # total 5s, < boundary_s=30 → short
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.interrupt_classification is InterruptionClass.short

    def test_no_yield_push_through_does_not_set_interrupt_event(self) -> None:
        # roll < tolerance → keep_talking (push through)
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0, rng=lambda: 0.0
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)

        tracker.vc.stop.assert_not_called()
        assert tracker.interrupt_event is None

    def test_on_transcript_only_accumulates_during_burst(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0,
            boundary_s=30.0,
            rng=lambda: 0.99,  # yield
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING

        # text before burst start is ignored
        detector.on_transcript(42, "before")

        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "during burst")
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(4.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert tracker.interrupt_transcript == "during burst"

    def test_idle_to_idle_clears_interrupt_fields(self) -> None:
        # transition(IDLE) must reset all Step 12 scratch
        t = ResponseTracker(guild_id=1)
        t.interruption_elapsed_ms = 500.0
        t.interrupt_event = asyncio.Event()
        t.interrupt_classification = InterruptionClass.long
        t.interrupt_transcript = "hello"
        t.interrupt_starter_name = "Bob"
        t.state = ResponseState.SPEAKING
        t.transition(ResponseState.IDLE)

        assert t.interruption_elapsed_ms is None
        assert t.interrupt_event is None
        assert t.interrupt_classification is None
        assert not t.interrupt_transcript
        assert not t.interrupt_starter_name

    def test_name_resolver_sets_starter_name(self) -> None:
        registry = ResponseTrackerRegistry()
        clock = _FakeClock()
        scheduler = _FakeScheduler()
        detector = InterruptionDetector(
            tracker_registry=registry,
            guild_id=1,
            min_interruption_s=2.0,
            short_long_boundary_s=30.0,
            lull_timeout_s=5.0,
            base_tolerance=0.30,
            clock=clock,
            scheduler=scheduler,
            rng=lambda: 0.99,  # roll > tolerance → yield
            name_resolver=lambda uid: f"Resolved-{uid}",
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = MagicMock()

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)

        assert tracker.interrupt_starter_name == "Resolved-42"


# ---------------------------------------------------------------------------
# Step 9 — Short interruption during GENERATING (polite wait)
# ---------------------------------------------------------------------------


class TestShortGeneratingDispatch:
    """Step 9: short interruption during GENERATING → polite wait.

    Familiar keeps generating; delivery gate already opens on finalize so
    playback proceeds naturally. Dispatch logs + stashes interrupter's
    transcript on the tracker for chronological flush in
    ``_run_voice_response`` after original buffer write.
    """

    def test_dispatch_log_format(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_transcript(42, "hello")
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        with caplog.at_level(logging.INFO):
            scheduler.fire_for(detector._finalize_burst)
        assert any(
            "Polite Wait" in _strip(r.message)
            and "speaker=42" in _strip(r.message)
            and "short@GENERATING" in _strip(r.message)
            for r in caplog.records
        )

    def test_transcript_stashed_on_tracker(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5,
            boundary_s=4.0,
            name_resolver=lambda uid: f"Name-{uid}",
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_transcript(42, "hello")
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert tracker.pending_interrupter_turns == [("Name-42", "hello")]

    def test_transcript_stashed_without_resolver_uses_fallback(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_transcript(42, "hello")
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert tracker.pending_interrupter_turns == [("User-42", "hello")]

    def test_llm_generation_task_not_cancelled(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        fake_task = MagicMock()
        tracker.generation_task = fake_task
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_transcript(42, "hello")
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        fake_task.cancel.assert_not_called()
        assert tracker.generation_task is fake_task

    @pytest.mark.asyncio
    async def test_delivery_gate_opens_with_short(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        wait_task = asyncio.create_task(detector.wait_for_lull())
        await asyncio.sleep(0)
        scheduler.fire_for(detector._finalize_burst)
        result = await wait_task
        assert result is InterruptionClass.short

    def test_empty_transcript_guarded(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5,
            boundary_s=4.0,
            name_resolver=lambda uid: f"Name-{uid}",
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        # no on_transcript call — transcript stays empty
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert tracker.pending_interrupter_turns == []

    def test_long_generating_leaves_pending_untouched(self) -> None:
        # Regression: long@GENERATING path does not populate
        # pending_interrupter_turns (early-cancel path uses its own callback).
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5,
            boundary_s=4.0,
            name_resolver=lambda uid: f"Name-{uid}",
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(5.0)  # exceeds boundary → long
        detector.on_transcript(42, "hey stop")
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert tracker.pending_interrupter_turns == []


class TestInterruptionLogNames:
    """Step 10: log lines use resolved display names instead of raw user IDs."""

    def test_min_crossed_log_uses_resolved_name(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5,
            boundary_s=4.0,
            name_resolver=lambda _uid: "Alice",
        )
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._on_min_crossed)
        assert any(
            "speaker=Alice" in _strip(r.message)
            and "min_threshold" in _strip(r.message)
            for r in caplog.records
        )

    def test_finalize_log_uses_resolved_name(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.5,
            boundary_s=4.0,
            name_resolver=lambda _uid: "Alice",
        )
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        assert any(
            "speaker=Alice" in _strip(r.message)
            and "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
            for r in caplog.records
        )

    def test_fallback_when_no_resolver(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # No name_resolver — raw ID rendered as string, no crash.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        registry.get(1).state = ResponseState.GENERATING
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        assert any(
            "speaker=42" in _strip(r.message)
            and "Interrupt" in _strip(r.message)
            and "min_threshold" not in _strip(r.message)
            for r in caplog.records
        )


class TestLongSpeakingPushThrough:
    """Step 11b: long interruption while SPEAKING with push-through.

    When the familiar decides to keep talking at Moment 1 (push-through)
    but the burst grows past ``short_long_boundary_s``, the interrupter
    transcript must still be forwarded to the push-through callback so it
    lands in history.
    """

    def test_dispatch_log_emitted(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        push_through_calls: list[tuple[int, str]] = []
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.99,  # high tolerance → rng=0.01 < 0.99 → keep
            rng=lambda: 0.01,
            on_push_through_transcript=lambda uid, text: push_through_calls.append((
                uid,
                text,
            )),
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = _FakeVC()  # ty: ignore[invalid-assignment]

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(2.0)  # crosses min → Moment 1 → keep
            scheduler.fire_for(detector._on_min_crossed)
            clock.advance(3.0)  # total 5s > boundary_s=4 → long
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)
        assert any(
            "long@SPEAKING" in _strip(r.message) and "Push Through" in _strip(r.message)
            for r in caplog.records
        )

    def test_push_through_callback_invoked(self) -> None:
        push_through_calls: list[tuple[int, str]] = []
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.99,
            rng=lambda: 0.01,
            on_push_through_transcript=lambda uid, text: push_through_calls.append((
                uid,
                text,
            )),
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = _FakeVC()  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hey what about this")
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(3.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert push_through_calls == [(42, "hey what about this")]

    def test_yield_path_does_not_invoke_push_through(self) -> None:
        # Roll > tolerance → yield, not push-through; callback must NOT fire.
        push_through_calls: list[tuple[int, str]] = []
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            base_tolerance=0.10,  # low tolerance → rng=0.99 > 0.10 → yield
            rng=lambda: 0.99,
            on_push_through_transcript=lambda uid, text: push_through_calls.append((
                uid,
                text,
            )),
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        tracker.playback_start_time = clock.now
        tracker.vc = _FakeVC()  # ty: ignore[invalid-assignment]

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "actually you know what")
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        clock.advance(3.0)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert push_through_calls == []

    def test_long_generating_does_not_invoke_push_through(self) -> None:
        # long@GENERATING must not fire the push-through callback.
        push_through_calls: list[tuple[int, str]] = []
        detector, registry, clock, scheduler = _make_detector_with_callbacks(
            min_s=1.5,
            boundary_s=4.0,
            on_push_through_transcript=lambda uid, text: push_through_calls.append((
                uid,
                text,
            )),
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING

        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "wait hold on")
        clock.advance(5.0)  # exceeds boundary → long
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)

        assert push_through_calls == []
