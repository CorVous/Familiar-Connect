"""Tests for the voice-response state machine scaffold (Steps 3 + 5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

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
    import asyncio
    from collections.abc import Callable


class TestResponseStateEnum:
    def test_has_three_states(self) -> None:
        assert len(list(ResponseState)) == 3

    def test_idle_generating_speaking(self) -> None:
        assert ResponseState.IDLE.value == "IDLE"
        assert ResponseState.GENERATING.value == "GENERATING"
        assert ResponseState.SPEAKING.value == "SPEAKING"


class TestResponseTrackerDefaults:
    def test_new_tracker_is_idle(self) -> None:
        t = ResponseTracker(guild_id=42)
        assert t.state is ResponseState.IDLE

    def test_new_tracker_has_no_task(self) -> None:
        t = ResponseTracker(guild_id=42)
        assert t.generation_task is None

    def test_new_tracker_has_no_response_text(self) -> None:
        t = ResponseTracker(guild_id=42)
        assert t.response_text is None

    def test_new_tracker_has_empty_timestamps(self) -> None:
        t = ResponseTracker(guild_id=42)
        assert t.timestamps == []

    def test_new_tracker_is_not_unsolicited(self) -> None:
        t = ResponseTracker(guild_id=42)
        assert t.is_unsolicited is False

    def test_new_tracker_has_zero_mood_modifier(self) -> None:
        t = ResponseTracker(guild_id=42)
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
            "tracker guild=7" in rec.message and "IDLE→GENERATING" in rec.message
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
        assert not any("GENERATING→GENERATING" in rec.message for rec in caplog.records)
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
        assert any("unsolicited=True" in rec.message for rec in caplog.records)


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
        assert not any("interruption:" in rec.message for rec in caplog.records)


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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "during SPEAKING" in msgs[0]
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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "by user=42" in msgs[0]

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
            r.message for r in caplog.records if "min threshold crossed" in r.message
        ]
        assert len(crossed) == 1
        assert "by user=42" in crossed[0]

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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "by user=42" in msgs[0]
        assert "user=43" not in msgs[0]


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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "during GENERATING" in msgs[0]
        assert "by user=42" in msgs[0]

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
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "during SPEAKING" in msgs[0]
        assert "by user=42" in msgs[0]

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
        assert not any("interruption:" in r.message for r in caplog.records)

    def test_generating_to_idle_with_active_speech_cancels_min_timer(
        self,
    ) -> None:
        # Same abort behaviour while a user is still actively speaking:
        # no pending timers should survive GENERATING → IDLE.
        detector, registry, _clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        tracker = registry.get(1)
        tracker.transition(ResponseState.GENERATING)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        assert scheduler.has_pending(detector._on_min_crossed)
        tracker.transition(ResponseState.IDLE)
        assert not scheduler.has_pending(detector._on_min_crossed)


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
        assert not any("min threshold crossed" in r.message for r in caplog.records)

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
            r.message for r in caplog.records if "min threshold crossed" in r.message
        ]
        assert len(crossed) == 1
        assert "during SPEAKING" in crossed[0]
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
            assert not any("min threshold crossed" in r.message for r in caplog.records)
            clock.advance(0.5)  # t=2.5
            # New speech: now-started = 2.5 ≥ 2 → log fires.
            detector.on_voice_activity(42, VoiceActivityEvent.started)
        crossed = [
            r.message for r in caplog.records if "min threshold crossed" in r.message
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
            assert not any("min threshold crossed" in r.message for r in caplog.records)
            clock.advance(1.0)  # t=2.5
            # ``last_ended_at - started_at`` = 2.5 ≥ 2 → log fires.
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
        crossed = [
            r.message for r in caplog.records if "min threshold crossed" in r.message
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
            r.message for r in caplog.records if "min threshold crossed" in r.message
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
        assert not any("min threshold crossed" in r.message for r in caplog.records)

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
        assert not any("min threshold crossed" in r.message for r in caplog.records)


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
        assert any("→ keep_talking" in r.message for r in caplog.records)

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
        assert any("→ yield" in r.message for r in caplog.records)

    def test_should_keep_talking_log_includes_all_components(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        t = ResponseTracker(guild_id=1, is_unsolicited=True, mood_modifier=0.10)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            t.should_keep_talking(0.30, rng=lambda: 0.50)
        msgs = [r.message for r in caplog.records if "toll:" in r.message]
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
        toll = [r.message for r in caplog.records if "toll:" in r.message]
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
        assert not any("toll:" in r.message for r in caplog.records)

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
        toll = [r.message for r in caplog.records if "toll:" in r.message]
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


class TestResponseTrackerInterruptionFields:
    """Step 8: pause-then-commit state on the tracker."""

    def test_new_tracker_delivery_gate_is_set(self) -> None:
        t = ResponseTracker(guild_id=1)
        assert t.delivery_gate.is_set()

    def test_new_tracker_cancel_committed_is_false(self) -> None:
        t = ResponseTracker(guild_id=1)
        assert t.cancel_committed is False

    def test_new_tracker_pending_buffer_is_empty(self) -> None:
        t = ResponseTracker(guild_id=1)
        assert t.pending_buffer == []

    def test_idle_resets_delivery_gate_cancel_and_buffer(self) -> None:
        from familiar_connect.chattiness import BufferedMessage  # noqa: PLC0415

        t = ResponseTracker(guild_id=1)
        t.state = ResponseState.GENERATING
        t.delivery_gate.clear()
        t.cancel_committed = True
        t.pending_buffer = [BufferedMessage(speaker="u", text="x", timestamp=0.0)]
        t.transition(ResponseState.IDLE)
        assert t.delivery_gate.is_set()
        assert t.cancel_committed is False
        assert t.pending_buffer == []


class TestInterruptionDetectorTranscriptAccumulator:
    """Step 8: burst-scoped transcript capture."""

    def test_on_transcript_ignored_when_no_active_burst(self) -> None:
        detector, _registry, _clock, _scheduler = _make_detector()
        detector.on_transcript(42, "pre-burst noise")
        assert detector._burst_transcript == []

    def test_on_transcript_accumulates_during_active_burst(self) -> None:
        detector, registry, clock, _scheduler = _make_detector()
        registry.get(1).state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hello")
        clock.advance(0.1)
        detector.on_transcript(42, "world")
        assert detector._burst_transcript == ["hello", "world"]

    def test_finalize_resets_transcript(self) -> None:
        detector, registry, clock, scheduler = _make_detector()
        registry.get(1).state = ResponseState.SPEAKING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "hey")
        clock.advance(0.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert detector._burst_transcript == []


class TestInterruptionDetectorMinCrossedClearsGate:
    """Step 8: gate cleared at min during GENERATING."""

    def test_min_crossed_generating_clears_delivery_gate(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        assert tracker.delivery_gate.is_set()
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        assert not tracker.delivery_gate.is_set()

    def test_min_crossed_speaking_does_not_clear_gate(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=2.0, boundary_s=30.0
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(2.0)
        scheduler.fire_for(detector._on_min_crossed)
        assert tracker.delivery_gate.is_set()

    def test_short_finalize_during_generating_releases_gate(self) -> None:
        detector, registry, clock, scheduler = _make_detector(
            min_s=1.0, boundary_s=30.0
        )
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        scheduler.fire_for(detector._on_min_crossed)
        assert not tracker.delivery_gate.is_set()
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert tracker.delivery_gate.is_set()


class _FakeTask:
    """Minimal stand-in for :class:`asyncio.Task` that records cancel()."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> bool:
        self.cancelled = True
        return True

    def done(self) -> bool:
        return self.cancelled


class TestInterruptionDetectorLongCommit:
    """Step 8: long boundary commits cancellation + dispatch."""

    def test_long_boundary_cancels_generation_task(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.0, boundary_s=3.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        task = _FakeTask()
        tracker.generation_task = cast("asyncio.Task[Any]", task)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(3.0)
        scheduler.fire_for(detector._on_long_boundary_crossed)
        assert task.cancelled is True

    def test_long_boundary_sets_cancel_committed_and_releases_gate(self) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.0, boundary_s=3.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.0)
        scheduler.fire_for(detector._on_min_crossed)
        assert not tracker.delivery_gate.is_set()
        clock.advance(2.0)
        scheduler.fire_for(detector._on_long_boundary_crossed)
        assert tracker.cancel_committed is True
        assert tracker.delivery_gate.is_set()

    def test_long_boundary_during_speaking_does_not_commit_or_cancel(
        self,
    ) -> None:
        detector, registry, clock, scheduler = _make_detector(min_s=1.0, boundary_s=3.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.SPEAKING
        task = _FakeTask()
        tracker.generation_task = cast("asyncio.Task[Any]", task)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(3.0)
        scheduler.fire_for(detector._on_long_boundary_crossed)
        assert tracker.cancel_committed is False
        assert task.cancelled is False

    def test_long_boundary_dispatches_with_transcript_and_buffer(self) -> None:
        from familiar_connect.chattiness import BufferedMessage  # noqa: PLC0415

        dispatched: list[dict[str, object]] = []

        def _dispatch(
            *,
            speaker_user_id: int,
            transcript: str,
            original_buffer: list[BufferedMessage],
            tracker: ResponseTracker,
        ) -> None:
            dispatched.append(
                {
                    "speaker_user_id": speaker_user_id,
                    "transcript": transcript,
                    "original_buffer": list(original_buffer),
                    "tracker": tracker,
                },
            )

        detector, registry, clock, scheduler = _make_detector(min_s=1.0, boundary_s=3.0)
        detector.set_dispatch(_dispatch)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        original = [
            BufferedMessage(speaker="alice", text="first", timestamp=0.0),
            BufferedMessage(speaker="alice", text="second", timestamp=0.0),
        ]
        tracker.pending_buffer = list(original)
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        detector.on_transcript(42, "wait no")
        clock.advance(3.0)
        scheduler.fire_for(detector._on_long_boundary_crossed)
        assert len(dispatched) == 1
        payload = dispatched[0]
        assert payload["speaker_user_id"] == 42
        assert payload["transcript"] == "wait no"
        assert payload["original_buffer"] == original
        assert payload["tracker"] is tracker

    def test_short_finalize_does_not_dispatch(self) -> None:
        dispatched: list[object] = []

        def _dispatch(**kwargs: object) -> None:
            dispatched.append(kwargs)

        detector, registry, clock, scheduler = _make_detector(
            min_s=1.0, boundary_s=30.0
        )
        detector.set_dispatch(_dispatch)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(1.5)
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        assert dispatched == []

    def test_long_finalize_is_idempotent_after_mid_burst_commit(self) -> None:
        dispatched: list[object] = []

        def _dispatch(**kwargs: object) -> None:
            dispatched.append(kwargs)

        detector, registry, clock, scheduler = _make_detector(min_s=1.0, boundary_s=3.0)
        detector.set_dispatch(_dispatch)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING
        detector.on_voice_activity(42, VoiceActivityEvent.started)
        clock.advance(3.0)
        scheduler.fire_for(detector._on_long_boundary_crossed)
        assert len(dispatched) == 1
        detector.on_voice_activity(42, VoiceActivityEvent.ended)
        scheduler.fire_for(detector._finalize_burst)
        # long finalize after mid-burst commit: no duplicate dispatch
        assert len(dispatched) == 1
