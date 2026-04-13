"""Tests for the voice-response state machine scaffold (Steps 3 + 5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect.voice.interruption import (
    InterruptionClass,
    InterruptionDetector,
    ResponseState,
    ResponseTracker,
    ResponseTrackerRegistry,
)
from familiar_connect.voice_lull import VoiceActivityEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


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
) -> tuple[InterruptionDetector, ResponseTrackerRegistry, _FakeClock, _FakeScheduler]:
    registry = ResponseTrackerRegistry()
    clock = _FakeClock()
    scheduler = _FakeScheduler()
    detector = InterruptionDetector(
        tracker_registry=registry,
        guild_id=guild_id,
        min_interruption_s=min_s,
        short_long_boundary_s=boundary_s,
        lull_timeout_s=lull_s,
        clock=clock,
        scheduler=scheduler,
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


class TestInterruptionDetectorStateSnapshot:
    def test_records_state_at_burst_start(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # If state transitions during the burst, the classification
        # should report the state observed when the burst began.
        detector, registry, clock, scheduler = _make_detector(min_s=1.5, boundary_s=4.0)
        tracker = registry.get(1)
        tracker.state = ResponseState.GENERATING

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.voice.interruption"
        ):
            detector.on_voice_activity(42, VoiceActivityEvent.started)
            clock.advance(1.0)
            tracker.state = ResponseState.SPEAKING  # state changed mid-burst
            clock.advance(2.0)
            detector.on_voice_activity(42, VoiceActivityEvent.ended)
            scheduler.fire_for(detector._finalize_burst)

        msgs = [
            r.message
            for r in caplog.records
            if "interruption:" in r.message and "min threshold" not in r.message
        ]
        assert len(msgs) == 1
        assert "GENERATING" in msgs[0]


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
