"""Tests for the voice-response state machine scaffold (Step 3)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect.voice.interruption import (
    ResponseState,
    ResponseTracker,
    ResponseTrackerRegistry,
)

if TYPE_CHECKING:
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
