"""Tests for voice interruption detection and response state management.

Covers familiar_connect.voice.interruption per the design in
future-features/interruption-flow.md.

Steps covered:
  1. ResponseState enum
  2. InterruptionKind enum
  3. classify_interruption() pure classifier
  4. should_keep_talking() RNG toll check
  5. split_at_elapsed() word position tracking
  6. ResponseTracker state machine
  7. InterruptionDetector dispatch
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import AsyncMock

import pytest

from familiar_connect.tts import WordTimestamp
from familiar_connect.voice.interruption import (
    InterruptionDetector,
    InterruptionKind,
    ResponseState,
    ResponseTracker,
    classify_interruption,
    should_keep_talking,
    split_at_elapsed,
)

# ---------------------------------------------------------------------------
# Step 1 — ResponseState enum
# ---------------------------------------------------------------------------


class TestResponseState:
    def test_has_three_members(self) -> None:
        assert len(list(ResponseState)) == 3

    def test_values(self) -> None:
        assert ResponseState.IDLE.value == "idle"
        assert ResponseState.GENERATING.value == "generating"
        assert ResponseState.SPEAKING.value == "speaking"


# ---------------------------------------------------------------------------
# Step 2 — InterruptionKind enum
# ---------------------------------------------------------------------------


class TestInterruptionKind:
    def test_has_two_members(self) -> None:
        assert len(list(InterruptionKind)) == 2

    def test_values(self) -> None:
        assert InterruptionKind.short.value == "short"
        assert InterruptionKind.long.value == "long"


# ---------------------------------------------------------------------------
# Step 3 — classify_interruption
# ---------------------------------------------------------------------------


class TestClassifyInterruption:
    def test_below_minimum_returns_none(self) -> None:
        assert (
            classify_interruption(
                1.0, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is None
        )

    def test_at_minimum_returns_short(self) -> None:
        assert (
            classify_interruption(
                1.5, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is InterruptionKind.short
        )

    def test_between_min_and_boundary_returns_short(self) -> None:
        assert (
            classify_interruption(
                2.5, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is InterruptionKind.short
        )

    def test_at_boundary_returns_long(self) -> None:
        assert (
            classify_interruption(
                4.0, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is InterruptionKind.long
        )

    def test_above_boundary_returns_long(self) -> None:
        assert (
            classify_interruption(
                7.0, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is InterruptionKind.long
        )

    def test_zero_duration_returns_none(self) -> None:
        assert (
            classify_interruption(
                0.0, min_interruption_s=1.5, short_long_boundary_s=4.0
            )
            is None
        )


# ---------------------------------------------------------------------------
# Step 4 — should_keep_talking
# ---------------------------------------------------------------------------


class TestShouldKeepTalking:
    def test_tolerance_zero_always_yields(self) -> None:
        """With tolerance 0.0, the familiar always yields."""
        rng = random.Random(42)  # noqa: S311
        for _ in range(100):
            assert not should_keep_talking(0.0, rng=rng)

    def test_tolerance_one_never_yields(self) -> None:
        """With tolerance 1.0, the familiar never yields."""
        rng = random.Random(42)  # noqa: S311
        for _ in range(100):
            assert should_keep_talking(1.0, rng=rng)

    def test_mood_modifier_positive_increases_tolerance(self) -> None:
        """A positive modifier makes the familiar more stubborn."""
        rng = random.Random(42)  # noqa: S311
        # With base 0.3 + modifier 0.7 = 1.0, should always keep talking
        for _ in range(100):
            assert should_keep_talking(0.3, 0.7, rng=rng)

    def test_mood_modifier_negative_decreases_tolerance(self) -> None:
        """A negative modifier makes the familiar yield more."""
        rng = random.Random(42)  # noqa: S311
        # With base 0.3 + modifier -0.3 = 0.0, should always yield
        for _ in range(100):
            assert not should_keep_talking(0.3, -0.3, rng=rng)

    def test_effective_clamped_to_0_1(self) -> None:
        """Effective tolerance is clamped to [0.0, 1.0]."""
        rng = random.Random(42)  # noqa: S311
        # base 0.8 + modifier 0.5 = 1.3, clamped to 1.0 → always keeps
        for _ in range(100):
            assert should_keep_talking(0.8, 0.5, rng=rng)
        # base 0.2 + modifier -0.5 = -0.3, clamped to 0.0 → always yields
        for _ in range(100):
            assert not should_keep_talking(0.2, -0.5, rng=rng)

    def test_default_mood_modifier_is_zero(self) -> None:
        """When mood_modifier is not provided, it defaults to 0.0."""
        rng = random.Random(42)  # noqa: S311
        # Just verify it runs without error with default
        result = should_keep_talking(0.5, rng=rng)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Step 5 — split_at_elapsed
# ---------------------------------------------------------------------------


class TestSplitAtElapsed:
    def _timestamps(self) -> list[WordTimestamp]:
        return [
            WordTimestamp(word="hello", start_ms=0.0, end_ms=400.0),
            WordTimestamp(word="world", start_ms=400.0, end_ms=800.0),
            WordTimestamp(word="how", start_ms=800.0, end_ms=1100.0),
            WordTimestamp(word="are", start_ms=1100.0, end_ms=1400.0),
            WordTimestamp(word="you", start_ms=1400.0, end_ms=1800.0),
        ]

    def test_split_at_zero_returns_empty_delivered(self) -> None:
        delivered, remaining = split_at_elapsed(self._timestamps(), 0.0)
        assert not delivered
        assert remaining == "hello world how are you"

    def test_split_at_end_returns_all_delivered(self) -> None:
        delivered, remaining = split_at_elapsed(self._timestamps(), 2000.0)
        assert delivered == "hello world how are you"
        assert not remaining

    def test_split_mid_sentence(self) -> None:
        # After 900ms, "hello" (end 400) and "world" (end 800) are delivered
        delivered, remaining = split_at_elapsed(self._timestamps(), 900.0)
        assert delivered == "hello world"
        assert remaining == "how are you"

    def test_empty_timestamps(self) -> None:
        delivered, remaining = split_at_elapsed([], 500.0)
        assert not delivered
        assert not remaining

    def test_split_exactly_at_word_end(self) -> None:
        # At exactly 800ms, "hello" and "world" are both delivered
        delivered, remaining = split_at_elapsed(self._timestamps(), 800.0)
        assert delivered == "hello world"
        assert remaining == "how are you"


# ---------------------------------------------------------------------------
# Step 6 — ResponseTracker
# ---------------------------------------------------------------------------


class TestResponseTracker:
    def test_starts_idle(self) -> None:
        tracker = ResponseTracker()
        assert tracker.state is ResponseState.IDLE

    def test_start_generating_transitions(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        assert tracker.state is ResponseState.GENERATING
        assert tracker.generation_task is task

    def test_start_generating_from_non_idle_raises(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        with pytest.raises(RuntimeError, match="Cannot start generating"):
            tracker.start_generating(task)

    def test_generation_complete_stores_text(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.generation_complete("Hello, how are you?")
        assert tracker.response_text == "Hello, how are you?"

    def test_start_speaking_transitions(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.start_speaking()
        assert tracker.state is ResponseState.SPEAKING
        assert tracker.playback_start_time is not None

    def test_start_speaking_from_idle_raises(self) -> None:
        tracker = ResponseTracker()
        with pytest.raises(RuntimeError, match="Cannot start speaking"):
            tracker.start_speaking()

    def test_start_speaking_with_timestamps(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        ts = [WordTimestamp(word="hi", start_ms=0.0, end_ms=300.0)]
        tracker.start_speaking(word_timestamps=ts)
        assert tracker.word_timestamps == ts

    def test_stop_speaking_returns_elapsed_ms(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.start_speaking()
        elapsed = tracker.stop_speaking()
        assert elapsed >= 0.0

    def test_stop_speaking_from_non_speaking_raises(self) -> None:
        tracker = ResponseTracker()
        with pytest.raises(RuntimeError, match="Cannot stop speaking"):
            tracker.stop_speaking()

    def test_reset_returns_to_idle(self) -> None:
        tracker = ResponseTracker()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.generation_complete("hello")
        tracker.mood_modifier = 0.3
        tracker.reset()
        assert tracker.state is ResponseState.IDLE
        assert tracker.generation_task is None
        assert tracker.response_text is None
        assert tracker.word_timestamps == []
        assert tracker.playback_start_time is None
        assert tracker.mood_modifier == 0.0  # noqa: RUF069

    def test_silence_event_cleared_on_start_generating(self) -> None:
        tracker = ResponseTracker()
        tracker.silence_event.set()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        assert not tracker.silence_event.is_set()


# ---------------------------------------------------------------------------
# Step 7 — InterruptionDetector
# ---------------------------------------------------------------------------


class TestInterruptionDetector:
    def _make_detector(
        self,
        tracker: ResponseTracker | None = None,
    ) -> tuple[
        InterruptionDetector,
        ResponseTracker,
        AsyncMock,
        AsyncMock,
        AsyncMock,
        AsyncMock,
    ]:
        t = tracker or ResponseTracker()
        on_short_gen = AsyncMock()
        on_long_gen = AsyncMock()
        on_short_speak = AsyncMock()
        on_long_speak = AsyncMock()
        detector = InterruptionDetector(
            tracker=t,
            min_interruption_s=1.5,
            short_long_boundary_s=4.0,
            on_short_during_generating=on_short_gen,
            on_long_during_generating=on_long_gen,
            on_short_during_speaking=on_short_speak,
            on_long_during_speaking=on_long_speak,
        )
        return detector, t, on_short_gen, on_long_gen, on_short_speak, on_long_speak

    def test_speech_during_idle_ignored(self) -> None:
        detector, _tracker, *mocks = self._make_detector()
        # Tracker is IDLE by default
        detector.on_speech_started(user_id=1, timestamp=0.0)
        result = asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="hey", timestamp=2.0)
        )
        assert result is None
        for mock in mocks:
            mock.assert_not_called()

    def test_short_interruption_during_generating(self) -> None:
        detector, tracker, on_short_gen, on_long_gen, *_ = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        detector.on_speech_started(user_id=1, timestamp=10.0)
        result = asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="wait", timestamp=12.0)
        )
        assert result is not None
        assert result.kind is InterruptionKind.short
        on_short_gen.assert_called_once()
        on_long_gen.assert_not_called()

    def test_long_interruption_during_generating(self) -> None:
        detector, tracker, on_short_gen, on_long_gen, *_ = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        detector.on_speech_started(user_id=1, timestamp=10.0)
        result = asyncio.run(
            detector.on_utterance_end(
                user_id=1, transcript="actually let me explain", timestamp=15.0
            )
        )
        assert result is not None
        assert result.kind is InterruptionKind.long
        on_long_gen.assert_called_once()
        on_short_gen.assert_not_called()

    def test_short_interruption_during_speaking(self) -> None:
        detector, tracker, _, _, on_short_speak, on_long_speak = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.start_speaking()

        detector.on_speech_started(user_id=1, timestamp=10.0)
        result = asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="hmm", timestamp=12.0)
        )
        assert result is not None
        assert result.kind is InterruptionKind.short
        on_short_speak.assert_called_once()
        on_long_speak.assert_not_called()

    def test_long_interruption_during_speaking(self) -> None:
        detector, tracker, _, _, on_short_speak, on_long_speak = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)
        tracker.start_speaking()

        detector.on_speech_started(user_id=1, timestamp=10.0)
        result = asyncio.run(
            detector.on_utterance_end(
                user_id=1, transcript="no stop I need to say something", timestamp=15.0
            )
        )
        assert result is not None
        assert result.kind is InterruptionKind.long
        on_long_speak.assert_called_once()
        on_short_speak.assert_not_called()

    def test_below_minimum_duration_ignored(self) -> None:
        detector, tracker, *mocks = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        detector.on_speech_started(user_id=1, timestamp=10.0)
        result = asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="mm", timestamp=11.0)
        )
        assert result is None
        for mock in mocks:
            mock.assert_not_called()

    def test_multiple_speakers_merged_into_single_event(self) -> None:
        detector, tracker, on_short_gen, *_ = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        detector.on_speech_started(user_id=1, timestamp=10.0)
        detector.on_speech_started(user_id=2, timestamp=10.5)
        result = asyncio.run(
            detector.on_utterance_end(user_id=2, transcript="wait wait", timestamp=12.0)
        )
        assert result is not None
        assert result.interrupter_ids == frozenset({1, 2})
        on_short_gen.assert_called_once()

    def test_on_utterance_end_without_speech_start_returns_none(self) -> None:
        detector, tracker, *mocks = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        # No speech_started before utterance_end
        result = asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="hey", timestamp=12.0)
        )
        assert result is None
        for mock in mocks:
            mock.assert_not_called()

    def test_accumulation_resets_after_dispatch(self) -> None:
        """After dispatching, the detector should reset its internal state."""
        detector, tracker, _on_short_gen, *_ = self._make_detector()
        task = AsyncMock()  # type: ignore[arg-type]
        tracker.start_generating(task)

        # First interruption
        detector.on_speech_started(user_id=1, timestamp=10.0)
        asyncio.run(
            detector.on_utterance_end(user_id=1, transcript="wait", timestamp=12.0)
        )

        # Second speech start - the internal state should be clean
        detector.on_speech_started(user_id=3, timestamp=15.0)
        result = asyncio.run(
            detector.on_utterance_end(user_id=3, transcript="hello", timestamp=17.0)
        )
        assert result is not None
        # Only user 3 should be in the interrupter set (user 1 was reset)
        assert result.interrupter_ids == frozenset({3})
