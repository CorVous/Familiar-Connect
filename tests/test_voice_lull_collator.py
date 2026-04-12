"""Tests for VoiceLullCollator.

Covers the event-flow properties described in
``docs/roadmap/conversation-flow.md`` and the voice lull collator
design: SPEAKING-gated dispatch, transcript buffering, the
``_wait_and_dispatch`` safety net, and hard-reset on SPEAKING=True.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice.lull_collator import VoiceLullCollator


def _final(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=True, start=0.0, end=0.0)


class TestDispatchAfterLull:
    @pytest.mark.asyncio
    async def test_final_then_silence_dispatches(
        self,
        event_loop_policy: object,  # noqa: ARG002
    ) -> None:
        """is_final followed by SPEAKING=False dispatches after lull_timeout."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=1.0
        )

        collator.on_speaking(42, True)
        collator.on_final(42, _final("hello world"))
        collator.on_speaking(42, False)

        # Wait past the lull timeout

        await asyncio.sleep(0.15)

        downstream.assert_awaited_once()
        user_id, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert user_id == 42
        assert result.text == "hello world"
        assert result.is_final is True

    @pytest.mark.asyncio
    async def test_two_finals_in_one_burst_fuse_into_single_dispatch(self) -> None:
        """Multiple is_finals within one SPEAKING burst collapse into one call."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=1.0
        )

        collator.on_speaking(42, True)
        collator.on_final(42, _final("hey Aria,"))
        collator.on_final(42, _final("what's the weather?"))
        collator.on_speaking(42, False)

        await asyncio.sleep(0.15)

        downstream.assert_awaited_once()
        _, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert result.text == "hey Aria, what's the weather?"


class TestWaitForTrailingFinal:
    @pytest.mark.asyncio
    async def test_lull_with_audio_pending_waits_for_final(self) -> None:
        """Lull fires while audio_pending=True → wait for the is_final."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=1.0
        )

        # User spoke, then went silent; Deepgram hasn't delivered yet
        collator.on_speaking(42, True)
        collator.on_speaking(42, False)

        # Let the lull timer fire
        await asyncio.sleep(0.1)
        # No dispatch yet — we're waiting for is_final
        downstream.assert_not_awaited()

        # Deepgram finally delivers
        collator.on_final(42, _final("late arrival"))
        await asyncio.sleep(0.05)

        downstream.assert_awaited_once()
        _, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert result.text == "late arrival"

    @pytest.mark.asyncio
    async def test_wait_timeout_logs_warning_and_no_dispatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """dispatch_timeout elapses with no is_final → warning, no dispatch."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=0.1
        )

        collator.on_speaking(42, True)
        collator.on_speaking(42, False)

        with caplog.at_level(
            logging.WARNING, logger="familiar_connect.voice.lull_collator"
        ):
            # lull (0.05) + dispatch wait (0.1) + slack
            await asyncio.sleep(0.25)

        downstream.assert_not_awaited()
        assert any("dispatch_timeout" in record.message for record in caplog.records)


class TestSpeakingTrueHardReset:
    @pytest.mark.asyncio
    async def test_speaking_true_cancels_pending_lull_timer(self) -> None:
        """SPEAKING=True while lull timer is pending cancels it."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(downstream, lull_timeout=0.1, dispatch_timeout=1.0)

        collator.on_speaking(42, True)
        collator.on_final(42, _final("first"))
        collator.on_speaking(42, False)
        # Before lull fires, user speaks again
        await asyncio.sleep(0.03)
        collator.on_speaking(42, True)

        # Sleep past where the original lull would have fired
        await asyncio.sleep(0.15)
        downstream.assert_not_awaited()

        # Now user finishes the second burst
        collator.on_final(42, _final("second"))
        collator.on_speaking(42, False)
        await asyncio.sleep(0.15)

        # Both buffered texts dispatch as one unit
        downstream.assert_awaited_once()
        _, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert result.text == "first second"

    @pytest.mark.asyncio
    async def test_speaking_true_cancels_running_wait_task(self) -> None:
        """SPEAKING=True during a wait task cancels it — no wait-task dispatch.

        A late ``is_final`` may still start a fresh transcript-gap timer
        (see :class:`TestTranscriptOnlyDispatch`), but the cancelled wait
        task itself never fires.
        """
        downstream = AsyncMock()
        collator = VoiceLullCollator(downstream, lull_timeout=0.2, dispatch_timeout=1.0)

        # Enter the wait-for-final branch (audio_pending=True, no text).
        collator.on_speaking(42, True)
        collator.on_speaking(42, False)
        await asyncio.sleep(0.25)  # let lull fire → wait task running

        # Hard reset before the is_final arrives
        collator.on_speaking(42, True)
        await asyncio.sleep(0.05)

        # The cancelled wait task has not fired — no dispatch yet.
        downstream.assert_not_awaited()

        # SPEAKING=True again (still no SPEAKING=False, no is_final) —
        # audio_pending stays True, no timer runs.
        await asyncio.sleep(0.3)
        downstream.assert_not_awaited()


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cancels_pending_timers_and_tasks(self) -> None:
        """close() cancels all per-user timers and in-flight wait tasks."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(downstream, lull_timeout=1.0, dispatch_timeout=5.0)

        # User A: pending lull timer (won't fire before close())
        collator.on_speaking(1, True)
        collator.on_final(1, _final("A"))
        collator.on_speaking(1, False)

        # User B: wait task in flight
        collator.on_speaking(2, True)
        collator.on_speaking(2, False)
        # Need a smaller lull for user B to reach the wait branch — use a
        # second collator with short lull solely for that branch.
        collator2 = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=5.0
        )
        collator2.on_speaking(2, True)
        collator2.on_speaking(2, False)
        await asyncio.sleep(0.08)  # let collator2's lull fire → wait task

        await collator.close()
        await collator2.close()

        # Nothing dispatches after close
        collator.on_final(1, _final("late-A"))
        collator2.on_final(2, _final("late-B"))
        await asyncio.sleep(0.1)
        downstream.assert_not_awaited()


class TestNeither:
    @pytest.mark.asyncio
    async def test_lull_with_no_audio_and_no_text_is_noop(self) -> None:
        """Lull fires with nothing buffered and no audio pending → no dispatch."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=1.0
        )

        # A lone SPEAKING=False starts the timer but audio_pending stays False
        # (SPEAKING=True was never seen).
        collator.on_speaking(42, False)
        await asyncio.sleep(0.15)
        downstream.assert_not_awaited()


class TestTranscriptOnlyDispatch:
    """Discord does not reliably emit SPEAKING=False for remote users.

    The collator must still dispatch after ``lull_timeout`` seconds of
    transcript silence, using each ``is_final`` arrival as an implicit
    "possibly paused" signal that (re)starts the lull timer.
    """

    @pytest.mark.asyncio
    async def test_final_without_speaking_false_dispatches_after_lull(
        self,
    ) -> None:
        """is_final + no SPEAKING=False → dispatch after lull_timeout."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(
            downstream, lull_timeout=0.05, dispatch_timeout=1.0
        )

        # SPEAKING=True arrives, a transcript lands, and Discord never
        # bothers to send SPEAKING=False.
        collator.on_speaking(42, True)
        collator.on_final(42, _final("hi"))

        await asyncio.sleep(0.15)

        downstream.assert_awaited_once()
        user_id, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert user_id == 42
        assert result.text == "hi"

    @pytest.mark.asyncio
    async def test_two_finals_restart_timer_and_fuse(self) -> None:
        """Back-to-back is_finals restart the timer and fuse into one reply."""
        downstream = AsyncMock()
        collator = VoiceLullCollator(downstream, lull_timeout=0.1, dispatch_timeout=1.0)

        collator.on_speaking(42, True)
        collator.on_final(42, _final("one"))
        # Second is_final arrives before the first one's lull would fire;
        # it must restart the timer, not leave the stale one running.
        await asyncio.sleep(0.05)
        collator.on_final(42, _final("two"))

        # Sleep past where the first is_final's original timer would have
        # fired, but before the restarted timer would fire.
        await asyncio.sleep(0.08)
        downstream.assert_not_awaited()

        # Now sleep past the restarted timer.
        await asyncio.sleep(0.1)
        downstream.assert_awaited_once()
        _, result = downstream.await_args.args  # ty: ignore[unresolved-attribute]
        assert result.text == "one two"
