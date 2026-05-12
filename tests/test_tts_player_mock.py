"""Tests for :class:`familiar_connect.tts_player.mock.MockTTSPlayer`.

Verify the mock actually honours :class:`TurnScope` cancellation —
otherwise the barge-in integration test would be meaningless.
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.bus.envelope import TurnScope
from familiar_connect.tts_player import MockTTSPlayer


class TestMockTTSPlayer:
    @pytest.mark.asyncio
    async def test_plays_full_text_when_not_cancelled(self) -> None:
        player = MockTTSPlayer(ms_per_word=5)
        scope = TurnScope(turn_id="t", session_id="s", started_at=0.0)
        await player.speak("hello world four words", scope=scope)
        # Four words x 5ms = 20ms min; allow slack
        assert player.total_played_ms >= 15
        assert player.calls == [("hello world four words", False)]

    @pytest.mark.asyncio
    async def test_stops_promptly_on_cancel(self) -> None:
        player = MockTTSPlayer(ms_per_word=20, poll_ms=5)
        scope = TurnScope(turn_id="t", session_id="s", started_at=0.0)

        async def canceller() -> None:
            await asyncio.sleep(0.03)  # 30ms
            scope.cancel()

        cancel_task = asyncio.create_task(canceller())
        # 10 words x 20ms = 200ms total if not cancelled
        text = "one two three four five six seven eight nine ten"
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await player.speak(text, scope=scope)
        elapsed_ms = int((loop.time() - t0) * 1000)
        await cancel_task

        # Should stop within ~cancel_delay + poll_ms, not play all 200ms
        assert elapsed_ms < 80
        assert player.total_played_ms < 80
        assert player.calls == [(text, True)]  # True = was cancelled

    @pytest.mark.asyncio
    async def test_stop_flushes_current_playback(self) -> None:
        player = MockTTSPlayer(ms_per_word=30, poll_ms=5)
        scope = TurnScope(turn_id="t", session_id="s", started_at=0.0)

        async def stopper() -> None:
            await asyncio.sleep(0.02)
            await player.stop()

        task = asyncio.create_task(stopper())
        t0 = asyncio.get_running_loop().time()
        await player.speak("one two three four", scope=scope)
        elapsed_ms = int((asyncio.get_running_loop().time() - t0) * 1000)
        await task

        assert elapsed_ms < 60
