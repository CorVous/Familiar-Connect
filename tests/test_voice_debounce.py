"""Tests for voice transcription debounce in _build_voice_response_handler.

Verifies that multiple rapid voice transcription results are buffered
and only trigger a single LLM generation after a lull timeout, gated
by VAD events (SpeechStarted / UtteranceEnd).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from familiar_connect.bot import _build_voice_response_handler
from familiar_connect.config import (
    ChannelConfig,
    ChannelMode,
    InterruptTolerance,
)
from familiar_connect.llm import Message
from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice.interruption import ResponseState


def _make_familiar_mock(*, lull_timeout: float = 0.05) -> MagicMock:
    """Build a minimal Familiar mock with just enough for the handler."""
    familiar = MagicMock()
    familiar.id = "test-familiar"

    # Config
    familiar.config.interrupt_tolerance = InterruptTolerance.average
    familiar.config.min_interruption_s = 1.5
    familiar.config.short_long_boundary_s = 4.0
    familiar.config.lull_timeout = lull_timeout
    familiar.config.history_window_size = 20
    familiar.config.depth_inject_position = 0
    familiar.config.depth_inject_role = "system"
    familiar.config.display_tz = "UTC"

    # Channel config
    channel_config = ChannelConfig(
        mode=ChannelMode.imitate_voice,
        budget_tokens=1000,
        deadline_s=5.0,
    )
    familiar.channel_configs.get.return_value = channel_config

    # Pipeline mock
    pipeline_mock = AsyncMock()
    pipeline_mock.assemble = AsyncMock(return_value=MagicMock(outcomes=[]))
    pipeline_mock.run_post_processors = AsyncMock(return_value="reply text")
    familiar.build_pipeline.return_value = pipeline_mock

    # LLM
    familiar.llm_client.chat = AsyncMock(
        return_value=Message(role="assistant", content="reply text")
    )

    # History store
    familiar.history_store.append_turn = MagicMock()
    familiar.history_store.recent = MagicMock(return_value=[])

    # No TTS for simpler tests
    familiar.tts_client = None

    # No mood evaluator
    familiar.mood_evaluator = None

    return familiar


def _make_vc_mock() -> MagicMock:
    """Build a minimal VoiceClient mock."""
    vc = MagicMock()
    vc.is_playing.return_value = False
    return vc


def _build(*, lull_timeout: float = 0.05) -> tuple[Any, Any, Any, Any, MagicMock]:
    """Build handler + mocks.

    Returns (handler, tracker, detector, vad_cb, familiar).
    """
    familiar = _make_familiar_mock(lull_timeout=lull_timeout)
    vc = _make_vc_mock()
    handler, tracker, detector, vad_cb = _build_voice_response_handler(
        vc=vc,
        familiar=familiar,
        voice_channel_id=100,
        guild_id=1,
        user_names={42: "Alice", 99: "Bob"},
    )
    return handler, tracker, detector, vad_cb, familiar


class TestVoiceDebounce:
    """Verify that voice results are buffered and generation waits for a lull."""

    @pytest.fixture(autouse=True)
    def _patch_chat(self) -> Iterator[None]:  # type: ignore[misc]
        """Patch assemble_chat_messages — we test timing, not rendering.

        Yields:
            None

        """
        with patch(
            "familiar_connect.bot.assemble_chat_messages",
            return_value=[Message(role="user", content="hi")],
        ):
            yield

    @pytest.mark.asyncio
    async def test_single_result_fallback_generates_after_lull(self) -> None:
        """Result with no active speakers uses the fallback timer."""
        handler, tracker, _, _, familiar = _build(lull_timeout=0.05)

        result = TranscriptionResult(
            text="hello there", is_final=True, start=0.0, end=1.0
        )
        await handler(42, result)

        # Immediately after, tracker should still be IDLE (debouncing).
        assert tracker.state is ResponseState.IDLE

        # Wait for lull + generation to complete.
        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_vad_speech_blocks_lull_timer(self) -> None:
        """SpeechStarted cancels the lull timer; UtteranceEnd restarts it."""
        handler, _, _, vad_cb, familiar = _build(lull_timeout=0.05)

        # Simulate: user starts speaking, then final arrives.
        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Even though a result arrived, no timer should fire because
        # the user is still speaking (SpeechStarted, no UtteranceEnd).
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_client.chat.assert_not_called()

        # Now the user stops.
        vad_cb(42, "UtteranceEnd")

        # Lull timer starts now.
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_vad_links_segments_across_brief_pause(self) -> None:
        """Segments separated by a brief pause are linked into one response.

        Simulates: user says "hello" → brief pause → "how are you",
        with Deepgram finalising "hello" during the pause but VAD
        showing continued speech.
        """
        handler, _, _, vad_cb, familiar = _build(lull_timeout=0.05)

        # Segment 1: user starts talking
        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Brief pause — Deepgram sends UtteranceEnd then SpeechStarted
        vad_cb(42, "UtteranceEnd")
        # User resumes before lull fires
        vad_cb(42, "SpeechStarted")

        # Segment 2: second part of the utterance
        r2 = TranscriptionResult(text="how are you", is_final=True, start=0.8, end=1.5)
        await handler(42, r2)

        # Still speaking — no generation yet.
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_client.chat.assert_not_called()

        # User finishes
        vad_cb(42, "UtteranceEnd")

        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)

        # Single generation with combined text
        familiar.llm_client.chat.assert_called_once()
        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "hello how are you"

    @pytest.mark.asyncio
    async def test_multiple_speakers_wait_for_all_silent(self) -> None:
        """Lull timer only starts when ALL speakers stop."""
        handler, _, _, vad_cb, familiar = _build(lull_timeout=0.05)

        vad_cb(42, "SpeechStarted")
        vad_cb(99, "SpeechStarted")

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # User 42 stops, but user 99 is still talking.
        vad_cb(42, "UtteranceEnd")
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_client.chat.assert_not_called()

        r2 = TranscriptionResult(text="world", is_final=True, start=0.5, end=1.0)
        await handler(99, r2)

        # Now user 99 stops.
        vad_cb(99, "UtteranceEnd")
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracker_idle_during_debounce(self) -> None:
        """Tracker stays IDLE while results are being buffered."""
        handler, tracker, _, vad_cb, _ = _build(lull_timeout=0.1)

        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hi", is_final=True, start=0.0, end=0.3)
        await handler(42, r1)
        await asyncio.sleep(0.01)
        assert tracker.state is ResponseState.IDLE

        r2 = TranscriptionResult(text="there", is_final=True, start=0.3, end=0.6)
        await handler(42, r2)
        await asyncio.sleep(0.01)
        assert tracker.state is ResponseState.IDLE

    @pytest.mark.asyncio
    async def test_combined_text_joins_all_utterances(self) -> None:
        """The LLM receives the combined text of all buffered utterances."""
        handler, _, _, _vad_cb, familiar = _build(lull_timeout=0.05)

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        r2 = TranscriptionResult(text="how are you", is_final=True, start=0.5, end=1.5)

        # No active speakers — fallback timer used.
        await handler(42, r1)
        await handler(42, r2)

        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "hello how are you"

    @pytest.mark.asyncio
    async def test_successive_results_reset_lull_timer(self) -> None:
        """Each new transcript resets the lull countdown.

        Simulates continuous speech where Deepgram finalises segments
        faster than the lull timeout.  Without timer-reset, the first
        timer could fire before later segments arrive.
        """
        handler, _, _, _vad_cb, familiar = _build(lull_timeout=0.08)

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Wait less than lull_timeout, then send another result.
        await asyncio.sleep(0.04)
        familiar.llm_client.chat.assert_not_called()

        r2 = TranscriptionResult(text="second", is_final=True, start=0.5, end=1.0)
        await handler(42, r2)

        # Wait less than lull_timeout again — timer was reset by r2.
        await asyncio.sleep(0.04)
        familiar.llm_client.chat.assert_not_called()

        r3 = TranscriptionResult(text="third", is_final=True, start=1.0, end=1.5)
        await handler(42, r3)

        # Now wait for the full lull to expire after the LAST result.
        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        # Single generation with all three segments.
        familiar.llm_client.chat.assert_called_once()
        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "first second third"
