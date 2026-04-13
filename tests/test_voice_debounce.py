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

from familiar_connect.bot import ResponseState, _build_voice_response_handler
from familiar_connect.config import ChannelConfig, ChannelMode
from familiar_connect.llm import Message
from familiar_connect.transcription import TranscriptionResult


def _make_familiar_mock(*, lull_timeout: float = 0.05) -> MagicMock:
    """Build a minimal Familiar mock with just enough for the handler."""
    familiar = MagicMock()
    familiar.id = "test-familiar"

    # Config
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
    familiar.llm_clients = {
        "main_prose": MagicMock(
            chat=AsyncMock(return_value=Message(role="assistant", content="reply text"))
        )
    }

    # History store
    familiar.history_store.append_turn = MagicMock()

    # No TTS for simpler tests
    familiar.tts_client = None

    return familiar


def _make_vc_mock() -> MagicMock:
    """Build a minimal VoiceClient mock."""
    vc = MagicMock()
    vc.is_playing.return_value = False
    return vc


def _build(*, lull_timeout: float = 0.05) -> tuple[Any, Any, Any, Any, MagicMock]:
    """Build handler + mocks.

    Returns (handler, tracker, vad_cb, deepgram_vad_cb, familiar).
    """
    familiar = _make_familiar_mock(lull_timeout=lull_timeout)
    vc = _make_vc_mock()
    handler, tracker, vad_cb, deepgram_vad_cb = _build_voice_response_handler(
        vc=vc,
        familiar=familiar,
        voice_channel_id=100,
        guild_id=1,
        user_names={42: "Alice", 99: "Bob"},
    )
    return handler, tracker, vad_cb, deepgram_vad_cb, familiar


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

        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_vad_speech_blocks_lull_timer(self) -> None:
        """SpeechStarted cancels the lull timer; UtteranceEnd restarts it."""
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        # Simulate: user starts speaking, then final arrives.
        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Even though a result arrived, no timer should fire because
        # the user is still speaking (SpeechStarted, no UtteranceEnd).
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # Now the user stops (Discord + Deepgram).
        vad_cb(42, "UtteranceEnd")
        dg_cb(42, "UtteranceEnd")

        # Lull timer starts now.
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_vad_links_segments_across_brief_pause(self) -> None:
        """Segments separated by a brief pause are linked into one response.

        Simulates: user says "hello" → brief pause → "how are you",
        with Deepgram finalising "hello" during the pause but VAD
        showing continued speech.
        """
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        # Segment 1: user starts talking
        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Brief pause — Discord detects silence then speech again
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
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # User finishes (Discord + Deepgram)
        vad_cb(42, "UtteranceEnd")
        dg_cb(42, "UtteranceEnd")

        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)

        # Single generation with combined text
        familiar.llm_clients["main_prose"].chat.assert_called_once()
        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "hello how are you"

    @pytest.mark.asyncio
    async def test_multiple_speakers_wait_for_all_silent(self) -> None:
        """Lull timer only starts when ALL speakers stop."""
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        vad_cb(42, "SpeechStarted")
        vad_cb(99, "SpeechStarted")

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # User 42 stops, but user 99 is still talking.
        vad_cb(42, "UtteranceEnd")
        dg_cb(42, "UtteranceEnd")
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        r2 = TranscriptionResult(text="world", is_final=True, start=0.5, end=1.0)
        await handler(99, r2)

        # Now user 99 stops.
        vad_cb(99, "UtteranceEnd")
        dg_cb(99, "UtteranceEnd")
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracker_idle_during_debounce(self) -> None:
        """Tracker stays IDLE while results are being buffered."""
        handler, tracker, vad_cb, _, _ = _build(lull_timeout=0.1)

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
        handler, _, _vad_cb, _, familiar = _build(lull_timeout=0.05)

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
        handler, _, _vad_cb, _, familiar = _build(lull_timeout=0.08)

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Wait less than lull_timeout, then send another result.
        await asyncio.sleep(0.04)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        r2 = TranscriptionResult(text="second", is_final=True, start=0.5, end=1.0)
        await handler(42, r2)

        # Wait less than lull_timeout again — timer was reset by r2.
        await asyncio.sleep(0.04)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        r3 = TranscriptionResult(text="third", is_final=True, start=1.0, end=1.5)
        await handler(42, r3)

        # Now wait for the full lull to expire after the LAST result.
        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        # Single generation with all three segments.
        familiar.llm_clients["main_prose"].chat.assert_called_once()
        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "first second third"


class TestLullStateTransition:
    """Verify IDLE→GENERATING happens at lull expiry, not later."""

    @pytest.fixture(autouse=True)
    def _patch_chat(self) -> Iterator[None]:  # type: ignore[misc]
        """Patch assemble_chat_messages."""
        with patch(
            "familiar_connect.bot.assemble_chat_messages",
            return_value=[Message(role="user", content="hi")],
        ):
            yield

    @pytest.mark.asyncio
    async def test_tracker_generating_after_lull_expires(self) -> None:
        """Tracker transitions to GENERATING when the lull timer fires."""
        handler, tracker, _, _, familiar = _build(lull_timeout=0.05)

        # Slow down the LLM so we can observe GENERATING before it finishes.
        gate = asyncio.Event()
        original_chat = familiar.llm_clients["main_prose"].chat

        async def _slow_chat(*args: object, **kwargs: object) -> object:
            await gate.wait()
            return await original_chat(*args, **kwargs)

        familiar.llm_clients["main_prose"].chat = AsyncMock(side_effect=_slow_chat)

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Before lull: still IDLE.
        assert tracker.state is ResponseState.IDLE

        # Wait for lull to expire + a tick for the task to run.
        await asyncio.sleep(0.07)
        for _ in range(10):
            await asyncio.sleep(0)

        # Lull expired — tracker should be GENERATING now.
        assert tracker.state is ResponseState.GENERATING

        # Let the LLM finish.
        gate.set()
        await asyncio.sleep(0.05)
        for _ in range(10):
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_speech_during_generating_does_not_cancel_timer(self) -> None:
        """New speech after GENERATING does not cancel the generation task."""
        handler, tracker, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        # Slow down the LLM so we can observe GENERATING.
        gate = asyncio.Event()
        original_chat = familiar.llm_clients["main_prose"].chat

        async def _slow_chat(*args: object, **kwargs: object) -> object:
            await gate.wait()
            return await original_chat(*args, **kwargs)

        familiar.llm_clients["main_prose"].chat = AsyncMock(side_effect=_slow_chat)

        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)
        vad_cb(42, "UtteranceEnd")
        dg_cb(42, "UtteranceEnd")

        # Wait for lull to expire.
        await asyncio.sleep(0.07)
        for _ in range(10):
            await asyncio.sleep(0)

        assert tracker.state is ResponseState.GENERATING

        # New speech arrives — should NOT cancel the generation.
        vad_cb(42, "SpeechStarted")
        r2 = TranscriptionResult(text="world", is_final=True, start=1.0, end=1.5)
        await handler(42, r2)

        # Still GENERATING — not cancelled.
        assert tracker.state is ResponseState.GENERATING

        # Let the LLM finish.
        gate.set()
        await asyncio.sleep(0.1)
        for _ in range(10):
            await asyncio.sleep(0)

        # Generation completed, tracker should be IDLE again.
        assert tracker.state is ResponseState.IDLE

        # The new utterance should be in pending_utterances, ready
        # for the next cycle.
        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_pending_utterances_trigger_new_cycle(self) -> None:
        """Utterances arriving during generation trigger a new cycle after."""
        handler, tracker, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)
        vad_cb(42, "UtteranceEnd")
        dg_cb(42, "UtteranceEnd")

        # Wait for generation to complete.
        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        familiar.llm_clients["main_prose"].chat.assert_called_once()
        assert tracker.state is ResponseState.IDLE

        # Now send a second utterance — it should trigger a new cycle.
        r2 = TranscriptionResult(text="world", is_final=True, start=1.0, end=1.5)
        await handler(42, r2)

        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        assert familiar.llm_clients["main_prose"].chat.call_count == 2


class TestDeepgramFlushGate:
    """Verify generation waits for Deepgram to flush in-transit transcripts."""

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
    async def test_lull_waits_for_deepgram_flush(self) -> None:
        """After lull expires, generation waits for Deepgram UtteranceEnd."""
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        # User speaks (Discord audio detected).
        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # User stops (Discord silence) — lull timer starts.
        vad_cb(42, "UtteranceEnd")

        # Lull expires, but Deepgram hasn't confirmed flush yet.
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # Deepgram confirms all transcriptions flushed.
        dg_cb(42, "UtteranceEnd")
        await asyncio.sleep(0.05)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_gate_collects_late_transcript(self) -> None:
        """A transcript arriving during the flush wait is included."""
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        vad_cb(42, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)
        vad_cb(42, "UtteranceEnd")

        # Lull expires — waiting for Deepgram flush.
        await asyncio.sleep(0.10)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # Late transcript arrives (Deepgram was slow).
        r2 = TranscriptionResult(text="world", is_final=True, start=0.5, end=1.0)
        await handler(42, r2)

        # The late transcript resets the lull timer (cancels the
        # flush wait).  After it also finishes, both are included.
        dg_cb(42, "UtteranceEnd")
        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)

        familiar.llm_clients["main_prose"].chat.assert_called_once()
        pipeline_mock = familiar.build_pipeline.return_value
        request = pipeline_mock.assemble.call_args[0][0]
        assert request.utterance == "hello world"

    @pytest.mark.asyncio
    async def test_no_vad_events_skips_flush_gate(self) -> None:
        """Without SpeechStarted, deepgram_ready stays set — no wait."""
        handler, _, _vad_cb, _dg_cb, familiar = _build(lull_timeout=0.05)

        # No Discord VAD events — fallback timer only.
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        await asyncio.sleep(0.10)
        for _ in range(10):
            await asyncio.sleep(0)
        # Should generate without waiting for Deepgram flush.
        familiar.llm_clients["main_prose"].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_gate_per_user_tracking(self) -> None:
        """Deepgram flush is tracked per-user; all must confirm."""
        handler, _, vad_cb, dg_cb, familiar = _build(lull_timeout=0.05)

        vad_cb(42, "SpeechStarted")
        vad_cb(99, "SpeechStarted")
        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)
        r2 = TranscriptionResult(text="world", is_final=True, start=0.0, end=0.5)
        await handler(99, r2)

        vad_cb(42, "UtteranceEnd")
        vad_cb(99, "UtteranceEnd")

        # Lull fires — waiting for both users' Deepgram flush.
        await asyncio.sleep(0.10)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # Only user 42 confirmed — still waiting.
        dg_cb(42, "UtteranceEnd")
        await asyncio.sleep(0.05)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_not_called()

        # User 99 confirms — now generate.
        dg_cb(99, "UtteranceEnd")
        await asyncio.sleep(0.05)
        for _ in range(10):
            await asyncio.sleep(0)
        familiar.llm_clients["main_prose"].chat.assert_called_once()
