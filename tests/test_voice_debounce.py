"""Tests for voice transcription debounce in _build_voice_response_handler.

Verifies that multiple rapid voice transcription results are buffered
and only trigger a single LLM generation after a lull timeout.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
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
    async def test_single_result_generates_after_lull(self) -> None:
        """A single voice result triggers generation after lull_timeout."""
        familiar = _make_familiar_mock(lull_timeout=0.05)
        vc = _make_vc_mock()

        handler, tracker, _detector = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=100,
            guild_id=1,
            user_names={42: "Alice"},
        )

        result = TranscriptionResult(
            text="hello there", is_final=True, start=0.0, end=1.0
        )
        await handler(42, result)

        # Immediately after, tracker should still be IDLE (debouncing).
        assert tracker.state is ResponseState.IDLE

        # Wait for lull + generation to complete.
        await asyncio.sleep(0.15)
        # Allow tasks to finish.
        for _ in range(10):
            await asyncio.sleep(0)

        # LLM should have been called once.
        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_rapid_results_generate_once(self) -> None:
        """Multiple results within the lull window produce a single generation."""
        familiar = _make_familiar_mock(lull_timeout=0.05)
        vc = _make_vc_mock()

        handler, tracker, _detector = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=100,
            guild_id=1,
            user_names={42: "Alice"},
        )

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        r2 = TranscriptionResult(text="there", is_final=True, start=0.5, end=1.0)
        r3 = TranscriptionResult(text="friend", is_final=True, start=1.0, end=1.5)

        await handler(42, r1)
        await handler(42, r2)
        await handler(42, r3)

        # Still IDLE during debounce.
        assert tracker.state is ResponseState.IDLE

        # Wait for lull + generation.
        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        # Only ONE LLM call with combined text.
        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_combined_text_joins_all_utterances(self) -> None:
        """The LLM receives the combined text of all buffered utterances."""
        familiar = _make_familiar_mock(lull_timeout=0.05)
        vc = _make_vc_mock()

        handler, _tracker, _detector = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=100,
            guild_id=1,
            user_names={42: "Alice"},
        )

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        r2 = TranscriptionResult(text="how are you", is_final=True, start=0.5, end=1.5)

        await handler(42, r1)
        await handler(42, r2)

        await asyncio.sleep(0.15)
        for _ in range(10):
            await asyncio.sleep(0)

        # Check that the pipeline received "hello how are you" as the utterance.
        pipeline_mock = familiar.build_pipeline.return_value
        assemble_call = pipeline_mock.assemble.call_args
        request = assemble_call[0][0]
        assert request.utterance == "hello how are you"

    @pytest.mark.asyncio
    async def test_new_result_resets_lull_timer(self) -> None:
        """A new result arriving before lull expires resets the timer."""
        familiar = _make_familiar_mock(lull_timeout=0.08)
        vc = _make_vc_mock()

        handler, _tracker, _detector = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=100,
            guild_id=1,
            user_names={42: "Alice"},
        )

        r1 = TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5)
        await handler(42, r1)

        # Wait almost the full lull period, then add another result.
        await asyncio.sleep(0.06)
        familiar.llm_client.chat.assert_not_called()

        r2 = TranscriptionResult(text="world", is_final=True, start=0.5, end=1.0)
        await handler(42, r2)

        # Wait another 0.06s — still within the RESET lull window.
        await asyncio.sleep(0.06)
        familiar.llm_client.chat.assert_not_called()

        # Wait for the full lull to expire from the last result.
        await asyncio.sleep(0.05)
        for _ in range(10):
            await asyncio.sleep(0)

        familiar.llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracker_idle_during_debounce(self) -> None:
        """Tracker stays IDLE while results are being buffered."""
        familiar = _make_familiar_mock(lull_timeout=0.1)
        vc = _make_vc_mock()

        handler, tracker, _detector = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=100,
            guild_id=1,
            user_names={42: "Alice"},
        )

        r1 = TranscriptionResult(text="hi", is_final=True, start=0.0, end=0.3)
        await handler(42, r1)
        await asyncio.sleep(0.01)
        assert tracker.state is ResponseState.IDLE

        r2 = TranscriptionResult(text="there", is_final=True, start=0.3, end=0.6)
        await handler(42, r2)
        await asyncio.sleep(0.01)
        assert tracker.state is ResponseState.IDLE
