"""Tests for the voice transcription pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice_pipeline import (
    PipelineError,
    VoicePipeline,
    _audio_pump,
    _audio_router,
    _transcript_forwarder,
    _transcript_logger,
    clear_pipeline,
    get_pipeline,
    set_pipeline,
    start_pipeline,
    stop_pipeline,
)


@pytest.fixture(autouse=True)
def _clean_pipeline():
    """Reset pipeline state between tests."""
    clear_pipeline()
    yield
    clear_pipeline()


# ---------------------------------------------------------------------------
# Phase 3: Registry
# ---------------------------------------------------------------------------


class TestPipelineRegistry:
    def test_get_pipeline_returns_none_initially(self) -> None:
        """No pipeline is active by default."""
        assert get_pipeline() is None

    def test_set_pipeline_stores_pipeline(self) -> None:
        """set_pipeline makes the pipeline retrievable via get_pipeline."""
        pipeline = MagicMock(spec=VoicePipeline)
        set_pipeline(pipeline)
        assert get_pipeline() is pipeline

    def test_set_pipeline_raises_when_already_set(self) -> None:
        """Setting a pipeline when one is already active raises PipelineError."""
        set_pipeline(MagicMock(spec=VoicePipeline))
        with pytest.raises(PipelineError):
            set_pipeline(MagicMock(spec=VoicePipeline))

    def test_clear_pipeline_resets_to_none(self) -> None:
        """Clearing an active pipeline resets to None."""
        set_pipeline(MagicMock(spec=VoicePipeline))
        clear_pipeline()
        assert get_pipeline() is None

    def test_clear_pipeline_is_idempotent(self) -> None:
        """Clearing when no pipeline is active does not raise."""
        clear_pipeline()
        clear_pipeline()


# ---------------------------------------------------------------------------
# Phase 4: Audio pump + transcript forwarder
# ---------------------------------------------------------------------------


class TestAudioPump:
    @pytest.mark.asyncio
    async def test_sends_bytes_to_transcriber(self) -> None:
        """Audio pump reads from the queue and calls send_audio."""
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock()

        await audio_queue.put(b"\x00\x01\x02\x03")

        task = asyncio.create_task(_audio_pump(audio_queue, transcriber))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        transcriber.send_audio.assert_awaited_once_with(b"\x00\x01\x02\x03")

    @pytest.mark.asyncio
    async def test_continues_on_send_error(self) -> None:
        """Audio pump logs and continues when send_audio raises."""
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock(side_effect=[RuntimeError("oops"), None])

        await audio_queue.put(b"\x01")
        await audio_queue.put(b"\x02")

        task = asyncio.create_task(_audio_pump(audio_queue, transcriber))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert transcriber.send_audio.await_count == 2


class TestTranscriptForwarder:
    @pytest.mark.asyncio
    async def test_tags_results_with_user_id(self) -> None:
        """Forwarder reads from user queue and puts tagged tuple on shared queue."""
        user_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()

        result = TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0)
        await user_queue.put(result)

        task = asyncio.create_task(_transcript_forwarder(42, user_queue, shared_queue))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not shared_queue.empty()
        user_id, tagged_result = shared_queue.get_nowait()
        assert user_id == 42
        assert tagged_result is result


# ---------------------------------------------------------------------------
# Phase 5: Transcript logger
# ---------------------------------------------------------------------------


def _make_pipeline_stub(
    user_names: dict[int, str],
    resolve_name: Callable[[int], str | None] | None = None,
    response_handler: object = None,
    lull_timeout: float = 0.8,
) -> VoicePipeline:
    """Create a minimal VoicePipeline for transcript logger tests."""
    return VoicePipeline(
        template=MagicMock(),
        tagged_audio_queue=asyncio.Queue(),
        shared_transcript_queue=asyncio.Queue(),
        router_task=None,
        logger_task=None,
        user_names=user_names,
        resolve_name=resolve_name,
        response_handler=response_handler,  # ty: ignore[invalid-argument-type]
        lull_timeout=lull_timeout,
    )


class TestTranscriptLogger:
    @pytest.mark.asyncio
    async def test_logs_final_with_user_name(self) -> None:
        """Final results are logged with the user's display name."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({42: "Alice"})

        result = TranscriptionResult(
            text="hello world", is_final=True, start=0.0, end=1.0
        )
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            mock_logger.info.assert_called_once_with(
                "[Transcription] %s: %s", "Alice", "hello world"
            )

    @pytest.mark.asyncio
    async def test_logs_interim_at_debug(self) -> None:
        """Interim results are logged at debug level."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({42: "Alice"})

        result = TranscriptionResult(text="hel", is_final=False, start=0.0, end=0.3)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            mock_logger.debug.assert_called_once_with(
                "[Transcription interim] %s: %s", "Alice", "hel"
            )
            mock_logger.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_user_id_for_unknown(self) -> None:
        """Unknown user IDs fall back to 'User-<id>' format."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({})

        result = TranscriptionResult(text="hi", is_final=True, start=0.0, end=0.5)
        await shared_queue.put((99999, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            mock_logger.info.assert_called_once_with(
                "[Transcription] %s: %s", "User-99999", "hi"
            )

    @pytest.mark.asyncio
    async def test_resolve_name_callback_for_late_joiner(self) -> None:
        """resolve_name callback resolves unknown users and caches the result."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub(
            {}, resolve_name=lambda uid: "Charlie" if uid == 77 else None
        )

        result = TranscriptionResult(text="hey", is_final=True, start=0.0, end=0.5)
        await shared_queue.put((77, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            mock_logger.info.assert_called_once_with(
                "[Transcription] %s: %s", "Charlie", "hey"
            )
        # Cached for future lookups
        assert pipeline.user_names[77] == "Charlie"

    @pytest.mark.asyncio
    async def test_response_handler_called_for_final(self) -> None:
        """response_handler is awaited after the lull timer fires."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_awaited_once()
        call_user_id, call_result = handler.call_args[0]
        assert call_user_id == 42
        assert call_result.text == "hello"

    @pytest.mark.asyncio
    async def test_response_handler_not_called_for_interim(self) -> None:
        """response_handler is NOT called for interim results."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub({42: "Alice"}, response_handler=handler)

        result = TranscriptionResult(text="hel", is_final=False, start=0.0, end=0.3)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_response_handler_error_does_not_crash(self) -> None:
        """An error in response_handler is logged but doesn't kill the logger."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock(side_effect=RuntimeError("llm failed"))
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.05
        )

        # Both results arrive before the lull fires — they are collated into one call.
        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=1.0)
        r2 = TranscriptionResult(text="second", is_final=True, start=1.0, end=2.0)
        await shared_queue.put((42, r1))
        await shared_queue.put((42, r2))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Collated into one call; handler failing doesn't crash the logger.
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_logger_not_blocked_by_slow_handler(self) -> None:
        """Logger logs both results immediately; handler is called once (collated)."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()

        # Handler that blocks for a long time
        handler_entered = asyncio.Event()
        handler_release = asyncio.Event()

        async def _slow_handler(
            user_id: int,  # noqa: ARG001
            result: TranscriptionResult,  # noqa: ARG001
        ) -> None:
            handler_entered.set()
            await handler_release.wait()

        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=_slow_handler, lull_timeout=0.05
        )

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=1.0)
        r2 = TranscriptionResult(text="second", is_final=True, start=1.0, end=2.0)
        await shared_queue.put((42, r1))
        await shared_queue.put((42, r2))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))

            # Both results are logged immediately regardless of the timer.
            await asyncio.sleep(0.02)
            info_calls = [
                c
                for c in mock_logger.info.call_args_list
                if c[0][0] == "[Transcription] %s: %s"
            ]
            assert len(info_calls) == 2

            # After the lull fires, the merged handler call begins.
            await asyncio.wait_for(handler_entered.wait(), timeout=1.0)

            handler_release.set()
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    # ------------------------------------------------------------------
    # Collation / lull-timer tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_response_handler_not_called_immediately_on_final(self) -> None:
        """Handler is NOT called right away — lull timer must expire first."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.5
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="hi", is_final=True, start=0.0, end=0.5),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)  # Well within lull_timeout=0.5
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lull_timer_fires_response_after_silence(self) -> None:
        """Single is_final result → one handler call after lull_timeout elapses."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_awaited_once()
        _, call_result = handler.call_args[0]
        assert call_result.text == "hello"

    @pytest.mark.asyncio
    async def test_multiple_finals_within_lull_collated_into_one_call(self) -> None:
        """Three rapid is_final results → one handler call with merged text."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="Hello,", is_final=True, start=0.0, end=0.5),
        ))
        await shared_queue.put((
            42,
            TranscriptionResult(text="how are", is_final=True, start=0.5, end=1.0),
        ))
        await shared_queue.put((
            42,
            TranscriptionResult(text="you doing?", is_final=True, start=1.0, end=1.8),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_awaited_once()
        _, call_result = handler.call_args[0]
        assert call_result.text == "Hello, how are you doing?"

    @pytest.mark.asyncio
    async def test_lull_resets_on_each_final(self) -> None:
        """A second is_final arriving within the lull window resets the timer."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.06
        )

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))

            # First result — starts the 60ms lull timer.
            await shared_queue.put((
                42,
                TranscriptionResult(text="one", is_final=True, start=0.0, end=0.5),
            ))
            # Wait 40ms (within the 60ms lull), then send the second result.
            await asyncio.sleep(0.04)
            await shared_queue.put((
                42,
                TranscriptionResult(text="two", is_final=True, start=0.5, end=1.0),
            ))
            # 40ms later — the reset timer has NOT fired yet.
            await asyncio.sleep(0.04)
            assert handler.await_count == 0

            # Wait past the reset lull (another 40ms).
            await asyncio.sleep(0.04)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_awaited_once()
        _, call_result = handler.call_args[0]
        assert call_result.text == "one two"

    @pytest.mark.asyncio
    async def test_different_users_have_independent_lull_timers(self) -> None:
        """Two users each get their own lull timer and separate handler calls."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice", 99: "Bob"}, response_handler=handler, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="hello", is_final=True, start=0.0, end=0.5),
        ))
        await shared_queue.put((
            99,
            TranscriptionResult(text="hi there", is_final=True, start=0.0, end=0.7),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert handler.await_count == 2
        user_ids = {call[0][0] for call in handler.call_args_list}
        assert user_ids == {42, 99}

    @pytest.mark.asyncio
    async def test_merged_result_fields(self) -> None:
        """Merged TranscriptionResult has correct start, end, confidence, and text."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=handler, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(
                text="part one", is_final=True, start=1.0, end=2.0, confidence=0.9
            ),
        ))
        await shared_queue.put((
            42,
            TranscriptionResult(
                text="part two", is_final=True, start=2.0, end=3.5, confidence=0.95
            ),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _, merged = handler.call_args[0]
        assert merged.text == "part one part two"
        assert merged.start == pytest.approx(1.0)
        assert merged.end == pytest.approx(3.5)
        assert merged.confidence == pytest.approx(0.95)
        assert merged.is_final is True

    @pytest.mark.asyncio
    async def test_no_call_when_no_response_handler(self) -> None:
        """No crash when response_handler is None and lull fires."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub(
            {42: "Alice"}, response_handler=None, lull_timeout=0.05
        )

        await shared_queue.put((
            42,
            TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0),
        ))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.12)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # No assertion needed — just verifying no exception was raised.


# ---------------------------------------------------------------------------
# Phase 6: Audio router
# ---------------------------------------------------------------------------


def _make_template() -> MagicMock:
    """Create a mock DeepgramTranscriber template with clone support."""
    template = MagicMock()

    def _clone() -> MagicMock:
        clone = MagicMock()
        clone.start = AsyncMock()
        clone.stop = AsyncMock()
        clone.send_audio = AsyncMock()
        return clone

    template.clone = MagicMock(side_effect=_clone)
    return template


class TestAudioRouter:
    @pytest.mark.asyncio
    async def test_creates_stream_for_new_user(self) -> None:
        """Router creates a new user stream when it sees a new user_id."""
        tagged_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        template = _make_template()
        pipeline = VoicePipeline(
            template=template,
            tagged_audio_queue=tagged_queue,
            shared_transcript_queue=shared_queue,
            router_task=MagicMock(),
            logger_task=MagicMock(),
            user_names={},
        )

        await tagged_queue.put((42, b"\x00\x01"))

        task = asyncio.create_task(_audio_router(tagged_queue, pipeline))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert 42 in pipeline.streams
        template.clone.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_existing_stream(self) -> None:
        """Router puts audio on the existing stream's queue for known users."""
        tagged_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        template = _make_template()
        pipeline = VoicePipeline(
            template=template,
            tagged_audio_queue=tagged_queue,
            shared_transcript_queue=shared_queue,
            router_task=MagicMock(),
            logger_task=MagicMock(),
            user_names={},
        )

        await tagged_queue.put((42, b"\x01"))
        await tagged_queue.put((42, b"\x02"))

        task = asyncio.create_task(_audio_router(tagged_queue, pipeline))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Only one stream created, clone called once
        template.clone.assert_called_once()
        assert 42 in pipeline.streams

    @pytest.mark.asyncio
    async def test_creates_separate_streams_per_user(self) -> None:
        """Different user_ids get different streams."""
        tagged_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        template = _make_template()
        pipeline = VoicePipeline(
            template=template,
            tagged_audio_queue=tagged_queue,
            shared_transcript_queue=shared_queue,
            router_task=MagicMock(),
            logger_task=MagicMock(),
            user_names={},
        )

        await tagged_queue.put((42, b"\x01"))
        await tagged_queue.put((99, b"\x02"))

        task = asyncio.create_task(_audio_router(tagged_queue, pipeline))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert 42 in pipeline.streams
        assert 99 in pipeline.streams
        assert template.clone.call_count == 2

    @pytest.mark.asyncio
    async def test_audio_router_cancels_lull_on_speaking_start(self) -> None:
        """Router cancels a pending lull handle when speech resumes after silence."""
        tagged_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        template = _make_template()
        pipeline = VoicePipeline(
            template=template,
            tagged_audio_queue=tagged_queue,
            shared_transcript_queue=shared_queue,
            router_task=MagicMock(),
            logger_task=MagicMock(),
            user_names={},
        )

        # Inject a fake pending lull handle for user 42.
        fake_handle = MagicMock()
        pipeline.lull_handles[42] = fake_handle

        # Send audio after a gap longer than the silence threshold.
        # The router starts with no prior audio time, so the initial gap is huge.
        await tagged_queue.put((42, b"\x01"))

        task = asyncio.create_task(_audio_router(tagged_queue, pipeline))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        fake_handle.cancel.assert_called_once()
        assert pipeline.lull_handles.get(42) is None

    @pytest.mark.asyncio
    async def test_audio_router_does_not_cancel_lull_for_continuous_audio(self) -> None:
        """Continuous audio (gap < threshold) does not cancel an absent lull handle."""
        tagged_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        template = _make_template()
        pipeline = VoicePipeline(
            template=template,
            tagged_audio_queue=tagged_queue,
            shared_transcript_queue=shared_queue,
            router_task=MagicMock(),
            logger_task=MagicMock(),
            user_names={},
        )

        # First packet — establishes last_audio_time.
        await tagged_queue.put((42, b"\x01"))
        task = asyncio.create_task(_audio_router(tagged_queue, pipeline))
        await asyncio.sleep(0.02)

        # Inject handle AFTER first packet has been processed.
        fake_handle = MagicMock()
        pipeline.lull_handles[42] = fake_handle

        # Second packet arrives immediately (gap << threshold).
        await tagged_queue.put((42, b"\x02"))
        await asyncio.sleep(0.02)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The gap was tiny — handle should NOT have been cancelled.
        fake_handle.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 7: start_pipeline / stop_pipeline
# ---------------------------------------------------------------------------


class TestStartPipeline:
    @pytest.mark.asyncio
    async def test_sets_registry(self) -> None:
        """start_pipeline registers the pipeline in the module."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        try:
            assert get_pipeline() is pipeline
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_spawns_router_and_logger(self) -> None:
        """start_pipeline creates router and logger tasks."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        try:
            assert pipeline.router_task is not None
            assert not pipeline.router_task.done()
            assert pipeline.logger_task is not None
            assert not pipeline.logger_task.done()
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_passes_response_handler_to_pipeline(self) -> None:
        """start_pipeline stores the response_handler on the pipeline."""
        template = _make_template()
        handler = AsyncMock()
        pipeline = await start_pipeline(template, {}, response_handler=handler)
        try:
            assert pipeline.response_handler is handler
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_raises_when_already_active(self) -> None:
        """Starting a pipeline when one is already active raises PipelineError."""
        template = _make_template()
        await start_pipeline(template, {})
        try:
            with pytest.raises(PipelineError):
                await start_pipeline(template, {})
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_lull_timeout_stored_on_pipeline(self) -> None:
        """start_pipeline stores a custom lull_timeout on the pipeline."""
        template = _make_template()
        pipeline = await start_pipeline(template, {}, lull_timeout=1.5)
        try:
            assert pipeline.lull_timeout == pytest.approx(1.5)
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_lull_timeout_default(self) -> None:
        """start_pipeline defaults lull_timeout to 0.8 seconds."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        try:
            assert pipeline.lull_timeout == pytest.approx(0.8)
        finally:
            await stop_pipeline()


class TestStopPipeline:
    @pytest.mark.asyncio
    async def test_stops_all_user_transcribers(self) -> None:
        """stop_pipeline stops every per-user transcriber."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})

        # Simulate two users by pushing tagged audio through the router
        await pipeline.tagged_audio_queue.put((42, b"\x01"))
        await pipeline.tagged_audio_queue.put((99, b"\x02"))
        await asyncio.sleep(0.1)

        await stop_pipeline()

        for stream in pipeline.streams.values():
            stop_mock: AsyncMock = getattr(stream.transcriber, "stop")  # noqa: B009
            stop_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancels_all_tasks(self) -> None:
        """stop_pipeline cancels router, logger, and per-user tasks."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        router = pipeline.router_task
        logger = pipeline.logger_task

        await stop_pipeline()

        assert router is not None
        assert logger is not None
        assert router.done()
        assert logger.done()

    @pytest.mark.asyncio
    async def test_clears_registry(self) -> None:
        """stop_pipeline clears the module registry."""
        template = _make_template()
        await start_pipeline(template, {})
        await stop_pipeline()
        assert get_pipeline() is None

    @pytest.mark.asyncio
    async def test_is_idempotent(self) -> None:
        """Calling stop_pipeline when no pipeline is active does not raise."""
        await stop_pipeline()
        await stop_pipeline()
