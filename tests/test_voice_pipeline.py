"""Tests for the voice transcription pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from familiar_connect.transcription import (
    SpeechStartedEvent,
    TranscriptionEvent,
    TranscriptionResult,
    UtteranceEndEvent,
)
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
        transcriber.finalize = AsyncMock()

        await audio_queue.put(b"\x01")
        await audio_queue.put(b"\x02")

        task = asyncio.create_task(_audio_pump(audio_queue, transcriber))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert transcriber.send_audio.await_count == 2

    @pytest.mark.asyncio
    async def test_finalizes_after_audio_idle(self) -> None:
        """Pump sends Deepgram ``Finalize`` when no audio for the idle window.

        Discord's client-side VAD drops RTP during silence; without an
        explicit flush Deepgram holds the buffered final until next speech.
        """
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock()
        transcriber.finalize = AsyncMock()

        await audio_queue.put(b"\x00\x01")

        task = asyncio.create_task(
            _audio_pump(audio_queue, transcriber, idle_finalize_s=0.05)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        transcriber.finalize.assert_awaited()

    @pytest.mark.asyncio
    async def test_does_not_finalize_before_any_audio(self) -> None:
        """Pump never finalizes while it has not yet seen real audio."""
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock()
        transcriber.finalize = AsyncMock()

        task = asyncio.create_task(
            _audio_pump(audio_queue, transcriber, idle_finalize_s=0.05)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        transcriber.finalize.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_finalize_twice_in_a_row(self) -> None:
        """Once flushed, pump waits for new audio before finalizing again."""
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock()
        transcriber.finalize = AsyncMock()

        await audio_queue.put(b"\x01")

        task = asyncio.create_task(
            _audio_pump(audio_queue, transcriber, idle_finalize_s=0.05)
        )
        # Run for several idle windows.
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert transcriber.finalize.await_count == 1

    @pytest.mark.asyncio
    async def test_real_audio_after_finalize_re_arms_window(self) -> None:
        """Fresh audio after a finalize re-enters the dirty state."""
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcriber = MagicMock()
        transcriber.send_audio = AsyncMock()
        transcriber.finalize = AsyncMock()

        async def feeder() -> None:
            await audio_queue.put(b"\x01")
            # Wait long enough for first finalize to fire.
            await asyncio.sleep(0.15)
            await audio_queue.put(b"\x02")

        feeder_task = asyncio.create_task(feeder())
        pump_task = asyncio.create_task(
            _audio_pump(audio_queue, transcriber, idle_finalize_s=0.05)
        )
        await asyncio.sleep(0.4)
        pump_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pump_task
        await feeder_task

        # Finalize once after first chunk, then again after second chunk.
        assert transcriber.finalize.await_count == 2


class TestTranscriptForwarder:
    @pytest.mark.asyncio
    async def test_tags_results_with_user_id(self) -> None:
        """Forwarder reads from user queue and puts tagged tuple on shared queue."""
        user_queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()

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
    on_speech_start: object = None,
    on_speech_end: object = None,
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
        on_speech_start=on_speech_start,  # ty: ignore[invalid-argument-type]
        on_speech_end=on_speech_end,  # ty: ignore[invalid-argument-type]
    )


class TestTranscriptLogger:
    @pytest.mark.asyncio
    async def test_logs_final_with_user_name(self) -> None:
        """Final results are logged with the user's display name."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        """response_handler is awaited for final transcriptions."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
        handler = AsyncMock()
        pipeline = _make_pipeline_stub({42: "Alice"}, response_handler=handler)

        result = TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        handler.assert_awaited_once_with(42, result)

    @pytest.mark.asyncio
    async def test_response_handler_not_called_for_interim(self) -> None:
        """response_handler is NOT called for interim results."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
        handler = AsyncMock(side_effect=RuntimeError("llm failed"))
        pipeline = _make_pipeline_stub({42: "Alice"}, response_handler=handler)

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=1.0)
        r2 = TranscriptionResult(text="second", is_final=True, start=1.0, end=2.0)
        await shared_queue.put((42, r1))
        await shared_queue.put((42, r2))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Both were attempted despite the first failing
        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_logger_not_blocked_by_slow_handler(self) -> None:
        """Logger keeps logging while the response handler is still running."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()

        # Handler that blocks for a long time
        handler_entered = asyncio.Event()
        handler_release = asyncio.Event()

        async def _slow_handler(
            user_id: int,  # noqa: ARG001
            result: TranscriptionResult,  # noqa: ARG001
        ) -> None:
            handler_entered.set()
            await handler_release.wait()

        pipeline = _make_pipeline_stub({42: "Alice"}, response_handler=_slow_handler)

        r1 = TranscriptionResult(text="first", is_final=True, start=0.0, end=1.0)
        r2 = TranscriptionResult(text="second", is_final=True, start=1.0, end=2.0)
        await shared_queue.put((42, r1))
        await shared_queue.put((42, r2))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))

            # Wait for handler to start processing r1
            await asyncio.wait_for(handler_entered.wait(), timeout=1.0)
            # Give logger a moment to process r2 while handler is blocked
            await asyncio.sleep(0.05)

            # BOTH transcriptions should be logged even though handler is stuck
            info_calls = [
                c
                for c in mock_logger.info.call_args_list
                if c[0][0] == "[Transcription] %s: %s"
            ]
            assert len(info_calls) == 2

            handler_release.set()
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestTranscriptLoggerVADEvents:
    @pytest.mark.asyncio
    async def test_speech_started_invokes_on_speech_start(self) -> None:
        """SpeechStartedEvent from Deepgram routes to pipeline.on_speech_start."""
        calls: list[int] = []

        def _on_start(user_id: int) -> None:
            calls.append(user_id)

        pipeline = _make_pipeline_stub({42: "Alice"}, on_speech_start=_on_start)
        shared_queue = pipeline.shared_transcript_queue

        await shared_queue.put((42, SpeechStartedEvent(timestamp=0.12)))

        task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert calls == [42]

    @pytest.mark.asyncio
    async def test_utterance_end_invokes_on_speech_end(self) -> None:
        """UtteranceEndEvent from Deepgram routes to pipeline.on_speech_end."""
        calls: list[int] = []

        def _on_end(user_id: int) -> None:
            calls.append(user_id)

        pipeline = _make_pipeline_stub({42: "Alice"}, on_speech_end=_on_end)
        shared_queue = pipeline.shared_transcript_queue

        await shared_queue.put((42, UtteranceEndEvent(last_word_end=1.0)))

        task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert calls == [42]

    @pytest.mark.asyncio
    async def test_vad_hooks_optional(self) -> None:
        """VAD events are a no-op when hooks are not wired."""
        pipeline = _make_pipeline_stub({42: "Alice"})
        shared_queue = pipeline.shared_transcript_queue

        await shared_queue.put((42, SpeechStartedEvent(timestamp=0.0)))
        await shared_queue.put((42, UtteranceEndEvent(last_word_end=1.0)))

        task = asyncio.create_task(_transcript_logger(shared_queue, pipeline))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # must not raise


class TestTranscriptForwarderVADEvents:
    @pytest.mark.asyncio
    async def test_forwards_speech_events_tagged_with_user_id(self) -> None:
        """Non-result events flow through the forwarder with user tagging."""
        user_queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()

        await user_queue.put(SpeechStartedEvent(timestamp=0.0))
        await user_queue.put(UtteranceEndEvent(last_word_end=1.0))

        task = asyncio.create_task(_transcript_forwarder(7, user_queue, shared_queue))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        first = shared_queue.get_nowait()
        second = shared_queue.get_nowait()
        assert first[0] == 7
        assert isinstance(first[1], SpeechStartedEvent)
        assert second[0] == 7
        assert isinstance(second[1], UtteranceEndEvent)


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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
        shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = asyncio.Queue()
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
