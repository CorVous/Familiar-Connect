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
    DISCORD_SILENCE_S,
    AudioActivityTracker,
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
        """response_handler is awaited for final transcriptions."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
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


# ---------------------------------------------------------------------------
# Discord audio activity tracker
# ---------------------------------------------------------------------------


class TestAudioActivityTracker:
    """Verify AudioActivityTracker fires events from Discord audio timing."""

    @pytest.mark.asyncio
    async def test_first_audio_fires_speech_started(self) -> None:
        """First audio chunk from a user fires SpeechStarted."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.1,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        try:
            tracker.on_audio(42)
            assert events == [(42, "SpeechStarted")]
        finally:
            tracker.cancel_all()

    @pytest.mark.asyncio
    async def test_continuous_audio_no_duplicate_speech_started(self) -> None:
        """Rapid audio chunks only fire SpeechStarted once."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.1,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        try:
            tracker.on_audio(42)
            tracker.on_audio(42)
            tracker.on_audio(42)
            assert events == [(42, "SpeechStarted")]
        finally:
            tracker.cancel_all()

    @pytest.mark.asyncio
    async def test_silence_fires_utterance_end(self) -> None:
        """After silence_s without audio, UtteranceEnd fires."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.05,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        tracker.on_audio(42)
        await asyncio.sleep(0.1)
        assert (42, "UtteranceEnd") in events

    @pytest.mark.asyncio
    async def test_audio_resets_silence_timer(self) -> None:
        """A new audio chunk resets the silence countdown."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.08,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        try:
            tracker.on_audio(42)
            await asyncio.sleep(0.04)
            # Timer not yet fired, send more audio to reset it
            tracker.on_audio(42)
            await asyncio.sleep(0.04)
            # Still within silence_s of the last audio — no UtteranceEnd yet
            assert (42, "UtteranceEnd") not in events
            # Wait for the full silence to expire
            await asyncio.sleep(0.1)
            assert (42, "UtteranceEnd") in events
        finally:
            tracker.cancel_all()

    @pytest.mark.asyncio
    async def test_resumed_speech_fires_new_speech_started(self) -> None:
        """After UtteranceEnd, new audio fires SpeechStarted again."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.05,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        tracker.on_audio(42)
        await asyncio.sleep(0.1)
        # Should have SpeechStarted + UtteranceEnd
        assert events == [(42, "SpeechStarted"), (42, "UtteranceEnd")]

        tracker.on_audio(42)
        assert events == [
            (42, "SpeechStarted"),
            (42, "UtteranceEnd"),
            (42, "SpeechStarted"),
        ]
        tracker.cancel_all()

    @pytest.mark.asyncio
    async def test_multiple_speakers_tracked_independently(self) -> None:
        """Each user gets independent SpeechStarted/UtteranceEnd."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.10,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        tracker.on_audio(42)
        tracker.on_audio(99)
        assert (42, "SpeechStarted") in events
        assert (99, "SpeechStarted") in events

        # User 42 goes silent, user 99 keeps talking
        await asyncio.sleep(0.05)
        tracker.on_audio(99)  # resets user 99's timer to now+0.10

        await asyncio.sleep(0.08)
        # At t≈0.13: user 42 silent since t=0 → UtteranceEnd (0+0.10=0.10 < 0.13)
        # user 99 reset at t=0.05 → fires at t=0.15 → still active
        assert (42, "UtteranceEnd") in events
        assert (99, "UtteranceEnd") not in events

        await asyncio.sleep(0.15)
        assert (99, "UtteranceEnd") in events
        tracker.cancel_all()

    @pytest.mark.asyncio
    async def test_cancel_all_prevents_utterance_end(self) -> None:
        """cancel_all prevents pending silence timers from firing."""
        events: list[tuple[int, str]] = []
        tracker = AudioActivityTracker(
            silence_s=0.05,
            callback=lambda uid, evt: events.append((uid, evt)),
        )
        tracker.on_audio(42)
        tracker.cancel_all()
        await asyncio.sleep(0.1)
        # Only SpeechStarted, no UtteranceEnd
        assert events == [(42, "SpeechStarted")]

    @pytest.mark.asyncio
    async def test_default_silence_matches_constant(self) -> None:
        """Default silence_s matches the module constant."""
        tracker = AudioActivityTracker(
            callback=lambda _uid, _evt: None,
        )
        assert tracker._silence_s == DISCORD_SILENCE_S
        tracker.cancel_all()
