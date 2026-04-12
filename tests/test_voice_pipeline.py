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
    _LullCollator,
    _speaking_monitor,
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
    lull_timeout: float = 9999.0,
) -> VoicePipeline:
    """Create a minimal VoicePipeline for transcript logger tests.

    ``lull_timeout`` defaults to a very large value so lull timers never fire
    during tests unless explicitly controlled.
    """
    return VoicePipeline(
        template=MagicMock(),
        tagged_audio_queue=asyncio.Queue(),
        shared_transcript_queue=asyncio.Queue(),
        router_task=None,
        logger_task=None,
        speaking_monitor_task=None,
        user_names=user_names,
        resolve_name=resolve_name,
        response_handler=response_handler,  # ty: ignore[invalid-argument-type]
        lull_timeout=lull_timeout,
    )


class TestTranscriptLogger:
    """Tests for _transcript_logger.

    The logger now delegates response dispatch to a _LullCollator.  These
    tests pass a MagicMock collator so they can verify the logger calls
    the right collator methods without worrying about timer mechanics.
    """

    def _make_mock_collator(self) -> MagicMock:
        collator = MagicMock(spec=_LullCollator)
        collator.on_final_transcript = MagicMock()
        return collator

    @pytest.mark.asyncio
    async def test_logs_final_with_user_name(self) -> None:
        """Final results are logged with the user's display name."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({42: "Alice"})
        collator = self._make_mock_collator()

        result = TranscriptionResult(
            text="hello world", is_final=True, start=0.0, end=1.0
        )
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
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
        collator = self._make_mock_collator()

        result = TranscriptionResult(text="hel", is_final=False, start=0.0, end=0.3)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
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
        collator = self._make_mock_collator()

        result = TranscriptionResult(text="hi", is_final=True, start=0.0, end=0.5)
        await shared_queue.put((99999, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
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
        collator = self._make_mock_collator()

        result = TranscriptionResult(text="hey", is_final=True, start=0.0, end=0.5)
        await shared_queue.put((77, result))

        with patch("familiar_connect.voice_pipeline._logger") as mock_logger:
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
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
    async def test_final_delegates_to_collator(self) -> None:
        """Final results are forwarded to the collator, not dispatched directly."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({42: "Alice"})
        collator = self._make_mock_collator()

        result = TranscriptionResult(text="hello", is_final=True, start=0.0, end=1.0)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        collator.on_final_transcript.assert_called_once_with(42, result)

    @pytest.mark.asyncio
    async def test_interim_not_forwarded_to_collator(self) -> None:
        """Interim results are NOT forwarded to the collator."""
        shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = asyncio.Queue()
        pipeline = _make_pipeline_stub({42: "Alice"})
        collator = self._make_mock_collator()

        result = TranscriptionResult(text="hel", is_final=False, start=0.0, end=0.3)
        await shared_queue.put((42, result))

        with patch("familiar_connect.voice_pipeline._logger"):
            task = asyncio.create_task(
                _transcript_logger(shared_queue, pipeline, collator)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        collator.on_final_transcript.assert_not_called()


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
    async def test_spawns_speaking_monitor(self) -> None:
        """start_pipeline creates a speaking_monitor_task."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        try:
            assert pipeline.speaking_monitor_task is not None
            assert not pipeline.speaking_monitor_task.done()
        finally:
            await stop_pipeline()

    @pytest.mark.asyncio
    async def test_stores_lull_timeout(self) -> None:
        """start_pipeline stores lull_timeout on the pipeline."""
        template = _make_template()
        pipeline = await start_pipeline(template, {}, lull_timeout=3.5)
        try:
            assert pipeline.lull_timeout == pytest.approx(3.5)
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
        """stop_pipeline cancels router, logger, speaking_monitor, and user tasks."""
        template = _make_template()
        pipeline = await start_pipeline(template, {})
        router = pipeline.router_task
        logger = pipeline.logger_task
        speaking_monitor = pipeline.speaking_monitor_task

        await stop_pipeline()

        assert router is not None
        assert logger is not None
        assert speaking_monitor is not None
        assert router.done()
        assert logger.done()
        assert speaking_monitor.done()

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
# Phase 8: _LullCollator
# ---------------------------------------------------------------------------


def _make_result(
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    confidence: float = 0.9,
    speaker: int | None = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        is_final=True,
        start=start,
        end=end,
        confidence=confidence,
        speaker=speaker,
    )


class TestLullCollator:
    """Unit tests for _LullCollator.

    Timer constants are kept very small (0.01 s) and asyncio.sleep(0.15) gives
    the event loop plenty of cycles to run call_later callbacks and any tasks
    spawned by _dispatch.  Every collator is created with dispatch_grace=0.01
    so tests don't have to sleep long.
    """

    TIMEOUT = 0.01  # lull timer
    GRACE = 0.01  # dispatch window
    WAIT = 0.15  # sleep long enough for both timers + task execution

    def _collator(self, handler: object = None) -> _LullCollator:
        return _LullCollator(
            lull_timeout=self.TIMEOUT,
            response_handler=handler,  # ty: ignore[invalid-argument-type]
            dispatch_grace=self.GRACE,
        )

    # ------------------------------------------------------------------
    # Basic SPEAKING → lull → dispatch flow
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_final_transcript_starts_lull_that_dispatches(self) -> None:
        """is_final → lull → dispatch window → handler called once."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator.on_final_transcript(42, _make_result("hello"))

        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        combined: TranscriptionResult = handler.call_args[0][1]
        assert combined.text == "hello"
        assert combined.is_final is True

    @pytest.mark.asyncio
    async def test_on_speaking_true_cancels_all_timers(self) -> None:
        """SPEAKING=True cancels both lull and dispatch-window timers."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator.on_final_transcript(42, _make_result("hello"))  # start lull
        collator.on_speaking(42, is_speaking=True)  # cancel everything

        await asyncio.sleep(self.WAIT)

        handler.assert_not_awaited()

    # ------------------------------------------------------------------
    # Collation: multiple transcripts → one response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_transcripts_before_lull_collated(self) -> None:
        """Transcripts buffered before lull fires are combined into one call."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator.on_final_transcript(42, _make_result("Hello"))
        collator.on_final_transcript(42, _make_result("world"))

        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcripts_during_dispatch_window_all_collated(self) -> None:
        """Transcripts arriving while the dispatch window is open are collected.

        This covers the race where Deepgram sends is_final results just after
        the lull timer fires but before the dispatch window closes.
        """
        handler = AsyncMock()
        collator = self._collator(handler)

        # Open the dispatch window directly (simulates lull firing with no text)
        collator._on_lull(42)

        # Two transcripts arrive during the open window
        collator.on_final_transcript(42, _make_result("Hello"))
        collator.on_final_transcript(42, _make_result("world"))

        await asyncio.sleep(self.WAIT)

        # Exactly ONE call with both texts joined
        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "Hello world"

    @pytest.mark.asyncio
    async def test_dispatch_fires_only_once_not_per_transcript(self) -> None:
        """Handler is called exactly once regardless of how many transcripts arrive."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator._on_lull(42)  # open dispatch window

        collator.on_final_transcript(42, _make_result("one"))
        collator.on_final_transcript(42, _make_result("two"))
        collator.on_final_transcript(42, _make_result("three"))

        await asyncio.sleep(self.WAIT)

        assert handler.await_count == 1
        assert handler.call_args[0][1].text == "one two three"

    # ------------------------------------------------------------------
    # Late / orphan transcripts
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_transcript_after_both_windows_closed_still_dispatched(self) -> None:
        """Transcript arriving after both windows closed reopens the dispatch window.

        Covers the fallback path in on_final_transcript: if an is_final arrives
        with no timers running it starts a fresh lull countdown rather than being
        silently dropped.
        """
        handler = AsyncMock()
        collator = self._collator(handler)

        # Open the dispatch window directly with no buffered text, then let it close.
        collator._on_lull(42)
        await asyncio.sleep(self.WAIT)  # dispatch window expires, nothing dispatched
        handler.assert_not_awaited()

        # Late transcript arrives — starts a new lull countdown
        collator.on_final_transcript(42, _make_result("late text"))
        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "late text"

    @pytest.mark.asyncio
    async def test_multiple_late_transcripts_still_collated_into_one(self) -> None:
        """Multiple transcripts arriving with no active timer are collated into one."""
        handler = AsyncMock()
        collator = self._collator(handler)

        # Two transcripts arrive in quick succession with no prior state.
        # The first starts the lull timer; the second resets it — both are
        # in the buffer when the timer eventually fires.
        collator.on_final_transcript(42, _make_result("late one"))
        collator.on_final_transcript(42, _make_result("late two"))
        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "late one late two"

    # ------------------------------------------------------------------
    # Timer interaction
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_speaking_true_also_cancels_dispatch_window(self) -> None:
        """SPEAKING=True cancels the dispatch window — no response for buffered text."""
        handler = AsyncMock()
        # Use a longer dispatch window so we can cancel it in time
        collator = _LullCollator(
            lull_timeout=self.TIMEOUT,
            response_handler=handler,
            dispatch_grace=self.WAIT * 2,
        )

        collator._on_lull(42)  # open dispatch window
        collator.on_final_transcript(42, _make_result("hi"))
        collator.on_speaking(42, is_speaking=True)  # cancel dispatch window

        await asyncio.sleep(self.WAIT)

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_speaking_true_then_new_transcript_restarts_chain(self) -> None:
        """SPEAKING=True stops timers; next is_final starts a fresh lull + window."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator.on_final_transcript(42, _make_result("first"))
        collator.on_speaking(42, is_speaking=True)  # cancel (user resumed)
        collator.on_final_transcript(42, _make_result("second"))

        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "first second"

    @pytest.mark.asyncio
    async def test_different_users_collated_independently(self) -> None:
        """Each user has their own independent timer and buffer."""
        handler = AsyncMock()
        collator = self._collator(handler)

        collator.on_final_transcript(1, _make_result("Alice says hello"))
        collator.on_final_transcript(2, _make_result("Bob says hi"))

        await asyncio.sleep(self.WAIT)

        assert handler.await_count == 2
        calls = {c[0][0]: c[0][1].text for c in handler.call_args_list}
        assert calls[1] == "Alice says hello"
        assert calls[2] == "Bob says hi"

    @pytest.mark.asyncio
    async def test_no_response_when_handler_is_none(self) -> None:
        """No error when response_handler is None — collator is a no-op."""
        collator = _LullCollator(
            lull_timeout=self.TIMEOUT,
            response_handler=None,
            dispatch_grace=self.GRACE,
        )
        collator.on_final_transcript(42, _make_result("ignored"))
        await asyncio.sleep(self.WAIT)  # should complete without error

    # ------------------------------------------------------------------
    # Regression: continuous speech must collate into one dispatch.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_continuous_speech_collated_into_one_dispatch(self) -> None:
        """All transcripts from a continuous speech session dispatch as one.

        Regression: Discord sends SPEAKING=False during natural mid-sentence
        pauses and SPEAKING=True when speech resumes.  After SPEAKING=True
        cancels all timers, a subsequent is_final used to trigger a fallback
        dispatch window and fire mid-utterance.  The fix drives the lull
        timer from is_final events so the timer resets on every transcript
        and can only fire after lull_timeout of transcript silence.
        """
        handler = AsyncMock()
        collator = self._collator(handler)

        # Simulate continuous speech with a Discord mid-sentence pause
        collator.on_speaking(42, is_speaking=True)
        collator.on_final_transcript(42, _make_result("first half"))
        collator.on_speaking(42, is_speaking=False)  # brief Discord pause (no-op)
        collator.on_speaking(42, is_speaking=True)  # user resumed → cancel lull
        collator.on_final_transcript(42, _make_result("second half"))

        # Both halves collated; exactly one dispatch fires after lull
        await asyncio.sleep(self.WAIT)

        handler.assert_awaited_once()
        assert handler.call_args[0][1].text == "first half second half"


# ---------------------------------------------------------------------------
# Phase 9: _speaking_monitor
# ---------------------------------------------------------------------------


class TestSpeakingMonitor:
    @pytest.mark.asyncio
    async def test_routes_speaking_true_to_collator(self) -> None:
        """speaking_monitor calls collator.on_speaking for each event."""
        speaking_queue: asyncio.Queue[tuple[int, bool]] = asyncio.Queue()
        collator = MagicMock(spec=_LullCollator)

        await speaking_queue.put((42, True))
        await speaking_queue.put((42, False))

        task = asyncio.create_task(_speaking_monitor(speaking_queue, collator))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert collator.on_speaking.call_count == 2
        collator.on_speaking.assert_any_call(42, is_speaking=True)
        collator.on_speaking.assert_any_call(42, is_speaking=False)
