"""Voice transcription pipeline — per-user Deepgram streams.

Manages the lifecycle of per-user transcription streams, routing audio
from a shared tagged queue to individual Deepgram WebSocket connections
and logging transcription results with the correct Discord user names.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from familiar_connect.transcription import DeepgramTranscriber, TranscriptionResult
    from familiar_connect.voice.lull_collator import VoiceLullCollator

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Raised when a pipeline operation is invalid (e.g. double-set)."""


# ---------------------------------------------------------------------------
# Per-user stream state
# ---------------------------------------------------------------------------


@dataclass
class _UserStream:
    """State for a single user's transcription stream."""

    transcriber: DeepgramTranscriber
    audio_queue: asyncio.Queue[bytes]
    transcript_queue: asyncio.Queue[TranscriptionResult]
    pump_task: asyncio.Task[None]
    forwarder_task: asyncio.Task[None]


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


@dataclass
class VoicePipeline:
    """State for the active voice transcription pipeline."""

    template: DeepgramTranscriber
    tagged_audio_queue: asyncio.Queue[tuple[int, bytes]]
    shared_transcript_queue: asyncio.Queue[tuple[int, TranscriptionResult]]
    router_task: asyncio.Task[None] | None
    logger_task: asyncio.Task[None] | None
    user_names: dict[int, str]
    resolve_name: Callable[[int], str | None] | None = None
    response_handler: (
        Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
    ) = None
    collator: VoiceLullCollator | None = None
    streams: dict[int, _UserStream] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level single-slot registry
# ---------------------------------------------------------------------------

_active: VoicePipeline | None = None


def get_pipeline() -> VoicePipeline | None:
    """Return the current active pipeline, or None."""
    return _active


def set_pipeline(pipeline: VoicePipeline) -> None:
    """Register *pipeline* as the active pipeline.

    :raises PipelineError: If a pipeline is already active.
    """
    global _active  # noqa: PLW0603
    if _active is not None:
        msg = "A pipeline is already active. Call clear_pipeline() first."
        raise PipelineError(msg)
    _active = pipeline


def clear_pipeline() -> None:
    """Clear the active pipeline (no-op if none is set)."""
    global _active  # noqa: PLW0603
    _active = None


def _get_user_name(pipeline: VoicePipeline, user_id: int) -> str:
    """Resolve a display name for *user_id*, caching new lookups.

    Checks the cached ``user_names`` dict first. On a miss, calls the
    optional ``resolve_name`` callback (e.g. scanning channel members)
    and caches the result so subsequent lookups are free.
    """
    name = pipeline.user_names.get(user_id)
    if name is not None:
        return name
    if pipeline.resolve_name is not None:
        resolved = pipeline.resolve_name(user_id)
        if resolved is not None:
            pipeline.user_names[user_id] = resolved
            return resolved
    return f"User-{user_id}"


# ---------------------------------------------------------------------------
# Per-user coroutines
# ---------------------------------------------------------------------------


async def _audio_pump(
    audio_queue: asyncio.Queue[bytes],
    transcriber: DeepgramTranscriber,
) -> None:
    """Drain *audio_queue* and send each chunk to the transcriber."""
    chunks_sent = 0
    consecutive_errors = 0
    while True:
        data = await audio_queue.get()
        try:
            await transcriber.send_audio(data)
            chunks_sent += 1
            consecutive_errors = 0
            if chunks_sent % 100 == 1:
                _logger.debug(
                    "[AudioPump] Sent chunk #%d (%d bytes) to Deepgram",
                    chunks_sent,
                    len(data),
                )
        except Exception:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                _logger.exception("Failed to send audio to transcriber")
            elif consecutive_errors == 4:
                _logger.error(
                    "Deepgram connection appears dead, suppressing further errors"
                )
            # Keep draining the queue so it doesn't back up


async def _transcript_forwarder(
    user_id: int,
    user_queue: asyncio.Queue[TranscriptionResult],
    shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]],
) -> None:
    """Read results from a per-user queue and tag them onto the shared queue."""
    while True:
        result = await user_queue.get()
        await shared_queue.put((user_id, result))


async def _create_user_stream(
    user_id: int,
    template: DeepgramTranscriber,
    shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]],
) -> _UserStream:
    """Create and start a per-user transcription stream."""
    transcriber = template.clone()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    transcript_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
    await transcriber.start(transcript_queue)
    pump_task = asyncio.create_task(_audio_pump(audio_queue, transcriber))
    forwarder_task = asyncio.create_task(
        _transcript_forwarder(user_id, transcript_queue, shared_queue)
    )
    return _UserStream(
        transcriber=transcriber,
        audio_queue=audio_queue,
        transcript_queue=transcript_queue,
        pump_task=pump_task,
        forwarder_task=forwarder_task,
    )


async def _audio_router(
    tagged_queue: asyncio.Queue[tuple[int, bytes]],
    pipeline: VoicePipeline,
) -> None:
    """Route tagged audio to per-user streams, creating them on demand."""
    chunks_routed = 0
    while True:
        user_id, data = await tagged_queue.get()
        if user_id not in pipeline.streams:
            name = _get_user_name(pipeline, user_id)
            _logger.info(
                "[Router] Creating transcription stream for %s (id=%d)",
                name,
                user_id,
            )
            stream = await _create_user_stream(
                user_id,
                pipeline.template,
                pipeline.shared_transcript_queue,
            )
            pipeline.streams[user_id] = stream
        await pipeline.streams[user_id].audio_queue.put(data)
        chunks_routed += 1
        if chunks_routed % 500 == 1:
            _logger.debug(
                "[Router] Routed %d chunks total (%d active streams)",
                chunks_routed,
                len(pipeline.streams),
            )


# ---------------------------------------------------------------------------
# Transcript logger
# ---------------------------------------------------------------------------


async def _run_response_handler(
    handler: Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]],
    user_id: int,
    result: TranscriptionResult,
) -> None:
    """Run a response handler, logging any errors."""
    try:
        await handler(user_id, result)
    except Exception:
        _logger.exception("Voice response handler failed")


async def _transcript_logger(
    shared_queue: asyncio.Queue[tuple[int, TranscriptionResult]],
    pipeline: VoicePipeline,
) -> None:
    """Log transcription results with the speaker's display name.

    When a :class:`VoiceLullCollator` is configured, final transcripts
    are routed through it so multiple ``is_final`` fragments within a
    single SPEAKING burst collate into one downstream dispatch. When
    no collator is set (e.g. tests, or a caller that wants raw
    per-``is_final`` dispatch) the legacy path fires the response
    handler directly as a background task so the logger is never
    blocked by slow LLM / TTS processing.
    """
    pending: set[asyncio.Task[None]] = set()
    while True:
        user_id, result = await shared_queue.get()
        name = _get_user_name(pipeline, user_id)
        if result.is_final:
            _logger.info("[Transcription] %s: %s", name, result.text)
            if pipeline.collator is not None:
                pipeline.collator.on_final(user_id, result)
            elif pipeline.response_handler is not None:
                task = asyncio.create_task(
                    _run_response_handler(
                        pipeline.response_handler,
                        user_id,
                        result,
                    ),
                )
                pending.add(task)
                task.add_done_callback(pending.discard)
        else:
            _logger.debug("[Transcription interim] %s: %s", name, result.text)


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


async def start_pipeline(  # noqa: RUF029
    template: DeepgramTranscriber,
    user_names: dict[int, str],
    resolve_name: Callable[[int], str | None] | None = None,
    response_handler: (
        Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
    ) = None,
    collator: VoiceLullCollator | None = None,
) -> VoicePipeline:
    """Create and register the voice transcription pipeline.

    :param template: A configured transcriber whose settings are cloned
        for each user stream.
    :param user_names: Mapping of Discord user IDs to display names.
    :param resolve_name: Optional callback to resolve a display name for
        user IDs not in *user_names* (e.g. late joiners). The result is
        cached in *user_names* automatically.
    :param response_handler: Optional async callback invoked with
        ``(user_id, result)`` for each final transcription. Used to
        trigger LLM + TTS responses.
    :param collator: Optional :class:`VoiceLullCollator`. When set,
        each ``is_final`` is routed into ``collator.on_final`` instead
        of directly invoking ``response_handler`` — the collator calls
        the handler itself after the SPEAKING-gated lull elapses. Its
        lifecycle is owned by the pipeline: :func:`stop_pipeline`
        closes it.
    :raises PipelineError: If a pipeline is already active.
    """
    tagged_audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
    shared_transcript_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = (
        asyncio.Queue()
    )

    # We create a placeholder pipeline first so the router can reference it.
    pipeline = VoicePipeline(
        template=template,
        tagged_audio_queue=tagged_audio_queue,
        shared_transcript_queue=shared_transcript_queue,
        router_task=None,
        logger_task=None,
        user_names=user_names,
        resolve_name=resolve_name,
        response_handler=response_handler,
        collator=collator,
    )

    pipeline.router_task = asyncio.create_task(
        _audio_router(tagged_audio_queue, pipeline)
    )
    pipeline.logger_task = asyncio.create_task(
        _transcript_logger(shared_transcript_queue, pipeline)
    )

    set_pipeline(pipeline)
    _logger.info("Voice transcription pipeline started")
    return pipeline


async def _stop_user_stream(stream: _UserStream) -> None:
    """Cancel tasks and stop the transcriber for a single user stream."""
    stream.pump_task.cancel()
    stream.forwarder_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stream.pump_task
    with contextlib.suppress(asyncio.CancelledError):
        await stream.forwarder_task
    await stream.transcriber.stop()


async def stop_pipeline() -> None:
    """Stop the active voice transcription pipeline.

    Cancels all tasks, stops all per-user transcribers, and clears the
    module registry. No-op if no pipeline is active.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        return

    # Stop per-user streams first.
    for stream in pipeline.streams.values():
        await _stop_user_stream(stream)

    # Cancel top-level tasks.
    if pipeline.router_task is not None:
        pipeline.router_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pipeline.router_task
    if pipeline.logger_task is not None:
        pipeline.logger_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pipeline.logger_task

    # Tear down the collator last so any in-flight wait task is cancelled.
    if pipeline.collator is not None:
        await pipeline.collator.close()

    clear_pipeline()
    _logger.info("Voice transcription pipeline stopped")
