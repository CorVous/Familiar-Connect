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

from familiar_connect import log_style as ls
from familiar_connect.transcription import TranscriptionResult

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from familiar_connect.transcription import DeepgramTranscriber, TranscriptionEvent

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
    transcript_queue: asyncio.Queue[TranscriptionEvent]
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
    shared_transcript_queue: asyncio.Queue[tuple[int, TranscriptionEvent]]
    router_task: asyncio.Task[None] | None
    logger_task: asyncio.Task[None] | None
    user_names: dict[int, str]
    channel_name: str = ""
    resolve_name: Callable[[int], str | None] | None = None
    response_handler: (
        Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
    ) = None
    on_speech_start: Callable[[int], None] | None = None
    on_speech_end: Callable[[int], None] | None = None
    feed_vad: Callable[[int, bytes], None] | None = None
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


DEFAULT_IDLE_FINALIZE_S: float = 0.5
"""How long to wait with no Discord audio before forcing a Deepgram flush.

Discord's client-side VAD stops RTP delivery during silence, so
Deepgram's endpointer never sees the in-stream silence it needs and
holds the buffered final until the next speech burst. After this many
seconds of no chunks arriving, the pump sends ``{"type":"Finalize"}``
to force Deepgram to emit immediately.
"""


async def _audio_pump(
    audio_queue: asyncio.Queue[bytes],
    transcriber: DeepgramTranscriber,
    *,
    idle_finalize_s: float = DEFAULT_IDLE_FINALIZE_S,
) -> None:
    """Drain *audio_queue* into the transcriber; flush on idle gaps.

    Two-state loop: ``dirty`` (audio buffered server-side, idle window
    armed) and ``drained`` (already flushed, blocking until next chunk).
    Real audio enters ``dirty``; an idle timeout in ``dirty`` sends
    ``Finalize`` and transitions to ``drained``.
    """
    chunks_sent = 0
    consecutive_errors = 0
    dirty = False
    while True:
        if dirty:
            try:
                data = await asyncio.wait_for(
                    audio_queue.get(), timeout=idle_finalize_s
                )
            except TimeoutError:
                with contextlib.suppress(Exception):
                    await transcriber.finalize()
                dirty = False
                continue
        else:
            data = await audio_queue.get()
        try:
            await transcriber.send_audio(data)
            dirty = True
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
    user_queue: asyncio.Queue[TranscriptionEvent],
    shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]],
) -> None:
    """Read events from a per-user queue and tag them onto the shared queue."""
    while True:
        event = await user_queue.get()
        await shared_queue.put((user_id, event))


async def _create_user_stream(
    user_id: int,
    template: DeepgramTranscriber,
    shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]],
) -> _UserStream:
    """Create and start a per-user transcription stream."""
    transcriber = template.clone()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    transcript_queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
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
    """Route tagged audio to per-user streams, creating them on demand.

    Also feeds each chunk into the optional local VAD (TEN VAD) so
    speech-activity edges arrive independently of Deepgram finals.
    """
    chunks_routed = 0
    while True:
        user_id, data = await tagged_queue.get()
        if pipeline.feed_vad is not None:
            try:
                pipeline.feed_vad(user_id, data)
            except Exception:
                _logger.exception("feed_vad hook failed")
        if user_id not in pipeline.streams:
            name = _get_user_name(pipeline, user_id)
            _logger.info(
                f"{ls.tag('🎙️ Stream Started', ls.C)} "
                f"{ls.kv('user', name, vc=ls.LC)} "
                f"{ls.kv('id', str(user_id), vc=ls.LW)}"
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
    shared_queue: asyncio.Queue[tuple[int, TranscriptionEvent]],
    pipeline: VoicePipeline,
) -> None:
    """Log results + fan finals to the response handler.

    VAD events (speech-start/end) are driven by the local TEN VAD
    detector via :attr:`VoicePipeline.feed_vad`, not by Deepgram —
    this logger only cares about transcription text.
    """
    pending: set[asyncio.Task[None]] = set()
    while True:
        user_id, event = await shared_queue.get()
        if not isinstance(event, TranscriptionResult):
            continue
        name = _get_user_name(pipeline, user_id)
        if event.is_final:
            _logger.info(
                f"{ls.tag('💬 Transcription', ls.LC)} "
                f"{ls.kv('user', name, vc=ls.LC)} "
                f"{ls.kv('text', ls.trunc(event.text))}"
            )
            if pipeline.response_handler is not None:
                task = asyncio.create_task(
                    _run_response_handler(
                        pipeline.response_handler,
                        user_id,
                        event,
                    ),
                )
                pending.add(task)
                task.add_done_callback(pending.discard)
        else:
            _logger.debug("[Transcription interim] %s: %s", name, event.text)


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


async def start_pipeline(  # noqa: RUF029
    template: DeepgramTranscriber,
    user_names: dict[int, str],
    channel_name: str = "",
    resolve_name: Callable[[int], str | None] | None = None,
    response_handler: (
        Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
    ) = None,
    on_speech_start: Callable[[int], None] | None = None,
    on_speech_end: Callable[[int], None] | None = None,
    feed_vad: Callable[[int, bytes], None] | None = None,
) -> VoicePipeline:
    """Create and register the voice transcription pipeline.

    :param template: Transcriber whose settings are cloned per user stream.
    :param user_names: Mapping of Discord user IDs to display names.
    :param channel_name: Human-readable channel name for log output.
    :param resolve_name: Optional display-name lookup for late joiners;
        resolved names are cached in *user_names*.
    :param response_handler: Optional async callback invoked with
        ``(user_id, result)`` for each final transcription.
    :param on_speech_start: Optional sync callback invoked with ``user_id``
        on TEN VAD speech-start edge (wired from the detector).
    :param on_speech_end: Optional sync callback invoked with ``user_id``
        on TEN VAD speech-end edge (wired from the detector).
    :param feed_vad: Optional sync callback ``(user_id, pcm)`` invoked
        for every tagged audio chunk so the local VAD can observe the
        stream in parallel with Deepgram.
    :raises PipelineError: If a pipeline is already active.
    """
    tagged_audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
    shared_transcript_queue: asyncio.Queue[tuple[int, TranscriptionEvent]] = (
        asyncio.Queue()
    )

    # placeholder pipeline so the router can reference it
    pipeline = VoicePipeline(
        template=template,
        tagged_audio_queue=tagged_audio_queue,
        shared_transcript_queue=shared_transcript_queue,
        router_task=None,
        logger_task=None,
        user_names=user_names,
        channel_name=channel_name,
        resolve_name=resolve_name,
        response_handler=response_handler,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        feed_vad=feed_vad,
    )

    pipeline.router_task = asyncio.create_task(
        _audio_router(tagged_audio_queue, pipeline)
    )
    pipeline.logger_task = asyncio.create_task(
        _transcript_logger(shared_transcript_queue, pipeline)
    )

    set_pipeline(pipeline)
    _logger.info(f"{ls.tag('STT Start', ls.G)} {ls.kv('channel', channel_name)}")
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

    ch = pipeline.channel_name
    clear_pipeline()
    _logger.info(f"{ls.tag('STT Stop', ls.Y)} {ls.kv('channel', ch, vc=ls.Y)}")
