"""Voice transcription pipeline — per-user Deepgram streams.

Manages the lifecycle of per-user transcription streams, routing audio
from a shared tagged queue to individual Deepgram WebSocket connections
and logging transcription results with the correct Discord user names.

Lull collation
--------------
Rapid speech produces multiple Deepgram ``is_final`` results in quick
succession.  Without collation each result would trigger a separate LLM
call.  :class:`_LullCollator` fixes this by:

1. Buffering every ``is_final`` transcript per user.
2. Relying on Discord SPEAKING events to gate the lull timer:

   * ``SPEAKING=True``  → cancel the lull timer (user is still talking).
   * ``SPEAKING=False`` → start the lull countdown.

3. When the lull timer fires:

   * If buffered text is present → combine it and invoke the response
     handler once.
   * If no text yet → set a ``lull_fired`` flag and wait.

4. When a transcript arrives after ``lull_fired`` → dispatch immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.transcription import TranscriptionResult

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from familiar_connect.transcription import DeepgramTranscriber

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
    speaking_monitor_task: asyncio.Task[None] | None = None
    lull_timeout: float = 1.5
    resolve_name: Callable[[int], str | None] | None = None
    response_handler: (
        Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
    ) = None
    streams: dict[int, _UserStream] = field(default_factory=dict)
    speaking_queue: asyncio.Queue[tuple[int, bool]] = field(
        default_factory=asyncio.Queue
    )


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
# Lull collator
# ---------------------------------------------------------------------------


class _LullCollator:
    """Buffer per-user transcripts and dispatch one combined response after a lull.

    The lull timer is driven exclusively by Discord SPEAKING events:

    * :meth:`on_speaking` with ``is_speaking=True``  → cancel timer.
    * :meth:`on_speaking` with ``is_speaking=False`` → start countdown.

    Deepgram ``is_final`` transcripts are buffered via
    :meth:`on_final_transcript`.  They are only dispatched after the lull
    timer fires.  If the timer fires before a transcript arrives, the
    ``lull_fired`` flag is set and the next transcript triggers an
    immediate dispatch.
    """

    def __init__(
        self,
        lull_timeout: float,
        response_handler: (
            Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None
        ),
    ) -> None:
        self._lull_timeout = lull_timeout
        self._response_handler = response_handler
        self._pending_texts: dict[int, list[str]] = {}
        self._lull_timers: dict[int, asyncio.TimerHandle] = {}
        self._lull_fired: dict[int, bool] = {}
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def on_speaking(self, user_id: int, *, is_speaking: bool) -> None:
        """Update lull timer based on Discord voice activity for *user_id*.

        * ``is_speaking=True``  → user is talking; cancel timer.
        * ``is_speaking=False`` → user stopped; start countdown.
        """
        if is_speaking:
            self._cancel_lull_timer(user_id)
        else:
            self._start_lull_timer(user_id)

    def on_final_transcript(
        self,
        user_id: int,
        result: TranscriptionResult,
    ) -> None:
        """Buffer a final Deepgram transcript for *user_id*.

        If the lull has already fired (user was silent before this
        transcript arrived), dispatch immediately.  Otherwise just
        accumulate — the lull timer will trigger dispatch later.
        """
        self._pending_texts.setdefault(user_id, []).append(result.text)

        if self._lull_fired.pop(user_id, False):
            # Lull fired while waiting for transcript — dispatch now.
            self._dispatch(user_id, last_result=result)

    # ------------------------------------------------------------------
    # Internal timer helpers
    # ------------------------------------------------------------------

    def _cancel_lull_timer(self, user_id: int) -> None:
        handle = self._lull_timers.pop(user_id, None)
        if handle is not None:
            handle.cancel()
        self._lull_fired.pop(user_id, None)

    def _start_lull_timer(self, user_id: int) -> None:
        self._cancel_lull_timer(user_id)
        loop = asyncio.get_event_loop()
        self._lull_timers[user_id] = loop.call_later(
            self._lull_timeout, self._on_lull, user_id
        )

    def _on_lull(self, user_id: int) -> None:
        """Handle lull timer expiry for *user_id*."""
        self._lull_timers.pop(user_id, None)
        if self._pending_texts.get(user_id):
            self._dispatch(user_id, last_result=None)
        else:
            # No transcript yet; mark as fired so the next one dispatches.
            self._lull_fired[user_id] = True

    def _dispatch(
        self,
        user_id: int,
        last_result: TranscriptionResult | None,
    ) -> None:
        """Combine buffered texts and schedule a response handler task."""
        texts = self._pending_texts.pop(user_id, [])
        self._lull_fired.pop(user_id, None)
        if not texts or self._response_handler is None:
            return

        combined = TranscriptionResult(
            text=" ".join(texts),
            is_final=True,
            start=last_result.start if last_result is not None else 0.0,
            end=last_result.end if last_result is not None else 0.0,
            confidence=last_result.confidence if last_result is not None else 0.0,
            speaker=last_result.speaker if last_result is not None else None,
        )
        task = asyncio.create_task(
            _run_response_handler(self._response_handler, user_id, combined),
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)


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
# Transcript logger + speaking monitor
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
    collator: _LullCollator,
) -> None:
    """Log transcription results and forward finals to the lull collator.

    The collator holds all buffering and timer logic; this coroutine is
    only responsible for reading the shared queue, resolving user names,
    and deciding whether a result is final or interim.
    """
    while True:
        user_id, result = await shared_queue.get()
        name = _get_user_name(pipeline, user_id)
        if result.is_final:
            _logger.info("[Transcription] %s: %s", name, result.text)
            collator.on_final_transcript(user_id, result)
        else:
            _logger.debug("[Transcription interim] %s: %s", name, result.text)


async def _speaking_monitor(
    speaking_queue: asyncio.Queue[tuple[int, bool]],
    collator: _LullCollator,
) -> None:
    """Forward Discord SPEAKING events to the lull collator.

    Each ``(user_id, is_speaking)`` tuple from the speaking queue calls
    :meth:`_LullCollator.on_speaking`, which either cancels or starts the
    per-user lull countdown.
    """
    while True:
        user_id, is_speaking = await speaking_queue.get()
        collator.on_speaking(user_id, is_speaking=is_speaking)


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
    lull_timeout: float = 1.5,
) -> VoicePipeline:
    """Create and register the voice transcription pipeline.

    :param template: A configured transcriber whose settings are cloned
        for each user stream.
    :param user_names: Mapping of Discord user IDs to display names.
    :param resolve_name: Optional callback to resolve a display name for
        user IDs not in *user_names* (e.g. late joiners). The result is
        cached in *user_names* automatically.
    :param response_handler: Optional async callback invoked with
        ``(user_id, result)`` for each collated voice turn after the lull
        timer fires.
    :param lull_timeout: Seconds of silence (no SPEAKING events) before
        the lull timer fires and a combined response is dispatched.
    :raises PipelineError: If a pipeline is already active.
    """
    tagged_audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
    shared_transcript_queue: asyncio.Queue[tuple[int, TranscriptionResult]] = (
        asyncio.Queue()
    )

    # Create the lull collator first so both the logger and monitor tasks
    # share the same instance.
    collator = _LullCollator(
        lull_timeout=lull_timeout,
        response_handler=response_handler,
    )

    # We create a placeholder pipeline first so the router can reference it.
    pipeline = VoicePipeline(
        template=template,
        tagged_audio_queue=tagged_audio_queue,
        shared_transcript_queue=shared_transcript_queue,
        router_task=None,
        logger_task=None,
        speaking_monitor_task=None,
        user_names=user_names,
        resolve_name=resolve_name,
        response_handler=response_handler,
        lull_timeout=lull_timeout,
    )

    pipeline.router_task = asyncio.create_task(
        _audio_router(tagged_audio_queue, pipeline)
    )
    pipeline.logger_task = asyncio.create_task(
        _transcript_logger(shared_transcript_queue, pipeline, collator)
    )
    pipeline.speaking_monitor_task = asyncio.create_task(
        _speaking_monitor(pipeline.speaking_queue, collator)
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
    for task in (
        pipeline.router_task,
        pipeline.logger_task,
        pipeline.speaking_monitor_task,
    ):
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    clear_pipeline()
    _logger.info("Voice transcription pipeline stopped")
