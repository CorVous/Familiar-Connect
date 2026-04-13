"""Deepgram streaming transcription client."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import aiohttp

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from typing import Self

_logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Transcription result
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """A single transcription result from the Deepgram streaming API."""

    text: str
    is_final: bool
    start: float
    end: float
    confidence: float = 0.0
    speaker: int | None = None

    def to_message(self: Self, speaker_names: dict[int, str] | None = None) -> Message:
        """Convert to an LLM Message.

        Content is prefixed with '[Voice] ' so the model can identify the
        source. The message name is resolved from *speaker_names* when a
        speaker ID is present, otherwise defaults to 'Voice'.
        """
        name = "Voice"
        if self.speaker is not None and speaker_names is not None:
            name = speaker_names.get(self.speaker, "Voice")
        return Message(role="user", content=f"[Voice] {self.text}", name=name)


# ---------------------------------------------------------------------------
# Deepgram streaming transcriber
# ---------------------------------------------------------------------------


class DeepgramTranscriber:
    """Client for streaming audio to Deepgram and receiving transcriptions.

    Connects to Deepgram's WebSocket streaming API, sends raw PCM audio
    frames, and outputs :class:`TranscriptionResult` objects to an
    ``asyncio.Queue``.
    """

    def __init__(
        self: Self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 48000,
        channels: int = 1,
        diarize: bool = False,
        interim_results: bool = True,
        utterance_end_ms: int = 1000,
        vad_events: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.diarize = diarize
        self.interim_results = interim_results
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        # Audio sent while the WebSocket is closed (e.g. between a
        # server-side session rotation and the reconnect completing) is
        # buffered here and flushed to the new socket once ``_reconnect``
        # finishes. Bounded by :attr:`_PENDING_AUDIO_MAX_BYTES`.
        self._pending_audio: deque[bytes] = deque()
        self._pending_audio_bytes: int = 0
        # Audio successfully sent to the current ws since the last
        # is_final Results. Replayed to the new socket on reconnect so
        # an utterance whose bytes made it to the old socket — but
        # whose transcription hadn't come back before the server-side
        # rotation closed the ws — still gets recognised. Cleared on
        # each is_final Results to avoid double-transcription.
        self._recent_audio: deque[bytes] = deque()
        self._recent_audio_bytes: int = 0
        # Serialises ``send_audio`` against ``_reconnect`` so the ws
        # swap + buffer flush happens atomically w.r.t. concurrent
        # sends from ``_audio_pump``. Without this, the event loop can
        # interleave a fresh ``send_bytes`` between the ws swap and
        # the flush, so Deepgram receives the new stream's audio out
        # of order (fresh chunk, then stale buffered chunks) and its
        # VAD rejects the mix as non-speech.
        self._send_lock: asyncio.Lock = asyncio.Lock()

    def build_ws_url(self: Self) -> str:
        """Build the Deepgram WebSocket URL with query parameters."""
        params: dict[str, str] = {
            "model": self.model,
            "language": self.language,
            "sample_rate": str(self.sample_rate),
            "channels": str(self.channels),
            "encoding": "linear16",
            "interim_results": str(self.interim_results).lower(),
            "utterance_end_ms": str(self.utterance_end_ms),
            "vad_events": str(self.vad_events).lower(),
        }
        if self.diarize:
            params["diarize"] = "true"
        return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"

    def build_headers(self: Self) -> dict[str, str]:
        """Build HTTP headers for the Deepgram WebSocket connection."""
        return {"Authorization": f"Token {self.api_key}"}

    def clone(self: Self) -> DeepgramTranscriber:
        """Create a new transcriber with the same configuration.

        Useful for spawning per-user streams that share the same API key
        and Deepgram settings but maintain independent WebSocket connections.
        """
        return DeepgramTranscriber(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate=self.sample_rate,
            channels=self.channels,
            diarize=self.diarize,
            interim_results=self.interim_results,
            utterance_end_ms=self.utterance_end_ms,
            vad_events=self.vad_events,
        )

    def _parse_response(self: Self, data: dict[str, Any]) -> TranscriptionResult | None:
        """Parse a Deepgram JSON response into a TranscriptionResult.

        Returns ``None`` for non-Results messages, empty transcripts, or
        responses with no alternatives.
        """
        if data.get("type") != "Results":
            return None

        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return None

        best = alternatives[0]
        transcript = best.get("transcript", "")
        if not transcript:
            return None

        # Extract speaker from the first word if diarization is active.
        speaker: int | None = None
        words = best.get("words", [])
        if words and "speaker" in words[0]:
            speaker = int(words[0]["speaker"])

        start = float(data.get("start", 0.0))
        duration = float(data.get("duration", 0.0))

        return TranscriptionResult(
            text=transcript,
            is_final=bool(data.get("is_final")),
            start=start,
            end=start + duration,
            confidence=float(best.get("confidence", 0.0)),
            speaker=speaker,
        )

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def _ws_connect(
        self: Self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
    ) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection. Extracted for testability."""
        return await session.ws_connect(url, headers=headers)

    async def start(self: Self, output: asyncio.Queue[TranscriptionResult]) -> None:
        """Connect to Deepgram and begin receiving transcription results.

        :param output: Queue where parsed :class:`TranscriptionResult` objects
            are placed as they arrive from Deepgram.
        """
        url = self.build_ws_url()
        _logger.info("Connecting to Deepgram: %s", url)
        self._session = aiohttp.ClientSession()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        _logger.info("Deepgram WebSocket connected (status=%s)", self._ws.close_code)
        self._receive_task = asyncio.create_task(self._receive_loop(output))
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def send_audio(self: Self, data: bytes) -> None:
        """Send raw PCM audio bytes to the Deepgram WebSocket.

        If the WebSocket has already been closed (e.g. server-side
        session rotation), the bytes are buffered in
        :attr:`_pending_audio` and flushed to the new socket once
        :meth:`_reconnect` succeeds. The buffer is bounded by
        :attr:`_PENDING_AUDIO_MAX_BYTES`; when it overflows the oldest
        frames are evicted so we retain the most recent audio.

        A :class:`ConnectionResetError` (including aiohttp's
        :class:`ClientConnectionResetError`) can also be raised by
        ``send_bytes`` when the transport is closing but
        ``ws.closed`` has not yet flipped to ``True``. That race is
        treated the same as a closed socket: the bytes are buffered
        so the upcoming reconnect can flush them.

        :raises RuntimeError: If called before :meth:`start`.
        """
        if self._ws is None:
            msg = "Transcriber is not connected — call start() first"
            raise RuntimeError(msg)
        # Serialise with ``_reconnect`` so the ws swap + buffer flush
        # appear atomic to audio_pump. Otherwise we can ship fresh
        # audio to the new socket before the stale buffered chunks,
        # which Deepgram's VAD sees as a non-speech burst.
        async with self._send_lock:
            if self._ws is None or self._ws.closed:
                self._buffer_pending_audio(data)
                return
            try:
                await self._ws.send_bytes(data)
            except ConnectionResetError:
                # Race: the transport is closing but ws.closed is still
                # False. Buffer the bytes so _reconnect can flush them to
                # the new socket instead of dropping them in _audio_pump.
                self._buffer_pending_audio(data)
                return
            # Record successful sends so we can replay them to the
            # new socket if Deepgram rotates before emitting a final
            # Results frame for this utterance.
            self._record_recent_audio(data)

    def _buffer_pending_audio(self: Self, data: bytes) -> None:
        """Append *data* to the pending-audio buffer.

        Evicts the oldest frames when the buffer exceeds
        :attr:`_PENDING_AUDIO_MAX_BYTES`.
        """
        self._pending_audio.append(data)
        self._pending_audio_bytes += len(data)
        while (
            self._pending_audio_bytes > self._PENDING_AUDIO_MAX_BYTES
            and self._pending_audio
        ):
            evicted = self._pending_audio.popleft()
            self._pending_audio_bytes -= len(evicted)

    def _record_recent_audio(self: Self, data: bytes) -> None:
        """Record a successfully-sent chunk for possible replay.

        Evicts the oldest frames when the buffer exceeds
        :attr:`_RECENT_AUDIO_MAX_BYTES`.
        """
        self._recent_audio.append(data)
        self._recent_audio_bytes += len(data)
        while (
            self._recent_audio_bytes > self._RECENT_AUDIO_MAX_BYTES
            and self._recent_audio
        ):
            evicted = self._recent_audio.popleft()
            self._recent_audio_bytes -= len(evicted)

    async def _flush_pending_audio(
        self: Self, ws: aiohttp.ClientWebSocketResponse
    ) -> int:
        """Send buffered audio frames to *ws*. Returns the number of bytes flushed."""
        flushed = 0
        while self._pending_audio:
            chunk = self._pending_audio.popleft()
            self._pending_audio_bytes -= len(chunk)
            try:
                await ws.send_bytes(chunk)
            except Exception:
                # Drop the remainder; the receive loop will either
                # recover on the next iteration or give up cleanly.
                _logger.debug("Failed to flush buffered audio chunk", exc_info=True)
                break
            flushed += len(chunk)
        return flushed

    async def stop(self: Self) -> None:
        """Gracefully close the Deepgram connection."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.send_json({"type": "CloseStream"})
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None

        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

        if self._session is not None:
            await self._session.close()
            self._session = None

    # Raised from 5 to 10 so a burst of Deepgram-server-initiated
    # drops (e.g. 1000 rotation followed by a transient 1006 shortly
    # after) doesn't exhaust retries before transcription resumes.
    # ``consecutive_reconnects`` is reset on any live server frame, so
    # this cap only triggers when Deepgram is truly unreachable.
    _MAX_RECONNECTS: int = 10
    _RECONNECT_DELAY: float = 0.1
    _KEEPALIVE_INTERVAL: float = 5.0
    # ~20 s of 48 kHz mono s16le — large enough to cover a reconnect
    # window comfortably, bounded so a persistently dead socket can't
    # grow the buffer without limit.
    _PENDING_AUDIO_MAX_BYTES: int = 2_000_000
    # ~3 s of 48 kHz mono s16le. Big enough to cover a short
    # utterance interrupted mid-flight by Deepgram's rotation, small
    # enough that the inevitable replay overlap stays brief.
    _RECENT_AUDIO_MAX_BYTES: int = 288_000

    async def _keepalive_loop(self: Self) -> None:
        """Send periodic KeepAlive frames to prevent Deepgram's idle timeout.

        Deepgram closes streaming connections after ~10 seconds of silence.
        During bot processing (LLM, TTS) the pump has no user audio to
        forward, so without a keepalive the connection is torn down and
        the transcriber enters a reconnect loop that can exhaust
        :attr:`_MAX_RECONNECTS`.

        The loop re-reads :attr:`_ws` on each tick so it transparently
        follows reconnects initiated by :meth:`_reconnect`.
        """
        while True:
            await asyncio.sleep(self._KEEPALIVE_INTERVAL)
            ws = self._ws
            if ws is None or ws.closed:
                continue
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "KeepAlive"})

    async def _reconnect(self: Self) -> None:
        """Re-establish the Deepgram WebSocket connection.

        Closes the old session and opens a fresh one so ``send_audio``
        and the receive loop can continue transparently. The ws swap
        and the buffer flush are performed under ``_send_lock`` so
        that concurrent ``send_audio`` calls from the audio pump can't
        interleave fresh audio with stale buffered chunks — Deepgram's
        VAD on a fresh session rejects out-of-order audio as
        non-speech, which silently suppresses transcription.
        """
        # Tear down old connection.
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()

        # Open a new connection outside the lock so network latency
        # doesn't block the audio pump any longer than necessary.
        new_session = aiohttp.ClientSession()
        url = self.build_ws_url()
        try:
            new_ws = await self._ws_connect(
                new_session,
                url,
                self.build_headers(),
            )
        except BaseException:
            # Don't leak the half-constructed session if ws_connect
            # raises (including asyncio.CancelledError from stop()).
            with contextlib.suppress(Exception):
                await new_session.close()
            raise

        # Atomically swap the new socket in and drain the pending
        # buffer before any other coroutine can call send_audio.
        async with self._send_lock:
            self._session = new_session
            self._ws = new_ws
            _logger.info("Deepgram WebSocket reconnected")
            # Replay in-flight audio (bytes we already sent to the
            # old socket but that weren't acknowledged by a final
            # Results frame before it closed) before flushing the
            # pending buffer, so Deepgram sees the audio in the
            # order the microphone produced it.
            if self._recent_audio:
                replayed = 0
                for chunk in list(self._recent_audio):
                    try:
                        await new_ws.send_bytes(chunk)
                    except Exception:
                        _logger.debug("Failed to replay recent audio", exc_info=True)
                        break
                    replayed += len(chunk)
                if replayed:
                    _logger.info(
                        "Replayed %d bytes of in-flight audio to new Deepgram socket",
                        replayed,
                    )
            if self._pending_audio:
                flushed = await self._flush_pending_audio(new_ws)
                if flushed:
                    _logger.info(
                        "Flushed %d buffered audio bytes to new Deepgram socket",
                        flushed,
                    )

    async def _receive_loop(
        self: Self,
        output: asyncio.Queue[TranscriptionResult],
    ) -> None:
        """Read messages from the WebSocket, reconnecting on drops."""
        consecutive_reconnects = 0
        # Track whether the current utterance (bounded by SpeechStarted
        # / UtteranceEnd) produced any non-empty final transcript. Lets
        # us flag utterances that Deepgram's VAD detected but its ASR
        # silently dropped — distinct from client-side audio loss.
        utterance_had_final_transcript = False

        while consecutive_reconnects <= self._MAX_RECONNECTS:
            if self._ws is None:
                return
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    # Any valid server-emitted frame proves the socket
                    # is alive; reset the reconnect counter so a burst
                    # of short-lived sessions doesn't exhaust retries.
                    consecutive_reconnects = 0
                    if msg_type == "Results":
                        result = self._parse_response(data)
                        if result is not None:
                            await output.put(result)
                            if result.is_final:
                                utterance_had_final_transcript = True
                                # Deepgram acknowledged this span; drop
                                # the replay buffer so a subsequent
                                # rotation doesn't re-submit already-
                                # transcribed audio.
                                self._recent_audio.clear()
                                self._recent_audio_bytes = 0
                    elif msg_type == "SpeechStarted":
                        utterance_had_final_transcript = False
                        _logger.info("[Deepgram] SpeechStarted")
                    elif msg_type == "UtteranceEnd":
                        if not utterance_had_final_transcript:
                            _logger.warning(
                                "[Deepgram] UtteranceEnd with no final "
                                "transcript — Deepgram heard speech "
                                "but produced no recognisable text."
                            )
                        else:
                            _logger.info("[Deepgram] UtteranceEnd")
                        utterance_had_final_transcript = False
                    else:
                        _logger.debug(
                            "[Deepgram] %s: %s",
                            msg_type,
                            msg.data[:200],
                        )
                elif msg.type in {
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    _logger.warning(
                        "Deepgram WebSocket closed/error: type=%s data=%s",
                        msg.type,
                        getattr(msg, "data", None),
                    )
                    break

            close_code = self._ws.close_code if self._ws else None
            consecutive_reconnects += 1
            if close_code == 1000:
                # 1000 is the standard WebSocket "normal closure" code.
                # Deepgram emits it when rotating long-lived streaming
                # sessions server-side; it is expected, not an error.
                _logger.info(
                    "Deepgram session rotated (close_code=1000), reconnecting (%d/%d)…",
                    consecutive_reconnects,
                    self._MAX_RECONNECTS,
                )
            else:
                _logger.warning(
                    "Deepgram WebSocket closed (close_code=%s), reconnecting (%d/%d)…",
                    close_code,
                    consecutive_reconnects,
                    self._MAX_RECONNECTS,
                )

            try:
                await asyncio.sleep(self._RECONNECT_DELAY)
                await self._reconnect()
            except Exception:
                _logger.exception("Deepgram reconnection failed")
                return

        _logger.error(
            "Deepgram: max reconnect attempts (%d) exhausted",
            self._MAX_RECONNECTS,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_transcriber_from_env() -> DeepgramTranscriber:
    """Create a :class:`DeepgramTranscriber` from environment variables.

    Required: ``DEEPGRAM_API_KEY``
    Optional: ``DEEPGRAM_MODEL``, ``DEEPGRAM_LANGUAGE``

    :raises ValueError: If ``DEEPGRAM_API_KEY`` is not set.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        msg = "DEEPGRAM_API_KEY environment variable is required"
        raise ValueError(msg)

    model = os.environ.get("DEEPGRAM_MODEL") or DEFAULT_MODEL
    language = os.environ.get("DEEPGRAM_LANGUAGE") or DEFAULT_LANGUAGE

    return DeepgramTranscriber(api_key=api_key, model=model, language=language)
