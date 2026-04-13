"""Deepgram streaming transcription client."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
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

        Silently returns if the WebSocket has already been closed (e.g.
        server-side disconnect). This avoids noisy errors when Deepgram
        terminates the connection while audio is still flowing.

        :raises RuntimeError: If called before :meth:`start`.
        """
        if self._ws is None:
            msg = "Transcriber is not connected — call start() first"
            raise RuntimeError(msg)
        if self._ws.closed:
            return
        await self._ws.send_bytes(data)

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

    _MAX_RECONNECTS: int = 5
    _RECONNECT_DELAY: float = 1.0
    _KEEPALIVE_INTERVAL: float = 3.0

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
        and the receive loop can continue transparently.
        """
        # Tear down old connection.
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()

        # Open a new connection.
        self._session = aiohttp.ClientSession()
        url = self.build_ws_url()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        _logger.info("Deepgram WebSocket reconnected")

    async def _receive_loop(
        self: Self,
        output: asyncio.Queue[TranscriptionResult],
    ) -> None:
        """Read messages from the WebSocket, reconnecting on drops."""
        consecutive_reconnects = 0

        while consecutive_reconnects <= self._MAX_RECONNECTS:
            if self._ws is None:
                return
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    if msg_type == "Results":
                        result = self._parse_response(data)
                        if result is not None:
                            await output.put(result)
                            # Got real data — reset reconnect counter.
                            consecutive_reconnects = 0
                    else:
                        _logger.info(
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
