"""TTS client — Cartesia streaming WebSocket.

Returns audio bytes + per-word timestamps (needed for mid-speech
yield in voice interruption flow).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import aiohttp

if TYPE_CHECKING:
    from typing import Any, Self


_logger = logging.getLogger(__name__)


CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2024-06-10"
DEFAULT_SAMPLE_RATE = 48000  # matches Discord's native rate


@dataclass(frozen=True)
class WordTimestamp:
    """Per-word playback window (ms from audio start)."""

    word: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TTSResult:
    """Synthesized audio + per-word timestamps."""

    audio: bytes
    timestamps: list[WordTimestamp] = field(default_factory=list)


class CartesiaTTSClient:
    """Cartesia TTS WebSocket client; one connection per :meth:`synthesize`."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        voice_id: str,
        model: str,
        base_url: str = CARTESIA_BASE_URL,
        ws_url: str = CARTESIA_WS_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.base_url = base_url
        self.ws_url = ws_url
        self.sample_rate = sample_rate

    def build_ws_url(self: Self) -> str:
        """Return the Cartesia WebSocket URL with auth in the query string."""
        query = urlencode(
            {
                "api_key": self.api_key,
                "cartesia_version": CARTESIA_API_VERSION,
            },
        )
        return f"{self.ws_url}?{query}"

    def build_headers(self: Self) -> dict[str, str]:
        """Return REST headers (kept for any non-WS call sites / tests)."""
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": CARTESIA_API_VERSION,
            "Content-Type": "application/json",
        }

    def build_payload(self: Self, text: str, *, context_id: str) -> dict[str, Any]:
        """Build JSON payload for one-shot TTS synthesis."""
        return {
            "context_id": context_id,
            "model_id": self.model,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "language": "en",
            "add_timestamps": True,
            "continue": False,
        }

    async def _ws_connect(
        self: Self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection. Extracted for testability."""
        return await session.ws_connect(url)

    async def synthesize(self: Self, text: str) -> TTSResult:
        """Synthesize *text* via WebSocket; return audio + word timestamps.

        :raises RuntimeError: on Cartesia ``error`` event or unexpected close.
        """
        context_id = uuid.uuid4().hex
        url = self.build_ws_url()
        payload = self.build_payload(text, context_id=context_id)

        audio_parts: list[bytes] = []
        timestamps: list[WordTimestamp] = []

        async with aiohttp.ClientSession() as session:
            ws = await self._ws_connect(session, url)
            try:
                await ws.send_json(payload)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        if self._handle_event(event, audio_parts, timestamps):
                            break
                    elif msg.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        msg_txt = (
                            f"Cartesia WebSocket closed unexpectedly (type={msg.type})"
                        )
                        raise RuntimeError(msg_txt)
            finally:
                if not ws.closed:
                    await ws.close()

        audio = b"".join(audio_parts)
        _logger.info(
            "tts: %d words, audio=%d bytes, timing=%.0fms→%.0fms",
            len(timestamps),
            len(audio),
            timestamps[0].start_ms if timestamps else 0.0,
            timestamps[-1].end_ms if timestamps else 0.0,
        )
        return TTSResult(audio=audio, timestamps=timestamps)

    def _handle_event(
        self: Self,
        event: dict[str, Any],
        audio_parts: list[bytes],
        timestamps: list[WordTimestamp],
    ) -> bool:
        """Dispatch a single parsed Cartesia event. Returns True when done."""
        event_type = event.get("type")
        if event_type == "chunk":
            data = event.get("data")
            if isinstance(data, str):
                audio_parts.append(base64.b64decode(data))
        elif event_type == "timestamps":
            word_ts = event.get("word_timestamps") or {}
            timestamps.extend(_parse_word_timestamps(word_ts))
        elif event_type == "done":
            return True
        elif event_type == "error":
            err = event.get("error") or "unknown error"
            status = event.get("status_code")
            msg = f"Cartesia TTS error (status={status}): {err}"
            raise RuntimeError(msg)
        return False


def _parse_word_timestamps(raw: dict[str, Any]) -> list[WordTimestamp]:
    """Convert Cartesia's parallel-array word_timestamps into objects.

    Cartesia emits ``{"words": [...], "start": [...], "end": [...]}``
    with times in seconds. We flatten to a list and convert to ms so
    all interruption math uses a single unit.
    """
    words = raw.get("words") or []
    starts = raw.get("start") or []
    ends = raw.get("end") or []
    count = min(len(words), len(starts), len(ends))
    return [
        WordTimestamp(
            word=str(words[i]),
            start_ms=float(starts[i]) * 1000.0,
            end_ms=float(ends[i]) * 1000.0,
        )
        for i in range(count)
    ]


def create_tts_client(voice_id: str, model: str) -> CartesiaTTSClient:
    """Create client from character-config values + ``CARTESIA_API_KEY`` env var.

    :raises ValueError: If API key missing or args empty.
    """
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        msg = "CARTESIA_API_KEY environment variable is required"
        raise ValueError(msg)
    if not voice_id:
        msg = "TTS voice_id is required (set [tts].voice_id in character.toml)"
        raise ValueError(msg)
    if not model:
        msg = "TTS model is required (set [tts].model in character.toml)"
        raise ValueError(msg)

    return CartesiaTTSClient(api_key=api_key, voice_id=voice_id, model=model)
