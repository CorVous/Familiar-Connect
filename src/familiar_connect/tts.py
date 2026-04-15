"""TTS clients — Cartesia (WebSocket) and Azure (Speech SDK).

Both return :class:`TTSResult` with raw PCM audio + per-word timestamps.
Timestamps drive mid-speech yield in the voice interruption flow.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlencode

import aiohttp

if TYPE_CHECKING:
    from datetime import timedelta

    from familiar_connect.config import TTSConfig


_logger = logging.getLogger(__name__)


CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2024-06-10"
DEFAULT_SAMPLE_RATE = 48000  # matches Discord's native rate

DEFAULT_AZURE_VOICE = "en-US-AmberNeural"
"""Default Azure Neural voice; mirrors ``config.DEFAULT_AZURE_TTS_VOICE``."""

_AZURE_TICKS_PER_MS: float = 10_000.0
"""100-nanosecond ticks per millisecond — Azure SDK offset unit."""

GEMINI_SAMPLE_RATE = 24_000
"""Gemini TTS native output rate; upsampled to 48kHz before returning."""

DEFAULT_GEMINI_VOICE = "Kore"
"""Default Gemini prebuilt voice; mirrors ``config.DEFAULT_GEMINI_TTS_VOICE``."""

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-tts-preview"
"""Default Gemini TTS model; mirrors ``config.DEFAULT_GEMINI_TTS_MODEL``."""


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


# ---------------------------------------------------------------------------
# Cartesia
# ---------------------------------------------------------------------------


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


def _synthesize_word_timestamps(
    text: str,
    audio: bytes,
    sample_rate: int,
) -> list[WordTimestamp]:
    """Synthesize evenly-distributed word timestamps from audio duration.

    Used when the TTS provider does not return per-word timing. Distributes
    whitespace-split words uniformly across total audio duration so interruption
    logic has something to work with.
    """
    words = text.split()
    if not words:
        return []
    # bytes / (2 bytes/sample) / sample_rate → duration in seconds → ms
    duration_ms = (len(audio) / 2 / sample_rate) * 1000.0
    step = duration_ms / len(words)
    return [
        WordTimestamp(
            word=word,
            start_ms=i * step,
            end_ms=(i + 1) * step,
        )
        for i, word in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# Greeting cache (shared across all TTS providers)
# ---------------------------------------------------------------------------

# File-based greeting audio cache: stored in `data/cache/greetings/` keyed by
# hash of (provider, voice_id, greeting). Shared across all TTS clients.

_GREETING_CACHE_DIR = Path("data/cache/greetings")


def _get_greeting_cache_path(provider: str, voice_id: str, greeting: str) -> Path:
    """Return filesystem path for cached greeting audio."""
    # Create a stable hash of the key parts.
    key = f"{provider}:{voice_id}:{greeting}"
    hash_hex = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _GREETING_CACHE_DIR / f"{hash_hex}.bin"


async def get_cached_greeting_audio(
    provider: str,
    voice_id: str,
    greeting: str,
    client: CartesiaTTSClient | AzureTTSClient | GeminiTTSClient,
) -> TTSResult:
    """Return TTS audio for *greeting*.

    Uses a file-based cache keyed by (provider, voice_id, greeting).
    On cache miss, synthesizes and stores the audio bytes to disk.
    Subsequent calls for the same (provider, voice_id, greeting) read
    from the file.
    """
    # Ensure cache directory exists (blocking I/O off the event loop).
    await asyncio.to_thread(_GREETING_CACHE_DIR.mkdir, parents=True, exist_ok=True)

    cache_path = _get_greeting_cache_path(provider, voice_id, greeting)
    if cache_path.is_file():
        # Cache hit: read audio bytes from file.
        audio_bytes = await asyncio.to_thread(cache_path.read_bytes)
        return TTSResult(audio=audio_bytes, timestamps=[])

    # Cache miss: synthesize via TTS client.
    tts_result = await client.synthesize(greeting)
    # Write audio bytes to cache file.
    await asyncio.to_thread(cache_path.write_bytes, tts_result.audio)
    return TTSResult(audio=tts_result.audio, timestamps=[])


# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------


class AzureTTSClient:
    """Azure Cognitive Services TTS client.

    Runs the blocking Speech SDK call in a thread-pool executor so the
    asyncio event loop stays free. Word-boundary events are collected on
    the same executor thread — no locking needed.
    """

    def __init__(
        self: Self,
        *,
        subscription_key: str,
        region: str,
        voice_name: str = DEFAULT_AZURE_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.subscription_key = subscription_key
        self.region = region
        self.voice_name = voice_name
        self.sample_rate = sample_rate

    def _make_synthesizer(self: Self) -> tuple[Any, Any]:
        """Return ``(speechsdk_module, SpeechSynthesizer)``; extracted for tests."""
        import azure.cognitiveservices.speech as speechsdk  # noqa: PLC0415

        speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region,
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
        )
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,  # audio returned via result.audio_data
        )
        return speechsdk, synthesizer

    def _synthesize_sync(self: Self, text: str) -> TTSResult:
        """Blocking synthesis; collect word boundaries, return TTSResult.

        :raises RuntimeError: if Azure synthesis fails or is cancelled.
        """
        speechsdk, synthesizer = self._make_synthesizer()

        word_timestamps: list[WordTimestamp] = []

        def _on_word_boundary(evt: Any) -> None:  # noqa: ANN401
            if evt.boundary_type != speechsdk.SpeechSynthesisBoundaryType.Word:
                return
            start_ms = evt.audio_offset / _AZURE_TICKS_PER_MS
            duration_td: timedelta = evt.duration
            end_ms = start_ms + duration_td.total_seconds() * 1000.0
            word_timestamps.append(
                WordTimestamp(word=evt.text, start_ms=start_ms, end_ms=end_ms),
            )

        synthesizer.synthesis_word_boundary.connect(_on_word_boundary)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio: bytes = result.audio_data
            _logger.info(
                "azure tts: %d words, audio=%d bytes, timing=%.0fms→%.0fms",
                len(word_timestamps),
                len(audio),
                word_timestamps[0].start_ms if word_timestamps else 0.0,
                word_timestamps[-1].end_ms if word_timestamps else 0.0,
            )
            return TTSResult(audio=audio, timestamps=word_timestamps)

        cancellation = speechsdk.CancellationDetails.from_result(result)
        msg = (
            f"Azure TTS synthesis failed: {cancellation.reason}"
            f" — {cancellation.error_details}"
        )
        raise RuntimeError(msg)

    async def synthesize(self: Self, text: str) -> TTSResult:
        """Synthesize *text* in a thread executor; return audio + word timestamps."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiTTSClient:
    """Google Gemini TTS client (google-genai SDK, synchronous, thread-wrapped).

    Gemini returns PCM at 24kHz; audio is upsampled 2x to 48kHz before
    returning so it matches the Discord pipeline. Word timestamps are not
    provided by the API — evenly-distributed estimates are synthesized instead.
    """

    def __init__(
        self: Self,
        *,
        api_key: str,
        voice: str = DEFAULT_GEMINI_VOICE,
        model: str = DEFAULT_GEMINI_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.sample_rate = sample_rate  # target output rate (48kHz)

    def _make_client(self: Self) -> Any:  # noqa: ANN401
        """Return configured genai.Client; extracted for testability."""
        from google import genai  # noqa: PLC0415

        return genai.Client(api_key=self.api_key)

    def _synthesize_sync(self: Self, text: str) -> TTSResult:
        """Blocking synthesis; upsample 24kHz → sample_rate; return TTSResult.

        :raises RuntimeError: on unexpected response structure or API error.
        """
        from google.genai import types  # noqa: PLC0415

        from familiar_connect.voice.audio import upsample_2x  # noqa: PLC0415

        client = self._make_client()
        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice,
                        ),
                    ),
                ),
            ),
        )
        try:
            raw_audio: bytes = response.candidates[0].content.parts[0].inline_data.data
        except (IndexError, AttributeError) as exc:
            msg = f"Gemini TTS returned unexpected response structure: {exc}"
            raise RuntimeError(msg) from exc

        audio = upsample_2x(raw_audio)
        timestamps = _synthesize_word_timestamps(text, audio, self.sample_rate)
        _logger.info(
            "gemini tts: %d words, audio=%d bytes, timing=%.0fms→%.0fms",
            len(timestamps),
            len(audio),
            timestamps[0].start_ms if timestamps else 0.0,
            timestamps[-1].end_ms if timestamps else 0.0,
        )
        return TTSResult(audio=audio, timestamps=timestamps)

    async def synthesize(self: Self, text: str) -> TTSResult:
        """Synthesize *text* in a thread executor; return audio + word timestamps."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tts_client(
    tts_config: TTSConfig,
) -> CartesiaTTSClient | AzureTTSClient | GeminiTTSClient:
    """Instantiate the TTS client for the active provider.

    Provider is taken from ``[tts].provider`` in ``character.toml``.
    Default provider is ``"azure"``.

    Reads credentials from environment variables; raises :class:`ValueError`
    if required variables are absent, config values are empty, or the
    provider name is unrecognised.

    :raises ValueError: unknown provider, missing env var, or empty field.
    """
    provider = tts_config.provider

    if provider == "azure":
        return _create_azure_client(tts_config)
    if provider == "cartesia":
        return _create_cartesia_client(tts_config)
    if provider == "gemini":
        return _create_gemini_client(tts_config)
    msg = (
        f"Unknown TTS provider {provider!r}; expected 'azure', 'cartesia', or 'gemini'"
    )
    raise ValueError(msg)


def _create_azure_client(tts_config: TTSConfig) -> AzureTTSClient:
    subscription_key = os.environ.get("AZURE_SPEECH_KEY")
    if not subscription_key:
        msg = "AZURE_SPEECH_KEY environment variable is required for Azure TTS"
        raise ValueError(msg)
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not region:
        msg = "AZURE_SPEECH_REGION environment variable is required for Azure TTS"
        raise ValueError(msg)
    return AzureTTSClient(
        subscription_key=subscription_key,
        region=region,
        voice_name=tts_config.azure_voice,
    )


def _create_cartesia_client(tts_config: TTSConfig) -> CartesiaTTSClient:
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        msg = "CARTESIA_API_KEY environment variable is required"
        raise ValueError(msg)
    voice_id = tts_config.cartesia_voice_id or ""
    if not voice_id:
        msg = (
            "TTS cartesia_voice_id is required "
            "(set [tts].cartesia_voice_id in character.toml)"
        )
        raise ValueError(msg)
    model = tts_config.cartesia_model or ""
    if not model:
        msg = (
            "TTS cartesia_model is required "
            "(set [tts].cartesia_model in character.toml)"
        )
        raise ValueError(msg)
    return CartesiaTTSClient(api_key=api_key, voice_id=voice_id, model=model)


def _create_gemini_client(tts_config: TTSConfig) -> GeminiTTSClient:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        msg = "GEMINI_API_KEY environment variable is required for Gemini TTS"
        raise ValueError(msg)
    return GeminiTTSClient(
        api_key=api_key,
        voice=tts_config.gemini_voice,
        model=tts_config.gemini_model,
    )
