"""TTS clients — Cartesia (WebSocket), Azure (Speech SDK), Gemini.

All return :class:`TTSResult` with raw PCM audio + per-word timestamps.
Timestamps drive mid-speech yield in voice interruption flow.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlencode

import aiohttp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta

    from google.genai import Client as _GenaiClient

    from familiar_connect.config import TTSConfig


from familiar_connect import log_style as ls

_logger = logging.getLogger(__name__)


CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2024-06-10"
DEFAULT_SAMPLE_RATE = 48000  # matches Discord's native rate

DEFAULT_AZURE_VOICE = "en-US-AmberNeural"
"""Default Azure Neural voice; mirrors ``config.DEFAULT_AZURE_TTS_VOICE``."""

_AZURE_TICKS_PER_MS: float = 10_000.0
"""100-nanosecond ticks per millisecond — Azure SDK offset unit."""

GEMINI_SAMPLE_RATE = 24000
"""Gemini TTS native rate (24 kHz); upsampled to 48 kHz before use."""


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
        """Cartesia WebSocket URL with auth in query string."""
        query = urlencode(
            {
                "api_key": self.api_key,
                "cartesia_version": CARTESIA_API_VERSION,
            },
        )
        return f"{self.ws_url}?{query}"

    def build_headers(self: Self) -> dict[str, str]:
        """REST headers (for non-WS call sites / tests)."""
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": CARTESIA_API_VERSION,
            "Content-Type": "application/json",
        }

    def build_payload(self: Self, text: str, *, context_id: str) -> dict[str, Any]:
        """JSON payload for one-shot TTS synthesis."""
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
        """Open WebSocket. Extracted for testability."""
        return await session.ws_connect(url)

    async def synthesize(self: Self, text: str) -> TTSResult:
        """Synthesize via WebSocket; return audio + word timestamps.

        Raises ``RuntimeError`` on Cartesia ``error`` event or unexpected close.
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
        start_ms = timestamps[0].start_ms if timestamps else 0.0
        end_ms = timestamps[-1].end_ms if timestamps else 0.0
        _logger.info(
            f"{ls.tag('🔉 TTS', ls.C)} "
            f"{ls.word('Cartesia', ls.C)} "
            f"{ls.kv('words', str(len(timestamps)), vc=ls.LW)} "
            f"{ls.kv('audio', f'{len(audio)}b', vc=ls.LW)} "
            f"{ls.kv('timing', f'{start_ms:.0f}ms→{end_ms:.0f}ms', vc=ls.LW)}"
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

    async def synthesize_stream(self: Self, text: str) -> AsyncIterator[bytes]:
        """Yield raw mono PCM chunks as Cartesia produces them.

        Lower-latency variant of :meth:`synthesize`: callers can start
        playback on the first chunk instead of waiting for the full
        utterance. ``timestamps`` events are dropped — chunk consumers
        get audio only. Errors and unexpected closes raise just like
        :meth:`synthesize`.

        Yields:
            bytes: raw mono ``pcm_s16le`` chunks at the configured sample rate.

        """
        context_id = uuid.uuid4().hex
        url = self.build_ws_url()
        payload = self.build_payload(text, context_id=context_id)
        first_at: float | None = None
        last_at: float | None = None
        total_bytes = 0

        async with aiohttp.ClientSession() as session:
            ws = await self._ws_connect(session, url)
            try:
                await ws.send_json(payload)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        event_type = event.get("type")
                        if event_type == "chunk":
                            data = event.get("data")
                            if isinstance(data, str):
                                chunk = base64.b64decode(data)
                                if not chunk:
                                    continue
                                now = asyncio.get_event_loop().time()
                                if first_at is None:
                                    first_at = now
                                last_at = now
                                total_bytes += len(chunk)
                                yield chunk
                        elif event_type == "done":
                            break
                        elif event_type == "error":
                            err = event.get("error") or "unknown error"
                            status = event.get("status_code")
                            msg_txt = f"Cartesia TTS error (status={status}): {err}"
                            raise RuntimeError(msg_txt)
                        # timestamps event silently dropped
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

        first_to_last_ms = (last_at - first_at) * 1000 if first_at and last_at else 0
        _logger.info(
            f"{ls.tag('🔉 TTS', ls.C)} "
            f"{ls.word('Cartesia/stream', ls.C)} "
            f"{ls.kv('audio', f'{total_bytes}b', vc=ls.LW)} "
            f"{ls.kv('span_ms', f'{first_to_last_ms:.0f}', vc=ls.LW)}"
        )


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
            start_ms = word_timestamps[0].start_ms if word_timestamps else 0.0
            end_ms = word_timestamps[-1].end_ms if word_timestamps else 0.0
            _logger.info(
                f"{ls.tag('🔉 TTS', ls.C)} "
                f"{ls.word('Azure', ls.C)} "
                f"{ls.kv('words', str(len(word_timestamps)), vc=ls.LW)} "
                f"{ls.kv('audio', f'{len(audio)}b', vc=ls.LW)} "
                f"{ls.kv('timing', f'{start_ms:.0f}ms→{end_ms:.0f}ms', vc=ls.LW)}"
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


def _upsample_s16le_2x(audio: bytes) -> bytes:
    """Upsample 16-bit signed LE mono PCM by 2x via linear interpolation.

    Each original sample pair (a, b) produces (a, (a+b)//2). The last
    sample is doubled. Input length must be a multiple of 2 bytes.
    """
    if not audio:
        return b""
    n = len(audio) // 2
    samples = struct.unpack(f"<{n}h", audio)
    out: list[int] = []
    for i, s in enumerate(samples):
        out.append(s)
        nxt = samples[i + 1] if i + 1 < n else s
        out.append((s + nxt) // 2)
    return struct.pack(f"<{len(out)}h", *out)


def _estimate_word_timestamps(text: str, total_ms: float) -> list[WordTimestamp]:
    """Distribute *total_ms* uniformly across whitespace-split words in *text*.

    Returns an empty list when *text* has no words or *total_ms* is zero.
    """
    words = text.split()
    if not words or total_ms <= 0:
        return []
    per = total_ms / len(words)
    return [
        WordTimestamp(word=w, start_ms=i * per, end_ms=(i + 1) * per)
        for i, w in enumerate(words)
    ]


def _compose_gemini_style_prompt(cfg: TTSConfig) -> str | None:
    """Compose Gemini style prompt from structured ``[tts]`` fields.

    Returns ``None`` when all six style fields are unset. Output follows
    Audio Profile / Scene / Director's Notes structure.
    """
    parts: list[str] = []
    if cfg.gemini_audio_profile:
        parts.append(f"Audio Profile: {cfg.gemini_audio_profile}")
    scene_bits = [b for b in (cfg.gemini_scene, cfg.gemini_context) if b]
    if scene_bits:
        parts.append("Scene: " + " ".join(scene_bits))
    notes: list[str] = []
    if cfg.gemini_style:
        notes.append(f"Style: {cfg.gemini_style}.")
    if cfg.gemini_pace:
        notes.append(f"Pace: {cfg.gemini_pace}.")
    if cfg.gemini_accent:
        notes.append(f"Accent: {cfg.gemini_accent}.")
    if notes:
        parts.append("Director's Notes: " + " ".join(notes))
    if not parts:
        return None
    return "\n".join(parts) + "\nSay:"


class GeminiTTSClient:
    """Google Gemini Flash TTS client.

    Uses the ``google-genai`` SDK (blocking call run in a thread executor).
    Gemini returns 24 kHz mono PCM; we upsample 2x to 48 kHz to match the
    Discord pipeline. Word timestamps are estimated uniformly from total
    audio duration — Gemini does not expose per-word timing.
    """

    def __init__(
        self: Self,
        *,
        api_key: str,
        voice_name: str,
        model: str,
        style_prompt: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.api_key = api_key
        self.voice_name = voice_name
        self.model = model
        self.style_prompt = style_prompt
        self.sample_rate = sample_rate

    def _make_client(self: Self) -> _GenaiClient:
        """Return a ``google.genai.Client``; extracted for testability."""
        from google import genai  # noqa: PLC0415

        return genai.Client(api_key=self.api_key)

    def _synthesize_sync(self: Self, text: str) -> TTSResult:
        """Blocking synthesis call; returns :class:`TTSResult`.

        :raises RuntimeError: when Gemini returns no audio part.
        """
        from google.genai import types  # noqa: PLC0415

        contents = f"{self.style_prompt}\n\n{text}" if self.style_prompt else text

        client = self._make_client()
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    ),
                ),
            ),
        )
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate is not None else None
        parts = content.parts if content is not None else None
        part = parts[0] if parts else None
        inline = part.inline_data if part is not None else None
        pcm_24k_or_none = inline.data if inline is not None else None
        if pcm_24k_or_none is None:
            msg = "Gemini TTS returned no audio part"
            raise RuntimeError(msg)
        pcm_24k: bytes = pcm_24k_or_none
        audio = _upsample_s16le_2x(pcm_24k)

        # duration of the 48 kHz audio in ms (16-bit = 2 bytes/sample)
        total_ms = len(audio) / 2 / self.sample_rate * 1000.0
        timestamps = _estimate_word_timestamps(text, total_ms)

        _logger.info(
            f"{ls.tag('🔉 TTS', ls.C)} "
            f"{ls.word('Gemini', ls.C)} "
            f"{ls.kv('words', str(len(timestamps)), vc=ls.LW)} "
            f"{ls.kv('audio', f'{len(audio)}b', vc=ls.LW)} "
            f"{ls.kv('duration', f'{total_ms:.0f}ms', vc=ls.LW)}"
        )
        return TTSResult(audio=audio, timestamps=timestamps)

    async def synthesize(self: Self, text: str) -> TTSResult:
        """Synthesize *text* in a thread executor; return audio + timestamps."""
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
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        msg = (
            "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable "
            "is required for Gemini TTS"
        )
        raise ValueError(msg)
    return GeminiTTSClient(
        api_key=api_key,
        voice_name=tts_config.gemini_voice,
        model=tts_config.gemini_model,
        style_prompt=_compose_gemini_style_prompt(tts_config),
    )
