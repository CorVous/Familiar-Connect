"""TTS client for synthesizing speech via Cartesia."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

# ---------------------------------------------------------------------------
# Data types for word-level timestamps (used by interruption system)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordTimestamp:
    """A single word's timing from TTS synthesis.

    :param word: The word spoken.
    :param start_ms: Milliseconds from audio start when the word begins.
    :param end_ms: Milliseconds from audio start when the word ends.
    """

    word: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TTSResult:
    """TTS synthesis result with audio and word-level timestamps.

    :param audio: Raw PCM mono audio bytes.
    :param timestamps: Ordered word-level timestamps from the TTS engine.
    """

    audio: bytes
    timestamps: list[WordTimestamp]


if TYPE_CHECKING:
    from typing import Any, Self


CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_API_VERSION = "2024-06-10"
DEFAULT_MODEL = "sonic-3"
DEFAULT_VOICE_ID = "999df508-4de5-40a7-8bd3-8c12f678c284"
DEFAULT_SAMPLE_RATE = 48000  # Hz — matches Discord's native rate


class CartesiaTTSClient:
    """Client for synthesizing speech via Cartesia's TTS API.

    Requests raw 16-bit signed PCM audio at the configured sample rate.
    The bytes returned by :meth:`synthesize` are mono PCM and can be
    converted to stereo for Discord playback via :mod:`familiar_connect.voice.audio`.
    """

    def __init__(
        self: Self,
        *,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        base_url: str = CARTESIA_BASE_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.base_url = base_url
        self.sample_rate = sample_rate

    def build_headers(self: Self) -> dict[str, str]:
        """Build HTTP headers for the Cartesia API request."""
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": CARTESIA_API_VERSION,
            "Content-Type": "application/json",
        }

    def build_payload(self: Self, text: str) -> dict[str, Any]:
        """Build the JSON payload for a TTS synthesis request."""
        return {
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
        }

    async def _post(
        self: Self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60.0) as http:
            return await http.post(url, headers=headers, json=payload)

    async def synthesize(self: Self, text: str) -> bytes:
        """Synthesize *text* to speech and return raw mono PCM bytes.

        The returned bytes are 16-bit signed little-endian PCM at
        :attr:`sample_rate` Hz (mono).

        :param text: The text to synthesize.
        :return: Raw PCM audio bytes.
        :raises httpx.HTTPStatusError: On a non-2xx response from Cartesia.
        """
        url = f"{self.base_url}/tts/bytes"
        headers = self.build_headers()
        payload = self.build_payload(text)

        response = await self._post(url, headers, payload)
        if not response.is_success:
            body = response.text
            msg = f"Cartesia TTS request failed ({response.status_code}): {body}"
            raise httpx.HTTPStatusError(
                msg, request=response.request, response=response
            )
        return response.content


def create_tts_client_from_env() -> CartesiaTTSClient:
    """Create a :class:`CartesiaTTSClient` from environment variables.

    Required: ``CARTESIA_API_KEY``
    Optional: ``CARTESIA_VOICE_ID``, ``CARTESIA_MODEL``

    :raises ValueError: If ``CARTESIA_API_KEY`` is not set.
    """
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        msg = "CARTESIA_API_KEY environment variable is required"
        raise ValueError(msg)

    voice_id = os.environ.get("CARTESIA_VOICE_ID") or DEFAULT_VOICE_ID
    model = os.environ.get("CARTESIA_MODEL") or DEFAULT_MODEL

    return CartesiaTTSClient(api_key=api_key, voice_id=voice_id, model=model)
