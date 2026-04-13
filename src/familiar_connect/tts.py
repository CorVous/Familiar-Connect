"""TTS client for synthesizing speech via Cartesia."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from typing import Any, Self


CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_API_VERSION = "2024-06-10"
DEFAULT_SAMPLE_RATE = 48000  # matches Discord's native rate


class CartesiaTTSClient:
    """Cartesia TTS client. Returns raw 16-bit signed mono PCM."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        voice_id: str,
        model: str,
        base_url: str = CARTESIA_BASE_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.base_url = base_url
        self.sample_rate = sample_rate

    def build_headers(self: Self) -> dict[str, str]:
        """HTTP headers for Cartesia API."""
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": CARTESIA_API_VERSION,
            "Content-Type": "application/json",
        }

    def build_payload(self: Self, text: str) -> dict[str, Any]:
        """JSON payload for TTS synthesis."""
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
        """Return raw mono PCM bytes for *text*.

        :raises httpx.HTTPStatusError: On non-2xx response.
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
