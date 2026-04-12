"""Tests for the Cartesia TTS client."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from familiar_connect.tts import (
    CARTESIA_API_VERSION,
    CARTESIA_BASE_URL,
    DEFAULT_SAMPLE_RATE,
    CartesiaTTSClient,
    create_tts_client,
)

_TEST_VOICE_ID = "test-voice-id"
_TEST_MODEL = "sonic-3"


def _client(
    *,
    api_key: str = "test-key",
    voice_id: str = _TEST_VOICE_ID,
    model: str = _TEST_MODEL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> CartesiaTTSClient:
    return CartesiaTTSClient(
        api_key=api_key,
        voice_id=voice_id,
        model=model,
        sample_rate=sample_rate,
    )


class TestCartesiaTTSClient:
    def test_init_stores_api_key(self) -> None:
        """Client stores the provided API key."""
        client = _client()
        assert client.api_key == "test-key"

    def test_init_stores_model(self) -> None:
        """Client stores the provided model."""
        client = _client(model="sonic-3")
        assert client.model == "sonic-3"

    def test_init_stores_voice_id(self) -> None:
        """Client stores the provided voice ID."""
        client = _client(voice_id="custom-uuid-1234")
        assert client.voice_id == "custom-uuid-1234"

    def test_init_default_sample_rate(self) -> None:
        """Client defaults to 48000 Hz (Discord native rate)."""
        client = _client()
        assert client.sample_rate == DEFAULT_SAMPLE_RATE
        assert client.sample_rate == 48000

    def test_init_custom_model(self) -> None:
        """Client accepts a custom model override."""
        client = _client(model="sonic-turbo")
        assert client.model == "sonic-turbo"

    def test_init_default_base_url(self) -> None:
        """Client defaults to the Cartesia API URL."""
        client = _client()
        assert client.base_url == CARTESIA_BASE_URL
        assert "cartesia.ai" in client.base_url

    def test_builds_request_headers(self) -> None:
        """Headers include X-API-Key, Cartesia-Version, and Content-Type."""
        client = _client(api_key="sk-cartesia-test-123")
        headers = client.build_headers()
        assert headers["X-API-Key"] == "sk-cartesia-test-123"
        assert headers["Cartesia-Version"] == CARTESIA_API_VERSION
        assert headers["Content-Type"] == "application/json"

    def test_builds_payload_structure(self) -> None:
        """Payload has model_id, transcript, voice, and output_format keys."""
        client = _client()
        payload = client.build_payload("Hello, world!")
        assert "model_id" in payload
        assert payload["transcript"] == "Hello, world!"
        assert "voice" in payload
        assert "output_format" in payload

    def test_payload_uses_raw_pcm_s16le(self) -> None:
        """Output format uses raw container and pcm_s16le encoding."""
        client = _client()
        payload = client.build_payload("test")
        fmt = payload["output_format"]
        assert fmt["container"] == "raw"
        assert fmt["encoding"] == "pcm_s16le"

    def test_payload_uses_configured_sample_rate(self) -> None:
        """Sample rate in the payload matches the client's sample_rate."""
        client = _client(sample_rate=22050)
        payload = client.build_payload("test")
        assert payload["output_format"]["sample_rate"] == 22050

    def test_payload_uses_configured_voice_id(self) -> None:
        """Voice ID in the payload matches the client's voice_id."""
        client = _client(voice_id="my-voice-abc")
        payload = client.build_payload("test")
        assert payload["voice"]["id"] == "my-voice-abc"

    def test_payload_voice_uses_id_mode(self) -> None:
        """Voice section uses mode=id for voice selection."""
        client = _client()
        payload = client.build_payload("test")
        assert payload["voice"]["mode"] == "id"


class TestCartesiaTTSClientSynthesize:
    @pytest.fixture
    def client(self) -> CartesiaTTSClient:
        return _client()

    def _make_mock_response(
        self,
        content: bytes = b"\x00\x01\x02\x03",
        *,
        status_code: int = 200,
        raise_error: Exception | None = None,
    ) -> Mock:
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = status_code
        mock_resp.content = content
        if raise_error:
            mock_resp.raise_for_status.side_effect = raise_error
        return mock_resp

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self, client: CartesiaTTSClient) -> None:
        """synthesize() returns the raw bytes from the API response."""
        pcm_bytes = b"\x10\x20\x30\x40"
        mock_resp = self._make_mock_response(content=pcm_bytes)

        with patch.object(client, "_post", new=AsyncMock(return_value=mock_resp)):
            result = await client.synthesize("Hello")

        assert result == pcm_bytes

    @pytest.mark.asyncio
    async def test_synthesize_calls_correct_endpoint(
        self, client: CartesiaTTSClient
    ) -> None:
        """synthesize() posts to {base_url}/tts/bytes."""
        mock_resp = self._make_mock_response()
        captured_url: list[str] = []

        def fake_post(url: str, _headers: dict, _payload: dict) -> Mock:
            captured_url.append(url)
            return mock_resp

        with patch.object(client, "_post", side_effect=fake_post):
            await client.synthesize("Hello")

        assert len(captured_url) == 1
        assert captured_url[0] == f"{CARTESIA_BASE_URL}/tts/bytes"

    @pytest.mark.asyncio
    async def test_synthesize_raises_on_http_error_with_body(
        self, client: CartesiaTTSClient
    ) -> None:
        """synthesize() raises HTTPStatusError including the response body."""
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.is_success = False
        mock_resp.text = "Invalid request: voice ID must not be empty"
        mock_resp.request = httpx.Request("POST", f"{CARTESIA_BASE_URL}/tts/bytes")

        with (
            patch.object(client, "_post", new=AsyncMock(return_value=mock_resp)),
            pytest.raises(httpx.HTTPStatusError, match=r"voice ID must not be empty"),
        ):
            await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_raises_on_5xx(self, client: CartesiaTTSClient) -> None:
        """synthesize() raises HTTPStatusError on a 5xx response.

        Pins the contract the bot.py TTS callers rely on: the client
        raises, the caller swallows. Any future refactor that turns
        this into a silent return would break the tests that expect
        TTS failures to be logged via the bot-layer try/except.
        """
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 503
        mock_resp.is_success = False
        mock_resp.text = "upstream voice model unavailable"
        mock_resp.request = httpx.Request("POST", f"{CARTESIA_BASE_URL}/tts/bytes")

        with (
            patch.object(client, "_post", new=AsyncMock(return_value=mock_resp)),
            pytest.raises(httpx.HTTPStatusError, match=r"503"),
        ):
            await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_raises_on_timeout(
        self, client: CartesiaTTSClient
    ) -> None:
        """A transport-level ``ReadTimeout`` propagates to the caller unchanged.

        The internal ``_post`` helper uses ``httpx.AsyncClient`` with a
        60 s timeout; when that fires, the exception is raised straight
        through. This test pins that behaviour so bot.py's TTS
        ``try/except`` clauses have a tested guarantee to anchor to.
        """
        with (
            patch.object(
                client,
                "_post",
                new=AsyncMock(side_effect=httpx.ReadTimeout("deadline exceeded")),
            ),
            pytest.raises(httpx.ReadTimeout, match=r"deadline exceeded"),
        ):
            await client.synthesize("Hello")


class TestCreateTTSClient:
    def test_creates_client_from_character_config(self) -> None:
        """Factory reads CARTESIA_API_KEY and takes voice/model as params."""
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}):
            client = create_tts_client(
                voice_id="some-voice-uuid",
                model="sonic-turbo",
            )

        assert client.api_key == "sk-cart-test-abc"
        assert client.voice_id == "some-voice-uuid"
        assert client.model == "sonic-turbo"

    def test_raises_when_api_key_missing(self) -> None:
        """Factory raises a clear error when CARTESIA_API_KEY is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"CARTESIA_API_KEY"),
        ):
            create_tts_client(voice_id="v", model="m")

    def test_raises_when_voice_id_empty(self) -> None:
        """Factory raises when voice_id is empty (required character.toml field)."""
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"voice_id"),
        ):
            create_tts_client(voice_id="", model="sonic-3")

    def test_raises_when_model_empty(self) -> None:
        """Factory raises when model is empty (required character.toml field)."""
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"model"),
        ):
            create_tts_client(voice_id="some-voice", model="")
