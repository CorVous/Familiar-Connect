"""Tests for the Cartesia TTS client (streaming WebSocket, word timestamps)."""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import aiohttp

if TYPE_CHECKING:
    from pathlib import Path

import familiar_connect.tts as tts_module
import pytest

from familiar_connect.tts import (
    CARTESIA_API_VERSION,
    CARTESIA_BASE_URL,
    CARTESIA_WS_URL,
    DEFAULT_SAMPLE_RATE,
    CartesiaTTSClient,
    TTSResult,
    WordTimestamp,
    create_tts_client,
    get_cached_greeting_audio,
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestWordTimestamp:
    def test_fields(self) -> None:
        ts = WordTimestamp(word="hello", start_ms=0.0, end_ms=420.0)
        assert ts.word == "hello"
        assert ts.start_ms == 0.0  # noqa: RUF069
        assert ts.end_ms == 420.0  # noqa: RUF069


class TestTTSResult:
    def test_audio_only_default_empty_timestamps(self) -> None:
        result = TTSResult(audio=b"\x00\x01")
        assert result.audio == b"\x00\x01"
        assert result.timestamps == []

    def test_with_timestamps(self) -> None:
        ts = [WordTimestamp("hi", 0.0, 200.0)]
        result = TTSResult(audio=b"\x00", timestamps=ts)
        assert result.timestamps == ts


# ---------------------------------------------------------------------------
# Client construction and payload shape
# ---------------------------------------------------------------------------


class TestCartesiaTTSClient:
    def test_init_stores_api_key(self) -> None:
        client = _client()
        assert client.api_key == "test-key"

    def test_init_stores_model(self) -> None:
        client = _client(model="sonic-3")
        assert client.model == "sonic-3"

    def test_init_stores_voice_id(self) -> None:
        client = _client(voice_id="custom-uuid-1234")
        assert client.voice_id == "custom-uuid-1234"

    def test_init_default_sample_rate(self) -> None:
        client = _client()
        assert client.sample_rate == DEFAULT_SAMPLE_RATE
        assert client.sample_rate == 48000

    def test_init_default_base_url(self) -> None:
        client = _client()
        assert client.base_url == CARTESIA_BASE_URL

    def test_init_default_ws_url(self) -> None:
        client = _client()
        assert client.ws_url == CARTESIA_WS_URL
        assert client.ws_url.startswith("wss://")

    def test_build_ws_url_includes_auth_query(self) -> None:
        client = _client(api_key="sk-cart-test-123")
        url = client.build_ws_url()
        assert url.startswith(CARTESIA_WS_URL + "?")
        assert "api_key=sk-cart-test-123" in url
        assert f"cartesia_version={CARTESIA_API_VERSION}" in url

    def test_build_headers_for_rest(self) -> None:
        client = _client(api_key="sk-cart-test-123")
        headers = client.build_headers()
        assert headers["X-API-Key"] == "sk-cart-test-123"
        assert headers["Cartesia-Version"] == CARTESIA_API_VERSION
        assert headers["Content-Type"] == "application/json"

    def test_payload_structure(self) -> None:
        client = _client()
        payload = client.build_payload("Hello, world!", context_id="ctx-1")
        assert payload["context_id"] == "ctx-1"
        assert payload["transcript"] == "Hello, world!"
        assert payload["model_id"] == _TEST_MODEL
        assert payload["voice"]["mode"] == "id"
        assert payload["voice"]["id"] == _TEST_VOICE_ID
        assert payload["add_timestamps"] is True
        assert payload["continue"] is False

    def test_payload_output_format(self) -> None:
        client = _client(sample_rate=22050)
        payload = client.build_payload("test", context_id="ctx")
        fmt = payload["output_format"]
        assert fmt["container"] == "raw"
        assert fmt["encoding"] == "pcm_s16le"
        assert fmt["sample_rate"] == 22050


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------


def _text_msg(obj: dict[str, Any]) -> Mock:
    msg = Mock(spec=aiohttp.WSMessage)
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(obj)
    return msg


class _FakeWS:
    """Mock aiohttp WebSocket that yields a pre-scripted list of events."""

    closed = False

    def __init__(self, events: list[Mock]) -> None:
        self._events = list(events)
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, obj: dict[str, Any]) -> None:
        self.sent.append(obj)

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> Mock:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def close(self) -> None:
        self.closed = True


def _fake_ws(events: list[Mock]) -> _FakeWS:
    """Build a mock aiohttp WebSocket that yields *events*."""
    return _FakeWS(events)


class TestCartesiaTTSClientSynthesize:
    @pytest.fixture
    def client(self) -> CartesiaTTSClient:
        return _client()

    @pytest.mark.asyncio
    async def test_returns_tts_result(self, client: CartesiaTTSClient) -> None:
        pcm = b"\x10\x20\x30\x40"
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(pcm).decode()}),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            result = await client.synthesize("Hello")
        assert isinstance(result, TTSResult)
        assert result.audio == pcm
        assert result.timestamps == []

    @pytest.mark.asyncio
    async def test_concatenates_multiple_chunks(
        self,
        client: CartesiaTTSClient,
    ) -> None:
        a = b"\xaa" * 4
        b = b"\xbb" * 4
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(a).decode()}),
            _text_msg({"type": "chunk", "data": base64.b64encode(b).decode()}),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            result = await client.synthesize("Hi")
        assert result.audio == a + b

    @pytest.mark.asyncio
    async def test_parses_word_timestamps_seconds_to_ms(
        self,
        client: CartesiaTTSClient,
    ) -> None:
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\x00").decode()}),
            _text_msg(
                {
                    "type": "timestamps",
                    "word_timestamps": {
                        "words": ["Hello", "world"],
                        "start": [0.0, 0.5],
                        "end": [0.42, 0.9],
                    },
                },
            ),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            result = await client.synthesize("Hello world")
        assert result.timestamps == [
            WordTimestamp("Hello", 0.0, 420.0),
            WordTimestamp("world", 500.0, 900.0),
        ]

    @pytest.mark.asyncio
    async def test_multiple_timestamp_events_accumulate(
        self,
        client: CartesiaTTSClient,
    ) -> None:
        events = [
            _text_msg(
                {
                    "type": "timestamps",
                    "word_timestamps": {
                        "words": ["A"],
                        "start": [0.0],
                        "end": [0.1],
                    },
                },
            ),
            _text_msg(
                {
                    "type": "timestamps",
                    "word_timestamps": {
                        "words": ["B"],
                        "start": [0.1],
                        "end": [0.2],
                    },
                },
            ),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            result = await client.synthesize("A B")
        assert [t.word for t in result.timestamps] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_sends_request_payload(self, client: CartesiaTTSClient) -> None:
        events = [_text_msg({"type": "done"})]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            await client.synthesize("Hello")
        assert len(fake_ws.sent) == 1
        payload = fake_ws.sent[0]
        assert payload["transcript"] == "Hello"
        assert payload["add_timestamps"] is True
        assert payload["continue"] is False
        assert "context_id" in payload

    @pytest.mark.asyncio
    async def test_error_event_raises(self, client: CartesiaTTSClient) -> None:
        events = [
            _text_msg(
                {"type": "error", "error": "voice id unknown", "status_code": 400},
            ),
        ]
        fake_ws = _fake_ws(events)
        with (
            patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)),
            pytest.raises(RuntimeError, match="voice id unknown"),
        ):
            await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_unexpected_close_raises(
        self,
        client: CartesiaTTSClient,
    ) -> None:
        closed_msg = Mock(spec=aiohttp.WSMessage)
        closed_msg.type = aiohttp.WSMsgType.CLOSED
        fake_ws = _fake_ws([closed_msg])
        with (
            patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)),
            pytest.raises(RuntimeError, match="closed unexpectedly"),
        ):
            await client.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_connect_failure_propagates(
        self,
        client: CartesiaTTSClient,
    ) -> None:
        with (
            patch.object(
                client,
                "_ws_connect",
                new=AsyncMock(
                    side_effect=aiohttp.ClientError("dns failure"),
                ),
            ),
            pytest.raises(aiohttp.ClientError, match="dns failure"),
        ):
            await client.synthesize("Hello")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTTSClient:
    def test_creates_client_from_character_config(self) -> None:
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}):
            client = create_tts_client(
                voice_id="some-voice-uuid",
                model="sonic-turbo",
            )
        assert client.api_key == "sk-cart-test-abc"
        assert client.voice_id == "some-voice-uuid"
        assert client.model == "sonic-turbo"

    def test_raises_when_api_key_missing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"CARTESIA_API_KEY"),
        ):
            create_tts_client(voice_id="v", model="m")

    def test_raises_when_voice_id_empty(self) -> None:
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"voice_id"),
        ):
            create_tts_client(voice_id="", model="sonic-3")

    def test_raises_when_model_empty(self) -> None:
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"model"),
        ):
            create_tts_client(voice_id="some-voice", model="")


class TestGreetingCache:
    """Tests for get_cached_greeting_audio file-based cache."""

    @pytest.fixture
    def fake_cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect _GREETING_CACHE_DIR to a temp directory for tests."""
        fake_dir = tmp_path / "greetings"
        monkeypatch.setattr(tts_module, "_GREETING_CACHE_DIR", fake_dir)
        return fake_dir

    @pytest.mark.asyncio
    async def test_cache_miss_synthesizes_and_writes_file(
        self,
        fake_cache_dir: Path,
    ) -> None:
        """First call synthesizes and writes audio to disk."""
        mock_client = AsyncMock(spec=CartesiaTTSClient)
        mock_result = TTSResult(audio=b"cached-audio")
        mock_client.synthesize.return_value = mock_result

        result = await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hello!",
            client=mock_client,
        )
        assert result.audio == b"cached-audio"
        mock_client.synthesize.assert_called_once_with("Hello!")
        # Verify file was created in the fake cache dir.
        audio_files = list(fake_cache_dir.glob("*.bin"))
        assert len(audio_files) == 1

    @pytest.mark.asyncio
    async def test_cache_hit_reads_file_without_synthesis(
        self,
        fake_cache_dir: Path,
    ) -> None:
        """Second call with same key reads from disk without calling synthesize."""
        mock_client = AsyncMock(spec=CartesiaTTSClient)
        mock_result = TTSResult(audio=b"cached-audio")
        mock_client.synthesize.return_value = mock_result

        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hello!",
            client=mock_client,
        )
        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hello!",
            client=mock_client,
        )
        # synthesize called only once
        assert mock_client.synthesize.call_count == 1

    @pytest.mark.asyncio
    async def test_different_voice_id_not_cached(
        self,
        fake_cache_dir: Path,
    ) -> None:
        """Different voice_id synthesizes separately."""
        mock_client = AsyncMock(spec=CartesiaTTSClient)
        mock_client.synthesize.return_value = TTSResult(audio=b"audio")

        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hello!",
            client=mock_client,
        )
        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-2",
            greeting="Hello!",
            client=mock_client,
        )
        assert mock_client.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_different_greeting_not_cached(
        self,
        fake_cache_dir: Path,
    ) -> None:
        """Different greeting text synthesizes separately."""
        mock_client = AsyncMock(spec=CartesiaTTSClient)
        mock_client.synthesize.return_value = TTSResult(audio=b"audio")

        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hello!",
            client=mock_client,
        )
        await get_cached_greeting_audio(
            provider="cartesia",
            voice_id="voice-1",
            greeting="Hi there!",
            client=mock_client,
        )
        assert mock_client.synthesize.call_count == 2
