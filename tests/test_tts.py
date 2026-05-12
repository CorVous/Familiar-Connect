"""Tests for TTS clients: Cartesia (WebSocket), Azure (Speech SDK), Gemini."""

from __future__ import annotations

import base64
import datetime
import json
import os
import struct
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp

if TYPE_CHECKING:
    from pathlib import Path

import familiar_connect.tts as tts_module  # noqa: I001
import pytest

from familiar_connect.config import (
    DEFAULT_AZURE_TTS_VOICE,
    TTSConfig,
)
from familiar_connect.tts import (
    CARTESIA_API_VERSION,
    CARTESIA_BASE_URL,
    CARTESIA_WS_URL,
    DEFAULT_AZURE_VOICE,
    DEFAULT_SAMPLE_RATE,
    GEMINI_SAMPLE_RATE,
    AzureTTSClient,
    CartesiaTTSClient,
    GeminiTTSClient,
    TTSResult,
    WordTimestamp,
    _compose_gemini_style_prompt,
    _estimate_word_timestamps,
    _upsample_s16le_2x,
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
    def test_init_stores_required_fields(self) -> None:
        """Constructor stores api_key, model, and voice_id."""
        client = _client(model="sonic-3", voice_id="custom-uuid-1234")
        assert client.api_key == "test-key"
        assert client.model == "sonic-3"
        assert client.voice_id == "custom-uuid-1234"

    def test_init_defaults(self) -> None:
        """Default sample_rate, base_url, and ws_url are set."""
        client = _client()
        assert client.sample_rate == DEFAULT_SAMPLE_RATE
        assert client.sample_rate == 48000
        assert client.base_url == CARTESIA_BASE_URL
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


class TestCartesiaTTSClientSynthesizeStream:
    """``synthesize_stream`` yields chunks as they arrive (no buffering).

    Lower-latency variant of ``synthesize``: enables byte-level playback
    so ``DiscordVoicePlayer`` can start ``vc.play`` on the first chunk
    instead of waiting for the full utterance.
    """

    @pytest.fixture
    def client(self) -> CartesiaTTSClient:
        return _client()

    @pytest.mark.asyncio
    async def test_yields_each_chunk_in_order(self, client: CartesiaTTSClient) -> None:
        chunks = [b"\x10\x20", b"\x30\x40", b"\x50\x60"]
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(c).decode()})
            for c in chunks
        ]
        events.append(_text_msg({"type": "done"}))
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            collected = [chunk async for chunk in client.synthesize_stream("hello")]
        assert collected == chunks

    @pytest.mark.asyncio
    async def test_empty_chunks_skipped(self, client: CartesiaTTSClient) -> None:
        events = [
            _text_msg({"type": "chunk", "data": ""}),
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\xab").decode()}),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            collected = [chunk async for chunk in client.synthesize_stream("hi")]
        assert collected == [b"\xab"]

    @pytest.mark.asyncio
    async def test_timestamps_events_silently_dropped(
        self, client: CartesiaTTSClient
    ) -> None:
        """Chunk consumers don't need timestamps; they should not break the iter."""
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\x01").decode()}),
            _text_msg({
                "type": "timestamps",
                "word_timestamps": {
                    "words": ["hi"],
                    "start": [0.0],
                    "end": [0.1],
                },
            }),
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\x02").decode()}),
            _text_msg({"type": "done"}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            collected = [chunk async for chunk in client.synthesize_stream("hi")]
        assert collected == [b"\x01", b"\x02"]

    @pytest.mark.asyncio
    async def test_error_event_raises(self, client: CartesiaTTSClient) -> None:
        events = [
            _text_msg({
                "type": "error",
                "error": "voice id unknown",
                "status_code": 400,
            }),
        ]
        fake_ws = _fake_ws(events)
        with (
            patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)),
            pytest.raises(RuntimeError, match="voice id unknown"),
        ):
            async for _ in client.synthesize_stream("hi"):
                pass

    @pytest.mark.asyncio
    async def test_unexpected_close_raises(self, client: CartesiaTTSClient) -> None:
        closed_msg = Mock(spec=aiohttp.WSMessage)
        closed_msg.type = aiohttp.WSMsgType.CLOSED
        fake_ws = _fake_ws([closed_msg])
        with (
            patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)),
            pytest.raises(RuntimeError, match="closed unexpectedly"),
        ):
            async for _ in client.synthesize_stream("hi"):
                pass

    @pytest.mark.asyncio
    async def test_sends_request_payload(self, client: CartesiaTTSClient) -> None:
        events = [_text_msg({"type": "done"})]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            async for _ in client.synthesize_stream("Hello"):
                pass
        assert len(fake_ws.sent) == 1
        payload = fake_ws.sent[0]
        assert payload["transcript"] == "Hello"
        assert payload["add_timestamps"] is True

    @pytest.mark.asyncio
    async def test_done_terminates_iteration(self, client: CartesiaTTSClient) -> None:
        """`done` ends the stream even if more events follow on the wire."""
        events = [
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\xaa").decode()}),
            _text_msg({"type": "done"}),
            _text_msg({"type": "chunk", "data": base64.b64encode(b"\xbb").decode()}),
        ]
        fake_ws = _fake_ws(events)
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=fake_ws)):
            collected = [chunk async for chunk in client.synthesize_stream("hi")]
        assert collected == [b"\xaa"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTTSClientCartesia:
    def test_creates_client_from_tts_config(self) -> None:
        cfg = TTSConfig(
            provider="cartesia",
            cartesia_voice_id="some-voice-uuid",
            cartesia_model="sonic-turbo",
        )
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}):
            client = create_tts_client(cfg)
        assert isinstance(client, CartesiaTTSClient)
        assert client.api_key == "sk-cart-test-abc"
        assert client.voice_id == "some-voice-uuid"
        assert client.model == "sonic-turbo"

    def test_raises_when_api_key_missing(self) -> None:
        cfg = TTSConfig(provider="cartesia", cartesia_voice_id="v", cartesia_model="m")
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"CARTESIA_API_KEY"),
        ):
            create_tts_client(cfg)

    def test_raises_when_voice_id_empty(self) -> None:
        cfg = TTSConfig(
            provider="cartesia", cartesia_voice_id="", cartesia_model="sonic-3"
        )
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"voice_id"),
        ):
            create_tts_client(cfg)

    def test_raises_when_model_empty(self) -> None:
        cfg = TTSConfig(
            provider="cartesia", cartesia_voice_id="some-voice", cartesia_model=""
        )
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"model"),
        ):
            create_tts_client(cfg)


class TestGreetingCache:
    """Tests for get_cached_greeting_audio file-based cache."""

    @pytest.fixture
    def fake_cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect _GREETING_CACHE_DIR to a temp directory for tests."""
        fake_dir = tmp_path / "greetings"
        monkeypatch.setattr(tts_module, "_GREETING_CACHE_DIR", fake_dir)
        return fake_dir

    @pytest.mark.usefixtures("fake_cache_dir")
    @pytest.mark.asyncio
    async def test_cache_miss_synthesizes_and_writes_file(self) -> None:
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

    @pytest.mark.usefixtures("fake_cache_dir")
    @pytest.mark.asyncio
    async def test_cache_hit_reads_file_without_synthesis(self) -> None:
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

    @pytest.mark.usefixtures("fake_cache_dir")
    @pytest.mark.asyncio
    async def test_different_voice_id_not_cached(self) -> None:
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

    @pytest.mark.usefixtures("fake_cache_dir")
    @pytest.mark.asyncio
    async def test_different_greeting_not_cached(self) -> None:
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


# ---------------------------------------------------------------------------
# Helpers for Azure SDK mocking
# ---------------------------------------------------------------------------

_AZURE_COMPLETED = object()  # sentinel for ResultReason.SynthesizingAudioCompleted


def _make_mock_speechsdk() -> MagicMock:
    """Return a mock ``azure.cognitiveservices.speech`` module."""
    sdk = MagicMock(name="speechsdk")
    sdk.ResultReason.SynthesizingAudioCompleted = _AZURE_COMPLETED
    sdk.SpeechSynthesisBoundaryType.Word = "Word"
    return sdk


def _make_fake_synth(
    audio_bytes: bytes,
    word_events: list[tuple[str, int, datetime.timedelta]],
    *,
    fail: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Return ``(mock_sdk, mock_synthesizer)`` for use with ``_make_synthesizer``.

    Each entry in *word_events* is
    ``(word_text, audio_offset_ticks, duration_timedelta)``.
    When *fail* is True the result reason is not SynthesizingAudioCompleted.
    """
    sdk = _make_mock_speechsdk()

    # Fake result
    result = MagicMock()
    result.audio_data = audio_bytes
    result.reason = _AZURE_COMPLETED if not fail else object()

    # Cancellation details for failure path
    cancel = MagicMock()
    cancel.reason = "Error"
    cancel.error_details = "something went wrong"
    sdk.CancellationDetails.from_result.return_value = cancel

    # Build boundary event mocks from word_events list
    boundary_events: list[MagicMock] = []
    for word_text, offset_ticks, duration_td in word_events:
        evt = MagicMock()
        evt.text = word_text
        evt.audio_offset = offset_ticks
        evt.duration = duration_td
        evt.boundary_type = "Word"  # matches sdk.SpeechSynthesisBoundaryType.Word
        boundary_events.append(evt)

    # Synthesizer: .synthesis_word_boundary.connect(cb) stores cb;
    # .speak_text_async(text).get() fires stored callbacks then returns result.
    callbacks: list[Any] = []

    synth = MagicMock()
    synth.synthesis_word_boundary.connect.side_effect = callbacks.append

    def _get() -> MagicMock:
        for evt in boundary_events:
            for cb in callbacks:
                cb(evt)
        return result

    synth.speak_text_async.return_value.get.side_effect = _get

    return sdk, synth


# ---------------------------------------------------------------------------
# Azure TTS client
# ---------------------------------------------------------------------------


class TestAzureTTSClientInit:
    def test_stores_required_fields(self) -> None:
        """Constructor stores subscription_key and region."""
        client = AzureTTSClient(subscription_key="sk-az", region="westus2")
        assert client.subscription_key == "sk-az"
        assert client.region == "westus2"

    def test_custom_voice_stored(self) -> None:
        client = AzureTTSClient(
            subscription_key="k",
            region="r",
            voice_name="en-US-JennyNeural",
        )
        assert client.voice_name == "en-US-JennyNeural"

    def test_defaults(self) -> None:
        """Default voice and sample_rate match module-level constants."""
        client = AzureTTSClient(subscription_key="k", region="r")
        assert client.voice_name == DEFAULT_AZURE_VOICE
        assert client.voice_name == DEFAULT_AZURE_TTS_VOICE
        assert client.sample_rate == DEFAULT_SAMPLE_RATE


class TestAzureTTSClientSynthesize:
    """Tests for _synthesize_sync logic via _make_synthesizer mock."""

    def _client(self) -> AzureTTSClient:
        return AzureTTSClient(subscription_key="sk-az", region="eastus")

    def test_returns_tts_result_with_audio(self) -> None:
        pcm = b"\x10\x20\x30\x40"
        sdk, synth = _make_fake_synth(pcm, [])
        client = self._client()
        with patch.object(client, "_make_synthesizer", return_value=(sdk, synth)):
            result = client._synthesize_sync("Hello")
        assert isinstance(result, TTSResult)
        assert result.audio == pcm

    def test_empty_timestamps_when_no_word_events(self) -> None:
        sdk, synth = _make_fake_synth(b"\x00", [])
        client = self._client()
        with patch.object(client, "_make_synthesizer", return_value=(sdk, synth)):
            result = client._synthesize_sync("Hi")
        assert result.timestamps == []

    def test_word_boundary_ticks_converted_to_ms(self) -> None:
        # audio_offset=100_000 ticks → 10ms; duration=50ms timedelta → end=60ms
        events = [
            ("Hello", 100_000, datetime.timedelta(milliseconds=50)),
            ("world", 700_000, datetime.timedelta(milliseconds=80)),
        ]
        sdk, synth = _make_fake_synth(b"\x00", events)
        client = self._client()
        with patch.object(client, "_make_synthesizer", return_value=(sdk, synth)):
            result = client._synthesize_sync("Hello world")
        assert result.timestamps == [
            WordTimestamp("Hello", 10.0, 60.0),
            WordTimestamp("world", 70.0, 150.0),
        ]

    def test_non_word_boundary_events_skipped(self) -> None:
        """Punctuation and sentence boundary events must be ignored."""
        sdk, synth = _make_fake_synth(b"\x00", [])
        client = self._client()

        # Inject a punctuation event directly by monkeypatching connect
        punct_evt = MagicMock()
        punct_evt.text = ","
        punct_evt.audio_offset = 50_000
        punct_evt.duration = datetime.timedelta(milliseconds=10)
        # differs from sdk.SpeechSynthesisBoundaryType.Word
        punct_evt.boundary_type = "Punctuation"

        captured: list[Any] = []
        synth.synthesis_word_boundary.connect.side_effect = captured.append

        def _get() -> MagicMock:
            for cb in captured:
                cb(punct_evt)
            r = MagicMock()
            r.reason = _AZURE_COMPLETED
            r.audio_data = b"\x00"
            return r

        synth.speak_text_async.return_value.get.side_effect = _get

        with patch.object(client, "_make_synthesizer", return_value=(sdk, synth)):
            result = client._synthesize_sync("Hello,")
        assert result.timestamps == []

    def test_raises_runtime_error_on_synthesis_failure(self) -> None:
        sdk, synth = _make_fake_synth(b"", [], fail=True)
        client = self._client()
        with (
            patch.object(client, "_make_synthesizer", return_value=(sdk, synth)),
            pytest.raises(RuntimeError, match="Azure TTS"),
        ):
            client._synthesize_sync("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_runs_in_executor(self) -> None:
        """synthesize() delegates to _synthesize_sync via run_in_executor."""
        expected = TTSResult(audio=b"\xab\xcd", timestamps=[])
        client = self._client()
        with patch.object(
            client,
            "_synthesize_sync",
            return_value=expected,
        ) as mock_sync:
            result = await client.synthesize("Hello")
        assert result is expected
        mock_sync.assert_called_once_with("Hello")


# ---------------------------------------------------------------------------
# Azure factory
# ---------------------------------------------------------------------------


class TestCreateTTSClientAzure:
    def test_creates_azure_client(self) -> None:
        cfg = TTSConfig(provider="azure", azure_voice="en-US-AmberNeural")
        with patch.dict(
            os.environ,
            {"AZURE_SPEECH_KEY": "az-key", "AZURE_SPEECH_REGION": "eastus"},
        ):
            client = create_tts_client(cfg)
        assert isinstance(client, AzureTTSClient)
        assert client.subscription_key == "az-key"
        assert client.region == "eastus"
        assert client.voice_name == "en-US-AmberNeural"

    def test_azure_client_uses_tts_config_voice(self) -> None:
        cfg = TTSConfig(provider="azure", azure_voice="en-US-JennyNeural")
        with patch.dict(
            os.environ,
            {"AZURE_SPEECH_KEY": "k", "AZURE_SPEECH_REGION": "westus2"},
        ):
            client = create_tts_client(cfg)
        assert isinstance(client, AzureTTSClient)
        assert client.voice_name == "en-US-JennyNeural"

    def test_raises_when_azure_key_missing(self) -> None:
        cfg = TTSConfig(provider="azure")
        with (
            patch.dict(os.environ, {"AZURE_SPEECH_REGION": "eastus"}, clear=True),
            pytest.raises(ValueError, match=r"AZURE_SPEECH_KEY"),
        ):
            create_tts_client(cfg)

    def test_raises_when_azure_region_missing(self) -> None:
        cfg = TTSConfig(provider="azure")
        with (
            patch.dict(os.environ, {"AZURE_SPEECH_KEY": "k"}, clear=True),
            pytest.raises(ValueError, match=r"AZURE_SPEECH_REGION"),
        ):
            create_tts_client(cfg)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _make_s16le(samples: list[int]) -> bytes:
    """Pack a list of s16le sample values into raw PCM bytes."""
    return struct.pack(f"<{len(samples)}h", *samples)


class TestUpsampleS16le2x:
    def test_doubles_length(self) -> None:
        audio = _make_s16le([100, 200])
        result = _upsample_s16le_2x(audio)
        assert len(result) == len(audio) * 2

    def test_first_sample_preserved(self) -> None:
        audio = _make_s16le([1000, 2000])
        result = _upsample_s16le_2x(audio)
        samples = list(struct.unpack(f"<{len(result) // 2}h", result))
        assert samples[0] == 1000

    def test_interpolates_midpoint(self) -> None:
        audio = _make_s16le([0, 2000])
        result = _upsample_s16le_2x(audio)
        samples = list(struct.unpack(f"<{len(result) // 2}h", result))
        # interp between 0 and 2000 → 1000
        assert samples[1] == 1000

    def test_last_sample_doubled(self) -> None:
        audio = _make_s16le([500])
        result = _upsample_s16le_2x(audio)
        samples = list(struct.unpack(f"<{len(result) // 2}h", result))
        assert samples == [500, 500]

    def test_empty_input(self) -> None:
        assert _upsample_s16le_2x(b"") == b""


class TestEstimateWordTimestamps:
    def test_empty_text_returns_empty(self) -> None:
        assert _estimate_word_timestamps("", 1000.0) == []

    def test_zero_duration_returns_empty(self) -> None:
        assert _estimate_word_timestamps("hello world", 0.0) == []

    def test_single_word_spans_full_duration(self) -> None:
        result = _estimate_word_timestamps("hello", 500.0)
        assert len(result) == 1
        assert result[0].word == "hello"
        assert result[0].start_ms == pytest.approx(0.0)
        assert result[0].end_ms == pytest.approx(500.0)

    def test_uniform_distribution(self) -> None:
        result = _estimate_word_timestamps("one two three four", 400.0)
        assert len(result) == 4
        assert result[0].start_ms == pytest.approx(0.0)
        assert result[0].end_ms == pytest.approx(100.0)
        assert result[1].start_ms == pytest.approx(100.0)
        assert result[3].end_ms == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Gemini style-prompt composer
# ---------------------------------------------------------------------------


class TestComposeGeminiStylePrompt:
    def test_returns_none_when_all_fields_empty(self) -> None:
        cfg = TTSConfig(provider="gemini")
        assert _compose_gemini_style_prompt(cfg) is None

    def test_audio_profile_only(self) -> None:
        cfg = TTSConfig(provider="gemini", gemini_audio_profile="warm contralto")
        result = _compose_gemini_style_prompt(cfg)
        assert result is not None
        assert "Audio Profile: warm contralto" in result
        assert "Scene:" not in result
        assert "Director" not in result
        assert result.endswith("\nSay:")

    def test_scene_and_context_joined(self) -> None:
        cfg = TTSConfig(
            provider="gemini",
            gemini_scene="quiet room",
            gemini_context="tavern keeper",
        )
        result = _compose_gemini_style_prompt(cfg)
        assert result is not None
        assert "Scene: quiet room tavern keeper" in result

    def test_scene_only(self) -> None:
        cfg = TTSConfig(provider="gemini", gemini_scene="foggy docks")
        result = _compose_gemini_style_prompt(cfg)
        assert result is not None
        assert "Scene: foggy docks" in result

    def test_director_notes_joined(self) -> None:
        cfg = TTSConfig(
            provider="gemini",
            gemini_style="playful",
            gemini_pace="relaxed",
            gemini_accent="Irish lilt",
        )
        result = _compose_gemini_style_prompt(cfg)
        assert result is not None
        assert "Director's Notes:" in result
        assert "Style: playful." in result
        assert "Pace: relaxed." in result
        assert "Accent: Irish lilt." in result

    def test_all_fields_ends_with_say(self) -> None:
        cfg = TTSConfig(
            provider="gemini",
            gemini_audio_profile="old wizard",
            gemini_scene="dark tower",
            gemini_context="foreboding",
            gemini_style="gravelly",
            gemini_pace="slow",
            gemini_accent="British",
        )
        result = _compose_gemini_style_prompt(cfg)
        assert result is not None
        assert result.endswith("\nSay:")
        assert "Audio Profile:" in result
        assert "Scene:" in result
        assert "Director's Notes:" in result


# ---------------------------------------------------------------------------
# GeminiTTSClient
# ---------------------------------------------------------------------------


def _make_gemini_mock(pcm_24k: bytes) -> MagicMock:
    """Return a mock google-genai client that yields *pcm_24k* audio bytes."""
    part = MagicMock()
    part.inline_data.data = pcm_24k
    part.inline_data.mime_type = f"audio/L16;codec=pcm;rate={GEMINI_SAMPLE_RATE}"

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


class TestGeminiTTSClientInit:
    def test_stores_fields(self) -> None:
        c = GeminiTTSClient(api_key="k", voice_name="Kore", model="m")
        assert c.api_key == "k"
        assert c.voice_name == "Kore"
        assert c.model == "m"
        assert c.style_prompt is None
        assert c.sample_rate == DEFAULT_SAMPLE_RATE

    def test_stores_style_prompt(self) -> None:
        c = GeminiTTSClient(
            api_key="k",
            voice_name="Puck",
            model="m",
            style_prompt="Audio Profile: wizard",
        )
        assert c.style_prompt == "Audio Profile: wizard"


class TestGeminiTTSClientSynthesize:
    def _client(self, *, style_prompt: str | None = None) -> GeminiTTSClient:
        return GeminiTTSClient(
            api_key="test-key",
            voice_name="Kore",
            model="m",
            style_prompt=style_prompt,
        )

    @pytest.mark.asyncio
    async def test_returns_upsampled_audio_and_timestamps(self) -> None:
        pcm_24k = _make_s16le([100, 200, 300, 400])  # 4 samples @ 24 kHz
        mock_client = _make_gemini_mock(pcm_24k)
        client = self._client()
        with patch.object(client, "_make_client", return_value=mock_client):
            result = await client.synthesize("hello world")
        # audio should be 2x length (upsampled to 48 kHz)
        assert len(result.audio) == len(pcm_24k) * 2
        assert isinstance(result, TTSResult)

    @pytest.mark.asyncio
    async def test_timestamps_cover_original_words(self) -> None:
        pcm_24k = _make_s16le([0] * 96000)  # 2 s @ 24 kHz (16-bit = 2 bytes/sample)
        mock_client = _make_gemini_mock(pcm_24k)
        client = self._client()
        with patch.object(client, "_make_client", return_value=mock_client):
            result = await client.synthesize("one two three")
        assert len(result.timestamps) == 3
        assert result.timestamps[0].word == "one"
        assert result.timestamps[2].word == "three"
        # spans should cover full duration approximately
        assert result.timestamps[0].start_ms == pytest.approx(0.0)
        assert result.timestamps[-1].end_ms > 0.0

    @pytest.mark.asyncio
    async def test_prepends_style_prompt_when_set(self) -> None:
        pcm_24k = _make_s16le([0] * 4)
        mock_client = _make_gemini_mock(pcm_24k)
        client = self._client(style_prompt="Audio Profile: wizard\nSay:")
        with patch.object(client, "_make_client", return_value=mock_client):
            await client.synthesize("hello there")
        call_kwargs = mock_client.models.generate_content.call_args
        contents_arg = call_kwargs[1].get("contents") or call_kwargs[0][1]
        assert "Audio Profile: wizard" in contents_arg
        assert "hello there" in contents_arg

    @pytest.mark.asyncio
    async def test_no_style_prompt_passes_text_unchanged(self) -> None:
        pcm_24k = _make_s16le([0] * 4)
        mock_client = _make_gemini_mock(pcm_24k)
        client = self._client()
        with patch.object(client, "_make_client", return_value=mock_client):
            await client.synthesize("just the text")
        call_kwargs = mock_client.models.generate_content.call_args
        contents_arg = call_kwargs[1].get("contents") or call_kwargs[0][1]
        assert contents_arg == "just the text"

    @pytest.mark.asyncio
    async def test_timestamps_from_original_text_not_prompt(self) -> None:
        pcm_24k = _make_s16le([0] * 4)
        mock_client = _make_gemini_mock(pcm_24k)
        client = self._client(style_prompt="Audio Profile: quiet\nSay:")
        with patch.object(client, "_make_client", return_value=mock_client):
            result = await client.synthesize("alpha beta")
        words = [ts.word for ts in result.timestamps]
        assert words == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_raises_when_no_candidates(self) -> None:
        response = MagicMock()
        response.candidates = []
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        client = self._client()
        with (
            patch.object(client, "_make_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="no audio"),
        ):
            await client.synthesize("hello")

    @pytest.mark.asyncio
    async def test_raises_when_no_parts(self) -> None:
        candidate = MagicMock()
        candidate.content.parts = []
        response = MagicMock()
        response.candidates = [candidate]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        client = self._client()
        with (
            patch.object(client, "_make_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="no audio"),
        ):
            await client.synthesize("hello")


# ---------------------------------------------------------------------------
# Gemini factory
# ---------------------------------------------------------------------------


class TestCreateTTSClientGemini:
    def test_raises_when_no_api_key(self) -> None:
        cfg = TTSConfig(provider="gemini")
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"GOOGLE_API_KEY"),
        ):
            create_tts_client(cfg)

    def test_accepts_gemini_api_key_alias(self) -> None:
        cfg = TTSConfig(provider="gemini")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g-key"}, clear=True):
            client = create_tts_client(cfg)
        assert isinstance(client, GeminiTTSClient)
        assert client.api_key == "g-key"

    def test_creates_gemini_client_from_tts_config(self) -> None:
        cfg = TTSConfig(
            provider="gemini",
            gemini_voice="Puck",
            gemini_model="gemini-3.1-flash-tts-preview",
        )
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "goog-key"}, clear=True):
            client = create_tts_client(cfg)
        assert isinstance(client, GeminiTTSClient)
        assert client.api_key == "goog-key"
        assert client.voice_name == "Puck"
        assert client.model == "gemini-3.1-flash-tts-preview"
        assert client.style_prompt is None

    def test_composes_style_prompt_from_config_fields(self) -> None:
        cfg = TTSConfig(
            provider="gemini",
            gemini_audio_profile="warm narrator",
            gemini_style="calm",
        )
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "k"}, clear=True):
            client = create_tts_client(cfg)
        assert isinstance(client, GeminiTTSClient)
        assert client.style_prompt is not None
        assert "warm narrator" in client.style_prompt
        assert "calm" in client.style_prompt

    def test_google_api_key_preferred_over_alias(self) -> None:
        cfg = TTSConfig(provider="gemini")
        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "primary", "GEMINI_API_KEY": "alias"},
            clear=True,
        ):
            client = create_tts_client(cfg)
        assert isinstance(client, GeminiTTSClient)
        assert client.api_key == "primary"
