"""Tests for TTS clients: Cartesia (WebSocket) and Azure (Speech SDK)."""

from __future__ import annotations

import base64
import datetime
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from familiar_connect.config import DEFAULT_AZURE_TTS_VOICE, TTSConfig
from familiar_connect.tts import (
    CARTESIA_API_VERSION,
    CARTESIA_BASE_URL,
    CARTESIA_WS_URL,
    DEFAULT_AZURE_VOICE,
    DEFAULT_SAMPLE_RATE,
    AzureTTSClient,
    CartesiaTTSClient,
    TTSResult,
    WordTimestamp,
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


class TestCreateTTSClientCartesia:
    def test_creates_client_from_tts_config(self) -> None:
        cfg = TTSConfig(
            provider="cartesia", voice_id="some-voice-uuid", model="sonic-turbo"
        )
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}):
            client = create_tts_client(cfg)
        assert isinstance(client, CartesiaTTSClient)
        assert client.api_key == "sk-cart-test-abc"
        assert client.voice_id == "some-voice-uuid"
        assert client.model == "sonic-turbo"

    def test_raises_when_api_key_missing(self) -> None:
        cfg = TTSConfig(provider="cartesia", voice_id="v", model="m")
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"CARTESIA_API_KEY"),
        ):
            create_tts_client(cfg)

    def test_raises_when_voice_id_empty(self) -> None:
        cfg = TTSConfig(provider="cartesia", voice_id="", model="sonic-3")
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"voice_id"),
        ):
            create_tts_client(cfg)

    def test_raises_when_model_empty(self) -> None:
        cfg = TTSConfig(provider="cartesia", voice_id="some-voice", model="")
        with (
            patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-cart-test-abc"}),
            pytest.raises(ValueError, match=r"model"),
        ):
            create_tts_client(cfg)


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
    def test_stores_subscription_key(self) -> None:
        client = AzureTTSClient(subscription_key="sk-az", region="eastus")
        assert client.subscription_key == "sk-az"

    def test_stores_region(self) -> None:
        client = AzureTTSClient(subscription_key="sk-az", region="westus2")
        assert client.region == "westus2"

    def test_default_voice_is_amber_neural(self) -> None:
        client = AzureTTSClient(subscription_key="sk-az", region="eastus")
        assert client.voice_name == DEFAULT_AZURE_VOICE
        assert client.voice_name == DEFAULT_AZURE_TTS_VOICE

    def test_custom_voice_stored(self) -> None:
        client = AzureTTSClient(
            subscription_key="k",
            region="r",
            voice_name="en-US-JennyNeural",
        )
        assert client.voice_name == "en-US-JennyNeural"

    def test_default_sample_rate(self) -> None:
        client = AzureTTSClient(subscription_key="k", region="r")
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
# TTS_PROVIDER env var override
# ---------------------------------------------------------------------------


class TestTTSProviderEnvOverride:
    """TTS_PROVIDER env var overrides [tts].provider from character.toml."""

    def test_env_override_selects_azure_over_cartesia_toml(self) -> None:
        cfg = TTSConfig(provider="cartesia")  # TOML says cartesia
        with patch.dict(
            os.environ,
            {
                "TTS_PROVIDER": "azure",
                "AZURE_SPEECH_KEY": "az-key",
                "AZURE_SPEECH_REGION": "eastus",
            },
        ):
            client = create_tts_client(cfg)
        assert isinstance(client, AzureTTSClient)

    def test_env_override_selects_cartesia_over_azure_toml(self) -> None:
        # TOML says azure, but TTS_PROVIDER=cartesia wins
        with patch.dict(
            os.environ,
            {
                "TTS_PROVIDER": "cartesia",
                "CARTESIA_API_KEY": "cart-key",
            },
        ):
            client = create_tts_client(
                TTSConfig(provider="azure", voice_id="v", model="sonic-3"),
            )
        assert isinstance(client, CartesiaTTSClient)

    def test_env_override_invalid_provider_raises(self) -> None:
        with (
            patch.dict(os.environ, {"TTS_PROVIDER": "fish"}),
            pytest.raises(ValueError, match=r"TTS_PROVIDER"),
        ):
            create_tts_client(TTSConfig(provider="azure"))

    def test_unset_env_uses_toml_provider(self) -> None:
        """No TTS_PROVIDER set → falls through to tts_config.provider."""
        cfg = TTSConfig(provider="azure", azure_voice="en-US-AmberNeural")
        env = {"AZURE_SPEECH_KEY": "k", "AZURE_SPEECH_REGION": "eastus"}
        # Explicitly exclude TTS_PROVIDER
        base = {k: v for k, v in os.environ.items() if k != "TTS_PROVIDER"}
        with patch.dict(os.environ, {**base, **env}, clear=True):
            client = create_tts_client(cfg)
        assert isinstance(client, AzureTTSClient)
