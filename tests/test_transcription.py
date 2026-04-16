"""Tests for the Deepgram streaming transcription module."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from familiar_connect.transcription import (
    DEEPGRAM_WS_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DeepgramTranscriber,
    TranscriptionEvent,
    TranscriptionResult,
    create_transcriber_from_env,
)


class TestTranscriptionResult:
    def test_dataclass_fields(self) -> None:
        """TranscriptionResult has all expected fields."""
        result = TranscriptionResult(
            text="hello world",
            is_final=True,
            start=0.0,
            end=1.5,
            confidence=0.98,
            speaker=0,
        )
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.start == pytest.approx(0.0)
        assert result.end == pytest.approx(1.5)
        assert result.confidence == pytest.approx(0.98)
        assert result.speaker == 0

    def test_default_values(self) -> None:
        """Speaker defaults to None, confidence defaults to 0.0."""
        result = TranscriptionResult(
            text="test",
            is_final=False,
            start=0.0,
            end=0.5,
        )
        assert result.speaker is None
        assert result.confidence == pytest.approx(0.0)

    def test_to_message(self) -> None:
        """to_message returns a Message with [Voice] prefix."""
        result = TranscriptionResult(
            text="hello there",
            is_final=True,
            start=0.0,
            end=1.0,
        )
        msg = result.to_message()
        assert msg.role == "user"
        assert msg.content == "[Voice] hello there"
        assert msg.name == "Voice"

    def test_to_message_with_speaker_names(self) -> None:
        """to_message uses speaker name from mapping when available."""
        result = TranscriptionResult(
            text="how are you",
            is_final=True,
            start=0.0,
            end=1.0,
            speaker=2,
        )
        names = {2: "Alice", 5: "Bob"}
        msg = result.to_message(speaker_names=names)
        assert msg.name == "Alice"
        assert msg.content == "[Voice] how are you"

    def test_to_message_with_unknown_speaker(self) -> None:
        """to_message falls back to 'Voice' for unmapped speaker IDs."""
        result = TranscriptionResult(
            text="hi",
            is_final=True,
            start=0.0,
            end=0.5,
            speaker=99,
        )
        names = {2: "Alice"}
        msg = result.to_message(speaker_names=names)
        assert msg.name == "Voice"


class TestDeepgramTranscriber:
    def test_init_stores_api_key(self) -> None:
        """Client stores the provided API key."""
        client = DeepgramTranscriber(api_key="test-key")
        assert client.api_key == "test-key"

    def test_init_default_model(self) -> None:
        """Client defaults to nova-3."""
        client = DeepgramTranscriber(api_key="test-key")
        assert client.model == DEFAULT_MODEL
        assert client.model == "nova-3"

    def test_init_default_language(self) -> None:
        """Client defaults to English."""
        client = DeepgramTranscriber(api_key="test-key")
        assert client.language == DEFAULT_LANGUAGE
        assert client.language == "en"

    def test_init_default_sample_rate(self) -> None:
        """Client defaults to 48000 Hz (Discord native rate)."""
        client = DeepgramTranscriber(api_key="test-key")
        assert client.sample_rate == 48000

    def test_init_custom_model(self) -> None:
        """Client accepts a custom model override."""
        client = DeepgramTranscriber(api_key="test-key", model="nova-2")
        assert client.model == "nova-2"

    def test_builds_ws_url(self) -> None:
        """build_ws_url includes all required query parameters."""
        client = DeepgramTranscriber(api_key="test-key")
        url = client.build_ws_url()
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        assert url.startswith(DEEPGRAM_WS_URL)
        assert params["model"] == ["nova-3"]
        assert params["language"] == ["en"]
        assert params["sample_rate"] == ["48000"]
        assert params["channels"] == ["1"]
        assert params["encoding"] == ["linear16"]
        # VAD now runs locally via TEN VAD; Deepgram VAD events are off.
        assert params["vad_events"] == ["false"]
        assert "diarize" not in params

    def test_builds_ws_url_omits_interim_results_by_default(self) -> None:
        """Default config disables interim results and utterance_end_ms."""
        client = DeepgramTranscriber(api_key="test-key")
        params = parse_qs(urlparse(client.build_ws_url()).query)
        # interim_results default is False → omitted (server default is false)
        assert "interim_results" not in params
        # utterance_end_ms requires interim_results → must be omitted when off
        assert "utterance_end_ms" not in params

    def test_builds_ws_url_emits_endpointing(self) -> None:
        """Default endpointing_ms is serialized on the URL."""
        client = DeepgramTranscriber(api_key="test-key")
        params = parse_qs(urlparse(client.build_ws_url()).query)
        assert params["endpointing"] == ["300"]

    def test_builds_ws_url_emits_utterance_end_ms_when_interim_on(self) -> None:
        """When interim_results is re-enabled, utterance_end_ms flows through."""
        client = DeepgramTranscriber(
            api_key="test-key", interim_results=True, utterance_end_ms=1500
        )
        params = parse_qs(urlparse(client.build_ws_url()).query)
        assert params["interim_results"] == ["true"]
        assert params["utterance_end_ms"] == ["1500"]

    def test_builds_ws_url_with_diarize(self) -> None:
        """build_ws_url includes diarize=true when enabled."""
        client = DeepgramTranscriber(api_key="test-key", diarize=True)
        url = client.build_ws_url()
        params = parse_qs(urlparse(url).query)
        assert params["diarize"] == ["true"]

    def test_builds_headers(self) -> None:
        """Headers include Authorization with Token prefix."""
        client = DeepgramTranscriber(api_key="sk-deepgram-test-123")
        headers = client.build_headers()
        assert headers["Authorization"] == "Token sk-deepgram-test-123"

    def test_clone_returns_new_instance_with_same_config(self) -> None:
        """clone() produces a distinct instance with identical configuration."""
        original = DeepgramTranscriber(
            api_key="test-key",
            model="nova-2",
            language="es",
            sample_rate=16000,
            channels=2,
            diarize=True,
            interim_results=False,
            utterance_end_ms=500,
            vad_events=False,
            endpointing_ms=450,
            replay_buffer_s=8.0,
        )
        cloned = original.clone()

        assert cloned is not original
        assert cloned.api_key == original.api_key
        assert cloned.model == original.model
        assert cloned.language == original.language
        assert cloned.sample_rate == original.sample_rate
        assert cloned.channels == original.channels
        assert cloned.diarize == original.diarize
        assert cloned.interim_results == original.interim_results
        assert cloned.utterance_end_ms == original.utterance_end_ms
        assert cloned.vad_events == original.vad_events
        assert cloned.endpointing_ms == original.endpointing_ms
        assert cloned.replay_buffer_s == pytest.approx(original.replay_buffer_s)


class TestDeepgramTranscriberParseResponse:
    """Tests for DeepgramTranscriber._parse_response."""

    def _make_client(self) -> DeepgramTranscriber:
        return DeepgramTranscriber(api_key="test-key")

    def test_parse_final_result(self) -> None:
        """Final result produces a TranscriptionResult with correct fields."""
        data = {
            "type": "Results",
            "channel": {
                "alternatives": [
                    {"transcript": "hello world", "confidence": 0.98},
                ],
            },
            "is_final": True,
            "start": 0.0,
            "duration": 1.5,
        }
        result = self._make_client()._parse_response(data)
        assert result is not None
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.confidence == pytest.approx(0.98)
        assert result.start == pytest.approx(0.0)
        assert result.end == pytest.approx(1.5)

    def test_parse_interim_result(self) -> None:
        """Interim result has is_final=False."""
        data = {
            "type": "Results",
            "channel": {
                "alternatives": [
                    {"transcript": "hel", "confidence": 0.75},
                ],
            },
            "is_final": False,
            "start": 0.0,
            "duration": 0.5,
        }
        result = self._make_client()._parse_response(data)
        assert result is not None
        assert result.is_final is False

    def test_parse_empty_transcript_returns_none(self) -> None:
        """Empty transcript string returns None (ignore silence)."""
        data = {
            "type": "Results",
            "channel": {
                "alternatives": [
                    {"transcript": "", "confidence": 0.0},
                ],
            },
            "is_final": True,
            "start": 0.0,
            "duration": 1.0,
        }
        assert self._make_client()._parse_response(data) is None

    def test_parse_no_alternatives_returns_none(self) -> None:
        """Response with no alternatives returns None."""
        data = {
            "type": "Results",
            "channel": {"alternatives": []},
            "is_final": True,
            "start": 0.0,
            "duration": 1.0,
        }
        assert self._make_client()._parse_response(data) is None

    def test_parse_non_results_type_returns_none(self) -> None:
        """Messages with type != 'Results' return None."""
        data = {"type": "Metadata", "request_id": "abc123"}
        assert self._make_client()._parse_response(data) is None

    def test_parse_with_diarize_speaker(self) -> None:
        """Speaker ID is captured from the first word when diarization is on."""
        data = {
            "type": "Results",
            "channel": {
                "alternatives": [
                    {
                        "transcript": "good morning",
                        "confidence": 0.95,
                        "words": [
                            {"word": "good", "speaker": 1},
                            {"word": "morning", "speaker": 1},
                        ],
                    },
                ],
            },
            "is_final": True,
            "start": 2.0,
            "duration": 1.0,
        }
        result = self._make_client()._parse_response(data)
        assert result is not None
        assert result.speaker == 1


class _AsyncIter:
    """Wrap a list of items into an async iterator for mocking __aiter__."""

    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class TestDeepgramTranscriberLifecycle:
    """Tests for start/send_audio/stop WebSocket lifecycle."""

    @pytest.fixture
    def client(self) -> DeepgramTranscriber:
        return DeepgramTranscriber(api_key="test-key")

    def _make_ws_mock(self, messages: list[object] | None = None) -> MagicMock:
        """Create a mock WebSocket with async methods."""
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        items = messages if messages is not None else []
        ws.__aiter__ = MagicMock(return_value=_AsyncIter(items))
        return ws

    @pytest.mark.asyncio
    async def test_start_connects_websocket(self, client: DeepgramTranscriber) -> None:
        """start() creates a session and connects via WebSocket."""
        ws_mock = self._make_ws_mock()
        connect_mock = AsyncMock(return_value=ws_mock)

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                connect_mock.assert_called_once()
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_send_audio_sends_bytes(self, client: DeepgramTranscriber) -> None:
        """send_audio() sends raw bytes over the WebSocket."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                audio = b"\x00\x01\x02\x03"
                await client.send_audio(audio)
                ws_mock.send_bytes.assert_called_once_with(audio)
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_send_audio_before_start_raises(
        self, client: DeepgramTranscriber
    ) -> None:
        """send_audio() raises RuntimeError if called before start()."""
        with pytest.raises(RuntimeError, match=r"not connected"):
            await client.send_audio(b"\x00\x01")

    @pytest.mark.asyncio
    async def test_send_audio_skips_when_ws_closed(
        self, client: DeepgramTranscriber
    ) -> None:
        """send_audio() silently returns when the WebSocket is already closed."""
        ws_mock = self._make_ws_mock()
        ws_mock.closed = True

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                # Should NOT raise — just returns silently.
                await client.send_audio(b"\x00\x01")
                ws_mock.send_bytes.assert_not_called()
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_finalize_sends_finalize_message(
        self, client: DeepgramTranscriber
    ) -> None:
        """finalize() sends Deepgram ``{"type":"Finalize"}`` to flush buffer."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                await client.finalize()
                ws_mock.send_json.assert_any_call({"type": "Finalize"})
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_finalize_skips_when_ws_closed(
        self, client: DeepgramTranscriber
    ) -> None:
        """finalize() silently no-ops when the WebSocket is already closed."""
        ws_mock = self._make_ws_mock()
        ws_mock.closed = True

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                await client.finalize()
                ws_mock.send_json.assert_not_called()
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_finalize_before_start_is_noop(
        self, client: DeepgramTranscriber
    ) -> None:
        """finalize() before start() is a silent no-op (never raises)."""
        await client.finalize()

    @pytest.mark.asyncio
    async def test_stop_sends_close_stream(self, client: DeepgramTranscriber) -> None:
        """stop() sends the CloseStream message and closes the WebSocket."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await client.stop()

            ws_mock.send_json.assert_called_once_with({"type": "CloseStream"})
            ws_mock.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, client: DeepgramTranscriber) -> None:
        """Calling stop() twice does not raise."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await client.stop()
            await client.stop()  # second call should not raise

    @pytest.mark.asyncio
    async def test_receive_loop_puts_results_on_queue(
        self, client: DeepgramTranscriber
    ) -> None:
        """The receive loop parses responses and puts results on the queue."""
        deepgram_response = json.dumps({
            "type": "Results",
            "channel": {
                "alternatives": [
                    {"transcript": "hello", "confidence": 0.99},
                ],
            },
            "is_final": True,
            "start": 0.0,
            "duration": 1.0,
        })

        ws_msg = MagicMock()
        ws_msg.type = 1  # aiohttp.WSMsgType.TEXT = 1
        ws_msg.data = deepgram_response

        ws_mock = self._make_ws_mock(messages=[ws_msg])

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)

            # Give the receive loop a moment to process
            await asyncio.sleep(0.05)
            await client.stop()

            assert not queue.empty()
            result = queue.get_nowait()
            assert isinstance(result, TranscriptionResult)
            assert result.text == "hello"
            assert result.is_final is True

    @pytest.mark.asyncio
    async def test_receive_loop_ignores_non_result_messages(
        self, client: DeepgramTranscriber
    ) -> None:
        """Metadata messages do not produce queue items."""
        metadata_response = json.dumps({
            "type": "Metadata",
            "request_id": "abc123",
        })

        ws_msg = MagicMock()
        ws_msg.type = 1  # aiohttp.WSMsgType.TEXT
        ws_msg.data = metadata_response

        ws_mock = self._make_ws_mock(messages=[ws_msg])

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)

            await asyncio.sleep(0.05)
            await client.stop()

            assert queue.empty()

    @pytest.mark.asyncio
    async def test_receive_loop_drops_vad_events(
        self, client: DeepgramTranscriber
    ) -> None:
        """`SpeechStarted` / `UtteranceEnd` are dropped — TEN VAD is the source."""
        msgs = []
        for payload in (
            {"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.1},
            {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.0},
        ):
            m = MagicMock()
            m.type = 1
            m.data = json.dumps(payload)
            msgs.append(m)

        ws_mock = self._make_ws_mock(messages=msgs)

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.05)
            await client.stop()

            assert queue.empty()

    @pytest.mark.asyncio
    async def test_receive_loop_passes_results_through(
        self, client: DeepgramTranscriber
    ) -> None:
        """Only `Results` reach the output queue now that VAD events are dropped."""
        msgs = []
        for payload in (
            {"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.1},
            {
                "type": "Results",
                "channel": {
                    "alternatives": [
                        {"transcript": "hello", "confidence": 0.99},
                    ],
                },
                "is_final": True,
                "start": 0.0,
                "duration": 1.0,
            },
            {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.0},
        ):
            m = MagicMock()
            m.type = 1
            m.data = json.dumps(payload)
            msgs.append(m)

        ws_mock = self._make_ws_mock(messages=msgs)

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.05)
            await client.stop()

            event = queue.get_nowait()
            assert isinstance(event, TranscriptionResult)
            assert event.is_final is True
            assert queue.empty()


class TestDeepgramReconnect:
    """Tests for automatic WebSocket reconnection."""

    @pytest.fixture
    def client(self) -> DeepgramTranscriber:
        t = DeepgramTranscriber(api_key="test-key")
        # Speed up tests — no real delay between reconnects.
        t._RECONNECT_DELAY = 0.01
        return t

    def _make_ws_mock(self, messages: list[object] | None = None) -> MagicMock:
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = 1006
        items = messages if messages is not None else []
        ws.__aiter__ = MagicMock(return_value=_AsyncIter(items))
        return ws

    @pytest.mark.asyncio
    async def test_reconnects_when_ws_closes(self, client: DeepgramTranscriber) -> None:
        """The receive loop reconnects when the WebSocket closes unexpectedly."""
        # First WS: empty iterator (simulates immediate close)
        ws1 = self._make_ws_mock()
        # Second WS: also empty, but proves reconnect happened
        ws2 = self._make_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            # Give the receive loop time to detect close and reconnect
            await asyncio.sleep(0.3)
            await client.stop()

        # start() calls once, reconnect calls again
        assert connect_mock.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_audio_works_after_reconnect(
        self, client: DeepgramTranscriber
    ) -> None:
        """After reconnection, send_audio sends to the new WebSocket."""
        ws1 = self._make_ws_mock()
        ws1.closed = True  # Mark closed so send_audio skips it
        ws2 = self._make_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.3)

            await client.send_audio(b"\x01\x02")
            await client.stop()

        ws2.send_bytes.assert_called_once_with(b"\x01\x02")

    @pytest.mark.asyncio
    async def test_gives_up_after_max_reconnects(
        self, client: DeepgramTranscriber
    ) -> None:
        """The receive loop stops retrying after max consecutive failures."""

        def _make_dying_ws() -> MagicMock:
            ws = self._make_ws_mock()
            ws.close_code = 1006
            return ws

        # Produce enough dying WSes for initial + max_reconnects + extra
        connect_mock = AsyncMock(side_effect=[_make_dying_ws() for _ in range(20)])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            # Give it enough time to exhaust retries
            await asyncio.sleep(1.5)
            await client.stop()

        # Should have stopped retrying (initial + max_reconnects)
        # Not unlimited — must be <= initial + max (default 5) + 1
        assert connect_mock.call_count <= 7

    @pytest.mark.asyncio
    async def test_does_not_reconnect_when_shutting_down(
        self, client: DeepgramTranscriber
    ) -> None:
        """Receive loop must not reconnect once ``_shutting_down`` is set.

        Simulates the race where ``stop()`` flips the flag before the
        receive loop notices the close frame.
        """
        ws1 = self._make_ws_mock()
        ws1.close_code = 1000  # clean close
        ws2 = self._make_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            # Flip the flag before the loop sees the close.
            client._shutting_down = True
            await asyncio.sleep(0.1)
            await client.stop()

        # Only the initial connect — no reconnect attempt.
        assert connect_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_does_not_reconnect_on_auth_close(
        self,
        client: DeepgramTranscriber,
    ) -> None:
        """Auth-style close codes (1008, 4001, 4008) must not be retried."""
        ws1 = self._make_ws_mock()
        ws1.close_code = 4001  # Deepgram auth failure style
        ws2 = self._make_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.1)
            await client.stop()

        # Only the initial connect — auth failures aren't retried.
        assert connect_mock.call_count == 1


class TestReplayBuffer:
    """Replay buffer — audio buffered during disconnect is resent on reconnect."""

    @pytest.fixture
    def client(self) -> DeepgramTranscriber:
        t = DeepgramTranscriber(api_key="test-key")
        t._RECONNECT_DELAY = 0.01
        return t

    def _make_ws_mock(
        self, messages: list[object] | None = None, *, close_code: int = 1006
    ) -> MagicMock:
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = close_code
        items = messages if messages is not None else []
        ws.__aiter__ = MagicMock(return_value=_AsyncIter(items))
        return ws

    def _make_open_ws_mock(self) -> MagicMock:
        """WS that stays open forever (pending future aiter)."""
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = None
        pending: asyncio.Future[object] = asyncio.get_event_loop().create_future()

        class _Never:
            def __aiter__(self) -> _Never:
                return self

            async def __anext__(self) -> object:
                await pending
                raise StopAsyncIteration

        ws.__aiter__ = MagicMock(return_value=_Never())
        return ws

    @pytest.mark.asyncio
    async def test_send_audio_buffers_chunk_when_ws_closed(
        self, client: DeepgramTranscriber
    ) -> None:
        """send_audio on a closed WS adds the chunk to the replay buffer."""
        ws = self._make_open_ws_mock()
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                ws.closed = True
                await client.send_audio(b"\x01\x02\x03\x04")
                # chunk must be in the buffer, not sent over the wire
                ws.send_bytes.assert_not_called()
                assert len(client._replay_buffer) > 0
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_replay_buffer_chunks_sent_to_new_ws_after_reconnect(
        self, client: DeepgramTranscriber
    ) -> None:
        """After reconnect, buffered chunks are replayed to the new WS in order."""
        chunk_a = b"\xaa" * 100
        chunk_b = b"\xbb" * 100

        ws1 = self._make_ws_mock()
        ws1.closed = True  # already closed so send_audio buffers immediately
        ws2 = self._make_open_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)

            # send while ws1 is closed → buffer
            await client.send_audio(chunk_a)
            await client.send_audio(chunk_b)

            # wait for reconnect to ws2
            await asyncio.sleep(0.2)

            call_args = [c.args[0] for c in ws2.send_bytes.call_args_list]
            assert chunk_a in call_args
            assert chunk_b in call_args
            # order preserved
            assert call_args.index(chunk_a) < call_args.index(chunk_b)
            # buffer cleared after drain so chunks aren't re-sent on a later drain
            assert len(client._replay_buffer) == 0

            await client.stop()

    @pytest.mark.asyncio
    async def test_replay_sends_finalize_after_drain(
        self, client: DeepgramTranscriber
    ) -> None:
        """After draining the replay, Finalize flushes the replayed audio.

        Without this, Deepgram holds the replayed segment until more audio
        arrives (or endpointing silence — which our VAD-gated pump doesn't
        send). The user sees no transcript until they speak again.
        """
        chunk = b"\xaa" * 100
        ws1 = self._make_ws_mock()
        ws1.closed = True  # already closed so send_audio buffers immediately
        ws2 = self._make_open_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await client.send_audio(chunk)
            await asyncio.sleep(0.2)  # let reconnect + drain run

            finalize_calls = [
                c.args[0]
                for c in ws2.send_json.call_args_list
                if c.args and c.args[0].get("type") == "Finalize"
            ]
            assert len(finalize_calls) == 1, (
                f"expected exactly one Finalize after replay; got {finalize_calls}"
            )

            await client.stop()

    @pytest.mark.asyncio
    async def test_empty_replay_does_not_send_finalize(
        self, client: DeepgramTranscriber
    ) -> None:
        """Zero-chunk replay should not trigger a spurious Finalize."""
        ws1 = self._make_ws_mock()  # closes without any buffered audio
        ws2 = self._make_open_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.2)  # let reconnect run

            finalize_calls = [
                c.args[0]
                for c in ws2.send_json.call_args_list
                if c.args and c.args[0].get("type") == "Finalize"
            ]
            assert finalize_calls == []

            await client.stop()

    @pytest.mark.asyncio
    async def test_replay_buffer_trims_oldest_when_over_budget(
        self, client: DeepgramTranscriber
    ) -> None:
        """Buffer trims oldest chunks when cumulative bytes exceed the cap."""
        # set a tiny cap: 1 s at 1 byte/s = 1 byte
        client.sample_rate = 1
        client.channels = 1

        ws = self._make_open_ws_mock()
        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            ws.closed = True  # force buffering

            try:
                chunk_a = b"\xaa" * 4
                chunk_b = b"\xbb" * 4
                chunk_c = b"\xcc" * 4

                await client.send_audio(chunk_a)
                await client.send_audio(chunk_b)
                await client.send_audio(chunk_c)

                # buffer is over budget; oldest chunk should be evicted
                buffered = list(client._replay_buffer)
                assert chunk_a not in buffered  # oldest dropped
                assert chunk_c in buffered  # newest retained
            finally:
                await client.stop()


class TestExponentialBackoff:
    """First reconnect is immediate; subsequent failures back off exponentially."""

    @pytest.fixture
    def client(self) -> DeepgramTranscriber:
        t = DeepgramTranscriber(api_key="test-key")
        # suppress keepalive loop so its sleep() calls don't pollute the list
        t._KEEPALIVE_INTERVAL = 999.0
        return t

    def _make_ws_mock(self, close_code: int = 1006) -> MagicMock:
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = close_code
        ws.__aiter__ = MagicMock(return_value=_AsyncIter([]))
        return ws

    def _make_open_ws_mock(self) -> MagicMock:
        """WS that stays open indefinitely (never emits messages)."""
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = None
        pending: asyncio.Future[object] = asyncio.get_event_loop().create_future()

        class _Never:
            def __aiter__(self) -> _Never:
                return self

            async def __anext__(self) -> object:
                await pending
                raise StopAsyncIteration

        ws.__aiter__ = MagicMock(return_value=_Never())
        return ws

    @pytest.mark.asyncio
    async def test_backoff_sequence(self, client: DeepgramTranscriber) -> None:
        """First reconnect is immediate; successive failures back off with growth."""
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def _track_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await original_sleep(0)

        # ws1+ws2+ws3 die immediately; ws4 stays open
        # reconnects: 1st (no sleep), 2nd (sleep 1.0), 3rd (sleep 2.0)
        dying = [self._make_ws_mock() for _ in range(3)]
        ws_list = [*dying, self._make_open_ws_mock()]
        connect_mock = AsyncMock(side_effect=ws_list)

        with (
            patch.object(client, "_ws_connect", new=connect_mock),
            patch("familiar_connect.transcription.asyncio.sleep", new=_track_sleep),
        ):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await original_sleep(0.2)
            await client.stop()

        # keepalive sleeps (999.0) excluded by the cap filter.
        cap = client._RECONNECT_BACKOFF_CAP
        backoff_sleeps = [d for d in sleep_calls if 0 < d <= cap]

        # 3 failures → 3 reconnect attempts. If the 1st attempt had a
        # pre-sleep we'd see 3 backoff delays; "first is immediate" means 2.
        assert len(backoff_sleeps) == 2, (
            f"expected exactly 2 backoff sleeps (first reconnect immediate, "
            f"subsequent two delayed); got {sleep_calls}"
        )
        assert backoff_sleeps == sorted(backoff_sleeps), (
            f"backoff sleeps must be non-decreasing: {backoff_sleeps}"
        )
        assert backoff_sleeps[-1] > backoff_sleeps[0], (
            f"backoff must grow; got: {backoff_sleeps}"
        )


class TestDeepgramKeepAlive:
    """Tests for the periodic KeepAlive frame that prevents Deepgram's idle timeout."""

    @pytest.fixture
    def client(self) -> DeepgramTranscriber:
        t = DeepgramTranscriber(api_key="test-key")
        # Speed up tests — tick much faster than 5 s so suite stays fast.
        t._KEEPALIVE_INTERVAL = 0.02
        return t

    def _make_ws_mock(self, messages: list[object] | None = None) -> MagicMock:
        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        ws.close_code = None
        items = messages if messages is not None else []
        # Keep the aiter alive for the duration of the test so the receive
        # loop doesn't exit and trigger a reconnect during keepalive checks.
        pending: asyncio.Future[object] = asyncio.get_event_loop().create_future()

        class _Never:
            def __aiter__(self) -> _Never:
                return self

            async def __anext__(self) -> object:
                if items:
                    return items.pop(0)
                await pending  # wait forever — test will cancel via stop()
                raise StopAsyncIteration

        ws.__aiter__ = MagicMock(return_value=_Never())
        return ws

    @staticmethod
    def _keepalive_calls(ws: MagicMock) -> list[object]:
        return [
            call.args[0]
            for call in ws.send_json.call_args_list
            if call.args and call.args[0] == {"type": "KeepAlive"}
        ]

    @pytest.mark.asyncio
    async def test_keepalive_sent_while_connected(
        self, client: DeepgramTranscriber
    ) -> None:
        """The KeepAlive loop ticks periodically while the WebSocket is open."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                # With interval=0.02, expect >=2 ticks within 0.1s.
                await asyncio.sleep(0.1)
                assert len(self._keepalive_calls(ws_mock)) >= 2
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_keepalive_task_cancelled_on_stop(
        self, client: DeepgramTranscriber
    ) -> None:
        """stop() cancels the keepalive task and no further KeepAlives fire."""
        ws_mock = self._make_ws_mock()

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            await asyncio.sleep(0.05)
            await client.stop()

            count_before = len(self._keepalive_calls(ws_mock))
            await asyncio.sleep(0.1)
            count_after = len(self._keepalive_calls(ws_mock))

        assert client._keepalive_task is None
        assert count_after == count_before

    @pytest.mark.asyncio
    async def test_keepalive_skips_when_ws_closed(
        self, client: DeepgramTranscriber
    ) -> None:
        """No KeepAlive frames are sent while the WS reports closed."""
        ws_mock = self._make_ws_mock()
        ws_mock.closed = True

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                await asyncio.sleep(0.1)
                assert self._keepalive_calls(ws_mock) == []
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_keepalive_survives_send_errors(
        self, client: DeepgramTranscriber
    ) -> None:
        """A transient send failure does not kill the keepalive loop."""
        ws_mock = self._make_ws_mock()
        # First call raises; subsequent calls succeed.
        ws_mock.send_json = AsyncMock(
            side_effect=[RuntimeError("transient"), None, None, None, None, None]
        )

        with patch.object(client, "_ws_connect", new=AsyncMock(return_value=ws_mock)):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                await asyncio.sleep(0.15)
                # At least one post-error KeepAlive should have fired.
                assert ws_mock.send_json.call_count >= 2
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_keepalive_follows_reconnect(
        self, client: DeepgramTranscriber
    ) -> None:
        """After reconnect, KeepAlive targets the new WebSocket."""
        client._RECONNECT_DELAY = 0.01
        # ws1 closes immediately (empty async iter). ws2 stays open.
        ws1 = MagicMock()
        ws1.send_bytes = AsyncMock()
        ws1.send_json = AsyncMock()
        ws1.close = AsyncMock()
        ws1.closed = False
        ws1.close_code = 1006
        ws1.__aiter__ = MagicMock(return_value=_AsyncIter([]))

        ws2 = self._make_ws_mock()
        connect_mock = AsyncMock(side_effect=[ws1, ws2])

        with patch.object(client, "_ws_connect", new=connect_mock):
            queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue()
            await client.start(queue)
            try:
                # Wait for reconnect + keepalive ticks on the new socket.
                await asyncio.sleep(0.15)
                assert len(self._keepalive_calls(ws2)) >= 1
            finally:
                await client.stop()


class TestCreateTranscriberFromEnv:
    def test_creates_from_env(self) -> None:
        """Factory reads DEEPGRAM_API_KEY and optional overrides."""
        env = {
            "DEEPGRAM_API_KEY": "sk-deepgram-test-abc",
            "DEEPGRAM_MODEL": "nova-2",
            "DEEPGRAM_LANGUAGE": "es",
        }
        with patch.dict(os.environ, env, clear=False):
            client = create_transcriber_from_env()

        assert client.api_key == "sk-deepgram-test-abc"
        assert client.model == "nova-2"
        assert client.language == "es"

    def test_uses_defaults_when_optional_missing(self) -> None:
        """Factory uses defaults for model and language when not in env."""
        env = {"DEEPGRAM_API_KEY": "sk-deepgram-test-abc"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("DEEPGRAM_MODEL", None)
            os.environ.pop("DEEPGRAM_LANGUAGE", None)
            client = create_transcriber_from_env()

        assert client.api_key == "sk-deepgram-test-abc"
        assert client.model == DEFAULT_MODEL
        assert client.language == DEFAULT_LANGUAGE

    def test_raises_when_api_key_missing(self) -> None:
        """Factory raises a clear error when DEEPGRAM_API_KEY is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match=r"DEEPGRAM_API_KEY"),
        ):
            create_transcriber_from_env()

    def test_reconnect_knobs_from_env(self) -> None:
        """Factory wires reconnect/buffer/keepalive knobs from env vars."""
        env = {
            "DEEPGRAM_API_KEY": "sk-test",
            "DEEPGRAM_REPLAY_BUFFER_S": "10.0",
            "DEEPGRAM_KEEPALIVE_INTERVAL_S": "5.0",
            "DEEPGRAM_RECONNECT_MAX_ATTEMPTS": "3",
            "DEEPGRAM_RECONNECT_BACKOFF_CAP_S": "32.0",
        }
        with patch.dict(os.environ, env, clear=False):
            client = create_transcriber_from_env()

        assert client.replay_buffer_s == pytest.approx(10.0)
        assert pytest.approx(5.0) == client._KEEPALIVE_INTERVAL
        assert client._MAX_RECONNECTS == 3
        assert pytest.approx(32.0) == client._RECONNECT_BACKOFF_CAP
