"""FasterWhisper (CTranslate2) transcriber — buffer + finalize semantics.

`faster_whisper.WhisperModel` is mocked so CI doesn't pull CT2 weights.
Numpy is provided by the ``local-turn`` extra already installed in the
test environment.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from familiar_connect.config import FasterWhisperSTTConfig
from familiar_connect.stt import Transcriber, TranscriptionResult
from familiar_connect.stt.faster_whisper import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL_SIZE,
    FasterWhisperTranscriber,
    create_faster_whisper_from_env,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _pcm48k(samples_16k_equivalent: int, amplitude: int = 5000) -> bytes:
    """Synthesise 48 kHz int16 PCM whose post-resample length matches input."""
    n48 = samples_16k_equivalent * 3
    return struct.pack(f"<{n48}h", *([amplitude] * n48))


def _make_segments(*texts: str) -> list[MagicMock]:
    out: list[MagicMock] = []
    for t in texts:
        seg = MagicMock()
        seg.text = t
        out.append(seg)
    return out


@dataclass
class Harness:
    """Bundles the patched transcriber, its fake model, and the load-mock."""

    transcriber: FasterWhisperTranscriber
    model: MagicMock
    load_mock: MagicMock


@pytest.fixture
def harness() -> Iterator[Harness]:
    """Patch ``_load_model`` to return a stub WhisperModel.

    Yields:
        :class:`Harness` — transcriber + the stub model + the load-mock.

    """
    model = MagicMock()
    info = MagicMock()
    model.transcribe.return_value = (iter(_make_segments("hello", " there")), info)
    with patch.object(
        FasterWhisperTranscriber, "_load_model", return_value=model
    ) as load_mock:
        yield Harness(
            transcriber=FasterWhisperTranscriber(),
            model=model,
            load_mock=load_mock,
        )


class TestProtocolContract:
    def test_satisfies_transcriber_protocol(self) -> None:
        t = FasterWhisperTranscriber()
        assert isinstance(t, Transcriber)


class TestConstruction:
    def test_default_model_size(self) -> None:
        t = FasterWhisperTranscriber()
        assert t.model_size == DEFAULT_MODEL_SIZE
        assert t.model_size == "small"

    def test_default_compute_type(self) -> None:
        t = FasterWhisperTranscriber()
        assert t.compute_type == DEFAULT_COMPUTE_TYPE

    def test_default_language_en(self) -> None:
        t = FasterWhisperTranscriber()
        assert t.language == "en"

    def test_clone_shares_model(self) -> None:
        sentinel = MagicMock()
        a = FasterWhisperTranscriber(model_size="medium", language="de")
        a._model = sentinel
        b = a.clone()
        assert b is not a
        assert b._model is sentinel
        assert b.model_size == "medium"
        assert b.language == "de"

    def test_clone_carries_idle_close_s(self) -> None:
        a = FasterWhisperTranscriber()
        a._IDLE_CLOSE_S = 12.5
        b = a.clone()
        assert pytest.approx(12.5) == b._IDLE_CLOSE_S


class TestStart:
    @pytest.mark.asyncio
    async def test_lazy_loads_model_on_first_start(self, harness: Harness) -> None:
        assert harness.transcriber._model is None
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        assert harness.transcriber._model is harness.model
        assert harness.load_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_subsequent_start_reuses_model(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.start(q)
        assert harness.load_mock.call_count == 1


class TestSendAudio:
    @pytest.mark.asyncio
    async def test_resamples_and_buffers(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=480))
        assert len(harness.transcriber._buffer) == 960


class TestFinalize:
    @pytest.mark.asyncio
    async def test_emits_final_joined_segments(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(
            _pcm48k(samples_16k_equivalent=16_000)
        )  # 1 s
        await harness.transcriber.finalize()

        assert q.qsize() == 1
        result = await q.get()
        assert result.is_final is True
        # Segments are concatenated; whisper conventionally embeds spacing.
        assert result.text == "hello there"
        assert result.end == pytest.approx(1.0, abs=1e-3)
        assert len(harness.transcriber._buffer) == 0

    @pytest.mark.asyncio
    async def test_passes_language_to_transcribe(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        t = FasterWhisperTranscriber(language="de")
        t._model = harness.model
        await t.start(q)
        await t.send_audio(_pcm48k(samples_16k_equivalent=16_000))
        await t.finalize()
        (_args, kwargs) = harness.model.transcribe.call_args
        assert kwargs.get("language") == "de"

    @pytest.mark.asyncio
    async def test_audio_is_float32_numpy(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=16_000))
        await harness.transcriber.finalize()
        (args, _kwargs) = harness.model.transcribe.call_args
        assert isinstance(args[0], np.ndarray)
        assert args[0].dtype == np.float32

    @pytest.mark.asyncio
    async def test_empty_buffer_is_noop(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.finalize()
        assert q.empty()
        harness.model.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_segments_skipped(self, harness: Harness) -> None:
        # Whisper sometimes returns no segments for pure silence/noise.
        info = MagicMock()
        harness.model.transcribe.return_value = (iter([]), info)
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=16_000))
        await harness.transcriber.finalize()
        assert q.empty()


class TestStop:
    @pytest.mark.asyncio
    async def test_clears_buffer(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=480))
        assert len(harness.transcriber._buffer) > 0
        await harness.transcriber.stop()
        assert len(harness.transcriber._buffer) == 0

    @pytest.mark.asyncio
    async def test_idempotent(self, harness: Harness) -> None:
        await harness.transcriber.stop()
        await harness.transcriber.stop()


class TestEnvFactory:
    def test_uses_config_defaults(self) -> None:
        cfg = FasterWhisperSTTConfig()
        t = create_faster_whisper_from_env(cfg)
        assert t.model_size == DEFAULT_MODEL_SIZE
        assert t.language == "en"

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FASTER_WHISPER_MODEL_SIZE", "medium")
        monkeypatch.setenv("FASTER_WHISPER_DEVICE", "cuda")
        monkeypatch.setenv("FASTER_WHISPER_COMPUTE_TYPE", "float16")
        monkeypatch.setenv("FASTER_WHISPER_LANGUAGE", "fr")
        monkeypatch.setenv("FASTER_WHISPER_IDLE_CLOSE_S", "45.0")
        t = create_faster_whisper_from_env()
        assert t.model_size == "medium"
        assert t.device == "cuda"
        assert t.compute_type == "float16"
        assert t.language == "fr"
        assert pytest.approx(45.0) == t._IDLE_CLOSE_S


class TestBackendKnobIgnored:
    """Bot sets ``endpointing_ms`` only when the attr exists (Deepgram-only)."""

    def test_no_endpointing_ms_attribute(self) -> None:
        t = FasterWhisperTranscriber()
        assert not hasattr(t, "endpointing_ms")
