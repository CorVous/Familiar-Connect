"""Parakeet-TDT 0.6B v3 transcriber — buffer + finalize semantics.

NeMo and the model itself are mocked so CI doesn't pull torch / 600 MB
of weights. Numpy is provided by the ``local-turn`` extra already
installed in the test environment.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from familiar_connect.config import ParakeetSTTConfig
from familiar_connect.stt import Transcriber, TranscriptionResult
from familiar_connect.stt.parakeet import (
    DEFAULT_MODEL_NAME,
    ParakeetTranscriber,
    create_parakeet_from_env,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _pcm48k(samples_16k_equivalent: int, amplitude: int = 5000) -> bytes:
    """Synthesise 48 kHz int16 PCM whose post-resample length matches input."""
    n48 = samples_16k_equivalent * 3  # 3:1 decimation
    return struct.pack(f"<{n48}h", *([amplitude] * n48))


@dataclass
class Harness:
    """Bundles the patched transcriber, its fake model, and the load-mock."""

    transcriber: ParakeetTranscriber
    model: MagicMock
    load_mock: MagicMock


@pytest.fixture
def harness() -> Iterator[Harness]:
    """Patch ``_load_model`` to return a stub ASR model.

    Yields:
        :class:`Harness` — transcriber + the stub model + the load-mock.

    """
    model = MagicMock()
    hyp = MagicMock()
    hyp.text = "hello there"
    model.transcribe.return_value = [hyp]
    with patch.object(
        ParakeetTranscriber, "_load_model", return_value=model
    ) as load_mock:
        yield Harness(
            transcriber=ParakeetTranscriber(), model=model, load_mock=load_mock
        )


class TestProtocolContract:
    def test_satisfies_transcriber_protocol(self) -> None:
        t = ParakeetTranscriber()
        assert isinstance(t, Transcriber)


class TestConstruction:
    def test_default_model_name(self) -> None:
        t = ParakeetTranscriber()
        assert t.model_name == DEFAULT_MODEL_NAME
        assert t.model_name == "nvidia/parakeet-tdt-0.6b-v3"

    def test_default_device_auto(self) -> None:
        t = ParakeetTranscriber()
        assert t.device == "auto"

    def test_default_sample_rate_48k(self) -> None:
        t = ParakeetTranscriber()
        assert t.sample_rate == 48_000

    def test_clone_shares_model(self) -> None:
        sentinel_model = MagicMock()
        a = ParakeetTranscriber(model_name="custom")
        a._model = sentinel_model
        b = a.clone()
        assert b is not a
        assert b._model is sentinel_model
        assert b.model_name == "custom"

    def test_clone_carries_idle_close_s(self) -> None:
        a = ParakeetTranscriber()
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
        # Second start sees the model already populated; no re-load.
        assert harness.load_mock.call_count == 1


class TestSendAudio:
    @pytest.mark.asyncio
    async def test_resamples_and_buffers(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        # 30 ms at 48 kHz mono = 1440 samples = 2880 bytes; resamples to 480
        # samples at 16 kHz = 960 bytes.
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=480))
        # buffer holds 16 kHz int16 PCM
        assert len(harness.transcriber._buffer) == 960


class TestFinalize:
    @pytest.mark.asyncio
    async def test_emits_final_result_and_clears_buffer(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(
            _pcm48k(samples_16k_equivalent=16_000)
        )  # 1 s
        await harness.transcriber.finalize()

        assert q.qsize() == 1
        result = await q.get()
        assert result.is_final is True
        assert result.text == "hello there"
        assert result.end == pytest.approx(1.0, abs=1e-3)
        assert len(harness.transcriber._buffer) == 0
        # transcribe called with one float32 numpy array, positional list
        (args, _kwargs) = harness.model.transcribe.call_args
        assert isinstance(args[0], list)
        assert isinstance(args[0][0], np.ndarray)
        assert args[0][0].dtype == np.float32

    @pytest.mark.asyncio
    async def test_empty_buffer_is_noop(self, harness: Harness) -> None:
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.finalize()
        assert q.empty()
        harness.model.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self, harness: Harness) -> None:
        empty = MagicMock()
        empty.text = "   "
        harness.model.transcribe.return_value = [empty]
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=16_000))
        await harness.transcriber.finalize()
        assert q.empty()

    @pytest.mark.asyncio
    async def test_handles_string_hypothesis_shape(self, harness: Harness) -> None:
        # Older NeMo versions return list[str] directly.
        harness.model.transcribe.return_value = ["plain string transcript"]
        q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await harness.transcriber.start(q)
        await harness.transcriber.send_audio(_pcm48k(samples_16k_equivalent=16_000))
        await harness.transcriber.finalize()
        result = await q.get()
        assert result.text == "plain string transcript"


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
        await harness.transcriber.stop()  # no exception


class TestEnvFactory:
    def test_uses_config_defaults(self) -> None:
        cfg = ParakeetSTTConfig()
        t = create_parakeet_from_env(cfg)
        assert t.model_name == DEFAULT_MODEL_NAME
        assert t.device == "auto"

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-1.1b")
        monkeypatch.setenv("PARAKEET_DEVICE", "cuda")
        monkeypatch.setenv("PARAKEET_IDLE_CLOSE_S", "45.0")
        t = create_parakeet_from_env()
        assert t.model_name == "nvidia/parakeet-tdt-1.1b"
        assert t.device == "cuda"
        assert pytest.approx(45.0) == t._IDLE_CLOSE_S


class TestBackendKnobIgnored:
    """Bot sets ``endpointing_ms`` only when the attr exists (Deepgram-only)."""

    def test_no_endpointing_ms_attribute(self) -> None:
        t = ParakeetTranscriber()
        assert not hasattr(t, "endpointing_ms")
