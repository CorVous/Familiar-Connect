"""Tests for the :class:`TenVAD` wrapper.

Mocks ``ten_vad.TenVad`` so the suite doesn't need TEN-VAD's native
shared library on disk. Verifies hop-size validation, threshold
behaviour, state recreation on reset.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from familiar_connect.voice.turn_detection import TenVAD

if TYPE_CHECKING:
    from collections.abc import Iterator


def _silence_pcm(samples: int) -> bytes:
    """Generate ``samples`` int16 zero samples — silence at any sample rate."""
    return struct.pack(f"<{samples}h", *([0] * samples))


def _tone_pcm(samples: int, amplitude: int = 5000) -> bytes:
    """Square-wave-ish PCM — non-silent for VAD probability rounding."""
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


@pytest.fixture
def fake_native() -> Iterator[MagicMock]:
    """Patch ``ten_vad.TenVad`` with a configurable mock.

    ``process`` returns ``(probability, flag)`` matching the upstream
    Python wrapper. Default to no-speech (low probability).

    Yields:
        MagicMock: the patched constructor — tests inspect
        ``return_value`` for the per-instance mock and override
        ``process.return_value`` to set probabilities.

    """
    with patch("familiar_connect.voice.turn_detection.ten_vad._TenVadNative") as ctor:
        instance = MagicMock()
        instance.process.return_value = (0.1, 0)
        ctor.return_value = instance
        yield ctor


class TestTenVADInit:
    def test_constructs_native_with_defaults(self, fake_native: MagicMock) -> None:
        vad = TenVAD()
        assert vad.sample_rate == 16000
        assert vad.hop_size == 256
        fake_native.assert_called_once_with(hop_size=256, threshold=0.5)

    def test_rejects_unsupported_hop_size(self, fake_native: MagicMock) -> None:  # noqa: ARG002
        """TEN-VAD only accepts 160 or 256 at 16 kHz; reject other sizes early."""
        with pytest.raises(ValueError, match="hop_size"):
            TenVAD(hop_size=512)

    def test_rejects_non_16k_sample_rate(self, fake_native: MagicMock) -> None:  # noqa: ARG002
        with pytest.raises(ValueError, match="sample_rate"):
            TenVAD(sample_rate=48000)


class TestSpeechProbability:
    def test_returns_probability_from_process(self, fake_native: MagicMock) -> None:
        fake_native.return_value.process.return_value = (0.87, 1)
        vad = TenVAD()
        prob = vad.speech_probability(_silence_pcm(256))
        assert prob == pytest.approx(0.87, abs=1e-5)

    def test_passes_int16_array_at_hop_size(self, fake_native: MagicMock) -> None:
        """Audio fed as int16 ndarray of length hop_size — no normalisation."""
        vad = TenVAD()
        vad.speech_probability(_tone_pcm(256))
        audio = fake_native.return_value.process.call_args[0][0]
        assert audio.dtype.kind == "i"
        assert audio.dtype.itemsize == 2
        assert audio.shape == (256,)
        assert audio[0] == 5000

    def test_supports_10ms_hop_size(self, fake_native: MagicMock) -> None:
        """160-sample hop (10 ms) is the alternate native frame."""
        vad = TenVAD(hop_size=160)
        vad.speech_probability(_silence_pcm(160))
        audio = fake_native.return_value.process.call_args[0][0]
        assert audio.shape == (160,)

    def test_rejects_wrong_chunk_length(self, fake_native: MagicMock) -> None:  # noqa: ARG002
        vad = TenVAD()
        with pytest.raises(ValueError, match="expected"):
            vad.speech_probability(_silence_pcm(128))


class TestThreshold:
    def test_is_speech_above_threshold(self, fake_native: MagicMock) -> None:
        fake_native.return_value.process.return_value = (0.6, 1)
        vad = TenVAD(threshold=0.5)
        assert vad.is_speech(_silence_pcm(256)) is True

    def test_is_not_speech_below_threshold(self, fake_native: MagicMock) -> None:
        fake_native.return_value.process.return_value = (0.4, 0)
        vad = TenVAD(threshold=0.5)
        assert vad.is_speech(_silence_pcm(256)) is False


class TestReset:
    def test_reset_recreates_native_handle(self, fake_native: MagicMock) -> None:
        """No public C reset — ``reset`` rebuilds the native instance."""
        vad = TenVAD(hop_size=256, threshold=0.4)
        assert fake_native.call_count == 1
        vad.reset()
        assert fake_native.call_count == 2
        # second construction reuses same hop/threshold
        last_call = fake_native.call_args_list[-1]
        assert last_call.kwargs == {"hop_size": 256, "threshold": 0.4}
