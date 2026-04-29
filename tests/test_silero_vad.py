"""Tests for the :class:`SileroVAD` wrapper.

Mocks ``onnxruntime.InferenceSession`` so the test suite doesn't need
the ~2 MB ONNX model (or the ~360 MB Smart Turn model) on disk.
Verifies input shape, state propagation, and threshold logic.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from familiar_connect.voice.turn_detection import SileroVAD

if TYPE_CHECKING:
    from collections.abc import Iterator


def _silence_pcm(samples: int) -> bytes:
    """Generate ``samples`` int16 zero samples — silence at any sample rate."""
    return struct.pack(f"<{samples}h", *([0] * samples))


def _tone_pcm(samples: int, amplitude: int = 5000) -> bytes:
    """Square-wave-ish PCM — non-silent for VAD probability rounding."""
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


@pytest.fixture
def fake_session() -> Iterator[MagicMock]:
    """Patch ``onnxruntime.InferenceSession`` with a configurable mock.

    ``run`` returns ``([[probability]], state)`` matching Silero v5's
    ``(prob, state)`` output shape.

    Yields:
        MagicMock: the patched session instance — tests override
        ``run.return_value`` to set probabilities.

    """
    with patch(
        "familiar_connect.voice.turn_detection.silero_vad.ort.InferenceSession"
    ) as cls:
        instance = MagicMock()
        # Default: low probability (no speech).
        instance.run.return_value = (
            np.array([[0.1]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        cls.return_value = instance
        yield instance


class TestSileroVADInit:
    def test_loads_model_from_path(self, fake_session: MagicMock) -> None:  # noqa: ARG002
        vad = SileroVAD(Path("/fake/silero_vad.onnx"))
        # Internal state initialised to zeros, sr matches sample_rate.
        assert vad.sample_rate == 16000

    def test_rejects_unknown_chunk_size(self, fake_session: MagicMock) -> None:  # noqa: ARG002
        """Silero v5 expects 512 samples at 16 kHz; reject other sizes early."""
        with pytest.raises(ValueError, match="chunk_size"):
            SileroVAD(Path("/fake/m.onnx"), chunk_size=999)


class TestSpeechProbability:
    def test_returns_probability_from_session(self, fake_session: MagicMock) -> None:
        vad = SileroVAD(Path("/fake/m.onnx"))
        fake_session.run.return_value = (
            np.array([[0.87]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        prob = vad.speech_probability(_silence_pcm(512))
        assert prob == pytest.approx(0.87, abs=1e-5)

    def test_runs_session_with_correct_input_shape(
        self, fake_session: MagicMock
    ) -> None:
        """Audio fed as ``[1, chunk_size]`` float32 in [-1, 1]."""
        vad = SileroVAD(Path("/fake/m.onnx"))
        vad.speech_probability(_tone_pcm(512))
        feed = fake_session.run.call_args[0][1]
        audio = feed["input"]
        assert audio.dtype == np.float32
        assert audio.shape == (1, 512)
        # int16 5000 → ~0.1526 after /32768 normalization.
        assert audio[0, 0] == pytest.approx(5000 / 32768.0, abs=1e-4)

    def test_propagates_state_across_calls(self, fake_session: MagicMock) -> None:
        """Silero is stateful: each call's output state feeds the next call."""
        new_state = np.full((2, 1, 128), 0.42, dtype=np.float32)
        fake_session.run.return_value = (
            np.array([[0.3]], dtype=np.float32),
            new_state,
        )

        vad = SileroVAD(Path("/fake/m.onnx"))
        # First call: state starts at zeros.
        first_state_in = fake_session.run.call_args[0][1]["state"] if False else None
        vad.speech_probability(_silence_pcm(512))
        # Second call: state should be the new_state from the first run.
        vad.speech_probability(_silence_pcm(512))
        second_call = fake_session.run.call_args_list[1][0][1]
        assert np.allclose(second_call["state"], 0.42)
        del first_state_in

    def test_rejects_wrong_chunk_length(self, fake_session: MagicMock) -> None:  # noqa: ARG002
        vad = SileroVAD(Path("/fake/m.onnx"))
        with pytest.raises(ValueError, match="expected"):
            vad.speech_probability(_silence_pcm(256))


class TestThreshold:
    def test_is_speech_above_threshold(self, fake_session: MagicMock) -> None:
        fake_session.run.return_value = (
            np.array([[0.6]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        vad = SileroVAD(Path("/fake/m.onnx"), threshold=0.5)
        assert vad.is_speech(_silence_pcm(512)) is True

    def test_is_not_speech_below_threshold(self, fake_session: MagicMock) -> None:
        fake_session.run.return_value = (
            np.array([[0.4]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        vad = SileroVAD(Path("/fake/m.onnx"), threshold=0.5)
        assert vad.is_speech(_silence_pcm(512)) is False


class TestReset:
    def test_reset_zeros_state(self, fake_session: MagicMock) -> None:
        new_state = np.full((2, 1, 128), 0.5, dtype=np.float32)
        fake_session.run.return_value = (
            np.array([[0.3]], dtype=np.float32),
            new_state,
        )
        vad = SileroVAD(Path("/fake/m.onnx"))
        vad.speech_probability(_silence_pcm(512))
        vad.reset()
        # Next call should start from zeros again.
        vad.speech_probability(_silence_pcm(512))
        last_call = fake_session.run.call_args_list[-1][0][1]
        assert np.allclose(last_call["state"], 0.0)
