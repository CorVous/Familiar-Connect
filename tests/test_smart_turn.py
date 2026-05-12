"""Tests for the :class:`SmartTurnDetector` wrapper.

Mocks ``onnxruntime.InferenceSession`` so the test suite doesn't
require the ~360 MB Smart Turn v3 ONNX model. Verifies input
normalisation, max-duration clamping, and both 2-class softmax /
sigmoid output shapes (Pipecat's published exports use one or the
other depending on version).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from familiar_connect.voice.turn_detection import SmartTurnDetector

if TYPE_CHECKING:
    from collections.abc import Iterator


def _pcm(samples: int, amplitude: int = 1000) -> bytes:
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


@pytest.fixture
def fake_session() -> Iterator[MagicMock]:
    with patch(
        "familiar_connect.voice.turn_detection.smart_turn.ort.InferenceSession"
    ) as cls:
        instance = MagicMock()
        # Default: 2-class logits leaning incomplete.
        instance.run.return_value = (np.array([[2.0, 0.0]], dtype=np.float32),)
        cls.return_value = instance
        yield instance


class TestCompletionProbability:
    def test_softmax_2class_logits(self, fake_session: MagicMock) -> None:
        """``[incomplete, complete]`` logits → softmax class-1 probability."""
        # logits [0, 2] → softmax → [≈0.119, ≈0.881]
        fake_session.run.return_value = (np.array([[0.0, 2.0]], dtype=np.float32),)
        det = SmartTurnDetector(Path("/fake/m.onnx"))
        prob = det.completion_probability(_pcm(8000))
        assert prob == pytest.approx(0.881, abs=1e-3)

    def test_sigmoid_1class_logit(self, fake_session: MagicMock) -> None:
        """Single-output sigmoid head → ``1/(1+exp(-x))``."""
        fake_session.run.return_value = (np.array([[1.0]], dtype=np.float32),)
        det = SmartTurnDetector(Path("/fake/m.onnx"))
        prob = det.completion_probability(_pcm(8000))
        # sigmoid(1) ≈ 0.731
        assert prob == pytest.approx(0.731, abs=1e-3)

    def test_input_is_float32_normalised(self, fake_session: MagicMock) -> None:
        det = SmartTurnDetector(Path("/fake/m.onnx"))
        det.completion_probability(_pcm(1000, amplitude=16384))
        feed = fake_session.run.call_args[0][1]
        audio = next(iter(feed.values()))
        assert audio.dtype == np.float32
        assert audio.shape == (1, 1000)
        assert audio[0, 0] == pytest.approx(16384 / 32768.0, abs=1e-4)

    def test_truncates_to_max_duration(self, fake_session: MagicMock) -> None:
        """Long buffers clamp to the most-recent ``max_duration_s`` window."""
        det = SmartTurnDetector(
            Path("/fake/m.onnx"), sample_rate=16000, max_duration_s=1.0
        )
        # 2 seconds (32000 samples) — should clamp to most-recent 16000.
        det.completion_probability(_pcm(32000))
        feed = fake_session.run.call_args[0][1]
        audio = next(iter(feed.values()))
        assert audio.shape == (1, 16000)


class TestThreshold:
    def test_is_complete_above_threshold(self, fake_session: MagicMock) -> None:
        fake_session.run.return_value = (np.array([[0.0, 5.0]], dtype=np.float32),)
        det = SmartTurnDetector(Path("/fake/m.onnx"), threshold=0.5)
        assert det.is_complete(_pcm(1000)) is True

    def test_is_not_complete_below_threshold(self, fake_session: MagicMock) -> None:
        fake_session.run.return_value = (np.array([[5.0, 0.0]], dtype=np.float32),)
        det = SmartTurnDetector(Path("/fake/m.onnx"), threshold=0.5)
        assert det.is_complete(_pcm(1000)) is False


class TestRejection:
    def test_rejects_unexpected_output_shape(self, fake_session: MagicMock) -> None:
        """Three-class output → unsupported, surface clearly."""
        fake_session.run.return_value = (np.array([[0.1, 0.2, 0.3]], dtype=np.float32),)
        det = SmartTurnDetector(Path("/fake/m.onnx"))
        with pytest.raises(ValueError, match="logits shape"):
            det.completion_probability(_pcm(1000))
