"""Tests for ``create_local_turn_detector``.

Knobs come from a typed :class:`LocalTurnConfig` — no env vars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import LocalTurnConfig
from familiar_connect.voice.turn_detection.factory import (
    LocalTurnDetector,
    create_local_turn_detector,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateLocalTurnDetector:
    def test_returns_none_when_model_missing(self, tmp_path: Path) -> None:
        cfg = LocalTurnConfig(smart_turn_model_path=str(tmp_path / "no-such.onnx"))
        assert create_local_turn_detector(cfg) is None

    def test_returns_detector_when_model_present(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        cfg = LocalTurnConfig(smart_turn_model_path=str(model))
        result = create_local_turn_detector(cfg)
        assert isinstance(result, LocalTurnDetector)
        assert result.smart_turn_path == model

    def test_default_knobs_applied(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        result = create_local_turn_detector(
            LocalTurnConfig(smart_turn_model_path=str(model))
        )
        assert result is not None
        assert result.silence_ms == 200
        assert result.speech_start_ms == 100
        assert result.vad_threshold == pytest.approx(0.5)
        assert result.smart_turn_threshold == pytest.approx(0.5)

    def test_passes_config_through(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        cfg = LocalTurnConfig(
            smart_turn_model_path=str(model),
            silence_ms=300,
            speech_start_ms=150,
            vad_threshold=0.7,
            smart_turn_threshold=0.6,
            vad_hop_size=160,
        )
        result = create_local_turn_detector(cfg)
        assert result is not None
        assert result.silence_ms == 300
        assert result.speech_start_ms == 150
        assert result.vad_threshold == pytest.approx(0.7)
        assert result.smart_turn_threshold == pytest.approx(0.6)
        assert result.vad_hop_size == 160
