"""Tests for the local turn detector factory functions.

Covers the new ``create_local_turn_detector`` (TOML-activated, no env
gate) alongside the existing env-var gate path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from familiar_connect.voice.turn_detection.factory import (
    LocalTurnDetector,
    create_local_turn_detector,
    create_local_turn_detector_from_env,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateLocalTurnDetector:
    """``create_local_turn_detector`` — no LOCAL_TURN_DETECTION gate."""

    def test_returns_none_when_model_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-model.onnx"
        result = create_local_turn_detector(smart_turn_path=missing)
        assert result is None

    def test_returns_detector_when_model_present(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        result = create_local_turn_detector(smart_turn_path=model)
        assert isinstance(result, LocalTurnDetector)
        assert result.smart_turn_path == model

    def test_default_knobs_applied(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        result = create_local_turn_detector(smart_turn_path=model)
        assert result is not None
        assert result.silence_ms == 200
        assert result.speech_start_ms == 100
        assert result.vad_threshold == pytest.approx(0.5)
        assert result.smart_turn_threshold == pytest.approx(0.5)

    def test_env_knobs_override_defaults(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.onnx"
        model.write_bytes(b"fake")
        env = {
            "LOCAL_TURN_SILENCE_MS": "300",
            "LOCAL_TURN_SPEECH_START_MS": "150",
            "LOCAL_TURN_VAD_THRESHOLD": "0.7",
            "LOCAL_TURN_SMART_TURN_THRESHOLD": "0.6",
        }
        with patch.dict("os.environ", env, clear=False):
            result = create_local_turn_detector(smart_turn_path=model)
        assert result is not None
        assert result.silence_ms == 300
        assert result.speech_start_ms == 150
        assert result.vad_threshold == pytest.approx(0.7)
        assert result.smart_turn_threshold == pytest.approx(0.6)


class TestCreateLocalTurnDetectorFromEnv:
    """Existing env-gate path stays backward-compatible."""

    def test_returns_none_when_flag_off(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = create_local_turn_detector_from_env()
        assert result is None

    def test_returns_none_when_model_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such.onnx"
        with patch.dict(
            "os.environ",
            {
                "LOCAL_TURN_DETECTION": "1",
                "SMART_TURN_MODEL_PATH": str(missing),
            },
            clear=False,
        ):
            result = create_local_turn_detector_from_env()
        assert result is None

    def test_returns_detector_when_enabled_and_model_present(
        self, tmp_path: Path
    ) -> None:
        model = tmp_path / "smart-turn.onnx"
        model.write_bytes(b"fake")
        with patch.dict(
            "os.environ",
            {
                "LOCAL_TURN_DETECTION": "1",
                "SMART_TURN_MODEL_PATH": str(model),
            },
            clear=False,
        ):
            result = create_local_turn_detector_from_env()
        assert isinstance(result, LocalTurnDetector)
