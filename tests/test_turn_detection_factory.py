"""Tests for ``create_local_turn_detector``.

Knobs come from a typed :class:`LocalTurnConfig`. Smart Turn's ONNX
weights are fetched on first use via ``huggingface_hub.hf_hub_download``;
its cache makes subsequent runs filesystem-only. Tests mock the
download so CI never hits the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from familiar_connect.config import LocalTurnConfig
from familiar_connect.voice.turn_detection.factory import (
    LocalTurnDetector,
    create_local_turn_detector,
)

if TYPE_CHECKING:
    from pathlib import Path


_HF_DOWNLOAD = "familiar_connect.voice.turn_detection.factory.hf_hub_download"


class TestCreateLocalTurnDetector:
    def test_returns_detector_with_downloaded_path(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.2-cpu.onnx"
        model.write_bytes(b"fake")
        with patch(_HF_DOWNLOAD, return_value=str(model)) as mock_dl:
            result = create_local_turn_detector(LocalTurnConfig())
        assert isinstance(result, LocalTurnDetector)
        assert result.smart_turn_path == model
        # default repo + filename flow through verbatim
        mock_dl.assert_called_once_with(
            repo_id="pipecat-ai/smart-turn-v3",
            filename="smart-turn-v3.2-cpu.onnx",
        )

    def test_returns_none_when_download_fails(self) -> None:
        with patch(_HF_DOWNLOAD, side_effect=OSError("offline")):
            result = create_local_turn_detector(LocalTurnConfig())
        assert result is None

    def test_returns_none_when_downloaded_path_missing(self) -> None:
        # hf_hub_download returns a path that no longer exists (cache rot).
        with patch(_HF_DOWNLOAD, return_value="/nonexistent/x.onnx"):
            result = create_local_turn_detector(LocalTurnConfig())
        assert result is None

    def test_default_knobs_applied(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.2-cpu.onnx"
        model.write_bytes(b"fake")
        with patch(_HF_DOWNLOAD, return_value=str(model)):
            result = create_local_turn_detector(LocalTurnConfig())
        assert result is not None
        assert result.silence_ms == 200
        assert result.speech_start_ms == 100
        assert result.vad_threshold == pytest.approx(0.5)
        assert result.smart_turn_threshold == pytest.approx(0.5)
        assert result.vad_hop_size == 256

    def test_passes_config_through(self, tmp_path: Path) -> None:
        model = tmp_path / "smart-turn-v3.2-gpu.onnx"
        model.write_bytes(b"fake")
        cfg = LocalTurnConfig(
            smart_turn_repo_id="pipecat-ai/smart-turn-v3",
            smart_turn_filename="smart-turn-v3.2-gpu.onnx",
            silence_ms=300,
            speech_start_ms=150,
            vad_threshold=0.7,
            smart_turn_threshold=0.6,
            vad_hop_size=160,
        )
        with patch(_HF_DOWNLOAD, return_value=str(model)) as mock_dl:
            result = create_local_turn_detector(cfg)
        assert result is not None
        assert result.silence_ms == 300
        assert result.speech_start_ms == 150
        assert result.vad_threshold == pytest.approx(0.7)
        assert result.smart_turn_threshold == pytest.approx(0.6)
        assert result.vad_hop_size == 160
        mock_dl.assert_called_once_with(
            repo_id="pipecat-ai/smart-turn-v3",
            filename="smart-turn-v3.2-gpu.onnx",
        )

    def test_custom_repo_id_flows_through(self, tmp_path: Path) -> None:
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"fake")
        cfg = LocalTurnConfig(
            smart_turn_repo_id="acme/custom-turn",
            smart_turn_filename="custom.onnx",
        )
        with patch(_HF_DOWNLOAD, return_value=str(model)) as mock_dl:
            create_local_turn_detector(cfg)
        mock_dl.assert_called_once_with(
            repo_id="acme/custom-turn",
            filename="custom.onnx",
        )
