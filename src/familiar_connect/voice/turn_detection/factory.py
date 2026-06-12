"""Factory for V1 phase 2 local turn-detection chain.

Bundles SmartTurn ONNX path + thresholds. ``make_endpointer`` called
once per Discord user (TenVAD is per-user — its native handle
accumulates state across frames; SmartTurn classifier is stateless
and shareable).

TEN-VAD ships its ONNX model inside the ``ten-vad`` Python package;
SmartTurn weights fetched from HuggingFace on first use via
``hf_hub_download``. Hub cache (``~/.cache/huggingface``) makes
subsequent runs filesystem-only; ``HF_HUB_OFFLINE=1`` forces
cache-only for air-gapped deployments.

Knobs come from ``[providers.turn_detection.local]`` in
``character.toml``. Returns ``None`` (not raises) when SmartTurn
weights can't be resolved — bot falls back to Deepgram-only endpointing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.voice.turn_detection.endpointer import UtteranceEndpointer
from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector
from familiar_connect.voice.turn_detection.ten_vad import TenVAD

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.config import LocalTurnConfig

_logger = logging.getLogger(__name__)


@dataclass
class LocalTurnDetector:
    """Per-process bundle: shared SmartTurn + per-user TenVAD factory.

    TenVAD native handle is per-user; SmartTurn is stateless beyond
    the loaded ONNX session. Lazy-load SmartTurn on first
    ``make_endpointer`` so import-time cost stays free for processes
    without voice subscription.
    """

    smart_turn_path: Path
    silence_ms: int = 200
    speech_start_ms: int = 100
    smart_turn_threshold: float = 0.5
    vad_threshold: float = 0.5
    vad_hop_size: int = 256
    _smart_turn: SmartTurnDetector | None = field(default=None, init=False, repr=False)

    def make_endpointer(
        self,
        *,
        on_turn_complete: Callable[[bytes], Awaitable[None]],
    ) -> UtteranceEndpointer:
        """Build fresh per-user endpointer.

        TenVAD constructed fresh per call (stateful native handle);
        SmartTurn loaded once, shared.
        """
        if self._smart_turn is None:
            self._smart_turn = SmartTurnDetector(
                self.smart_turn_path, threshold=self.smart_turn_threshold
            )
        vad = TenVAD(hop_size=self.vad_hop_size, threshold=self.vad_threshold)
        return UtteranceEndpointer(
            vad=vad,
            smart_turn=self._smart_turn,
            on_turn_complete=on_turn_complete,
            silence_ms=self.silence_ms,
            speech_start_ms=self.speech_start_ms,
        )


def create_local_turn_detector(config: LocalTurnConfig) -> LocalTurnDetector | None:
    """Build :class:`LocalTurnDetector` from typed *config*.

    Pulls SmartTurn ONNX weights from HuggingFace
    (``config.smart_turn_repo_id`` / ``config.smart_turn_filename``);
    Hub cache covers offline reruns. Returns ``None`` (with warning)
    on any download/FS error — bot falls back to Deepgram-only
    endpointing rather than crashing.
    """
    # Lazy: huggingface_hub is the `local-turn` extra, not always installed
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ModuleNotFoundError:
        _logger.warning(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('local_turn_detection', 'disabled', vc=ls.LY)} "
            f"{ls.kv('reason', 'huggingface_hub_missing', vc=ls.LW)} "
            f"{ls.kv('hint', 'uv sync --extra local-turn', vc=ls.LW)}"
        )
        return None
    try:
        resolved = Path(
            hf_hub_download(
                repo_id=config.smart_turn_repo_id,
                filename=config.smart_turn_filename,
            )
        )
    except Exception as exc:  # noqa: BLE001 — broad: any HF/network/FS issue degrades
        _logger.warning(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('local_turn_detection', 'disabled', vc=ls.LY)} "
            f"{ls.kv('reason', 'smart_turn_download_failed', vc=ls.LW)} "
            f"{ls.kv('repo', config.smart_turn_repo_id, vc=ls.LW)} "
            f"{ls.kv('file', config.smart_turn_filename, vc=ls.LW)} "
            f"{ls.kv('exc', repr(exc), vc=ls.LW)}"
        )
        return None
    if not resolved.exists():
        # Cache rot — hf_hub_download returned a path that's gone
        _logger.warning(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('local_turn_detection', 'disabled', vc=ls.LY)} "
            f"{ls.kv('reason', 'smart_turn_cache_missing', vc=ls.LW)} "
            f"{ls.kv('path', str(resolved), vc=ls.LW)}"
        )
        return None
    detector = LocalTurnDetector(
        smart_turn_path=resolved,
        silence_ms=config.silence_ms,
        speech_start_ms=config.speech_start_ms,
        vad_threshold=config.vad_threshold,
        smart_turn_threshold=config.smart_turn_threshold,
        vad_hop_size=config.vad_hop_size,
    )
    _logger.info(
        f"{ls.tag('🎙️  Voice', ls.G)} "
        f"{ls.kv('local_turn_detection', 'enabled', vc=ls.LG)} "
        f"{ls.kv('vad', 'ten-vad', vc=ls.LG)} "
        f"{ls.kv('smart_turn', config.smart_turn_filename, vc=ls.LW)} "
        f"{ls.kv('hop_size', str(detector.vad_hop_size), vc=ls.LW)} "
        f"{ls.kv('silence_ms', str(detector.silence_ms), vc=ls.LW)} "
        f"{ls.kv('speech_start_ms', str(detector.speech_start_ms), vc=ls.LW)}"
    )
    return detector
