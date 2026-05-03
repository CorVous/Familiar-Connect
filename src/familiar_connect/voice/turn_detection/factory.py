"""Factory for the V1 phase 2 local turn-detection chain.

Bundles SmartTurn ONNX path + thresholds. ``make_endpointer`` is called
once per Discord user (the TenVAD instance is per-user because its
native handle accumulates state across frames; the SmartTurn classifier
is stateless and can be shared).

TEN-VAD ships its ONNX model inside the ``ten-vad`` Python package, so
no VAD model path is required — only Smart Turn needs an on-disk file.

All knobs come from ``[providers.turn_detection.local]`` in
``character.toml``. Returns ``None`` (rather than raising) when the
SmartTurn model file is missing — the bot falls back to Deepgram-only
endpointing.
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

    TenVAD's native handle is per-user; SmartTurn is stateless beyond
    the loaded ONNX session. Lazy-load SmartTurn on first
    ``make_endpointer`` so import-time cost stays free for processes
    without a voice subscription.
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
        """Build a fresh per-user endpointer.

        TenVAD is constructed fresh per call (stateful native handle);
        SmartTurn is loaded once and shared.
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

    Returns ``None`` (with a warning) when the SmartTurn ONNX file at
    ``config.smart_turn_model_path`` is missing — the bot falls back
    to Deepgram-only endpointing rather than crashing.
    """
    smart_turn = Path(config.smart_turn_model_path)
    if not smart_turn.exists():
        _logger.warning(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('local_turn_detection', 'disabled', vc=ls.LY)} "
            f"{ls.kv('reason', 'smart_turn_model_missing', vc=ls.LW)} "
            f"{ls.kv('smart_turn', str(smart_turn), vc=ls.LW)}"
        )
        return None
    detector = LocalTurnDetector(
        smart_turn_path=smart_turn,
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
        f"{ls.kv('hop_size', str(detector.vad_hop_size), vc=ls.LW)} "
        f"{ls.kv('silence_ms', str(detector.silence_ms), vc=ls.LW)} "
        f"{ls.kv('speech_start_ms', str(detector.speech_start_ms), vc=ls.LW)}"
    )
    return detector
