"""Factory for the V1 phase 2 local turn-detection chain.

Bundles ONNX paths + thresholds. ``make_endpointer`` is called once
per Discord user (the SileroVAD instance is per-user because it carries
hidden state across 32 ms frames; the SmartTurn classifier is stateless
and can be shared).

Env-driven setup lives in :func:`create_local_turn_detector_from_env`
so ``commands/run.py`` can mirror the existing ``create_transcriber_from_env``
pattern. Returns ``None`` (rather than raising) when the feature flag
is off or model files are missing — the bot falls back to Deepgram-
only endpointing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.voice.turn_detection.endpointer import UtteranceEndpointer
from familiar_connect.voice.turn_detection.silero_vad import SileroVAD
from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = logging.getLogger(__name__)

# default model paths under the repo's gitignored ``data/models/`` tree
DEFAULT_SILERO_PATH: Path = Path("data/models/silero_vad.onnx")
DEFAULT_SMART_TURN_PATH: Path = Path("data/models/smart-turn-v3.onnx")


@dataclass
class LocalTurnDetector:
    """Per-process bundle: shared SmartTurn + per-user SileroVAD factory.

    SileroVAD's hidden state is per-user; SmartTurn is stateless beyond
    the loaded ONNX session. Lazy-load both on first ``make_endpointer``
    so import-time cost stays free for processes without a voice
    subscription.
    """

    silero_path: Path
    smart_turn_path: Path
    silence_ms: int = 200
    speech_start_ms: int = 100
    smart_turn_threshold: float = 0.5
    vad_threshold: float = 0.5
    _smart_turn: SmartTurnDetector | None = field(default=None, init=False, repr=False)

    def make_endpointer(
        self,
        *,
        on_turn_complete: Callable[[bytes], Awaitable[None]],
    ) -> UtteranceEndpointer:
        """Build a fresh per-user endpointer.

        SileroVAD is constructed fresh per call (stateful); SmartTurn
        is loaded once and shared.
        """
        if self._smart_turn is None:
            self._smart_turn = SmartTurnDetector(
                self.smart_turn_path, threshold=self.smart_turn_threshold
            )
        vad = SileroVAD(self.silero_path, threshold=self.vad_threshold)
        return UtteranceEndpointer(
            vad=vad,
            smart_turn=self._smart_turn,
            on_turn_complete=on_turn_complete,
            silence_ms=self.silence_ms,
            speech_start_ms=self.speech_start_ms,
        )


def _env_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(raw: str | None, default: int) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(raw: str | None, default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def create_local_turn_detector_from_env() -> LocalTurnDetector | None:
    """Build :class:`LocalTurnDetector` from env, or ``None`` if disabled.

    Knobs:

    - ``LOCAL_TURN_DETECTION`` — ``1/true/yes/on`` to enable.
    - ``SILERO_VAD_MODEL_PATH`` — default ``data/models/silero_vad.onnx``.
    - ``SMART_TURN_MODEL_PATH`` — default ``data/models/smart-turn-v3.onnx``.
    - ``LOCAL_TURN_SILENCE_MS`` — default ``200``. Silence after speech
      before SmartTurn fires.
    - ``LOCAL_TURN_SPEECH_START_MS`` — default ``100``. Consecutive
      speech before "speaking" latches.
    - ``LOCAL_TURN_VAD_THRESHOLD`` — default ``0.5``.
    - ``LOCAL_TURN_SMART_TURN_THRESHOLD`` — default ``0.5``.

    Returns ``None`` (with a warning) if the feature is enabled but
    a model file is missing — the bot keeps running on Deepgram-only.
    """
    if not _env_bool(os.environ.get("LOCAL_TURN_DETECTION"), default=False):
        return None
    silero = Path(os.environ.get("SILERO_VAD_MODEL_PATH") or str(DEFAULT_SILERO_PATH))
    smart_turn = Path(
        os.environ.get("SMART_TURN_MODEL_PATH") or str(DEFAULT_SMART_TURN_PATH)
    )
    if not silero.exists() or not smart_turn.exists():
        _logger.warning(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('local_turn_detection', 'disabled', vc=ls.LY)} "
            f"{ls.kv('reason', 'model_files_missing', vc=ls.LW)} "
            f"{ls.kv('silero', str(silero), vc=ls.LW)} "
            f"{ls.kv('smart_turn', str(smart_turn), vc=ls.LW)}"
        )
        return None
    detector = LocalTurnDetector(
        silero_path=silero,
        smart_turn_path=smart_turn,
        silence_ms=_env_int(os.environ.get("LOCAL_TURN_SILENCE_MS"), 200),
        speech_start_ms=_env_int(os.environ.get("LOCAL_TURN_SPEECH_START_MS"), 100),
        vad_threshold=_env_float(os.environ.get("LOCAL_TURN_VAD_THRESHOLD"), 0.5),
        smart_turn_threshold=_env_float(
            os.environ.get("LOCAL_TURN_SMART_TURN_THRESHOLD"), 0.5
        ),
    )
    _logger.info(
        f"{ls.tag('🎙️  Voice', ls.G)} "
        f"{ls.kv('local_turn_detection', 'enabled', vc=ls.LG)} "
        f"{ls.kv('silence_ms', str(detector.silence_ms), vc=ls.LW)} "
        f"{ls.kv('speech_start_ms', str(detector.speech_start_ms), vc=ls.LW)}"
    )
    return detector
