"""Local turn detection — TEN-VAD + Smart Turn v3 + endpointer state machine.

Phase 1 landed the wrappers; phase 2 added :class:`UtteranceEndpointer`
composing them into a per-user state machine. Audio pump in
``bot._start_voice_intake`` forks PCM into both transcriber and
endpointer when ``[providers.turn_detection].strategy = "ten+smart_turn"``;
on turn-complete verdict, endpointer calls ``transcriber.finalize()``.
See ``docs/architecture/voice-pipeline.md``.
"""

from familiar_connect.voice.turn_detection.endpointer import UtteranceEndpointer
from familiar_connect.voice.turn_detection.factory import (
    LocalTurnDetector,
    create_local_turn_detector,
)
from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector
from familiar_connect.voice.turn_detection.ten_vad import TenVAD

__all__ = [
    "LocalTurnDetector",
    "SmartTurnDetector",
    "TenVAD",
    "UtteranceEndpointer",
    "create_local_turn_detector",
]
