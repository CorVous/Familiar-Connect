"""Local turn detection — TEN-VAD + Smart Turn v3 + endpointer state machine.

Phase 1 landed the wrappers; phase 2 adds :class:`UtteranceEndpointer`
which composes them into a per-user state machine. The audio pump in
``bot._start_voice_intake`` forks PCM into both Deepgram and the
endpointer when ``LOCAL_TURN_DETECTION=1``; on a turn-complete verdict
the endpointer calls ``transcriber.finalize()`` to flush Deepgram.
See ``docs/architecture/voice-pipeline.md``.
"""

from familiar_connect.voice.turn_detection.endpointer import UtteranceEndpointer
from familiar_connect.voice.turn_detection.factory import (
    LocalTurnDetector,
    create_local_turn_detector_from_env,
)
from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector
from familiar_connect.voice.turn_detection.ten_vad import TenVAD

__all__ = [
    "LocalTurnDetector",
    "SmartTurnDetector",
    "TenVAD",
    "UtteranceEndpointer",
    "create_local_turn_detector_from_env",
]
