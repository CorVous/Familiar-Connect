"""Local turn detection — Silero VAD + Smart Turn v3.

Phase 1 — wrappers only, not wired into the audio pipeline yet. See
``docs/architecture/voice-pipeline.md`` for the integration plan.
"""

from familiar_connect.voice.turn_detection.silero_vad import SileroVAD
from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector

__all__ = ["SileroVAD", "SmartTurnDetector"]
