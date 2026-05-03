"""Speech-to-text — Transcriber Protocol + backend selector.

V3 phase 1 lifted ``DeepgramTranscriber`` behind a Protocol so V3 phases
2/3 (Parakeet, FasterWhisper) drop in behind ``[providers.stt].backend``
without touching ``bot.py`` / ``sources/voice.py``. See
``docs/architecture/voice-pipeline.md``.
"""

from familiar_connect.stt.factory import create_transcriber
from familiar_connect.stt.protocol import (
    Transcriber,
    TranscriptionEvent,
    TranscriptionResult,
)

__all__ = [
    "Transcriber",
    "TranscriptionEvent",
    "TranscriptionResult",
    "create_transcriber",
]
