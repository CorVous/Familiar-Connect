"""Event processors. Phase 1 ships the debug processor only.

See plan § Design.2, plan § Rollout Phase 1.
"""

from __future__ import annotations

from familiar_connect.processors.debug_logger import DebugLoggerProcessor
from familiar_connect.processors.voice_responder import VoiceResponder

__all__ = ["DebugLoggerProcessor", "VoiceResponder"]
