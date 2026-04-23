"""Event processors. Phase 1 ships the debug processor only.

See plan § Design.2, plan § Rollout Phase 1.
"""

from __future__ import annotations

from familiar_connect.processors.debug_logger import DebugLoggerProcessor

__all__ = ["DebugLoggerProcessor"]
