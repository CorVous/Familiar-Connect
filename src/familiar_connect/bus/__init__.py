"""Event bus for streaming-first architecture.

Topic-keyed in-process fan-out. Sources publish events; processors
subscribe to topics. See ``docs/architecture/overview.md``.
"""

from __future__ import annotations

from familiar_connect.bus.bus import InProcessEventBus, Lifecycle
from familiar_connect.bus.envelope import Event, TurnScope
from familiar_connect.bus.protocols import (
    BackpressurePolicy,
    EventBus,
    Processor,
    StreamSource,
)
from familiar_connect.bus.router import TurnRouter

__all__ = [
    "BackpressurePolicy",
    "Event",
    "EventBus",
    "InProcessEventBus",
    "Lifecycle",
    "Processor",
    "StreamSource",
    "TurnRouter",
    "TurnScope",
]
