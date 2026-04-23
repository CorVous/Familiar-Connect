"""Debug processor: one log line per event on subscribed topics.

Phase-1 "is the bus alive?" signal. Colourised via
:mod:`familiar_connect.log_style`. Matches the Phase-1 deliverable in
the rollout plan — no user-visible behaviour beyond logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import Event
    from familiar_connect.bus.protocols import EventBus

_logger = logging.getLogger("familiar_connect.processors.debug_logger")


class DebugLoggerProcessor:
    """Log one line per event. Does not republish."""

    name: str = "debug-logger"

    def __init__(self, *, topics: tuple[str, ...]) -> None:
        self.topics: tuple[str, ...] = topics

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        payload_repr = (
            ls.trunc(repr(event.payload), 160) if event.payload is not None else "-"
        )
        _logger.info(
            f"{ls.tag('📥 Event', ls.LG)} "
            f"{ls.kv('topic', event.topic, vc=ls.LM)} "
            f"{ls.kv('event_id', event.event_id, vc=ls.LC)} "
            f"{ls.kv('turn_id', event.turn_id, vc=ls.LC)} "
            f"{ls.kv('session', event.session_id, vc=ls.LC)} "
            f"{ls.kv('seq', str(event.sequence_number), vc=ls.LY)} "
            f"{ls.kv('payload', payload_repr, vc=ls.LW)}"
        )
