"""Event envelope + per-turn cancel scope.

:class:`Event` — immutable, topic-addressed envelope flowing over the
bus. ``parent_event_ids`` tuple carries lineage (session-lifetime
provenance); derived SQLite rows carry ``source_turn_ids`` (forever).

:class:`TurnScope` — one per inbound turn. ``cancel()`` is idempotent;
downstream tasks await :meth:`wait_cancelled` or check
:meth:`is_cancelled` at await points. See plan § Design.3.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable topic-addressed envelope.

    :param event_id: unique per publish
    :param turn_id: scopes derived work; matches enclosing
        :class:`TurnScope`
    :param session_id: usually a channel id
    :param parent_event_ids: lineage; empty for source events
    :param topic: routing key (see :mod:`familiar_connect.bus.topics`)
    :param sequence_number: monotonic per session; consumers that
        care about ordering sort or reject
    """

    event_id: str
    turn_id: str
    session_id: str
    parent_event_ids: tuple[str, ...]
    topic: str
    timestamp: datetime
    sequence_number: int
    payload: Any


@dataclass(slots=True)
class TurnScope:
    """Per-turn cancel handle. Not frozen — ``_cancel_event`` mutates."""

    turn_id: str
    session_id: str
    started_at: float
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    async def wait_cancelled(self) -> None:
        """Block until :meth:`cancel` is called."""
        await self._cancel_event.wait()
