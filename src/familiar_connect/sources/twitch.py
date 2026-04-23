"""Twitch event queue → bus event.

Consumes the ``asyncio.Queue`` produced by
:class:`familiar_connect.twitch_watcher.TwitchWatcher`, wraps each
event in an envelope, and publishes on
:data:`familiar_connect.bus.topics.TOPIC_TWITCH_EVENT`.

Unbounded topic policy per plan § Design.1 — Twitch volume is low and
dropping a cheer is costly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_TWITCH_EVENT

if TYPE_CHECKING:
    import asyncio

    from familiar_connect.bus.protocols import EventBus


class TwitchSource:
    """Drains a Twitch queue into the bus."""

    name: str = "twitch"

    def __init__(
        self,
        *,
        bus: EventBus,
        familiar_id: str,
        queue: asyncio.Queue[object],
    ) -> None:
        self._bus = bus
        self._familiar_id = familiar_id
        self._queue = queue
        self._seq = 0

    async def run(self) -> None:
        """Forever loop: drain queue, publish. Cancel to stop."""
        while True:
            twitch_event = await self._queue.get()
            await self._publish(twitch_event)

    async def _publish(self, twitch_event: object) -> Event:
        self._seq += 1
        event_id = f"twitch-{uuid4().hex[:12]}"
        ev = Event(
            event_id=event_id,
            turn_id=event_id,
            session_id=f"twitch:{self._familiar_id}",
            parent_event_ids=(),
            topic=TOPIC_TWITCH_EVENT,
            timestamp=datetime.now(tz=UTC),
            sequence_number=self._seq,
            payload={
                "familiar_id": self._familiar_id,
                "twitch": twitch_event,
            },
        )
        await self._bus.publish(ev)
        return ev
