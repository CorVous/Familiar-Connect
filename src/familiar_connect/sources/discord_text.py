"""Discord text message → bus event.

Not a pull-loop source; ``py-cord`` owns the event loop. Bot
``on_message`` hands off to :meth:`DiscordTextSource.publish_text`
which constructs the envelope and publishes. See plan § Design.2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT

if TYPE_CHECKING:
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.identity import Author


class DiscordTextSource:
    """Publishes ``discord.text`` events for messages the bot observes."""

    name: str = "discord-text"

    def __init__(self, *, bus: EventBus, familiar_id: str) -> None:
        self._bus = bus
        self._familiar_id = familiar_id
        self._seq = 0

    async def publish_text(
        self,
        *,
        channel_id: int,
        guild_id: int | None,
        author: Author,
        content: str,
    ) -> Event:
        """Construct and publish a text event; return the envelope."""
        self._seq += 1
        event_id = f"discord-text-{uuid4().hex[:12]}"
        ev = Event(
            event_id=event_id,
            turn_id=event_id,  # source event: turn_id == event_id
            session_id=f"discord:{channel_id}",
            parent_event_ids=(),
            topic=TOPIC_DISCORD_TEXT,
            timestamp=datetime.now(tz=UTC),
            sequence_number=self._seq,
            payload={
                "familiar_id": self._familiar_id,
                "channel_id": channel_id,
                "guild_id": guild_id,
                "author": author,
                "content": content,
            },
        )
        await self._bus.publish(ev)
        return ev
