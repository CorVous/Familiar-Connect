"""Discord text message → bus event.

Not a pull-loop source; ``py-cord`` owns event loop. Bot
``on_message`` hands off to :meth:`DiscordTextSource.publish_text`
which builds envelope and publishes. See plan § Design.2.
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
    """Publishes ``discord.text`` events for messages bot observes."""

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
        message_id: str | None = None,
        reply_to_message_id: str | None = None,
        mentions: tuple[Author, ...] = (),
        images: dict[str, str] | None = None,
        pings_bot: bool = False,
    ) -> Event:
        """Construct + publish text event; return envelope.

        ``message_id`` — platform-native message id (Discord snowflake
        as string). ``reply_to_message_id`` set when
        ``discord.Message.reference`` carried parent message.
        ``mentions`` — users the message pinged (already converted to
        :class:`Author`); empty when no user mentions.
        ``images`` — img_id → URL map for view_image tool; empty when
        no images detected in message.
        ``pings_bot`` — bot user in ``message.mentions`` (``<@id>``
        mentions AND reply-pings; roles/@everyone excluded). Needed
        because ``mentions`` filters bot users and reply-pings carry
        no mention string in content.
        """
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
                "message_id": message_id,
                "reply_to_message_id": reply_to_message_id,
                "mentions": mentions,
                "images": images or {},
                "pings_bot": pings_bot,
            },
        )
        await self._bus.publish(ev)
        return ev
