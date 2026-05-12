"""Tests for :class:`familiar_connect.sources.discord_text.DiscordTextSource`."""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.bus import InProcessEventBus
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.identity import Author
from familiar_connect.sources.discord_text import DiscordTextSource


def _author(
    user_id: str = "42",
    display_name: str = "Alice",
) -> Author:
    return Author(
        platform="discord",
        user_id=user_id,
        username="alice",
        display_name=display_name,
    )


class TestPublishText:
    @pytest.mark.asyncio
    async def test_publishes_on_topic(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        source = DiscordTextSource(bus=bus, familiar_id="fam")

        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_DISCORD_TEXT,)):
                received.append(ev)
                return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await source.publish_text(
            channel_id=111,
            guild_id=222,
            author=_author(),
            content="hello",
        )
        await asyncio.wait_for(task, timeout=1.0)
        await bus.shutdown()

        assert len(received) == 1
        ev = received[0]
        assert ev.topic == TOPIC_DISCORD_TEXT
        assert ev.session_id == "discord:111"
        assert ev.payload["channel_id"] == 111
        assert ev.payload["guild_id"] == 222
        assert ev.payload["content"] == "hello"
        assert ev.payload["author"].display_name == "Alice"
        assert ev.payload["familiar_id"] == "fam"

    @pytest.mark.asyncio
    async def test_sequence_number_monotonic(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        source = DiscordTextSource(bus=bus, familiar_id="fam")
        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_DISCORD_TEXT,)):
                received.append(ev)
                if len(received) == 3:
                    return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        for content in ("a", "b", "c"):
            await source.publish_text(
                channel_id=1,
                guild_id=None,
                author=_author(),
                content=content,
            )
        await asyncio.wait_for(task, timeout=1.0)
        await bus.shutdown()

        seqs = [e.sequence_number for e in received]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 3

    @pytest.mark.asyncio
    async def test_carries_message_id_reply_and_mentions(self) -> None:
        """Discord's message_id, reply target, and mentions must round-trip."""
        bus = InProcessEventBus()
        await bus.start()
        source = DiscordTextSource(bus=bus, familiar_id="fam")
        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_DISCORD_TEXT,)):
                received.append(ev)
                return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        bob = _author(user_id="222", display_name="Bob")
        await source.publish_text(
            channel_id=111,
            guild_id=222,
            author=_author(),
            content="hey @bob",
            message_id="msg-9999",
            reply_to_message_id="msg-prev",
            mentions=(bob,),
        )
        await asyncio.wait_for(task, timeout=1.0)
        await bus.shutdown()

        ev = received[0]
        assert ev.payload["message_id"] == "msg-9999"
        assert ev.payload["reply_to_message_id"] == "msg-prev"
        assert ev.payload["mentions"] == (bob,)

    @pytest.mark.asyncio
    async def test_message_id_reply_and_mentions_default(self) -> None:
        """Backwards-compatible defaults: callers can omit the new fields."""
        bus = InProcessEventBus()
        await bus.start()
        source = DiscordTextSource(bus=bus, familiar_id="fam")
        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_DISCORD_TEXT,)):
                received.append(ev)
                return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await source.publish_text(
            channel_id=111,
            guild_id=222,
            author=_author(),
            content="hello",
        )
        await asyncio.wait_for(task, timeout=1.0)
        await bus.shutdown()

        ev = received[0]
        assert ev.payload["message_id"] is None
        assert ev.payload["reply_to_message_id"] is None
        assert ev.payload["mentions"] == ()

    @pytest.mark.asyncio
    async def test_turn_id_equals_event_id_for_source_events(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        source = DiscordTextSource(bus=bus, familiar_id="fam")
        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_DISCORD_TEXT,)):
                received.append(ev)
                return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await source.publish_text(
            channel_id=1,
            guild_id=None,
            author=_author(),
            content="x",
        )
        await asyncio.wait_for(task, timeout=1.0)
        await bus.shutdown()

        ev = received[0]
        # source events are the turn's root; turn_id == event_id
        assert ev.turn_id == ev.event_id
