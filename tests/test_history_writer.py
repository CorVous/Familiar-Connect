"""Tests for :class:`familiar_connect.processors.history_writer.HistoryWriter`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from familiar_connect.bus import InProcessEventBus
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.processors.history_writer import HistoryWriter


def _discord_text_event(
    *,
    event_id: str = "e-1",
    channel_id: int = 42,
    content: str = "hello",
    author: Author | None = None,
    seq: int = 1,
) -> Event:
    return Event(
        event_id=event_id,
        turn_id=event_id,
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "familiar_id": "fam",
            "channel_id": channel_id,
            "guild_id": 99,
            "author": author
            or Author(
                platform="discord",
                user_id="1",
                username="alice",
                display_name="Alice",
            ),
            "content": content,
        },
    )


class TestHistoryWriter:
    @pytest.mark.asyncio
    async def test_persists_discord_text_event(self) -> None:
        store = HistoryStore(":memory:")
        bus = InProcessEventBus()
        await bus.start()

        writer = HistoryWriter(store=store, familiar_id="fam")
        await writer.handle(_discord_text_event(content="hello"), bus)

        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        assert len(turns) == 1
        assert turns[0].content == "hello"
        assert turns[0].role == "user"
        assert turns[0].author is not None
        assert turns[0].author.display_name == "Alice"

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_dedups_on_event_id(self) -> None:
        store = HistoryStore(":memory:")
        bus = InProcessEventBus()
        await bus.start()

        writer = HistoryWriter(store=store, familiar_id="fam")
        ev = _discord_text_event(event_id="dup-1", content="once")
        await writer.handle(ev, bus)
        await writer.handle(ev, bus)  # same event_id

        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        assert len(turns) == 1

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_ignores_events_for_other_familiars(self) -> None:
        store = HistoryStore(":memory:")
        bus = InProcessEventBus()
        await bus.start()

        writer = HistoryWriter(store=store, familiar_id="fam")
        ev = _discord_text_event(content="not mine")
        # re-stamp for the other familiar
        payload = dict(ev.payload)
        payload["familiar_id"] = "other"
        other = Event(
            event_id="other-1",
            turn_id="other-1",
            session_id=ev.session_id,
            parent_event_ids=(),
            topic=ev.topic,
            timestamp=ev.timestamp,
            sequence_number=ev.sequence_number,
            payload=payload,
        )
        await writer.handle(other, bus)
        assert store.count(familiar_id="fam") == 0

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_ignores_empty_content(self) -> None:
        store = HistoryStore(":memory:")
        bus = InProcessEventBus()
        await bus.start()

        writer = HistoryWriter(store=store, familiar_id="fam")
        await writer.handle(_discord_text_event(content=""), bus)
        assert store.count(familiar_id="fam") == 0

        await bus.shutdown()

    def test_processor_surface(self) -> None:
        store = HistoryStore(":memory:")
        writer = HistoryWriter(store=store, familiar_id="fam")
        assert writer.name == "history-writer"
        assert TOPIC_DISCORD_TEXT in writer.topics
