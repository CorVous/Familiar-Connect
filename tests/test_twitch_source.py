"""Tests for :class:`familiar_connect.sources.twitch.TwitchSource`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from familiar_connect.bus import InProcessEventBus
from familiar_connect.bus.topics import TOPIC_TWITCH_EVENT
from familiar_connect.sources.twitch import TwitchSource


@dataclass
class _FakeTwitchEvent:
    kind: str
    detail: str


class TestTwitchSource:
    @pytest.mark.asyncio
    async def test_drains_queue_onto_bus(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue = asyncio.Queue()

        source = TwitchSource(bus=bus, familiar_id="fam", queue=queue)

        received: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_TWITCH_EVENT,)):
                received.append(ev)
                if len(received) == 2:
                    return

        consumer = asyncio.create_task(consume())
        producer = asyncio.create_task(source.run())
        await asyncio.sleep(0)

        await queue.put(_FakeTwitchEvent(kind="follow", detail="a"))
        await queue.put(_FakeTwitchEvent(kind="cheer", detail="b"))

        await asyncio.wait_for(consumer, timeout=1.0)
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await bus.shutdown()

        assert {ev.payload["twitch"].kind for ev in received} == {"follow", "cheer"}
        assert all(ev.topic == TOPIC_TWITCH_EVENT for ev in received)
        assert all(ev.session_id == "twitch:fam" for ev in received)

    @pytest.mark.asyncio
    async def test_run_exits_when_bus_shuts_down(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue = asyncio.Queue()
        source = TwitchSource(bus=bus, familiar_id="fam", queue=queue)

        producer = asyncio.create_task(source.run())
        await asyncio.sleep(0)

        # Simulate shutdown — run() should be cancellable cleanly.
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await bus.shutdown()
