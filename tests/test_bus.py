"""Tests for :class:`familiar_connect.bus.bus.InProcessEventBus`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from familiar_connect.bus.bus import InProcessEventBus, Lifecycle
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.protocols import BackpressurePolicy


def _mk_event(*, topic: str = "t", seq: int = 0, payload: object = None) -> Event:
    return Event(
        event_id=f"e-{seq}",
        turn_id="turn-0",
        session_id="sess-0",
        parent_event_ids=(),
        topic=topic,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload=payload,
    )


class TestLifecycle:
    def test_starts_in_starting(self) -> None:
        bus = InProcessEventBus()
        assert bus.lifecycle is Lifecycle.STARTING

    @pytest.mark.asyncio
    async def test_running_after_start(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        assert bus.lifecycle is Lifecycle.RUNNING
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_draining_then_stopped_on_shutdown(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        await bus.shutdown()
        assert bus.lifecycle is Lifecycle.STOPPED


class TestFanout:
    @pytest.mark.asyncio
    async def test_single_subscriber_receives_published_event(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        received: list[Event] = []

        async def consume() -> None:
            async for ev in bus.subscribe(("t",)):
                received.append(ev)
                break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let consumer subscribe
        await bus.publish(_mk_event(topic="t", seq=0))
        await asyncio.wait_for(task, timeout=1.0)
        assert len(received) == 1
        assert received[0].sequence_number == 0
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_two_subscribers_each_get_every_event(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        got_a: list[Event] = []
        got_b: list[Event] = []

        async def consume(dest: list[Event]) -> None:
            async for ev in bus.subscribe(("t",)):
                dest.append(ev)
                if len(dest) == 3:
                    return

        a = asyncio.create_task(consume(got_a))
        b = asyncio.create_task(consume(got_b))
        await asyncio.sleep(0)
        for i in range(3):
            await bus.publish(_mk_event(topic="t", seq=i))
        await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)
        assert [e.sequence_number for e in got_a] == [0, 1, 2]
        assert [e.sequence_number for e in got_b] == [0, 1, 2]
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_topic_isolation(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        got: list[Event] = []

        async def consume() -> None:
            async for ev in bus.subscribe(("wanted",)):
                got.append(ev)
                if len(got) == 1:
                    return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await bus.publish(_mk_event(topic="ignored", seq=99))
        await bus.publish(_mk_event(topic="wanted", seq=1))
        await asyncio.wait_for(task, timeout=1.0)
        assert [e.sequence_number for e in got] == [1]
        await bus.shutdown()


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_drop_oldest_keeps_newest(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        # never consume; fill queue; newest must survive.
        got: list[Event] = []
        sub = bus.subscribe(
            ("audio",),
            policy=BackpressurePolicy.DROP_OLDEST,
            maxsize=2,
        )
        for i in range(5):
            await bus.publish(_mk_event(topic="audio", seq=i))
        # now drain

        async def drain() -> None:
            async for ev in sub:
                got.append(ev)
                if len(got) == 2:
                    return

        await asyncio.wait_for(drain(), timeout=1.0)
        assert [e.sequence_number for e in got] == [3, 4]
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_drop_newest_keeps_oldest(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        got: list[Event] = []
        sub = bus.subscribe(
            ("t",),
            policy=BackpressurePolicy.DROP_NEWEST,
            maxsize=2,
        )
        for i in range(5):
            await bus.publish(_mk_event(topic="t", seq=i))

        async def drain() -> None:
            async for ev in sub:
                got.append(ev)
                if len(got) == 2:
                    return

        await asyncio.wait_for(drain(), timeout=1.0)
        assert [e.sequence_number for e in got] == [0, 1]
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_block_policy_waits_on_slow_consumer(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        got: list[Event] = []
        sub = bus.subscribe(
            ("t",),
            policy=BackpressurePolicy.BLOCK,
            maxsize=1,
        )

        async def slow_consumer() -> None:
            async for ev in sub:
                got.append(ev)
                await asyncio.sleep(0.02)
                if len(got) == 3:
                    return

        task = asyncio.create_task(slow_consumer())
        await asyncio.sleep(0)
        # publisher must block; this should take at least ~2 sleeps
        start = asyncio.get_running_loop().time()
        for i in range(3):
            await bus.publish(_mk_event(topic="t", seq=i))
        await asyncio.wait_for(task, timeout=1.0)
        elapsed = asyncio.get_running_loop().time() - start
        assert [e.sequence_number for e in got] == [0, 1, 2]
        # two sleeps of 20ms → at least 30ms elapsed
        assert elapsed >= 0.03
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_unbounded_never_drops(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        got: list[Event] = []
        sub = bus.subscribe(
            ("t",),
            policy=BackpressurePolicy.UNBOUNDED,
        )
        for i in range(100):
            await bus.publish(_mk_event(topic="t", seq=i))

        async def drain() -> None:
            async for ev in sub:
                got.append(ev)
                if len(got) == 100:
                    return

        await asyncio.wait_for(drain(), timeout=1.0)
        assert [e.sequence_number for e in got] == list(range(100))
        await bus.shutdown()


class TestShutdown:
    @pytest.mark.asyncio
    async def test_subscribers_exit_cleanly_on_shutdown(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        seen = 0

        async def consume() -> int:
            nonlocal seen
            async for _ev in bus.subscribe(("t",)):
                seen += 1
            return seen

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await bus.publish(_mk_event(topic="t", seq=0))
        await asyncio.sleep(0.01)
        await bus.shutdown()
        await asyncio.wait_for(task, timeout=1.0)
        assert seen == 1
