"""Tests for :class:`AlarmScheduler`.

The scheduler:
* Persists alarms to ``HistoryStore`` (so they survive restart).
* Maintains an ``asyncio.Task`` per pending alarm that sleeps until
  ``scheduled_at`` then publishes :data:`TOPIC_ALARM_FIRED` on the bus.
* On startup, loads pending rows and reschedules them; past-due
  alarms fire immediately.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.bus.protocols import BackpressurePolicy
from familiar_connect.bus.topics import TOPIC_ALARM_FIRED
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.tools.scheduler import AlarmScheduler

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.bus.envelope import Event


_FAMILIAR = "aria"


async def _drain_one(
    bus: InProcessEventBus,
    *,
    timeout: float = 1.0,  # noqa: ASYNC109
) -> Event:
    """Wait for the next alarm.fired event with a timeout."""
    sub = bus.subscribe((TOPIC_ALARM_FIRED,), policy=BackpressurePolicy.UNBOUNDED)
    return await asyncio.wait_for(anext(sub), timeout=timeout)


def _make_store(tmp_path: Path) -> AsyncHistoryStore:
    return AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))


class TestAlarmSchedulerFires:
    @pytest.mark.asyncio
    async def test_alarm_fires_at_scheduled_time(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
        await scheduler.start()
        try:
            alarm_id = await scheduler.add(
                channel_id=42,
                channel_kind="text",
                scheduled_at=datetime.now(tz=UTC) + timedelta(milliseconds=80),
                reason="ping",
            )
            event = await _drain_one(bus, timeout=2.0)
            assert event.topic == TOPIC_ALARM_FIRED
            payload = event.payload
            assert payload["alarm_id"] == alarm_id
            assert payload["channel_id"] == 42
            assert payload["channel_kind"] == "text"
            assert payload["reason"] == "ping"
        finally:
            await scheduler.shutdown()
            await bus.shutdown()

        # marks fired in DB
        pending = store.sync.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending == []

    @pytest.mark.asyncio
    async def test_past_due_alarm_fires_immediately_on_start(
        self, tmp_path: Path
    ) -> None:
        store = _make_store(tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        # seed a row that's already overdue
        past = (datetime.now(tz=UTC) - timedelta(seconds=10)).isoformat()
        store.sync.insert_alarm(
            familiar_id=_FAMILIAR,
            channel_id=7,
            channel_kind="text",
            scheduled_at=past,
            reason="overdue",
        )
        scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
        await scheduler.start()
        try:
            event = await _drain_one(bus, timeout=2.0)
            assert event.payload["reason"] == "overdue"
        finally:
            await scheduler.shutdown()
            await bus.shutdown()


class TestAlarmSchedulerCancel:
    @pytest.mark.asyncio
    async def test_cancel_prevents_fire(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
        await scheduler.start()
        try:
            alarm_id = await scheduler.add(
                channel_id=1,
                channel_kind="text",
                scheduled_at=datetime.now(tz=UTC) + timedelta(seconds=30),
                reason="should-cancel",
            )
            ok = await scheduler.cancel(alarm_id=alarm_id)
            assert ok is True

            # No event should arrive in 200ms
            with pytest.raises(asyncio.TimeoutError):
                await _drain_one(bus, timeout=0.2)
        finally:
            await scheduler.shutdown()
            await bus.shutdown()

        pending = store.sync.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending == []

    @pytest.mark.asyncio
    async def test_cancel_unknown_returns_false(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
        await scheduler.start()
        try:
            ok = await scheduler.cancel(alarm_id="no-such-id")
            assert ok is False
        finally:
            await scheduler.shutdown()
            await bus.shutdown()


class TestAlarmSchedulerPayload:
    @pytest.mark.asyncio
    async def test_payload_carries_originating_turn_id(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
        await scheduler.start()
        try:
            await scheduler.add(
                channel_id=11,
                channel_kind="voice",
                scheduled_at=datetime.now(tz=UTC) + timedelta(milliseconds=50),
                reason="audit",
                originating_turn_id="turn-abc",
            )
            event = await _drain_one(bus, timeout=2.0)
            assert event.payload["originating_turn_id"] == "turn-abc"
            assert event.payload["channel_kind"] == "voice"
            # payload should be JSON-serializable
            json.dumps(event.payload)
        finally:
            await scheduler.shutdown()
            await bus.shutdown()
