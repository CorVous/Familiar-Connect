"""Tests for the ``set_alarm`` / ``cancel_alarm`` tool builders."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.tools.alarm import (
    build_alarm_tool,
    build_cancel_alarm_tool,
)
from familiar_connect.tools.registry import ToolContext
from familiar_connect.tools.scheduler import AlarmScheduler

if TYPE_CHECKING:
    from pathlib import Path


_FAMILIAR = "aria"


def _ctx(history: AsyncHistoryStore, bus: InProcessEventBus) -> ToolContext:
    return ToolContext(
        familiar_id=_FAMILIAR,
        channel_id=42,
        channel_kind="text",
        turn_id="turn-test",
        history=history,
        bus=bus,
    )


@pytest.mark.asyncio
async def test_set_alarm_with_delay_seconds_inserts_row(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler
        tool = build_alarm_tool(scheduler)
        result = await tool.handler({"reason": "ping", "delay_seconds": 30}, ctx)
        body = json.loads(result)
        assert "alarm_id" in body
        assert "scheduled_at" in body
        assert body["ack"] == "ok"

        pending = store.sync.list_pending_alarms(familiar_id=_FAMILIAR)
        assert len(pending) == 1
        assert pending[0]["reason"] == "ping"
        # cancel so shutdown doesn't fire the alarm
        await scheduler.cancel(alarm_id=body["alarm_id"])
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_set_alarm_with_iso_when_inserts_row(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler
        future = (datetime.now(tz=UTC) + timedelta(minutes=1)).isoformat()
        tool = build_alarm_tool(scheduler)
        result = await tool.handler({"reason": "later", "when": future}, ctx)
        body = json.loads(result)
        assert body["scheduled_at"] == future
        await scheduler.cancel(alarm_id=body["alarm_id"])
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_set_alarm_rejects_past_timestamp(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler
        past = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
        tool = build_alarm_tool(scheduler)
        result = await tool.handler({"reason": "rip", "when": past}, ctx)
        body = json.loads(result)
        assert "error" in body
        assert "past" in body["error"].lower()
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_set_alarm_rejects_missing_reason(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler
        tool = build_alarm_tool(scheduler)
        result = await tool.handler({"delay_seconds": 10}, ctx)
        body = json.loads(result)
        assert "error" in body
        assert "reason" in body["error"].lower()
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_set_alarm_uses_caller_channel_from_ctx(tmp_path: Path) -> None:
    """The alarm should fire back into the channel the user spoke in."""
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = ToolContext(
            familiar_id=_FAMILIAR,
            channel_id=777,
            channel_kind="voice",
            turn_id="turn-x",
            history=store,
            bus=bus,
            scheduler=scheduler,
        )
        tool = build_alarm_tool(scheduler)
        result = await tool.handler({"reason": "echo", "delay_seconds": 60}, ctx)
        body = json.loads(result)

        pending = store.sync.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending[0]["channel_id"] == 777
        assert pending[0]["channel_kind"] == "voice"
        await scheduler.cancel(alarm_id=body["alarm_id"])
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_cancel_alarm_tool_cancels_pending(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler

        # First, set
        set_tool = build_alarm_tool(scheduler)
        set_result = json.loads(
            await set_tool.handler({"reason": "x", "delay_seconds": 60}, ctx)
        )
        alarm_id = set_result["alarm_id"]

        # Then cancel
        cancel_tool = build_cancel_alarm_tool(scheduler)
        cancel_result = json.loads(
            await cancel_tool.handler({"alarm_id": alarm_id}, ctx)
        )
        assert cancel_result["ack"] == "ok"
        assert store.sync.list_pending_alarms(familiar_id=_FAMILIAR) == []
    finally:
        await scheduler.shutdown()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_cancel_unknown_returns_error(tmp_path: Path) -> None:
    store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
    bus = InProcessEventBus()
    await bus.start()
    scheduler = AlarmScheduler(history=store, bus=bus, familiar_id=_FAMILIAR)
    await scheduler.start()
    try:
        ctx = _ctx(store, bus)
        ctx.scheduler = scheduler
        tool = build_cancel_alarm_tool(scheduler)
        result = json.loads(await tool.handler({"alarm_id": "no-such"}, ctx))
        assert "error" in result
    finally:
        await scheduler.shutdown()
        await bus.shutdown()
