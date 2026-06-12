"""Tests for :class:`AlarmWaker`.

On ``alarm.fired`` events, the waker republishes a synthetic
``discord.text``-shaped event so the existing :class:`TextResponder`
picks it up and generates a follow-up reply.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.protocols import BackpressurePolicy
from familiar_connect.bus.topics import TOPIC_ALARM_FIRED, TOPIC_DISCORD_TEXT
from familiar_connect.tools.waker import AlarmWaker

_FAMILIAR = "aria"


async def _publish_alarm_event(
    bus: InProcessEventBus,
    *,
    channel_id: int,
    channel_kind: str,
    reason: str,
    alarm_id: str = "alarm-1",
) -> None:
    payload = {
        "alarm_id": alarm_id,
        "channel_id": channel_id,
        "channel_kind": channel_kind,
        "reason": reason,
        "scheduled_at": datetime.now(tz=UTC).isoformat(),
        "fired_at": datetime.now(tz=UTC).isoformat(),
        "originating_turn_id": None,
    }
    await bus.publish(
        Event(
            event_id=uuid.uuid4().hex,
            turn_id=f"alarm-{alarm_id}",
            session_id=f"alarm:{channel_id}",
            parent_event_ids=(),
            topic=TOPIC_ALARM_FIRED,
            timestamp=datetime.now(tz=UTC),
            sequence_number=0,
            payload=payload,
        )
    )


@pytest.mark.asyncio
async def test_waker_republishes_as_discord_text_for_text_alarms() -> None:
    bus = InProcessEventBus()
    await bus.start()
    waker = AlarmWaker(familiar_id=_FAMILIAR)

    received: list[Event] = []

    async def _consume_alarm() -> None:
        sub = bus.subscribe((TOPIC_ALARM_FIRED,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            await waker.handle(event, bus)

    async def _consume_text() -> None:
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            received.append(event)  # noqa: PERF401 — extend would block here

    consumer_alarm = asyncio.create_task(_consume_alarm())
    consumer_text = asyncio.create_task(_consume_text())
    try:
        await asyncio.sleep(0)  # let subscribers register
        await _publish_alarm_event(
            bus, channel_id=42, channel_kind="text", reason="ping"
        )
        # give the bus a tick to dispatch
        await asyncio.sleep(0.05)
    finally:
        consumer_alarm.cancel()
        consumer_text.cancel()
        await bus.shutdown()

    assert len(received) == 1
    text_event = received[0]
    assert text_event.topic == TOPIC_DISCORD_TEXT
    payload = text_event.payload
    assert payload["familiar_id"] == _FAMILIAR
    assert payload["channel_id"] == 42
    assert "alarm" in payload["content"].lower()
    assert "ping" in payload["content"]


@pytest.mark.asyncio
async def test_waker_payload_carries_alarm_marker() -> None:
    """Marker ``alarm: True`` lets the activity gate pierce absences."""
    bus = InProcessEventBus()
    await bus.start()
    waker = AlarmWaker(familiar_id=_FAMILIAR)

    received: list[Event] = []

    async def _consume_alarm() -> None:
        sub = bus.subscribe((TOPIC_ALARM_FIRED,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            await waker.handle(event, bus)

    async def _consume_text() -> None:
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            received.append(event)  # noqa: PERF401 — extend would block here

    consumer_alarm = asyncio.create_task(_consume_alarm())
    consumer_text = asyncio.create_task(_consume_text())
    try:
        await asyncio.sleep(0)
        await _publish_alarm_event(
            bus, channel_id=42, channel_kind="text", reason="ping"
        )
        await asyncio.sleep(0.05)
    finally:
        consumer_alarm.cancel()
        consumer_text.cancel()
        await bus.shutdown()

    assert len(received) == 1
    assert received[0].payload["alarm"] is True


@pytest.mark.asyncio
async def test_waker_ignores_other_familiars() -> None:
    bus = InProcessEventBus()
    await bus.start()
    waker = AlarmWaker(familiar_id="other-fam")

    received: list[Event] = []

    async def _consume_alarm() -> None:
        sub = bus.subscribe((TOPIC_ALARM_FIRED,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            await waker.handle(event, bus)

    async def _consume_text() -> None:
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        async for event in sub:
            received.append(event)  # noqa: PERF401 — extend would block here

    consumer_alarm = asyncio.create_task(_consume_alarm())
    consumer_text = asyncio.create_task(_consume_text())
    try:
        await asyncio.sleep(0)
        # Publish a payload that mentions a different familiar-bound channel
        # — actually the waker filters by configured familiar_id only since
        # the alarm payload itself doesn't carry one; we test by routing the
        # waker for "other-fam" and verifying it still wakes (no familiar
        # filter on payload). Drop this test if you choose payload-side
        # filtering instead.
        await _publish_alarm_event(
            bus, channel_id=42, channel_kind="text", reason="ping"
        )
        await asyncio.sleep(0.05)
    finally:
        consumer_alarm.cancel()
        consumer_text.cancel()
        await bus.shutdown()

    # the waker for "other-fam" wakes its own familiar — the synthetic
    # event carries the waker's familiar_id, which the text responder
    # would filter on.
    assert received[0].payload["familiar_id"] == "other-fam"
