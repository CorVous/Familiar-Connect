"""Tests for the debug logger processor.

This processor subscribes to every live topic and logs one line per
event. It's the Phase-1 "something useful is happening on the bus"
signal — without it, the bus is silent from a user perspective.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

import pytest

from familiar_connect.bus import InProcessEventBus
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.processors.debug_logger import DebugLoggerProcessor

_ANSI_RE = re.compile(r"\x1b\[\d+m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _event(topic: str, seq: int, payload: object = None) -> Event:
    return Event(
        event_id=f"e-{seq}",
        turn_id=f"t-{seq}",
        session_id="s",
        parent_event_ids=(),
        topic=topic,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload=payload,
    )


class TestDebugLogger:
    @pytest.mark.asyncio
    async def test_logs_every_event_on_subscribed_topics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()

        proc = DebugLoggerProcessor(topics=(TOPIC_DISCORD_TEXT,))

        async def consume_subscribe() -> None:
            async for ev in bus.subscribe(proc.topics):
                await proc.handle(ev, bus)

        task = asyncio.create_task(consume_subscribe())
        await asyncio.sleep(0)

        with caplog.at_level(logging.INFO, logger="familiar_connect.processors"):
            await bus.publish(_event(TOPIC_DISCORD_TEXT, 1, {"text": "hi"}))
            await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await bus.shutdown()

        messages = [_strip(r.getMessage()) for r in caplog.records]
        assert any("discord.text" in m for m in messages)
        assert any("e-1" in m for m in messages)

    def test_topics_attribute_is_tuple(self) -> None:
        proc = DebugLoggerProcessor(topics=(TOPIC_DISCORD_TEXT,))
        assert isinstance(proc.topics, tuple)
        assert proc.name == "debug-logger"
