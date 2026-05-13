"""Alarm scheduler.

Owns one ``asyncio.Task`` per pending alarm. Tasks sleep until the
target time, then mark the row fired and publish
:data:`TOPIC_ALARM_FIRED` on the bus. On startup, reloads any rows
left pending from a previous process; past-due rows fire immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_ALARM_FIRED

if TYPE_CHECKING:
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore

_logger = logging.getLogger(__name__)


class AlarmScheduler:
    """Per-familiar wake scheduler — DB-backed, asyncio-driven."""

    def __init__(
        self,
        *,
        history: AsyncHistoryStore,
        bus: EventBus,
        familiar_id: str,
    ) -> None:
        self._history = history
        self._bus = bus
        self._familiar_id = familiar_id
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False

    async def start(self) -> None:
        """Load any pending alarms and schedule each."""
        if self._started:
            return
        self._started = True
        pending = await self._history.list_pending_alarms(familiar_id=self._familiar_id)
        for row in pending:
            scheduled_at = datetime.fromisoformat(row["scheduled_at"])
            self._spawn_task(row, scheduled_at)
        _logger.info(
            f"{ls.tag('Alarm', ls.LC)} "
            f"{ls.kv('loaded_pending', str(len(pending)), vc=ls.LC)} "
            f"{ls.kv('familiar_id', self._familiar_id, vc=ls.LW)}"
        )

    async def shutdown(self) -> None:
        """Cancel all in-flight sleep tasks. Does not modify the DB."""
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()

    async def add(
        self,
        *,
        channel_id: int,
        channel_kind: str,
        scheduled_at: datetime,
        reason: str,
        originating_turn_id: str | None = None,
    ) -> str:
        """Insert + schedule a new alarm. Returns the alarm id."""
        alarm_id = await self._history.insert_alarm(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            channel_kind=channel_kind,
            scheduled_at=scheduled_at.isoformat(),
            reason=reason,
            originating_turn_id=originating_turn_id,
        )
        row = {
            "id": alarm_id,
            "channel_id": channel_id,
            "channel_kind": channel_kind,
            "scheduled_at": scheduled_at.isoformat(),
            "reason": reason,
            "originating_turn_id": originating_turn_id,
        }
        self._spawn_task(row, scheduled_at)
        return alarm_id

    async def cancel(self, *, alarm_id: str) -> bool:
        """Stop the in-flight sleep and mark the row cancelled."""
        task = self._tasks.pop(alarm_id, None)
        if task is not None:
            task.cancel()
        return await self._history.cancel_alarm(
            alarm_id=alarm_id,
            cancelled_at=datetime.now(tz=UTC).isoformat(),
        )

    def _spawn_task(self, row: dict, scheduled_at: datetime) -> None:
        task = asyncio.create_task(
            self._sleep_then_fire(row, scheduled_at),
            name=f"alarm-{row['id']}",
        )
        self._tasks[row["id"]] = task

    async def _sleep_then_fire(self, row: dict, scheduled_at: datetime) -> None:
        try:
            delay = (scheduled_at - datetime.now(tz=UTC)).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            fired_at = datetime.now(tz=UTC)
            updated = await self._history.mark_alarm_fired(
                alarm_id=row["id"],
                fired_at=fired_at.isoformat(),
            )
            if not updated:
                # already fired or cancelled by another path — skip publish.
                return
            payload = {
                "alarm_id": row["id"],
                "channel_id": row["channel_id"],
                "channel_kind": row["channel_kind"],
                "reason": row["reason"],
                "scheduled_at": row["scheduled_at"],
                "fired_at": fired_at.isoformat(),
                "originating_turn_id": row.get("originating_turn_id"),
            }
            session_id = f"alarm:{row['channel_id']}"
            await self._bus.publish(
                Event(
                    event_id=uuid.uuid4().hex,
                    turn_id=f"alarm-{row['id']}",
                    session_id=session_id,
                    parent_event_ids=(),
                    topic=TOPIC_ALARM_FIRED,
                    timestamp=fired_at,
                    sequence_number=0,
                    payload=payload,
                )
            )
            _logger.info(
                f"{ls.tag('Alarm', ls.LM)} "
                f"{ls.kv('fired', row['id'], vc=ls.LM)} "
                f"{ls.kv('reason', ls.trunc(row['reason'], limit=80), vc=ls.LW)}"
            )
        finally:
            self._tasks.pop(row["id"], None)
