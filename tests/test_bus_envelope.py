"""Tests for :mod:`familiar_connect.bus.envelope`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from familiar_connect.bus.envelope import Event, TurnScope


class TestEvent:
    def test_event_has_core_fields(self) -> None:
        ts = datetime.now(tz=UTC)
        ev = Event(
            event_id="e-1",
            turn_id="t-1",
            session_id="chan-42",
            parent_event_ids=(),
            topic="discord.text",
            timestamp=ts,
            sequence_number=0,
            payload={"text": "hi"},
        )
        assert ev.event_id == "e-1"
        assert ev.turn_id == "t-1"
        assert ev.session_id == "chan-42"
        assert ev.parent_event_ids == ()
        assert ev.topic == "discord.text"
        assert ev.timestamp == ts
        assert ev.sequence_number == 0
        assert ev.payload == {"text": "hi"}

    def test_event_is_immutable(self) -> None:
        ev = Event(
            event_id="e-1",
            turn_id="t-1",
            session_id="chan-42",
            parent_event_ids=(),
            topic="discord.text",
            timestamp=datetime.now(tz=UTC),
            sequence_number=0,
            payload={},
        )
        with pytest.raises((AttributeError, TypeError)):
            setattr(ev, "event_id", "e-2")  # noqa: B010 — exercising frozen dataclass

    def test_parent_event_ids_is_tuple_for_hashability(self) -> None:
        ev = Event(
            event_id="e-2",
            turn_id="t-1",
            session_id="chan-42",
            parent_event_ids=("e-1",),
            topic="derived",
            timestamp=datetime.now(tz=UTC),
            sequence_number=1,
            payload={},
        )
        assert isinstance(ev.parent_event_ids, tuple)


class TestTurnScope:
    def test_scope_carries_identity(self) -> None:
        scope = TurnScope(turn_id="t-1", session_id="chan-42", started_at=1.5)
        assert scope.turn_id == "t-1"
        assert scope.session_id == "chan-42"
        # started_at is a timestamp we pass through unchanged
        assert scope.started_at > 0
        assert scope.is_cancelled() is False

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self) -> None:
        scope = TurnScope(turn_id="t-1", session_id="chan-42", started_at=0.0)
        assert scope.is_cancelled() is False
        scope.cancel()
        assert scope.is_cancelled() is True
        # cancel is idempotent
        scope.cancel()
        assert scope.is_cancelled() is True

    @pytest.mark.asyncio
    async def test_wait_cancelled_resolves_after_cancel(self) -> None:
        scope = TurnScope(turn_id="t-1", session_id="chan-42", started_at=0.0)

        async def canceller() -> None:
            await asyncio.sleep(0.01)
            scope.cancel()

        await asyncio.gather(
            asyncio.wait_for(scope.wait_cancelled(), timeout=0.5),
            canceller(),
        )
