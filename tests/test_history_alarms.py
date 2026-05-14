"""Alarm CRUD tests for :class:`HistoryStore`.

Schema additions:

* ``alarms`` table — scheduled wakes, append-only with soft-delete via
  ``cancelled_at``; ``fired_at`` flips when the scheduler dispatches.
* ``turns.tool_calls_json`` + ``turns.tool_call_id`` — assistant turns
  that invoked tools and ``role=tool`` results, persisted alongside
  the rest of history.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.store import HistoryStore

if TYPE_CHECKING:
    from pathlib import Path


_FAMILIAR = "aria"
_CHANNEL = 555


def _store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _future_iso(seconds: int) -> str:
    return (datetime.now(tz=UTC) + timedelta(seconds=seconds)).isoformat()


# ---------------------------------------------------------------------------
# alarms table CRUD
# ---------------------------------------------------------------------------


class TestAlarmsCRUD:
    def test_insert_and_list_pending(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alarm_id = store.insert_alarm(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            channel_kind="text",
            scheduled_at=_future_iso(60),
            reason="wake-test",
        )
        assert alarm_id

        pending = store.list_pending_alarms(familiar_id=_FAMILIAR)
        assert len(pending) == 1
        assert pending[0]["id"] == alarm_id
        assert pending[0]["reason"] == "wake-test"
        assert pending[0]["channel_kind"] == "text"
        assert pending[0]["fired_at"] is None
        assert pending[0]["cancelled_at"] is None

    def test_mark_fired_excludes_from_pending(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alarm_id = store.insert_alarm(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            channel_kind="text",
            scheduled_at=_future_iso(60),
            reason="wake-test",
        )
        store.mark_alarm_fired(alarm_id=alarm_id, fired_at=_now_iso())

        pending = store.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending == []

    def test_cancel_excludes_from_pending(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alarm_id = store.insert_alarm(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            channel_kind="text",
            scheduled_at=_future_iso(60),
            reason="wake-test",
        )
        ok = store.cancel_alarm(alarm_id=alarm_id, cancelled_at=_now_iso())
        assert ok is True

        pending = store.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending == []

    def test_cancel_unknown_returns_false(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ok = store.cancel_alarm(alarm_id="no-such-id", cancelled_at=_now_iso())
        assert ok is False

    def test_list_pending_scoped_per_familiar(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.insert_alarm(
            familiar_id="fam-a",
            channel_id=1,
            channel_kind="text",
            scheduled_at=_future_iso(60),
            reason="a",
        )
        store.insert_alarm(
            familiar_id="fam-b",
            channel_id=2,
            channel_kind="text",
            scheduled_at=_future_iso(60),
            reason="b",
        )
        assert len(store.list_pending_alarms(familiar_id="fam-a")) == 1
        assert len(store.list_pending_alarms(familiar_id="fam-b")) == 1

    def test_channel_kind_constrained(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(Exception):  # noqa: B017, PT011
            store.insert_alarm(
                familiar_id=_FAMILIAR,
                channel_id=_CHANNEL,
                channel_kind="other",
                scheduled_at=_future_iso(60),
                reason="x",
            )

    def test_originating_turn_id_round_trip(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alarm_id = store.insert_alarm(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            channel_kind="voice",
            scheduled_at=_future_iso(60),
            reason="x",
            originating_turn_id="turn-42",
        )
        pending = store.list_pending_alarms(familiar_id=_FAMILIAR)
        assert pending[0]["id"] == alarm_id
        assert pending[0]["originating_turn_id"] == "turn-42"


# ---------------------------------------------------------------------------
# turns.tool_calls_json + tool_call_id columns
# ---------------------------------------------------------------------------


class TestTurnsToolColumns:
    def test_append_turn_round_trips_tool_calls_json(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        tool_calls = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "set_alarm", "arguments": "{}"},
            }
        ]
        turn = store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="assistant",
            content="",
            tool_calls_json=json.dumps(tool_calls),
        )
        row = store._conn.execute(
            "SELECT tool_calls_json, tool_call_id FROM turns WHERE id = ?",
            (turn.id,),
        ).fetchone()
        assert json.loads(row["tool_calls_json"]) == tool_calls
        assert row["tool_call_id"] is None

    def test_append_turn_round_trips_tool_call_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        turn = store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="tool",
            content='{"ok": true}',
            tool_call_id="c-abc",
        )
        row = store._conn.execute(
            "SELECT tool_calls_json, tool_call_id FROM turns WHERE id = ?",
            (turn.id,),
        ).fetchone()
        assert row["tool_call_id"] == "c-abc"
        assert row["tool_calls_json"] is None

    def test_append_turn_without_tool_fields_is_unchanged(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        turn = store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="assistant",
            content="hello",
        )
        row = store._conn.execute(
            "SELECT content, tool_calls_json, tool_call_id FROM turns WHERE id = ?",
            (turn.id,),
        ).fetchone()
        assert row["content"] == "hello"
        assert row["tool_calls_json"] is None
        assert row["tool_call_id"] is None
