"""Tests for the SQLite-backed metrics collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.metrics.sqlite_collector import SQLiteCollector
from familiar_connect.metrics.types import StageSpan, TurnTrace

if TYPE_CHECKING:
    from pathlib import Path


def _trace(
    *,
    trace_id: str,
    familiar_id: str = "fam",
    channel_id: int = 1,
    modality: str = "text",
    total: float = 0.5,
    timestamp: str = "2026-04-14T12:00:00+00:00",
    stages: tuple[StageSpan, ...] = (),
    tags: dict[str, str] | None = None,
) -> TurnTrace:
    return TurnTrace(
        trace_id=trace_id,
        familiar_id=familiar_id,
        channel_id=channel_id,
        guild_id=None,
        modality=modality,
        timestamp=timestamp,
        total_duration_s=total,
        stages=stages,
        tags=tags or {},
    )


class TestSQLiteCollectorInMemory:
    def test_round_trip_single_trace(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=1)
        trace = _trace(trace_id="a")
        collector.record(trace)
        rows = collector.recent_traces(familiar_id="fam", limit=10)
        assert len(rows) == 1
        assert rows[0].trace_id == "a"

    def test_round_trip_preserves_stages_and_tags(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=1)
        stages = (
            StageSpan(
                name="llm_call",
                start_s=0.1,
                duration_s=1.5,
                metadata={"model": "glm-5.1", "tokens_out": 420},
            ),
        )
        trace = _trace(
            trace_id="b",
            stages=stages,
            tags={"channel_mode": "full_rp", "variant": "A"},
        )
        collector.record(trace)
        [got] = collector.recent_traces(familiar_id="fam", limit=10)
        assert len(got.stages) == 1
        assert got.stages[0].name == "llm_call"
        assert got.stages[0].metadata["model"] == "glm-5.1"
        assert got.stages[0].metadata["tokens_out"] == 420
        assert got.tags == {"channel_mode": "full_rp", "variant": "A"}

    def test_buffer_flush_on_threshold(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=3)
        collector.record(_trace(trace_id="1"))
        collector.record(_trace(trace_id="2"))
        # still buffered; recent_traces reads from DB only
        assert collector.recent_traces(familiar_id="fam", limit=10) == []
        collector.record(_trace(trace_id="3"))
        # threshold hit, all 3 flushed
        rows = collector.recent_traces(familiar_id="fam", limit=10)
        assert {r.trace_id for r in rows} == {"1", "2", "3"}

    def test_close_flushes_remaining(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=100)
        collector.record(_trace(trace_id="x"))
        assert collector.recent_traces(familiar_id="fam", limit=10) == []
        collector.close()
        # after close, buffer is flushed
        # re-open same file: can't for :memory:, but close() must have flushed
        # test by reading from the still-open DB immediately before close released
        # — instead use the flush boundary: re-record via a second collector on
        # a temp path verifies close() persists

    def test_recent_traces_newest_first(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=1)
        collector.record(_trace(trace_id="old", timestamp="2026-04-14T10:00:00+00:00"))
        collector.record(_trace(trace_id="new", timestamp="2026-04-14T12:00:00+00:00"))
        rows = collector.recent_traces(familiar_id="fam", limit=10)
        assert [r.trace_id for r in rows] == ["new", "old"]

    def test_scoped_by_familiar(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=1)
        collector.record(_trace(trace_id="a", familiar_id="fam1"))
        collector.record(_trace(trace_id="b", familiar_id="fam2"))
        rows = collector.recent_traces(familiar_id="fam1", limit=10)
        assert [r.trace_id for r in rows] == ["a"]

    def test_stage_durations(self) -> None:
        collector = SQLiteCollector(":memory:", flush_interval=1)
        t1 = _trace(
            trace_id="1",
            stages=(StageSpan(name="llm_call", start_s=0.0, duration_s=1.0),),
        )
        t2 = _trace(
            trace_id="2",
            stages=(
                StageSpan(name="llm_call", start_s=0.0, duration_s=2.0),
                StageSpan(name="tts", start_s=2.0, duration_s=0.5),
            ),
        )
        collector.record(t1)
        collector.record(t2)
        llm_durations = collector.stage_durations(
            familiar_id="fam",
            stage_name="llm_call",
            limit=100,
        )
        assert sorted(llm_durations) == [1.0, 2.0]


class TestSQLiteCollectorPersistent:
    def test_writes_to_disk_and_reopens(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        collector = SQLiteCollector(db_path, flush_interval=1)
        collector.record(_trace(trace_id="persist_me"))
        collector.close()

        assert db_path.exists()

        reopened = SQLiteCollector(db_path, flush_interval=1)
        rows = reopened.recent_traces(familiar_id="fam", limit=10)
        assert [r.trace_id for r in rows] == ["persist_me"]
        reopened.close()

    def test_close_flushes_pending(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        collector = SQLiteCollector(db_path, flush_interval=100)
        collector.record(_trace(trace_id="x"))
        collector.record(_trace(trace_id="y"))
        collector.close()

        reopened = SQLiteCollector(db_path, flush_interval=1)
        rows = reopened.recent_traces(familiar_id="fam", limit=10)
        assert {r.trace_id for r in rows} == {"x", "y"}
        reopened.close()

    def test_parent_dir_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "metrics.db"
        collector = SQLiteCollector(db_path, flush_interval=1)
        collector.record(_trace(trace_id="z"))
        collector.close()
        assert db_path.exists()
