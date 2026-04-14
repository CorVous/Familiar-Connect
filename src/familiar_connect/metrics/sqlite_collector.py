"""SQLite-backed :class:`MetricsCollector` with batched writes.

One DB per familiar. Mirrors ``history/store.py`` pattern. Single
``turn_traces`` table with JSON columns for variable-shape stage +
tag data — new dimensions absorbed without migration.

Writes are buffered in memory under a ``threading.Lock`` and flushed
at a configurable threshold or on ``close()``. The hot path cost is
one ``list.append`` + lock acquire (uncontended, nanoseconds).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from familiar_connect.metrics.types import StageSpan, TurnTrace

PathLike = str | Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_traces (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT    NOT NULL UNIQUE,
    familiar_id      TEXT    NOT NULL,
    channel_id       INTEGER NOT NULL,
    guild_id         INTEGER,
    modality         TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    total_duration_s REAL    NOT NULL,
    stages_json      TEXT    NOT NULL,
    tags_json        TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_familiar_ts
    ON turn_traces (familiar_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_traces_channel
    ON turn_traces (familiar_id, channel_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_traces_modality
    ON turn_traces (familiar_id, modality, timestamp);
"""


class SQLiteCollector:
    """Persist :class:`TurnTrace` records to SQLite with batched writes.

    Pass ``":memory:"`` for an ephemeral in-process DB (tests).
    """

    def __init__(
        self,
        db_path: PathLike,
        *,
        flush_interval: int = 50,
    ) -> None:
        if db_path == ":memory:":
            self._path: Path | None = None
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._flush_interval = flush_interval
        self._buffer: list[TurnTrace] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # MetricsCollector protocol
    # ------------------------------------------------------------------

    def record(self, trace: TurnTrace) -> None:
        """Append trace to buffer; flush when threshold reached."""
        with self._lock:
            self._buffer.append(trace)
            if len(self._buffer) >= self._flush_interval:
                self._flush_locked()

    def close(self) -> None:
        """Flush remaining buffer and close the DB connection."""
        with self._lock:
            self._flush_locked()
            self._conn.close()

    # ------------------------------------------------------------------
    # Query API (for the report layer)
    # ------------------------------------------------------------------

    def recent_traces(
        self,
        *,
        familiar_id: str,
        limit: int = 100,
    ) -> list[TurnTrace]:
        """Return the ``limit`` most recent traces for *familiar_id*, newest first."""
        rows = self._conn.execute(
            """
            SELECT trace_id, familiar_id, channel_id, guild_id, modality,
                   timestamp, total_duration_s, stages_json, tags_json
              FROM turn_traces
             WHERE familiar_id = ?
             ORDER BY timestamp DESC, id DESC
             LIMIT ?
            """,
            (familiar_id, limit),
        ).fetchall()
        return [_row_to_trace(r) for r in rows]

    def traces_in_range(
        self,
        *,
        familiar_id: str,
        start: str,
        end: str,
    ) -> list[TurnTrace]:
        """Return traces whose ``timestamp`` falls in ``[start, end]`` (ISO 8601)."""
        rows = self._conn.execute(
            """
            SELECT trace_id, familiar_id, channel_id, guild_id, modality,
                   timestamp, total_duration_s, stages_json, tags_json
              FROM turn_traces
             WHERE familiar_id = ?
               AND timestamp >= ?
               AND timestamp <= ?
             ORDER BY timestamp ASC, id ASC
            """,
            (familiar_id, start, end),
        ).fetchall()
        return [_row_to_trace(r) for r in rows]

    def stage_durations(
        self,
        *,
        familiar_id: str,
        stage_name: str,
        limit: int = 1000,
    ) -> list[float]:
        """Return durations for all spans with ``stage_name``, most recent first.

        Uses ``recent_traces`` + in-Python filter. Acceptable at current
        volume; a targeted ``json_each`` query is a future optimization.
        """
        traces = self.recent_traces(familiar_id=familiar_id, limit=limit)
        return [
            span.duration_s
            for trace in traces
            for span in trace.stages
            if span.name == stage_name
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """Write buffered traces to the DB. Caller must hold ``_lock``."""
        if not self._buffer:
            return
        now = datetime.now(tz=UTC).isoformat()
        rows = [
            (
                t.trace_id,
                t.familiar_id,
                t.channel_id,
                t.guild_id,
                t.modality,
                t.timestamp,
                t.total_duration_s,
                json.dumps([_stage_to_dict(s) for s in t.stages]),
                json.dumps(t.tags),
                now,
            )
            for t in self._buffer
        ]
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO turn_traces
                (trace_id, familiar_id, channel_id, guild_id, modality,
                 timestamp, total_duration_s, stages_json, tags_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        self._buffer.clear()


def _stage_to_dict(span: StageSpan) -> dict[str, object]:
    return {
        "name": span.name,
        "start_s": span.start_s,
        "duration_s": span.duration_s,
        "metadata": span.metadata,
    }


def _dict_to_stage(data: dict[str, object]) -> StageSpan:
    raw_meta = data.get("metadata")
    metadata: dict[str, object] = (
        cast("dict[str, object]", raw_meta) if isinstance(raw_meta, dict) else {}
    )
    start_s = data["start_s"]
    duration_s = data["duration_s"]
    return StageSpan(
        name=str(data["name"]),
        start_s=float(start_s) if isinstance(start_s, (int, float)) else 0.0,
        duration_s=(float(duration_s) if isinstance(duration_s, (int, float)) else 0.0),
        metadata=metadata,
    )


def _row_to_trace(row: sqlite3.Row) -> TurnTrace:
    stages_data = json.loads(row["stages_json"])
    tags_data = json.loads(row["tags_json"])
    return TurnTrace(
        trace_id=str(row["trace_id"]),
        familiar_id=str(row["familiar_id"]),
        channel_id=int(row["channel_id"]),
        guild_id=row["guild_id"] if row["guild_id"] is None else int(row["guild_id"]),
        modality=str(row["modality"]),
        timestamp=str(row["timestamp"]),
        total_duration_s=float(row["total_duration_s"]),
        stages=tuple(_dict_to_stage(s) for s in stages_data),
        tags=dict(tags_data),
    )
