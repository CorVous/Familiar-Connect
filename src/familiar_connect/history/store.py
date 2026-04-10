"""SQLite-backed persistent conversation history.

The HistoryStore owns one SQLite database with two tables:

- ``turns`` — every conversational turn the bot has seen, scoped by
  ``(familiar_id, channel_id)``. Each turn carries a monotonically
  increasing primary key, a role (user/assistant/system), an
  optional speaker name, the textual content, an optional
  ``guild_id`` (observability only), and a UTC timestamp.
- ``summaries`` — at most one rolling summary per ``familiar_id``,
  with a ``last_summarised_id`` watermark so the
  :class:`HistoryProvider` can tell whether the cache still covers
  every turn the familiar has heard. The summary is *global per
  familiar* — see ``plan.md`` § Context Management for the rationale.
  The recent rolling window is partitioned per channel; the rolling
  summary is not.

A Familiar-Connect install runs exactly one familiar at a time, so
``familiar_id`` effectively identifies the single active character on
this install. It's still carried through the API surface (rather than
implicit) so that tests can exercise multiple familiars against a
single store and future work that revisits the single-character
constraint has a clean extension point.

The store deliberately exposes a small, synchronous API. SQLite is
fast enough for the volumes a per-host bot will see, and keeping the
API sync means callers (the bot's text/voice loops, the
:class:`HistoryProvider`) don't need to manage a connection pool.
Sync calls from async code can be wrapped with :func:`asyncio.to_thread`
if they ever start blocking the event loop, but at the scale of "tens
of turns per second worst case" it's not worth pre-empting.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.config import ChannelMode

PathLike = str | Path


@dataclass(frozen=True)
class HistoryTurn:
    """A single conversational turn as stored on disk."""

    id: int
    timestamp: datetime
    role: str
    speaker: str | None
    content: str


@dataclass(frozen=True)
class SummaryEntry:
    """A cached rolling summary for one ``familiar_id``."""

    last_summarised_id: int
    summary_text: str
    created_at: datetime


@dataclass(frozen=True)
class OtherChannelInfo:
    """Summary info about another channel's recent activity."""

    channel_id: int
    mode: str | None
    latest_id: int
    latest_timestamp: datetime


@dataclass(frozen=True)
class CrossContextEntry:
    """A cached cross-context summary for one source channel."""

    source_last_id: int
    summary_text: str
    created_at: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id    TEXT    NOT NULL,
    channel_id     INTEGER NOT NULL,
    guild_id       INTEGER,
    role           TEXT    NOT NULL,
    speaker        TEXT,
    content        TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    mode           TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_channel
    ON turns (familiar_id, channel_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_global
    ON turns (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_channel_mode
    ON turns (familiar_id, channel_id, mode, id);

CREATE TABLE IF NOT EXISTS summaries (
    familiar_id         TEXT    NOT NULL,
    channel_id          INTEGER NOT NULL DEFAULT 0,
    last_summarised_id  INTEGER NOT NULL,
    summary_text        TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, channel_id)
);

CREATE TABLE IF NOT EXISTS cross_context_summaries (
    familiar_id        TEXT    NOT NULL,
    viewer_mode        TEXT    NOT NULL,
    source_channel_id  INTEGER NOT NULL,
    source_last_id     INTEGER NOT NULL,
    summary_text       TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, viewer_mode, source_channel_id)
);
"""


class HistoryStore:
    """Persistent SQLite store for turns + rolling summaries.

    Pass a filesystem path (str or :class:`Path`) for a real
    persistent database, or the literal string ``":memory:"`` for an
    ephemeral in-process database (useful in tests).
    """

    def __init__(self, db_path: PathLike) -> None:
        if db_path == ":memory:":
            self._path: Path | None = None
            self._conn = sqlite3.connect(":memory:")
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._migrate_if_needed()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate_if_needed(self) -> None:
        """Run idempotent migrations for schema changes.

        Currently handles:
        - Adding the ``mode`` column to ``turns`` (+ backfill index).
        """
        # Check whether the turns table exists at all — if not, the
        # CREATE TABLE IF NOT EXISTS in _SCHEMA will handle it.
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='turns'"
        ).fetchone()
        if row is None:
            return

        columns = {
            col["name"]
            for col in self._conn.execute("PRAGMA table_info(turns)").fetchall()
        }
        if "mode" not in columns:
            self._conn.execute("ALTER TABLE turns ADD COLUMN mode TEXT")
            self._conn.commit()

        # Migrate summaries table: old schema had PK (familiar_id) only.
        # New schema adds channel_id to the PK. Summaries are a cache —
        # dropping and recreating is safe (they rebuild on next request).
        summary_row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='summaries'"
        ).fetchone()
        if summary_row is not None:
            summary_cols = {
                col["name"]
                for col in self._conn.execute("PRAGMA table_info(summaries)").fetchall()
            }
            if "channel_id" not in summary_cols:
                self._conn.execute("DROP TABLE summaries")
                self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------

    def append_turn(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        role: str,
        content: str,
        speaker: str | None = None,
        guild_id: int | None = None,
        mode: ChannelMode | None = None,
    ) -> HistoryTurn:
        """Append a single turn and return its persisted form."""
        timestamp = datetime.now(tz=UTC)
        mode_value = mode.value if mode is not None else None
        cur = self._conn.execute(
            """
            INSERT INTO turns
                (familiar_id, channel_id, guild_id,
                 role, speaker, content, timestamp, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                channel_id,
                guild_id,
                role,
                speaker,
                content,
                timestamp.isoformat(),
                mode_value,
            ),
        )
        self._conn.commit()
        return HistoryTurn(
            id=int(cur.lastrowid or 0),
            timestamp=timestamp,
            role=role,
            speaker=speaker,
            content=content,
        )

    def recent(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        limit: int,
        mode: ChannelMode | None = None,
    ) -> list[HistoryTurn]:
        """Return up to *limit* most recent turns *in this channel*.

        The recent window is partitioned per channel so two
        simultaneous conversations don't bleed into each other.
        Results are sorted oldest-first so callers can render them
        without re-reversing.

        When *mode* is provided, only turns recorded under that mode
        are returned. This prevents cross-mode style contamination in
        the verbatim example window.
        """
        if limit <= 0:
            return []
        if mode is not None:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, role, speaker, content
                  FROM turns
                 WHERE familiar_id = ? AND channel_id = ? AND mode = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (familiar_id, channel_id, mode.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, role, speaker, content
                  FROM turns
                 WHERE familiar_id = ? AND channel_id = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (familiar_id, channel_id, limit),
            ).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    def older_than(
        self,
        *,
        familiar_id: str,
        max_id: int,
        channel_id: int | None = None,
        limit: int = 10_000,
    ) -> list[HistoryTurn]:
        """Return turns whose ``id`` is *<= max_id*, oldest first.

        When *channel_id* is provided, scopes to that single channel.
        When omitted, returns turns globally across all channels (the
        legacy behaviour).
        """
        if channel_id is not None:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, role, speaker, content
                  FROM turns
                 WHERE familiar_id = ?
                   AND channel_id = ?
                   AND id <= ?
                 ORDER BY id ASC
                 LIMIT ?
                """,
                (familiar_id, channel_id, max_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, timestamp, role, speaker, content
                  FROM turns
                 WHERE familiar_id = ?
                   AND id <= ?
                 ORDER BY id ASC
                 LIMIT ?
                """,
                (familiar_id, max_id, limit),
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def latest_id(
        self,
        *,
        familiar_id: str,
        channel_id: int | None = None,
    ) -> int | None:
        """Return the highest turn id for the familiar.

        When *channel_id* is provided, scopes to that single channel.
        When omitted, returns the global max across all channels.

        Used by :class:`HistoryProvider` as the watermark for cache
        freshness.
        """
        if channel_id is not None:
            row = self._conn.execute(
                """
                SELECT MAX(id) AS max_id
                  FROM turns
                 WHERE familiar_id = ? AND channel_id = ?
                """,
                (familiar_id, channel_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT MAX(id) AS max_id
                  FROM turns
                 WHERE familiar_id = ?
                """,
                (familiar_id,),
            ).fetchone()
        if row is None or row["max_id"] is None:
            return None
        return int(row["max_id"])

    def count(
        self,
        *,
        familiar_id: str,
        channel_id: int | None = None,
    ) -> int:
        """Return the number of stored turns.

        With ``channel_id`` set, scoped to that channel; without it,
        scoped to the whole familiar.
        """
        if channel_id is None:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n
                  FROM turns
                 WHERE familiar_id = ?
                """,
                (familiar_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n
                  FROM turns
                 WHERE familiar_id = ?
                   AND channel_id = ?
                """,
                (familiar_id, channel_id),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # summaries
    # ------------------------------------------------------------------

    def get_summary(
        self,
        *,
        familiar_id: str,
        channel_id: int = 0,
    ) -> SummaryEntry | None:
        """Return the cached summary for the familiar + channel, or ``None``."""
        row = self._conn.execute(
            """
            SELECT last_summarised_id, summary_text, created_at
              FROM summaries
             WHERE familiar_id = ? AND channel_id = ?
            """,
            (familiar_id, channel_id),
        ).fetchone()
        if row is None:
            return None
        return SummaryEntry(
            last_summarised_id=int(row["last_summarised_id"]),
            summary_text=str(row["summary_text"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put_summary(
        self,
        *,
        familiar_id: str,
        last_summarised_id: int,
        summary_text: str,
        channel_id: int = 0,
    ) -> None:
        """Insert or replace the summary for the familiar + channel."""
        timestamp = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO summaries
                (familiar_id, channel_id,
                 last_summarised_id, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (familiar_id, channel_id)
            DO UPDATE SET
                last_summarised_id = excluded.last_summarised_id,
                summary_text       = excluded.summary_text,
                created_at         = excluded.created_at
            """,
            (
                familiar_id,
                channel_id,
                last_summarised_id,
                summary_text,
                timestamp,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # cross-context summaries
    # ------------------------------------------------------------------

    def distinct_other_channels(
        self,
        *,
        familiar_id: str,
        exclude_channel_id: int,
    ) -> list[OtherChannelInfo]:
        """Return info about other channels with activity for this familiar.

        Each row carries the channel's most recent mode, latest turn id,
        and latest timestamp. Channels with *only* the excluded channel_id
        are omitted. Results are ordered by latest turn id descending
        (most-recently-active first).
        """
        rows = self._conn.execute(
            """
            SELECT channel_id, mode, MAX(id) AS latest_id,
                   MAX(timestamp) AS latest_ts
              FROM turns
             WHERE familiar_id = ? AND channel_id != ?
             GROUP BY channel_id
             ORDER BY latest_id DESC
            """,
            (familiar_id, exclude_channel_id),
        ).fetchall()
        return [
            OtherChannelInfo(
                channel_id=int(row["channel_id"]),
                mode=row["mode"],
                latest_id=int(row["latest_id"]),
                latest_timestamp=datetime.fromisoformat(row["latest_ts"]),
            )
            for row in rows
        ]

    def get_cross_context(
        self,
        *,
        familiar_id: str,
        viewer_mode: str,
        source_channel_id: int,
    ) -> CrossContextEntry | None:
        """Return the cached cross-context summary, or ``None``."""
        row = self._conn.execute(
            """
            SELECT source_last_id, summary_text, created_at
              FROM cross_context_summaries
             WHERE familiar_id = ?
               AND viewer_mode = ?
               AND source_channel_id = ?
            """,
            (familiar_id, viewer_mode, source_channel_id),
        ).fetchone()
        if row is None:
            return None
        return CrossContextEntry(
            source_last_id=int(row["source_last_id"]),
            summary_text=str(row["summary_text"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put_cross_context(
        self,
        *,
        familiar_id: str,
        viewer_mode: str,
        source_channel_id: int,
        source_last_id: int,
        summary_text: str,
    ) -> None:
        """Insert or replace a cross-context summary."""
        timestamp = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO cross_context_summaries
                (familiar_id, viewer_mode, source_channel_id,
                 source_last_id, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (familiar_id, viewer_mode, source_channel_id)
            DO UPDATE SET
                source_last_id = excluded.source_last_id,
                summary_text   = excluded.summary_text,
                created_at     = excluded.created_at
            """,
            (
                familiar_id,
                viewer_mode,
                source_channel_id,
                source_last_id,
                summary_text,
                timestamp,
            ),
        )
        self._conn.commit()


def _row_to_turn(row: sqlite3.Row) -> HistoryTurn:
    return HistoryTurn(
        id=int(row["id"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        role=str(row["role"]),
        speaker=row["speaker"] if row["speaker"] is not None else None,
        content=str(row["content"]),
    )
