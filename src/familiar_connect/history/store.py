"""SQLite-backed persistent conversation history.

One database per familiar. Two core tables:

- ``turns`` — every turn scoped by ``(familiar_id, channel_id)``;
  monotonic PK, role, optional :class:`Author`, content, optional
  guild_id, UTC ts
- ``summaries`` — at most one rolling summary per (familiar, channel)
  with a ``last_summarised_id`` watermark for cache freshness.
  See ``docs/architecture/context-pipeline.md`` for rationale.

``familiar_id`` is explicit (not implicit) so tests can exercise
multiple familiars against one store. Synchronous API — SQLite is
fast enough at per-host volumes; wrap with ``asyncio.to_thread`` if
needed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.identity import Author

if TYPE_CHECKING:
    from familiar_connect.config import ChannelMode

PathLike = str | Path


@dataclass(frozen=True)
class HistoryTurn:
    """Single persisted conversational turn."""

    id: int
    timestamp: datetime
    role: str
    author: Author | None
    content: str
    channel_id: int = 0


@dataclass(frozen=True)
class SummaryEntry:
    """Cached rolling summary for one familiar."""

    last_summarised_id: int
    summary_text: str
    created_at: datetime


@dataclass(frozen=True)
class OtherChannelInfo:
    """Recent-activity info for another channel."""

    channel_id: int
    mode: str | None
    latest_id: int
    latest_timestamp: datetime


@dataclass(frozen=True)
class CrossContextEntry:
    """Cached cross-context summary for one source channel."""

    source_last_id: int
    summary_text: str
    created_at: datetime


@dataclass(frozen=True)
class WatermarkEntry:
    """Tracks the last turn id written to long-term memory by the memory writer."""

    last_written_id: int
    created_at: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id          TEXT    NOT NULL,
    channel_id           INTEGER NOT NULL,
    guild_id             INTEGER,
    role                 TEXT    NOT NULL,
    author_platform      TEXT,
    author_user_id       TEXT,
    author_username      TEXT,
    author_display_name  TEXT,
    content              TEXT    NOT NULL,
    timestamp            TEXT    NOT NULL,
    mode                 TEXT
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

CREATE TABLE IF NOT EXISTS memory_writer_watermark (
    familiar_id       TEXT    PRIMARY KEY,
    last_written_id   INTEGER NOT NULL,
    created_at        TEXT    NOT NULL
);
"""

_TURN_COLS = (
    "id, timestamp, role, author_platform, author_user_id, "
    "author_username, author_display_name, content, channel_id"
)


class HistoryStore:
    """Persistent SQLite store for turns + rolling summaries.

    Pass ``":memory:"`` for an ephemeral in-process database (tests).
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
        """Idempotent migrations for the ``turns`` and ``summaries`` tables."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='turns'"
        ).fetchone()
        if row is None:
            return

        columns = {
            col["name"]
            for col in self._conn.execute("PRAGMA table_info(turns)").fetchall()
        }

        # legacy: add mode column if missing
        if "mode" not in columns:
            self._conn.execute("ALTER TABLE turns ADD COLUMN mode TEXT")
            self._conn.commit()

        # identity migration: drop bare ``speaker`` in favour of four
        # author_* columns. Legacy speaker strings are preserved as
        # ``author_display_name`` with a synthesised ``legacy-discord``
        # platform key so historical turns keep their attribution.
        # TODO(cleanup): remove this branch + the related legacy test
        # once every live install has been upgraded past the speaker
        # schema. See docs/architecture/memory.md § Legacy history
        # migration.
        if "speaker" in columns:
            for col in (
                "author_platform",
                "author_user_id",
                "author_username",
                "author_display_name",
            ):
                if col not in columns:
                    self._conn.execute(f"ALTER TABLE turns ADD COLUMN {col} TEXT")
            self._conn.execute("""
                UPDATE turns
                   SET author_display_name = speaker,
                       author_platform = 'legacy-discord',
                       author_user_id = speaker
                 WHERE speaker IS NOT NULL
                   AND author_platform IS NULL
            """)
            self._conn.execute("ALTER TABLE turns DROP COLUMN speaker")
            self._conn.commit()
            columns = {
                col["name"]
                for col in self._conn.execute("PRAGMA table_info(turns)").fetchall()
            }

        for col in (
            "author_platform",
            "author_user_id",
            "author_username",
            "author_display_name",
        ):
            if col not in columns:
                self._conn.execute(f"ALTER TABLE turns ADD COLUMN {col} TEXT")
        self._conn.commit()

        # summaries: old PK was (familiar_id) only; new adds channel_id.
        # summaries are a cache — drop + recreate is safe
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
        """Close underlying SQLite connection."""
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
        author: Author | None = None,
        guild_id: int | None = None,
        mode: ChannelMode | None = None,
    ) -> HistoryTurn:
        """Append a single turn and return its persisted form."""
        timestamp = datetime.now(tz=UTC)
        mode_value = mode.value if mode is not None else None
        platform = author.platform if author is not None else None
        user_id = author.user_id if author is not None else None
        username = author.username if author is not None else None
        display_name = author.display_name if author is not None else None
        cur = self._conn.execute(
            """
            INSERT INTO turns
                (familiar_id, channel_id, guild_id,
                 role, author_platform, author_user_id,
                 author_username, author_display_name,
                 content, timestamp, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                channel_id,
                guild_id,
                role,
                platform,
                user_id,
                username,
                display_name,
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
            author=author,
            content=content,
            channel_id=channel_id,
        )

    def recent(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        limit: int,
        mode: ChannelMode | None = None,
    ) -> list[HistoryTurn]:
        """Return most recent turns in channel, oldest-first.

        Per-channel partitioning prevents bleed between conversations.
        When *mode* is set, only matching turns returned (prevents
        cross-mode style contamination).
        """
        if limit <= 0:
            return []
        if mode is not None:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ? AND channel_id = ? AND mode = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,  # noqa: S608
                (familiar_id, channel_id, mode.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ? AND channel_id = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,  # noqa: S608
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
        """Return turns with ``id <= max_id``, oldest first.

        *channel_id* scopes to one channel; omit for global.
        """
        if channel_id is not None:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ?
                   AND channel_id = ?
                   AND id <= ?
                 ORDER BY id ASC
                 LIMIT ?
                """,  # noqa: S608
                (familiar_id, channel_id, max_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ?
                   AND id <= ?
                 ORDER BY id ASC
                 LIMIT ?
                """,  # noqa: S608
                (familiar_id, max_id, limit),
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def latest_id(
        self,
        *,
        familiar_id: str,
        channel_id: int | None = None,
    ) -> int | None:
        """Return highest turn id (watermark for cache freshness).

        *channel_id* scopes to one channel; omit for global max.
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
        """Return number of stored turns. *channel_id* scopes to one channel."""
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
        """Return other channels with activity, most-recently-active first.

        Each row carries latest mode, turn id, and timestamp.
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

    # ------------------------------------------------------------------
    # memory-writer watermark
    # ------------------------------------------------------------------

    def get_writer_watermark(
        self,
        *,
        familiar_id: str,
    ) -> WatermarkEntry | None:
        """Return the memory-writer watermark for *familiar_id*, or ``None``."""
        row = self._conn.execute(
            """
            SELECT last_written_id, created_at
              FROM memory_writer_watermark
             WHERE familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return None
        return WatermarkEntry(
            last_written_id=int(row["last_written_id"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put_writer_watermark(
        self,
        *,
        familiar_id: str,
        last_written_id: int,
    ) -> None:
        """Insert or replace the memory-writer watermark for *familiar_id*."""
        timestamp = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO memory_writer_watermark
                (familiar_id, last_written_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT (familiar_id)
            DO UPDATE SET
                last_written_id = excluded.last_written_id,
                created_at      = excluded.created_at
            """,
            (familiar_id, last_written_id, timestamp),
        )
        self._conn.commit()

    def turns_since_watermark(
        self,
        *,
        familiar_id: str,
        limit: int = 10_000,
    ) -> list[HistoryTurn]:
        """Return turns after the memory-writer watermark, oldest first.

        If no watermark has been set, returns all turns for the familiar.
        """
        wm = self.get_writer_watermark(familiar_id=familiar_id)
        min_id = wm.last_written_id if wm is not None else 0
        rows = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ?
               AND id > ?
             ORDER BY id ASC
             LIMIT ?
            """,  # noqa: S608
            (familiar_id, min_id, limit),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]


def _row_to_turn(row: sqlite3.Row) -> HistoryTurn:
    """Rebuild a HistoryTurn from a SELECT row. Author is reconstructed.

    channel_id missing from older SELECTs that don't need it; fall
    back to 0 so those callers keep working. Writer-facing SELECTs
    include it explicitly.
    """
    try:
        channel_id = int(row["channel_id"])
    except (IndexError, KeyError):
        channel_id = 0

    platform = row["author_platform"]
    user_id = row["author_user_id"]
    if platform is not None and user_id is not None:
        author: Author | None = Author(
            platform=str(platform),
            user_id=str(user_id),
            username=row["author_username"],
            display_name=row["author_display_name"],
        )
    else:
        author = None

    return HistoryTurn(
        id=int(row["id"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        role=str(row["role"]),
        author=author,
        content=str(row["content"]),
        channel_id=channel_id,
    )
