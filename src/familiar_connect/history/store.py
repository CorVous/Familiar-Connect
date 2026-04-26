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

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.identity import Author

if TYPE_CHECKING:
    from collections.abc import Iterable

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


@dataclass(frozen=True)
class FactSubject:
    """Soft link from a fact to one canonical identity.

    The extractor's best guess at *who* a fact is about. Provisional —
    mic-sharing, relayed quotes, ambiguous mentions all break a clean
    1:1 mapping. Stored to enable display-name resolution at read time
    without claiming authoritative subject identification.

    :param canonical_key: stable ``platform:user_id`` from
        :class:`~familiar_connect.identity.Author`.
    :param display_at_write: display name as seen by the extractor
        when the fact was authored. Used as a substring anchor at
        read time when the current display name differs.
    """

    canonical_key: str
    display_at_write: str


@dataclass(frozen=True)
class Fact:
    """Atomic fact extracted from one or more turns.

    :param source_turn_ids: ids in ``turns`` the fact was distilled
        from — forever provenance, per plan § Design.5.
    :param superseded_at: when this fact was retired, or ``None`` if
        still current. Supersession keeps the row (no delete) so the
        prior state stays visible for audit.
    :param superseded_by: id of the replacement fact, or ``None`` if
        still current.
    :param subjects: best-effort canonical-key annotations. Empty
        tuple for legacy rows or when the extractor couldn't link a
        name to any participant.
    """

    id: int
    familiar_id: str
    channel_id: int | None
    text: str
    source_turn_ids: tuple[int, ...]
    created_at: datetime
    superseded_at: datetime | None = None
    superseded_by: int | None = None
    subjects: tuple[FactSubject, ...] = ()


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

CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id       TEXT    NOT NULL,
    channel_id        INTEGER,
    text              TEXT    NOT NULL,
    source_turn_ids   TEXT    NOT NULL,  -- JSON array of ids in ``turns``
    created_at        TEXT    NOT NULL,
    superseded_at     TEXT,               -- NULL = current
    superseded_by     INTEGER,            -- NULL = current; FK-by-convention
    subjects_json     TEXT                -- JSON list; NULL = legacy fact
);

CREATE INDEX IF NOT EXISTS idx_facts_familiar
    ON facts (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_facts_familiar_current
    ON facts (familiar_id, superseded_at, id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(
    text,
    content='facts',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS facts_ai_fts
AFTER INSERT ON facts BEGIN
    INSERT INTO fts_facts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad_fts
AFTER DELETE ON facts BEGIN
    INSERT INTO fts_facts (fts_facts, rowid, text)
        VALUES ('delete', old.id, old.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS fts_turns USING fts5(
    content,
    content='turns',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Keep the FTS index in sync with ``turns``.
CREATE TRIGGER IF NOT EXISTS turns_ai_fts
AFTER INSERT ON turns BEGIN
    INSERT INTO fts_turns (rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad_fts
AFTER DELETE ON turns BEGIN
    INSERT INTO fts_turns (fts_turns, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_au_fts
AFTER UPDATE ON turns BEGIN
    INSERT INTO fts_turns (fts_turns, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO fts_turns (rowid, content) VALUES (new.id, new.content);
END;
"""

_TURN_COLS = (
    "id, timestamp, role, author_platform, author_user_id, "
    "author_username, author_display_name, content, channel_id"
)


_FTS_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

# Drop common English stopwords before FTS matching. Without this, casual
# chat cues like "hey do you know about X" dilute BM25 scoring (every token
# is OR'd) and produce noisy hits on conversational filler. Keep the list
# small and high-confidence — over-filtering hurts recall.
_FTS_STOPWORDS = frozenset({
    "a",
    "about",
    "an",
    "and",
    "any",
    "anything",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hey",
    "hi",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "know",
    "lol",
    "me",
    "my",
    "no",
    "not",
    "of",
    "ok",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "yes",
    "you",
    "your",
    "yours",
})


def _tokenize_fts_query(query: str) -> list[str]:
    """Tokenize free-form text into FTS5 MATCH tokens.

    Strips punctuation, lowercases, drops English stopwords, and
    appends the FTS5 prefix operator (``*``) to tokens of 3+ chars so
    ``fox`` also matches ``foxes`` (default tokenizer has no stemmer).
    Returns ``[]`` for empty / all-stopword input — caller short-
    circuits when no useful tokens remain.

    See :func:`_build_fts_match` for how callers compose the result
    into an OR-joined MATCH expression.
    """
    if not query or not query.strip():
        return []
    out: list[str] = []
    for tok in _FTS_TOKEN_RE.findall(query):
        low = tok.lower()
        if low in _FTS_STOPWORDS:
            continue
        if len(low) >= 3:
            out.append(f"{low}*")
        else:
            out.append(low)
    return out


def _build_fts_match(query: str) -> str:
    """OR-joined MATCH expression for free-form text.

    FTS5's default operator is implicit AND — every token must appear
    in the indexed row. That destroys recall on multi-word chat cues
    (a 16-token query that mixes filler with substantive nouns
    almost never matches a 6-word fact). OR-joining lets BM25 rank
    on whichever substantive tokens hit; common tokens contribute
    little weight.
    """
    return " OR ".join(_tokenize_fts_query(query))


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
        # Cleanup debt: remove this branch + the related legacy test
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

        # facts: add supersession columns if missing. Existing facts
        # default to current (NULL on both columns).
        facts_row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
        ).fetchone()
        if facts_row is not None:
            facts_cols = {
                col["name"]
                for col in self._conn.execute("PRAGMA table_info(facts)").fetchall()
            }
            if "superseded_at" not in facts_cols:
                self._conn.execute("ALTER TABLE facts ADD COLUMN superseded_at TEXT")
            if "superseded_by" not in facts_cols:
                self._conn.execute("ALTER TABLE facts ADD COLUMN superseded_by INTEGER")
            if "subjects_json" not in facts_cols:
                self._conn.execute("ALTER TABLE facts ADD COLUMN subjects_json TEXT")
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
        mode: str | None = None,
    ) -> HistoryTurn:
        """Append a single turn and return its persisted form.

        *mode* is a free-form string tag on the ``turns.mode`` column.
        """
        timestamp = datetime.now(tz=UTC)
        mode_value = mode
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
        mode: str | None = None,
    ) -> list[HistoryTurn]:
        """Return most recent turns in channel, oldest-first.

        Per-channel partitioning prevents bleed between conversations.
        When *mode* is set, only matching legacy-tag turns returned.
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
                (familiar_id, channel_id, mode, limit),
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

    def recent_distinct_authors(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        limit: int,
    ) -> list[Author]:
        """Return up to *limit* most-recently-seen distinct user authors.

        Most-recent-first ordering by canonical key (platform + user_id).
        Skips turns without an author (assistant replies, system events).
        Scoped to one channel — matches :meth:`recent`.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT author_platform, author_user_id,
                   author_username, author_display_name,
                   MAX(id) AS max_id
              FROM turns
             WHERE familiar_id = ?
               AND channel_id = ?
               AND author_platform IS NOT NULL
               AND author_user_id IS NOT NULL
             GROUP BY author_platform, author_user_id
             ORDER BY max_id DESC
             LIMIT ?
            """,
            (familiar_id, channel_id, limit),
        ).fetchall()
        return [
            Author(
                platform=str(row["author_platform"]),
                user_id=str(row["author_user_id"]),
                username=row["author_username"],
                display_name=row["author_display_name"],
            )
            for row in rows
        ]

    def latest_author_for(
        self,
        *,
        familiar_id: str,
        canonical_key: str,
    ) -> Author | None:
        """Return the :class:`Author` from the most recent turn with this key.

        Display names rotate (Discord/Twitch nicks); the latest turn
        carries the freshest one. Returns ``None`` if no turn matches —
        e.g. the user hasn't spoken in this familiar, or the
        canonical_key isn't well-formed. Used by
        :class:`RagContextLayer` to resolve stale fact-subject names
        at read time.
        """
        if ":" not in canonical_key:
            return None
        platform, _, user_id = canonical_key.partition(":")
        if not platform or not user_id:
            return None
        row = self._conn.execute(
            """
            SELECT author_platform, author_user_id,
                   author_username, author_display_name
              FROM turns
             WHERE familiar_id = ?
               AND author_platform = ?
               AND author_user_id = ?
             ORDER BY id DESC
             LIMIT 1
            """,
            (familiar_id, platform, user_id),
        ).fetchone()
        if row is None:
            return None
        return Author(
            platform=str(row["author_platform"]),
            user_id=str(row["author_user_id"]),
            username=row["author_username"],
            display_name=row["author_display_name"],
        )

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

    # ------------------------------------------------------------------
    # FTS side-index over ``turns.content``
    # ------------------------------------------------------------------

    def search_turns(
        self,
        *,
        familiar_id: str,
        query: str,
        limit: int,
        channel_id: int | None = None,
        max_id: int | None = None,
    ) -> list[HistoryTurn]:
        """Return turns whose content matches the FTS *query*.

        Empty/whitespace *query* and queries that reduce to only
        stopwords return ``[]``. Tokens are OR-joined and BM25-ranked;
        see :func:`_build_fts_match`.

        :param max_id: if set, only turns with ``id <= max_id`` are
            considered. Used by :class:`RagContextLayer` to keep RAG
            from re-surfacing turns already covered by
            :class:`RecentHistoryLayer`.
        """
        if limit <= 0:
            return []
        match_expr = _build_fts_match(query)
        if not match_expr:
            return []

        params: list[object] = [match_expr, familiar_id]
        where_extra = ""
        if channel_id is not None:
            where_extra += "AND t.channel_id = ?\n"
            params.append(channel_id)
        if max_id is not None:
            where_extra += "AND t.id <= ?\n"
            params.append(max_id)
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT {", ".join("t." + c for c in _TURN_COLS.split(", "))}
              FROM fts_turns AS fts
              JOIN turns AS t ON t.id = fts.rowid
             WHERE fts_turns MATCH ?
               AND t.familiar_id = ?
               {where_extra}
             ORDER BY bm25(fts_turns) ASC, t.id DESC
             LIMIT ?
            """,  # noqa: S608
            params,
        ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def rebuild_fts(self) -> None:
        """Drop and repopulate the ``fts_turns`` index from ``turns``.

        Cheap relative to re-running every LLM call; cheap enough to
        run at startup if triggers ever get out of sync.
        """
        self._conn.executescript(
            """
            DELETE FROM fts_turns;
            INSERT INTO fts_turns (rowid, content)
            SELECT id, content FROM turns;
            """
        )
        self._conn.commit()

    def latest_fts_id(self, *, familiar_id: str) -> int:
        """Return the highest turn id currently indexed for ``familiar_id``.

        The FTS index is updated by trigger in lockstep with ``turns``
        writes, so this is an ``id``-lookup, not a separate watermark.
        """
        row = self._conn.execute(
            """
            SELECT MAX(t.id) AS max_id
              FROM fts_turns AS fts
              JOIN turns AS t ON t.id = fts.rowid
             WHERE t.familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        max_id = row["max_id"] if row is not None else None
        return int(max_id) if max_id is not None else 0

    # ------------------------------------------------------------------
    # Facts — atomic distilled statements with provenance
    # ------------------------------------------------------------------

    def append_fact(
        self,
        *,
        familiar_id: str,
        channel_id: int | None,
        text: str,
        source_turn_ids: Iterable[int],
        subjects: Iterable[FactSubject] = (),
    ) -> Fact:
        """Persist one fact. ``source_turn_ids`` and ``subjects`` stored as JSON.

        ``subjects`` is the extractor's best-effort link to canonical
        identities — see :class:`FactSubject`.
        """
        ids = [int(i) for i in source_turn_ids]
        subjects_tuple = tuple(subjects)
        subjects_blob: str | None = (
            json.dumps([
                {
                    "canonical_key": s.canonical_key,
                    "display_at_write": s.display_at_write,
                }
                for s in subjects_tuple
            ])
            if subjects_tuple
            else None
        )
        ts = datetime.now(tz=UTC)
        cur = self._conn.execute(
            """
            INSERT INTO facts (familiar_id, channel_id, text,
                               source_turn_ids, created_at, subjects_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                channel_id,
                text,
                json.dumps(ids),
                ts.isoformat(),
                subjects_blob,
            ),
        )
        self._conn.commit()
        return Fact(
            id=int(cur.lastrowid or 0),
            familiar_id=familiar_id,
            channel_id=channel_id,
            text=text,
            source_turn_ids=tuple(ids),
            created_at=ts,
            subjects=subjects_tuple,
        )

    def recent_facts(
        self,
        *,
        familiar_id: str,
        limit: int,
        include_superseded: bool = False,
    ) -> list[Fact]:
        """Return the ``limit`` most recent facts, newest first.

        Excludes superseded facts unless ``include_superseded`` is set —
        the default is "what's currently true". Audit / contradiction
        inspection passes ``include_superseded=True``.
        """
        if limit <= 0:
            return []
        where_super = "" if include_superseded else "AND superseded_at IS NULL"
        rows = self._conn.execute(
            f"""
            SELECT id, familiar_id, channel_id, text,
                   source_turn_ids, created_at,
                   superseded_at, superseded_by, subjects_json
              FROM facts
             WHERE familiar_id = ?
               {where_super}
             ORDER BY id DESC
             LIMIT ?
            """,  # noqa: S608
            (familiar_id, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def search_facts(
        self,
        *,
        familiar_id: str,
        query: str,
        limit: int,
        include_superseded: bool = False,
    ) -> list[Fact]:
        """FTS search over ``facts.text``.

        See :meth:`search_turns` for tokenisation notes. Superseded
        facts are excluded by default; the FTS index itself still
        covers them (so unsupersede via re-supersede stays cheap).
        """
        if limit <= 0:
            return []
        match_expr = _build_fts_match(query)
        if not match_expr:
            return []
        where_super = "" if include_superseded else "AND f.superseded_at IS NULL"
        rows = self._conn.execute(
            f"""
            SELECT f.id, f.familiar_id, f.channel_id, f.text,
                   f.source_turn_ids, f.created_at,
                   f.superseded_at, f.superseded_by, f.subjects_json
              FROM fts_facts AS fts
              JOIN facts AS f ON f.id = fts.rowid
             WHERE fts_facts MATCH ?
               AND f.familiar_id = ?
               {where_super}
             ORDER BY bm25(fts_facts) ASC, f.id DESC
             LIMIT ?
            """,  # noqa: S608
            (match_expr, familiar_id, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def latest_fact_id(self, *, familiar_id: str) -> int:
        """Return highest ``facts.id`` for ``familiar_id``; 0 if none.

        Counts superseded rows too — the cache invalidation key only
        needs to change on writes, and supersession-by-replacement
        already adds a new row so the id ticks up naturally.
        """
        row = self._conn.execute(
            "SELECT MAX(id) AS max_id FROM facts WHERE familiar_id = ?",
            (familiar_id,),
        ).fetchone()
        max_id = row["max_id"] if row is not None else None
        return int(max_id) if max_id is not None else 0

    def supersede_fact(
        self,
        *,
        familiar_id: str,
        old_id: int,
        new_id: int,
    ) -> None:
        """Mark ``old_id`` as superseded by ``new_id``.

        Both ids must belong to ``familiar_id``. The old row keeps its
        text and provenance; only ``superseded_at`` (now, UTC) and
        ``superseded_by`` are written. Re-superseding a row that's
        already superseded raises ``ValueError`` — that signals an
        upstream bug (double-write) rather than something to silently
        absorb.
        """
        row = self._conn.execute(
            "SELECT superseded_at FROM facts WHERE id = ? AND familiar_id = ?",
            (old_id, familiar_id),
        ).fetchone()
        if row is None:
            msg = f"unknown fact id={old_id} for familiar={familiar_id}"
            raise ValueError(msg)
        if row["superseded_at"] is not None:
            msg = f"fact id={old_id} already superseded"
            raise ValueError(msg)
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            UPDATE facts
               SET superseded_at = ?, superseded_by = ?
             WHERE id = ? AND familiar_id = ?
            """,
            (ts, new_id, old_id, familiar_id),
        )
        self._conn.commit()


def _row_to_fact(row: sqlite3.Row) -> Fact:
    ids_raw = row["source_turn_ids"]
    try:
        ids = tuple(int(x) for x in json.loads(ids_raw))
    except (ValueError, TypeError):
        ids = ()
    channel = row["channel_id"]
    superseded_at_raw: str | None
    superseded_by_raw: int | None
    try:
        superseded_at_raw = row["superseded_at"]
    except (IndexError, KeyError):
        superseded_at_raw = None
    try:
        superseded_by_raw = row["superseded_by"]
    except (IndexError, KeyError):
        superseded_by_raw = None
    try:
        subjects_raw = row["subjects_json"]
    except (IndexError, KeyError):
        subjects_raw = None
    subjects: tuple[FactSubject, ...] = ()
    if subjects_raw:
        try:
            parsed = json.loads(subjects_raw)
        except (ValueError, TypeError):
            parsed = []
        if isinstance(parsed, list):
            subjects = tuple(
                FactSubject(
                    canonical_key=str(item["canonical_key"]),
                    display_at_write=str(item["display_at_write"]),
                )
                for item in parsed
                if isinstance(item, dict)
                and "canonical_key" in item
                and "display_at_write" in item
            )
    return Fact(
        id=int(row["id"]),
        familiar_id=str(row["familiar_id"]),
        channel_id=int(channel) if channel is not None else None,
        text=str(row["text"]),
        source_turn_ids=ids,
        created_at=datetime.fromisoformat(row["created_at"]),
        superseded_at=(
            datetime.fromisoformat(superseded_at_raw)
            if superseded_at_raw is not None
            else None
        ),
        superseded_by=int(superseded_by_raw) if superseded_by_raw is not None else None,
        subjects=subjects,
    )


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
