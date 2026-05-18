"""Turso-backed persistent conversation history (FTS via tantivy).

One database per familiar. Two core tables:

- ``turns`` — every turn scoped by ``(familiar_id, channel_id)``;
  monotonic PK, role, optional :class:`Author`, content, optional
  guild_id, UTC ts
- ``summaries`` — at most one rolling summary per (familiar, channel)
  with a ``last_summarised_id`` watermark for cache freshness.
  See ``docs/architecture/context-pipeline.md`` for rationale.

Relational storage uses Turso (SQLite-compatible Rust rewrite); FTS
lives in a sibling tantivy index under ``fts/turns/`` and ``fts/facts/``
because pyturso wheels don't ship the FTS module yet. See
``docs/architecture/turso-migration.md`` for the migration story.

``familiar_id`` is explicit (not implicit) so tests can exercise
multiple familiars against one store. Synchronous API; the
:class:`AsyncHistoryStore` wrapper dispatches calls to a thread pool
that holds per-thread Turso connections.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import turso

from familiar_connect import log_style as ls
from familiar_connect.history.fts import FtsIndex
from familiar_connect.history.turso_compat import TursoConnection
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from collections.abc import Iterable

_logger = logging.getLogger(__name__)

# Result rows come back as ``turso.Row``; spell as ``Any`` to keep the
# private ``_row_to_*`` helpers free of a third-party type alias.
Row = Any

PathLike = str | Path


@dataclass(frozen=True)
class HistoryTurn:
    """Single persisted conversational turn.

    :param platform_message_id: native id from the platform of origin
        (Discord snowflake, etc.). ``None`` for legacy rows or
        platform-less inputs.
    :param reply_to_message_id: parent ``platform_message_id`` when
        this turn is a reply (e.g. ``message.reference`` on Discord).
    :param guild_id: Discord guild scoping per-guild nicknames for
        rendering. ``None`` for DMs / non-Discord platforms / legacy
        rows where the column wasn't populated.
    """

    id: int
    timestamp: datetime
    role: str
    author: Author | None
    content: str
    channel_id: int = 0
    platform_message_id: str | None = None
    reply_to_message_id: str | None = None
    guild_id: int | None = None


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
class AccountProfile:
    """Read-side projection of ``accounts`` profile metadata.

    Populated on Discord ingestion via :meth:`HistoryStore.upsert_account`;
    consumed by :class:`PeopleDossierLayer` to surface basic identity
    (username, pronouns, bio) on the prompt header.
    """

    canonical_key: str
    username: str | None
    global_name: str | None
    pronouns: str | None
    bio: str | None


@dataclass(frozen=True)
class PeopleDossierEntry:
    """Cached per-person dossier compounded from facts mentioning ``canonical_key``.

    ``last_fact_id`` is a watermark over ``facts.id`` — the worker
    refreshes when ``subjects_with_facts`` reports a higher id for
    this subject. Mirrors :class:`SummaryEntry`.
    """

    canonical_key: str
    last_fact_id: int
    dossier_text: str
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
class Reflection:
    """Higher-order synthesis over recent turns + facts (M3).

    Written by :class:`ReflectionWorker`; read by
    :class:`ReflectionLayer`. ``cited_turn_ids`` / ``cited_fact_ids``
    are forever-provenance — never edited, never trimmed. A reflection
    citing a superseded fact stays in the table; the read path flags
    it stale rather than dropping it (audit trail beats silent loss).

    :param last_turn_id: highest ``turns.id`` visible to the worker at
        write time. Doubles as the next tick's watermark — one row per
        write, not a separate table.
    :param last_fact_id: highest ``facts.id`` visible to the worker at
        write time. Same role as ``last_turn_id`` for the facts axis.
    """

    id: int
    familiar_id: str
    channel_id: int | None
    text: str
    cited_turn_ids: tuple[int, ...]
    cited_fact_ids: tuple[int, ...]
    created_at: datetime
    last_turn_id: int
    last_fact_id: int


@dataclass(frozen=True)
class Fact:
    """Atomic fact extracted from one or more turns.

    :param source_turn_ids: ids in ``turns`` the fact was distilled
        from — forever provenance, per plan § Design.5.
    :param superseded_at: system-time — when this fact was retired,
        or ``None`` if still current. Supersession keeps the row (no
        delete) so the prior state stays visible for audit.
    :param superseded_by: id of the replacement fact, or ``None`` if
        still current.
    :param subjects: best-effort canonical-key annotations. Empty
        tuple for legacy rows or when the extractor couldn't link a
        name to any participant.
    :param valid_from: world-time — when the fact began applying.
        Default is the source turn's timestamp; LLM may override when
        an explicit "as of …" phrase is detected. ``None`` only on
        legacy rows pre-M1.
    :param valid_to: world-time — when the fact stopped applying;
        ``None`` while still in effect.
    :param importance: 1-10 hint for retrieval ranking (M2). ``None``
        on legacy rows or when the extractor declined to score.
        Treated as the neutral midpoint by rank-time consumers.
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
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    importance: int | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id            TEXT    NOT NULL,
    channel_id             INTEGER NOT NULL,
    guild_id               INTEGER,
    role                   TEXT    NOT NULL,
    author_platform        TEXT,
    author_user_id         TEXT,
    author_username        TEXT,
    author_display_name    TEXT,
    content                TEXT    NOT NULL,
    timestamp              TEXT    NOT NULL,
    mode                   TEXT,
    platform_message_id    TEXT,
    reply_to_message_id    TEXT,
    tool_calls_json        TEXT,
    tool_call_id           TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_channel
    ON turns (familiar_id, channel_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_global
    ON turns (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_channel_mode
    ON turns (familiar_id, channel_id, mode, id);

CREATE INDEX IF NOT EXISTS idx_turns_platform_msg
    ON turns (familiar_id, platform_message_id);

-- Discord reactions on persisted messages. Keyed by the platform-
-- native message id so we can update without touching ``turns``;
-- per (familiar, message, emoji) row stores the live count from
-- gateway events. ``count = 0`` rows are deleted at write time.
CREATE TABLE IF NOT EXISTS message_reactions (
    familiar_id          TEXT    NOT NULL,
    platform_message_id  TEXT    NOT NULL,
    emoji                TEXT    NOT NULL,
    count                INTEGER NOT NULL,
    updated_at           TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, platform_message_id, emoji)
);

CREATE INDEX IF NOT EXISTS idx_message_reactions_lookup
    ON message_reactions (familiar_id, platform_message_id);

CREATE TABLE IF NOT EXISTS turn_mentions (
    turn_id        INTEGER NOT NULL,
    canonical_key  TEXT    NOT NULL,
    PRIMARY KEY (turn_id, canonical_key)
);

CREATE INDEX IF NOT EXISTS idx_turn_mentions_canonical
    ON turn_mentions (canonical_key, turn_id);

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

CREATE TABLE IF NOT EXISTS people_dossiers (
    familiar_id    TEXT    NOT NULL,
    canonical_key  TEXT    NOT NULL,
    last_fact_id   INTEGER NOT NULL,
    dossier_text   TEXT    NOT NULL,
    created_at     TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, canonical_key)
);

-- Reflections (M3). Higher-order syntheses over recent turns + facts.
-- Provenance forever via cited_turn_ids / cited_fact_ids (JSON arrays).
-- ``last_turn_id`` / ``last_fact_id`` snapshot the worker's view at
-- write time so the next tick can detect freshness without a separate
-- watermark table. Citations to superseded facts surface as "stale" at
-- read time; the row itself is never deleted.
CREATE TABLE IF NOT EXISTS reflections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id     TEXT    NOT NULL,
    channel_id      INTEGER,
    text            TEXT    NOT NULL,
    cited_turn_ids  TEXT    NOT NULL,  -- JSON array
    cited_fact_ids  TEXT    NOT NULL,  -- JSON array
    created_at      TEXT    NOT NULL,
    last_turn_id    INTEGER NOT NULL,  -- watermark over turns at write
    last_fact_id    INTEGER NOT NULL   -- watermark over facts at write
);

CREATE INDEX IF NOT EXISTS idx_reflections_familiar
    ON reflections (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_reflections_familiar_channel
    ON reflections (familiar_id, channel_id, id);

-- Identity. One row per (platform, user_id). Last-write wins.
CREATE TABLE IF NOT EXISTS accounts (
    canonical_key  TEXT PRIMARY KEY,           -- "discord:123" / "twitch:456"
    platform       TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    username       TEXT,                        -- global handle
    global_name    TEXT,                        -- global display name
    pronouns       TEXT,                        -- profile pronouns; NULL when unknown
    bio            TEXT,                        -- profile bio; NULL when unknown
    last_seen_at   TEXT NOT NULL
);

-- Per-guild nickname cache. NULL nick = explicit "no override".
CREATE TABLE IF NOT EXISTS account_guild_nicks (
    canonical_key  TEXT NOT NULL,
    guild_id       INTEGER NOT NULL,
    nick           TEXT,
    last_seen_at   TEXT NOT NULL,
    PRIMARY KEY (canonical_key, guild_id)
);

CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id       TEXT    NOT NULL,
    channel_id        INTEGER,
    text              TEXT    NOT NULL,
    source_turn_ids   TEXT    NOT NULL,  -- JSON array of ids in ``turns``
    created_at        TEXT    NOT NULL,  -- system-time write
    superseded_at     TEXT,               -- system-time retire; NULL = current
    superseded_by     INTEGER,            -- NULL = current; FK-by-convention
    subjects_json     TEXT,               -- JSON list; NULL = legacy fact
    valid_from        TEXT,               -- world-time start; NULL = legacy
    valid_to          TEXT,               -- world-time end; NULL = still applies
    importance        INTEGER             -- 1-10; NULL = unknown / legacy
);

CREATE INDEX IF NOT EXISTS idx_facts_familiar
    ON facts (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_facts_familiar_current
    ON facts (familiar_id, superseded_at, id);

CREATE INDEX IF NOT EXISTS idx_facts_familiar_validity
    ON facts (familiar_id, valid_from, valid_to);

-- Per-fact embeddings (M6). Vectors stored as packed float32 BLOBs;
-- ``model`` is the embedder's ``name`` so a model swap creates a new
-- row rather than overwriting (audit history preserved). The
-- :class:`FactEmbeddingWorker` projector populates this table from
-- ``facts``; ``RagContextLayer`` reads it at rerank time.
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id     INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (fact_id, model)
);

CREATE INDEX IF NOT EXISTS idx_fact_embeddings_model
    ON fact_embeddings (model, fact_id);

-- Scheduled wakes (tool-driven). Append-only with soft-delete:
-- ``fired_at`` flips when the scheduler dispatches; ``cancelled_at``
-- flips when a user/tool cancels. Pending rows have both NULL.
CREATE TABLE IF NOT EXISTS alarms (
    id                   TEXT    PRIMARY KEY,
    familiar_id          TEXT    NOT NULL,
    channel_id           INTEGER NOT NULL,
    channel_kind         TEXT    NOT NULL CHECK(channel_kind IN ('text','voice')),
    scheduled_at         TEXT    NOT NULL,
    reason               TEXT    NOT NULL,
    originating_turn_id  TEXT,
    fired_at             TEXT,
    cancelled_at         TEXT,
    created_at           TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alarms_pending
    ON alarms (familiar_id, fired_at, cancelled_at, scheduled_at);
"""

_TURN_COLS = (
    "id, timestamp, role, author_platform, author_user_id, "
    "author_username, author_display_name, content, channel_id, "
    "platform_message_id, reply_to_message_id, guild_id"
)

# Table names declared in _SCHEMA — used by HistoryStore to verify the
# schema is intact after init (pyturso 0.5.1 on Windows occasionally
# drops CREATE TABLE statements; see _ensure_expected_tables).
_EXPECTED_TABLES: frozenset[str] = frozenset(
    re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA)
)


def _facts_validity_where(
    *,
    include_superseded: bool,
    as_of: datetime | None,
    alias: str = "",
) -> tuple[str, tuple[object, ...]]:
    """Build SQL fragment + params for the ``facts`` validity filter.

    Default ("current truth"):
        ``superseded_at IS NULL AND (valid_to IS NULL OR valid_to > now)``

    With ``as_of``:
        bi-temporal slice — ``valid_from`` IS NULL or <= as_of, and
        ``valid_to`` IS NULL or > as_of. Includes superseded rows so
        audit queries can recover prior beliefs (overrides
        ``include_superseded``).
    """
    prefix = f"{alias}." if alias else ""
    if as_of is not None:
        ts = as_of.isoformat()
        clause = (
            f"AND ({prefix}valid_from IS NULL OR {prefix}valid_from <= ?) "
            f"AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)"
        )
        return clause, (ts, ts)
    parts: list[str] = []
    params: list[object] = []
    if not include_superseded:
        parts.append(f"AND {prefix}superseded_at IS NULL")
    now_ts = datetime.now(tz=UTC).isoformat()
    parts.append(f"AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)")
    params.append(now_ts)
    return " ".join(parts), tuple(params)


class HistoryStore:
    """Persistent SQLite store for turns + rolling summaries.

    Pass ``":memory:"`` for an ephemeral in-process database (tests).
    """

    def __init__(self, db_path: PathLike) -> None:
        if db_path == ":memory:":
            self._path: Path | None = None
            self._conn = TursoConnection(":memory:")
            fts_turns_path: Path | None = None
            fts_facts_path: Path | None = None
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._conn = TursoConnection(path)
            fts_root = path.parent / "fts"
            fts_turns_path = fts_root / "turns"
            fts_facts_path = fts_root / "facts"
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db")
        self._migrate_if_needed()
        self._conn.commit()
        # pyturso 0.5.1 on Windows: migration's ``sqlite_master`` /
        # ``PRAGMA table_info`` queries pollute the schema cache so
        # the following ``CREATE TABLE IF NOT EXISTS`` silently fails
        # with ``Parse error: no such table: …`` (swallowed by
        # ``_execute_schema``). Reopening clears the cache so schema
        # creation runs against a fresh read of sqlite_master. Skip
        # for ``:memory:`` — reopening would lose the migration.
        if self._path is not None:
            self._conn.reopen()
        self._execute_schema(_SCHEMA)
        self._conn.commit()
        # Reopen again so runtime queries see the just-created tables
        # (same cache-staleness bug applies on the read side).
        if self._path is not None:
            self._conn.reopen()
        # Defensive verify + one retry on a fresh connection: if the
        # first ``_execute_schema`` pass silently swallowed a CREATE
        # for any expected table, re-run on the fresh cache.
        self._ensure_expected_tables()
        self._fts_turns = FtsIndex(fts_turns_path)
        self._fts_facts = FtsIndex(fts_facts_path)
        # Tantivy indexes are independent files; on first run after the
        # sqlite→turso migration (or after the user nukes ``fts/``)
        # they're empty while ``turns``/``facts`` already have rows.
        # Detect and bulk-reindex.
        self._reindex_if_empty()

    # pyturso 0.5.1 (Windows) emits a grab-bag of parse errors when
    # re-running ``CREATE … IF NOT EXISTS`` against a populated DB:
    # ``already exists`` (index cache stale), ``no such table`` /
    # ``does not exist`` (schema cache lags behind the CREATE TABLE
    # we just ran). All of them are benign on an idempotent schema —
    # the table / index is either already present or will be created
    # on a later run once Turso catches up.
    _SCHEMA_PARSE_ERROR_FRAGMENTS = (
        "already exists",
        "no such table",
        "does not exist",
    )

    def _execute_schema(self, script: str) -> None:
        """Run an idempotent schema script, tolerating Turso parse quirks.

        Plain ``executescript`` aborts on the first statement that
        raises. Strip line + trailing comments (some contain ``;``)
        then split and execute one statement at a time, swallowing
        the parse-error variants listed in
        ``_SCHEMA_PARSE_ERROR_FRAGMENTS`` because every statement in
        ``_SCHEMA`` is shaped ``CREATE … IF NOT EXISTS``.
        """
        cleaned_lines: list[str] = []
        for raw_line in script.splitlines():
            comment_pos = raw_line.find("--")
            line = raw_line[:comment_pos] if comment_pos != -1 else raw_line
            if line.strip():
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        for raw in cleaned.split(";"):
            stmt = raw.strip()
            if not stmt:
                continue
            try:
                self._conn.execute(stmt)
            except turso.DatabaseError as exc:
                msg = str(exc).lower()
                if any(f in msg for f in self._SCHEMA_PARSE_ERROR_FRAGMENTS):
                    continue
                raise

    def _ensure_expected_tables(self) -> None:
        """Re-run ``_SCHEMA`` if any expected table is missing.

        pyturso 0.5.1 on Windows can silently drop a CREATE TABLE
        statement when the schema cache is polluted by earlier
        ``sqlite_master`` / ``PRAGMA`` queries: ``_execute_schema``
        sees ``Parse error: no such table: <name>`` and swallows it,
        but the table never lands on disk. After the post-schema
        reopen, sweep ``sqlite_master`` for the expected set; if any
        are missing, run ``_execute_schema`` again on the now-fresh
        cache and reopen once more. Raise on the second miss.
        """
        if self._path is None:
            return
        for attempt in (1, 2):
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            actual = {row["name"] for row in rows}
            _logger.info(
                f"{ls.tag('History', ls.C)} "
                f"{ls.kv('attempt', str(attempt))} "
                f"{ls.kv('sqlite_master_tables', ','.join(sorted(actual)) or '<none>')}"
            )
            missing = _EXPECTED_TABLES - actual
            if not missing:
                self._probe_expected_tables()
                return
            if attempt == 2:
                msg = (
                    f"history schema incomplete after retry; "
                    f"missing tables: {sorted(missing)}"
                )
                raise RuntimeError(msg)
            _logger.warning(
                f"{ls.tag('History', ls.Y)} "
                f"{ls.kv('missing_tables', ','.join(sorted(missing)))} "
                f"{ls.kv('action', 'retry-schema')}"
            )
            self._execute_schema(_SCHEMA)
            self._conn.commit()
            self._conn.reopen()

    def _probe_expected_tables(self) -> None:
        """Run a no-op ``SELECT`` against each expected table.

        ``sqlite_master`` reports a row for the table, but pyturso
        0.5.1 on Windows has been seen to refuse statement
        preparation against tables that *should* exist
        (``Parse error: no such table: <name>``). Probe each table
        explicitly and log the result so we can tell — from a
        single boot log — whether the schema is consistent or
        ``sqlite_master`` is reporting phantom entries.
        """
        for table in sorted(_EXPECTED_TABLES):
            try:
                self._conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchall()  # noqa: S608
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    f"{ls.tag('History', ls.R)} "
                    f"{ls.kv('probe_table', table)} "
                    f"{ls.kv('error', type(exc).__name__)} "
                    f"{ls.kv('detail', str(exc))}"
                )
            else:
                _logger.info(
                    f"{ls.tag('History', ls.G)} "
                    f"{ls.kv('probe_table', table)} "
                    f"{ls.kv('result', 'ok')}"
                )

    def _safe_add_column(self, table: str, column: str, type_: str) -> None:
        """ALTER TABLE ADD COLUMN, swallowing benign migration errors.

        Tolerates two parse-error variants:

        * ``duplicate column`` — column already added on a prior run
        * ``no such table`` — table absent (or pyturso 0.5.1 reporting
          phantom state inconsistent with ``sqlite_master``/``PRAGMA
          table_info``; observed on Windows). ``_SCHEMA``'s
          ``CREATE TABLE IF NOT EXISTS`` runs after migration and
          creates the table fresh with the new columns.

        Other ``DatabaseError`` instances propagate.
        """
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_}")
        except turso.DatabaseError as exc:
            msg = str(exc).lower()
            if "duplicate column" in msg or "no such table" in msg:
                return
            raise

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
            self._safe_add_column("turns", "mode", "TEXT")
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
                    self._safe_add_column("turns", col, "TEXT")
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
            "platform_message_id",
            "reply_to_message_id",
        ):
            if col not in columns:
                self._safe_add_column("turns", col, "TEXT")
        if "guild_id" not in columns:
            self._safe_add_column("turns", "guild_id", "INTEGER")
        # tool calling: assistant tool_calls stored as JSON, tool-role
        # turns reference the call id they answered.
        if "tool_calls_json" not in columns:
            self._safe_add_column("turns", "tool_calls_json", "TEXT")
        if "tool_call_id" not in columns:
            self._safe_add_column("turns", "tool_call_id", "TEXT")
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

        # accounts: add profile columns if missing. Pre-existing rows
        # default to NULL — populated next time the user is observed.
        accounts_row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'"
        ).fetchone()
        if accounts_row is not None:
            accounts_cols = {
                col["name"]
                for col in self._conn.execute("PRAGMA table_info(accounts)").fetchall()
            }
            if "pronouns" not in accounts_cols:
                self._safe_add_column("accounts", "pronouns", "TEXT")
            if "bio" not in accounts_cols:
                self._safe_add_column("accounts", "bio", "TEXT")
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
                self._safe_add_column("facts", "superseded_at", "TEXT")
            if "superseded_by" not in facts_cols:
                self._safe_add_column("facts", "superseded_by", "INTEGER")
            if "subjects_json" not in facts_cols:
                self._safe_add_column("facts", "subjects_json", "TEXT")
            if "valid_from" not in facts_cols:
                self._safe_add_column("facts", "valid_from", "TEXT")
            if "valid_to" not in facts_cols:
                self._safe_add_column("facts", "valid_to", "TEXT")
            if "importance" not in facts_cols:
                self._safe_add_column("facts", "importance", "INTEGER")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down executor, FTS writers, and Turso connections."""
        self._executor.shutdown(wait=False)
        try:
            self._fts_turns.close()
        finally:
            try:
                self._fts_facts.close()
            finally:
                self._conn.close()

    def _reindex_if_empty(self) -> None:
        """Bulk-rebuild a tantivy index when relational rows lack matching docs.

        Triggers on first run after the sqlite→turso migration script
        (which copies relational data but doesn't rebuild FTS) and on
        any later run where the user dropped ``fts/`` to recover from
        a corrupted index.
        """
        if self._fts_turns.is_empty():
            rows = self._conn.execute(
                "SELECT id, content FROM turns ORDER BY id ASC"
            ).fetchall()
            if rows:
                self._fts_turns.add_many([
                    (int(r["id"]), str(r["content"])) for r in rows
                ])
        if self._fts_facts.is_empty():
            rows = self._conn.execute(
                "SELECT id, text FROM facts ORDER BY id ASC"
            ).fetchall()
            if rows:
                self._fts_facts.add_many([(int(r["id"]), str(r["text"])) for r in rows])

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
        platform_message_id: str | None = None,
        reply_to_message_id: str | None = None,
        tool_calls_json: str | None = None,
        tool_call_id: str | None = None,
    ) -> HistoryTurn:
        """Append a single turn and return its persisted form.

        *mode* is a free-form string tag on the ``turns.mode`` column.
        *platform_message_id* / *reply_to_message_id* are platform-
        native ids (Discord snowflakes, etc.) stored as TEXT so any
        platform's id format fits.
        *tool_calls_json* carries the assistant's invoked tool calls
        (JSON-encoded list); *tool_call_id* references the call a
        ``role=tool`` turn is answering.
        """
        timestamp = datetime.now(tz=UTC)
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
                 content, timestamp, mode,
                 platform_message_id, reply_to_message_id,
                 tool_calls_json, tool_call_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                mode,
                platform_message_id,
                reply_to_message_id,
                tool_calls_json,
                tool_call_id,
            ),
        )
        self._conn.commit()
        turn_id = int(cur.lastrowid or 0)
        self._fts_turns.add(turn_id, content)
        return HistoryTurn(
            id=turn_id,
            timestamp=timestamp,
            role=role,
            author=author,
            content=content,
            channel_id=channel_id,
        )

    def lookup_turn_by_platform_message_id(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
    ) -> HistoryTurn | None:
        """Find the turn carrying ``platform_message_id`` for ``familiar_id``.

        Returns ``None`` if no row matches — used by the read path to
        resolve reply parents. The column index
        ``idx_turns_platform_msg`` keeps this O(1) per familiar.
        """
        row = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ? AND platform_message_id = ?
             ORDER BY id DESC
             LIMIT 1
            """,  # noqa: S608
            (familiar_id, platform_message_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_turn(row)

    def update_turn_content_by_message_id(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
        content: str,
    ) -> None:
        """Rewrite ``turns.content`` for one platform message id.

        Used when Discord delivers a URL unfurl after the original
        ``on_message`` (typical: ``message.embeds`` populates via a
        follow-up edit a second or two later). Silently no-ops when
        no row matches — the bot may have come up after the message
        landed. Refreshes the tantivy index for the affected row.
        """
        rows = self._conn.execute(
            "SELECT id FROM turns WHERE familiar_id = ? AND platform_message_id = ?",
            (familiar_id, platform_message_id),
        ).fetchall()
        self._conn.execute(
            """
            UPDATE turns
               SET content = ?
             WHERE familiar_id = ? AND platform_message_id = ?
            """,
            (content, familiar_id, platform_message_id),
        )
        self._conn.commit()
        for row in rows:
            self._fts_turns.add(int(row["id"]), content)

    def turns_by_ids(
        self,
        *,
        familiar_id: str,
        ids: Iterable[int],
    ) -> list[HistoryTurn]:
        """Fetch turns by id, scoped to ``familiar_id``, oldest first.

        Used by RAG to expand each FTS hit into a small surrounding
        window (hit ± neighbours) without a per-id round trip.
        """
        unique_ids = sorted({int(i) for i in ids})
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ?
               AND id IN ({placeholders})
             ORDER BY id ASC
            """,  # noqa: S608
            (familiar_id, *unique_ids),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]

    # ------------------------------------------------------------------
    # turn_mentions: many-to-many from a turn to mentioned canonical_keys
    # ------------------------------------------------------------------

    def record_mentions(
        self,
        *,
        turn_id: int,
        canonical_keys: Iterable[str],
    ) -> None:
        """Record the canonical keys mentioned in ``turn_id``.

        Idempotent — re-recording the same keys is a no-op thanks to
        the (turn_id, canonical_key) primary key. Empty input is a
        no-op too. Order is not preserved; reads come back sorted by
        canonical_key for determinism.
        """
        seen: set[str] = set()
        rows: list[tuple[int, str]] = []
        for key in canonical_keys:
            if key in seen:
                continue
            seen.add(key)
            rows.append((turn_id, key))
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO turn_mentions (turn_id, canonical_key)
            VALUES (?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def mentions_for_turn(self, *, turn_id: int) -> tuple[str, ...]:
        """Return the canonical keys mentioned in ``turn_id``, sorted."""
        rows = self._conn.execute(
            """
            SELECT canonical_key FROM turn_mentions
             WHERE turn_id = ?
             ORDER BY canonical_key ASC
            """,
            (turn_id,),
        ).fetchall()
        return tuple(str(r["canonical_key"]) for r in rows)

    # ------------------------------------------------------------------
    # message_reactions: emoji counts keyed by platform_message_id
    # ------------------------------------------------------------------

    def set_reaction(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
        emoji: str,
        count: int,
    ) -> None:
        """Upsert reaction count for one ``(message, emoji)`` pair.

        ``count <= 0`` deletes the row — mirrors Discord semantics
        where the last user removing a reaction collapses the entry.
        Idempotent for repeated identical writes (gateway dedup is
        cheap insurance here).
        """
        if count <= 0:
            self._conn.execute(
                """
                DELETE FROM message_reactions
                 WHERE familiar_id = ?
                   AND platform_message_id = ?
                   AND emoji = ?
                """,
                (familiar_id, platform_message_id, emoji),
            )
            self._conn.commit()
            return
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO message_reactions
                (familiar_id, platform_message_id, emoji, count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(familiar_id, platform_message_id, emoji)
            DO UPDATE SET count = excluded.count, updated_at = excluded.updated_at
            """,
            (familiar_id, platform_message_id, emoji, count, ts),
        )
        self._conn.commit()

    def bump_reaction(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
        emoji: str,
        delta: int,
    ) -> None:
        """Atomic ±delta on one ``(message, emoji)`` row.

        Drives the gateway hot path —``on_raw_reaction_add`` /
        ``on_raw_reaction_remove`` deliver per-user toggles, never
        absolute counts. Floors at zero (a stray remove without a
        matching add — e.g. bot was offline when the reaction was
        added — leaves no row rather than persisting a negative).
        """
        if delta == 0:
            return
        ts = datetime.now(tz=UTC).isoformat()
        cur = self._conn.execute(
            """
            UPDATE message_reactions
               SET count = count + ?, updated_at = ?
             WHERE familiar_id = ?
               AND platform_message_id = ?
               AND emoji = ?
            """,
            (delta, ts, familiar_id, platform_message_id, emoji),
        )
        if cur.rowcount == 0 and delta > 0:
            self._conn.execute(
                """
                INSERT INTO message_reactions
                    (familiar_id, platform_message_id, emoji, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (familiar_id, platform_message_id, emoji, delta, ts),
            )
        self._conn.execute(
            """
            DELETE FROM message_reactions
             WHERE familiar_id = ?
               AND platform_message_id = ?
               AND emoji = ?
               AND count <= 0
            """,
            (familiar_id, platform_message_id, emoji),
        )
        self._conn.commit()

    def clear_reactions(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
        emoji: str | None = None,
    ) -> None:
        """Drop reactions on one message — all (``emoji=None``) or a single emoji.

        Mirrors ``on_raw_reaction_clear`` (no emoji) and
        ``on_raw_reaction_clear_emoji`` (single emoji wiped).
        """
        if emoji is None:
            self._conn.execute(
                """
                DELETE FROM message_reactions
                 WHERE familiar_id = ?
                   AND platform_message_id = ?
                """,
                (familiar_id, platform_message_id),
            )
        else:
            self._conn.execute(
                """
                DELETE FROM message_reactions
                 WHERE familiar_id = ?
                   AND platform_message_id = ?
                   AND emoji = ?
                """,
                (familiar_id, platform_message_id, emoji),
            )
        self._conn.commit()

    def reactions_for_messages(
        self,
        *,
        familiar_id: str,
        platform_message_ids: Iterable[str],
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        """Batch lookup reactions for many messages in one query.

        Returns ``{platform_message_id: ((emoji, count), ...)}`` —
        messages with no reactions are absent. Per-message tuples are
        ordered by descending count, then emoji asc for stable ties.
        """
        ids = [str(m) for m in platform_message_ids if m]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT platform_message_id, emoji, count
              FROM message_reactions
             WHERE familiar_id = ?
               AND platform_message_id IN ({placeholders})
             ORDER BY platform_message_id ASC, count DESC, emoji ASC
            """,  # noqa: S608
            (familiar_id, *ids),
        ).fetchall()
        out: dict[str, list[tuple[str, int]]] = {}
        for r in rows:
            out.setdefault(str(r["platform_message_id"]), []).append((
                str(r["emoji"]),
                int(r["count"]),
            ))
        return {k: tuple(v) for k, v in out.items()}

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

    # ------------------------------------------------------------------
    # accounts + per-guild nicknames
    # ------------------------------------------------------------------

    def upsert_account(self, author: Author) -> None:
        """Insert or refresh the canonical identity row for an Author.

        Last-write wins on ``username`` / ``global_name`` / ``pronouns``
        / ``bio``; ``last_seen_at`` always stamps now. ``canonical_key``
        is the primary key, so re-upserting an existing user is cheap.
        Profile fields (pronouns, bio) only overwrite when the new
        value is non-NULL — bot tokens often can't read them, so a
        later read shouldn't clobber a richer earlier observation.
        Does not touch per-guild nicks — see :meth:`upsert_guild_nick`.
        """
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO accounts
                (canonical_key, platform, user_id, username, global_name,
                 pronouns, bio, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (canonical_key) DO UPDATE SET
                username     = excluded.username,
                global_name  = excluded.global_name,
                pronouns     = COALESCE(excluded.pronouns, accounts.pronouns),
                bio          = COALESCE(excluded.bio, accounts.bio),
                last_seen_at = excluded.last_seen_at
            """,
            (
                author.canonical_key,
                author.platform,
                author.user_id,
                author.username,
                author.global_name,
                author.pronouns,
                author.bio,
                ts,
            ),
        )
        self._conn.commit()

    def get_account_profile(self, *, canonical_key: str) -> AccountProfile | None:
        """Return cached profile fields for ``canonical_key``.

        Powers the per-person header in :class:`PeopleDossierLayer` —
        cheap lookup against the ``accounts`` table. ``None`` when no
        row exists; missing columns surface as ``None`` on the result.
        """
        row = self._conn.execute(
            """
            SELECT username, global_name, pronouns, bio
              FROM accounts
             WHERE canonical_key = ?
            """,
            (canonical_key,),
        ).fetchone()
        if row is None:
            return None
        return AccountProfile(
            canonical_key=canonical_key,
            username=row["username"],
            global_name=row["global_name"],
            pronouns=row["pronouns"],
            bio=row["bio"],
        )

    def upsert_guild_nick(
        self,
        *,
        canonical_key: str,
        guild_id: int,
        nick: str | None,
    ) -> None:
        """Cache a per-guild nickname. ``nick=None`` records "no override".

        Per-guild row is keyed by ``(canonical_key, guild_id)``, so a
        user with distinct nicks per guild gets distinct rows. NULL
        ``nick`` is meaningful: it says "we observed this user in
        this guild and they had no nickname override" — distinct from
        "we've never seen them in this guild" (no row at all).
        """
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO account_guild_nicks
                (canonical_key, guild_id, nick, last_seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (canonical_key, guild_id) DO UPDATE SET
                nick         = excluded.nick,
                last_seen_at = excluded.last_seen_at
            """,
            (canonical_key, guild_id, nick, ts),
        )
        self._conn.commit()

    def resolve_label(
        self,
        *,
        canonical_key: str,
        guild_id: int | None,
        familiar_id: str | None = None,
    ) -> str:
        """Return the best display name for ``canonical_key`` in *guild_id*.

        Preference order:
        1. ``account_guild_nicks.nick`` for ``(canonical_key, guild_id)``
        2. ``accounts.global_name``
        3. ``accounts.username``
        4. ``latest_author_for(familiar_id, canonical_key).label`` —
           snapshot from the most recent turn. Useful for legacy rows
           where no ``accounts`` upsert has happened. Skipped when
           *familiar_id* is omitted.
        5. bare ``user_id`` parsed from canonical_key

        Always returns a non-empty string — unknown keys still produce
        ``"<user_id>"`` so callers don't have to handle ``None``.
        """
        if guild_id is not None:
            row = self._conn.execute(
                """
                SELECT nick FROM account_guild_nicks
                 WHERE canonical_key = ? AND guild_id = ?
                """,
                (canonical_key, guild_id),
            ).fetchone()
            if row is not None and row["nick"]:
                return str(row["nick"])
        row = self._conn.execute(
            """
            SELECT global_name, username FROM accounts
             WHERE canonical_key = ?
            """,
            (canonical_key,),
        ).fetchone()
        if row is not None:
            if row["global_name"]:
                return str(row["global_name"])
            if row["username"]:
                return str(row["username"])
        # Snapshot fallback: the latest turn carries an Author whose
        # display_name was correct at the moment of writing. Cheaper
        # than a join, and a sensible "last we saw them" answer for
        # legacy rows that pre-date the accounts table.
        if familiar_id is not None:
            snapshot = self.latest_author_for(
                familiar_id=familiar_id, canonical_key=canonical_key
            )
            if snapshot is not None:
                return snapshot.label
        # Fall back to the user_id portion of the canonical_key.
        if ":" in canonical_key:
            return canonical_key.partition(":")[2] or canonical_key
        return canonical_key

    # ------------------------------------------------------------------
    # latest_author_for (legacy shim)
    # ------------------------------------------------------------------

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

    def all_channel_ids(self, *, familiar_id: str) -> set[int]:
        """Return the set of all channel ids that have turns for *familiar_id*."""
        rows = self._conn.execute(
            "SELECT DISTINCT channel_id FROM turns WHERE familiar_id = ?",
            (familiar_id,),
        ).fetchall()
        return {int(r["channel_id"]) for r in rows}

    def turns_in_id_range(
        self,
        *,
        familiar_id: str,
        min_id_exclusive: int,
        max_id_inclusive: int,
        channel_id: int | None = None,
    ) -> list[HistoryTurn]:
        """Return turns whose id falls in ``(min_id_exclusive, max_id_inclusive]``.

        When *channel_id* is given, restricts to that channel.
        """
        if channel_id is not None:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ?
                   AND channel_id = ?
                   AND id > ?
                   AND id <= ?
                 ORDER BY id ASC
                """,  # noqa: S608
                (familiar_id, channel_id, min_id_exclusive, max_id_inclusive),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""
                SELECT {_TURN_COLS}
                  FROM turns
                 WHERE familiar_id = ?
                   AND id > ?
                   AND id <= ?
                 ORDER BY id ASC
                """,  # noqa: S608
                (familiar_id, min_id_exclusive, max_id_inclusive),
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def all_fact_ids(self, *, familiar_id: str) -> set[int]:
        """Return all fact ids for *familiar_id*, including superseded ones."""
        rows = self._conn.execute(
            "SELECT id FROM facts WHERE familiar_id = ?",
            (familiar_id,),
        ).fetchall()
        return {int(r["id"]) for r in rows}

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
    # people_dossiers — per-person summaries compounded from facts
    # ------------------------------------------------------------------

    def get_people_dossier(
        self,
        *,
        familiar_id: str,
        canonical_key: str,
    ) -> PeopleDossierEntry | None:
        """Return the cached dossier for ``canonical_key``, or ``None``."""
        row = self._conn.execute(
            """
            SELECT canonical_key, last_fact_id, dossier_text, created_at
              FROM people_dossiers
             WHERE familiar_id = ? AND canonical_key = ?
            """,
            (familiar_id, canonical_key),
        ).fetchone()
        if row is None:
            return None
        return PeopleDossierEntry(
            canonical_key=str(row["canonical_key"]),
            last_fact_id=int(row["last_fact_id"]),
            dossier_text=str(row["dossier_text"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put_people_dossier(
        self,
        *,
        familiar_id: str,
        canonical_key: str,
        last_fact_id: int,
        dossier_text: str,
    ) -> None:
        """Insert or replace the dossier for ``canonical_key``."""
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO people_dossiers
                (familiar_id, canonical_key,
                 last_fact_id, dossier_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (familiar_id, canonical_key)
            DO UPDATE SET
                last_fact_id = excluded.last_fact_id,
                dossier_text = excluded.dossier_text,
                created_at   = excluded.created_at
            """,
            (familiar_id, canonical_key, last_fact_id, dossier_text, ts),
        )
        self._conn.commit()

    def subjects_with_facts(self, *, familiar_id: str) -> dict[str, int]:
        """Map ``canonical_key`` → ``max(facts.id)`` across current facts.

        Excludes superseded facts — the dossier should track current
        truth, and a subject whose only facts are stale shouldn't keep
        showing up as a refresh candidate. Scans ``subjects_json`` in
        Python; fine at expected per-familiar volumes (a SQLite virtual
        index would be a later optimisation if profiling demands it).
        """
        rows = self._conn.execute(
            """
            SELECT id, subjects_json
              FROM facts
             WHERE familiar_id = ?
               AND subjects_json IS NOT NULL
               AND superseded_at IS NULL
             ORDER BY id ASC
            """,
            (familiar_id,),
        ).fetchall()
        out: dict[str, int] = {}
        for row in rows:
            try:
                parsed = json.loads(row["subjects_json"])
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, list):
                continue
            fact_id = int(row["id"])
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                key = item.get("canonical_key")
                if not isinstance(key, str):
                    continue
                # ORDER BY id ASC ⇒ later assignment wins ⇒ max id.
                out[key] = fact_id
        return out

    def facts_for_subject(
        self,
        *,
        familiar_id: str,
        canonical_key: str,
        min_id_exclusive: int = 0,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[Fact]:
        """Return facts mentioning ``canonical_key``, ASC by id.

        Pre-filters with ``subjects_json LIKE`` (cheap; the JSON form
        wraps each key in quotes so substring collisions like
        ``discord:1`` vs ``discord:11`` don't false-positive). Final
        membership check parses the JSON in Python. ``as_of`` mirrors
        :meth:`recent_facts` semantics.
        """
        where, params = _facts_validity_where(
            include_superseded=include_superseded, as_of=as_of
        )
        like_pattern = f'%"{canonical_key}"%'
        rows = self._conn.execute(
            f"""
            SELECT id, familiar_id, channel_id, text,
                   source_turn_ids, created_at,
                   superseded_at, superseded_by, subjects_json,
                   valid_from, valid_to, importance
              FROM facts
             WHERE familiar_id = ?
               AND id > ?
               AND subjects_json IS NOT NULL
               AND subjects_json LIKE ?
               {where}
             ORDER BY id ASC
            """,  # noqa: S608
            (familiar_id, min_id_exclusive, like_pattern, *params),
        ).fetchall()
        out: list[Fact] = []
        for row in rows:
            fact = _row_to_fact(row)
            if any(s.canonical_key == canonical_key for s in fact.subjects):
                out.append(fact)
        return out

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
        stopwords return ``[]``. Tantivy's English analyzer
        (lowercase + ascii_fold + stopwords + english stemmer) handles
        tokenisation; the default disjunctive parse ORs the substantive
        terms together so chat-style cues still rank by BM25 on the
        nouns that hit.

        :param max_id: if set, only turns with ``id <= max_id`` are
            considered. Used by :class:`RagContextLayer` to keep RAG
            from re-surfacing turns already covered by
            :class:`RecentHistoryLayer`.
        """
        if limit <= 0:
            return []
        # Overfetch from FTS so post-filter (familiar/channel/max_id)
        # doesn't starve the result. Cap at 10x to bound work.
        fts_limit = max(limit * 4, limit)
        hits = self._fts_turns.search(query, limit=fts_limit)
        if not hits:
            return []
        score_by_id = dict(hits)
        candidate_ids = list(score_by_id)

        params: list[object] = [familiar_id, *candidate_ids]
        placeholders = ",".join("?" for _ in candidate_ids)
        where_extra = ""
        if channel_id is not None:
            where_extra += "AND t.channel_id = ?\n"
            params.append(channel_id)
        if max_id is not None:
            where_extra += "AND t.id <= ?\n"
            params.append(max_id)
        rows = self._conn.execute(
            f"""
            SELECT {", ".join("t." + c for c in _TURN_COLS.split(", "))}
              FROM turns AS t
             WHERE t.familiar_id = ?
               AND t.id IN ({placeholders})
               {where_extra}
            """,  # noqa: S608
            params,
        ).fetchall()
        turns = [_row_to_turn(r) for r in rows]
        # Re-rank by BM25 desc (higher = better in tantivy), tie-break by
        # newer-first to match the old ``ORDER BY ..., t.id DESC``.
        turns.sort(key=lambda t: (-score_by_id.get(t.id, 0.0), -t.id))
        return turns[:limit]

    def rebuild_fts(self) -> None:
        """Drop and repopulate the tantivy turns index from ``turns``.

        Cheap relative to re-running every LLM call; cheap enough to
        run at startup if the index ever gets out of sync.
        """
        self._fts_turns.clear()
        rows = self._conn.execute(
            "SELECT id, content FROM turns ORDER BY id ASC"
        ).fetchall()
        self._fts_turns.add_many([(int(r["id"]), str(r["content"])) for r in rows])

    def latest_fts_id(self, *, familiar_id: str) -> int:
        """Return the highest turn id currently indexed for ``familiar_id``.

        The tantivy index is updated synchronously with each
        :meth:`append_turn`, so the highest indexed id equals the
        highest ``turns.id`` for the familiar. Cheap MAX query rather
        than a tantivy round trip.
        """
        row = self._conn.execute(
            "SELECT MAX(id) AS max_id FROM turns WHERE familiar_id = ?",
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
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        importance: int | None = None,
    ) -> Fact:
        """Persist one fact. ``source_turn_ids`` and ``subjects`` stored as JSON.

        ``subjects`` is the extractor's best-effort link to canonical
        identities — see :class:`FactSubject`.

        ``valid_from`` / ``valid_to`` are world-time (when the fact
        applied in the world). When ``valid_from`` is omitted it
        defaults to ``created_at``; callers (e.g. ``FactExtractor``)
        pass the source turn's timestamp explicitly. ``valid_to``
        defaults to ``None`` — fact still applies.

        ``importance`` is the extractor's 1-10 ranking hint (M2).
        Out-of-range values clamp to ``[1, 10]`` so a stray LLM number
        can't poison rank-time math. ``None`` is preserved verbatim —
        downstream consumers treat it as a neutral midpoint.
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
        valid_from_eff = valid_from if valid_from is not None else ts
        importance_eff: int | None
        if importance is None:
            importance_eff = None
        else:
            importance_eff = max(1, min(10, int(importance)))
        cur = self._conn.execute(
            """
            INSERT INTO facts (familiar_id, channel_id, text,
                               source_turn_ids, created_at, subjects_json,
                               valid_from, valid_to, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                channel_id,
                text,
                json.dumps(ids),
                ts.isoformat(),
                subjects_blob,
                valid_from_eff.isoformat(),
                valid_to.isoformat() if valid_to is not None else None,
                importance_eff,
            ),
        )
        self._conn.commit()
        fact_id = int(cur.lastrowid or 0)
        self._fts_facts.add(fact_id, text)
        return Fact(
            id=fact_id,
            familiar_id=familiar_id,
            channel_id=channel_id,
            text=text,
            source_turn_ids=tuple(ids),
            created_at=ts,
            subjects=subjects_tuple,
            valid_from=valid_from_eff,
            valid_to=valid_to,
            importance=importance_eff,
        )

    def recent_facts(
        self,
        *,
        familiar_id: str,
        limit: int,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[Fact]:
        """Return the ``limit`` most recent facts, newest first.

        Default ("current truth"): excludes superseded facts and any
        whose world-time ``valid_to`` is in the past.

        ``as_of`` switches to a bi-temporal world-time slice — returns
        facts whose ``valid_from <= as_of`` and (``valid_to`` is NULL
        or > ``as_of``). Includes superseded rows so audit queries can
        recover prior beliefs (overrides ``include_superseded``).
        """
        if limit <= 0:
            return []
        where, params = _facts_validity_where(
            include_superseded=include_superseded, as_of=as_of
        )
        rows = self._conn.execute(
            f"""
            SELECT id, familiar_id, channel_id, text,
                   source_turn_ids, created_at,
                   superseded_at, superseded_by, subjects_json,
                   valid_from, valid_to, importance
              FROM facts
             WHERE familiar_id = ?
               {where}
             ORDER BY id DESC
             LIMIT ?
            """,  # noqa: S608
            (familiar_id, *params, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def _fact_candidates_by_fts(
        self,
        *,
        familiar_id: str,
        query: str,
        limit: int,
        include_superseded: bool,
        as_of: datetime | None,
    ) -> list[tuple[Fact, float]]:
        """Shared FTS lookup for fact search methods.

        Runs the tantivy query, joins back to ``facts`` with the
        validity filter, and re-ranks by BM25 desc then id desc.
        Returns ``[(fact, score)]`` truncated to *limit*.
        """
        if limit <= 0:
            return []
        # Overfetch from FTS so validity/familiar filters don't starve
        # the result. 4x is enough in practice (validity is cheap).
        fts_limit = max(limit * 4, limit)
        hits = self._fts_facts.search(query, limit=fts_limit)
        if not hits:
            return []
        score_by_id = dict(hits)
        candidate_ids = list(score_by_id)
        placeholders = ",".join("?" for _ in candidate_ids)
        where, params = _facts_validity_where(
            include_superseded=include_superseded, as_of=as_of, alias="f"
        )
        rows = self._conn.execute(
            f"""
            SELECT f.id, f.familiar_id, f.channel_id, f.text,
                   f.source_turn_ids, f.created_at,
                   f.superseded_at, f.superseded_by, f.subjects_json,
                   f.valid_from, f.valid_to, f.importance
              FROM facts AS f
             WHERE f.familiar_id = ?
               AND f.id IN ({placeholders})
               {where}
            """,  # noqa: S608
            (familiar_id, *candidate_ids, *params),
        ).fetchall()
        scored = [(_row_to_fact(r), score_by_id.get(int(r["id"]), 0.0)) for r in rows]
        scored.sort(key=lambda pair: (-pair[1], -pair[0].id))
        return scored[:limit]

    def search_facts(
        self,
        *,
        familiar_id: str,
        query: str,
        limit: int,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[Fact]:
        """FTS search over ``facts.text``.

        See :meth:`search_turns` for tokenisation notes. Validity
        filtering matches :meth:`recent_facts`: default = current
        truth (not superseded, not expired); ``as_of`` switches to a
        bi-temporal world-time slice including superseded rows.
        """
        return [
            fact
            for fact, _ in self._fact_candidates_by_fts(
                familiar_id=familiar_id,
                query=query,
                limit=limit,
                include_superseded=include_superseded,
                as_of=as_of,
            )
        ]

    def search_facts_scored(
        self,
        *,
        familiar_id: str,
        query: str,
        limit: int,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[tuple[Fact, float]]:
        """Like :meth:`search_facts`, but pairs each row with its BM25 score.

        Tantivy's BM25 is positive (higher = better). Callers fusing
        with other signals (importance, recency, embedding similarity)
        should treat the score as a non-negative weight; the prior
        SQLite FTS5 ``bm25()`` returned negative numbers (lower =
        better), so consumers may have an inverted-sign assumption to
        revisit.
        """
        return self._fact_candidates_by_fts(
            familiar_id=familiar_id,
            query=query,
            limit=limit,
            include_superseded=include_superseded,
            as_of=as_of,
        )

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

    # ------------------------------------------------------------------
    # fact embeddings (M6) — semantic recall side-index
    # ------------------------------------------------------------------

    def set_fact_embedding(
        self,
        *,
        fact_id: int,
        model: str,
        vector: list[float],
    ) -> None:
        """Persist *vector* for ``(fact_id, model)``; upsert.

        Stored as packed little-endian float32. ``model`` is the
        embedder's :attr:`Embedder.name`; pairing it with ``fact_id``
        lets a model swap accumulate new rows beside the old without
        destroying audit history.
        """
        if not vector:
            msg = "set_fact_embedding requires a non-empty vector"
            raise ValueError(msg)
        dim = len(vector)
        blob = struct.pack(f"<{dim}f", *vector)
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO fact_embeddings (fact_id, model, dim, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (fact_id, model)
            DO UPDATE SET
                dim        = excluded.dim,
                vector     = excluded.vector,
                created_at = excluded.created_at
            """,
            (int(fact_id), model, dim, blob, ts),
        )
        self._conn.commit()

    def get_fact_embeddings(
        self,
        *,
        fact_ids: Iterable[int],
        model: str,
    ) -> dict[int, list[float]]:
        """Return ``{fact_id: vector}`` for the requested ids + model.

        Missing rows are simply absent from the result — the caller
        treats them as "not yet embedded" and skips the embedding
        signal for that candidate.
        """
        ids = [int(i) for i in fact_ids]
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT fact_id, dim, vector
              FROM fact_embeddings
             WHERE model = ? AND fact_id IN ({placeholders})
            """,  # noqa: S608
            (model, *ids),
        ).fetchall()
        out: dict[int, list[float]] = {}
        for r in rows:
            dim = int(r["dim"])
            blob = bytes(r["vector"])
            out[int(r["fact_id"])] = list(struct.unpack(f"<{dim}f", blob))
        return out

    def unembedded_facts(
        self,
        *,
        familiar_id: str,
        model: str,
        limit: int,
    ) -> list[Fact]:
        """Return current facts lacking an embedding row for ``model``.

        "Current" matches :meth:`recent_facts` defaults — superseded
        rows are excluded. The projector embeds in id order so an
        interrupted run resumes deterministically.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT f.id, f.familiar_id, f.channel_id, f.text,
                   f.source_turn_ids, f.created_at,
                   f.superseded_at, f.superseded_by, f.subjects_json,
                   f.valid_from, f.valid_to, f.importance
              FROM facts AS f
              LEFT JOIN fact_embeddings AS fe
                ON fe.fact_id = f.id AND fe.model = ?
             WHERE f.familiar_id = ?
               AND f.superseded_at IS NULL
               AND fe.fact_id IS NULL
             ORDER BY f.id ASC
             LIMIT ?
            """,
            (model, familiar_id, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def latest_embedded_fact_id(
        self,
        *,
        familiar_id: str,
        model: str,
    ) -> int:
        """Highest ``fact_id`` with an embedding row for ``model``; 0 if none."""
        row = self._conn.execute(
            """
            SELECT MAX(fe.fact_id) AS max_id
              FROM fact_embeddings AS fe
              JOIN facts AS f ON f.id = fe.fact_id
             WHERE fe.model = ? AND f.familiar_id = ?
            """,
            (model, familiar_id),
        ).fetchone()
        max_id = row["max_id"] if row is not None else None
        return int(max_id) if max_id is not None else 0

    # ------------------------------------------------------------------
    # reflections (M3) — higher-order syntheses
    # ------------------------------------------------------------------

    def append_reflection(
        self,
        *,
        familiar_id: str,
        channel_id: int | None,
        text: str,
        cited_turn_ids: Iterable[int],
        cited_fact_ids: Iterable[int],
        last_turn_id: int,
        last_fact_id: int,
    ) -> Reflection:
        """Insert a new reflection row.

        ``last_turn_id`` / ``last_fact_id`` snapshot the worker's view
        at write time — also serve as the next tick's watermark.
        """
        turn_ids = [int(i) for i in cited_turn_ids]
        fact_ids = [int(i) for i in cited_fact_ids]
        ts = datetime.now(tz=UTC)
        cur = self._conn.execute(
            """
            INSERT INTO reflections
                (familiar_id, channel_id, text,
                 cited_turn_ids, cited_fact_ids, created_at,
                 last_turn_id, last_fact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                channel_id,
                text,
                json.dumps(turn_ids),
                json.dumps(fact_ids),
                ts.isoformat(),
                int(last_turn_id),
                int(last_fact_id),
            ),
        )
        self._conn.commit()
        return Reflection(
            id=int(cur.lastrowid or 0),
            familiar_id=familiar_id,
            channel_id=channel_id,
            text=text,
            cited_turn_ids=tuple(turn_ids),
            cited_fact_ids=tuple(fact_ids),
            created_at=ts,
            last_turn_id=int(last_turn_id),
            last_fact_id=int(last_fact_id),
        )

    def recent_reflections(
        self,
        *,
        familiar_id: str,
        channel_id: int | None = None,
        limit: int,
    ) -> list[Reflection]:
        """Return ``limit`` most recent reflections, newest first.

        ``channel_id`` scopes to one channel; ``None`` returns reflections
        regardless of channel scope (including channel-agnostic rows).
        """
        if limit <= 0:
            return []
        if channel_id is None:
            rows = self._conn.execute(
                """
                SELECT id, familiar_id, channel_id, text,
                       cited_turn_ids, cited_fact_ids, created_at,
                       last_turn_id, last_fact_id
                  FROM reflections
                 WHERE familiar_id = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (familiar_id, limit),
            ).fetchall()
        else:
            # Include channel-agnostic rows (channel_id IS NULL) so a
            # global reflection still surfaces in any channel.
            rows = self._conn.execute(
                """
                SELECT id, familiar_id, channel_id, text,
                       cited_turn_ids, cited_fact_ids, created_at,
                       last_turn_id, last_fact_id
                  FROM reflections
                 WHERE familiar_id = ?
                   AND (channel_id = ? OR channel_id IS NULL)
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (familiar_id, channel_id, limit),
            ).fetchall()
        return [_row_to_reflection(r) for r in rows]

    def latest_reflection_watermarks(
        self,
        *,
        familiar_id: str,
    ) -> tuple[int, int]:
        """Return (last_turn_id, last_fact_id) of the newest reflection.

        ``(0, 0)`` if no reflections exist for *familiar_id*. Used by
        :class:`ReflectionWorker` to decide whether enough new turns
        / facts have accumulated to write again.
        """
        row = self._conn.execute(
            """
            SELECT last_turn_id, last_fact_id
              FROM reflections
             WHERE familiar_id = ?
             ORDER BY id DESC
             LIMIT 1
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return (0, 0)
        return (int(row["last_turn_id"]), int(row["last_fact_id"]))

    def superseded_fact_ids(
        self,
        *,
        familiar_id: str,
        fact_ids: Iterable[int],
    ) -> set[int]:
        """Return the subset of ``fact_ids`` that are superseded.

        Used by :class:`ReflectionLayer` to flag stale citations on
        read. Empty input returns ``set()`` without a query.
        """
        ids = [int(i) for i in fact_ids]
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT id
              FROM facts
             WHERE familiar_id = ?
               AND id IN ({placeholders})
               AND superseded_at IS NOT NULL
            """,  # noqa: S608
            (familiar_id, *ids),
        ).fetchall()
        return {int(r["id"]) for r in rows}

    # ------------------------------------------------------------------
    # alarms
    # ------------------------------------------------------------------

    def insert_alarm(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        channel_kind: str,
        scheduled_at: str,
        reason: str,
        originating_turn_id: str | None = None,
    ) -> str:
        """Insert a new alarm row; return its id.

        ``scheduled_at`` is an ISO-8601 UTC timestamp. ``channel_kind``
        must be ``"text"`` or ``"voice"`` (enforced by CHECK constraint).
        """
        alarm_id = uuid.uuid4().hex
        created_at = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO alarms
                (id, familiar_id, channel_id, channel_kind,
                 scheduled_at, reason, originating_turn_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alarm_id,
                familiar_id,
                channel_id,
                channel_kind,
                scheduled_at,
                reason,
                originating_turn_id,
                created_at,
            ),
        )
        self._conn.commit()
        return alarm_id

    def list_pending_alarms(self, *, familiar_id: str) -> list[dict[str, Any]]:
        """Return pending alarms (not fired, not cancelled) for ``familiar_id``.

        Rows are dicts; ordered by ``scheduled_at`` ascending.
        """
        rows = self._conn.execute(
            """
            SELECT id, familiar_id, channel_id, channel_kind,
                   scheduled_at, reason, originating_turn_id,
                   fired_at, cancelled_at, created_at
              FROM alarms
             WHERE familiar_id = ?
               AND fired_at IS NULL
               AND cancelled_at IS NULL
             ORDER BY scheduled_at ASC
            """,
            (familiar_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_alarm_fired(self, *, alarm_id: str, fired_at: str) -> bool:
        """Stamp ``fired_at`` on one alarm; ``True`` if a row was updated."""
        cur = self._conn.execute(
            """
            UPDATE alarms
               SET fired_at = ?
             WHERE id = ?
               AND fired_at IS NULL
               AND cancelled_at IS NULL
            """,
            (fired_at, alarm_id),
        )
        self._conn.commit()
        return (cur.rowcount or 0) > 0

    def cancel_alarm(self, *, alarm_id: str, cancelled_at: str) -> bool:
        """Stamp ``cancelled_at`` on one alarm; ``True`` if a row was updated."""
        cur = self._conn.execute(
            """
            UPDATE alarms
               SET cancelled_at = ?
             WHERE id = ?
               AND fired_at IS NULL
               AND cancelled_at IS NULL
            """,
            (cancelled_at, alarm_id),
        )
        self._conn.commit()
        return (cur.rowcount or 0) > 0


def _row_to_reflection(row: Row) -> Reflection:
    try:
        turn_ids = tuple(int(x) for x in json.loads(row["cited_turn_ids"]))
    except (ValueError, TypeError):
        turn_ids = ()
    try:
        fact_ids = tuple(int(x) for x in json.loads(row["cited_fact_ids"]))
    except (ValueError, TypeError):
        fact_ids = ()
    channel = row["channel_id"]
    return Reflection(
        id=int(row["id"]),
        familiar_id=str(row["familiar_id"]),
        channel_id=int(channel) if channel is not None else None,
        text=str(row["text"]),
        cited_turn_ids=turn_ids,
        cited_fact_ids=fact_ids,
        created_at=datetime.fromisoformat(row["created_at"]),
        last_turn_id=int(row["last_turn_id"]),
        last_fact_id=int(row["last_fact_id"]),
    )


def _row_to_fact(row: Row) -> Fact:
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
    try:
        valid_from_raw = row["valid_from"]
    except (IndexError, KeyError):
        valid_from_raw = None
    try:
        valid_to_raw = row["valid_to"]
    except (IndexError, KeyError):
        valid_to_raw = None
    try:
        importance_raw = row["importance"]
    except (IndexError, KeyError):
        importance_raw = None
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
        valid_from=(
            datetime.fromisoformat(valid_from_raw)
            if valid_from_raw is not None
            else None
        ),
        valid_to=(
            datetime.fromisoformat(valid_to_raw) if valid_to_raw is not None else None
        ),
        importance=int(importance_raw) if importance_raw is not None else None,
    )


def _row_to_turn(row: Row) -> HistoryTurn:
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

    try:
        platform_message_id = row["platform_message_id"]
    except (IndexError, KeyError):
        platform_message_id = None
    try:
        reply_to_message_id = row["reply_to_message_id"]
    except (IndexError, KeyError):
        reply_to_message_id = None
    try:
        guild_id_raw = row["guild_id"]
    except (IndexError, KeyError):
        guild_id_raw = None
    return HistoryTurn(
        id=int(row["id"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        role=str(row["role"]),
        author=author,
        content=str(row["content"]),
        channel_id=channel_id,
        platform_message_id=(
            str(platform_message_id) if platform_message_id is not None else None
        ),
        reply_to_message_id=(
            str(reply_to_message_id) if reply_to_message_id is not None else None
        ),
        guild_id=int(guild_id_raw) if guild_id_raw is not None else None,
    )
