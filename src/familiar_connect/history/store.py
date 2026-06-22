"""Turso-backed persistent conversation history (FTS via tantivy).

One database per familiar. Two core tables:

- ``turns`` — every turn scoped by ``(familiar_id, channel_id)``;
  monotonic PK, role, optional :class:`Author`, content, optional
  guild_id, UTC ts
- ``summaries`` — at most one rolling summary per (familiar, channel)
  with ``last_summarised_id`` watermark for cache freshness.
  See ``docs/architecture/context-pipeline.md`` for rationale.

Relational storage uses Turso (SQLite-compatible Rust rewrite); FTS
lives in a sibling tantivy index under ``fts/turns/`` +
``fts/facts/`` since pyturso wheels don't ship FTS.

``familiar_id`` is explicit (not implicit) so tests can exercise
multiple familiars against one store. Sync API;
:class:`AsyncHistoryStore` wrapper dispatches calls to a thread pool
that funnels every Turso call onto one dedicated OS thread inside
:class:`TursoConnection`.
"""

from __future__ import annotations

import json
import logging
import struct
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.history.fts import FtsIndex
from familiar_connect.history.turso_compat import TursoConnection
from familiar_connect.identity import Author

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Result rows come back as ``turso.Row``; spell as ``Any`` to keep
# private ``_row_to_*`` helpers free of a third-party type alias
Row = Any

PathLike = str | Path


@dataclass(frozen=True)
class HistoryTurn:
    """Single persisted conversational turn.

    :param platform_message_id: native id from platform of origin
        (Discord snowflake, etc.). ``None`` for legacy rows or
        platform-less inputs.
    :param reply_to_message_id: parent ``platform_message_id`` when
        this turn is a reply (e.g. ``message.reference`` on Discord).
    :param guild_id: Discord guild scoping per-guild nicknames.
        ``None`` for DMs / non-Discord / legacy rows where column
        wasn't populated.
    """

    id: int
    timestamp: datetime
    role: str
    author: Author | None
    content: str
    channel_id: int = 0
    mode: str | None = None  # free-form tag, e.g. ACTIVITY_RETURN_MODE
    platform_message_id: str | None = None
    reply_to_message_id: str | None = None
    guild_id: int | None = None
    arrived_at: datetime | None = None  # Immutable ingest time; None on legacy rows
    consumed_at: datetime | None = None  # None = staged


@dataclass(frozen=True)
class ActivityRecord:
    """One activity row (append-only log).

    Active while ``actual_return_at`` is ``None``; ``status`` is
    ``"completed"`` | ``"cut_short"`` once finished.
    """

    id: int
    familiar_id: str
    type_id: str
    label: str
    started_at: datetime
    planned_return_at: datetime
    note: str | None
    status: str | None
    actual_return_at: datetime | None
    experience_text: str | None


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
    """Last turn id written to long-term memory by the memory writer."""

    last_written_id: int
    created_at: datetime


@dataclass(frozen=True)
class SleepWatermark:
    """Highest fact/turn ids the last sleep consolidation pass saw.

    Bounds the *turns* window the next pass covers (re-attribution
    context above ``last_turn_id``); a missed night widens it. The
    *facts* window is always the current fact base, not id-bounded —
    consolidation reasons over all live facts. ``last_fact_id`` is the
    high-water mark the watermark advances to, recording progress.
    Distinct from memory-writer/reflection watermarks — sleep runs on
    its own (nightly) cadence.
    """

    last_fact_id: int
    last_turn_id: int
    updated_at: datetime


@dataclass(frozen=True)
class FocusPointers:
    """Current text/voice channel focus for a familiar."""

    text_channel_id: int | None
    voice_channel_id: int | None
    updated_at: datetime


@dataclass(frozen=True)
class AccountProfile:
    """Read-side projection of ``accounts`` profile metadata.

    Populated on Discord ingestion via
    :meth:`HistoryStore.upsert_account`; consumed by
    :class:`PeopleDossierLayer` to surface basic identity (username,
    pronouns, bio) on the prompt header.
    """

    canonical_key: str
    username: str | None
    global_name: str | None
    pronouns: str | None
    bio: str | None


@dataclass(frozen=True)
class PeopleDossierEntry:
    """Cached per-person dossier compounded from facts mentioning ``canonical_key``.

    ``last_fact_id`` is a watermark over ``facts.id`` — worker
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

    Extractor's best guess at *who* a fact is about. Provisional —
    mic-sharing, relayed quotes, ambiguous mentions all break a
    clean 1:1 mapping. Stored to enable display-name resolution at
    read time without claiming authoritative subject identification.

    :param canonical_key: stable ``platform:user_id`` from
        :class:`~familiar_connect.identity.Author`.
    :param display_at_write: display name as seen by extractor when
        fact was authored. Substring anchor at read time when
        current display name differs.
    """

    canonical_key: str
    display_at_write: str


@dataclass(frozen=True)
class FactDraft:
    """Consolidated *content* for a merge, before the store mints it.

    The caller of :meth:`HistoryStore.supersede` supplies only the
    merged content — text, channel, and resolved subject displays.
    The store supplies the *lineage* (obsolete rows point at the
    minted fact) and *provenance* (``source_turn_ids`` = union of the
    obsolete rows'), so a draft deliberately carries no turn ids.
    """

    channel_id: int | None
    text: str
    subjects: tuple[FactSubject, ...] = ()


@dataclass(frozen=True)
class SupersedeResult:
    """Outcome of one :meth:`HistoryStore.supersede` call.

    :param minted: the freshly minted replacement when ``new_fact``
        was a :class:`FactDraft`; ``None`` for retire or when an
        already-minted fact/id was supplied (nothing new was minted).
    :param superseded: obsolete ids actually marked this call.
    :param skipped: ``(id, reason)`` for obsolete rows left untouched
        — e.g. a concurrent writer already retired them. A skip is
        tolerated, never fatal: the merge is not rolled back.
    """

    minted: Fact | None
    superseded: tuple[int, ...]
    skipped: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Reflection:
    """Higher-order synthesis over recent turns + facts (M3).

    Written by :class:`ReflectionWorker`, read by
    :class:`ReflectionLayer`. ``cited_turn_ids`` / ``cited_fact_ids``
    are forever-provenance — never edited, never trimmed. A
    reflection citing a superseded fact stays in the table; read
    path flags it stale rather than dropping (audit trail beats
    silent loss).

    :param last_turn_id: highest ``turns.id`` visible to worker at
        write time. Doubles as next tick's watermark — one row per
        write, not a separate table.
    :param last_fact_id: highest ``facts.id`` visible to worker at
        write time. Same role as ``last_turn_id`` for facts axis.
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
    :param superseded_at: system-time — when fact was retired, or
        ``None`` if still current. Supersession keeps the row (no
        delete) so prior state stays visible for audit.
    :param superseded_by: id of replacement fact, or ``None`` if
        still current.
    :param subjects: best-effort canonical-key annotations. Empty
        tuple for legacy rows or when extractor couldn't link a name
        to any participant.
    :param valid_from: world-time — when fact began applying.
        Default = source turn's timestamp; LLM may override when an
        explicit "as of …" phrase is detected. ``None`` only on
        legacy rows pre-M1.
    :param valid_to: world-time — when fact stopped applying;
        ``None`` while still in effect.
    :param importance: 1-10 hint for retrieval ranking (M2). ``None``
        on legacy rows or when extractor declined to score. Treated
        as neutral midpoint by rank-time consumers.
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

-- Reflection watermark. Advanced every tick (even when the LLM
-- returns "[]" or all items are dropped) so a no-op tick can't pin
-- the worker to an ever-growing turn window. Without this the
-- reflection prompt balloons until the LLM bills the caller for the
-- entire chat history every 60s.
CREATE TABLE IF NOT EXISTS reflection_watermark (
    familiar_id   TEXT    PRIMARY KEY,
    last_turn_id  INTEGER NOT NULL,
    last_fact_id  INTEGER NOT NULL,
    updated_at    TEXT    NOT NULL
);

-- Sleep-consolidation watermark. Highest fact/turn ids the last
-- nightly hygiene pass saw. Window the next pass covers = ids above
-- these; a missed night just widens it. One row per familiar.
CREATE TABLE IF NOT EXISTS sleep_watermark (
    familiar_id   TEXT    PRIMARY KEY,
    last_fact_id  INTEGER NOT NULL,
    last_turn_id  INTEGER NOT NULL,
    updated_at    TEXT    NOT NULL
);

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

-- Attentional stream. Current text/voice focus for each familiar.
CREATE TABLE IF NOT EXISTS focus_pointers (
    familiar_id      TEXT PRIMARY KEY,
    text_channel_id  INTEGER,
    voice_channel_id INTEGER,
    updated_at       TEXT NOT NULL
);

-- Unread digest high-water mark. max(arrived_at) of last-delivered digest.
CREATE TABLE IF NOT EXISTS unread_digest_watermark (
    familiar_id    TEXT PRIMARY KEY,
    watermark_at   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- Activities archive watermark. Per (familiar, channel) turn id set
-- at the departure turn when an absence crosses the archive
-- threshold; ``recent_cross_channel(respect_archive=True)`` hides
-- ids at/below it.
CREATE TABLE IF NOT EXISTS channel_archive_watermark (
    familiar_id  TEXT    NOT NULL,
    channel_id   INTEGER NOT NULL,
    turn_id      INTEGER NOT NULL,
    updated_at   TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, channel_id)
);

-- Activities (append-only, restart-safe). Active row has
-- ``actual_return_at`` NULL; finishing stamps status
-- ('completed' | 'cut_short'), return time, experience text.
CREATE TABLE IF NOT EXISTS activities (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id        TEXT    NOT NULL,
    type_id            TEXT    NOT NULL,
    label              TEXT    NOT NULL,
    started_at         TEXT    NOT NULL,
    planned_return_at  TEXT    NOT NULL,
    note               TEXT,
    status             TEXT    CHECK(status IN ('completed','cut_short')),
    actual_return_at   TEXT,
    experience_text    TEXT
);

CREATE INDEX IF NOT EXISTS idx_activities_active
    ON activities (familiar_id, actual_return_at, id);

"""

_TURN_COLS = (
    "id, timestamp, role, author_platform, author_user_id, "
    "author_username, author_display_name, content, channel_id, "
    "mode, platform_message_id, reply_to_message_id, guild_id, "
    "arrived_at, consumed_at"
)


def _normalize_fact_text(text: str) -> str:
    """Deterministic key for near-duplicate fact detection.

    Lowercase, collapse whitespace, remove quote chars everywhere,
    strip surrounding punctuation. Exact-match-after-normalization
    only — no semantic similarity (kills the "called 'Cor'"
    restatement class without risking false-positive suppression of
    genuinely distinct facts). Internal non-quote punct kept (stripping
    it risks false positives).
    """
    collapsed = " ".join(text.lower().split())
    dequoted = collapsed.replace("'", "").replace('"', "")
    return dequoted.strip(".,!?;:()[]{} \t\n")


def _subject_key_set(subjects: Iterable[FactSubject]) -> frozenset[str]:
    """Canonical-key set identifying a fact's subjects.

    Dups scoped per subject set: a NULL-subject fact and a keyed one
    with same text are NOT dups.
    """
    return frozenset(s.canonical_key for s in subjects)


def _union_provenance(facts: Iterable[Fact | None]) -> list[int]:
    """Order-preserving union of ``source_turn_ids`` across ``facts``.

    A merged fact's provenance is the union of the rows it consolidates
    — forever-provenance, no row dropped. ``None`` entries (an obsolete
    id with no live row) contribute nothing.
    """
    out: list[int] = []
    for fact in facts:
        if fact is None:
            continue
        for tid in fact.source_turn_ids:
            if tid not in out:
                out.append(tid)
    return out


def _canonical_keys_from_subjects_json(subjects_json: str | None) -> frozenset[str]:
    """Canonical str keys from a fact row's ``subjects_json``.

    Empty on NULL / unparseable / non-list / no valid keys. Tolerant
    of malformed items: skips non-dict entries and non-``str`` keys.
    """
    if not subjects_json:
        return frozenset()
    try:
        parsed = json.loads(subjects_json)
    except (TypeError, ValueError):
        return frozenset()
    if not isinstance(parsed, list):
        return frozenset()
    keys: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = item.get("canonical_key")
        if isinstance(key, str):
            keys.add(key)
    return frozenset(keys)


def _facts_validity_where(
    *,
    include_superseded: bool,
    as_of: datetime | None,
    alias: str = "",
) -> tuple[str, tuple[object, ...]]:
    """Build SQL fragment + params for ``facts`` validity filter.

    Default ("current truth"):
        ``superseded_at IS NULL AND (valid_to IS NULL OR valid_to > now)``

    With ``as_of``:
        bi-temporal slice — ``valid_from`` IS NULL or <= as_of;
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

    ``":memory:"`` for ephemeral in-process DB (tests).
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
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()
        self._fts_turns = FtsIndex(fts_turns_path)
        self._fts_facts = FtsIndex(fts_facts_path)

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Add arrived_at / consumed_at to turns if missing; backfill."""
        for col in ("arrived_at", "consumed_at"):
            try:
                self._conn.execute(f"ALTER TABLE turns ADD COLUMN {col} TEXT")  # noqa: S608,RUF100
                self._conn.commit()
            except Exception:  # noqa: BLE001, S110  # column already exists
                pass
        # Backfill: existing rows treat timestamp as arrived_at
        self._conn.execute(
            "UPDATE turns SET arrived_at = timestamp WHERE arrived_at IS NULL"
        )
        # Backfill: existing rows are considered already consumed
        self._conn.execute(
            "UPDATE turns SET consumed_at = timestamp WHERE consumed_at IS NULL"
        )
        self._conn.commit()
        # Create cross-channel ordering index after columns exist
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turns_consumed
                ON turns (familiar_id, consumed_at, arrived_at, id)
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # FTS write helper
    # ------------------------------------------------------------------

    def _safe_fts_add(
        self, index: FtsIndex, row_id: int, content: str, *, kind: str
    ) -> None:
        """Index ``content`` under ``row_id``; never raise.

        Tantivy retries transient Windows AV file locks internally; any
        error reaching here is either persistent or non-lock-shaped.
        SQL row already committed — losing one FTS doc is recoverable
        via :meth:`rebuild_fts`. Crashing the bot is not.
        """
        try:
            index.add(row_id, content)
        except ValueError as exc:
            _logger.warning(
                f"{ls.tag('FTS', ls.Y)} "
                f"{ls.kv('skip', kind, vc=ls.LY)} "
                f"{ls.kv('row_id', str(row_id), vc=ls.LY)} "
                f"{ls.kv('err', ls.trunc(str(exc), 160), vc=ls.LY)}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down executor, FTS writers, Turso connections."""
        self._executor.shutdown(wait=False)
        try:
            self._fts_turns.close()
        finally:
            try:
                self._fts_facts.close()
            finally:
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
        platform_message_id: str | None = None,
        reply_to_message_id: str | None = None,
        tool_calls_json: str | None = None,
        tool_call_id: str | None = None,
        arrived_at: datetime | None = None,
        consumed: bool = True,
    ) -> HistoryTurn:
        """Append a single turn; return persisted form.

        *mode* is a free-form string tag on ``turns.mode``.
        *platform_message_id* / *reply_to_message_id* are platform-
        native ids (Discord snowflakes, etc.) stored as TEXT so any
        platform's id format fits. *tool_calls_json* carries the
        assistant's invoked tool calls (JSON-encoded list);
        *tool_call_id* references the call a ``role=tool`` turn is
        answering. *arrived_at* defaults to now (UTC) when omitted.
        *consumed* = ``True`` sets ``consumed_at = arrived_at``; when
        ``False`` the turn is staged (``consumed_at`` remains NULL).
        """
        timestamp = datetime.now(tz=UTC)
        arrived_at_eff = arrived_at if arrived_at is not None else timestamp
        consumed_at_eff = arrived_at_eff if consumed else None
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
                 tool_calls_json, tool_call_id,
                 arrived_at, consumed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                arrived_at_eff.isoformat(),
                consumed_at_eff.isoformat() if consumed_at_eff is not None else None,
            ),
        )
        self._conn.commit()
        turn_id = int(cur.lastrowid or 0)
        self._safe_fts_add(self._fts_turns, turn_id, content, kind="turn")
        return HistoryTurn(
            id=turn_id,
            timestamp=timestamp,
            role=role,
            author=author,
            content=content,
            channel_id=channel_id,
            mode=mode,
            arrived_at=arrived_at_eff,
            consumed_at=consumed_at_eff,
        )

    def lookup_turn_by_platform_message_id(
        self,
        *,
        familiar_id: str,
        platform_message_id: str,
    ) -> HistoryTurn | None:
        """Find turn carrying ``platform_message_id`` for ``familiar_id``.

        ``None`` if no row matches — used by read path to resolve
        reply parents. Index ``idx_turns_platform_msg`` keeps this
        O(1) per familiar.
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
        no row matches — bot may have come up after the message
        landed. Refreshes tantivy index for the affected row.
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
            self._safe_fts_add(
                self._fts_turns, int(row["id"]), content, kind="turn_edit"
            )

    def turns_by_ids(
        self,
        *,
        familiar_id: str,
        ids: Iterable[int],
    ) -> list[HistoryTurn]:
        """Fetch turns by id, scoped to ``familiar_id``, oldest first.

        Used by RAG to expand each FTS hit into a small surrounding
        window (hit ± neighbours) without per-id round trips.
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
        """Record canonical keys mentioned in ``turn_id``.

        Idempotent — re-recording same keys is a no-op thanks to
        (turn_id, canonical_key) primary key. Empty input is a no-op
        too. Order not preserved; reads come back sorted by
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
        """Canonical keys mentioned in ``turn_id``, sorted."""
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
        where last user removing a reaction collapses the entry.
        Idempotent for repeated identical writes (gateway dedup is
        cheap insurance).
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

        Drives gateway hot path — ``on_raw_reaction_add`` /
        ``on_raw_reaction_remove`` deliver per-user toggles, never
        absolute counts. Floors at zero (stray remove without a
        matching add — e.g. bot was offline when reaction was added
        — leaves no row rather than persisting a negative).
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
        """Drop reactions on one message — all (``emoji=None``) or single emoji.

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
        """Batch reaction lookup for many messages in one query.

        Returns ``{platform_message_id: ((emoji, count), ...)}`` —
        messages with no reactions are absent. Per-message tuples
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
        before_id: int | None = None,
    ) -> list[HistoryTurn]:
        """Most recent turns in channel, oldest-first.

        Per-channel partitioning prevents bleed between conversations.
        When *mode* is set, only matching legacy-tag turns returned.
        *before_id* restricts to ``id < before_id`` (paging anchor).
        Archive watermark not applied here — prompt-window filtering
        lives in :meth:`recent_cross_channel`.
        """
        if limit <= 0:
            return []
        where_extra = ""
        params: list[object] = [familiar_id, channel_id]
        if mode is not None:
            where_extra += "AND mode = ?\n"
            params.append(mode)
        if before_id is not None:
            where_extra += "AND id < ?\n"
            params.append(before_id)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ? AND channel_id = ?
               {where_extra}
             ORDER BY id DESC
             LIMIT ?
            """,  # noqa: S608
            params,
        ).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    def turns_around(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        turn_id: int,
        before: int = 5,
        after: int = 5,
    ) -> list[HistoryTurn]:
        """Window of turns centred on anchor ``turn_id``, oldest first.

        Anchor included; *before* / *after* count turns each side
        within the channel partition. Clips at history edges.
        """
        before_rows = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ? AND channel_id = ? AND id < ?
             ORDER BY id DESC
             LIMIT ?
            """,  # noqa: S608
            (familiar_id, channel_id, turn_id, max(before, 0)),
        ).fetchall()
        anchor_after_rows = self._conn.execute(
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ? AND channel_id = ? AND id >= ?
             ORDER BY id ASC
             LIMIT ?
            """,  # noqa: S608
            (familiar_id, channel_id, turn_id, max(after, 0) + 1),
        ).fetchall()
        rows = [*reversed(before_rows), *anchor_after_rows]
        return [_row_to_turn(r) for r in rows]

    def recent_distinct_authors(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        limit: int,
    ) -> list[Author]:
        """Up to *limit* most-recently-seen distinct user authors.

        Most-recent-first by canonical key (platform + user_id).
        Skips turns without an author (assistant replies, system
        events). Scoped to one channel — matches :meth:`recent`.
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
        """Insert or refresh canonical identity row for an Author.

        Last-write wins on ``username`` / ``global_name`` /
        ``pronouns`` / ``bio``; ``last_seen_at`` always stamps now.
        ``canonical_key`` is the primary key, so re-upserting an
        existing user is cheap. Profile fields (pronouns, bio) only
        overwrite when new value is non-NULL — bot tokens often
        can't read them, so a later read shouldn't clobber a richer
        earlier observation. Does not touch per-guild nicks — see
        :meth:`upsert_guild_nick`.
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

        Powers per-person header in :class:`PeopleDossierLayer` —
        cheap lookup against ``accounts``. ``None`` when no row
        exists; missing columns surface as ``None`` on the result.
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
        """Cache per-guild nickname. ``nick=None`` records "no override".

        Per-guild row keyed by ``(canonical_key, guild_id)`` — a
        user with distinct nicks per guild gets distinct rows. NULL
        ``nick`` is meaningful: "observed this user in this guild,
        no nickname override" — distinct from "never seen them in
        this guild" (no row at all).
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
        """Best display name for ``canonical_key`` in *guild_id*.

        Preference order:
        1. ``account_guild_nicks.nick`` for ``(canonical_key, guild_id)``
        2. ``accounts.global_name``
        3. ``accounts.username``
        4. ``latest_author_for(familiar_id, canonical_key).label`` —
           snapshot from most recent turn. Useful for legacy rows
           where no ``accounts`` upsert has happened. Skipped when
           *familiar_id* omitted.
        5. bare ``user_id`` parsed from canonical_key

        Always non-empty string — unknown keys still produce
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
        # Snapshot fallback: latest turn carries an Author whose
        # display_name was correct at write time. cheaper than a
        # join; sensible "last we saw them" answer for legacy rows
        # pre-dating the accounts table.
        if familiar_id is not None:
            snapshot = self.latest_author_for(
                familiar_id=familiar_id, canonical_key=canonical_key
            )
            if snapshot is not None:
                return snapshot.label
        # Fall back to user_id portion of canonical_key
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
        """:class:`Author` from most recent turn with this key.

        Display names rotate (Discord/Twitch nicks); latest turn
        carries the freshest one. ``None`` if no turn matches — e.g.
        user hasn't spoken in this familiar, or canonical_key isn't
        well-formed. Used by :class:`RagContextLayer` to resolve
        stale fact-subject names at read time.
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
        """Highest turn id (watermark for cache freshness).

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
        """Count stored turns. *channel_id* scopes to one channel."""
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
        """Fetch cached summary for familiar + channel, or ``None``."""
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
        """Insert or replace summary for familiar + channel."""
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
        """Other channels with activity, most-recently-active first.

        Each row carries latest mode, turn id, timestamp.
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
        """All channel ids with turns for *familiar_id*."""
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

        *channel_id* restricts to that channel when given.
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
        """All fact ids for *familiar_id*, including superseded."""
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
        """Fetch cached cross-context summary, or ``None``."""
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
        """Memory-writer watermark for *familiar_id*, or ``None``."""
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
        """Insert or replace memory-writer watermark for *familiar_id*."""
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

    def get_sleep_watermark(
        self,
        *,
        familiar_id: str,
    ) -> SleepWatermark | None:
        """Last sleep-consolidation watermark for *familiar_id*, or ``None``."""
        row = self._conn.execute(
            """
            SELECT last_fact_id, last_turn_id, updated_at
              FROM sleep_watermark
             WHERE familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return None
        return SleepWatermark(
            last_fact_id=int(row["last_fact_id"]),
            last_turn_id=int(row["last_turn_id"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def advance_sleep_watermark(
        self,
        *,
        familiar_id: str,
        last_fact_id: int | None = None,
        last_turn_id: int | None = None,
    ) -> None:
        """Advance one or both watermark axes; leave omitted axes intact.

        Partial-update by design: hygiene owns ``last_fact_id``, dream
        owns ``last_turn_id``. Passing only one updates only that column,
        so neither pass can clobber the other's progress. On first insert
        an omitted axis defaults to 0. Both ``None`` is a no-op.
        """
        if last_fact_id is None and last_turn_id is None:
            return
        timestamp = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO sleep_watermark
                (familiar_id, last_fact_id, last_turn_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (familiar_id)
            DO UPDATE SET
                last_fact_id = COALESCE(?, last_fact_id),
                last_turn_id = COALESCE(?, last_turn_id),
                updated_at   = excluded.updated_at
            """,
            (
                familiar_id,
                last_fact_id if last_fact_id is not None else 0,
                last_turn_id if last_turn_id is not None else 0,
                timestamp,
                last_fact_id,
                last_turn_id,
            ),
        )
        self._conn.commit()

    def turns_since_watermark(
        self,
        *,
        familiar_id: str,
        limit: int = 10_000,
    ) -> list[HistoryTurn]:
        """Return turns after memory-writer watermark, oldest first.

        Returns all turns for the familiar when no watermark set.
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
        """Fetch cached dossier for ``canonical_key``, or ``None``."""
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
        """Insert or replace dossier for ``canonical_key``."""
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

        Excludes superseded facts — dossier tracks current truth; a
        subject whose only facts are stale shouldn't keep showing up
        as refresh candidate. Scans ``subjects_json`` in Python;
        fine at expected per-familiar volumes (a SQLite virtual
        index would be a later optimisation if profiling demands).
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
            fact_id = int(row["id"])
            for key in _canonical_keys_from_subjects_json(row["subjects_json"]):
                # ORDER BY id ASC ⇒ later assignment wins ⇒ max id
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
        """Facts mentioning ``canonical_key``, ASC by id.

        Pre-filters with ``subjects_json LIKE`` — cheap; JSON form
        wraps each key in quotes so substring collisions like
        ``discord:1`` vs ``discord:11`` don't false-positive. Final
        membership check parses JSON in Python. ``as_of`` mirrors
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
        """Search ``turns.content`` via FTS *query*.

        Empty/whitespace *query* and queries reducing to only
        stopwords return ``[]``. Tantivy's English analyzer
        (lowercase + ascii_fold + stopwords + english stemmer)
        handles tokenisation; default disjunctive parse ORs
        substantive terms so chat-style cues still rank by BM25 on
        nouns that hit.

        :param max_id: if set, only turns with ``id <= max_id``
            considered. Used by :class:`RagContextLayer` to keep RAG
            from re-surfacing turns already covered by
            :class:`RecentHistoryLayer`.
        """
        if limit <= 0:
            return []
        # Overfetch from FTS so post-filter (familiar/channel/max_id)
        # doesn't starve the result. cap at 10x to bound work
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
        # Re-rank by BM25 desc (higher = better in tantivy), tie-break
        # newer-first to match old ``ORDER BY ..., t.id DESC``
        turns.sort(key=lambda t: (-score_by_id.get(t.id, 0.0), -t.id))
        return turns[:limit]

    def rebuild_fts(self) -> None:
        """Drop and repopulate tantivy turns index from ``turns``.

        Cheap relative to re-running every LLM call; cheap enough to
        run at startup if index ever gets out of sync.
        """
        self._fts_turns.clear()
        rows = self._conn.execute(
            "SELECT id, content FROM turns ORDER BY id ASC"
        ).fetchall()
        self._fts_turns.add_many([(int(r["id"]), str(r["content"])) for r in rows])

    def latest_fts_id(self, *, familiar_id: str) -> int:
        """Highest turn id currently indexed for ``familiar_id``.

        Tantivy index updates synchronously with each
        :meth:`append_turn`, so highest indexed id equals highest
        ``turns.id`` for the familiar. Cheap MAX query rather than a
        tantivy round trip.
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
        dedup: bool = True,
    ) -> Fact:
        """Persist one fact. ``source_turn_ids`` + ``subjects`` stored as JSON.

        ``subjects`` is extractor's best-effort link to canonical
        identities — see :class:`FactSubject`.

        ``valid_from`` / ``valid_to`` are world-time (when fact
        applied in the world). When ``valid_from`` omitted, defaults
        to ``created_at``; callers (e.g. ``FactExtractor``) pass the
        source turn's timestamp explicitly. ``valid_to`` defaults to
        ``None`` — fact still applies.

        ``importance`` is extractor's 1-10 ranking hint (M2).
        Out-of-range values clamp to ``[1, 10]`` so a stray LLM
        number can't poison rank-time math. ``None`` preserved
        verbatim — downstream consumers treat as neutral midpoint.

        ``dedup`` (default True) gates near-duplicate suppression. The
        sleep-hygiene merge passes ``dedup=False`` so a consolidated
        fact whose text equals one of the rows it supersedes still
        inserts a fresh row (the olds are superseded right after).
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
        # near-duplicate suppression: skip insert when a CURRENT fact
        # for the same familiar + same subject-key set has matching
        # normalized text (kills alias/nickname restatement pile-up).
        # bypass when valid_to is set — such a fact may close/bound an
        # existing one, not restate it.
        if dedup and valid_to is None:
            existing = self._find_current_dup(
                familiar_id=familiar_id,
                norm_text=_normalize_fact_text(text),
                subject_keys=_subject_key_set(subjects_tuple),
            )
            if existing is not None:
                return existing
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
        self._safe_fts_add(self._fts_facts, fact_id, text, kind="fact")
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

    def _find_current_dup(
        self,
        *,
        familiar_id: str,
        norm_text: str,
        subject_keys: frozenset[str],
    ) -> Fact | None:
        """Existing current fact matching normalized text + subject set.

        Scans current (non-superseded, non-expired) facts for the
        familiar. Comparison is exact-match after normalization;
        subject scoping is set-equality on canonical keys.
        """
        where, params = _facts_validity_where(include_superseded=False, as_of=None)
        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE familiar_id = ? {where}",  # noqa: S608
            (familiar_id, *params),
        ).fetchall()
        for row in rows:
            fact = _row_to_fact(row)
            if (
                _normalize_fact_text(fact.text) == norm_text
                and _subject_key_set(fact.subjects) == subject_keys
            ):
                return fact
        return None

    def facts_by_ids(
        self,
        *,
        familiar_id: str,
        ids: Iterable[int],
    ) -> list[Fact]:
        """Fetch facts by id, scoped to ``familiar_id``, oldest first.

        Includes superseded rows — callers (sleep-apply) snapshot
        source facts that a concurrent writer may retire mid-pass.
        """
        unique_ids = sorted({int(i) for i in ids})
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self._conn.execute(
            f"""
            SELECT id, familiar_id, channel_id, text,
                   source_turn_ids, created_at,
                   superseded_at, superseded_by, subjects_json,
                   valid_from, valid_to, importance
              FROM facts
             WHERE familiar_id = ?
               AND id IN ({placeholders})
             ORDER BY id ASC
            """,  # noqa: S608
            (familiar_id, *unique_ids),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def recent_facts(
        self,
        *,
        familiar_id: str,
        limit: int,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[Fact]:
        """``limit`` most recent facts, newest first.

        Default ("current truth"): excludes superseded facts and any
        whose world-time ``valid_to`` is in the past.

        ``as_of`` switches to bi-temporal world-time slice — facts
        whose ``valid_from <= as_of`` and (``valid_to`` IS NULL or >
        ``as_of``). Includes superseded rows so audit queries can
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

        Runs tantivy query, joins back to ``facts`` with validity
        filter, re-ranks by BM25 desc then id desc. Returns
        ``[(fact, score)]`` truncated to *limit*.
        """
        if limit <= 0:
            return []
        # Overfetch from FTS so validity/familiar filters don't
        # starve the result. 4x is enough in practice (validity cheap)
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

        Tantivy BM25 is positive (higher = better). Callers fusing
        with other signals (importance, recency, embedding sim)
        should treat score as a non-negative weight; prior SQLite
        FTS5 ``bm25()`` returned negative numbers (lower = better),
        so consumers may have inverted-sign assumptions to revisit.
        """
        return self._fact_candidates_by_fts(
            familiar_id=familiar_id,
            query=query,
            limit=limit,
            include_superseded=include_superseded,
            as_of=as_of,
        )

    def latest_fact_id(self, *, familiar_id: str) -> int:
        """Highest ``facts.id`` for ``familiar_id``; 0 if none.

        Counts superseded rows too — cache invalidation key only
        needs to change on writes, and supersession-by-replacement
        already adds a new row so id ticks up naturally.
        """
        row = self._conn.execute(
            "SELECT MAX(id) AS max_id FROM facts WHERE familiar_id = ?",
            (familiar_id,),
        ).fetchone()
        max_id = row["max_id"] if row is not None else None
        return int(max_id) if max_id is not None else 0

    def supersede(
        self,
        *,
        familiar_id: str,
        obsolete_facts: Iterable[int],
        new_fact: FactDraft | Fact | int | None,
    ) -> SupersedeResult:
        """Unified mutation: retire, merge, or repoint obsolete facts.

        Three replacement shapes, distinguished by ``new_fact``:

        * ``None`` — **retire**: each obsolete row gets ``superseded_at``
          set, ``superseded_by`` left NULL (nothing replaces it).
          Per-id skip-and-record: a stale id is recorded, the rest
          still process.
        * :class:`FactDraft` — **merge**: ATOMIC / all-or-nothing. Pre-flight
          every obsolete row; if ANY is unknown or already superseded —
          or ``obsolete_facts`` is empty — the merge is declined WHOLE
          (nothing minted, ``minted=None``, every obsolete id recorded in
          ``skipped``). Only when every row is current do we MINT the
          replacement (``dedup=False``, so a consolidated text matching an
          obsolete row still inserts) and point every obsolete row's
          ``superseded_by`` at the minted id (many->one lineage). A
          phantom merge that supersedes nothing is impossible by
          construction.
        * :class:`Fact` / ``int`` — **repoint at an existing row**: the
          obsolete rows point at the supplied id; nothing is minted.
          Per-id skip-and-record like retire.

        The store owns merge lineage + provenance. For a draft, the
        minted fact's ``source_turn_ids`` is the order-preserving UNION
        of the obsolete rows' provenance — the caller never supplies it.
        Ancestry is then resolvable from the store alone via
        :meth:`ancestors_of` (reverse ``superseded_by`` walk). Because the
        merge mints only when every ancestor is valid, provenance-union
        equals ancestry exactly — no partial merge can break the
        invariant.
        """
        ids = [int(i) for i in obsolete_facts]
        obsolete_rows = {
            f.id: f for f in self.facts_by_ids(familiar_id=familiar_id, ids=ids)
        }

        if isinstance(new_fact, FactDraft):
            return self._merge_atomically(
                familiar_id=familiar_id,
                ids=ids,
                obsolete_rows=obsolete_rows,
                draft=new_fact,
            )

        if new_fact is None:
            new_id: int | None = None
        else:
            new_id = new_fact.id if isinstance(new_fact, Fact) else int(new_fact)

        superseded: list[int] = []
        skipped: list[tuple[int, str]] = []
        ts = datetime.now(tz=UTC).isoformat()
        for fid in ids:
            row = obsolete_rows.get(fid)
            if row is None:
                skipped.append((fid, f"unknown fact id={fid}"))
                continue
            if row.superseded_at is not None:
                skipped.append((fid, f"fact id={fid} already superseded"))
                continue
            self._conn.execute(
                "UPDATE facts SET superseded_at = ?, superseded_by = ? "
                "WHERE id = ? AND familiar_id = ?",
                (ts, new_id, fid, familiar_id),
            )
            self._invalidate_dossiers_for_keys(
                familiar_id=familiar_id,
                keys=_subject_key_set(row.subjects),
            )
            superseded.append(fid)
        self._conn.commit()
        # retire (new_id None) and existing-id repoint mint nothing.
        return SupersedeResult(
            minted=None,
            superseded=tuple(superseded),
            skipped=tuple(skipped),
        )

    def _merge_atomically(
        self,
        *,
        familiar_id: str,
        ids: list[int],
        obsolete_rows: dict[int, Fact],
        draft: FactDraft,
    ) -> SupersedeResult:
        """Mint a merge only if EVERY obsolete row is current — else decline.

        All-or-nothing pre-flight (mirrors ``sleep/apply.py``'s rewrite
        semantics): an empty ``ids`` or any obsolete id that's unknown or
        already superseded means we mint nothing and supersede nothing —
        the merge is declined whole, every obsolete id recorded in
        ``skipped``. This makes a phantom (a minted CURRENT fact that
        supersedes no ancestor) impossible, so provenance-union always
        equals ancestry.
        """
        stale: list[tuple[int, str]] = []
        for fid in ids:
            row = obsolete_rows.get(fid)
            if row is None:
                stale.append((fid, f"unknown fact id={fid}"))
            elif row.superseded_at is not None:
                stale.append((fid, f"fact id={fid} already superseded"))
        if not ids or stale:
            return SupersedeResult(minted=None, superseded=(), skipped=tuple(stale))

        minted = self.append_fact(
            familiar_id=familiar_id,
            channel_id=draft.channel_id,
            text=draft.text,
            source_turn_ids=_union_provenance(obsolete_rows[fid] for fid in ids),
            subjects=draft.subjects,
            dedup=False,
        )
        ts = datetime.now(tz=UTC).isoformat()
        for fid in ids:
            self._conn.execute(
                "UPDATE facts SET superseded_at = ?, superseded_by = ? "
                "WHERE id = ? AND familiar_id = ?",
                (ts, minted.id, fid, familiar_id),
            )
            self._invalidate_dossiers_for_keys(
                familiar_id=familiar_id,
                keys=_subject_key_set(obsolete_rows[fid].subjects),
            )
        self._conn.commit()
        return SupersedeResult(minted=minted, superseded=tuple(ids), skipped=())

    def ancestors_of(
        self,
        *,
        familiar_id: str,
        fact_id: int,
    ) -> list[Fact]:
        """Facts directly superseded by ``fact_id`` — its merge ancestors.

        Reverse ``superseded_by`` walk: a merge points every obsolete
        row at the minted fact (many->one), so "ancestors of X" is
        exactly the rows whose ``superseded_by = X``. One hop — direct
        parents, oldest first; chase the chain by re-querying each.
        """
        rows = self._conn.execute(
            """
            SELECT id, familiar_id, channel_id, text,
                   source_turn_ids, created_at,
                   superseded_at, superseded_by, subjects_json,
                   valid_from, valid_to, importance
              FROM facts
             WHERE familiar_id = ? AND superseded_by = ?
             ORDER BY id ASC
            """,
            (familiar_id, int(fact_id)),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def _invalidate_dossiers_for_keys(
        self, *, familiar_id: str, keys: Iterable[str]
    ) -> None:
        """DELETE baked dossiers for ``keys`` — authoritative primitive.

        Sole owner of the dossier-drop knowledge; :meth:`supersede`
        routes here (it holds parsed :class:`FactSubject`s).

        PeopleDossierWorker compounds prior dossier text + only newer
        facts; never un-bakes a retired fact. DELETE the row (not reset
        the watermark) — a surviving row re-compounds stale/poisoned
        prose via the "Previous dossier" path. Absent row → prior=None →
        clean full rebuild. Empty ``keys`` → no-op. Caller commits.
        """
        for key in keys:
            self._conn.execute(
                "DELETE FROM people_dossiers "
                "WHERE familiar_id = ? AND canonical_key = ?",
                (familiar_id, key),
            )

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

        Stored as packed little-endian float32. ``model`` is
        embedder's :attr:`Embedder.name`; pairing with ``fact_id``
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
        """``{fact_id: vector}`` for requested ids + model.

        Missing rows simply absent from result — caller treats as
        "not yet embedded" and skips embedding signal for that
        candidate.
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
        """List current facts lacking an embedding row for ``model``.

        "Current" matches :meth:`recent_facts` defaults — superseded
        rows excluded. Projector embeds in id order so interrupted
        run resumes deterministically.
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

        ``last_turn_id`` / ``last_fact_id`` snapshot worker's view at
        write time — also serve as next tick's watermark.
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
        """``limit`` most recent reflections, newest first.

        ``channel_id`` scopes to one channel; ``None`` returns
        reflections regardless of channel scope (including
        channel-agnostic rows).
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
            # global reflection still surfaces in any channel
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
        """(last_turn_id, last_fact_id) the worker last processed.

        Prefers explicit ``reflection_watermark`` row — advanced
        every tick, including no-op ticks. Falls back to newest
        reflection row for back-compat with databases written before
        the watermark table existed. ``(0, 0)`` if neither exists.
        """
        wm = self._conn.execute(
            """
            SELECT last_turn_id, last_fact_id
              FROM reflection_watermark
             WHERE familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        if wm is not None:
            return (int(wm["last_turn_id"]), int(wm["last_fact_id"]))
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

    def set_reflection_watermark(
        self,
        *,
        familiar_id: str,
        last_turn_id: int,
        last_fact_id: int,
    ) -> None:
        """Upsert reflection watermark for *familiar_id*.

        Called by :class:`ReflectionWorker` at end of every tick —
        regardless of whether a reflection row was written — so a
        no-substance LLM reply can't pin the worker to an
        ever-growing turn window.
        """
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO reflection_watermark
                (familiar_id, last_turn_id, last_fact_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (familiar_id)
            DO UPDATE SET
                last_turn_id = excluded.last_turn_id,
                last_fact_id = excluded.last_fact_id,
                updated_at   = excluded.updated_at
            """,
            (familiar_id, int(last_turn_id), int(last_fact_id), ts),
        )
        self._conn.commit()

    def superseded_fact_ids(
        self,
        *,
        familiar_id: str,
        fact_ids: Iterable[int],
    ) -> set[int]:
        """Subset of ``fact_ids`` that are superseded.

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

        ``scheduled_at`` is ISO-8601 UTC timestamp. ``channel_kind``
        must be ``"text"`` or ``"voice"`` (enforced by CHECK).
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
        """Pending alarms (not fired, not cancelled) for ``familiar_id``.

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

    # ------------------------------------------------------------------
    # Attentional stream — staging, promotion, cross-channel recall
    # ------------------------------------------------------------------

    def stage_turn(
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
        arrived_at: datetime | None = None,
    ) -> HistoryTurn:
        """Wrap append_turn with consumed=False."""
        return self.append_turn(
            familiar_id=familiar_id,
            channel_id=channel_id,
            role=role,
            content=content,
            author=author,
            guild_id=guild_id,
            mode=mode,
            platform_message_id=platform_message_id,
            reply_to_message_id=reply_to_message_id,
            tool_calls_json=tool_calls_json,
            tool_call_id=tool_call_id,
            arrived_at=arrived_at,
            consumed=False,
        )

    def promote_staged_turns(
        self,
        *,
        familiar_id: str,
        channel_id: int,
    ) -> int:
        """Set consumed_at = NOW() for all staged turns in channel.

        Returns count of promoted rows.
        """
        now = datetime.now(tz=UTC).isoformat()
        cur = self._conn.execute(
            """
            UPDATE turns SET consumed_at = ?
             WHERE familiar_id = ? AND channel_id = ? AND consumed_at IS NULL
            """,
            (now, familiar_id, channel_id),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def promote_staged_turns_since(
        self,
        *,
        familiar_id: str,
        after_turn_id: int,
    ) -> int:
        """Set consumed_at = NOW() for staged turns with id > after_turn_id.

        All channels — return-from-absence promotion ("she reads the
        screen when she gets back"). Pre-absence staged turns
        (id <= after_turn_id) keep their attentional semantics.
        Returns count of promoted rows.
        """
        now = datetime.now(tz=UTC).isoformat()
        cur = self._conn.execute(
            """
            UPDATE turns SET consumed_at = ?
             WHERE familiar_id = ? AND consumed_at IS NULL AND id > ?
            """,
            (now, familiar_id, after_turn_id),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def count_staged(self, *, familiar_id: str, channel_id: int) -> int:
        """Count turns with consumed_at IS NULL for familiar + channel."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
              FROM turns
             WHERE familiar_id = ? AND channel_id = ? AND consumed_at IS NULL
            """,
            (familiar_id, channel_id),
        ).fetchone()
        return int(row["n"])

    def staged_channels(self, *, familiar_id: str) -> dict[int, int]:
        """Map channel_id → staged_count for all channels with staged turns."""
        rows = self._conn.execute(
            """
            SELECT channel_id, COUNT(*) AS n
              FROM turns
             WHERE familiar_id = ? AND consumed_at IS NULL
             GROUP BY channel_id
            """,
            (familiar_id,),
        ).fetchall()
        return {int(r["channel_id"]): int(r["n"]) for r in rows}

    def recent_cross_channel(
        self,
        *,
        familiar_id: str,
        limit: int,
        respect_archive: bool = False,
    ) -> list[HistoryTurn]:
        """Last *limit* consumed turns across all channels, oldest-first.

        Fetches DESC then reverses so callers get temporal order
        without a subquery.

        *respect_archive* drops turns at/below their channel's archive
        watermark — applied *outside* the latest-*limit* window, so the
        window shrinks rather than backfills past the watermark
        (archive marks a break, not a paging anchor). Missing
        watermark row ⇒ no filter for that channel.
        """
        if limit <= 0:
            return []
        inner = f"""
            SELECT {_TURN_COLS}, familiar_id
              FROM turns
             WHERE familiar_id = ? AND consumed_at IS NOT NULL
             ORDER BY arrived_at DESC, id DESC
             LIMIT ?
        """  # noqa: S608
        if respect_archive:
            query = f"""
                SELECT * FROM ({inner}) AS t
                 WHERE t.id > COALESCE(
                           (SELECT turn_id
                              FROM channel_archive_watermark w
                             WHERE w.familiar_id = t.familiar_id
                               AND w.channel_id = t.channel_id), 0)
                 ORDER BY t.arrived_at DESC, t.id DESC
            """  # noqa: S608
        else:
            query = inner
        rows = self._conn.execute(query, (familiar_id, limit)).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Focus pointers
    # ------------------------------------------------------------------

    def get_focus_pointers(self, familiar_id: str) -> FocusPointers | None:
        """Return current focus pointers for familiar, or None."""
        row = self._conn.execute(
            """
            SELECT text_channel_id, voice_channel_id, updated_at
              FROM focus_pointers
             WHERE familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return None
        return FocusPointers(
            text_channel_id=(
                int(row["text_channel_id"])
                if row["text_channel_id"] is not None
                else None
            ),
            voice_channel_id=(
                int(row["voice_channel_id"])
                if row["voice_channel_id"] is not None
                else None
            ),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def set_focus_pointers(
        self,
        familiar_id: str,
        *,
        text_channel_id: int | None,
        voice_channel_id: int | None,
    ) -> None:
        """Upsert focus pointers for familiar."""
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO focus_pointers
                (familiar_id, text_channel_id, voice_channel_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (familiar_id) DO UPDATE SET
                text_channel_id  = excluded.text_channel_id,
                voice_channel_id = excluded.voice_channel_id,
                updated_at       = excluded.updated_at
            """,
            (familiar_id, text_channel_id, voice_channel_id, ts),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Channel archive watermark (activities)
    # ------------------------------------------------------------------

    def set_archive_watermark(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        turn_id: int,
    ) -> None:
        """Upsert archive watermark for familiar + channel.

        Set at departure turn id when absence exceeds the archive
        threshold; :meth:`recent_cross_channel` with
        ``respect_archive=True`` hides turns at/below it.
        """
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO channel_archive_watermark
                (familiar_id, channel_id, turn_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (familiar_id, channel_id) DO UPDATE SET
                turn_id    = excluded.turn_id,
                updated_at = excluded.updated_at
            """,
            (familiar_id, channel_id, int(turn_id), ts),
        )
        self._conn.commit()

    def set_archive_watermark_all(
        self,
        *,
        familiar_id: str,
        turn_id: int,
    ) -> None:
        """Upsert archive watermark for every channel with turns.

        Absence is global — one departure point breaks every channel's
        window (turn ids are globally monotonic).
        """
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO channel_archive_watermark
                (familiar_id, channel_id, turn_id, updated_at)
            SELECT DISTINCT familiar_id, channel_id, ?, ?
              FROM turns
             WHERE familiar_id = ?
            ON CONFLICT (familiar_id, channel_id) DO UPDATE SET
                turn_id    = excluded.turn_id,
                updated_at = excluded.updated_at
            """,
            (int(turn_id), ts, familiar_id),
        )
        self._conn.commit()

    def latest_id_at_or_before(
        self,
        *,
        familiar_id: str,
        ts: datetime,
    ) -> int | None:
        """Highest turn id with timestamp ≤ *ts*, across all channels.

        Departure-point recovery after restart (timestamps stored
        isoformat UTC — lexicographic compare is chronological).
        """
        row = self._conn.execute(
            """
            SELECT MAX(id) AS max_id
              FROM turns
             WHERE familiar_id = ? AND timestamp <= ?
            """,
            (familiar_id, ts.isoformat()),
        ).fetchone()
        if row is None or row["max_id"] is None:
            return None
        return int(row["max_id"])

    def get_archive_watermark(
        self,
        *,
        familiar_id: str,
        channel_id: int,
    ) -> int | None:
        """Archive watermark turn id, or ``None`` when unset."""
        row = self._conn.execute(
            """
            SELECT turn_id
              FROM channel_archive_watermark
             WHERE familiar_id = ? AND channel_id = ?
            """,
            (familiar_id, channel_id),
        ).fetchone()
        if row is None:
            return None
        return int(row["turn_id"])

    # ------------------------------------------------------------------
    # Activities — append-only absence log
    # ------------------------------------------------------------------

    def create_activity(
        self,
        *,
        familiar_id: str,
        type_id: str,
        label: str,
        started_at: datetime,
        planned_return_at: datetime,
        note: str | None,
    ) -> int:
        """Insert activity row; return its id.

        Row is "active" until :meth:`finish_activity` stamps
        ``actual_return_at``.
        """
        cur = self._conn.execute(
            """
            INSERT INTO activities
                (familiar_id, type_id, label,
                 started_at, planned_return_at, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                familiar_id,
                type_id,
                label,
                started_at.isoformat(),
                planned_return_at.isoformat(),
                note,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def finish_activity(
        self,
        *,
        activity_id: int,
        status: str,
        actual_return_at: datetime,
        experience_text: str | None,
    ) -> None:
        """Stamp return fields on one activity.

        *status* must be ``"completed"`` or ``"cut_short"`` —
        anything else raises ``ValueError``.
        """
        if status not in {"completed", "cut_short"}:
            msg = f"invalid activity status: {status!r}"
            raise ValueError(msg)
        self._conn.execute(
            """
            UPDATE activities
               SET status = ?, actual_return_at = ?, experience_text = ?
             WHERE id = ?
            """,
            (status, actual_return_at.isoformat(), experience_text, activity_id),
        )
        self._conn.commit()

    def set_activity_experience(
        self, *, activity_id: int, experience_text: str
    ) -> None:
        """Persist dream/experience prose on one activity row (idempotent)."""
        self._conn.execute(
            "UPDATE activities SET experience_text = ? WHERE id = ?",
            (experience_text, activity_id),
        )
        self._conn.commit()

    def active_activity(self, *, familiar_id: str) -> ActivityRecord | None:
        """Newest activity with ``actual_return_at IS NULL``, or ``None``."""
        row = self._conn.execute(
            """
            SELECT id, familiar_id, type_id, label,
                   started_at, planned_return_at, note,
                   status, actual_return_at, experience_text
              FROM activities
             WHERE familiar_id = ? AND actual_return_at IS NULL
             ORDER BY id DESC
             LIMIT 1
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_activity(row)

    def latest_activity(
        self, *, familiar_id: str, type_id: str
    ) -> ActivityRecord | None:
        """Newest activity of *type_id* (active or finished), or ``None``.

        Sleep-window guard keys on this: most recent sleep row's
        ``started_at`` vs the current window occurrence's start.
        """
        row = self._conn.execute(
            """
            SELECT id, familiar_id, type_id, label,
                   started_at, planned_return_at, note,
                   status, actual_return_at, experience_text
              FROM activities
             WHERE familiar_id = ? AND type_id = ?
             ORDER BY id DESC
             LIMIT 1
            """,
            (familiar_id, type_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_activity(row)

    # ------------------------------------------------------------------
    # Unread digest watermark
    # ------------------------------------------------------------------

    def get_digest_watermark(self, familiar_id: str) -> datetime | None:
        """Return max(arrived_at) of last-delivered digest, or None."""
        row = self._conn.execute(
            """
            SELECT watermark_at
              FROM unread_digest_watermark
             WHERE familiar_id = ?
            """,
            (familiar_id,),
        ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row["watermark_at"])

    def set_digest_watermark(
        self,
        familiar_id: str,
        watermark_at: datetime,
    ) -> None:
        """Upsert digest watermark for familiar."""
        ts = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO unread_digest_watermark
                (familiar_id, watermark_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (familiar_id) DO UPDATE SET
                watermark_at = excluded.watermark_at,
                updated_at   = excluded.updated_at
            """,
            (familiar_id, watermark_at.isoformat(), ts),
        )
        self._conn.commit()


def _row_to_activity(row: Row) -> ActivityRecord:
    return ActivityRecord(
        id=int(row["id"]),
        familiar_id=str(row["familiar_id"]),
        type_id=str(row["type_id"]),
        label=str(row["label"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        planned_return_at=datetime.fromisoformat(row["planned_return_at"]),
        note=str(row["note"]) if row["note"] is not None else None,
        status=str(row["status"]) if row["status"] is not None else None,
        actual_return_at=(
            datetime.fromisoformat(row["actual_return_at"])
            if row["actual_return_at"] is not None
            else None
        ),
        experience_text=(
            str(row["experience_text"]) if row["experience_text"] is not None else None
        ),
    )


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
    """Rebuild a HistoryTurn from a SELECT row. Author reconstructed.

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
        mode_raw = row["mode"]
    except (IndexError, KeyError):
        mode_raw = None
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
    try:
        arrived_at_raw = row["arrived_at"]
    except (IndexError, KeyError):
        arrived_at_raw = None
    try:
        consumed_at_raw = row["consumed_at"]
    except (IndexError, KeyError):
        consumed_at_raw = None
    return HistoryTurn(
        id=int(row["id"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        role=str(row["role"]),
        author=author,
        content=str(row["content"]),
        channel_id=channel_id,
        mode=str(mode_raw) if mode_raw is not None else None,
        platform_message_id=(
            str(platform_message_id) if platform_message_id is not None else None
        ),
        reply_to_message_id=(
            str(reply_to_message_id) if reply_to_message_id is not None else None
        ),
        guild_id=int(guild_id_raw) if guild_id_raw is not None else None,
        arrived_at=(
            datetime.fromisoformat(arrived_at_raw)
            if arrived_at_raw is not None
            else None
        ),
        consumed_at=(
            datetime.fromisoformat(consumed_at_raw)
            if consumed_at_raw is not None
            else None
        ),
    )
