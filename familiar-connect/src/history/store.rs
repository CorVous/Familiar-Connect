//! `HistoryStore`: turns log + watermarked side-index projections
//! (subsystem 03; Python `history/store.py`).
//!
//! One SQLite database per familiar. The append-only `turns` table is the
//! source of truth; every side-index (summaries, facts, fact embeddings, people
//! dossiers, reflections, alarms, activities, identity cache, reactions,
//! focus/attention state) is a projection that can be deleted and rebuilt from
//! `turns`. Full-text search lives outside the DB behind the [`FtsIndex`] seam
//! (tantivy in stage B; a no-op in stage A).
//!
//! Timestamps are always emitted through [`iso_utc`] (fixed-width microseconds,
//! `+00:00`) so lexicographic ordering equals chronological ordering — a
//! correctness dependency in five query paths (DESIGN §4.2).
//!
//! Author identity is [`crate::identity::Author`] (re-exported at the module
//! root as `history::Author`): the store round-trips its `platform` / `user_id`
//! / `username` / `display_name` / `global_name` / `pronouns` / `bio` fields.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{Row, params};

use super::StoreError;
use super::db::{Db, Value};
use super::fts::TantivyFts;
use crate::identity::Author;
use crate::support::time::{iso_utc, parse_iso};

/// Reserved channel id for the per-familiar focus-stream summary (the consumed
/// cross-channel stream). Distinct from real (large positive) channel ids and
/// from the channel-less bucket (`0`).
pub const FOCUS_STREAM_CHANNEL_ID: i64 = -1;

/// Fallback catch-up window when a caller omits one; the canonical knob is
/// `[focus].catch_up_limit` (subsystem 02), threaded by FocusManager /
/// ActivityEngine.
const DEFAULT_CATCH_UP_LIMIT: usize = 20;

/// Promotion `UPDATE ... WHERE id IN (...)` chunk size (SQLite param cap).
const STAMP_CHUNK: usize = 500;

const TURN_COLS: &str = "id, timestamp, role, author_platform, author_user_id, \
     author_username, author_display_name, content, channel_id, mode, \
     platform_message_id, reply_to_message_id, guild_id, arrived_at, consumed_at, pings_bot";

const TURN_COLS_T: &str = "t.id, t.timestamp, t.role, t.author_platform, t.author_user_id, \
     t.author_username, t.author_display_name, t.content, t.channel_id, t.mode, \
     t.platform_message_id, t.reply_to_message_id, t.guild_id, t.arrived_at, t.consumed_at, \
     t.pings_bot";

const FACT_COLS: &str = "id, familiar_id, channel_id, text, source_turn_ids, created_at, \
     superseded_at, superseded_by, subjects_json, valid_from, valid_to, importance";

const FACT_COLS_F: &str = "f.id, f.familiar_id, f.channel_id, f.text, f.source_turn_ids, \
     f.created_at, f.superseded_at, f.superseded_by, f.subjects_json, f.valid_from, f.valid_to, \
     f.importance";

const REFLECTION_COLS: &str = "id, familiar_id, channel_id, text, cited_turn_ids, cited_fact_ids, \
     created_at, last_turn_id, last_fact_id";

const ACTIVITY_COLS: &str = "id, familiar_id, type_id, label, started_at, planned_return_at, note, \
     status, actual_return_at, experience_text";

const SCHEMA: &str = "
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
    tool_call_id           TEXT,
    pings_bot              INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_channel
    ON turns (familiar_id, channel_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_global
    ON turns (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_turns_channel_mode
    ON turns (familiar_id, channel_id, mode, id);

CREATE INDEX IF NOT EXISTS idx_turns_platform_msg
    ON turns (familiar_id, platform_message_id);

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

CREATE TABLE IF NOT EXISTS reflections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id     TEXT    NOT NULL,
    channel_id      INTEGER,
    text            TEXT    NOT NULL,
    cited_turn_ids  TEXT    NOT NULL,
    cited_fact_ids  TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    last_turn_id    INTEGER NOT NULL,
    last_fact_id    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflections_familiar
    ON reflections (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_reflections_familiar_channel
    ON reflections (familiar_id, channel_id, id);

CREATE TABLE IF NOT EXISTS reflection_watermark (
    familiar_id   TEXT    PRIMARY KEY,
    last_turn_id  INTEGER NOT NULL,
    last_fact_id  INTEGER NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sleep_watermark (
    familiar_id   TEXT    PRIMARY KEY,
    last_fact_id  INTEGER NOT NULL,
    last_turn_id  INTEGER NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    canonical_key  TEXT PRIMARY KEY,
    platform       TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    username       TEXT,
    global_name    TEXT,
    pronouns       TEXT,
    bio            TEXT,
    last_seen_at   TEXT NOT NULL
);

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
    source_turn_ids   TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    superseded_at     TEXT,
    superseded_by     INTEGER,
    subjects_json     TEXT,
    valid_from        TEXT,
    valid_to          TEXT,
    importance        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_facts_familiar
    ON facts (familiar_id, id);

CREATE INDEX IF NOT EXISTS idx_facts_familiar_current
    ON facts (familiar_id, superseded_at, id);

CREATE INDEX IF NOT EXISTS idx_facts_familiar_validity
    ON facts (familiar_id, valid_from, valid_to);

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

CREATE TABLE IF NOT EXISTS focus_pointers (
    familiar_id      TEXT PRIMARY KEY,
    text_channel_id  INTEGER,
    voice_channel_id INTEGER,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unread_digest_watermark (
    familiar_id    TEXT PRIMARY KEY,
    watermark_at   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_archive_watermark (
    familiar_id  TEXT    NOT NULL,
    channel_id   INTEGER NOT NULL,
    turn_id      INTEGER NOT NULL,
    updated_at   TEXT    NOT NULL,
    PRIMARY KEY (familiar_id, channel_id)
);

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
";

// ---------------------------------------------------------------------------
// Value types (Python frozen dataclasses / NamedTuples)
// ---------------------------------------------------------------------------

/// A single persisted conversational turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryTurn {
    /// Engine-assigned monotonic id.
    pub id: i64,
    /// Write time (now, UTC).
    pub timestamp: DateTime<Utc>,
    /// `"user"` | `"assistant"` | `"tool"` | free-form.
    pub role: String,
    /// Author, or `None` for assistant/system turns.
    pub author: Option<Author>,
    /// Turn text.
    pub content: String,
    /// Channel id (`0` = channel-less, `-1` = focus stream).
    pub channel_id: i64,
    /// Free-form mode tag.
    pub mode: Option<String>,
    /// Platform-native message id.
    pub platform_message_id: Option<String>,
    /// Parent platform message id for replies.
    pub reply_to_message_id: Option<String>,
    /// Discord guild id (observability only).
    pub guild_id: Option<i64>,
    /// Immutable ingest time; `None` only on pre-migration rows.
    pub arrived_at: Option<DateTime<Utc>>,
    /// Consumption time; `None` = staged (or missed).
    pub consumed_at: Option<DateTime<Utc>>,
    /// Did the incoming message ping the bot?
    pub pings_bot: bool,
}

/// Staged-turn tally for one channel: total unread + bot-ping subset.
///
/// Tuple-struct so consumers can destructure it as a plain 2-tuple
/// (`let ChannelUnread(unread, pings) = x;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelUnread(pub i64, pub i64);

impl ChannelUnread {
    /// Total staged turns.
    #[must_use]
    pub const fn unread(&self) -> i64 {
        self.0
    }
    /// Bot-ping subset.
    #[must_use]
    pub const fn pings(&self) -> i64 {
        self.1
    }
}

/// Outcome of a staged-turn promotion: consumed vs. missed counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Promotion {
    /// Turns that gained `consumed_at`.
    pub consumed: usize,
    /// Turns that gained `missed_at` (terminal).
    pub missed: usize,
}

/// One activity row (append-only log).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityRecord {
    pub id: i64,
    pub familiar_id: String,
    pub type_id: String,
    pub label: String,
    pub started_at: DateTime<Utc>,
    pub planned_return_at: DateTime<Utc>,
    pub note: Option<String>,
    pub status: Option<String>,
    pub actual_return_at: Option<DateTime<Utc>>,
    pub experience_text: Option<String>,
}

/// Cached rolling summary for one (familiar, channel).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryEntry {
    pub last_summarised_id: i64,
    pub summary_text: String,
    pub created_at: DateTime<Utc>,
    /// Composite-watermark cursor kept as an opaque string; `None` on legacy rows.
    pub last_consumed_at: Option<String>,
}

/// Recent-activity info for another channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherChannelInfo {
    pub channel_id: i64,
    pub mode: Option<String>,
    pub latest_id: i64,
    pub latest_timestamp: DateTime<Utc>,
}

/// Last turn id written to long-term memory by the memory writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatermarkEntry {
    pub last_written_id: i64,
    pub created_at: DateTime<Utc>,
}

/// Highest fact/turn ids the last sleep consolidation pass saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SleepWatermark {
    pub last_fact_id: i64,
    pub last_turn_id: i64,
    pub updated_at: DateTime<Utc>,
}

/// Current text/voice channel focus for a familiar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusPointers {
    pub text_channel_id: Option<i64>,
    pub voice_channel_id: Option<i64>,
    pub updated_at: DateTime<Utc>,
}

/// Read-side projection of `accounts` profile metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountProfile {
    pub canonical_key: String,
    pub username: Option<String>,
    pub global_name: Option<String>,
    pub pronouns: Option<String>,
    pub bio: Option<String>,
}

/// Cached per-person dossier compounded from facts mentioning `canonical_key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeopleDossierEntry {
    pub canonical_key: String,
    pub last_fact_id: i64,
    pub dossier_text: String,
    pub created_at: DateTime<Utc>,
}

/// Soft link from a fact to one canonical identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactSubject {
    pub canonical_key: String,
    pub display_at_write: String,
}

/// Consolidated *content* for a merge, before the store mints it. Carries no
/// turn ids — the store owns provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactDraft {
    pub channel_id: Option<i64>,
    pub text: String,
    pub subjects: Vec<FactSubject>,
}

/// Outcome of one [`HistoryStore::supersede`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupersedeResult {
    /// Freshly minted replacement (merge form only), else `None`.
    pub minted: Option<Fact>,
    /// Obsolete ids actually marked this call.
    pub superseded: Vec<i64>,
    /// `(id, reason)` for obsolete rows left untouched.
    pub skipped: Vec<(i64, String)>,
}

/// Replacement shape for [`HistoryStore::supersede`] (Python's
/// `FactDraft | Fact | int | None` union).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewFact {
    /// Retire each obsolete row (`superseded_by` stays NULL).
    Retire,
    /// Mint a merged replacement (atomic / all-or-nothing).
    Merge(FactDraft),
    /// Repoint obsolete rows at an existing fact id.
    Repoint(i64),
}

/// Higher-order synthesis over recent turns + facts (M3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reflection {
    pub id: i64,
    pub familiar_id: String,
    pub channel_id: Option<i64>,
    pub text: String,
    pub cited_turn_ids: Vec<i64>,
    pub cited_fact_ids: Vec<i64>,
    pub created_at: DateTime<Utc>,
    pub last_turn_id: i64,
    pub last_fact_id: i64,
}

/// Atomic fact extracted from one or more turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub id: i64,
    pub familiar_id: String,
    pub channel_id: Option<i64>,
    pub text: String,
    pub source_turn_ids: Vec<i64>,
    pub created_at: DateTime<Utc>,
    pub superseded_at: Option<DateTime<Utc>>,
    pub superseded_by: Option<i64>,
    pub subjects: Vec<FactSubject>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub importance: Option<i64>,
}

/// One pending/terminal alarm row (Python returned raw dicts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlarmRow {
    pub id: String,
    pub familiar_id: String,
    pub channel_id: i64,
    pub channel_kind: String,
    pub scheduled_at: String,
    pub reason: String,
    pub originating_turn_id: Option<String>,
    pub fired_at: Option<String>,
    pub cancelled_at: Option<String>,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// append_turn / append_fact builders (avoid unwieldy positional arg lists)
// ---------------------------------------------------------------------------

/// Builder for [`HistoryStore::append_turn`] / [`HistoryStore::stage_turn`].
///
/// `consumed` defaults to `true` (matching the Python default); `stage_turn`
/// forces it to `false`.
#[derive(Debug, Clone)]
pub struct AppendTurn {
    familiar_id: String,
    channel_id: i64,
    role: String,
    content: String,
    author: Option<Author>,
    guild_id: Option<i64>,
    mode: Option<String>,
    platform_message_id: Option<String>,
    reply_to_message_id: Option<String>,
    tool_calls_json: Option<String>,
    tool_call_id: Option<String>,
    arrived_at: Option<DateTime<Utc>>,
    consumed: bool,
    pings_bot: bool,
}

impl AppendTurn {
    /// The four required fields; all optionals default empty, `consumed = true`.
    pub fn new(
        familiar_id: impl Into<String>,
        channel_id: i64,
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            channel_id,
            role: role.into(),
            content: content.into(),
            author: None,
            guild_id: None,
            mode: None,
            platform_message_id: None,
            reply_to_message_id: None,
            tool_calls_json: None,
            tool_call_id: None,
            arrived_at: None,
            consumed: true,
            pings_bot: false,
        }
    }

    #[must_use]
    pub fn author(mut self, author: Author) -> Self {
        self.author = Some(author);
        self
    }
    #[must_use]
    pub const fn guild_id(mut self, guild_id: i64) -> Self {
        self.guild_id = Some(guild_id);
        self
    }
    #[must_use]
    pub fn mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = Some(mode.into());
        self
    }
    #[must_use]
    pub fn platform_message_id(mut self, id: impl Into<String>) -> Self {
        self.platform_message_id = Some(id.into());
        self
    }
    #[must_use]
    pub fn reply_to_message_id(mut self, id: impl Into<String>) -> Self {
        self.reply_to_message_id = Some(id.into());
        self
    }
    #[must_use]
    pub fn tool_calls_json(mut self, json: impl Into<String>) -> Self {
        self.tool_calls_json = Some(json.into());
        self
    }
    #[must_use]
    pub fn tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
    #[must_use]
    pub const fn arrived_at(mut self, at: DateTime<Utc>) -> Self {
        self.arrived_at = Some(at);
        self
    }
    #[must_use]
    pub const fn consumed(mut self, consumed: bool) -> Self {
        self.consumed = consumed;
        self
    }
    #[must_use]
    pub const fn pings_bot(mut self, pings_bot: bool) -> Self {
        self.pings_bot = pings_bot;
        self
    }
}

/// Builder for [`HistoryStore::append_fact`]. `dedup` defaults to `true`.
#[derive(Debug, Clone)]
pub struct AppendFact {
    familiar_id: String,
    channel_id: Option<i64>,
    text: String,
    source_turn_ids: Vec<i64>,
    subjects: Vec<FactSubject>,
    valid_from: Option<DateTime<Utc>>,
    valid_to: Option<DateTime<Utc>>,
    importance: Option<i64>,
    dedup: bool,
}

impl AppendFact {
    pub fn new(
        familiar_id: impl Into<String>,
        channel_id: Option<i64>,
        text: impl Into<String>,
        source_turn_ids: Vec<i64>,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            channel_id,
            text: text.into(),
            source_turn_ids,
            subjects: Vec::new(),
            valid_from: None,
            valid_to: None,
            importance: None,
            dedup: true,
        }
    }

    #[must_use]
    pub fn subjects(mut self, subjects: Vec<FactSubject>) -> Self {
        self.subjects = subjects;
        self
    }
    #[must_use]
    pub const fn valid_from(mut self, at: DateTime<Utc>) -> Self {
        self.valid_from = Some(at);
        self
    }
    #[must_use]
    pub const fn valid_to(mut self, at: DateTime<Utc>) -> Self {
        self.valid_to = Some(at);
        self
    }
    #[must_use]
    pub const fn importance(mut self, importance: i64) -> Self {
        self.importance = Some(importance);
        self
    }
    #[must_use]
    pub const fn dedup(mut self, dedup: bool) -> Self {
        self.dedup = dedup;
        self
    }
}

// ---------------------------------------------------------------------------
// FTS seam
// ---------------------------------------------------------------------------

/// Full-text index seam over `(row_id, content)` for one relational table.
///
/// Stage B implements this with tantivy in [`super::fts`]; stage A ships a
/// no-op so the store (and its non-FTS tests) build and run without the index.
pub trait FtsIndex: Send + Sync {
    /// Upsert one document (delete-by-row_id then add), committing immediately.
    fn add(&self, row_id: i64, content: &str) -> Result<(), StoreError>;
    /// Bulk upsert with a single commit; empty input is a no-op.
    fn add_many(&self, rows: &[(i64, String)]) -> Result<(), StoreError>;
    /// Delete one document by row id.
    fn delete(&self, row_id: i64) -> Result<(), StoreError>;
    /// Drop every document.
    fn clear(&self) -> Result<(), StoreError>;
    /// Return `[(row_id, bm25_score)]` for `query`, highest score first.
    fn search(&self, query: &str, limit: usize) -> Vec<(i64, f32)>;
}

/// No-op FTS: writes are dropped, searches return nothing.
///
/// Useful for callers that want to disable full-text indexing, and for tests
/// that inject a non-indexing side (see [`HistoryStore::open_with_fts`]). The
/// default [`HistoryStore::open`] uses the real tantivy index
/// ([`super::fts::TantivyFts`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopFtsIndex;

impl FtsIndex for NoopFtsIndex {
    fn add(&self, _row_id: i64, _content: &str) -> Result<(), StoreError> {
        Ok(())
    }
    fn add_many(&self, _rows: &[(i64, String)]) -> Result<(), StoreError> {
        Ok(())
    }
    fn delete(&self, _row_id: i64) -> Result<(), StoreError> {
        Ok(())
    }
    fn clear(&self) -> Result<(), StoreError> {
        Ok(())
    }
    fn search(&self, _query: &str, _limit: usize) -> Vec<(i64, f32)> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Free helpers (Python module-level functions)
// ---------------------------------------------------------------------------

fn v_str(s: impl Into<String>) -> Value {
    Value::Text(s.into())
}
const fn v_int(i: i64) -> Value {
    Value::Integer(i)
}
fn v_opt_str(s: Option<String>) -> Value {
    s.map_or(Value::Null, Value::Text)
}
fn v_opt_int(i: Option<i64>) -> Value {
    i.map_or(Value::Null, Value::Integer)
}

fn clamp_usize(n: i64) -> usize {
    usize::try_from(n).unwrap_or(0)
}

fn epoch() -> DateTime<Utc> {
    DateTime::from_timestamp(0, 0).unwrap_or_else(Utc::now)
}

fn parse_required(s: &str) -> DateTime<Utc> {
    parse_iso(s).unwrap_or_else(epoch)
}

fn parse_optional(s: Option<String>) -> Option<DateTime<Utc>> {
    s.and_then(|s| parse_iso(&s))
}

/// SQL `IN`-clause placeholder list: `n` bound `?` marks (`""` when `n == 0`).
fn placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

/// Deterministic key for near-duplicate fact detection (behavior 26).
fn normalize_fact_text(text: &str) -> String {
    let collapsed = text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let dequoted: String = collapsed
        .chars()
        .filter(|&c| c != '\'' && c != '"')
        .collect();
    dequoted
        .trim_matches(|c: char| ".,!?;:()[]{} \t\n".contains(c))
        .to_owned()
}

fn subject_key_set(subjects: &[FactSubject]) -> HashSet<String> {
    subjects.iter().map(|s| s.canonical_key.clone()).collect()
}

/// Order-preserving union of `source_turn_ids` across `facts`.
fn union_provenance<'a>(facts: impl Iterator<Item = &'a Fact>) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for fact in facts {
        for &tid in &fact.source_turn_ids {
            if !out.contains(&tid) {
                out.push(tid);
            }
        }
    }
    out
}

/// Canonical str keys from a fact row's `subjects_json` (tolerant of malformed
/// items: skips non-dict entries and non-str keys).
fn canonical_keys_from_subjects_json(subjects_json: Option<&str>) -> HashSet<String> {
    let Some(raw) = subjects_json else {
        return HashSet::new();
    };
    if raw.is_empty() {
        return HashSet::new();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return HashSet::new();
    };
    let Some(items) = value.as_array() else {
        return HashSet::new();
    };
    let mut keys = HashSet::new();
    for item in items {
        if let Some(obj) = item.as_object() {
            if let Some(key) = obj.get("canonical_key").and_then(serde_json::Value::as_str) {
                keys.insert(key.to_owned());
            }
        }
    }
    keys
}

fn parse_id_array(raw: &str) -> Vec<i64> {
    serde_json::from_str::<Vec<i64>>(raw).unwrap_or_default()
}

/// Coerce a `subjects_json` field value to a string, mirroring Python
/// `_row_to_fact`'s `str(item["canonical_key"])` / `str(item["display_at_write"])`:
/// a JSON string passes through unchanged; any other present value is stringified
/// (so `{"canonical_key": 5, ...}` yields `"5"` rather than a dropped item). Bool
/// and null follow Python's `str()` (`True`/`False`/`None`); arrays/objects are
/// absurd inputs no writer produces, so their exact rendering is not pinned.
fn subject_field_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(true) => "True".to_owned(),
        serde_json::Value::Bool(false) => "False".to_owned(),
        serde_json::Value::Null => "None".to_owned(),
        other => other.to_string(),
    }
}

fn parse_subjects(raw: Option<&str>) -> Vec<FactSubject> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return Vec::new();
    };
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    // Mirror Python `_row_to_fact`: keep any object that has BOTH keys present
    // (even with non-string values), coercing each value via `str(...)`; skip
    // non-object items and objects missing either key.
    items
        .iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            let key = obj.get("canonical_key")?;
            let disp = obj.get("display_at_write")?;
            Some(FactSubject {
                canonical_key: subject_field_to_string(key),
                display_at_write: subject_field_to_string(disp),
            })
        })
        .collect()
}

fn subjects_to_json(subjects: &[FactSubject]) -> Option<String> {
    if subjects.is_empty() {
        return None;
    }
    let arr: Vec<serde_json::Value> = subjects
        .iter()
        .map(|s| {
            serde_json::json!({
                "canonical_key": s.canonical_key,
                "display_at_write": s.display_at_write,
            })
        })
        .collect();
    Some(serde_json::Value::Array(arr).to_string())
}

/// SQL fragment + params for the `facts` validity filter (behavior 28). `now`
/// is captured per call so lexicographic text comparison stays chronological.
fn facts_validity_where(
    include_superseded: bool,
    as_of: Option<DateTime<Utc>>,
    alias: &str,
) -> (String, Vec<Value>) {
    let prefix = if alias.is_empty() {
        String::new()
    } else {
        format!("{alias}.")
    };
    if let Some(as_of) = as_of {
        let ts = iso_utc(as_of);
        let clause = format!(
            "AND ({prefix}valid_from IS NULL OR {prefix}valid_from <= ?) \
             AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)"
        );
        return (clause, vec![v_str(ts.clone()), v_str(ts)]);
    }
    let mut parts: Vec<String> = Vec::new();
    if !include_superseded {
        parts.push(format!("AND {prefix}superseded_at IS NULL"));
    }
    let now = iso_utc(Utc::now());
    parts.push(format!(
        "AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)"
    ));
    (parts.join(" "), vec![v_str(now)])
}

// ---------------------------------------------------------------------------
// Row mappers
// ---------------------------------------------------------------------------

fn map_turn(row: &Row) -> rusqlite::Result<HistoryTurn> {
    let author_platform: Option<String> = row.get("author_platform")?;
    let author_user_id: Option<String> = row.get("author_user_id")?;
    let author = match (author_platform, author_user_id) {
        (Some(platform), Some(user_id)) => Some(Author::new(
            platform,
            user_id,
            row.get("author_username")?,
            row.get("author_display_name")?,
        )),
        _ => None,
    };
    let timestamp: String = row.get("timestamp")?;
    let pings_bot: i64 = row.get("pings_bot")?;
    Ok(HistoryTurn {
        id: row.get("id")?,
        timestamp: parse_required(&timestamp),
        role: row.get("role")?,
        author,
        content: row.get("content")?,
        channel_id: row.get("channel_id")?,
        mode: row.get("mode")?,
        platform_message_id: row.get("platform_message_id")?,
        reply_to_message_id: row.get("reply_to_message_id")?,
        guild_id: row.get("guild_id")?,
        arrived_at: parse_optional(row.get("arrived_at")?),
        consumed_at: parse_optional(row.get("consumed_at")?),
        pings_bot: pings_bot != 0,
    })
}

fn map_author(row: &Row) -> rusqlite::Result<Author> {
    Ok(Author::new(
        row.get::<_, String>("author_platform")?,
        row.get::<_, String>("author_user_id")?,
        row.get("author_username")?,
        row.get("author_display_name")?,
    ))
}

fn map_fact(row: &Row) -> rusqlite::Result<Fact> {
    let source_raw: String = row.get("source_turn_ids")?;
    let subjects_raw: Option<String> = row.get("subjects_json")?;
    let created_at: String = row.get("created_at")?;
    Ok(Fact {
        id: row.get("id")?,
        familiar_id: row.get("familiar_id")?,
        channel_id: row.get("channel_id")?,
        text: row.get("text")?,
        source_turn_ids: parse_id_array(&source_raw),
        created_at: parse_required(&created_at),
        superseded_at: parse_optional(row.get("superseded_at")?),
        superseded_by: row.get("superseded_by")?,
        subjects: parse_subjects(subjects_raw.as_deref()),
        valid_from: parse_optional(row.get("valid_from")?),
        valid_to: parse_optional(row.get("valid_to")?),
        importance: row.get("importance")?,
    })
}

fn map_reflection(row: &Row) -> rusqlite::Result<Reflection> {
    let turn_raw: String = row.get("cited_turn_ids")?;
    let fact_raw: String = row.get("cited_fact_ids")?;
    let created_at: String = row.get("created_at")?;
    Ok(Reflection {
        id: row.get("id")?,
        familiar_id: row.get("familiar_id")?,
        channel_id: row.get("channel_id")?,
        text: row.get("text")?,
        cited_turn_ids: parse_id_array(&turn_raw),
        cited_fact_ids: parse_id_array(&fact_raw),
        created_at: parse_required(&created_at),
        last_turn_id: row.get("last_turn_id")?,
        last_fact_id: row.get("last_fact_id")?,
    })
}

fn map_activity(row: &Row) -> rusqlite::Result<ActivityRecord> {
    let started_at: String = row.get("started_at")?;
    let planned_return_at: String = row.get("planned_return_at")?;
    Ok(ActivityRecord {
        id: row.get("id")?,
        familiar_id: row.get("familiar_id")?,
        type_id: row.get("type_id")?,
        label: row.get("label")?,
        started_at: parse_required(&started_at),
        planned_return_at: parse_required(&planned_return_at),
        note: row.get("note")?,
        status: row.get("status")?,
        actual_return_at: parse_optional(row.get("actual_return_at")?),
        experience_text: row.get("experience_text")?,
    })
}

fn map_alarm(row: &Row) -> rusqlite::Result<AlarmRow> {
    Ok(AlarmRow {
        id: row.get("id")?,
        familiar_id: row.get("familiar_id")?,
        channel_id: row.get("channel_id")?,
        channel_kind: row.get("channel_kind")?,
        scheduled_at: row.get("scheduled_at")?,
        reason: row.get("reason")?,
        originating_turn_id: row.get("originating_turn_id")?,
        fired_at: row.get("fired_at")?,
        cancelled_at: row.get("cancelled_at")?,
        created_at: row.get("created_at")?,
    })
}

// ---------------------------------------------------------------------------
// HistoryStore
// ---------------------------------------------------------------------------

/// Persistent SQLite store for turns + every side-index projection.
pub struct HistoryStore {
    db: Db,
    fts_turns: Box<dyn FtsIndex>,
    fts_facts: Box<dyn FtsIndex>,
}

impl HistoryStore {
    /// Open (or create) a store. `":memory:"` gives a fully in-memory database
    /// plus two in-memory tantivy indexes (tests); any other path is a file
    /// whose parent dirs are created, with the FTS indexes on disk beside it at
    /// `<db_dir>/fts/turns` and `<db_dir>/fts/facts`.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = db_path.as_ref();
        if path == Path::new(":memory:") {
            let db = Db::open_memory()?;
            let fts_turns = Box::new(TantivyFts::in_memory()?);
            let fts_facts = Box::new(TantivyFts::in_memory()?);
            return Self::init(db, fts_turns, fts_facts);
        }
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(parent) = parent {
            std::fs::create_dir_all(parent)?;
        }
        let fts_root = parent.unwrap_or_else(|| Path::new(".")).join("fts");
        let (fts_turns, turns_recreated) = TantivyFts::open_dir(&fts_root.join("turns"))?;
        let (fts_facts, facts_recreated) = TantivyFts::open_dir(&fts_root.join("facts"))?;
        // An index that was wiped-and-recreated, or is otherwise empty, has lost
        // whatever the retired Python impl indexed; repopulate it from the DB.
        let turns_stale = turns_recreated || fts_turns.is_empty();
        let facts_stale = facts_recreated || fts_facts.is_empty();
        let store = Self::init(Db::open(path)?, Box::new(fts_turns), Box::new(fts_facts))?;
        store.repopulate_stale_fts(turns_stale, facts_stale)?;
        Ok(store)
    }

    /// Open a store with caller-supplied FTS indexes. The DB is set up exactly
    /// as [`open`](Self::open) (in-memory for `":memory:"`, file otherwise) but
    /// no tantivy index is created. This is the injection seam behind the
    /// "append survives an FTS failure" test (Python monkeypatched
    /// `store._fts_turns.add`).
    pub fn open_with_fts(
        db_path: impl AsRef<Path>,
        fts_turns: Box<dyn FtsIndex>,
        fts_facts: Box<dyn FtsIndex>,
    ) -> Result<Self, StoreError> {
        let path = db_path.as_ref();
        let db = if path == Path::new(":memory:") {
            Db::open_memory()?
        } else {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            Db::open(path)?
        };
        Self::init(db, fts_turns, fts_facts)
    }

    /// Wire the DB + FTS into a store, run the schema repair pass and the
    /// idempotent migrations (Python `__init__` order: schema → migrate → FTS).
    fn init(
        db: Db,
        fts_turns: Box<dyn FtsIndex>,
        fts_facts: Box<dyn FtsIndex>,
    ) -> Result<Self, StoreError> {
        let store = Self {
            db,
            fts_turns,
            fts_facts,
        };
        store.db.execute_batch(SCHEMA)?;
        store.migrate()?;
        Ok(store)
    }

    /// The DB actor handle (mirrors Python `store._conn`; test/diagnostic use).
    #[must_use]
    pub const fn conn(&self) -> &Db {
        &self.db
    }

    /// Shut down the DB actor. Idempotent; further calls error with
    /// [`StoreError::Closed`].
    pub fn close(&self) {
        self.db.close();
    }

    fn migrate(&self) -> Result<(), StoreError> {
        // arrived_at / consumed_at: backfill ONLY when the column is newly added
        // (one-time scoping — a deliberately-staged turn must survive restart).
        for col in ["arrived_at", "consumed_at"] {
            if self
                .db
                .execute(format!("ALTER TABLE turns ADD COLUMN {col} TEXT"), vec![])
                .is_ok()
            {
                self.db.execute(
                    format!("UPDATE turns SET {col} = timestamp WHERE {col} IS NULL"),
                    vec![],
                )?;
            }
        }
        // Idempotent column adds (swallow "already exists").
        let _ = self.db.execute(
            "ALTER TABLE turns ADD COLUMN pings_bot INTEGER NOT NULL DEFAULT 0",
            vec![],
        );
        let _ = self.db.execute(
            "ALTER TABLE summaries ADD COLUMN last_consumed_at TEXT",
            vec![],
        );
        let _ = self
            .db
            .execute("ALTER TABLE turns ADD COLUMN missed_at TEXT", vec![]);
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_consumed \
             ON turns (familiar_id, consumed_at, arrived_at, id)",
            vec![],
        )?;
        // Ego-key rewrite (issue #154): idempotent, whitespace-tolerant (D9).
        self.db.execute(
            "UPDATE people_dossiers SET canonical_key = 'ego:' || substr(canonical_key, 6) \
             WHERE canonical_key LIKE 'self:%'",
            vec![],
        )?;
        self.db.execute(
            "UPDATE facts SET subjects_json = \
             replace(subjects_json, '\"canonical_key\": \"self:', '\"canonical_key\": \"ego:') \
             WHERE subjects_json LIKE '%\"canonical_key\": \"self:%'",
            vec![],
        )?;
        self.db.execute(
            "UPDATE facts SET subjects_json = \
             replace(subjects_json, '\"canonical_key\":\"self:', '\"canonical_key\":\"ego:') \
             WHERE subjects_json LIKE '%\"canonical_key\":\"self:%'",
            vec![],
        )?;
        Ok(())
    }

    fn safe_fts_add(index: &dyn FtsIndex, row_id: i64, content: &str, kind: &str) {
        if let Err(err) = index.add(row_id, content) {
            tracing::warn!(target: "familiar_connect.history", kind, row_id, %err, "FTS add skipped");
        }
    }

    // -- turns -----------------------------------------------------------

    /// Append a single turn; return its persisted form. Note the QUIRK
    /// (behavior 11): the returned value leaves `guild_id` /
    /// `platform_message_id` / `reply_to_message_id` at `None` even when
    /// persisted — callers that need them re-read.
    pub fn append_turn(&self, p: AppendTurn) -> Result<HistoryTurn, StoreError> {
        let timestamp = Utc::now();
        let arrived = p.arrived_at.unwrap_or(timestamp);
        let consumed_at = if p.consumed { Some(arrived) } else { None };
        let params = vec![
            v_str(p.familiar_id.clone()),
            v_int(p.channel_id),
            v_opt_int(p.guild_id),
            v_str(p.role.clone()),
            v_opt_str(p.author.as_ref().map(|a| a.platform.clone())),
            v_opt_str(p.author.as_ref().map(|a| a.user_id.clone())),
            v_opt_str(p.author.as_ref().and_then(|a| a.username.clone())),
            v_opt_str(p.author.as_ref().and_then(|a| a.display_name.clone())),
            v_str(p.content.clone()),
            v_str(iso_utc(timestamp)),
            v_opt_str(p.mode.clone()),
            v_opt_str(p.platform_message_id.clone()),
            v_opt_str(p.reply_to_message_id.clone()),
            v_opt_str(p.tool_calls_json.clone()),
            v_opt_str(p.tool_call_id.clone()),
            v_str(iso_utc(arrived)),
            v_opt_str(consumed_at.map(iso_utc)),
            v_int(i64::from(p.pings_bot)),
        ];
        let turn_id = self.db.execute_returning_id(
            "INSERT INTO turns \
                (familiar_id, channel_id, guild_id, role, author_platform, author_user_id, \
                 author_username, author_display_name, content, timestamp, mode, \
                 platform_message_id, reply_to_message_id, tool_calls_json, tool_call_id, \
                 arrived_at, consumed_at, pings_bot) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params,
        )?;
        Self::safe_fts_add(self.fts_turns.as_ref(), turn_id, &p.content, "turn");
        Ok(HistoryTurn {
            id: turn_id,
            timestamp,
            role: p.role,
            author: p.author,
            content: p.content,
            channel_id: p.channel_id,
            mode: p.mode,
            platform_message_id: None,
            reply_to_message_id: None,
            guild_id: None,
            arrived_at: Some(arrived),
            consumed_at,
            pings_bot: p.pings_bot,
        })
    }

    /// `append_turn` with `consumed = false` (staged).
    pub fn stage_turn(&self, p: AppendTurn) -> Result<HistoryTurn, StoreError> {
        self.append_turn(p.consumed(false))
    }

    /// Find the turn carrying `platform_message_id` (highest id on duplicates).
    pub fn lookup_turn_by_platform_message_id(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
    ) -> Result<Option<HistoryTurn>, StoreError> {
        let rows = self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns \
                 WHERE familiar_id = ? AND platform_message_id = ? \
                 ORDER BY id DESC LIMIT 1"
            ),
            vec![v_str(familiar_id), v_str(platform_message_id)],
            map_turn,
        )?;
        Ok(rows.into_iter().next())
    }

    /// Rewrite `turns.content` for one platform message id; silent no-op when
    /// unmatched. Re-indexes FTS for each matched row.
    pub fn update_turn_content_by_message_id(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        content: &str,
    ) -> Result<(), StoreError> {
        let ids = self.db.query_map(
            "SELECT id FROM turns WHERE familiar_id = ? AND platform_message_id = ?",
            vec![v_str(familiar_id), v_str(platform_message_id)],
            |r| r.get::<_, i64>("id"),
        )?;
        self.db.execute(
            "UPDATE turns SET content = ? WHERE familiar_id = ? AND platform_message_id = ?",
            vec![
                v_str(content),
                v_str(familiar_id),
                v_str(platform_message_id),
            ],
        )?;
        for id in ids {
            Self::safe_fts_add(self.fts_turns.as_ref(), id, content, "turn_edit");
        }
        Ok(())
    }

    /// Fetch turns by id, scoped to `familiar_id`, oldest first (deduped input).
    pub fn turns_by_ids(
        &self,
        familiar_id: &str,
        ids: &[i64],
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        let unique: Vec<i64> = sorted_unique(ids);
        if unique.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = vec![v_str(familiar_id)];
        params.extend(unique.iter().map(|&i| v_int(i)));
        self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns \
                 WHERE familiar_id = ? AND id IN ({}) ORDER BY id ASC",
                placeholders(unique.len())
            ),
            params,
            map_turn,
        )
    }

    // -- mentions --------------------------------------------------------

    /// Record canonical keys mentioned in `turn_id` (idempotent; empty no-op).
    pub fn record_mentions(&self, turn_id: i64, canonical_keys: &[&str]) -> Result<(), StoreError> {
        let mut seen = HashSet::new();
        let unique: Vec<String> = canonical_keys
            .iter()
            .filter(|k| seen.insert((*k).to_owned()))
            .map(|k| (*k).to_owned())
            .collect();
        if unique.is_empty() {
            return Ok(());
        }
        self.db.run(move |conn| {
            let tx = conn.unchecked_transaction()?;
            for key in &unique {
                tx.execute(
                    "INSERT OR IGNORE INTO turn_mentions (turn_id, canonical_key) VALUES (?1, ?2)",
                    params![turn_id, key],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    /// Canonical keys mentioned in `turn_id`, sorted ascending.
    pub fn mentions_for_turn(&self, turn_id: i64) -> Result<Vec<String>, StoreError> {
        self.db.query_map(
            "SELECT canonical_key FROM turn_mentions WHERE turn_id = ? ORDER BY canonical_key ASC",
            vec![v_int(turn_id)],
            |r| r.get::<_, String>("canonical_key"),
        )
    }

    // -- reactions -------------------------------------------------------

    /// Upsert a reaction count; `count <= 0` deletes the row.
    pub fn set_reaction(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: &str,
        count: i64,
    ) -> Result<(), StoreError> {
        if count <= 0 {
            self.db.execute(
                "DELETE FROM message_reactions \
                 WHERE familiar_id = ? AND platform_message_id = ? AND emoji = ?",
                vec![v_str(familiar_id), v_str(platform_message_id), v_str(emoji)],
            )?;
            return Ok(());
        }
        self.db.execute(
            "INSERT INTO message_reactions \
                (familiar_id, platform_message_id, emoji, count, updated_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(familiar_id, platform_message_id, emoji) \
             DO UPDATE SET count = excluded.count, updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_str(platform_message_id),
                v_str(emoji),
                v_int(count),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Atomic ±delta on one `(message, emoji)` row; floors at zero.
    pub fn bump_reaction(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: &str,
        delta: i64,
    ) -> Result<(), StoreError> {
        if delta == 0 {
            return Ok(());
        }
        let ts = iso_utc(Utc::now());
        let (fam, pmid, em) = (
            familiar_id.to_owned(),
            platform_message_id.to_owned(),
            emoji.to_owned(),
        );
        self.db.run(move |conn| {
            let tx = conn.unchecked_transaction()?;
            let updated = tx.execute(
                "UPDATE message_reactions SET count = count + ?1, updated_at = ?2 \
                 WHERE familiar_id = ?3 AND platform_message_id = ?4 AND emoji = ?5",
                params![delta, ts, fam, pmid, em],
            )?;
            if updated == 0 && delta > 0 {
                tx.execute(
                    "INSERT INTO message_reactions \
                        (familiar_id, platform_message_id, emoji, count, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![fam, pmid, em, delta, ts],
                )?;
            }
            tx.execute(
                "DELETE FROM message_reactions \
                 WHERE familiar_id = ?1 AND platform_message_id = ?2 AND emoji = ?3 AND count <= 0",
                params![fam, pmid, em],
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Drop reactions on one message — all (`emoji = None`) or a single emoji.
    pub fn clear_reactions(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: Option<&str>,
    ) -> Result<(), StoreError> {
        match emoji {
            None => self.db.execute(
                "DELETE FROM message_reactions WHERE familiar_id = ? AND platform_message_id = ?",
                vec![v_str(familiar_id), v_str(platform_message_id)],
            )?,
            Some(emoji) => self.db.execute(
                "DELETE FROM message_reactions \
                 WHERE familiar_id = ? AND platform_message_id = ? AND emoji = ?",
                vec![v_str(familiar_id), v_str(platform_message_id), v_str(emoji)],
            )?,
        };
        Ok(())
    }

    /// Batch reaction lookup (one SQL round-trip). Per-message tuples ordered by
    /// count desc, then emoji asc. Messages with no reactions are absent.
    pub fn reactions_for_messages(
        &self,
        familiar_id: &str,
        platform_message_ids: &[&str],
    ) -> Result<HashMap<String, Vec<(String, i64)>>, StoreError> {
        let ids: Vec<&&str> = platform_message_ids
            .iter()
            .filter(|m| !m.is_empty())
            .collect();
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut params = vec![v_str(familiar_id)];
        params.extend(ids.iter().map(|m| v_str(**m)));
        let rows = self.db.query_map(
            format!(
                "SELECT platform_message_id, emoji, count FROM message_reactions \
                 WHERE familiar_id = ? AND platform_message_id IN ({}) \
                 ORDER BY platform_message_id ASC, count DESC, emoji ASC",
                placeholders(ids.len())
            ),
            params,
            |r| {
                Ok((
                    r.get::<_, String>("platform_message_id")?,
                    r.get::<_, String>("emoji")?,
                    r.get::<_, i64>("count")?,
                ))
            },
        )?;
        let mut out: HashMap<String, Vec<(String, i64)>> = HashMap::new();
        for (pmid, emoji, count) in rows {
            out.entry(pmid).or_default().push((emoji, count));
        }
        Ok(out)
    }

    // -- reads -----------------------------------------------------------

    /// Most recent turns in a channel, oldest-first.
    pub fn recent(
        &self,
        familiar_id: &str,
        channel_id: i64,
        limit: i64,
        mode: Option<&str>,
        before_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let mut where_extra = String::new();
        let mut params = vec![v_str(familiar_id), v_int(channel_id)];
        if let Some(mode) = mode {
            where_extra.push_str("AND mode = ?\n");
            params.push(v_str(mode));
        }
        if let Some(before_id) = before_id {
            where_extra.push_str("AND id < ?\n");
            params.push(v_int(before_id));
        }
        params.push(v_int(limit));
        let mut rows = self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns \
                 WHERE familiar_id = ? AND channel_id = ? {where_extra} \
                 ORDER BY id DESC LIMIT ?"
            ),
            params,
            map_turn,
        )?;
        rows.reverse();
        Ok(rows)
    }

    /// Window of turns centred on `turn_id`, oldest first (per-channel).
    pub fn turns_around(
        &self,
        familiar_id: &str,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        let before_rows = self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns \
                 WHERE familiar_id = ? AND channel_id = ? AND id < ? \
                 ORDER BY id DESC LIMIT ?"
            ),
            vec![
                v_str(familiar_id),
                v_int(channel_id),
                v_int(turn_id),
                v_int(before.max(0)),
            ],
            map_turn,
        )?;
        let anchor_after = self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns \
                 WHERE familiar_id = ? AND channel_id = ? AND id >= ? \
                 ORDER BY id ASC LIMIT ?"
            ),
            vec![
                v_str(familiar_id),
                v_int(channel_id),
                v_int(turn_id),
                v_int(after.max(0) + 1),
            ],
            map_turn,
        )?;
        let mut out: Vec<HistoryTurn> = before_rows.into_iter().rev().collect();
        out.extend(anchor_after);
        Ok(out)
    }

    /// Up to `limit` most-recently-seen distinct user authors (most recent first).
    pub fn recent_distinct_authors(
        &self,
        familiar_id: &str,
        channel_id: i64,
        limit: i64,
    ) -> Result<Vec<Author>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        self.db.query_map(
            "SELECT author_platform, author_user_id, author_username, author_display_name, \
                    MAX(id) AS max_id \
               FROM turns \
              WHERE familiar_id = ? AND channel_id = ? \
                AND author_platform IS NOT NULL AND author_user_id IS NOT NULL \
              GROUP BY author_platform, author_user_id \
              ORDER BY max_id DESC LIMIT ?",
            vec![v_str(familiar_id), v_int(channel_id), v_int(limit)],
            map_author,
        )
    }

    /// Turns with `id <= max_id`, oldest first (optionally per-channel).
    pub fn older_than(
        &self,
        familiar_id: &str,
        max_id: i64,
        channel_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        channel_id.map_or_else(
            || {
                self.db.query_map(
                    format!(
                        "SELECT {TURN_COLS} FROM turns \
                         WHERE familiar_id = ? AND id <= ? ORDER BY id ASC LIMIT ?"
                    ),
                    vec![v_str(familiar_id), v_int(max_id), v_int(limit)],
                    map_turn,
                )
            },
            |channel_id| {
                self.db.query_map(
                    format!(
                        "SELECT {TURN_COLS} FROM turns \
                         WHERE familiar_id = ? AND channel_id = ? AND id <= ? \
                         ORDER BY id ASC LIMIT ?"
                    ),
                    vec![
                        v_str(familiar_id),
                        v_int(channel_id),
                        v_int(max_id),
                        v_int(limit),
                    ],
                    map_turn,
                )
            },
        )
    }

    /// Highest turn id, or `None` when empty (optionally per-channel).
    pub fn latest_id(
        &self,
        familiar_id: &str,
        channel_id: Option<i64>,
    ) -> Result<Option<i64>, StoreError> {
        channel_id.map_or_else(
            || {
                self.db.query_scalar_i64(
                    "SELECT MAX(id) AS max_id FROM turns WHERE familiar_id = ?",
                    vec![v_str(familiar_id)],
                )
            },
            |channel_id| {
                self.db.query_scalar_i64(
                    "SELECT MAX(id) AS max_id FROM turns WHERE familiar_id = ? AND channel_id = ?",
                    vec![v_str(familiar_id), v_int(channel_id)],
                )
            },
        )
    }

    /// Count stored turns (optionally per-channel).
    pub fn count(&self, familiar_id: &str, channel_id: Option<i64>) -> Result<i64, StoreError> {
        let n = match channel_id {
            None => self.db.query_scalar_i64(
                "SELECT COUNT(*) AS n FROM turns WHERE familiar_id = ?",
                vec![v_str(familiar_id)],
            )?,
            Some(channel_id) => self.db.query_scalar_i64(
                "SELECT COUNT(*) AS n FROM turns WHERE familiar_id = ? AND channel_id = ?",
                vec![v_str(familiar_id), v_int(channel_id)],
            )?,
        };
        Ok(n.unwrap_or(0))
    }

    // -- summaries -------------------------------------------------------

    /// Cached summary for (familiar, channel), or `None`.
    pub fn get_summary(
        &self,
        familiar_id: &str,
        channel_id: i64,
    ) -> Result<Option<SummaryEntry>, StoreError> {
        let rows = self.db.query_map(
            "SELECT last_summarised_id, summary_text, created_at, last_consumed_at \
               FROM summaries WHERE familiar_id = ? AND channel_id = ?",
            vec![v_str(familiar_id), v_int(channel_id)],
            |r| {
                let created_at: String = r.get("created_at")?;
                Ok(SummaryEntry {
                    last_summarised_id: r.get("last_summarised_id")?,
                    summary_text: r.get("summary_text")?,
                    created_at: parse_required(&created_at),
                    last_consumed_at: r.get("last_consumed_at")?,
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Insert or replace the summary for (familiar, channel).
    pub fn put_summary(
        &self,
        familiar_id: &str,
        last_summarised_id: i64,
        summary_text: &str,
        channel_id: i64,
        last_consumed_at: Option<&str>,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO summaries \
                (familiar_id, channel_id, last_summarised_id, summary_text, created_at, \
                 last_consumed_at) \
             VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT (familiar_id, channel_id) DO UPDATE SET \
                 last_summarised_id = excluded.last_summarised_id, \
                 summary_text = excluded.summary_text, \
                 created_at = excluded.created_at, \
                 last_consumed_at = excluded.last_consumed_at",
            vec![
                v_str(familiar_id),
                v_int(channel_id),
                v_int(last_summarised_id),
                v_str(summary_text),
                v_str(iso_utc(Utc::now())),
                v_opt_str(last_consumed_at.map(ToOwned::to_owned)),
            ],
        )?;
        Ok(())
    }

    /// Other channels with activity, most-recently-active first.
    pub fn distinct_other_channels(
        &self,
        familiar_id: &str,
        exclude_channel_id: i64,
    ) -> Result<Vec<OtherChannelInfo>, StoreError> {
        self.db.query_map(
            "SELECT channel_id, mode, MAX(id) AS latest_id, MAX(timestamp) AS latest_ts \
               FROM turns WHERE familiar_id = ? AND channel_id != ? \
              GROUP BY channel_id ORDER BY latest_id DESC",
            vec![v_str(familiar_id), v_int(exclude_channel_id)],
            |r| {
                let latest_ts: String = r.get("latest_ts")?;
                Ok(OtherChannelInfo {
                    channel_id: r.get("channel_id")?,
                    mode: r.get("mode")?,
                    latest_id: r.get("latest_id")?,
                    latest_timestamp: parse_required(&latest_ts),
                })
            },
        )
    }

    /// All channel ids with turns for `familiar_id`.
    pub fn all_channel_ids(&self, familiar_id: &str) -> Result<HashSet<i64>, StoreError> {
        Ok(self
            .db
            .query_map(
                "SELECT DISTINCT channel_id FROM turns WHERE familiar_id = ?",
                vec![v_str(familiar_id)],
                |r| r.get::<_, i64>("channel_id"),
            )?
            .into_iter()
            .collect())
    }

    /// Turns whose id falls in `(min_id_exclusive, max_id_inclusive]`, asc.
    pub fn turns_in_id_range(
        &self,
        familiar_id: &str,
        min_id_exclusive: i64,
        max_id_inclusive: i64,
        channel_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        channel_id.map_or_else(
            || {
                self.db.query_map(
                    format!(
                        "SELECT {TURN_COLS} FROM turns \
                         WHERE familiar_id = ? AND id > ? AND id <= ? ORDER BY id ASC"
                    ),
                    vec![
                        v_str(familiar_id),
                        v_int(min_id_exclusive),
                        v_int(max_id_inclusive),
                    ],
                    map_turn,
                )
            },
            |channel_id| {
                self.db.query_map(
                    format!(
                        "SELECT {TURN_COLS} FROM turns \
                         WHERE familiar_id = ? AND channel_id = ? AND id > ? AND id <= ? \
                         ORDER BY id ASC"
                    ),
                    vec![
                        v_str(familiar_id),
                        v_int(channel_id),
                        v_int(min_id_exclusive),
                        v_int(max_id_inclusive),
                    ],
                    map_turn,
                )
            },
        )
    }

    /// All fact ids for `familiar_id`, including superseded.
    pub fn all_fact_ids(&self, familiar_id: &str) -> Result<HashSet<i64>, StoreError> {
        Ok(self
            .db
            .query_map(
                "SELECT id FROM facts WHERE familiar_id = ?",
                vec![v_str(familiar_id)],
                |r| r.get::<_, i64>("id"),
            )?
            .into_iter()
            .collect())
    }

    // -- watermarks ------------------------------------------------------

    /// Memory-writer watermark for `familiar_id`, or `None`.
    pub fn get_writer_watermark(
        &self,
        familiar_id: &str,
    ) -> Result<Option<WatermarkEntry>, StoreError> {
        let rows = self.db.query_map(
            "SELECT last_written_id, created_at FROM memory_writer_watermark WHERE familiar_id = ?",
            vec![v_str(familiar_id)],
            |r| {
                let created_at: String = r.get("created_at")?;
                Ok(WatermarkEntry {
                    last_written_id: r.get("last_written_id")?,
                    created_at: parse_required(&created_at),
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Insert or replace the memory-writer watermark.
    pub fn put_writer_watermark(
        &self,
        familiar_id: &str,
        last_written_id: i64,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO memory_writer_watermark (familiar_id, last_written_id, created_at) \
             VALUES (?, ?, ?) \
             ON CONFLICT (familiar_id) DO UPDATE SET \
                 last_written_id = excluded.last_written_id, created_at = excluded.created_at",
            vec![
                v_str(familiar_id),
                v_int(last_written_id),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Last sleep-consolidation watermark, or `None`.
    pub fn get_sleep_watermark(
        &self,
        familiar_id: &str,
    ) -> Result<Option<SleepWatermark>, StoreError> {
        let rows = self.db.query_map(
            "SELECT last_fact_id, last_turn_id, updated_at FROM sleep_watermark \
             WHERE familiar_id = ?",
            vec![v_str(familiar_id)],
            |r| {
                let updated_at: String = r.get("updated_at")?;
                Ok(SleepWatermark {
                    last_fact_id: r.get("last_fact_id")?,
                    last_turn_id: r.get("last_turn_id")?,
                    updated_at: parse_required(&updated_at),
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Advance one or both sleep-watermark axes; omitted axes are preserved
    /// (default 0 on first insert). Both `None` is a no-op.
    pub fn advance_sleep_watermark(
        &self,
        familiar_id: &str,
        last_fact_id: Option<i64>,
        last_turn_id: Option<i64>,
    ) -> Result<(), StoreError> {
        if last_fact_id.is_none() && last_turn_id.is_none() {
            return Ok(());
        }
        self.db.execute(
            "INSERT INTO sleep_watermark (familiar_id, last_fact_id, last_turn_id, updated_at) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT (familiar_id) DO UPDATE SET \
                 last_fact_id = COALESCE(?, last_fact_id), \
                 last_turn_id = COALESCE(?, last_turn_id), \
                 updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_int(last_fact_id.unwrap_or(0)),
                v_int(last_turn_id.unwrap_or(0)),
                v_str(iso_utc(Utc::now())),
                v_opt_int(last_fact_id),
                v_opt_int(last_turn_id),
            ],
        )?;
        Ok(())
    }

    /// Turns after the memory-writer watermark (0 when unset), oldest first.
    pub fn turns_since_watermark(
        &self,
        familiar_id: &str,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        let min_id = self
            .get_writer_watermark(familiar_id)?
            .map_or(0, |w| w.last_written_id);
        self.db.query_map(
            format!(
                "SELECT {TURN_COLS} FROM turns WHERE familiar_id = ? AND id > ? \
                 ORDER BY id ASC LIMIT ?"
            ),
            vec![v_str(familiar_id), v_int(min_id), v_int(limit)],
            map_turn,
        )
    }

    // -- dossiers --------------------------------------------------------

    /// Cached dossier for `canonical_key`, or `None`.
    pub fn get_people_dossier(
        &self,
        familiar_id: &str,
        canonical_key: &str,
    ) -> Result<Option<PeopleDossierEntry>, StoreError> {
        let rows = self.db.query_map(
            "SELECT canonical_key, last_fact_id, dossier_text, created_at FROM people_dossiers \
             WHERE familiar_id = ? AND canonical_key = ?",
            vec![v_str(familiar_id), v_str(canonical_key)],
            |r| {
                let created_at: String = r.get("created_at")?;
                Ok(PeopleDossierEntry {
                    canonical_key: r.get("canonical_key")?,
                    last_fact_id: r.get("last_fact_id")?,
                    dossier_text: r.get("dossier_text")?,
                    created_at: parse_required(&created_at),
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Insert or replace the dossier for `canonical_key`.
    pub fn put_people_dossier(
        &self,
        familiar_id: &str,
        canonical_key: &str,
        last_fact_id: i64,
        dossier_text: &str,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO people_dossiers \
                (familiar_id, canonical_key, last_fact_id, dossier_text, created_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT (familiar_id, canonical_key) DO UPDATE SET \
                 last_fact_id = excluded.last_fact_id, \
                 dossier_text = excluded.dossier_text, \
                 created_at = excluded.created_at",
            vec![
                v_str(familiar_id),
                v_str(canonical_key),
                v_int(last_fact_id),
                v_str(dossier_text),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Map `canonical_key` → `max(facts.id)` across current (non-superseded) facts.
    pub fn subjects_with_facts(
        &self,
        familiar_id: &str,
    ) -> Result<HashMap<String, i64>, StoreError> {
        let rows = self.db.query_map(
            "SELECT id, subjects_json FROM facts \
             WHERE familiar_id = ? AND subjects_json IS NOT NULL AND superseded_at IS NULL \
             ORDER BY id ASC",
            vec![v_str(familiar_id)],
            |r| {
                Ok((
                    r.get::<_, i64>("id")?,
                    r.get::<_, Option<String>>("subjects_json")?,
                ))
            },
        )?;
        let mut out: HashMap<String, i64> = HashMap::new();
        for (fact_id, subjects_json) in rows {
            for key in canonical_keys_from_subjects_json(subjects_json.as_deref()) {
                out.insert(key, fact_id);
            }
        }
        Ok(out)
    }

    /// Facts mentioning `canonical_key`, ascending by id.
    pub fn facts_for_subject(
        &self,
        familiar_id: &str,
        canonical_key: &str,
        min_id_exclusive: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        let (where_clause, mut vparams) = facts_validity_where(include_superseded, as_of, "");
        let mut params = vec![
            v_str(familiar_id),
            v_int(min_id_exclusive),
            v_str(format!("%\"{canonical_key}\"%")),
        ];
        params.append(&mut vparams);
        let facts = self.db.query_map(
            format!(
                "SELECT {FACT_COLS} FROM facts \
                 WHERE familiar_id = ? AND id > ? AND subjects_json IS NOT NULL \
                   AND subjects_json LIKE ? {where_clause} ORDER BY id ASC"
            ),
            params,
            map_fact,
        )?;
        Ok(facts
            .into_iter()
            .filter(|f| f.subjects.iter().any(|s| s.canonical_key == canonical_key))
            .collect())
    }

    // -- accounts / identity ---------------------------------------------

    /// Insert or refresh the canonical identity row for an author. `pronouns` /
    /// `bio` only overwrite when the new value is non-NULL (`COALESCE`).
    pub fn upsert_account(&self, author: &Author) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO accounts \
                (canonical_key, platform, user_id, username, global_name, pronouns, bio, \
                 last_seen_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT (canonical_key) DO UPDATE SET \
                 username = excluded.username, \
                 global_name = excluded.global_name, \
                 pronouns = COALESCE(excluded.pronouns, accounts.pronouns), \
                 bio = COALESCE(excluded.bio, accounts.bio), \
                 last_seen_at = excluded.last_seen_at",
            vec![
                v_str(author.canonical_key()),
                v_str(author.platform.clone()),
                v_str(author.user_id.clone()),
                v_opt_str(author.username.clone()),
                v_opt_str(author.global_name.clone()),
                v_opt_str(author.pronouns.clone()),
                v_opt_str(author.bio.clone()),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Cached profile fields for `canonical_key`, or `None`.
    pub fn get_account_profile(
        &self,
        canonical_key: &str,
    ) -> Result<Option<AccountProfile>, StoreError> {
        let key = canonical_key.to_owned();
        let rows = self.db.query_map(
            "SELECT username, global_name, pronouns, bio FROM accounts WHERE canonical_key = ?",
            vec![v_str(canonical_key)],
            move |r| {
                Ok(AccountProfile {
                    canonical_key: key.clone(),
                    username: r.get("username")?,
                    global_name: r.get("global_name")?,
                    pronouns: r.get("pronouns")?,
                    bio: r.get("bio")?,
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Cache a per-guild nickname (`nick = None` records an explicit "no override").
    pub fn upsert_guild_nick(
        &self,
        canonical_key: &str,
        guild_id: i64,
        nick: Option<&str>,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO account_guild_nicks (canonical_key, guild_id, nick, last_seen_at) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT (canonical_key, guild_id) DO UPDATE SET \
                 nick = excluded.nick, last_seen_at = excluded.last_seen_at",
            vec![
                v_str(canonical_key),
                v_int(guild_id),
                v_opt_str(nick.map(ToOwned::to_owned)),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Best display name for `canonical_key` in `guild_id` (always non-empty).
    pub fn resolve_label(
        &self,
        canonical_key: &str,
        guild_id: Option<i64>,
        familiar_id: Option<&str>,
    ) -> Result<String, StoreError> {
        if let Some(guild_id) = guild_id {
            let nick = self.db.query_scalar_string(
                "SELECT nick FROM account_guild_nicks WHERE canonical_key = ? AND guild_id = ?",
                vec![v_str(canonical_key), v_int(guild_id)],
            )?;
            if let Some(nick) = nick {
                if !nick.is_empty() {
                    return Ok(nick);
                }
            }
        }
        let account = self.db.query_map(
            "SELECT global_name, username FROM accounts WHERE canonical_key = ?",
            vec![v_str(canonical_key)],
            |r| {
                Ok((
                    r.get::<_, Option<String>>("global_name")?,
                    r.get::<_, Option<String>>("username")?,
                ))
            },
        )?;
        if let Some((global_name, username)) = account.into_iter().next() {
            if let Some(global_name) = global_name.filter(|s| !s.is_empty()) {
                return Ok(global_name);
            }
            if let Some(username) = username.filter(|s| !s.is_empty()) {
                return Ok(username);
            }
        }
        if let Some(familiar_id) = familiar_id {
            if let Some(author) = self.latest_author_for(familiar_id, canonical_key)? {
                return Ok(author.label());
            }
        }
        Ok(match canonical_key.split_once(':') {
            Some((_, tail)) if !tail.is_empty() => tail.to_owned(),
            _ => canonical_key.to_owned(),
        })
    }

    /// `Author` from the most recent turn with `canonical_key`, or `None`.
    pub fn latest_author_for(
        &self,
        familiar_id: &str,
        canonical_key: &str,
    ) -> Result<Option<Author>, StoreError> {
        let Some((platform, user_id)) = canonical_key.split_once(':') else {
            return Ok(None);
        };
        if platform.is_empty() || user_id.is_empty() {
            return Ok(None);
        }
        let rows = self.db.query_map(
            "SELECT author_platform, author_user_id, author_username, author_display_name \
               FROM turns \
              WHERE familiar_id = ? AND author_platform = ? AND author_user_id = ? \
              ORDER BY id DESC LIMIT 1",
            vec![v_str(familiar_id), v_str(platform), v_str(user_id)],
            map_author,
        )?;
        Ok(rows.into_iter().next())
    }

    // -- facts -----------------------------------------------------------

    /// Persist one fact (see [`AppendFact`]).
    pub fn append_fact(&self, p: AppendFact) -> Result<Fact, StoreError> {
        let AppendFact {
            familiar_id,
            channel_id,
            text,
            source_turn_ids,
            subjects,
            valid_from,
            valid_to,
            importance,
            dedup,
        } = p;
        let importance_eff = importance.map(|i| i.clamp(1, 10));
        let subjects_blob = subjects_to_json(&subjects);
        let source_json =
            serde_json::to_string(&source_turn_ids).unwrap_or_else(|_| "[]".to_owned());
        let norm_text = normalize_fact_text(&text);
        let subject_keys = subject_key_set(&subjects);

        let outcome = self.db.run(move |conn| {
            if dedup && valid_to.is_none() {
                if let Some(existing) =
                    find_current_dup(conn, &familiar_id, &norm_text, &subject_keys)?
                {
                    return Ok(FactInsert::Existing(existing));
                }
            }
            let now = Utc::now();
            let valid_from_eff = valid_from.unwrap_or(now);
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "INSERT INTO facts \
                    (familiar_id, channel_id, text, source_turn_ids, created_at, subjects_json, \
                     valid_from, valid_to, importance) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    familiar_id,
                    channel_id,
                    text,
                    source_json,
                    iso_utc(now),
                    subjects_blob,
                    iso_utc(valid_from_eff),
                    valid_to.map(iso_utc),
                    importance_eff,
                ],
            )?;
            let fact_id = tx.last_insert_rowid();
            tx.commit()?;
            Ok(FactInsert::Minted(Fact {
                id: fact_id,
                familiar_id,
                channel_id,
                text,
                source_turn_ids,
                created_at: now,
                superseded_at: None,
                superseded_by: None,
                subjects,
                valid_from: Some(valid_from_eff),
                valid_to,
                importance: importance_eff,
            }))
        })?;
        match outcome {
            FactInsert::Existing(fact) => Ok(fact),
            FactInsert::Minted(fact) => {
                Self::safe_fts_add(self.fts_facts.as_ref(), fact.id, &fact.text, "fact");
                Ok(fact)
            }
        }
    }

    /// Fetch facts by id (includes superseded), scoped to `familiar_id`, asc.
    pub fn facts_by_ids(&self, familiar_id: &str, ids: &[i64]) -> Result<Vec<Fact>, StoreError> {
        let unique = sorted_unique(ids);
        if unique.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = vec![v_str(familiar_id)];
        params.extend(unique.iter().map(|&i| v_int(i)));
        self.db.query_map(
            format!(
                "SELECT {FACT_COLS} FROM facts WHERE familiar_id = ? AND id IN ({}) \
                 ORDER BY id ASC",
                placeholders(unique.len())
            ),
            params,
            map_fact,
        )
    }

    /// `limit` most recent facts, newest first (current truth by default).
    pub fn recent_facts(
        &self,
        familiar_id: &str,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let (where_clause, mut vparams) = facts_validity_where(include_superseded, as_of, "");
        let mut params = vec![v_str(familiar_id)];
        params.append(&mut vparams);
        params.push(v_int(limit));
        self.db.query_map(
            format!(
                "SELECT {FACT_COLS} FROM facts WHERE familiar_id = ? {where_clause} \
                 ORDER BY id DESC LIMIT ?"
            ),
            params,
            map_fact,
        )
    }

    /// Highest `facts.id` for `familiar_id` (0 if none; counts superseded).
    pub fn latest_fact_id(&self, familiar_id: &str) -> Result<i64, StoreError> {
        Ok(self
            .db
            .query_scalar_i64(
                "SELECT MAX(id) AS max_id FROM facts WHERE familiar_id = ?",
                vec![v_str(familiar_id)],
            )?
            .unwrap_or(0))
    }

    /// Facts directly superseded by `fact_id` (its merge ancestors), asc.
    pub fn ancestors_of(&self, familiar_id: &str, fact_id: i64) -> Result<Vec<Fact>, StoreError> {
        self.db.query_map(
            format!(
                "SELECT {FACT_COLS} FROM facts WHERE familiar_id = ? AND superseded_by = ? \
                 ORDER BY id ASC"
            ),
            vec![v_str(familiar_id), v_int(fact_id)],
            map_fact,
        )
    }

    /// Subset of `fact_ids` that are superseded (empty input short-circuits).
    pub fn superseded_fact_ids(
        &self,
        familiar_id: &str,
        fact_ids: &[i64],
    ) -> Result<HashSet<i64>, StoreError> {
        if fact_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let mut params = vec![v_str(familiar_id)];
        params.extend(fact_ids.iter().map(|&i| v_int(i)));
        Ok(self
            .db
            .query_map(
                format!(
                    "SELECT id FROM facts WHERE familiar_id = ? AND id IN ({}) \
                     AND superseded_at IS NOT NULL",
                    placeholders(fact_ids.len())
                ),
                params,
                |r| r.get::<_, i64>("id"),
            )?
            .into_iter()
            .collect())
    }

    /// Unified mutation: retire, merge, or repoint obsolete facts (behavior 31).
    pub fn supersede(
        &self,
        familiar_id: &str,
        obsolete_facts: &[i64],
        new_fact: NewFact,
    ) -> Result<SupersedeResult, StoreError> {
        let ids: Vec<i64> = obsolete_facts.to_vec();
        match new_fact {
            NewFact::Merge(draft) => self.merge_atomically(familiar_id, &ids, draft),
            NewFact::Retire => self.retire_or_repoint(familiar_id, &ids, None),
            NewFact::Repoint(id) => self.retire_or_repoint(familiar_id, &ids, Some(id)),
        }
    }

    fn retire_or_repoint(
        &self,
        familiar_id: &str,
        ids: &[i64],
        new_id: Option<i64>,
    ) -> Result<SupersedeResult, StoreError> {
        let obsolete = self.facts_by_ids(familiar_id, ids)?;
        let by_id: HashMap<i64, Fact> = obsolete.into_iter().map(|f| (f.id, f)).collect();
        let ids = ids.to_vec();
        let fam = familiar_id.to_owned();
        self.db.run(move |conn| {
            let now = iso_utc(Utc::now());
            let mut superseded: Vec<i64> = Vec::new();
            let mut skipped: Vec<(i64, String)> = Vec::new();
            let tx = conn.unchecked_transaction()?;
            for fid in &ids {
                let Some(row) = by_id.get(fid) else {
                    skipped.push((*fid, format!("unknown fact id={fid}")));
                    continue;
                };
                if row.superseded_at.is_some() {
                    skipped.push((*fid, format!("fact id={fid} already superseded")));
                    continue;
                }
                tx.execute(
                    "UPDATE facts SET superseded_at = ?1, superseded_by = ?2 \
                     WHERE id = ?3 AND familiar_id = ?4",
                    params![now, new_id, fid, fam],
                )?;
                for key in subject_key_set(&row.subjects) {
                    tx.execute(
                        "DELETE FROM people_dossiers WHERE familiar_id = ?1 AND canonical_key = ?2",
                        params![fam, key],
                    )?;
                }
                superseded.push(*fid);
            }
            tx.commit()?;
            Ok(SupersedeResult {
                minted: None,
                superseded,
                skipped,
            })
        })
    }

    fn merge_atomically(
        &self,
        familiar_id: &str,
        ids: &[i64],
        draft: FactDraft,
    ) -> Result<SupersedeResult, StoreError> {
        let obsolete = self.facts_by_ids(familiar_id, ids)?;
        let by_id: HashMap<i64, Fact> = obsolete.into_iter().map(|f| (f.id, f)).collect();
        // Pre-flight: decline whole if empty or any id unknown/already superseded.
        let mut stale: Vec<(i64, String)> = Vec::new();
        for fid in ids {
            match by_id.get(fid) {
                None => stale.push((*fid, format!("unknown fact id={fid}"))),
                Some(row) if row.superseded_at.is_some() => {
                    stale.push((*fid, format!("fact id={fid} already superseded")));
                }
                Some(_) => {}
            }
        }
        if ids.is_empty() || !stale.is_empty() {
            return Ok(SupersedeResult {
                minted: None,
                superseded: Vec::new(),
                skipped: stale,
            });
        }

        let provenance = union_provenance(ids.iter().filter_map(|fid| by_id.get(fid)));
        let ordered_ids = ids.to_vec();
        let fam = familiar_id.to_owned();
        let source_json = serde_json::to_string(&provenance).unwrap_or_else(|_| "[]".to_owned());
        let subjects_blob = subjects_to_json(&draft.subjects);
        let minted = self.db.run(move |conn| {
            let now = Utc::now();
            let now_iso = iso_utc(now);
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "INSERT INTO facts \
                    (familiar_id, channel_id, text, source_turn_ids, created_at, subjects_json, \
                     valid_from, valid_to, importance) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    fam,
                    draft.channel_id,
                    draft.text,
                    source_json,
                    now_iso,
                    subjects_blob,
                    now_iso,
                    Option::<String>::None,
                    Option::<i64>::None,
                ],
            )?;
            let minted_id = tx.last_insert_rowid();
            for fid in &ordered_ids {
                tx.execute(
                    "UPDATE facts SET superseded_at = ?1, superseded_by = ?2 \
                     WHERE id = ?3 AND familiar_id = ?4",
                    params![now_iso, minted_id, fid, fam],
                )?;
                if let Some(row) = by_id.get(fid) {
                    for key in subject_key_set(&row.subjects) {
                        tx.execute(
                            "DELETE FROM people_dossiers \
                             WHERE familiar_id = ?1 AND canonical_key = ?2",
                            params![fam, key],
                        )?;
                    }
                }
            }
            tx.commit()?;
            Ok(Fact {
                id: minted_id,
                familiar_id: fam,
                channel_id: draft.channel_id,
                text: draft.text,
                source_turn_ids: provenance,
                created_at: now,
                superseded_at: None,
                superseded_by: None,
                subjects: draft.subjects,
                valid_from: Some(now),
                valid_to: None,
                importance: None,
            })
        })?;
        Self::safe_fts_add(self.fts_facts.as_ref(), minted.id, &minted.text, "fact");
        Ok(SupersedeResult {
            superseded: ids.to_vec(),
            minted: Some(minted),
            skipped: Vec::new(),
        })
    }

    // -- fact embeddings -------------------------------------------------

    /// Persist `vector` for `(fact_id, model)` as packed little-endian f32.
    pub fn set_fact_embedding(
        &self,
        fact_id: i64,
        model: &str,
        vector: &[f32],
    ) -> Result<(), StoreError> {
        if vector.is_empty() {
            return Err(StoreError::EmptyVector);
        }
        let dim = i64::try_from(vector.len()).unwrap_or(i64::MAX);
        let mut blob = Vec::with_capacity(vector.len() * 4);
        for &x in vector {
            byteorder::WriteBytesExt::write_f32::<byteorder::LittleEndian>(&mut blob, x)
                .expect("writing to a Vec cannot fail");
        }
        self.db.execute(
            "INSERT INTO fact_embeddings (fact_id, model, dim, vector, created_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT (fact_id, model) DO UPDATE SET \
                 dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at",
            vec![
                v_int(fact_id),
                v_str(model),
                v_int(dim),
                Value::Blob(blob),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// `{fact_id: vector}` for the requested ids + model (missing ids absent).
    pub fn get_fact_embeddings(
        &self,
        fact_ids: &[i64],
        model: &str,
    ) -> Result<HashMap<i64, Vec<f32>>, StoreError> {
        if fact_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut params = vec![v_str(model)];
        params.extend(fact_ids.iter().map(|&i| v_int(i)));
        let rows = self.db.query_map(
            format!(
                "SELECT fact_id, dim, vector FROM fact_embeddings \
                 WHERE model = ? AND fact_id IN ({})",
                placeholders(fact_ids.len())
            ),
            params,
            |r| {
                Ok((
                    r.get::<_, i64>("fact_id")?,
                    r.get::<_, i64>("dim")?,
                    r.get::<_, Vec<u8>>("vector")?,
                ))
            },
        )?;
        let mut out: HashMap<i64, Vec<f32>> = HashMap::new();
        for (fact_id, dim, blob) in rows {
            let n = clamp_usize(dim);
            let mut cursor = std::io::Cursor::new(blob);
            let mut vec = Vec::with_capacity(n);
            for _ in 0..n {
                match byteorder::ReadBytesExt::read_f32::<byteorder::LittleEndian>(&mut cursor) {
                    Ok(x) => vec.push(x),
                    Err(_) => break,
                }
            }
            out.insert(fact_id, vec);
        }
        Ok(out)
    }

    /// Current facts lacking an embedding row for `model`, ascending by id.
    pub fn unembedded_facts(
        &self,
        familiar_id: &str,
        model: &str,
        limit: i64,
    ) -> Result<Vec<Fact>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        self.db.query_map(
            format!(
                "SELECT {FACT_COLS_F} FROM facts AS f \
                 LEFT JOIN fact_embeddings AS fe ON fe.fact_id = f.id AND fe.model = ? \
                 WHERE f.familiar_id = ? AND f.superseded_at IS NULL AND fe.fact_id IS NULL \
                 ORDER BY f.id ASC LIMIT ?"
            ),
            vec![v_str(model), v_str(familiar_id), v_int(limit)],
            map_fact,
        )
    }

    /// Highest `fact_id` with an embedding row for `model` (0 if none).
    pub fn latest_embedded_fact_id(
        &self,
        familiar_id: &str,
        model: &str,
    ) -> Result<i64, StoreError> {
        Ok(self
            .db
            .query_scalar_i64(
                "SELECT MAX(fe.fact_id) AS max_id FROM fact_embeddings AS fe \
                 JOIN facts AS f ON f.id = fe.fact_id \
                 WHERE fe.model = ? AND f.familiar_id = ?",
                vec![v_str(model), v_str(familiar_id)],
            )?
            .unwrap_or(0))
    }

    // -- reflections -----------------------------------------------------

    /// Insert a new reflection row.
    #[allow(clippy::too_many_arguments)]
    pub fn append_reflection(
        &self,
        familiar_id: &str,
        channel_id: Option<i64>,
        text: &str,
        cited_turn_ids: &[i64],
        cited_fact_ids: &[i64],
        last_turn_id: i64,
        last_fact_id: i64,
    ) -> Result<Reflection, StoreError> {
        let turn_ids: Vec<i64> = cited_turn_ids.to_vec();
        let fact_ids: Vec<i64> = cited_fact_ids.to_vec();
        let created = Utc::now();
        let id = self.db.execute_returning_id(
            "INSERT INTO reflections \
                (familiar_id, channel_id, text, cited_turn_ids, cited_fact_ids, created_at, \
                 last_turn_id, last_fact_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                v_str(familiar_id),
                v_opt_int(channel_id),
                v_str(text),
                v_str(serde_json::to_string(&turn_ids).unwrap_or_else(|_| "[]".to_owned())),
                v_str(serde_json::to_string(&fact_ids).unwrap_or_else(|_| "[]".to_owned())),
                v_str(iso_utc(created)),
                v_int(last_turn_id),
                v_int(last_fact_id),
            ],
        )?;
        Ok(Reflection {
            id,
            familiar_id: familiar_id.to_owned(),
            channel_id,
            text: text.to_owned(),
            cited_turn_ids: turn_ids,
            cited_fact_ids: fact_ids,
            created_at: created,
            last_turn_id,
            last_fact_id,
        })
    }

    /// `limit` most recent reflections, newest first. `channel_id = None`
    /// returns every channel (including channel-agnostic rows); a set channel
    /// includes channel-agnostic rows too.
    pub fn recent_reflections(
        &self,
        familiar_id: &str,
        channel_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<Reflection>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        channel_id.map_or_else(
            || {
                self.db.query_map(
                    format!(
                        "SELECT {REFLECTION_COLS} FROM reflections WHERE familiar_id = ? \
                         ORDER BY id DESC LIMIT ?"
                    ),
                    vec![v_str(familiar_id), v_int(limit)],
                    map_reflection,
                )
            },
            |channel_id| {
                self.db.query_map(
                    format!(
                        "SELECT {REFLECTION_COLS} FROM reflections \
                         WHERE familiar_id = ? AND (channel_id = ? OR channel_id IS NULL) \
                         ORDER BY id DESC LIMIT ?"
                    ),
                    vec![v_str(familiar_id), v_int(channel_id), v_int(limit)],
                    map_reflection,
                )
            },
        )
    }

    /// `(last_turn_id, last_fact_id)` the reflection worker last processed.
    pub fn latest_reflection_watermarks(
        &self,
        familiar_id: &str,
    ) -> Result<(i64, i64), StoreError> {
        let wm = self.db.query_map(
            "SELECT last_turn_id, last_fact_id FROM reflection_watermark WHERE familiar_id = ?",
            vec![v_str(familiar_id)],
            |r| {
                Ok((
                    r.get::<_, i64>("last_turn_id")?,
                    r.get::<_, i64>("last_fact_id")?,
                ))
            },
        )?;
        if let Some(pair) = wm.into_iter().next() {
            return Ok(pair);
        }
        let row = self.db.query_map(
            "SELECT last_turn_id, last_fact_id FROM reflections WHERE familiar_id = ? \
             ORDER BY id DESC LIMIT 1",
            vec![v_str(familiar_id)],
            |r| {
                Ok((
                    r.get::<_, i64>("last_turn_id")?,
                    r.get::<_, i64>("last_fact_id")?,
                ))
            },
        )?;
        Ok(row.into_iter().next().unwrap_or((0, 0)))
    }

    /// Upsert the reflection watermark unconditionally (called every tick).
    pub fn set_reflection_watermark(
        &self,
        familiar_id: &str,
        last_turn_id: i64,
        last_fact_id: i64,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO reflection_watermark (familiar_id, last_turn_id, last_fact_id, updated_at) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT (familiar_id) DO UPDATE SET \
                 last_turn_id = excluded.last_turn_id, \
                 last_fact_id = excluded.last_fact_id, \
                 updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_int(last_turn_id),
                v_int(last_fact_id),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    // -- alarms ----------------------------------------------------------

    /// Insert a new alarm row; return its id (`uuid4().hex`).
    pub fn insert_alarm(
        &self,
        familiar_id: &str,
        channel_id: i64,
        channel_kind: &str,
        scheduled_at: &str,
        reason: &str,
        originating_turn_id: Option<&str>,
    ) -> Result<String, StoreError> {
        let alarm_id = uuid::Uuid::new_v4().simple().to_string();
        self.db.execute(
            "INSERT INTO alarms \
                (id, familiar_id, channel_id, channel_kind, scheduled_at, reason, \
                 originating_turn_id, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                v_str(alarm_id.clone()),
                v_str(familiar_id),
                v_int(channel_id),
                v_str(channel_kind),
                v_str(scheduled_at),
                v_str(reason),
                v_opt_str(originating_turn_id.map(ToOwned::to_owned)),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(alarm_id)
    }

    /// Pending alarms (not fired, not cancelled), ordered `scheduled_at` asc.
    pub fn list_pending_alarms(&self, familiar_id: &str) -> Result<Vec<AlarmRow>, StoreError> {
        self.db.query_map(
            "SELECT id, familiar_id, channel_id, channel_kind, scheduled_at, reason, \
                    originating_turn_id, fired_at, cancelled_at, created_at \
               FROM alarms \
              WHERE familiar_id = ? AND fired_at IS NULL AND cancelled_at IS NULL \
              ORDER BY scheduled_at ASC",
            vec![v_str(familiar_id)],
            map_alarm,
        )
    }

    /// Stamp `fired_at`; `true` if a row changed.
    pub fn mark_alarm_fired(&self, alarm_id: &str, fired_at: &str) -> Result<bool, StoreError> {
        let n = self.db.execute(
            "UPDATE alarms SET fired_at = ? \
             WHERE id = ? AND fired_at IS NULL AND cancelled_at IS NULL",
            vec![v_str(fired_at), v_str(alarm_id)],
        )?;
        Ok(n > 0)
    }

    /// Stamp `cancelled_at`; `true` if a row changed.
    pub fn cancel_alarm(&self, alarm_id: &str, cancelled_at: &str) -> Result<bool, StoreError> {
        let n = self.db.execute(
            "UPDATE alarms SET cancelled_at = ? \
             WHERE id = ? AND fired_at IS NULL AND cancelled_at IS NULL",
            vec![v_str(cancelled_at), v_str(alarm_id)],
        )?;
        Ok(n > 0)
    }

    // -- attentional stream ---------------------------------------------

    /// Promote the catch-up window of staged turns in one channel; miss the rest.
    pub fn promote_staged_turns(
        &self,
        familiar_id: &str,
        channel_id: i64,
        catch_up_limit: Option<usize>,
    ) -> Result<Promotion, StoreError> {
        let limit = catch_up_limit.unwrap_or(DEFAULT_CATCH_UP_LIMIT);
        let rows = self.db.query_map(
            "SELECT id, pings_bot FROM turns \
             WHERE familiar_id = ? AND channel_id = ? \
               AND consumed_at IS NULL AND missed_at IS NULL \
             ORDER BY arrived_at DESC, id DESC",
            vec![v_str(familiar_id), v_int(channel_id)],
            |r| Ok((r.get::<_, i64>("id")?, r.get::<_, i64>("pings_bot")?, 0_i64)),
        )?;
        self.resolve_promotion(&rows, limit, false)
    }

    /// Promote a per-channel catch-up window of absence backlog (`id > after`).
    pub fn promote_staged_turns_since(
        &self,
        familiar_id: &str,
        after_turn_id: i64,
        catch_up_limit: Option<usize>,
    ) -> Result<Promotion, StoreError> {
        let limit = catch_up_limit.unwrap_or(DEFAULT_CATCH_UP_LIMIT);
        let rows = self.db.query_map(
            "SELECT id, channel_id, pings_bot FROM turns \
             WHERE familiar_id = ? AND consumed_at IS NULL AND missed_at IS NULL AND id > ? \
             ORDER BY channel_id ASC, arrived_at DESC, id DESC",
            vec![v_str(familiar_id), v_int(after_turn_id)],
            |r| {
                Ok((
                    r.get::<_, i64>("id")?,
                    r.get::<_, i64>("pings_bot")?,
                    r.get::<_, i64>("channel_id")?,
                ))
            },
        )?;
        self.resolve_promotion(&rows, limit, true)
    }

    /// Split staged `rows` (`(id, pings_bot, channel_id)`, newest-first) into
    /// consume/miss sets and stamp them in one transaction.
    fn resolve_promotion(
        &self,
        rows: &[(i64, i64, i64)],
        catch_up_limit: usize,
        per_channel: bool,
    ) -> Result<Promotion, StoreError> {
        let mut consume_ids: Vec<i64> = Vec::new();
        let mut miss_ids: Vec<i64> = Vec::new();
        let mut seen_per_channel: HashMap<i64, usize> = HashMap::new();
        for &(id, pings_bot, channel_id) in rows {
            let rank = if per_channel {
                let entry = seen_per_channel.entry(channel_id).or_insert(0);
                let rank = *entry;
                *entry += 1;
                rank
            } else {
                consume_ids.len() + miss_ids.len()
            };
            let within_window = rank < catch_up_limit;
            let is_ping = pings_bot == 1;
            if within_window || is_ping {
                consume_ids.push(id);
            } else {
                miss_ids.push(id);
            }
        }
        let now = iso_utc(Utc::now());
        let consumed = consume_ids.len();
        let missed = miss_ids.len();
        self.db.run(move |conn| {
            let tx = conn.unchecked_transaction()?;
            stamp_turns(&tx, "consumed_at", &consume_ids, &now)?;
            stamp_turns(&tx, "missed_at", &miss_ids, &now)?;
            tx.commit()?;
            Ok(())
        })?;
        Ok(Promotion { consumed, missed })
    }

    /// Count still-staged turns (`consumed_at IS NULL AND missed_at IS NULL`).
    pub fn count_staged(&self, familiar_id: &str, channel_id: i64) -> Result<i64, StoreError> {
        Ok(self
            .db
            .query_scalar_i64(
                "SELECT COUNT(*) AS n FROM turns \
                 WHERE familiar_id = ? AND channel_id = ? \
                   AND consumed_at IS NULL AND missed_at IS NULL",
                vec![v_str(familiar_id), v_int(channel_id)],
            )?
            .unwrap_or(0))
    }

    /// Map channel_id → [`ChannelUnread`] for staged channels.
    pub fn staged_channels(
        &self,
        familiar_id: &str,
    ) -> Result<HashMap<i64, ChannelUnread>, StoreError> {
        let rows = self.db.query_map(
            "SELECT channel_id, COUNT(*) AS n, COALESCE(SUM(pings_bot), 0) AS pings \
               FROM turns \
              WHERE familiar_id = ? AND consumed_at IS NULL AND missed_at IS NULL \
              GROUP BY channel_id",
            vec![v_str(familiar_id)],
            |r| {
                Ok((
                    r.get::<_, i64>("channel_id")?,
                    r.get::<_, i64>("n")?,
                    r.get::<_, i64>("pings")?,
                ))
            },
        )?;
        Ok(rows
            .into_iter()
            .map(|(cid, n, pings)| (cid, ChannelUnread(n, pings)))
            .collect())
    }

    /// Last `limit` consumed turns across all channels, oldest-first.
    pub fn recent_cross_channel(
        &self,
        familiar_id: &str,
        limit: i64,
        respect_archive: bool,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let inner = format!(
            "SELECT {TURN_COLS}, familiar_id FROM turns \
             WHERE familiar_id = ? AND consumed_at IS NOT NULL \
             ORDER BY arrived_at DESC, id DESC LIMIT ?"
        );
        let sql = if respect_archive {
            format!(
                "SELECT * FROM ({inner}) AS t \
                 WHERE t.id > COALESCE( \
                     (SELECT turn_id FROM channel_archive_watermark w \
                       WHERE w.familiar_id = t.familiar_id AND w.channel_id = t.channel_id), 0) \
                 ORDER BY t.arrived_at DESC, t.id DESC"
            )
        } else {
            inner
        };
        let mut rows = self
            .db
            .query_map(sql, vec![v_str(familiar_id), v_int(limit)], map_turn)?;
        rows.reverse();
        Ok(rows)
    }

    /// Consumed turns past the `(consumed_at, id)` composite cursor, in order.
    /// Empty `after_consumed_at` matches all consumed turns (cold start).
    pub fn consumed_turns_after(
        &self,
        familiar_id: &str,
        after_consumed_at: &str,
        after_id: i64,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        self.db.query_map(
            format!(
                "SELECT {TURN_COLS}, familiar_id FROM turns \
                 WHERE familiar_id = ? AND consumed_at IS NOT NULL \
                   AND (consumed_at > ? OR (consumed_at = ? AND id > ?)) \
                 ORDER BY consumed_at ASC, id ASC LIMIT ?"
            ),
            vec![
                v_str(familiar_id),
                v_str(after_consumed_at),
                v_str(after_consumed_at),
                v_int(after_id),
                v_int(limit),
            ],
            map_turn,
        )
    }

    // -- focus pointers / digest watermark -------------------------------

    /// Current focus pointers for `familiar_id`, or `None`.
    pub fn get_focus_pointers(
        &self,
        familiar_id: &str,
    ) -> Result<Option<FocusPointers>, StoreError> {
        let rows = self.db.query_map(
            "SELECT text_channel_id, voice_channel_id, updated_at FROM focus_pointers \
             WHERE familiar_id = ?",
            vec![v_str(familiar_id)],
            |r| {
                let updated_at: String = r.get("updated_at")?;
                Ok(FocusPointers {
                    text_channel_id: r.get("text_channel_id")?,
                    voice_channel_id: r.get("voice_channel_id")?,
                    updated_at: parse_required(&updated_at),
                })
            },
        )?;
        Ok(rows.into_iter().next())
    }

    /// Upsert focus pointers for `familiar_id`.
    pub fn set_focus_pointers(
        &self,
        familiar_id: &str,
        text_channel_id: Option<i64>,
        voice_channel_id: Option<i64>,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO focus_pointers (familiar_id, text_channel_id, voice_channel_id, updated_at) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT (familiar_id) DO UPDATE SET \
                 text_channel_id = excluded.text_channel_id, \
                 voice_channel_id = excluded.voice_channel_id, \
                 updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_opt_int(text_channel_id),
                v_opt_int(voice_channel_id),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Max `arrived_at` of the last-delivered digest, or `None`.
    pub fn get_digest_watermark(
        &self,
        familiar_id: &str,
    ) -> Result<Option<DateTime<Utc>>, StoreError> {
        let raw = self.db.query_scalar_string(
            "SELECT watermark_at FROM unread_digest_watermark WHERE familiar_id = ?",
            vec![v_str(familiar_id)],
        )?;
        Ok(raw.map(|s| parse_required(&s)))
    }

    /// Upsert the digest watermark for `familiar_id`.
    pub fn set_digest_watermark(
        &self,
        familiar_id: &str,
        watermark_at: DateTime<Utc>,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO unread_digest_watermark (familiar_id, watermark_at, updated_at) \
             VALUES (?, ?, ?) \
             ON CONFLICT (familiar_id) DO UPDATE SET \
                 watermark_at = excluded.watermark_at, updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_str(iso_utc(watermark_at)),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    // -- archive watermark ----------------------------------------------

    /// Upsert the archive watermark for (familiar, channel).
    pub fn set_archive_watermark(
        &self,
        familiar_id: &str,
        channel_id: i64,
        turn_id: i64,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO channel_archive_watermark (familiar_id, channel_id, turn_id, updated_at) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT (familiar_id, channel_id) DO UPDATE SET \
                 turn_id = excluded.turn_id, updated_at = excluded.updated_at",
            vec![
                v_str(familiar_id),
                v_int(channel_id),
                v_int(turn_id),
                v_str(iso_utc(Utc::now())),
            ],
        )?;
        Ok(())
    }

    /// Upsert the archive watermark for every channel with turns.
    pub fn set_archive_watermark_all(
        &self,
        familiar_id: &str,
        turn_id: i64,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO channel_archive_watermark (familiar_id, channel_id, turn_id, updated_at) \
             SELECT DISTINCT familiar_id, channel_id, ?, ? FROM turns WHERE familiar_id = ? \
             ON CONFLICT (familiar_id, channel_id) DO UPDATE SET \
                 turn_id = excluded.turn_id, updated_at = excluded.updated_at",
            vec![
                v_int(turn_id),
                v_str(iso_utc(Utc::now())),
                v_str(familiar_id),
            ],
        )?;
        Ok(())
    }

    /// Highest turn id with `timestamp <= ts`, across channels (or `None`).
    pub fn latest_id_at_or_before(
        &self,
        familiar_id: &str,
        ts: DateTime<Utc>,
    ) -> Result<Option<i64>, StoreError> {
        self.db.query_scalar_i64(
            "SELECT MAX(id) AS max_id FROM turns WHERE familiar_id = ? AND timestamp <= ?",
            vec![v_str(familiar_id), v_str(iso_utc(ts))],
        )
    }

    /// Archive watermark turn id for (familiar, channel), or `None`.
    pub fn get_archive_watermark(
        &self,
        familiar_id: &str,
        channel_id: i64,
    ) -> Result<Option<i64>, StoreError> {
        self.db.query_scalar_i64(
            "SELECT turn_id FROM channel_archive_watermark WHERE familiar_id = ? AND channel_id = ?",
            vec![v_str(familiar_id), v_int(channel_id)],
        )
    }

    // -- activities ------------------------------------------------------

    /// Insert an activity row; return its id.
    pub fn create_activity(
        &self,
        familiar_id: &str,
        type_id: &str,
        label: &str,
        started_at: DateTime<Utc>,
        planned_return_at: DateTime<Utc>,
        note: Option<&str>,
    ) -> Result<i64, StoreError> {
        self.db.execute_returning_id(
            "INSERT INTO activities \
                (familiar_id, type_id, label, started_at, planned_return_at, note) \
             VALUES (?, ?, ?, ?, ?, ?)",
            vec![
                v_str(familiar_id),
                v_str(type_id),
                v_str(label),
                v_str(iso_utc(started_at)),
                v_str(iso_utc(planned_return_at)),
                v_opt_str(note.map(ToOwned::to_owned)),
            ],
        )
    }

    /// Stamp return fields on one activity (status must be
    /// `"completed"`/`"cut_short"`).
    pub fn finish_activity(
        &self,
        activity_id: i64,
        status: &str,
        actual_return_at: DateTime<Utc>,
        experience_text: Option<&str>,
    ) -> Result<(), StoreError> {
        if status != "completed" && status != "cut_short" {
            return Err(StoreError::InvalidActivityStatus(status.to_owned()));
        }
        self.db.execute(
            "UPDATE activities SET status = ?, actual_return_at = ?, experience_text = ? \
             WHERE id = ?",
            vec![
                v_str(status),
                v_str(iso_utc(actual_return_at)),
                v_opt_str(experience_text.map(ToOwned::to_owned)),
                v_int(activity_id),
            ],
        )?;
        Ok(())
    }

    /// Persist experience prose on one activity (idempotent).
    pub fn set_activity_experience(
        &self,
        activity_id: i64,
        experience_text: &str,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "UPDATE activities SET experience_text = ? WHERE id = ?",
            vec![v_str(experience_text), v_int(activity_id)],
        )?;
        Ok(())
    }

    /// Newest activity with `actual_return_at IS NULL`, or `None`.
    pub fn active_activity(&self, familiar_id: &str) -> Result<Option<ActivityRecord>, StoreError> {
        let rows = self.db.query_map(
            format!(
                "SELECT {ACTIVITY_COLS} FROM activities \
                 WHERE familiar_id = ? AND actual_return_at IS NULL ORDER BY id DESC LIMIT 1"
            ),
            vec![v_str(familiar_id)],
            map_activity,
        )?;
        Ok(rows.into_iter().next())
    }

    /// Newest activity of `type_id` (active or finished), or `None`.
    pub fn latest_activity(
        &self,
        familiar_id: &str,
        type_id: &str,
    ) -> Result<Option<ActivityRecord>, StoreError> {
        let rows = self.db.query_map(
            format!(
                "SELECT {ACTIVITY_COLS} FROM activities \
                 WHERE familiar_id = ? AND type_id = ? ORDER BY id DESC LIMIT 1"
            ),
            vec![v_str(familiar_id), v_str(type_id)],
            map_activity,
        )?;
        Ok(rows.into_iter().next())
    }

    // -- FTS-backed reads ------------------------------------------------

    /// Search `turns.content` via FTS `query`, re-ranked and joined back.
    pub fn search_turns(
        &self,
        familiar_id: &str,
        query: &str,
        limit: i64,
        channel_id: Option<i64>,
        max_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let fts_limit = clamp_usize((limit * 4).max(limit));
        let hits = self.fts_turns.search(query, fts_limit);
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let score_by_id: HashMap<i64, f32> = hits.into_iter().collect();
        let candidate_ids: Vec<i64> = score_by_id.keys().copied().collect();
        let mut params = vec![v_str(familiar_id)];
        params.extend(candidate_ids.iter().map(|&i| v_int(i)));
        let mut where_extra = String::new();
        if let Some(channel_id) = channel_id {
            where_extra.push_str("AND t.channel_id = ?\n");
            params.push(v_int(channel_id));
        }
        if let Some(max_id) = max_id {
            where_extra.push_str("AND t.id <= ?\n");
            params.push(v_int(max_id));
        }
        let mut turns = self.db.query_map(
            format!(
                "SELECT {TURN_COLS_T} FROM turns AS t \
                 WHERE t.familiar_id = ? AND t.id IN ({}) {where_extra}",
                placeholders(candidate_ids.len())
            ),
            params,
            map_turn,
        )?;
        turns.sort_by(|a, b| {
            let sa = score_by_id.get(&a.id).copied().unwrap_or(0.0);
            let sb = score_by_id.get(&b.id).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.id.cmp(&a.id))
        });
        turns.truncate(clamp_usize(limit));
        Ok(turns)
    }

    /// Drop and repopulate the tantivy turns index from `turns`.
    pub fn rebuild_fts(&self) -> Result<(), StoreError> {
        self.rebuild_turns_fts().map(|_| ())
    }

    /// Clear and repopulate the turns index from the `turns` table; returns the
    /// number of rows indexed.
    fn rebuild_turns_fts(&self) -> Result<usize, StoreError> {
        self.fts_turns.clear()?;
        let rows = self.db.query_map(
            "SELECT id, content FROM turns ORDER BY id ASC",
            vec![],
            |r| Ok((r.get::<_, i64>("id")?, r.get::<_, String>("content")?)),
        )?;
        let count = rows.len();
        self.fts_turns.add_many(&rows)?;
        Ok(count)
    }

    /// Clear and repopulate the facts index from the `facts` table; returns the
    /// number of rows indexed.
    fn rebuild_facts_fts(&self) -> Result<usize, StoreError> {
        self.fts_facts.clear()?;
        let rows =
            self.db
                .query_map("SELECT id, text FROM facts ORDER BY id ASC", vec![], |r| {
                    Ok((r.get::<_, i64>("id")?, r.get::<_, String>("text")?))
                })?;
        let count = rows.len();
        self.fts_facts.add_many(&rows)?;
        Ok(count)
    }

    /// Repopulate any FTS index that was wiped/recreated or is empty while its
    /// source table still holds rows (the Python-index migration failure mode).
    /// Synchronous — the live corpus is ~10k rows.
    fn repopulate_stale_fts(&self, turns_stale: bool, facts_stale: bool) -> Result<(), StoreError> {
        if turns_stale && self.table_has_rows("turns")? {
            let rows = self.rebuild_turns_fts()?;
            tracing::info!(
                target: "familiar_connect.history",
                index = "turns",
                rows,
                "rebuilt FTS index from DB (on-disk index was wiped or empty)"
            );
        }
        if facts_stale && self.table_has_rows("facts")? {
            let rows = self.rebuild_facts_fts()?;
            tracing::info!(
                target: "familiar_connect.history",
                index = "facts",
                rows,
                "rebuilt FTS index from DB (on-disk index was wiped or empty)"
            );
        }
        Ok(())
    }

    /// Cheap existence probe for one of the fixed source tables.
    fn table_has_rows(&self, table: &str) -> Result<bool, StoreError> {
        // `table` is a fixed literal from `repopulate_stale_fts`, never user input.
        let present = self
            .db
            .query_scalar_i64(format!("SELECT EXISTS(SELECT 1 FROM {table})"), vec![])?
            .unwrap_or(0);
        Ok(present != 0)
    }

    /// Highest turn id indexed for `familiar_id` (a `MAX(turns.id)` query; the
    /// index is write-synchronous). 0 when none.
    pub fn latest_fts_id(&self, familiar_id: &str) -> Result<i64, StoreError> {
        Ok(self
            .db
            .query_scalar_i64(
                "SELECT MAX(id) AS max_id FROM turns WHERE familiar_id = ?",
                vec![v_str(familiar_id)],
            )?
            .unwrap_or(0))
    }

    /// FTS search over `facts.text` (current truth by default).
    pub fn search_facts(
        &self,
        familiar_id: &str,
        query: &str,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        Ok(self
            .fact_candidates_by_fts(familiar_id, query, limit, include_superseded, as_of)?
            .into_iter()
            .map(|(fact, _)| fact)
            .collect())
    }

    /// Like [`search_facts`](Self::search_facts) but pairs each fact with its
    /// BM25 score (positive, higher = better).
    pub fn search_facts_scored(
        &self,
        familiar_id: &str,
        query: &str,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<(Fact, f32)>, StoreError> {
        self.fact_candidates_by_fts(familiar_id, query, limit, include_superseded, as_of)
    }

    fn fact_candidates_by_fts(
        &self,
        familiar_id: &str,
        query: &str,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<(Fact, f32)>, StoreError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let fts_limit = clamp_usize((limit * 4).max(limit));
        let hits = self.fts_facts.search(query, fts_limit);
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let score_by_id: HashMap<i64, f32> = hits.into_iter().collect();
        let candidate_ids: Vec<i64> = score_by_id.keys().copied().collect();
        let (where_clause, mut vparams) = facts_validity_where(include_superseded, as_of, "f");
        let mut params = vec![v_str(familiar_id)];
        params.extend(candidate_ids.iter().map(|&i| v_int(i)));
        params.append(&mut vparams);
        let facts = self.db.query_map(
            format!(
                "SELECT {FACT_COLS_F} FROM facts AS f \
                 WHERE f.familiar_id = ? AND f.id IN ({}) {where_clause}",
                placeholders(candidate_ids.len())
            ),
            params,
            map_fact,
        )?;
        let mut scored: Vec<(Fact, f32)> = facts
            .into_iter()
            .map(|f| {
                let score = score_by_id.get(&f.id).copied().unwrap_or(0.0);
                (f, score)
            })
            .collect();
        scored.sort_by(|(fa, sa), (fb, sb)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(fb.id.cmp(&fa.id))
        });
        scored.truncate(clamp_usize(limit));
        Ok(scored)
    }
}

impl Drop for HistoryStore {
    fn drop(&mut self) {
        self.db.close();
    }
}

// ---------------------------------------------------------------------------
// Private free helpers over &Connection (run on the actor thread)
// ---------------------------------------------------------------------------

/// Existing current fact matching normalized text + subject set, or `None`.
fn find_current_dup(
    conn: &rusqlite::Connection,
    familiar_id: &str,
    norm_text: &str,
    subject_keys: &HashSet<String>,
) -> Result<Option<Fact>, StoreError> {
    let (where_clause, vparams) = facts_validity_where(false, None, "");
    let mut params: Vec<Value> = vec![v_str(familiar_id)];
    params.extend(vparams);
    let mut stmt = conn.prepare(&format!(
        "SELECT {FACT_COLS} FROM facts WHERE familiar_id = ? {where_clause}"
    ))?;
    let facts = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), map_fact)?
        .collect::<rusqlite::Result<Vec<Fact>>>()?;
    for fact in facts {
        if normalize_fact_text(&fact.text) == norm_text
            && &subject_key_set(&fact.subjects) == subject_keys
        {
            return Ok(Some(fact));
        }
    }
    Ok(None)
}

/// Set `column = value` for `ids`, chunked under the SQLite param cap.
fn stamp_turns(
    tx: &rusqlite::Transaction,
    column: &str,
    ids: &[i64],
    value: &str,
) -> rusqlite::Result<()> {
    for batch in ids.chunks(STAMP_CHUNK) {
        let mut params: Vec<Value> = vec![v_str(value)];
        params.extend(batch.iter().map(|&i| v_int(i)));
        tx.execute(
            &format!(
                "UPDATE turns SET {column} = ? WHERE id IN ({})",
                placeholders(batch.len())
            ),
            rusqlite::params_from_iter(params.iter()),
        )?;
    }
    Ok(())
}

fn sorted_unique(ids: &[i64]) -> Vec<i64> {
    let set: std::collections::BTreeSet<i64> = ids.iter().copied().collect();
    set.into_iter().collect()
}

enum FactInsert {
    Existing(Fact),
    Minted(Fact),
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_keys_from_subjects_json, normalize_fact_text, parse_subjects, placeholders,
    };
    use std::collections::HashSet;

    #[test]
    fn placeholders_emits_bound_marks() {
        assert_eq!(placeholders(3), "?,?,?");
        assert_eq!(placeholders(1), "?");
    }

    #[test]
    fn placeholders_empty_for_zero() {
        assert_eq!(placeholders(0), "");
    }

    #[test]
    fn placeholders_only_question_marks_and_commas() {
        assert!(placeholders(5).chars().all(|c| c == '?' || c == ','));
    }

    #[test]
    fn canonical_keys_none_and_empty() {
        assert_eq!(canonical_keys_from_subjects_json(None), HashSet::new());
        assert_eq!(canonical_keys_from_subjects_json(Some("")), HashSet::new());
    }

    #[test]
    fn canonical_keys_unparseable_or_non_list() {
        assert_eq!(
            canonical_keys_from_subjects_json(Some("{not json")),
            HashSet::new()
        );
        assert_eq!(
            canonical_keys_from_subjects_json(Some("{\"a\": 1}")),
            HashSet::new()
        );
    }

    #[test]
    fn canonical_keys_extracts_str_keys_skips_malformed() {
        let blob = "[{\"canonical_key\": \"discord:1\", \"display_at_write\": \"Cor\"}, \
             {\"canonical_key\": 5}, \"not a dict\", {\"display_at_write\": \"no key\"}, \
             {\"canonical_key\": \"discord:2\", \"display_at_write\": \"Aria\"}]";
        let expected: HashSet<String> = ["discord:1".to_owned(), "discord:2".to_owned()]
            .into_iter()
            .collect();
        assert_eq!(canonical_keys_from_subjects_json(Some(blob)), expected);
    }

    #[test]
    fn parse_subjects_none_empty_and_non_list() {
        assert!(parse_subjects(None).is_empty());
        assert!(parse_subjects(Some("")).is_empty());
        assert!(parse_subjects(Some("{not json")).is_empty());
        assert!(parse_subjects(Some("{\"a\": 1}")).is_empty());
    }

    #[test]
    fn parse_subjects_coerces_non_string_values_like_python_str() {
        // Python `_row_to_fact` keeps any dict with BOTH keys, coercing each via
        // `str(...)`; only non-dict items or items missing a key are dropped.
        let blob = "[{\"canonical_key\": \"discord:1\", \"display_at_write\": \"Cor\"}, \
             {\"canonical_key\": 5, \"display_at_write\": \"X\"}, \
             \"not a dict\", \
             {\"canonical_key\": \"discord:2\"}, \
             {\"canonical_key\": \"discord:3\", \"display_at_write\": 7}]";
        let got = parse_subjects(Some(blob));
        let pairs: Vec<(String, String)> = got
            .into_iter()
            .map(|s| (s.canonical_key, s.display_at_write))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("discord:1".to_owned(), "Cor".to_owned()),
                ("5".to_owned(), "X".to_owned()),
                ("discord:3".to_owned(), "7".to_owned()),
            ]
        );
    }

    #[test]
    fn normalize_collapses_and_dequotes() {
        assert_eq!(
            normalize_fact_text("  \"Postbirb   Prime is called Cor\"  "),
            "postbirb prime is called cor"
        );
        assert_eq!(
            normalize_fact_text("Postbirb Prime is called 'Cor'."),
            "postbirb prime is called cor"
        );
    }
}
