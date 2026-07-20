//! Durable per-familiar SQLite store + tantivy full-text indexes
//! (subsystem 03; Python `history/`).
//!
//! Layout mirrors the Python package but reshapes the threading sandwich into a
//! single DB actor (see [`db`] and DESIGN §4.4 / decision D5):
//!
//! - [`db`] — the single-owner DB actor over `rusqlite`. One OS thread owns the
//!   [`rusqlite::Connection`]; callers submit whole-operation closures over an
//!   `mpsc` channel and block on a reply. This supersedes Python
//!   `history/turso_compat.py` (`TursoConnection`).
//! - [`store`] — [`HistoryStore`]: the append-only `turns` log plus every
//!   watermarked side-index projection and all query shapes. The full schema is
//!   declared up front in `SCHEMA`; the Python era's incremental `_migrate()`
//!   was folded in and removed (issue #202). Ports Python `history/store.py`.
//! - [`fts`] — the tantivy full-text seam (`familiar_en` analyzer). **Stage B.**
//! - [`async_store`] — the async facade over the store. **Stage B.**
//!
//! Value types, the [`FtsIndex`] seam, and [`HistoryStore`] are re-exported at
//! the module root for consumers (subsystems 02/04/05/06/07/08/10/11).

pub mod async_store;
pub mod db;
pub mod fts;
pub mod store;

pub use crate::identity::Author;
pub use async_store::AsyncHistoryStore;
pub use db::Db;
pub use fts::{CommitFault, TantivyFts};
pub use store::{
    AccountProfile, ActivityRecord, AlarmRow, AppendFact, AppendTurn, ChannelUnread,
    FOCUS_STREAM_CHANNEL_ID, Fact, FactDraft, FactSubject, FocusPointers, FtsIndex, HistoryStore,
    HistoryTurn, NewFact, NoopFtsIndex, OtherChannelInfo, PeopleDossierEntry, Promotion,
    Reflection, SleepWatermark, SummaryEntry, SupersedeResult, WatermarkEntry,
};

/// One error enum for the whole history subsystem (DESIGN §4.1).
///
/// Genuine faults only: a closed connection, an engine error (including CHECK
/// violations such as an out-of-range `alarms.channel_kind`), an empty
/// embedding vector, or an invalid `finish_activity` status. Reads that hit
/// malformed *stored* data degrade to empty/`None` rather than erroring
/// (behavior 27) — those paths never surface a `StoreError`.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A call arrived after [`Db::close`] — the owning actor thread is gone.
    #[error("history database connection is closed")]
    Closed,
    /// An underlying `rusqlite` engine error (I/O, constraint/CHECK violation…).
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    /// Failed to spawn the dedicated DB actor thread.
    #[error(transparent)]
    Thread(#[from] std::io::Error),
    /// `set_fact_embedding` was handed an empty vector.
    #[error("set_fact_embedding requires a non-empty vector")]
    EmptyVector,
    /// `finish_activity` status was neither `"completed"` nor `"cut_short"`.
    #[error("invalid activity status: {0:?}")]
    InvalidActivityStatus(String),
    /// A tantivy full-text index error (open, write, or exhausted commit retry).
    /// The store's `_safe_fts_add` guard swallows this on `append_turn` /
    /// `append_fact` (the SQL row is already committed); rebuild/clear/delete
    /// surface it.
    #[error("full-text index error: {0}")]
    Fts(String),
}
