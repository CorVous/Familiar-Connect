//! Durable per-familiar SQLite store + tantivy full-text indexes
//! (subsystem 03).
//!
//! The threading sandwich is reshaped into a
//! single DB actor (see [`db`]):
//!
//! - [`db`] — the single-owner DB actor over `rusqlite`. One OS thread owns the
//!   [`rusqlite::Connection`]; callers submit whole-operation closures over an
//!   `mpsc` channel and block on a reply.
//! - [`store`] — [`HistoryStore`]: the append-only `turns` log plus every
//!   watermarked side-index projection and all query shapes. The full schema is
//!   declared up front in `SCHEMA`; the earlier incremental `_migrate()`
//!   was folded in and removed (issue #202).
//! - [`fts`] — the tantivy full-text seam (`familiar_en` analyzer). **Stage B.**
//! - [`llm_mirror`] — the `llm_calls` table writer behind the subsystem-01
//!   [`LlmCallSink`](crate::diagnostics::llm_mirror::LlmCallSink) seam.
//! - [`async_store`] — the async facade over the store. **Stage B.**
//!
//! Value types, the [`FtsIndex`] seam, and [`HistoryStore`] are re-exported at
//! the module root for consumers (subsystems 02/04/05/06/07/08/10/11).

pub mod async_store;
pub mod db;
pub mod fts;
pub mod llm_mirror;
pub mod store;

pub use crate::identity::Author;
pub use async_store::AsyncHistoryStore;
pub use db::Db;
pub use fts::{CommitFault, TantivyFts};
pub use store::{
    AccountProfile, ActivityRecord, AlarmRow, AppendFact, AppendLlmCall, AppendTurn, ChannelUnread,
    FOCUS_STREAM_CHANNEL_ID, Fact, FactDraft, FactSubject, FocusPointers, FtsIndex, HistoryStore,
    HistoryTurn, LlmCallRow, NewFact, NoopFtsIndex, OtherChannelInfo, PeopleDossierEntry,
    Promotion, Reflection, SleepWatermark, SummaryEntry, SupersedeResult, WatermarkEntry,
};

/// One error enum for the whole history subsystem.
///
/// Genuine faults only: a closed connection, an engine error (including CHECK
/// violations such as an out-of-range `alarms.channel_kind`), an empty
/// embedding vector, or an invalid `finish_activity` status. Reads that hit
/// malformed *stored* data degrade to empty/`None` rather than erroring
/// — those paths never surface a `StoreError`.
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
