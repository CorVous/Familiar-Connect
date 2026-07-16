//! Shared fixtures for the context-layer integration tests (subsystem 05).
//! Included via `#[path = "context_helpers/mod.rs"] mod helpers;` — not a test
//! binary itself.
#![allow(dead_code)]

use std::sync::Arc;

use chrono::{DateTime, Utc};
use familiar_connect::context::AssemblyContext;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::db::Value;
use familiar_connect::history::store::HistoryStore;
use familiar_connect::identity::Author;
use familiar_connect::support::time::iso_utc;

/// A fresh in-memory store wrapped in the async facade.
#[must_use]
pub fn store() -> Arc<AsyncHistoryStore> {
    let hs = HistoryStore::open(":memory:").expect("open in-memory store");
    Arc::new(AsyncHistoryStore::new(hs))
}

/// A voice-tier context for `channel_id` under familiar `"fam"`.
#[must_use]
pub fn vctx(channel_id: i64) -> AssemblyContext {
    AssemblyContext::new("fam", Some(channel_id)).with_viewer_mode("voice")
}

/// A text-tier context for `channel_id` under familiar `"fam"`.
#[must_use]
pub fn tctx(channel_id: i64) -> AssemblyContext {
    AssemblyContext::new("fam", Some(channel_id)).with_viewer_mode("text")
}

/// A voice-tier context with a guild id.
#[must_use]
pub fn vctx_guild(channel_id: i64, guild_id: i64) -> AssemblyContext {
    vctx(channel_id).with_guild_id(guild_id)
}

/// A Discord author (`username = display.lower()`), mirroring the Python
/// `_author` fixture.
#[must_use]
pub fn author(user_id: &str, display: &str) -> Author {
    Author::new(
        "discord",
        user_id,
        Some(display.to_lowercase()),
        Some(display.to_owned()),
    )
}

/// Overwrite a turn's timestamp (mirrors the Python `store._conn.execute(...)`
/// fixtures that pin clocks for coalesce / date-header assertions).
pub fn set_ts(store: &AsyncHistoryStore, turn_id: i64, dt: DateTime<Utc>) {
    store
        .sync()
        .conn()
        .execute(
            "UPDATE turns SET timestamp = ? WHERE id = ?",
            vec![Value::Text(iso_utc(dt)), Value::Integer(turn_id)],
        )
        .expect("update turn timestamp");
}
