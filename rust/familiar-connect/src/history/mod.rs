//! Durable per-familiar SQLite store + tantivy full-text indexes
//! (subsystem 03; Python `history/`).

pub mod async_store;
pub mod db;
pub mod fts;
pub mod store;
