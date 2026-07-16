//! Bus sources (subsystem 10; Python `sources/`).
//!
//! Discord text/embeds, voice queue drain, Twitch queue drain. Per-source
//! queue->bus semantics for `voice`/`twitch` are pinned by specs 10 (B-VS/B-TS)
//! and 09/11.

pub mod discord_embed_text;
pub mod discord_text;
pub mod twitch;
pub mod voice;
