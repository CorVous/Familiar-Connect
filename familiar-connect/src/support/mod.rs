//! Shared helpers that resolve cross-cutting conventions ONCE, so call sites
//! across the crate do not diverge.
//!
//! - `time`:  ISO-8601 UTC emission/parse (fixed-width microseconds, `+00:00`).
//! - `round`: half-to-even ("banker's") rounding.
//! - `text`:  Unicode-scalar-safe truncation with the U+2026 ellipsis.

pub mod round;
pub mod text;
pub mod time;
