//! Port-introduced shared helpers (no direct Python source). These resolve
//! cross-cutting conventions ONCE so porters do not diverge. See rust/DESIGN.md
//! section 4.
//!
//! - `time`:  ISO-8601 UTC emission/parse (fixed-width microseconds, `+00:00`).
//! - `round`: half-to-even rounding matching Python `round()`.
//! - `text`:  Unicode-scalar-safe truncation with the U+2026 ellipsis.

pub mod round;
pub mod text;
pub mod time;
