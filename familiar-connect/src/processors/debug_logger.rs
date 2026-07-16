//! One log line per event on subscribed topics (subsystem 06; Python
//! `processors/debug_logger.py`).
//!
//! The Phase-1 "is the bus alive?" signal. Passive — never republishes.
//!
//! Port note: Python renders the payload as `repr(...)` truncated at 160 chars
//! (`-` when `None`). The Rust bus payload is a type-erased `Arc<dyn Any>` the
//! logger cannot inspect generically, so the payload field renders as `-`. The
//! topic / ids / sequence fields — the only ones the tests assert — are
//! reproduced exactly.

use crate::bus::envelope::Event;
use crate::bus::protocols::EventBus;
use crate::log_style as ls;

/// Logs one line per event; does not republish.
pub struct DebugLoggerProcessor {
    topics: Vec<String>,
}

impl DebugLoggerProcessor {
    /// The processor's human label.
    pub const NAME: &'static str = "debug-logger";

    /// Construct a logger over the injected `topics`.
    #[must_use]
    pub fn new(topics: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            topics: topics.into_iter().map(Into::into).collect(),
        }
    }

    /// The processor name (`"debug-logger"`).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        Self::NAME
    }

    /// The injected topics (borrowed for `bus.subscribe`).
    #[must_use]
    pub fn topics(&self) -> Vec<&str> {
        self.topics.iter().map(String::as_str).collect()
    }

    /// Log one INFO line for `event`.
    ///
    /// # Errors
    /// Never fails (returns `Ok` unconditionally) — signature matches the
    /// processor contract.
    #[allow(clippy::unused_async, reason = "processor contract is async")]
    pub async fn handle(&self, event: &Event, _bus: &dyn EventBus) -> anyhow::Result<()> {
        tracing::info!(
            "{} {} {} {} {} {} {}",
            ls::tag("\u{1f4e5} Event", ls::LG),
            ls::kv_styled("topic", &event.topic, ls::W, ls::LM),
            ls::kv_styled("event_id", &event.event_id, ls::W, ls::LC),
            ls::kv_styled("turn_id", &event.turn_id, ls::W, ls::LC),
            ls::kv_styled("session", &event.session_id, ls::W, ls::LC),
            ls::kv_styled("seq", &event.sequence_number.to_string(), ls::W, ls::LY),
            ls::kv_styled("payload", "-", ls::W, ls::LW),
        );
        Ok(())
    }
}
