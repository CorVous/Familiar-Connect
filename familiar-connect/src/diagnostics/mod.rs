//! Logs-first timing/diagnostics: spans, collectors, voice budget, cold-cache
//! signals, summary table (subsystem 01; Python `diagnostics/`).
//!
//! Layer 0: imports only [`crate::log_style`]. The public surface mirrors the
//! Python package — [`spans::timed_sync`]/[`spans::timed_async`] (the `@span`
//! decorator has no Rust analog), the [`collector::SpanCollector`] ring +
//! singleton, the [`voice_budget::VoiceBudgetRecorder`] funnel, the
//! [`cold_cache`] signal detectors, and [`report::render_summary_table`].

pub mod cold_cache;
pub mod collector;
pub mod report;
pub mod spans;
pub mod voice_budget;

#[cfg(test)]
pub(crate) use testutil::{Capture, strip_ansi};

/// Shared test harness: singleton serialization guard + a `tracing` capture
/// layer + panic-silencing catch helpers. Only compiled for the crate's own
/// unit tests (`cfg(test)`); a `test-util` feature would be needed to expose the
/// singleton `reset_*` seams to cross-subsystem integration tests.
#[cfg(test)]
pub(crate) mod testutil {
    use std::any::Any;
    use std::panic::AssertUnwindSafe;
    use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

    use futures::FutureExt;
    use regex::Regex;
    use tracing::field::{Field, Visit};
    use tracing::subscriber::DefaultGuard;
    use tracing::{Event, Level, Subscriber};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    // Serializes every test that reads or mutates a process-wide singleton
    // (SpanCollector / VoiceBudgetRecorder), mirroring pytest's serial run +
    // autouse reset fixtures. Poison is recovered so one failing test does not
    // cascade.
    static GUARD: Mutex<()> = Mutex::new(());

    /// Acquire the process-wide singleton test lock (recovers from poison).
    pub fn singleton_guard() -> MutexGuard<'static, ()> {
        GUARD.lock().unwrap_or_else(PoisonError::into_inner)
    }

    static ANSI_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").expect("valid ansi regex"));

    /// Strip single-parameter (and compound) SGR codes for assertions.
    pub fn strip_ansi(text: &str) -> String {
        ANSI_RE.replace_all(text, "").into_owned()
    }

    /// One captured `tracing` event.
    #[derive(Clone, Debug)]
    pub struct Rec {
        pub level: Level,
        #[allow(dead_code)] // available for target-scoped assertions
        pub target: String,
        pub message: String,
    }

    /// A `tracing` layer that records every event's level/target/message.
    #[derive(Clone, Default)]
    pub struct Capture {
        recs: Arc<Mutex<Vec<Rec>>>,
    }

    impl Capture {
        /// Snapshot of captured records in emission order.
        pub fn records(&self) -> Vec<Rec> {
            self.recs
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    struct MsgVisitor<'a>(&'a mut String);

    impl Visit for MsgVisitor<'_> {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "message" {
                self.0.push_str(value);
            }
        }
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                use std::fmt::Write;
                // `fmt::Arguments::Debug == Display`, so no escaping is added.
                let _ = write!(self.0, "{value:?}");
            }
        }
    }

    impl<S: Subscriber> Layer<S> for Capture {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut message = String::new();
            event.record(&mut MsgVisitor(&mut message));
            let meta = event.metadata();
            self.recs
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(Rec {
                    level: *meta.level(),
                    target: meta.target().to_string(),
                    message,
                });
        }
    }

    /// Install `cap` as the thread-local default subscriber for this test; the
    /// returned guard restores the previous default on drop.
    pub fn install_capture(cap: &Capture) -> DefaultGuard {
        let subscriber = tracing_subscriber::registry().with(cap.clone());
        tracing::subscriber::set_default(subscriber)
    }

    /// Await `fut`, catching a panic without the noisy default hook output.
    /// Restores the previous hook afterwards. Used by the span error-path tests.
    pub async fn catch_silent_async<F: Future>(fut: F) -> Result<F::Output, Box<dyn Any + Send>> {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = AssertUnwindSafe(fut).catch_unwind().await;
        std::panic::set_hook(prev);
        result
    }
}
