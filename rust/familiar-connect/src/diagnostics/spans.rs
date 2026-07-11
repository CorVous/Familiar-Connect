//! `@span` timing — structured timing logs.
//!
//! Port of `familiar_connect/diagnostics/spans.py`. Python exposes a `@span`
//! decorator that wraps sync **or** async callables; Rust has no decorators, so
//! this exposes two helpers — [`timed_sync`] and [`timed_async`] — that time an
//! operation, emit one DEBUG line per call on the `familiar_connect.diagnostics`
//! target, and record into the process-wide [`SpanCollector`](super::collector).
//!
//! Timing is a wall-clock delta reported as `int(elapsed_seconds * 1000)`
//! (truncation toward zero, DESIGN §4.3). Status is `"ok"` on normal return,
//! `"error"` on **any** early exit — a panic (the Rust analog of a raised
//! exception) or, for [`timed_async`], the future being dropped before it
//! completes (cancellation: barge-in, a `JoinSet`/task abort of an `@span`
//! worker tick — the Rust analog of `CancelledError`). The span line and record
//! are emitted in a `finally`-like position on every path (spec 01 §20); a panic
//! is re-raised unchanged afterwards.

use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use crate::diagnostics::collector::get_span_collector;
use crate::log_style as ls;

/// Time a synchronous closure, emitting the span line and recording the result.
///
/// A panic in `f` yields `status="error"` and is resumed unchanged after emit.
pub fn timed_sync<T, F: FnOnce() -> T>(name: &str, f: F) -> T {
    let start = Instant::now();
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    let status = if result.is_ok() { "ok" } else { "error" };
    emit(name, start.elapsed(), status);
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Time a future, emitting the span line and recording the result.
///
/// The span line + collector record are emitted on **every** exit path, exactly
/// like Python's `try/finally` (spec 01 §20): normal completion reports
/// `status="ok"`; a panic while polling **or** the future being dropped before it
/// completes (cancellation — barge-in, a `JoinSet`/task abort of an `@span`
/// worker tick) reports `status="error"`, matching Python re-raising
/// `CancelledError` through its `except BaseException` arm. A panic propagates
/// unchanged after the span is emitted.
///
/// Cancellation matters here because Rust has no async `Drop`: dropping the
/// future runs no user code except `Drop` impls, so the emit is anchored in a
/// guard whose synchronous `Drop` fires when the future is cancelled. Without it,
/// `/diagnostics` and the diagnose CLI would silently omit timings for every
/// cancelled/aborted turn — common on the voice path.
pub async fn timed_async<T, F: Future<Output = T>>(name: &str, fut: F) -> T {
    // Armed on construction; its `Drop` emits `status="error"` on any exit we do
    // not reach normally — a panic unwind or a cancellation drop, the two cases
    // where no further statement in this fn runs. Reaching the end calls
    // `finish("ok")`, which emits and disarms the guard.
    let mut guard = SpanGuard::new(name);
    let value = fut.await;
    guard.finish("ok");
    value
}

/// Emits a span exactly once, on the first of [`SpanGuard::finish`] or `Drop`.
///
/// [`timed_async`] arms one across the awaited future. Reaching normal completion
/// calls `finish` (explicit status, disarms). If instead the future panics or is
/// dropped mid-poll (cancellation), the value is never produced and `Drop` emits
/// `status="error"` — the Rust analog of Python's `finally` running after
/// `except BaseException` (spec 01 §20).
struct SpanGuard<'a> {
    name: &'a str,
    start: Instant,
    armed: bool,
}

impl<'a> SpanGuard<'a> {
    fn new(name: &'a str) -> Self {
        Self {
            name,
            start: Instant::now(),
            armed: true,
        }
    }

    fn finish(&mut self, status: &str) {
        self.armed = false;
        emit(self.name, self.start.elapsed(), status);
    }
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            emit(self.name, self.start.elapsed(), "error");
        }
    }
}

/// Emit the DEBUG span line and record into the singleton collector.
///
/// The line is a wire format (spec 01 §21): after ANSI stripping it reads
/// `[span] span=<name> ms=<int> status=<ok|error>` in that order. DEBUG level is
/// pinned by test — visible at `-vv` only. Recording never raises into the
/// caller (the collector is poison-safe).
fn emit(name: &str, elapsed: Duration, status: &str) {
    // `@span` ms truncates toward zero (distinct from voice-budget rounding).
    #[allow(clippy::cast_possible_truncation)]
    let ms = (elapsed.as_secs_f64() * 1000.0) as i64;
    let line = format!(
        "{} {} {} {}",
        ls::tag("span", ls::LM),
        ls::kv_styled("span", name, ls::W, ls::LM),
        ls::kv_styled("ms", &ms.to_string(), ls::W, ls::LC),
        ls::kv_styled(
            "status",
            status,
            ls::W,
            if status == "ok" { ls::LG } else { ls::R }
        ),
    );
    tracing::debug!(target: "familiar_connect.diagnostics", "{line}");
    get_span_collector().record(name, ms, status);
}

#[cfg(test)]
mod tests {
    use super::{timed_async, timed_sync};
    use crate::diagnostics::collector::reset_span_collector;
    use crate::diagnostics::testutil::{catch_silent_async, install_capture, singleton_guard};
    use crate::diagnostics::{Capture, strip_ansi};
    use regex::Regex;
    use std::sync::LazyLock;
    use std::time::Duration;

    static MS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"ms=\d+").expect("valid"));

    fn span_lines(cap: &Capture) -> Vec<crate::diagnostics::testutil::Rec> {
        cap.records()
            .into_iter()
            .filter(|r| strip_ansi(&r.message).contains("span="))
            .collect()
    }

    // Guard held across await to serialize singleton access; safe on the
    // current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn emits_ms_log_on_async_function() {
        let _g = singleton_guard();
        reset_span_collector();
        let cap = Capture::default();
        let _sub = install_capture(&cap);

        let result = timed_async("demo", async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            42
        })
        .await;
        assert_eq!(result, 42);

        let spans = span_lines(&cap);
        assert!(!spans.is_empty(), "no span log emitted");
        let last = strip_ansi(&spans.last().unwrap().message);
        assert!(last.contains("span=demo"));
        assert!(MS_RE.is_match(&last));
        // spans are DEBUG-level — shown at -vv, not -v
        assert_eq!(spans.last().unwrap().level, tracing::Level::DEBUG);
    }

    #[test]
    fn emits_ms_log_on_sync_function() {
        let _g = singleton_guard();
        reset_span_collector();
        let cap = Capture::default();
        let _sub = install_capture(&cap);

        assert_eq!(timed_sync("sync-demo", || "ok"), "ok");

        let spans = span_lines(&cap);
        assert!(!spans.is_empty());
        assert!(strip_ansi(&spans.last().unwrap().message).contains("span=sync-demo"));
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn logs_even_on_exception() {
        let _g = singleton_guard();
        reset_span_collector();
        let cap = Capture::default();
        let _sub = install_capture(&cap);

        let outcome = catch_silent_async(timed_async("failing", async {
            tokio::time::sleep(Duration::from_millis(0)).await;
            panic!("boom");
        }))
        .await;
        assert!(outcome.is_err(), "panic must propagate");

        let spans = span_lines(&cap);
        assert!(!spans.is_empty());
        let last = strip_ansi(&spans.last().unwrap().message);
        assert!(last.contains("span=failing"));
        assert!(last.contains("status=error"));
    }

    // A future dropped before completion (barge-in / JoinSet abort) must still
    // emit an `error` span — Rust's analog of Python's `finally` firing on
    // `CancelledError`. Drive it by entering the async body (which arms the
    // guard) and parking at the inner await, then dropping without completing.
    #[test]
    fn emits_error_span_when_future_dropped_cancelled() {
        use crate::diagnostics::collector::get_span_collector;
        use std::future::{Future, pending};
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        let _g = singleton_guard();
        reset_span_collector();

        {
            let mut fut = pin!(timed_async("cancelled", pending::<i32>()));
            let mut cx = Context::from_waker(Waker::noop());
            assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Pending));
        } // `fut` dropped here — the cancellation path.

        let records = get_span_collector().all();
        assert_eq!(records.len(), 1, "cancelled future must still record");
        assert_eq!(records[0].name, "cancelled");
        assert_eq!(records[0].status, "error");
    }
}
