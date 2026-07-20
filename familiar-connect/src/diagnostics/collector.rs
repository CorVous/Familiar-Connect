//! In-process span collector — ring buffer of recent `@span` calls.
//!
//! Port of `familiar_connect/diagnostics/collector.py`. Feeds the
//! `/diagnostics` slash command with a breakdown of last-turn timings without
//! re-parsing logs at runtime. Logs-first remains the durable record; the
//! collector is a live convenience only.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

/// One recorded timing span.
#[derive(Clone, Debug)]
pub struct SpanRecord {
    /// Span name (e.g. `voice.stt_to_ttft`).
    pub name: String,
    /// Elapsed milliseconds (truncated toward zero for `@span`).
    pub ms: i64,
    /// `"ok"` or `"error"`.
    pub status: String,
    /// Wall-clock stamp (UTC) at record time.
    pub at: DateTime<Utc>,
}

/// Per-name aggregate stats, mirroring Python's
/// `{count, p50, p95, last_ms}` dict (all numeric floats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanStats {
    /// Number of records for the name.
    pub count: f64,
    /// Interpolated 50th percentile of `ms`.
    pub p50: f64,
    /// Interpolated 95th percentile of `ms`.
    pub p95: f64,
    /// The most recent record's `ms` (insertion order, not max).
    pub last_ms: f64,
}

/// Bounded ring buffer. Thread-safe appends + reads.
#[derive(Debug)]
pub struct SpanCollector {
    buf: Mutex<VecDeque<SpanRecord>>,
    maxlen: usize,
}

impl SpanCollector {
    /// Create a collector holding at most `maxlen` records.
    #[must_use]
    pub const fn new(maxlen: usize) -> Self {
        Self {
            buf: Mutex::new(VecDeque::new()),
            maxlen,
        }
    }

    /// Append a record stamped `at = now(UTC)`, evicting the oldest past
    /// capacity. Never panics (a poisoned lock is recovered) — recording must
    /// never raise into the caller (spec 01 §22).
    // The guard's scope is the whole method (push + evict both need it).
    #[allow(clippy::significant_drop_tightening)]
    pub fn record(&self, name: &str, ms: i64, status: &str) {
        let rec = SpanRecord {
            name: name.to_string(),
            ms,
            status: status.to_string(),
            at: Utc::now(),
        };
        let mut buf = self
            .buf
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        buf.push_back(rec);
        while buf.len() > self.maxlen {
            buf.pop_front();
        }
    }

    /// Snapshot copy of all records, oldest first.
    #[must_use]
    pub fn all(&self) -> Vec<SpanRecord> {
        let buf = self
            .buf
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        buf.iter().cloned().collect()
    }

    /// Records grouped by name; each group preserves insertion order.
    #[must_use]
    pub fn by_name(&self) -> HashMap<String, Vec<SpanRecord>> {
        let mut out: HashMap<String, Vec<SpanRecord>> = HashMap::new();
        for rec in self.all() {
            out.entry(rec.name.clone()).or_default().push(rec);
        }
        out
    }

    /// Per-name `{count, p50, p95, last_ms}`. Returned sorted by name (the
    /// renderer sorts anyway; an empty collector yields no keys).
    #[must_use]
    pub fn summary(&self) -> BTreeMap<String, SpanStats> {
        let mut out = BTreeMap::new();
        for (name, records) in self.by_name() {
            let last = records.last().map_or(0, |r| r.ms);
            let mut ms_values: Vec<i64> = records.iter().map(|r| r.ms).collect();
            ms_values.sort_unstable();
            out.insert(
                name,
                SpanStats {
                    #[allow(clippy::cast_precision_loss)] // counts are small
                    count: records.len() as f64,
                    p50: percentile(&ms_values, 50),
                    p95: percentile(&ms_values, 95),
                    #[allow(clippy::cast_precision_loss)] // ms fits f64 exactly here
                    last_ms: last as f64,
                },
            );
        }
        out
    }

    /// Drop all records.
    pub fn clear(&self) {
        self.buf
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Linear-interpolated percentile over pre-sorted values.
///
/// `rank = pct/100 * (n-1)`; empty → 0.0; single value → that value. Ported once
/// and shared with `commands::diagnose` (spec 01 §24, §44).
#[must_use]
#[allow(clippy::cast_precision_loss)] // values/indices are bounded (<= ring size)
pub fn percentile(sorted_values: &[i64], pct: u32) -> f64 {
    match sorted_values.len() {
        0 => return 0.0,
        1 => return sorted_values[0] as f64,
        _ => {}
    }
    let n = sorted_values.len();
    let rank = (f64::from(pct) / 100.0) * (n - 1) as f64;
    // `int(rank)` — truncation toward zero; rank is non-negative.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = rank as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    // Separate multiply/add (not fused `mul_add`) for bit-parity with Python.
    #[allow(clippy::suboptimal_flops)]
    {
        sorted_values[lo] as f64 * (1.0 - frac) + sorted_values[hi] as f64 * frac
    }
}

// ---------------------------------------------------------------------------
// process-wide singleton
// ---------------------------------------------------------------------------

/// Default ring capacity (spec 01 § Config knobs).
const DEFAULT_MAXLEN: usize = 2000;

static COLLECTOR: Mutex<Option<Arc<SpanCollector>>> = Mutex::new(None);

/// Return the process-wide [`SpanCollector`], creating it on first use.
///
/// Every producer path fetches the singleton at call time (not at import time)
/// so `reset_span_collector` takes effect immediately (spec 01 §25).
#[must_use]
pub fn get_span_collector() -> Arc<SpanCollector> {
    let mut guard = COLLECTOR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .get_or_insert_with(|| Arc::new(SpanCollector::new(DEFAULT_MAXLEN)))
        .clone()
}

/// Reset the singleton so the next `get` creates a fresh instance — tests only.
#[cfg(any(test, feature = "test-util"))]
pub fn reset_span_collector() {
    *COLLECTOR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

#[cfg(test)]
mod tests {
    use super::{SpanCollector, get_span_collector, percentile, reset_span_collector};
    use crate::diagnostics::spans;
    use crate::diagnostics::testutil::singleton_guard;
    use std::sync::Arc;

    // --- SpanCollector direct (no singleton; parallel-safe) ---

    #[test]
    fn record_then_read() {
        let coll = SpanCollector::new(10);
        coll.record("a", 10, "ok");
        coll.record("a", 20, "ok");
        coll.record("b", 5, "error");
        assert_eq!(coll.all().len(), 3);
        let by_name = coll.by_name();
        assert_eq!(by_name["a"].len(), 2);
        assert_eq!(by_name["b"].len(), 1);
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let coll = SpanCollector::new(3);
        for i in 0..5 {
            coll.record("x", i, "ok");
        }
        let records = coll.all();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records.iter().map(|r| r.ms).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn summary_computes_percentiles() {
        let coll = SpanCollector::new(100);
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            coll.record("llm", ms, "ok");
        }
        let summary = coll.summary();
        let s = summary["llm"];
        assert!((s.count - 10.0).abs() < f64::EPSILON);
        assert!((s.p50 - 55.0).abs() <= 1.0);
        assert!((s.p95 - 95.5).abs() <= 1.0);
        assert!((s.last_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn summary_empty_buckets() {
        let coll = SpanCollector::new(10);
        assert!(coll.summary().is_empty());
    }

    #[test]
    fn percentile_edge_cases() {
        assert!((percentile(&[], 50) - 0.0).abs() < f64::EPSILON);
        assert!((percentile(&[7], 95) - 7.0).abs() < f64::EPSILON);
        let vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert!((percentile(&vals, 50) - 55.0).abs() < f64::EPSILON);
        assert!((percentile(&vals, 95) - 95.5).abs() < f64::EPSILON);
    }

    // --- singleton (guarded: shared process-wide state) ---

    #[test]
    fn singleton_reset() {
        let _g = singleton_guard();
        reset_span_collector();
        let a = get_span_collector();
        let b = get_span_collector();
        assert!(Arc::ptr_eq(&a, &b));
        reset_span_collector();
        let c = get_span_collector();
        assert!(!Arc::ptr_eq(&a, &c));
    }

    // --- @span integration (ported from tests/test_span_collector.py) ---

    // Guard held across await to serialize singleton access; safe on the
    // current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn span_decorator_records_into_collector() {
        let _g = singleton_guard();
        reset_span_collector();
        let result = spans::timed_async("demo", async {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            42
        })
        .await;
        assert_eq!(result, 42);
        let records = get_span_collector().all();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "demo");
        assert_eq!(records[0].status, "ok");
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn span_records_errors() {
        let _g = singleton_guard();
        reset_span_collector();
        let outcome =
            crate::diagnostics::testutil::catch_silent_async(spans::timed_async("boom", async {
                tokio::time::sleep(std::time::Duration::from_millis(0)).await;
                panic!("x");
            }))
            .await;
        assert!(outcome.is_err());
        let records = get_span_collector().all();
        assert!(!records.is_empty());
        assert_eq!(records.last().unwrap().status, "error");
    }
}
