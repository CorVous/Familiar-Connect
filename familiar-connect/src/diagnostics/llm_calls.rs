//! `[LLM call]` log-line parser + per-slot/model cache aggregation (#206).
//!
//! `CallMetrics::emit` (`crate::llm`) writes one INFO line per LLM request; the
//! line is a wire format. This module re-reads it from a captured log so the
//! prompt-cache question can be answered from data instead of from reasoning
//! about layer order: how often the provider reports a cache hit, how much of
//! the prompt prefix was reused, and what a miss costs in `ttfb_ms`.
//!
//! Renders through [`super::report::render_llm_call_report`].

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;

use super::collector::percentile;

/// One parsed `[LLM call]` line.
///
/// Every key past `slot`/`model`/`status` is optional: a failed call reports no
/// tokens, `cached` rides only on providers that report it, `cal_ratio` only
/// once a model has a usage-bearing call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallRecord {
    /// `slot=` — `"-"` when the call named no slot.
    pub slot: String,
    /// `model=`.
    pub model: String,
    /// `status=` — open vocabulary (`ok`/`error`/`cancelled`/`silent`/…).
    pub status: String,
    /// `ttfb_ms=` — first response byte.
    pub ttfb_ms: Option<i64>,
    /// `ttft_ms=` — first content delta.
    pub ttft_ms: Option<i64>,
    /// `total_ms=`.
    pub total_ms: Option<i64>,
    /// `est_in_tokens=` — the `len/4` heuristic's guess.
    pub est_in_tokens: Option<i64>,
    /// `in_tokens=` — provider-reported prompt tokens.
    pub in_tokens: Option<i64>,
    /// `out_tokens=`.
    pub out_tokens: Option<i64>,
    /// `cached=` — provider-reported cached prompt tokens.
    pub cached: Option<i64>,
    /// `cal_ratio=` — running Σtrue/Σestimated for the model (#183).
    pub cal_ratio: Option<f64>,
}

/// Latency samples split by whether the call reported a cache hit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HitMissLatency {
    /// Samples from calls with `cached > 0`, sorted ascending.
    pub hit: Vec<i64>,
    /// Samples from usage-bearing calls with no cached tokens, sorted ascending.
    pub miss: Vec<i64>,
}

/// Per-`(slot, model)` cache + latency aggregate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallGroupStats {
    /// Lines parsed into this group.
    pub calls: usize,
    /// `status` → count. Open vocabulary, so a map rather than fixed fields.
    pub by_status: BTreeMap<String, usize>,
    /// Calls reporting `in_tokens` — the only ones scorable hit/miss.
    pub usage_calls: usize,
    /// Usage-bearing calls with `cached > 0`.
    pub hit_calls: usize,
    /// Σ`cached` over usage-bearing calls.
    pub cached_sum: i64,
    /// Σ`in_tokens` over usage-bearing calls.
    pub in_sum: i64,
    /// `ttfb_ms` split by hit/miss.
    pub ttfb: HitMissLatency,
    /// `ttft_ms` split by hit/miss.
    pub ttft: HitMissLatency,
}

/// Per-model estimator accuracy (#184 item 3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EstimatorStats {
    /// Calls contributing an estimated/true pair.
    pub calls: usize,
    /// Σ`est_in_tokens` over those calls.
    pub est_sum: i64,
    /// Σ`in_tokens` over those calls.
    pub in_sum: i64,
    /// Last `cal_ratio` seen for the model (file order) — the most-converged.
    pub cal_ratio: Option<f64>,
}

/// Everything the `[LLM call]` table needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallSummary {
    /// `(slot, model)` → cache/latency stats.
    pub groups: BTreeMap<(String, String), CallGroupStats>,
    /// `model` → estimator accuracy.
    pub models: BTreeMap<String, EstimatorStats>,
}

impl HitMissLatency {
    /// Interpolated percentile of the hit samples; `None` when empty.
    #[must_use]
    pub fn hit_p(&self, pct: u32) -> Option<f64> {
        (!self.hit.is_empty()).then(|| percentile(&self.hit, pct))
    }

    /// Interpolated percentile of the miss samples; `None` when empty.
    #[must_use]
    pub fn miss_p(&self, pct: u32) -> Option<f64> {
        (!self.miss.is_empty()).then(|| percentile(&self.miss, pct))
    }

    /// `miss p50 - hit p50` — what a miss costs; `None` unless both sides have
    /// samples.
    #[must_use]
    pub fn cost_p50(&self) -> Option<f64> {
        Some(self.miss_p(50)? - self.hit_p(50)?)
    }

    /// No samples on either side.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hit.is_empty() && self.miss.is_empty()
    }

    // Sort both sides once, after ingest — `percentile` requires sorted input.
    fn sort(&mut self) {
        self.hit.sort_unstable();
        self.miss.sort_unstable();
    }
}

/// `a / b` as `f64`, or `None` when `b` is zero.
#[allow(clippy::cast_precision_loss)] // counts and token sums stay well inside f64
fn ratio(a: i64, b: i64) -> Option<f64> {
    (b != 0).then(|| a as f64 / b as f64)
}

impl CallGroupStats {
    /// Fraction of usage-bearing calls that reported any cached tokens.
    ///
    /// Calls with no reported usage (a failed or cancelled call) are *not*
    /// counted as misses — they are unobservable, not cold.
    #[must_use]
    pub fn call_hit_rate(&self) -> Option<f64> {
        ratio(
            i64::try_from(self.hit_calls).unwrap_or(i64::MAX),
            i64::try_from(self.usage_calls).unwrap_or(i64::MAX),
        )
    }

    /// Σ`cached` / Σ`in_tokens` — the fraction of prompt prefix actually reused.
    /// The number that speaks to #206: a call can "hit" on a sliver.
    #[must_use]
    pub fn token_hit_rate(&self) -> Option<f64> {
        ratio(self.cached_sum, self.in_sum)
    }

    /// Mean `in_tokens` over usage-bearing calls — a small stable prefix makes a
    /// cache fix not worth the write premium.
    #[must_use]
    pub fn mean_in_tokens(&self) -> Option<f64> {
        ratio(self.in_sum, i64::try_from(self.usage_calls).unwrap_or(0))
    }
}

impl EstimatorStats {
    /// Observed Σ`in_tokens` / Σ`est_in_tokens`.
    #[must_use]
    pub fn observed_ratio(&self) -> Option<f64> {
        ratio(self.in_sum, self.est_sum)
    }
}

impl CallSummary {
    /// No `[LLM call]` lines parsed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// The emitter's leading tag, ANSI-stripped — the line's only marker.
const CALL_TAG: &str = "[LLM call]";

// Single-parameter SGR only — the `log_style` wire convention. Stripping first
// (rather than threading `(?:\x1b\[\d+m)*` through every key, as `SPAN_RE`
// does) keeps a twelve-key line readable and handles either capture mode: logs
// written with `log_style::init(true)` carry no codes at all.
static SGR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[\d+m").expect("static SGR regex is valid"));

// One `key=value` chunk; no emitted value on this line contains whitespace.
static KV_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([a-z_]+)=(\S+)").expect("static kv regex is valid"));

/// Parse one `[LLM call]` line; `None` for any other line or a malformed one.
///
/// Malformed = missing `slot`/`model`/`status`, or a numeric key whose value
/// will not parse. Absent optional keys are simply `None`: `cached` and
/// `cal_ratio` ride only some calls, and a failed call reports no tokens.
#[must_use]
pub fn parse_call_line(line: &str) -> Option<CallRecord> {
    let plain = SGR_RE.replace_all(line, "");
    if !plain.contains(CALL_TAG) {
        return None;
    }
    let mut kv: BTreeMap<&str, &str> = BTreeMap::new();
    for caps in KV_RE.captures_iter(&plain) {
        let (Some(key), Some(val)) = (caps.get(1), caps.get(2)) else {
            continue;
        };
        kv.insert(key.as_str(), val.as_str());
    }

    // A key present but unparseable is corruption, not absence — flagged here
    // and checked once at the end, so the whole line is dropped.
    let corrupt = Cell::new(false);
    let rec = CallRecord {
        slot: (*kv.get("slot")?).to_owned(),
        model: (*kv.get("model")?).to_owned(),
        status: (*kv.get("status")?).to_owned(),
        ttfb_ms: num(&kv, "ttfb_ms", &corrupt),
        ttft_ms: num(&kv, "ttft_ms", &corrupt),
        total_ms: num(&kv, "total_ms", &corrupt),
        est_in_tokens: num(&kv, "est_in_tokens", &corrupt),
        in_tokens: num(&kv, "in_tokens", &corrupt),
        out_tokens: num(&kv, "out_tokens", &corrupt),
        cached: num(&kv, "cached", &corrupt),
        cal_ratio: num(&kv, "cal_ratio", &corrupt),
    };
    (!corrupt.get()).then_some(rec)
}

/// One optional numeric key. Absent → `None`; unparseable → `None` *and*
/// `corrupt` set.
fn num<T: std::str::FromStr>(
    kv: &BTreeMap<&str, &str>,
    key: &str,
    corrupt: &Cell<bool>,
) -> Option<T> {
    kv.get(key)
        .and_then(|raw| raw.parse().map_err(|_| corrupt.set(true)).ok())
}

/// Aggregate `[LLM call]` lines by `(slot, model)` and by model.
///
/// Non-call and malformed lines are skipped.
#[must_use]
pub fn aggregate_calls<I, S>(lines: I) -> CallSummary
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut summary = CallSummary::default();
    for line in lines {
        let Some(rec) = parse_call_line(line.as_ref()) else {
            continue;
        };
        ingest(&mut summary, &rec);
    }
    for group in summary.groups.values_mut() {
        group.ttfb.sort();
        group.ttft.sort();
    }
    summary
}

/// Fold one record into the group + model buckets.
fn ingest(summary: &mut CallSummary, rec: &CallRecord) {
    let group = summary
        .groups
        .entry((rec.slot.clone(), rec.model.clone()))
        .or_default();
    group.calls += 1;
    *group.by_status.entry(rec.status.clone()).or_default() += 1;

    // `cached > 0` is a hit; a usage-bearing call reporting no cached tokens is
    // a miss. Everything else (no usage reported) stays unscored.
    let hit = rec.cached.unwrap_or(0) > 0;
    if let Some(in_tokens) = rec.in_tokens {
        group.usage_calls += 1;
        group.in_sum += in_tokens;
        group.cached_sum += rec.cached.unwrap_or(0);
        if hit {
            group.hit_calls += 1;
        }
        for (bucket, sample) in [
            (&mut group.ttfb, rec.ttfb_ms),
            (&mut group.ttft, rec.ttft_ms),
        ] {
            if let Some(ms) = sample {
                if hit {
                    &mut bucket.hit
                } else {
                    &mut bucket.miss
                }
                .push(ms);
            }
        }
    }

    // Estimator bucket only for calls that carry something to compare — a
    // model with no usage-bearing call never reaches the table.
    let pair = rec.est_in_tokens.zip(rec.in_tokens);
    if pair.is_none() && rec.cal_ratio.is_none() {
        return;
    }
    let model = summary.models.entry(rec.model.clone()).or_default();
    if let Some(ratio) = rec.cal_ratio {
        model.cal_ratio = Some(ratio);
    }
    if let Some((est, actual)) = pair {
        model.calls += 1;
        model.est_sum += est;
        model.in_sum += actual;
    }
}

/// `[LLM call]` line fixture, shared by the `diagnostics` + `commands::diagnose`
/// unit tests.
#[cfg(test)]
pub(crate) mod fixture {
    use crate::log_style as ls;

    /// Byte-exact `[LLM call]` line, composed the way `CallMetrics::emit` does
    /// (same key order, same colours). Copying the emitter keeps the fixture
    /// honest about the wire format.
    #[derive(Clone)]
    pub struct Line {
        pub slot: &'static str,
        pub model: &'static str,
        pub status: &'static str,
        pub chars: i64,
        pub ttfb: Option<i64>,
        pub ttft: Option<i64>,
        pub total: Option<i64>,
        pub provider: Option<&'static str>,
        pub est_in: Option<i64>,
        pub in_tok: Option<i64>,
        pub out_tok: Option<i64>,
        pub cached: Option<i64>,
        pub cal_ratio: Option<f64>,
    }

    impl Default for Line {
        fn default() -> Self {
            Self {
                slot: "fast",
                model: "anthropic/claude-haiku-4.5",
                status: "ok",
                chars: 8000,
                ttfb: Some(400),
                ttft: Some(450),
                total: Some(900),
                provider: Some("anthropic"),
                est_in: Some(2000),
                in_tok: Some(2100),
                out_tok: Some(80),
                cached: Some(0),
                cal_ratio: Some(1.05),
            }
        }
    }

    impl Line {
        pub fn render(&self) -> String {
            let mut parts = vec![
                ls::tag("LLM call", ls::LM),
                ls::kv_styled("slot", self.slot, ls::W, ls::LC),
                ls::kv_styled("model", self.model, ls::W, ls::LW),
                ls::kv_styled(
                    "status",
                    self.status,
                    ls::W,
                    if self.status == "ok" { ls::LG } else { ls::R },
                ),
                ls::kv_styled("chars", &self.chars.to_string(), ls::W, ls::LC),
            ];
            for (key, val) in [
                ("ttfb_ms", self.ttfb),
                ("ttft_ms", self.ttft),
                ("total_ms", self.total),
            ] {
                if let Some(v) = val {
                    parts.push(ls::kv_styled(key, &v.to_string(), ls::W, ls::LC));
                }
            }
            if let Some(p) = self.provider {
                parts.push(ls::kv_styled("provider", p, ls::W, ls::LM));
            }
            if let Some(v) = self.est_in {
                parts.push(ls::kv_styled(
                    "est_in_tokens",
                    &v.to_string(),
                    ls::W,
                    ls::LC,
                ));
            }
            for (key, val) in [
                ("in_tokens", self.in_tok),
                ("out_tokens", self.out_tok),
                ("cached", self.cached),
            ] {
                if let Some(v) = val {
                    parts.push(ls::kv_styled(key, &v.to_string(), ls::W, ls::LW));
                }
            }
            if let Some(r) = self.cal_ratio {
                parts.push(ls::kv_styled(
                    "cal_ratio",
                    &format!("{r:.3}"),
                    ls::W,
                    ls::LC,
                ));
            }
            format!("2026-08-16 12:00:00 INFO {}", parts.join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::Line;
    use super::{CallSummary, aggregate_calls, parse_call_line};
    use crate::diagnostics::strip_ansi;
    use crate::log_style as ls;

    fn summary_of(lines: &[String]) -> CallSummary {
        aggregate_calls(lines)
    }

    fn group<'a>(summary: &'a CallSummary, slot: &str, model: &str) -> &'a super::CallGroupStats {
        summary
            .groups
            .get(&(slot.to_owned(), model.to_owned()))
            .expect("group present")
    }

    // --- parse ---

    #[test]
    fn parses_ansi_coloured_call_line() {
        let rec = parse_call_line(&Line::default().render()).expect("parsed");
        assert_eq!(rec.slot, "fast");
        assert_eq!(rec.model, "anthropic/claude-haiku-4.5");
        assert_eq!(rec.status, "ok");
        assert_eq!(rec.ttfb_ms, Some(400));
        assert_eq!(rec.ttft_ms, Some(450));
        assert_eq!(rec.total_ms, Some(900));
        assert_eq!(rec.est_in_tokens, Some(2000));
        assert_eq!(rec.in_tokens, Some(2100));
        assert_eq!(rec.out_tokens, Some(80));
        assert_eq!(rec.cached, Some(0));
        let ratio = rec.cal_ratio.expect("cal_ratio");
        assert!((ratio - 1.05).abs() < 1e-9);
    }

    #[test]
    fn parses_ansi_stripped_call_line_identically() {
        // Logs captured with ANSI stripping on must parse the same.
        let coloured = Line::default().render();
        let plain = strip_ansi(&coloured);
        assert!(!plain.contains('\x1b'));
        assert_eq!(parse_call_line(&coloured), parse_call_line(&plain));
        assert!(parse_call_line(&plain).is_some());
    }

    #[test]
    fn tolerates_absent_cached_cal_ratio_and_token_keys() {
        // A failed call: no timings, no usage, no cached, no cal_ratio.
        let line = Line {
            status: "error",
            ttfb: None,
            ttft: None,
            total: None,
            provider: None,
            est_in: Some(2000),
            in_tok: None,
            out_tok: None,
            cached: None,
            cal_ratio: None,
            ..Line::default()
        };
        let rec = parse_call_line(&line.render()).expect("parsed");
        assert_eq!(rec.status, "error");
        assert_eq!(rec.ttfb_ms, None);
        assert_eq!(rec.in_tokens, None);
        assert_eq!(rec.cached, None);
        assert_eq!(rec.cal_ratio, None);
        assert_eq!(rec.est_in_tokens, Some(2000));
    }

    #[test]
    fn non_call_lines_are_not_parsed() {
        assert!(parse_call_line("2026-08-16 INFO [span] span=llm ms=42 status=ok").is_none());
        assert!(parse_call_line("").is_none());
    }

    #[test]
    fn malformed_call_lines_are_skipped() {
        // Tag present but the required keys are missing.
        let headless = format!("INFO {} chars=10", ls::tag("LLM call", ls::LM));
        assert!(parse_call_line(&headless).is_none());
        // A numeric key with a non-numeric value: skip, never panic.
        let garbled = Line::default().render().replace("400", "four hundred");
        assert!(parse_call_line(&garbled).is_none());
        // Aggregation tolerates the same line among good ones.
        let good = Line::default().render();
        let summary = summary_of(&[headless, garbled, good]);
        assert_eq!(
            group(&summary, "fast", "anthropic/claude-haiku-4.5").calls,
            1
        );
    }

    // --- grouping ---

    #[test]
    fn groups_by_slot_then_model_across_a_model_change() {
        let lines = vec![
            Line::default().render(),
            Line::default().render(),
            // Mid-log model swap on the same slot.
            Line {
                model: "anthropic/claude-sonnet-4.5",
                ..Line::default()
            }
            .render(),
            Line {
                slot: "prose",
                model: "z-ai/glm-5.2",
                ..Line::default()
            }
            .render(),
        ];
        let summary = summary_of(&lines);
        assert_eq!(summary.groups.len(), 3);
        assert_eq!(
            group(&summary, "fast", "anthropic/claude-haiku-4.5").calls,
            2
        );
        assert_eq!(
            group(&summary, "fast", "anthropic/claude-sonnet-4.5").calls,
            1
        );
        assert_eq!(group(&summary, "prose", "z-ai/glm-5.2").calls, 1);
    }

    #[test]
    fn status_counts_are_open_vocabulary() {
        let lines: Vec<String> = ["ok", "error", "cancelled", "silent", "suppressed", "ok"]
            .into_iter()
            .map(|status| {
                Line {
                    status,
                    ..Line::default()
                }
                .render()
            })
            .collect();
        let summary = summary_of(&lines);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        assert_eq!(g.calls, 6);
        assert_eq!(g.by_status["ok"], 2);
        assert_eq!(g.by_status["suppressed"], 1);
        assert_eq!(g.by_status["silent"], 1);
    }

    // --- cache rates ---

    #[test]
    fn reports_call_and_token_weighted_hit_rates() {
        // Three usage-bearing calls; one reports cached tokens.
        let lines = vec![
            Line {
                in_tok: Some(1000),
                cached: Some(0),
                ..Line::default()
            }
            .render(),
            Line {
                in_tok: Some(1000),
                cached: Some(0),
                ..Line::default()
            }
            .render(),
            Line {
                in_tok: Some(1000),
                cached: Some(600),
                ..Line::default()
            }
            .render(),
        ];
        let summary = summary_of(&lines);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        // 1 of 3 calls hit.
        assert!((g.call_hit_rate().expect("rate") - 1.0 / 3.0).abs() < 1e-9);
        // 600 of 3000 prompt tokens reused.
        assert!((g.token_hit_rate().expect("rate") - 0.2).abs() < 1e-9);
        assert!((g.mean_in_tokens().expect("mean") - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn calls_without_usage_do_not_score_as_misses() {
        let lines = vec![
            Line {
                status: "error",
                in_tok: None,
                cached: None,
                ..Line::default()
            }
            .render(),
            Line {
                in_tok: Some(500),
                cached: Some(500),
                ..Line::default()
            }
            .render(),
        ];
        let summary = summary_of(&lines);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        assert_eq!(g.calls, 2);
        assert_eq!(g.usage_calls, 1);
        assert!((g.call_hit_rate().expect("rate") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hit_rates_absent_without_usage_bearing_calls() {
        let line = Line {
            status: "cancelled",
            in_tok: None,
            out_tok: None,
            cached: None,
            cal_ratio: None,
            ..Line::default()
        };
        let summary = summary_of(&[line.render()]);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        assert!(g.call_hit_rate().is_none());
        assert!(g.token_hit_rate().is_none());
        assert!(g.mean_in_tokens().is_none());
    }

    // --- latency split ---

    #[test]
    fn splits_latency_percentiles_by_cache_hit_and_miss() {
        let mk = |ttfb: i64, ttft: i64, cached: i64| {
            Line {
                ttfb: Some(ttfb),
                ttft: Some(ttft),
                cached: Some(cached),
                ..Line::default()
            }
            .render()
        };
        let lines = vec![
            mk(100, 150, 900),
            mk(200, 250, 900),
            mk(500, 600, 0),
            mk(700, 800, 0),
        ];
        let summary = summary_of(&lines);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        assert_eq!(g.ttfb.hit.len(), 2);
        assert_eq!(g.ttfb.miss.len(), 2);
        assert!((g.ttfb.hit_p(50).expect("p50") - 150.0).abs() < 1e-9);
        assert!((g.ttfb.miss_p(50).expect("p50") - 600.0).abs() < 1e-9);
        assert!((g.ttfb.miss_p(95).expect("p95") - 690.0).abs() < 1e-9);
        // Cost of a miss at p50.
        assert!((g.ttfb.cost_p50().expect("cost") - 450.0).abs() < 1e-9);
        assert!((g.ttft.hit_p(50).expect("p50") - 200.0).abs() < 1e-9);
        assert!((g.ttft.cost_p50().expect("cost") - 500.0).abs() < 1e-9);
    }

    #[test]
    fn cost_absent_when_one_side_has_no_samples() {
        let summary = summary_of(&[Line {
            cached: Some(900),
            ..Line::default()
        }
        .render()]);
        let g = group(&summary, "fast", "anthropic/claude-haiku-4.5");
        assert!(g.ttfb.miss_p(50).is_none());
        assert!(g.ttfb.cost_p50().is_none());
        assert!(!g.ttfb.is_empty());
    }

    // --- estimator accuracy ---

    #[test]
    fn reports_observed_ratio_and_last_cal_ratio_per_model() {
        let lines = vec![
            Line {
                est_in: Some(1000),
                in_tok: Some(1100),
                cal_ratio: Some(1.100),
                ..Line::default()
            }
            .render(),
            Line {
                slot: "prose",
                est_in: Some(1000),
                in_tok: Some(1300),
                cal_ratio: Some(1.200),
                ..Line::default()
            }
            .render(),
        ];
        let summary = summary_of(&lines);
        let est = &summary.models["anthropic/claude-haiku-4.5"];
        assert_eq!(est.calls, 2);
        // 2400 true over 2000 estimated, across both slots.
        assert!((est.observed_ratio().expect("ratio") - 1.2).abs() < 1e-9);
        // Last-seen running ratio, not the first.
        assert!((est.cal_ratio.expect("cal") - 1.2).abs() < 1e-9);
    }

    #[test]
    fn empty_input_summarises_to_nothing() {
        assert!(aggregate_calls(Vec::<String>::new()).is_empty());
        assert!(aggregate_calls(["not a call line"]).is_empty());
    }
}
