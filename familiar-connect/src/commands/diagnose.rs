//! diagnose subcommand: log grep + summary tables (subsystem 10).
//!
//! Reads `span=<name> … ms=<int> … status=<word>` markers from one or more log
//! files (or stdin for `-`), groups by span, and prints the same code-fenced
//! `count / p50 / p95 / last` table `/diagnostics` renders. The percentile /
//! aggregation function is the one ported once in
//! [`crate::diagnostics::collector`] and shared here; the
//! in-process [`SpanCollector`](crate::diagnostics::collector::SpanCollector)
//! resets on restart, so the durable log is the only cross-run record.
//!
//! A log carrying `[LLM call]` lines gets a second pass
//! ([`crate::diagnostics::llm_calls`]) and prints the prompt-cache tables after
//! the span table — the #206 measurement surface. A span-only log renders
//! exactly as before.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::diagnostics::collector::{SpanStats, percentile};
use crate::diagnostics::llm_calls::aggregate_calls;
use crate::diagnostics::report::{render_llm_call_report, render_summary_table};

/// Matches `span=<name>` + `ms=<int>` + `status=<word>` KV markers, tolerating
/// interleaved single-parameter ANSI codes and arbitrary intervening tokens
/// (DOTALL).
static SPAN_RE: LazyLock<Regex> = LazyLock::new(|| {
    // `_ANSI = (?:\x1b\[\d+m)*` — zero or more single-parameter SGR codes.
    Regex::new(concat!(
        r"(?s)",
        r"span=(?:\x1b\[\d+m)*(?P<name>[\w.\-]+)",
        r".*?",
        r"ms=(?:\x1b\[\d+m)*(?P<ms>\d+)",
        r".*?",
        r"status=(?:\x1b\[\d+m)*(?P<status>\w+)",
    ))
    .expect("static span regex is valid")
});

/// Build a `{name: SpanStats}` summary from log lines, matching
/// `SpanCollector::summary`'s shape.
///
/// `last_ms` is the most recently *seen* value for the span (file order), not
/// the maximum — each occurrence overwrites the previous.
#[must_use]
pub fn aggregate<I, S>(lines: I) -> BTreeMap<String, SpanStats>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut buckets: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    let mut last_ms: BTreeMap<String, i64> = BTreeMap::new();
    for line in lines {
        let Some(caps) = SPAN_RE.captures(line.as_ref()) else {
            continue;
        };
        let name = caps["name"].to_owned();
        // `ms` is `\d+`; parse failure only on overflow, which we skip.
        let Ok(ms) = caps["ms"].parse::<i64>() else {
            continue;
        };
        buckets.entry(name.clone()).or_default().push(ms);
        last_ms.insert(name, ms);
    }

    let mut summary = BTreeMap::new();
    for (name, mut ms_list) in buckets {
        ms_list.sort_unstable();
        let last = last_ms.get(&name).copied().unwrap_or(0);
        summary.insert(
            name,
            SpanStats {
                #[allow(clippy::cast_precision_loss)] // counts are small
                count: ms_list.len() as f64,
                p50: percentile(&ms_list, 50),
                p95: percentile(&ms_list, 95),
                #[allow(clippy::cast_precision_loss)] // ms fits f64 exactly here
                last_ms: last as f64,
            },
        );
    }
    summary
}

/// Yield the lines of every path in order; `-` reads stdin. An unreadable file
/// logs an error and is skipped (the rest still aggregate).
fn read_lines(paths: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for path in paths {
        if path == "-" {
            let mut buf = String::new();
            if std::io::stdin().read_to_string(&mut buf).is_ok() {
                out.extend(buf.lines().map(str::to_owned));
            }
            continue;
        }
        match std::fs::read(Path::new(path)) {
            Ok(bytes) => {
                // Lossy-decode so an invalid UTF-8 byte replaces itself with
                // U+FFFD and every line still aggregates. A `BufRead::lines`
                // + `map_while(Result::ok)` would instead STOP at the first bad
                // byte, silently dropping every later span.
                let text = String::from_utf8_lossy(&bytes);
                out.extend(text.lines().map(str::to_owned));
            }
            Err(err) => {
                tracing::error!("could not read {path}: {err}");
            }
        }
    }
    out
}

/// The whole printed report: the span table, plus the `[LLM call]` cache tables
/// when the log carries any such line.
///
/// A span-only log renders byte-identically to the span table alone — the
/// existing `diagnose` contract.
#[must_use]
pub fn render_report(lines: &[String]) -> String {
    let mut out = render_summary_table(&aggregate(lines));
    let calls = aggregate_calls(lines);
    if !calls.is_empty() {
        out.push_str("\n\n");
        out.push_str(&render_llm_call_report(&calls));
    }
    out
}

/// Aggregate the given log files and print the report; always `0`.
#[must_use]
pub fn diagnose(paths: &[String]) -> i32 {
    println!("{}", render_report(&read_lines(paths)));
    0
}

#[cfg(test)]
mod tests {
    use super::{aggregate, diagnose, render_report};
    use crate::diagnostics::llm_calls::fixture::Line;
    use crate::diagnostics::report::render_summary_table;

    fn span_line(name: &str, ms: i64, status: &str) -> String {
        // The span-line template, filled directly.
        format!("2026-04-22 12:00:00 INFO [span] span={name} ms={ms} status={status}")
    }

    // --- aggregate ---

    #[test]
    fn parses_simple_span_lines() {
        let lines = vec![
            span_line("llm", 100, "ok"),
            span_line("llm", 200, "ok"),
            span_line("tts", 80, "ok"),
            "junk line".to_owned(),
        ];
        let summary = aggregate(lines);
        assert!((summary["llm"].count - 2.0).abs() < f64::EPSILON);
        assert!((summary["tts"].count - 1.0).abs() < f64::EPSILON);
        // p50 of [100, 200] = 150 (linear-interpolated).
        assert!((summary["llm"].p50 - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tolerates_ansi_coloured_lines() {
        // Byte-exact ANSI line.
        let line = "\x1b[37mspan=\x1b[0m\x1b[95mllm\x1b[0m \
             \x1b[37mms=\x1b[0m\x1b[96m42\x1b[0m \
             \x1b[37mstatus=\x1b[0m\x1b[32mok\x1b[0m";
        let summary = aggregate([line]);
        assert!((summary["llm"].count - 1.0).abs() < f64::EPSILON);
        assert!((summary["llm"].last_ms - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn last_ms_is_most_recent_not_max() {
        let summary = aggregate([span_line("llm", 500, "ok"), span_line("llm", 30, "ok")]);
        assert!((summary["llm"].last_ms - 30.0).abs() < f64::EPSILON);
    }

    // --- diagnose CLI ---

    #[test]
    fn runs_against_a_log_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("bot.log");
        let body = format!(
            "{}\n{}\n{}\n",
            span_line("llm", 50, "ok"),
            span_line("llm", 150, "ok"),
            span_line("tts", 20, "ok"),
        );
        std::fs::write(&log_path, body).expect("write log");
        let paths = vec![log_path.to_string_lossy().into_owned()];
        assert_eq!(diagnose(&paths), 0);
        // Table content check is on the shared renderer (stdout is not captured).
        let summary = aggregate(std::fs::read_to_string(&log_path).unwrap().lines());
        let table = render_summary_table(&summary);
        assert!(table.contains("llm"));
        assert!(table.contains("tts"));
    }

    #[test]
    fn runs_against_multiple_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.log");
        let b = dir.path().join("b.log");
        std::fs::write(&a, format!("{}\n", span_line("llm", 10, "ok"))).unwrap();
        std::fs::write(&b, format!("{}\n", span_line("llm", 30, "ok"))).unwrap();
        let paths = vec![
            a.to_string_lossy().into_owned(),
            b.to_string_lossy().into_owned(),
        ];
        assert_eq!(diagnose(&paths), 0);
        let mut lines = std::fs::read_to_string(&a).unwrap();
        lines.push_str(&std::fs::read_to_string(&b).unwrap());
        let summary = aggregate(lines.lines());
        assert!((summary["llm"].count - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_log_shows_placeholder() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("empty.log");
        std::fs::write(&log, "nothing here\n").unwrap();
        let paths = vec![log.to_string_lossy().into_owned()];
        assert_eq!(diagnose(&paths), 0);
        // The empty summary renders the "no spans" placeholder.
        let summary = aggregate(std::fs::read_to_string(&log).unwrap().lines());
        assert!(render_summary_table(&summary).contains("no spans"));
    }

    #[test]
    fn aggregates_lines_after_invalid_utf8() {
        // A partial-write / mixed-encoding byte mid-file must not truncate the
        // aggregation: lossy decoding yields every line,
        // so spans AFTER the bad byte still count. (Regression: a prior
        // `lines().map_while(Result::ok)` stopped at the first decode error.)
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("mixed.log");
        let mut bytes = span_line("llm", 100, "ok").into_bytes();
        bytes.push(b'\n');
        bytes.push(0xFF); // lone invalid UTF-8 byte
        bytes.push(b'\n');
        bytes.extend_from_slice(span_line("llm", 200, "ok").as_bytes());
        bytes.push(b'\n');
        std::fs::write(&log, &bytes).expect("write log");
        let paths = vec![log.to_string_lossy().into_owned()];
        assert_eq!(diagnose(&paths), 0);
        let summary = aggregate(super::read_lines(&paths));
        // Both spans survive the bad byte between them.
        assert!((summary["llm"].count - 2.0).abs() < f64::EPSILON);
        assert!((summary["llm"].last_ms - 200.0).abs() < f64::EPSILON);
    }

    // --- render_report: the span table stays untouched ---

    #[test]
    fn span_only_log_renders_byte_identically_to_the_span_table() {
        // The #206 addition must not perturb the existing contract: with no
        // `[LLM call]` line in the log, output is exactly today's table.
        let lines = vec![
            span_line("llm", 50, "ok"),
            span_line("llm", 150, "ok"),
            span_line("tts", 20, "ok"),
            "junk line".to_owned(),
        ];
        assert_eq!(
            render_report(&lines),
            render_summary_table(&aggregate(&lines))
        );
    }

    #[test]
    fn log_with_no_recognised_lines_renders_only_the_placeholder() {
        let lines = vec!["nothing here".to_owned()];
        assert_eq!(
            render_report(&lines),
            render_summary_table(&aggregate(&lines))
        );
        assert!(render_report(&lines).contains("no spans"));
    }

    #[test]
    fn llm_call_lines_append_the_cache_report() {
        let lines = vec![span_line("llm", 50, "ok"), Line::default().render()];
        let out = render_report(&lines);
        // Span table first, unchanged, then the new report.
        let span_table = render_summary_table(&aggregate(&lines));
        assert!(out.starts_with(&span_table), "{out}");
        assert!(out.contains("LLM calls by slot / model"), "{out}");
        assert!(out.contains("anthropic/claude-haiku-4.5"), "{out}");
    }

    #[test]
    fn unreadable_file_is_skipped() {
        // A missing path logs an error and yields an empty summary (exit 0).
        let paths = vec!["/nonexistent/does-not-exist.log".to_owned()];
        assert_eq!(diagnose(&paths), 0);
    }
}
