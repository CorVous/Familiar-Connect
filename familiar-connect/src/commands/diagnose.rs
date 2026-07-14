//! diagnose subcommand: log grep + summary table (subsystem 10; Python
//! commands/diagnose.py).
//!
//! Reads `span=<name> … ms=<int> … status=<word>` markers from one or more log
//! files (or stdin for `-`), groups by span, and prints the same code-fenced
//! `count / p50 / p95 / last` table `/diagnostics` renders. The percentile /
//! aggregation function is the one ported once in
//! [`crate::diagnostics::collector`] and shared here (spec 01 §24, §44); the
//! in-process [`SpanCollector`](crate::diagnostics::collector::SpanCollector)
//! resets on restart, so the durable log is the only cross-run record.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::diagnostics::collector::{SpanStats, percentile};
use crate::diagnostics::report::render_summary_table;

/// Matches `span=<name>` + `ms=<int>` + `status=<word>` KV markers, tolerating
/// interleaved single-parameter ANSI codes and arbitrary intervening tokens
/// (DOTALL). Byte-for-byte the Python `_SPAN_RE` (`commands/diagnose.py`).
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
/// the maximum — mirroring the Python `last_ms[name] = ms` overwrite.
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
/// logs an error and is skipped (the rest still aggregate), mirroring Python's
/// `_iter_lines`.
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
                // U+FFFD and every line still aggregates — mirroring Python's
                // `open(..., errors="replace")` (spec 01 §44). A `BufRead::lines()`
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

/// Aggregate the given log files and print the summary table; always `0`.
#[must_use]
pub fn diagnose(paths: &[String]) -> i32 {
    let summary = aggregate(read_lines(paths));
    println!("{}", render_summary_table(&summary));
    0
}

#[cfg(test)]
mod tests {
    use super::{aggregate, diagnose};
    use crate::diagnostics::report::render_summary_table;

    fn span_line(name: &str, ms: i64, status: &str) -> String {
        // The `_SPAN_LINE` template from test_diagnose_cmd.py, filled directly.
        format!("2026-04-22 12:00:00 INFO [span] span={name} ms={ms} status={status}")
    }

    // --- aggregate (ported from test_diagnose_cmd.py::TestAggregate) ---

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
        // Byte-exact ANSI line from test_diagnose_cmd.py.
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

    // --- diagnose CLI (ported from test_diagnose_cmd.py::TestDiagnoseCLI) ---

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
        // aggregation: Python opens with errors="replace" and yields every line,
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

    #[test]
    fn unreadable_file_is_skipped() {
        // A missing path logs an error and yields an empty summary (exit 0).
        let paths = vec!["/nonexistent/does-not-exist.log".to_owned()];
        assert_eq!(diagnose(&paths), 0);
    }
}
