//! Plain-text renderers for span summaries.
//!
//! Port of `familiar_connect/diagnostics/report.py`. Shared by the
//! `/diagnostics` slash command and the `familiar-connect diagnose` CLI — both
//! consume [`SpanCollector::summary`](super::collector::SpanCollector::summary)
//! output (or log-file aggregates in the same [`SpanStats`] shape) and both want
//! the same terse, Discord-friendly code-fenced table.

use std::collections::BTreeMap;

use crate::diagnostics::collector::SpanStats;

/// Render `{name: {count, p50, p95, last_ms}}` as a code-fenced monospace table.
///
/// Empty summary → the exact placeholder ```` ```\nno spans recorded yet\n``` ````.
/// Otherwise: rows sorted by name (the `BTreeMap` is already sorted); name
/// column width = `max(longest name, len("span"))`; count and last as integers,
/// p50/p95 as `%.0f`; two-space column gaps.
#[must_use]
pub fn render_summary_table(summary: &BTreeMap<String, SpanStats>) -> String {
    if summary.is_empty() {
        return "```\nno spans recorded yet\n```".to_string();
    }

    let name_width = summary
        .keys()
        .map(|name| name.chars().count())
        .max()
        .unwrap_or(0)
        .max("span".len());

    let header = format!(
        "{:<name_width$}  {:>5}  {:>6}  {:>6}  {:>6}",
        "span", "n", "p50", "p95", "last"
    );
    let sep = "-".repeat(header.chars().count());
    let mut lines = vec!["```".to_string(), header, sep];
    for (name, stats) in summary {
        // count and last render as integers (truncated); p50/p95 as %.0f.
        #[allow(clippy::cast_possible_truncation)]
        let count = stats.count as i64;
        #[allow(clippy::cast_possible_truncation)]
        let last = stats.last_ms as i64;
        lines.push(format!(
            "{name:<name_width$}  {count:>5}  {:>6.0}  {:>6.0}  {last:>6}",
            stats.p50, stats.p95
        ));
    }
    lines.push("```".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::render_summary_table;
    use crate::diagnostics::collector::SpanStats;
    use std::collections::BTreeMap;

    fn stat(count: f64, p50: f64, p95: f64, last_ms: f64) -> SpanStats {
        SpanStats {
            count,
            p50,
            p95,
            last_ms,
        }
    }

    #[test]
    fn empty_summary_produces_placeholder() {
        let out = render_summary_table(&BTreeMap::new());
        assert!(out.contains("no spans"));
        assert!(out.starts_with("```"));
        assert!(out.ends_with("```"));
    }

    #[test]
    fn rows_sorted_by_name() {
        let mut summary = BTreeMap::new();
        summary.insert("zeta".to_string(), stat(1.0, 10.0, 10.0, 10.0));
        summary.insert("alpha".to_string(), stat(2.0, 5.0, 5.0, 5.0));
        let out = render_summary_table(&summary);
        let alpha_at = out.find("alpha").expect("alpha present");
        let zeta_at = out.find("zeta").expect("zeta present");
        assert!(alpha_at < zeta_at);
    }

    #[test]
    fn renders_expected_columns() {
        let mut summary = BTreeMap::new();
        summary.insert("llm".to_string(), stat(3.0, 12.5, 30.0, 18.0));
        let out = render_summary_table(&summary);
        assert!(out.contains("span"));
        assert!(out.contains("p50"));
        assert!(out.contains("p95"));
        assert!(out.contains("last"));
        assert!(out.contains("llm"));
        assert!(out.contains(" 3 ")); // count
    }
}
