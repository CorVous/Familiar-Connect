//! Plain-text renderers for span summaries.
//!
//! Shared by the `/diagnostics` slash command and the `familiar-connect
//! diagnose` CLI — both consume
//! [`SpanCollector::summary`](super::collector::SpanCollector::summary) output
//! (or log-file aggregates in the same [`SpanStats`] shape) and both want the
//! same terse, Discord-friendly code-fenced table.
//!
//! [`render_llm_call_report`] renders the `diagnose`-only prompt-cache tables
//! over [`CallSummary`] in the same style.

use std::collections::BTreeMap;

use crate::diagnostics::collector::SpanStats;
use crate::diagnostics::llm_calls::CallSummary;

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

/// Column alignment for [`fenced_table`].
#[derive(Clone, Copy)]
enum Align {
    /// Text columns (slot, model, status).
    Left,
    /// Numeric columns.
    Right,
}

/// Code-fenced fixed-width table: header, dash rule, rows.
///
/// Column width = max(header, cells); two-space gaps; trailing padding trimmed
/// so no row carries dead whitespace. Mirrors [`render_summary_table`]'s shape.
fn fenced_table(headers: &[(&str, Align)], rows: &[Vec<String>]) -> String {
    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(col, (label, _))| {
            rows.iter()
                .filter_map(|row| row.get(col))
                .map(|cell| cell.chars().count())
                .chain(std::iter::once(label.chars().count()))
                .max()
                .unwrap_or(0)
        })
        .collect();
    let pad = |cell: &str, col: usize| -> String {
        let fill = " ".repeat(widths[col].saturating_sub(cell.chars().count()));
        match headers[col].1 {
            Align::Left => format!("{cell}{fill}"),
            Align::Right => format!("{fill}{cell}"),
        }
    };
    let row_text = |cells: &[String]| -> String {
        let joined: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(col, cell)| pad(cell, col))
            .collect();
        joined.join("  ").trim_end().to_owned()
    };

    let header: Vec<String> = headers
        .iter()
        .map(|(label, _)| (*label).to_owned())
        .collect();
    // Rule spans the full padded width, not the trimmed header.
    let rule_width = widths.iter().sum::<usize>() + 2 * widths.len().saturating_sub(1);
    let mut lines = vec!["```".to_owned(), row_text(&header), "-".repeat(rule_width)];
    lines.extend(rows.iter().map(|row| row_text(row)));
    lines.push("```".to_owned());
    lines.join("\n")
}

/// `-` for an absent value, else `value` at `places` decimals.
fn opt(value: Option<f64>, places: usize) -> String {
    value.map_or_else(|| "-".to_owned(), |v| format!("{v:.places$}"))
}

/// `-` for an absent rate, else the percentage at one decimal.
fn opt_pct(value: Option<f64>) -> String {
    opt(value.map(|v| v * 100.0), 1)
}

/// Section 1 — per `(slot, model)`: call count, both cache-hit rates, mean
/// prompt size, status breakdown.
fn call_overview_section(summary: &CallSummary) -> String {
    let rows: Vec<Vec<String>> = summary
        .groups
        .iter()
        .map(|((slot, model), stats)| {
            let status = stats
                .by_status
                .iter()
                .map(|(name, n)| format!("{name}={n}"))
                .collect::<Vec<_>>()
                .join(" ");
            vec![
                slot.clone(),
                model.clone(),
                stats.calls.to_string(),
                opt_pct(stats.call_hit_rate()),
                opt_pct(stats.token_hit_rate()),
                opt(stats.mean_in_tokens(), 0),
                status,
            ]
        })
        .collect();
    format!(
        "LLM calls by slot / model\n{}",
        fenced_table(
            &[
                ("slot", Align::Left),
                ("model", Align::Left),
                ("n", Align::Right),
                ("hit%", Align::Right),
                ("ctok%", Align::Right),
                ("in_tok", Align::Right),
                ("status", Align::Left),
            ],
            &rows,
        )
    )
}

/// Section 2 — `ttfb_ms`/`ttft_ms` percentiles split by cache hit vs miss, plus
/// the p50 cost of a miss. Empty when no call reported a timing.
fn latency_split_section(summary: &CallSummary) -> Option<String> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for ((slot, model), stats) in &summary.groups {
        for (metric, split) in [("ttfb", &stats.ttfb), ("ttft", &stats.ttft)] {
            if split.is_empty() {
                continue;
            }
            rows.push(vec![
                slot.clone(),
                model.clone(),
                metric.to_owned(),
                split.hit.len().to_string(),
                opt(split.hit_p(50), 0),
                opt(split.hit_p(95), 0),
                split.miss.len().to_string(),
                opt(split.miss_p(50), 0),
                opt(split.miss_p(95), 0),
                opt(split.cost_p50(), 0),
            ]);
        }
    }
    (!rows.is_empty()).then(|| {
        format!(
            "cache hit vs miss latency (ms)\n{}",
            fenced_table(
                &[
                    ("slot", Align::Left),
                    ("model", Align::Left),
                    ("metric", Align::Left),
                    ("hit_n", Align::Right),
                    ("hit_p50", Align::Right),
                    ("hit_p95", Align::Right),
                    ("miss_n", Align::Right),
                    ("miss_p50", Align::Right),
                    ("miss_p95", Align::Right),
                    ("cost_p50", Align::Right),
                ],
                &rows,
            )
        )
    })
}

/// Section 3 — per model: observed true/estimated token ratio next to the
/// logged `cal_ratio` (#184 item 3). Empty when no call reported usage.
fn estimator_section(summary: &CallSummary) -> Option<String> {
    if summary.models.is_empty() {
        return None;
    }
    let rows: Vec<Vec<String>> = summary
        .models
        .iter()
        .map(|(model, stats)| {
            vec![
                model.clone(),
                stats.calls.to_string(),
                stats.est_sum.to_string(),
                stats.in_sum.to_string(),
                opt(stats.observed_ratio(), 3),
                opt(stats.cal_ratio, 3),
            ]
        })
        .collect();
    Some(format!(
        "token estimator accuracy by model\n{}",
        fenced_table(
            &[
                ("model", Align::Left),
                ("n", Align::Right),
                ("est_tok", Align::Right),
                ("in_tok", Align::Right),
                ("obs", Align::Right),
                ("cal", Align::Right),
            ],
            &rows,
        )
    ))
}

/// Render the `[LLM call]` cache/latency report as up to three code-fenced
/// tables. Empty summary → empty string (the caller prints nothing).
///
/// Reading it for #206: see `docs/architecture/tuning.md`.
#[must_use]
pub fn render_llm_call_report(summary: &CallSummary) -> String {
    if summary.is_empty() {
        return String::new();
    }
    let sections = [
        Some(call_overview_section(summary)),
        latency_split_section(summary),
        estimator_section(summary),
    ];
    sections
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::{render_llm_call_report, render_summary_table};
    use crate::diagnostics::collector::SpanStats;
    use crate::diagnostics::llm_calls::{CallSummary, aggregate_calls, fixture::Line};
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

    // --- render_llm_call_report ---

    fn two_path_summary() -> CallSummary {
        let lines = vec![
            // fast: every call a full miss (the #206 hypothesis's signature).
            Line {
                ttfb: Some(600),
                in_tok: Some(4000),
                cached: Some(0),
                ..Line::default()
            }
            .render(),
            Line {
                ttfb: Some(700),
                in_tok: Some(4000),
                cached: Some(0),
                ..Line::default()
            }
            .render(),
            // prose: a warm prefix.
            Line {
                slot: "prose",
                model: "z-ai/glm-5.2",
                ttfb: Some(300),
                in_tok: Some(4000),
                cached: Some(3600),
                ..Line::default()
            }
            .render(),
        ];
        aggregate_calls(&lines)
    }

    #[test]
    fn llm_report_is_blank_without_calls() {
        assert_eq!(render_llm_call_report(&CallSummary::default()), "");
    }

    #[test]
    fn llm_report_renders_all_three_sections() {
        let out = render_llm_call_report(&two_path_summary());
        assert!(out.contains("LLM calls by slot / model"));
        assert!(out.contains("cache hit vs miss latency (ms)"));
        assert!(out.contains("token estimator accuracy by model"));
        // Fenced like the span table.
        assert!(out.starts_with("LLM calls"));
        assert!(out.ends_with("```"));
    }

    #[test]
    fn llm_report_shows_rates_status_and_cost() {
        let out = render_llm_call_report(&two_path_summary());
        assert!(out.contains("fast"), "{out}");
        assert!(out.contains("prose"), "{out}");
        assert!(out.contains("z-ai/glm-5.2"), "{out}");
        // fast: 0% of calls and 0% of tokens cached; prose: 100% / 90%.
        assert!(out.contains("0.0"), "{out}");
        assert!(out.contains("90.0"), "{out}");
        // Status breakdown rides the last column.
        assert!(out.contains("ok=2"), "{out}");
        // Both latency metrics get a row.
        assert!(out.contains("ttfb"), "{out}");
        assert!(out.contains("ttft"), "{out}");
        // Estimator: 2100/2000 default pair → observed 1.050.
        assert!(out.contains("1.050"), "{out}");
    }

    #[test]
    fn llm_report_marks_absent_values_with_a_dash() {
        // A cancelled call reports no usage and no timings at all.
        let lines = vec![
            Line {
                status: "cancelled",
                ttfb: None,
                ttft: None,
                total: None,
                in_tok: None,
                out_tok: None,
                cached: None,
                cal_ratio: None,
                ..Line::default()
            }
            .render(),
        ];
        let out = render_llm_call_report(&aggregate_calls(&lines));
        assert!(out.contains("cancelled=1"), "{out}");
        assert!(out.contains('-'), "{out}");
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
