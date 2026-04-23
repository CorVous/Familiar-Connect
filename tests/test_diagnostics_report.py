"""Tests for the shared span-summary text renderer."""

from __future__ import annotations

from familiar_connect.diagnostics.report import render_summary_table


class TestRenderSummaryTable:
    def test_empty_summary_produces_placeholder(self) -> None:
        out = render_summary_table({})
        assert "no spans" in out
        assert out.startswith("```")
        assert out.endswith("```")

    def test_rows_sorted_by_name(self) -> None:
        out = render_summary_table({
            "zeta": {"count": 1, "p50": 10, "p95": 10, "last_ms": 10},
            "alpha": {"count": 2, "p50": 5, "p95": 5, "last_ms": 5},
        })
        alpha_line = next(line for line in out.splitlines() if "alpha" in line)
        zeta_line = next(line for line in out.splitlines() if "zeta" in line)
        assert out.index(alpha_line) < out.index(zeta_line)

    def test_renders_expected_columns(self) -> None:
        out = render_summary_table({
            "llm": {"count": 3, "p50": 12.5, "p95": 30.0, "last_ms": 18}
        })
        assert "span" in out
        assert "p50" in out
        assert "p95" in out
        assert "last" in out
        assert "llm" in out
        assert " 3 " in out  # count
