"""Plain-text renderers for span summaries.

Shared by :meth:`/diagnostics` slash command and the
``familiar-connect diagnose`` CLI — both consume
:class:`SpanCollector.summary` output or log-file aggregates, and
both want the same terse, Discord-friendly table.
"""

from __future__ import annotations


def render_summary_table(summary: dict[str, dict[str, float]]) -> str:
    """Render ``{name: {count, p50, p95, last_ms}}`` as a code-fenced table.

    Discord renders triple-backtick blocks as monospace, which lines
    up columns reliably across web / desktop / mobile clients.
    """
    if not summary:
        return "```\nno spans recorded yet\n```"

    rows = sorted(summary.items())
    name_width = max(len(name) for name, _ in rows)
    name_width = max(name_width, len("span"))

    header = f"{'span':<{name_width}}  {'n':>5}  {'p50':>6}  {'p95':>6}  {'last':>6}"
    sep = "-" * len(header)
    lines = ["```", header, sep]
    for name, stats in rows:
        lines.append(
            f"{name:<{name_width}}  "
            f"{int(stats['count']):>5}  "
            f"{stats['p50']:>6.0f}  "
            f"{stats['p95']:>6.0f}  "
            f"{int(stats['last_ms']):>6}"
        )
    lines.append("```")
    return "\n".join(lines)
