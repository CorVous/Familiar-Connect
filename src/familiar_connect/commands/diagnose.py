"""``familiar-connect diagnose`` — aggregate span timings from log files.

Reads one or more log files for ``span=<name> ms=<int>`` markers,
groups by span name, and prints a p50/p95 histogram. Logs are the
durable record of span timings (the in-process
:class:`SpanCollector` is for the live ``/diagnostics`` slash command
only and gets wiped on restart).

Usage::

    familiar-connect diagnose path/to/bot.log

Multiple files combine into one aggregate. ``-`` reads from stdin.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.diagnostics.report import render_summary_table

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable

_logger = logging.getLogger(__name__)

# Matches log lines carrying ``span=<name>`` and ``ms=<int>`` KV pairs,
# tolerating intervening ANSI codes and other tokens.
_ANSI = r"(?:\x1b\[\d+m)*"
_SPAN_RE = re.compile(
    rf"span={_ANSI}(?P<name>[\w.-]+)"
    r".*?"
    rf"ms={_ANSI}(?P<ms>\d+)"
    r".*?"
    rf"status={_ANSI}(?P<status>\w+)",
    re.DOTALL,
)


def _aggregate(lines: Iterable[str]) -> dict[str, dict[str, float]]:
    """Return the same summary shape as :meth:`SpanCollector.summary`."""
    buckets: dict[str, list[int]] = {}
    last_ms: dict[str, int] = {}
    for line in lines:
        m = _SPAN_RE.search(line)
        if m is None:
            continue
        name = m.group("name")
        ms = int(m.group("ms"))
        buckets.setdefault(name, []).append(ms)
        last_ms[name] = ms
    summary: dict[str, dict[str, float]] = {}
    for name, ms_list in buckets.items():
        ms_sorted = sorted(ms_list)
        summary[name] = {
            "count": len(ms_sorted),
            "p50": _percentile(ms_sorted, 50),
            "p95": _percentile(ms_sorted, 95),
            "last_ms": last_ms[name],
        }
    return summary


def _percentile(sorted_values: list[int], pct: int) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _iter_lines(paths: list[str]) -> Iterable[str]:
    for path in paths:
        if path == "-":
            yield from sys.stdin
            continue
        try:
            with Path(path).open(encoding="utf-8", errors="replace") as fh:
                yield from fh
        except OSError as exc:
            _logger.error("could not read %s: %s", path, exc)


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the ``diagnose`` subcommand."""
    parser = subparsers.add_parser(
        "diagnose",
        parents=[common_parser],
        help="Aggregate span timings from log files",
        description=(
            "Aggregate span=... ms=... markers from log files; print "
            "p50/p95 per span. Pass ``-`` to read from stdin."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="LOG_FILE",
        help="One or more log files to aggregate (``-`` for stdin).",
    )
    parser.set_defaults(func=diagnose)
    return parser


def diagnose(args: argparse.Namespace) -> int:
    """Print the aggregate span summary for ``args.paths``."""
    summary = _aggregate(_iter_lines(list(args.paths)))
    sys.stdout.write(render_summary_table(summary) + "\n")
    return 0
