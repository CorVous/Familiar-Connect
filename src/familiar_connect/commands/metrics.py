"""Metrics subcommand — view performance data collected by the bot.

Reads from ``data/familiars/<id>/metrics.db`` and prints aggregate
statistics to stdout. Pass ``--compare KEY`` to group by a tag (A/B
testing). ``--plot`` emits matplotlib histograms if matplotlib is
installed; otherwise falls back to text output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from familiar_connect.metrics.report import (
    provider_success_rates,
    stage_breakdown,
    summary_stats,
    tag_comparison,
    throughput_stats,
)
from familiar_connect.metrics.sqlite_collector import SQLiteCollector

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from familiar_connect.metrics.types import TurnTrace

_logger = logging.getLogger(__name__)

_DEFAULT_FAMILIARS_ROOT = Path("data") / "familiars"


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the ``metrics`` subcommand."""
    parser = subparsers.add_parser(
        "metrics",
        parents=[common_parser],
        help="View performance metrics collected by the bot",
        description=(
            "Analyze per-turn performance traces from data/familiars/<id>/metrics.db."
        ),
    )
    parser.add_argument(
        "--familiar",
        metavar="ID",
        default=None,
        help="Folder name of the familiar to report on (under data/familiars/).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=100,
        metavar="N",
        help="Report on the most recent N traces (default: 100).",
    )
    parser.add_argument(
        "--stage",
        default=None,
        metavar="NAME",
        help="Filter to traces that include a span with this stage name.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        metavar="KEY=VALUE",
        help="Filter traces by a tag equality (e.g. channel_mode=full_rp).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="TAG_KEY",
        help="A/B comparison: group traces by tag value.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Emit histogram plot (requires matplotlib; falls back to text).",
    )
    parser.add_argument(
        "--familiars-root",
        default=None,
        metavar="PATH",
        help="Root directory containing familiar folders (default: data/familiars).",
    )
    parser.set_defaults(func=run)
    return parser


def _resolve_db_path(args: argparse.Namespace) -> Path | None:
    """Return the metrics.db path for the requested familiar, or ``None``."""
    if not args.familiar:
        print("error: --familiar ID is required")  # noqa: T201
        return None

    root = Path(args.familiars_root) if args.familiars_root else _DEFAULT_FAMILIARS_ROOT
    db_path = root / args.familiar / "metrics.db"
    if not db_path.exists():
        print(f"error: metrics.db not found at {db_path}")  # noqa: T201
        return None
    return db_path


def run(args: argparse.Namespace) -> int:
    """Execute the metrics command."""
    db_path = _resolve_db_path(args)
    if db_path is None:
        return 1

    collector = SQLiteCollector(db_path, flush_interval=1)
    try:
        traces = collector.recent_traces(familiar_id=args.familiar, limit=args.last)
    finally:
        collector.close()

    # apply --tag KEY=VALUE filter in-memory
    if args.tag:
        if "=" not in args.tag:
            print(f"error: --tag must be KEY=VALUE, got: {args.tag}")  # noqa: T201
            return 1
        key, _, value = args.tag.partition("=")
        traces = [t for t in traces if t.tags.get(key) == value]

    # apply --stage filter: traces must have at least one span with that name
    if args.stage:
        traces = [t for t in traces if any(s.name == args.stage for s in t.stages)]

    print(summary_stats(traces))  # noqa: T201
    print()  # noqa: T201
    print(throughput_stats(traces))  # noqa: T201
    print()  # noqa: T201
    print(stage_breakdown(traces))  # noqa: T201
    print()  # noqa: T201
    print(provider_success_rates(traces))  # noqa: T201

    if args.compare:
        print()  # noqa: T201
        print(tag_comparison(traces, args.compare))  # noqa: T201

    if args.plot:
        _emit_plot_or_fallback(traces)

    return 0


def _emit_plot_or_fallback(traces: Sequence[TurnTrace]) -> None:
    """Render histogram if matplotlib is available; else a fallback message."""
    try:
        import importlib  # noqa: PLC0415

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("matplotlib not installed; install to enable --plot")  # noqa: T201
        return

    totals = [t.total_duration_s for t in traces]
    if not totals:
        print("no traces to plot")  # noqa: T201
        return
    plt.hist(totals, bins=20)
    plt.xlabel("total latency (s)")
    plt.ylabel("count")
    plt.title("per-turn latency distribution")
    plt.show()
