"""Pure reporting functions over :class:`TurnTrace` sequences.

No I/O, no side effects. Consumed by the ``metrics`` CLI subcommand.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from familiar_connect.metrics.types import TurnTrace


def _percentile(values: Sequence[float], pct: float) -> float:
    """Return the *pct*'th percentile of *values* (0-100). Empty -> 0.0."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    # nearest-rank method; good enough at our sample sizes
    k = max(0, min(len(sorted_vals) - 1, round(pct / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def summary_stats(traces: Sequence[TurnTrace]) -> str:
    """Total count + latency percentiles for the trace set."""
    if not traces:
        return "no traces"

    totals = [t.total_duration_s for t in traces]
    p50 = _percentile(totals, 50)
    p95 = _percentile(totals, 95)
    p99 = _percentile(totals, 99)
    mean = statistics.fmean(totals)

    lines = [
        f"traces: {len(traces)}",
        f"total latency mean={mean:.3f}s p50={p50:.3f}s p95={p95:.3f}s p99={p99:.3f}s",
    ]
    return "\n".join(lines)


def stage_breakdown(traces: Sequence[TurnTrace]) -> str:
    """Per-stage p50/p95/max table across all traces."""
    if not traces:
        return "no traces"

    per_stage: dict[str, list[float]] = defaultdict(list)
    for trace in traces:
        for span in trace.stages:
            per_stage[span.name].append(span.duration_s)

    if not per_stage:
        return "no stages recorded"

    header = f"{'stage':<22} {'n':>5} {'p50':>8} {'p95':>8} {'max':>8}"
    lines = [header, "-" * len(header)]
    for name in sorted(per_stage):
        durs = per_stage[name]
        lines.append(
            f"{name:<22} {len(durs):>5} "
            f"{_percentile(durs, 50):>7.3f}s "
            f"{_percentile(durs, 95):>7.3f}s "
            f"{max(durs):>7.3f}s",
        )
    return "\n".join(lines)


def throughput_stats(traces: Sequence[TurnTrace]) -> str:
    """Turns/minute over the observed time window."""
    if not traces:
        return "no traces"
    try:
        timestamps = [datetime.fromisoformat(t.timestamp) for t in traces]
    except ValueError:
        return "traces have unparseable timestamps"

    earliest = min(timestamps)
    latest = max(timestamps)
    window_s = max((latest - earliest).total_seconds(), 1.0)
    turns_per_min = len(traces) / window_s * 60

    return f"turns: {len(traces)} over {window_s:.1f}s ({turns_per_min:.2f} turns/min)"


def provider_success_rates(traces: Sequence[TurnTrace]) -> str:
    """Per-provider ok/error/timeout counts from ``pipeline_assembly`` metadata."""
    if not traces:
        return "no traces"

    # provider_id -> {status -> count}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for trace in traces:
        for span in trace.stages:
            if span.name != "pipeline_assembly":
                continue
            raw = span.metadata.get("provider_outcomes")
            if not isinstance(raw, list):
                continue
            for outcome in raw:
                if not isinstance(outcome, dict):
                    continue
                entry = cast("dict[str, object]", outcome)
                pid = str(entry.get("id", "?"))
                status = str(entry.get("status", "?"))
                counts[pid][status] += 1

    if not counts:
        return "no provider outcomes recorded"

    header = f"{'provider':<20} {'ok':>6} {'error':>6} {'timeout':>8}"
    lines = [header, "-" * len(header)]
    for pid in sorted(counts):
        c = counts[pid]
        lines.append(
            f"{pid:<20} {c.get('ok', 0):>6} "
            f"{c.get('error', 0):>6} {c.get('timeout', 0):>8}",
        )
    return "\n".join(lines)


def tag_comparison(traces: Sequence[TurnTrace], tag_key: str) -> str:
    """Group traces by tag value, compare latency distributions."""
    groups: dict[str, list[float]] = defaultdict(list)
    for trace in traces:
        val = trace.tags.get(tag_key)
        if val is None:
            continue
        groups[val].append(trace.total_duration_s)

    if not groups:
        return f"no traces tagged with {tag_key}"

    header = f"{tag_key + '=value':<28} {'n':>5} {'p50':>8} {'p95':>8} {'mean':>8}"
    lines = [header, "-" * len(header)]
    for val in sorted(groups):
        durs = groups[val]
        mean = statistics.fmean(durs)
        lines.append(
            f"{tag_key}={val:<22} {len(durs):>5} "
            f"{_percentile(durs, 50):>7.3f}s "
            f"{_percentile(durs, 95):>7.3f}s "
            f"{mean:>7.3f}s",
        )
    return "\n".join(lines)
