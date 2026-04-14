"""Profiling and telemetry for Familiar-Connect.

Captures per-turn performance data as structured ``TurnTrace`` records
and persists them to SQLite for later analysis.

- Use :mod:`logging` for ephemeral operational messages (errors, warnings, status)
- Use :class:`TraceBuilder` spans for anything you want to measure or compare
- Spans emit log lines automatically; never dual-write durations manually

See ``docs/guides/metrics.md`` for the full guide.
"""

from __future__ import annotations

from familiar_connect.metrics.collector import MetricsCollector, NullCollector
from familiar_connect.metrics.sqlite_collector import SQLiteCollector
from familiar_connect.metrics.timing import TraceBuilder
from familiar_connect.metrics.types import StageSpan, TurnTrace

__all__ = [
    "MetricsCollector",
    "NullCollector",
    "SQLiteCollector",
    "StageSpan",
    "TraceBuilder",
    "TurnTrace",
]
