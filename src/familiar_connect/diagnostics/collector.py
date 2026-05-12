"""In-process span collector — ring buffer of recent ``@span`` calls.

Feeds the :meth:`/diagnostics` slash command (Phase 5.2) with a
breakdown of the last-turn's timings without needing to re-parse logs
at runtime.

Logs-first remains the durable record; the collector is a convenience
for the live slash command and nothing more.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock


@dataclass(frozen=True)
class SpanRecord:
    """One recorded timing span."""

    name: str
    ms: int
    status: str
    at: datetime


class SpanCollector:
    """Bounded ring buffer. Thread-safe appends + reads."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._buf: deque[SpanRecord] = deque(maxlen=maxlen)
        self._lock = Lock()

    def record(self, *, name: str, ms: int, status: str) -> None:
        rec = SpanRecord(name=name, ms=ms, status=status, at=datetime.now(tz=UTC))
        with self._lock:
            self._buf.append(rec)

    def all(self) -> list[SpanRecord]:
        with self._lock:
            return list(self._buf)

    def by_name(self) -> dict[str, list[SpanRecord]]:
        out: dict[str, list[SpanRecord]] = {}
        for rec in self.all():
            out.setdefault(rec.name, []).append(rec)
        return out

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{span_name: {count, p50, p95, last_ms}}``.

        Percentiles computed from in-buffer records; if the buffer
        has been cycling they reflect only the recent window.
        """
        buckets = self.by_name()
        out: dict[str, dict[str, float]] = {}
        for name, records in buckets.items():
            ms_values = sorted(r.ms for r in records)
            last = records[-1].ms
            out[name] = {
                "count": len(ms_values),
                "p50": _percentile(ms_values, 50),
                "p95": _percentile(ms_values, 95),
                "last_ms": last,
            }
        return out

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


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


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: SpanCollector | None = None


def get_span_collector(maxlen: int = 2000) -> SpanCollector:
    """Return the process-wide :class:`SpanCollector`, creating on first use."""
    global _collector  # noqa: PLW0603
    if _collector is None:
        _collector = SpanCollector(maxlen=maxlen)
    return _collector


def reset_span_collector() -> None:
    """Reset the singleton — for tests only."""
    global _collector  # noqa: PLW0603
    _collector = None
