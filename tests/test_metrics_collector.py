"""Tests for the MetricsCollector Protocol and NullCollector."""

from __future__ import annotations

from familiar_connect.metrics.collector import MetricsCollector, NullCollector
from familiar_connect.metrics.types import TurnTrace


def _sample_trace() -> TurnTrace:
    return TurnTrace(
        trace_id="t",
        familiar_id="fam",
        channel_id=1,
        guild_id=None,
        modality="text",
        timestamp="2026-04-14T12:00:00+00:00",
        total_duration_s=0.1,
        stages=(),
    )


class TestNullCollector:
    def test_conforms_to_protocol(self) -> None:
        collector: MetricsCollector = NullCollector()
        assert isinstance(collector, MetricsCollector)

    def test_record_does_not_raise(self) -> None:
        NullCollector().record(_sample_trace())

    def test_close_does_not_raise(self) -> None:
        NullCollector().close()

    def test_multiple_records_ok(self) -> None:
        collector = NullCollector()
        for _ in range(5):
            collector.record(_sample_trace())
        collector.close()
