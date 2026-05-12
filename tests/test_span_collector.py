"""Tests for :class:`SpanCollector` + integration with ``@span``."""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.diagnostics.collector import (
    SpanCollector,
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.diagnostics.spans import span


class TestSpanCollector:
    def test_record_then_read(self) -> None:
        coll = SpanCollector(maxlen=10)
        coll.record(name="a", ms=10, status="ok")
        coll.record(name="a", ms=20, status="ok")
        coll.record(name="b", ms=5, status="error")
        all_records = coll.all()
        assert len(all_records) == 3
        by_name = coll.by_name()
        assert len(by_name["a"]) == 2
        assert len(by_name["b"]) == 1

    def test_ring_buffer_evicts_oldest(self) -> None:
        coll = SpanCollector(maxlen=3)
        for i in range(5):
            coll.record(name="x", ms=i, status="ok")
        records = coll.all()
        assert len(records) == 3
        assert [r.ms for r in records] == [2, 3, 4]

    def test_summary_computes_percentiles(self) -> None:
        coll = SpanCollector(maxlen=100)
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            coll.record(name="llm", ms=ms, status="ok")
        summary = coll.summary()
        assert summary["llm"]["count"] == 10
        assert summary["llm"]["p50"] == pytest.approx(55, abs=1)
        assert summary["llm"]["p95"] == pytest.approx(95.5, abs=1)
        assert summary["llm"]["last_ms"] == 100

    def test_summary_empty_buckets(self) -> None:
        coll = SpanCollector(maxlen=10)
        assert coll.summary() == {}

    def test_singleton_reset(self) -> None:
        reset_span_collector()
        a = get_span_collector()
        b = get_span_collector()
        assert a is b
        reset_span_collector()
        c = get_span_collector()
        assert c is not a


class TestSpanIntegration:
    def setup_method(self) -> None:
        reset_span_collector()

    def teardown_method(self) -> None:
        reset_span_collector()

    @pytest.mark.asyncio
    async def test_span_decorator_records_into_collector(self) -> None:
        @span("demo")
        async def work() -> int:
            await asyncio.sleep(0.005)
            return 42

        await work()
        coll = get_span_collector()
        records = coll.all()
        assert len(records) == 1
        assert records[0].name == "demo"
        assert records[0].status == "ok"

    @pytest.mark.asyncio
    async def test_span_records_errors(self) -> None:
        @span("boom")
        async def work() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await work()
        records = get_span_collector().all()
        assert records
        assert records[-1].status == "error"
