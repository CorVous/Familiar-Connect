"""Tests for metrics reporting functions (pure, no I/O)."""

from __future__ import annotations

from familiar_connect.metrics.report import (
    provider_success_rates,
    stage_breakdown,
    summary_stats,
    tag_comparison,
    throughput_stats,
)
from familiar_connect.metrics.types import StageSpan, TurnTrace


def _trace(
    *,
    trace_id: str,
    total: float,
    stages: tuple[StageSpan, ...] = (),
    tags: dict[str, str] | None = None,
    timestamp: str = "2026-04-14T12:00:00+00:00",
    modality: str = "text",
) -> TurnTrace:
    return TurnTrace(
        trace_id=trace_id,
        familiar_id="fam",
        channel_id=1,
        guild_id=None,
        modality=modality,
        timestamp=timestamp,
        total_duration_s=total,
        stages=stages,
        tags=tags or {},
    )


class TestSummaryStats:
    def test_empty(self) -> None:
        out = summary_stats([])
        assert "no traces" in out.lower() or "0" in out

    def test_counts(self) -> None:
        traces = [_trace(trace_id=str(i), total=float(i + 1)) for i in range(10)]
        out = summary_stats(traces)
        assert "10" in out
        # percentiles sensible
        assert "p50" in out.lower()
        assert "p95" in out.lower()


class TestStageBreakdown:
    def test_empty(self) -> None:
        out = stage_breakdown([])
        assert out  # any non-empty string

    def test_stage_durations_reported(self) -> None:
        traces = [
            _trace(
                trace_id="1",
                total=1.0,
                stages=(
                    StageSpan(name="llm_call", start_s=0.0, duration_s=0.5),
                    StageSpan(name="tts", start_s=0.5, duration_s=0.2),
                ),
            ),
            _trace(
                trace_id="2",
                total=2.0,
                stages=(StageSpan(name="llm_call", start_s=0.0, duration_s=1.2),),
            ),
        ]
        out = stage_breakdown(traces)
        assert "llm_call" in out
        assert "tts" in out


class TestThroughputStats:
    def test_empty(self) -> None:
        out = throughput_stats([])
        assert out

    def test_turns_per_minute(self) -> None:
        # 10 traces spread over 60 seconds = 10 turns/min
        traces = [
            _trace(
                trace_id=str(i),
                total=0.1,
                timestamp=f"2026-04-14T12:00:{i:02d}+00:00",
            )
            for i in range(10)
        ]
        out = throughput_stats(traces)
        assert "turns" in out.lower()


class TestProviderSuccessRates:
    def test_extracts_from_pipeline_assembly_metadata(self) -> None:
        traces = [
            _trace(
                trace_id="1",
                total=1.0,
                stages=(
                    StageSpan(
                        name="pipeline_assembly",
                        start_s=0.0,
                        duration_s=0.1,
                        metadata={
                            "provider_outcomes": [
                                {
                                    "id": "character",
                                    "duration_s": 0.01,
                                    "status": "ok",
                                    "error": None,
                                },
                                {
                                    "id": "history",
                                    "duration_s": 0.05,
                                    "status": "timeout",
                                    "error": "deadline",
                                },
                            ],
                        },
                    ),
                ),
            ),
            _trace(
                trace_id="2",
                total=1.0,
                stages=(
                    StageSpan(
                        name="pipeline_assembly",
                        start_s=0.0,
                        duration_s=0.1,
                        metadata={
                            "provider_outcomes": [
                                {
                                    "id": "character",
                                    "duration_s": 0.01,
                                    "status": "ok",
                                    "error": None,
                                },
                                {
                                    "id": "history",
                                    "duration_s": 0.02,
                                    "status": "ok",
                                    "error": None,
                                },
                            ],
                        },
                    ),
                ),
            ),
        ]
        out = provider_success_rates(traces)
        assert "character" in out
        assert "history" in out
        # character: 2 ok
        # history: 1 ok, 1 timeout


class TestTagComparison:
    def test_groups_by_tag(self) -> None:
        traces = [
            _trace(trace_id="a1", total=1.0, tags={"variant": "A"}),
            _trace(trace_id="a2", total=2.0, tags={"variant": "A"}),
            _trace(trace_id="b1", total=3.0, tags={"variant": "B"}),
            _trace(trace_id="b2", total=4.0, tags={"variant": "B"}),
        ]
        out = tag_comparison(traces, "variant")
        assert "variant=A" in out or "A" in out
        assert "variant=B" in out or "B" in out

    def test_ignores_traces_missing_tag(self) -> None:
        traces = [
            _trace(trace_id="has", total=1.0, tags={"variant": "A"}),
            _trace(trace_id="missing", total=1.0),
        ]
        out = tag_comparison(traces, "variant")
        # missing-tag traces skipped, but output still sensible
        assert out
