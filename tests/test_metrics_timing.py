"""Tests for TraceBuilder and span() timing helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

import pytest

from familiar_connect.context.pipeline import ProviderOutcome
from familiar_connect.metrics.timing import TraceBuilder
from familiar_connect.metrics.types import TurnTrace


def test_empty_trace_finalizes() -> None:
    builder = TraceBuilder(
        familiar_id="fam",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    trace = builder.finalize()
    assert isinstance(trace, TurnTrace)
    assert trace.familiar_id == "fam"
    assert trace.channel_id == 1
    assert trace.modality == "text"
    assert trace.stages == ()
    assert trace.tags == {}
    assert trace.total_duration_s >= 0.0
    assert trace.trace_id  # non-empty uuid


@pytest.mark.asyncio
async def test_single_span_recorded() -> None:
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    async with builder.span("work") as meta:
        await asyncio.sleep(0.01)
        meta["note"] = "hi"

    trace = builder.finalize()
    assert len(trace.stages) == 1
    span = trace.stages[0]
    assert span.name == "work"
    assert span.duration_s >= 0.01
    assert span.start_s >= 0.0
    assert span.metadata["note"] == "hi"


@pytest.mark.asyncio
async def test_sequential_spans_ordered() -> None:
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    async with builder.span("a"):
        await asyncio.sleep(0.005)
    async with builder.span("b"):
        await asyncio.sleep(0.005)
    async with builder.span("c"):
        await asyncio.sleep(0.005)

    trace = builder.finalize()
    assert [s.name for s in trace.stages] == ["a", "b", "c"]
    # each later start_s is >= previous start_s + duration_s
    starts = [s.start_s for s in trace.stages]
    assert starts[0] <= starts[1] <= starts[2]


@pytest.mark.asyncio
async def test_span_initial_metadata() -> None:
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    async with builder.span("x", model="glm-5.1", mode="full_rp"):
        pass
    trace = builder.finalize()
    assert trace.stages[0].metadata == {"model": "glm-5.1", "mode": "full_rp"}


def test_tag_attached() -> None:
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    builder.tag("channel_mode", "full_rp")
    builder.tag("model_variant", "gpt-4o")
    trace = builder.finalize()
    assert trace.tags == {"channel_mode": "full_rp", "model_variant": "gpt-4o"}


@pytest.mark.asyncio
async def test_span_logs_debug_on_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="familiar_connect.metrics.timing"):
        builder = TraceBuilder(
            familiar_id="f",
            channel_id=1,
            guild_id=None,
            modality="text",
        )
        async with builder.span("llm_call") as meta:
            meta["model"] = "glm-5.1"

    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "llm_call" in r.getMessage()
    ]
    assert len(debug_records) == 1
    msg = debug_records[0].getMessage()
    assert "duration=" in msg
    assert "model=glm-5.1" in msg


@pytest.mark.asyncio
async def test_finalize_logs_info_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="familiar_connect.metrics.timing"):
        builder = TraceBuilder(
            familiar_id="f",
            channel_id=42,
            guild_id=None,
            modality="text",
        )
        async with builder.span("a"):
            pass
        trace = builder.finalize()

    info_records = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "turn" in r.getMessage()
    ]
    assert len(info_records) == 1
    msg = info_records[0].getMessage()
    assert trace.trace_id[:8] in msg  # at least a prefix present
    assert "channel=42" in msg
    assert "modality=text" in msg
    assert "stages=1" in msg


@pytest.mark.asyncio
async def test_add_provider_outcomes() -> None:
    """Convenience method embeds provider outcomes into a span's metadata."""
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    outcomes = [
        ProviderOutcome(
            provider_id="character",
            duration_s=0.01,
            contributions=[],
            status="ok",
        ),
        ProviderOutcome(
            provider_id="history",
            duration_s=0.05,
            contributions=[],
            status="timeout",
            error_message="exceeded deadline",
        ),
    ]
    async with builder.span("pipeline_assembly") as meta:
        builder.add_provider_outcomes(meta, outcomes)

    trace = builder.finalize()
    recorded = cast(
        "list[dict[str, object]]",
        trace.stages[0].metadata["provider_outcomes"],
    )
    assert len(recorded) == 2
    assert recorded[0]["id"] == "character"
    assert recorded[0]["status"] == "ok"
    assert recorded[1]["id"] == "history"
    assert recorded[1]["status"] == "timeout"


async def _raise_in_span(builder: TraceBuilder) -> None:
    async with builder.span("fails"):
        await asyncio.sleep(0.005)
        raise ValueError("boom")


@pytest.mark.asyncio
async def test_span_records_duration_on_exception() -> None:
    """A span that raises still records its duration — loss of data is worse."""
    builder = TraceBuilder(
        familiar_id="f",
        channel_id=1,
        guild_id=None,
        modality="text",
    )
    with pytest.raises(ValueError, match="boom"):
        await _raise_in_span(builder)

    trace = builder.finalize()
    assert len(trace.stages) == 1
    assert trace.stages[0].name == "fails"
    assert trace.stages[0].duration_s >= 0.005
