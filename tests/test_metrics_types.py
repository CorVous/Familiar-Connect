"""Tests for the metrics data model."""

from __future__ import annotations

import dataclasses

import pytest

from familiar_connect.metrics.types import StageSpan, TurnTrace


class TestStageSpan:
    def test_construct_minimal(self) -> None:
        span = StageSpan(name="llm_call", start_s=0.0, duration_s=1.5)
        assert span.name == "llm_call"
        assert span.start_s == pytest.approx(0.0)
        assert span.duration_s == pytest.approx(1.5)
        assert span.metadata == {}

    def test_construct_with_metadata(self) -> None:
        span = StageSpan(
            name="llm_call",
            start_s=0.1,
            duration_s=2.3,
            metadata={"model": "glm-5.1", "tokens_out": 420},
        )
        assert span.metadata["model"] == "glm-5.1"
        assert span.metadata["tokens_out"] == 420

    def test_frozen(self) -> None:
        span = StageSpan(name="x", start_s=0.0, duration_s=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(span, "name", "y")  # noqa: B010 — static analysis escape


class TestTurnTrace:
    def test_construct_minimal(self) -> None:
        trace = TurnTrace(
            trace_id="abc123",
            familiar_id="fam",
            channel_id=42,
            guild_id=None,
            modality="text",
            timestamp="2026-04-14T12:00:00+00:00",
            total_duration_s=3.2,
            stages=(),
        )
        assert trace.trace_id == "abc123"
        assert trace.familiar_id == "fam"
        assert trace.channel_id == 42
        assert trace.guild_id is None
        assert trace.modality == "text"
        assert trace.total_duration_s == pytest.approx(3.2)
        assert trace.stages == ()
        assert trace.tags == {}

    def test_construct_with_stages_and_tags(self) -> None:
        stages = (
            StageSpan(name="a", start_s=0.0, duration_s=0.5),
            StageSpan(name="b", start_s=0.5, duration_s=1.0),
        )
        trace = TurnTrace(
            trace_id="t1",
            familiar_id="fam",
            channel_id=1,
            guild_id=7,
            modality="voice",
            timestamp="2026-04-14T12:00:00+00:00",
            total_duration_s=1.5,
            stages=stages,
            tags={"channel_mode": "full_rp", "model_variant": "gpt-4o"},
        )
        assert len(trace.stages) == 2
        assert trace.tags["channel_mode"] == "full_rp"
        assert trace.tags["model_variant"] == "gpt-4o"

    def test_frozen(self) -> None:
        trace = TurnTrace(
            trace_id="t",
            familiar_id="f",
            channel_id=1,
            guild_id=None,
            modality="text",
            timestamp="2026-04-14T12:00:00+00:00",
            total_duration_s=0.0,
            stages=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(trace, "trace_id", "other")  # noqa: B010

    def test_stages_is_tuple(self) -> None:
        """Stages must be a tuple so the frozen trace can't be mutated via it."""
        trace = TurnTrace(
            trace_id="t",
            familiar_id="f",
            channel_id=1,
            guild_id=None,
            modality="text",
            timestamp="2026-04-14T12:00:00+00:00",
            total_duration_s=0.0,
            stages=(),
        )
        assert isinstance(trace.stages, tuple)
