"""Tests for Phase-3 cold-cache signal detectors + logging."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta

import pytest  # noqa: TC002 — runtime use via caplog fixture typing

from familiar_connect.diagnostics.cold_cache import (
    detect_silence_gap,
    detect_topic_shift,
    detect_unknown_proper_noun,
    log_signals,
)

_ANSI_RE = re.compile(r"\x1b\[\d+m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestTopicShift:
    def test_fires_when_vocab_disjoint(self) -> None:
        assert (
            detect_topic_shift(
                new_text="Tell me about the submarine propulsion",
                prior_context="Discussed fox diets and foraging patterns",
            )
            is True
        )

    def test_quiet_when_vocab_overlaps(self) -> None:
        assert (
            detect_topic_shift(
                new_text="More about foxes and foraging",
                prior_context="Discussed fox diets and foraging patterns",
                min_overlap=0.1,
            )
            is False
        )

    def test_quiet_when_prior_is_empty(self) -> None:
        assert detect_topic_shift(new_text="anything", prior_context="") is False


class TestUnknownProperNoun:
    def test_finds_new_capitalized_tokens(self) -> None:
        nouns = detect_unknown_proper_noun(
            new_text="I'm meeting Aria and Boris at the pub.",
            prior_context="We were talking about puddles.",
        )
        assert "Aria" in nouns
        assert "Boris" in nouns

    def test_skips_known_capitalized_tokens(self) -> None:
        nouns = detect_unknown_proper_noun(
            new_text="Aria said hi.",
            prior_context="Aria joined the channel.",
        )
        assert nouns == []

    def test_ignores_short_tokens(self) -> None:
        nouns = detect_unknown_proper_noun(
            new_text="Is OK with me",
            prior_context="nothing relevant",
        )
        assert nouns == []


class TestSilenceGap:
    def test_fires_past_threshold(self) -> None:
        t0 = datetime.now(tz=UTC)
        t1 = t0 + timedelta(minutes=10)
        gap = detect_silence_gap(
            prev_turn_at=t0, current_turn_at=t1, threshold_seconds=300
        )
        assert gap is not None
        assert gap >= 600

    def test_quiet_below_threshold(self) -> None:
        t0 = datetime.now(tz=UTC)
        t1 = t0 + timedelta(seconds=30)
        assert (
            detect_silence_gap(
                prev_turn_at=t0, current_turn_at=t1, threshold_seconds=300
            )
            is None
        )

    def test_quiet_when_no_prior(self) -> None:
        t1 = datetime.now(tz=UTC)
        assert detect_silence_gap(prev_turn_at=None, current_turn_at=t1) is None


class TestLogSignals:
    def test_emits_spans_for_firing_signals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        t0 = datetime.now(tz=UTC)
        t1 = t0 + timedelta(minutes=10)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.diagnostics.cold_cache"
        ):
            fired = log_signals(
                channel_id=42,
                turn_id="t-1",
                new_text="Meet Aria about the submarine.",
                prior_context="We discussed foxes.",
                prev_turn_at=t0,
                current_turn_at=t1,
            )
        messages = [_strip(r.getMessage()) for r in caplog.records]
        assert "topic_shift" in fired
        assert "unknown_proper_nouns" in fired
        assert "silence_gap_s" in fired
        assert any("signal=topic_shift" in m for m in messages)
        assert any("signal=unknown_proper_noun" in m for m in messages)
        assert any("signal=silence_gap" in m for m in messages)

    def test_no_spans_when_nothing_fires(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = datetime.now(tz=UTC)
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.diagnostics.cold_cache"
        ):
            fired = log_signals(
                channel_id=1,
                turn_id="t",
                new_text="same topic as before Aria mentioned it",
                prior_context="aria was discussing this topic before",
                prev_turn_at=now - timedelta(seconds=5),
                current_turn_at=now,
            )
        assert fired == {}
        assert not [r for r in caplog.records if "signal=" in r.getMessage()]
