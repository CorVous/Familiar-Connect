"""Tests for MoodEvaluator — Step 13 real LLM side-call.

TDD red-first: all tests fail until mood.py is rewritten.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.mood import MoodEvaluator

if TYPE_CHECKING:
    from collections.abc import Iterable

_FAMILIAR = "aria"
_CHANNEL = 42
_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _FakeLLMClient(LLMClient):
    """Scripted stub: returns queued replies, records calls."""

    def __init__(self, replies: Iterable[str] | None = None) -> None:
        super().__init__(api_key="fake", model="fake/model")
        self.calls: list[list[Message]] = []
        self._replies: list[str] = list(replies or [])

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        content = self._replies.pop(0) if self._replies else ""
        return Message(role="assistant", content=content)


def _make_store(*turns: tuple[str, str]) -> HistoryStore:
    """Return in-memory store seeded with (role, content) turns."""
    store = HistoryStore(":memory:")
    for role, content in turns:
        author = _ALICE if role == "user" else None
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role=role,
            content=content,
            author=author,
        )
    return store


def _make_evaluator(
    replies: list[str],
    *turns: tuple[str, str],
) -> tuple[MoodEvaluator, _FakeLLMClient]:
    """Return (evaluator, client) with a seeded in-memory store."""
    store = _make_store(*turns)
    client = _FakeLLMClient(replies)
    evaluator = MoodEvaluator(llm_client=client, history_store=store)
    return evaluator, client


class TestMoodEvaluatorRealCall:
    def test_returns_float_in_range(self) -> None:
        evaluator, _ = _make_evaluator(
            ["0.3"],
            ("user", "this is great!"),
            ("assistant", "glad you think so!"),
        )
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(0.3)

    def test_clamps_positive_out_of_range(self) -> None:
        evaluator, _ = _make_evaluator(["1.5"], ("user", "amazing!"))
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(0.5)

    def test_clamps_negative_out_of_range(self) -> None:
        evaluator, _ = _make_evaluator(["-0.8"], ("user", "wrong again..."))
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(-0.5)

    def test_logs_modifier(self, caplog: pytest.LogCaptureFixture) -> None:
        evaluator, _ = _make_evaluator(["0.3"], ("user", "great!"))
        with caplog.at_level(logging.INFO, logger="familiar_connect.mood"):
            asyncio.run(evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR))
        messages = [_ANSI_RE.sub("", r.message) for r in caplog.records]
        assert any("Mood" in m and "modifier=" in m for m in messages)

    def test_returns_zero_when_no_history(self) -> None:
        store = HistoryStore(":memory:")
        client = _FakeLLMClient([])
        evaluator = MoodEvaluator(llm_client=client, history_store=store)
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(0.0)
        assert client.calls == []  # no LLM call made

    def test_returns_zero_on_llm_error(self) -> None:
        store = _make_store(("user", "hello"))
        error_client = AsyncMock()
        error_client.chat = AsyncMock(side_effect=Exception("network error"))
        evaluator = MoodEvaluator(llm_client=error_client, history_store=store)
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(0.0)


class TestMoodEvaluatorStubMode:
    """No-arg MoodEvaluator behaves as the original stub."""

    def test_stub_returns_zero(self) -> None:
        evaluator = MoodEvaluator()
        result = asyncio.run(
            evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR)
        )
        assert result == pytest.approx(0.0)

    def test_stub_logs_stub_label(self, caplog: pytest.LogCaptureFixture) -> None:
        evaluator = MoodEvaluator()
        with caplog.at_level(logging.INFO, logger="familiar_connect.mood"):
            asyncio.run(evaluator.evaluate(channel_id=_CHANNEL, familiar_id=_FAMILIAR))
        messages = [r.message for r in caplog.records]
        assert any("stub" in m for m in messages)
