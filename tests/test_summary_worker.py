"""Tests for :class:`familiar_connect.processors.summary_worker.SummaryWorker`.

Watermark-driven regeneration for both per-channel rolling summaries
and cross-channel summaries. See plan § Design.4, § Design.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.summary_worker import SummaryWorker


class _ScriptedLLM(LLMClient):
    """Scripted LLM for summary tests. Returns scripted replies."""

    def __init__(self, *, replies: list[str]) -> None:
        super().__init__(api_key="k", model="m")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="(nothing to summarise)")
        return Message(role="assistant", content=self._replies.pop(0))

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content


def _seed_turns(store: HistoryStore, count: int, channel_id: int = 1) -> None:
    for i in range(count):
        store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            author=None,
        )


class TestRollingSummary:
    @pytest.mark.asyncio
    async def test_generates_summary_when_threshold_crossed(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12)  # 12 > threshold 10
        llm = _ScriptedLLM(replies=["Alice said hi and things happened."])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        summary = store.get_summary(familiar_id="fam", channel_id=1)
        assert summary is not None
        assert "Alice" in summary.summary_text
        assert summary.last_summarised_id == 12

    @pytest.mark.asyncio
    async def test_noop_below_threshold(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 3)
        llm = _ScriptedLLM(replies=["should not be used"])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        assert store.get_summary(familiar_id="fam", channel_id=1) is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_compounding_uses_prior_summary(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12)
        llm = _ScriptedLLM(
            replies=[
                "Round 1 summary: early chat.",
                "Round 2 summary: extended with 10 more.",
            ]
        )

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        # Add 10 more turns, trigger another tick
        _seed_turns(store, 10)
        await worker.tick()

        summary = store.get_summary(familiar_id="fam", channel_id=1)
        assert summary is not None
        assert "Round 2" in summary.summary_text
        # Second call passes the prior summary in the prompt
        second_call = llm.calls[1]
        joined = "\n".join(m.content for m in second_call)
        assert "Round 1 summary" in joined

    @pytest.mark.asyncio
    async def test_multiple_channels_independently(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12, channel_id=1)
        _seed_turns(store, 12, channel_id=2)
        llm = _ScriptedLLM(replies=["ch1 summary", "ch2 summary"])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        s1 = store.get_summary(familiar_id="fam", channel_id=1)
        s2 = store.get_summary(familiar_id="fam", channel_id=2)
        assert s1 is not None
        assert s1.summary_text
        assert s2 is not None
        assert s2.summary_text
        assert s1.summary_text != s2.summary_text


class TestCrossChannelSummary:
    @pytest.mark.asyncio
    async def test_generates_when_viewer_wants_other_channel(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 6, channel_id=100)  # source (e.g. #general)
        llm = _ScriptedLLM(replies=["In #general: chit-chat."])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
            cross_channel_map={1: [100]},  # viewer channel 1 watches source 100
            cross_k=3,
        )
        await worker.tick()

        cross = store.get_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
        )
        assert cross is not None
        assert "chit-chat" in cross.summary_text
        assert cross.source_last_id == 6

    @pytest.mark.asyncio
    async def test_cross_noop_when_source_below_k(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 2, channel_id=100)
        llm = _ScriptedLLM(replies=["unused"])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
            cross_channel_map={1: [100]},
            cross_k=5,
        )
        await worker.tick()

        cross = store.get_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
        )
        assert cross is None

    @pytest.mark.asyncio
    async def test_cross_refreshes_after_k_more_turns(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 5, channel_id=100)
        llm = _ScriptedLLM(replies=["first cross", "second cross"])

        worker = SummaryWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
            cross_channel_map={1: [100]},
            cross_k=3,
        )
        await worker.tick()
        first = store.get_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
        )
        assert first is not None
        assert first.source_last_id == 5

        _seed_turns(store, 4, channel_id=100)  # now source at 9
        await worker.tick()
        second = store.get_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
        )
        assert second is not None
        assert second.source_last_id == 9
        assert "second cross" in second.summary_text
