"""Tests for :class:`familiar_connect.processors.summary_worker.SummaryWorker`.

Watermark-driven regeneration of the focus-stream rolling summary
(consumed cross-channel stream) and the retired cross-channel summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import (
    FOCUS_STREAM_CHANNEL_ID,
    HistoryStore,
    SummaryEntry,
)
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
        yield reply.content_str


def _seed_turns(
    store: HistoryStore,
    count: int,
    channel_id: int = 1,
    *,
    consumed: bool = True,
) -> None:
    for i in range(count):
        store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            author=None,
            consumed=consumed,
        )


def _focus_summary(store: HistoryStore) -> SummaryEntry | None:
    return store.get_summary(familiar_id="fam", channel_id=FOCUS_STREAM_CHANNEL_ID)


class TestFocusStreamSummary:
    @pytest.mark.asyncio
    async def test_summarises_consumed_cross_channel_stream(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 6, channel_id=1)
        _seed_turns(store, 6, channel_id=2)  # 12 consumed across two channels
        llm = _ScriptedLLM(replies=["Cross-channel: hi everywhere."])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        summary = _focus_summary(store)
        assert summary is not None
        assert summary.last_summarised_id == 12
        assert summary.last_consumed_at is not None
        assert summary.summary_text

    @pytest.mark.asyncio
    async def test_noop_below_threshold(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 3)
        llm = _ScriptedLLM(replies=["should not be used"])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        assert _focus_summary(store) is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ignores_staged_turns(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 3, channel_id=1)  # consumed
        _seed_turns(store, 20, channel_id=2, consumed=False)  # staged, unfocused
        llm = _ScriptedLLM(replies=["should not fire"])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        assert _focus_summary(store) is None  # 3 consumed < threshold
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_compounds_prior_summary_into_prompt(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12)
        llm = _ScriptedLLM(
            replies=[
                "Round 1 summary: early chat.",
                "Round 2 summary: extended with 10 more.",
            ]
        )

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()
        _seed_turns(store, 10)
        await worker.tick()

        summary = _focus_summary(store)
        assert summary is not None
        assert "Round 2" in summary.summary_text
        second_call = llm.calls[1]
        joined = "\n".join(m.content_str for m in second_call)
        assert "Round 1 summary" in joined

    @pytest.mark.asyncio
    async def test_backfill_cap_bounds_first_run(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 500)
        llm = _ScriptedLLM(replies=["batch 1", "batch 2"])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
            backfill_cap=200,
        )
        await worker.tick()
        first = _focus_summary(store)
        assert first is not None
        assert first.last_summarised_id == 200

        await worker.tick()
        second = _focus_summary(store)
        assert second is not None
        assert second.last_summarised_id == 400

    @pytest.mark.asyncio
    async def test_late_promoted_turn_picked_up_on_next_tick(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12, channel_id=1)  # consumed; ids 1..12
        llm = _ScriptedLLM(replies=["round 1", "round 2 with dormant content"])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()
        first = _focus_summary(store)
        assert first is not None
        assert first.last_summarised_id == 12

        # 10 staged turns in a dormant channel; below-watermark consumed_at NULL
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=2,
                role="user",
                content=f"dormant {i}",
                author=None,
                consumed=False,
            )
        # nothing consumed yet -> noop
        await worker.tick()
        unchanged = _focus_summary(store)
        assert unchanged is not None
        assert unchanged.last_summarised_id == 12

        # focus shifts -> promote; consumed_at = NOW (> watermark)
        store.promote_staged_turns(familiar_id="fam", channel_id=2)
        await worker.tick()
        second = _focus_summary(store)
        assert second is not None
        assert second.last_summarised_id == 22
        third_call = llm.calls[1]
        joined = "\n".join(m.content_str for m in third_call)
        assert "dormant" in joined

    @pytest.mark.asyncio
    async def test_per_channel_summary_no_longer_written(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12, channel_id=1)
        llm = _ScriptedLLM(replies=["focus summary"])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=10,
        )
        await worker.tick()

        assert store.get_summary(familiar_id="fam", channel_id=1) is None
        assert _focus_summary(store) is not None


class TestCrossChannelSummary:
    @pytest.mark.asyncio
    async def test_generates_when_viewer_wants_other_channel(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 6, channel_id=100)  # source (e.g. #general)
        llm = _ScriptedLLM(replies=["In #general: chit-chat."])

        worker = SummaryWorker(
            store=AsyncHistoryStore(store),
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
            store=AsyncHistoryStore(store),
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
            store=AsyncHistoryStore(store),
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
