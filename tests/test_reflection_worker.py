"""Tests for :class:`ReflectionWorker` (M3).

Watermark-driven reflection writes — fires when enough new turns have
accumulated since the previous reflection's watermark. Asks the LLM
for high-level questions about recent events; persists each answer
with ``cited_turn_ids`` / ``cited_fact_ids`` provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.reflection_worker import ReflectionWorker


class _ScriptedLLM(LLMClient):
    """Scripted LLM stub. Pops replies in order; records messages."""

    def __init__(self, *, replies: list[str]) -> None:
        super().__init__(api_key="k", model="m")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="[]")
        return Message(role="assistant", content=self._replies.pop(0))

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content


def _seed_turns(store: HistoryStore, count: int, channel_id: int = 1) -> list[int]:
    out: list[int] = []
    for i in range(count):
        t = store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            author=None,
        )
        out.append(t.id)
    return out


def _seed_facts(store: HistoryStore, count: int) -> list[int]:
    out: list[int] = []
    for i in range(count):
        f = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text=f"fact {i}",
            source_turn_ids=[i + 1],
        )
        out.append(f.id)
    return out


class TestReflectionWorker:
    @pytest.mark.asyncio
    async def test_noop_below_threshold(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 5)  # below threshold
        llm = _ScriptedLLM(replies=["should not be called"])

        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()

        assert llm.calls == []
        assert store.recent_reflections(familiar_id="fam", limit=10) == []

    @pytest.mark.asyncio
    async def test_writes_reflection_when_threshold_crossed(self) -> None:
        store = HistoryStore(":memory:")
        turn_ids = _seed_turns(store, 25)
        fact_ids = _seed_facts(store, 4)
        reply = (
            '[{"text": "Crew morale dipped after Friday.", '
            f'"cited_turn_ids": [{turn_ids[3]}, {turn_ids[10]}], '
            f'"cited_fact_ids": [{fact_ids[1]}]'
            "}]"
        )
        llm = _ScriptedLLM(replies=[reply])

        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()

        rows = store.recent_reflections(familiar_id="fam", limit=10)
        assert len(rows) == 1
        r = rows[0]
        assert "morale" in r.text
        assert turn_ids[3] in r.cited_turn_ids
        assert fact_ids[1] in r.cited_fact_ids
        # Watermark snapshots the worker's view at write time.
        assert r.last_turn_id == turn_ids[-1]
        assert r.last_fact_id == fact_ids[-1]

    @pytest.mark.asyncio
    async def test_does_not_refire_until_more_turns_arrive(self) -> None:
        store = HistoryStore(":memory:")
        turn_ids = _seed_turns(store, 25)
        first_reply = (
            f'[{{"text": "first reflection", '
            f'"cited_turn_ids": [{turn_ids[0]}], "cited_fact_ids": []}}]'
        )
        llm = _ScriptedLLM(replies=[first_reply, "[]"])

        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()
        assert len(llm.calls) == 1
        # No new turns — should be a noop.
        await worker.tick()
        assert len(llm.calls) == 1
        # Now add 25 more turns to cross the threshold.
        _seed_turns(store, 25)
        await worker.tick()
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_writes_multiple_rows_when_llm_returns_multiple(self) -> None:
        store = HistoryStore(":memory:")
        turn_ids = _seed_turns(store, 25)
        reply = (
            "["
            f'{{"text": "first", "cited_turn_ids": [{turn_ids[0]}], '
            '"cited_fact_ids": []},'
            f'{{"text": "second", "cited_turn_ids": [{turn_ids[1]}], '
            '"cited_fact_ids": []}'
            "]"
        )
        llm = _ScriptedLLM(replies=[reply])

        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()

        rows = store.recent_reflections(familiar_id="fam", limit=10)
        assert {r.text for r in rows} == {"first", "second"}

    @pytest.mark.asyncio
    async def test_drops_rows_with_unknown_citations(self) -> None:
        store = HistoryStore(":memory:")
        turn_ids = _seed_turns(store, 25)
        # 999 isn't a real turn id — citation must be dropped from the
        # row, but the row itself should still land if any cited id is
        # valid.
        reply = (
            "["
            f'{{"text": "valid", "cited_turn_ids": [{turn_ids[0]}, 999], '
            '"cited_fact_ids": []}'
            "]"
        )
        llm = _ScriptedLLM(replies=[reply])
        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()
        rows = store.recent_reflections(familiar_id="fam", limit=10)
        assert len(rows) == 1
        assert rows[0].cited_turn_ids == (turn_ids[0],)

    @pytest.mark.asyncio
    async def test_skips_row_with_empty_text(self) -> None:
        store = HistoryStore(":memory:")
        turn_ids = _seed_turns(store, 25)
        reply = (
            f'[{{"text": "", "cited_turn_ids": [{turn_ids[0]}], "cited_fact_ids": []}}]'
        )
        llm = _ScriptedLLM(replies=[reply])
        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        await worker.tick()
        assert store.recent_reflections(familiar_id="fam", limit=10) == []

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_output(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 25)
        llm = _ScriptedLLM(replies=["not json at all"])
        worker = ReflectionWorker(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            turns_threshold=20,
        )
        # Should not raise.
        await worker.tick()
        assert store.recent_reflections(familiar_id="fam", limit=10) == []
