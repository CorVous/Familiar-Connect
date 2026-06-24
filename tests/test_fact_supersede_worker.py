"""Tests for :class:`FactSupersedeWorker` — LLM-driven fact retirement."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.fact_supersede_worker import FactSupersedeWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _ScriptedLLM(LLMClient):
    """LLM stub returning canned replies; records every prompt."""

    def __init__(self, *, replies: list[str]) -> None:
        super().__init__(api_key="k", model="m")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content='{"superseded_ids": []}')
        return Message(role="assistant", content=self._replies.pop(0))

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content_str


def _seed_subject_facts(
    store: HistoryStore,
    *,
    worker: FactSupersedeWorker | None = None,
    key: str = "discord:111",
    display: str = "Aria",
) -> tuple[int, int]:
    """Pre-existing fact, prime watermark, then a new fact arrives.

    Returns ``(old_id, new_id)``. If a *worker* is supplied, its
    watermark is primed between the two appends — matching real flow
    where the old fact was already in the store at startup.
    """
    subjects = (FactSubject(canonical_key=key, display_at_write=display),)
    old = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text=f"{display} loves hiking.",
        source_turn_ids=[1],
        subjects=subjects,
    )
    if worker is not None:
        worker.prime_watermark()
    new = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text=f"{display} hates hiking now.",
        source_turn_ids=[2],
        subjects=subjects,
    )
    return old.id, new.id


def _ids_json(ids: list[int]) -> str:
    return json.dumps({"superseded_ids": ids})


class TestFactSupersedeWorkerTick:
    @pytest.mark.asyncio
    async def test_no_new_facts_is_noop(self) -> None:
        store = HistoryStore(":memory:")
        _seed_subject_facts(store)
        llm = _ScriptedLLM(replies=[_ids_json([])])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        # Pre-advance watermark past everything.
        worker.prime_watermark()
        await worker.tick()
        # No LLM call when there's nothing new.
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_supersedes_prior_when_llm_flags_it(self) -> None:
        store = HistoryStore(":memory:")
        llm = _ScriptedLLM(replies=[])  # filled after we know old_id
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        old_id, new_id = _seed_subject_facts(store, worker=worker)
        llm._replies.append(_ids_json([old_id]))
        await worker.tick()

        current = store.recent_facts(familiar_id="fam", limit=10)
        # Old retired -> only the new one is current.
        assert len(current) == 1
        assert current[0].id == new_id
        # And the old row carries the supersede metadata.
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        old_row = next(f for f in all_facts if f.id == old_id)
        assert old_row.superseded_by == new_id
        assert old_row.superseded_at is not None

    @pytest.mark.asyncio
    async def test_empty_reply_leaves_facts_untouched(self) -> None:
        store = HistoryStore(":memory:")
        llm = _ScriptedLLM(replies=[_ids_json([])])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        old_id, new_id = _seed_subject_facts(store, worker=worker)
        await worker.tick()

        current = store.recent_facts(familiar_id="fam", limit=10)
        assert {f.id for f in current} == {old_id, new_id}

    @pytest.mark.asyncio
    async def test_hallucinated_id_outside_candidate_set_ignored(self) -> None:
        store = HistoryStore(":memory:")
        # 9999 isn't a real fact id; supersede must skip it, not crash.
        llm = _ScriptedLLM(replies=[_ids_json([9999])])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        old_id, new_id = _seed_subject_facts(store, worker=worker)
        await worker.tick()

        current = store.recent_facts(familiar_id="fam", limit=10)
        assert {f.id for f in current} == {old_id, new_id}

    @pytest.mark.asyncio
    async def test_bad_json_is_swallowed(self) -> None:
        store = HistoryStore(":memory:")
        llm = _ScriptedLLM(replies=["not json at all"])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        old_id, new_id = _seed_subject_facts(store, worker=worker)
        await worker.tick()

        # Bad reply -> no supersession; both facts current.
        current = store.recent_facts(familiar_id="fam", limit=10)
        assert {f.id for f in current} == {old_id, new_id}

    @pytest.mark.asyncio
    async def test_does_not_propose_self_supersede(self) -> None:
        """Worker must never instruct the LLM to retire the new fact itself."""
        store = HistoryStore(":memory:")
        llm = _ScriptedLLM(replies=[])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        old_id, new_id = _seed_subject_facts(store, worker=worker)
        # If the LLM erroneously names the new fact, the worker filters it.
        llm._replies.append(_ids_json([new_id]))
        await worker.tick()

        current = store.recent_facts(familiar_id="fam", limit=10)
        # New fact still current.
        assert new_id in {f.id for f in current}
        # And old still current too — LLM's only suggestion was filtered.
        assert old_id in {f.id for f in current}

    @pytest.mark.asyncio
    async def test_advances_watermark_so_next_tick_is_noop(self) -> None:
        store = HistoryStore(":memory:")
        llm = _ScriptedLLM(replies=[_ids_json([]), _ids_json([])])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        _seed_subject_facts(store, worker=worker)
        await worker.tick()
        first_count = len(llm.calls)
        # Same set, no new facts.
        await worker.tick()
        assert len(llm.calls) == first_count, (
            "second tick should not re-evaluate a fact already seen"
        )

    @pytest.mark.asyncio
    async def test_skips_facts_with_no_subjects(self) -> None:
        """Facts without canonical-key subjects can't be paired with priors."""
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="The weather is nice today.",
            source_turn_ids=[1],
        )
        llm = _ScriptedLLM(replies=[])
        worker = FactSupersedeWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        await worker.tick()

        # No subjects -> no LLM call.
        assert llm.calls == []
