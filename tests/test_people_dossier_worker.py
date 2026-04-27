"""Tests for :class:`PeopleDossierWorker`.

Mirrors :mod:`test_summary_worker`: scripted LLM, in-memory
:class:`HistoryStore`, watermark-driven refresh — but per
``canonical_key`` instead of per ``channel_id``.

Trigger: a subject's max ``facts.id`` exceeds its dossier's
``last_fact_id``. Compounding: prior dossier text is fed back into
the prompt with the new facts. Cadence and threshold mirror
:class:`SummaryWorker`'s shape so the worker family stays consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.people_dossier_worker import PeopleDossierWorker


class _ScriptedLLM(LLMClient):
    """Scripted LLM: pops one canned reply per ``chat`` call."""

    def __init__(self, *, replies: list[str]) -> None:
        super().__init__(api_key="k", model="m")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="(no more scripted replies)")
        return Message(role="assistant", content=self._replies.pop(0))

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content


def _seed_subject_fact(
    store: HistoryStore,
    *,
    text: str,
    canonical_key: str,
    display: str,
    familiar_id: str = "fam",
) -> int:
    fact = store.append_fact(
        familiar_id=familiar_id,
        channel_id=1,
        text=text,
        source_turn_ids=[1],
        subjects=(FactSubject(canonical_key=canonical_key, display_at_write=display),),
    )
    return fact.id


class TestPeopleDossierWorker:
    @pytest.mark.asyncio
    async def test_creates_dossier_for_new_subject(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="1",
                username="cass_login",
                display_name="Cass",
            )
        )
        _seed_subject_fact(
            store, text="Cass likes pho.", canonical_key="discord:1", display="Cass"
        )
        llm = _ScriptedLLM(replies=["Cass: enjoys pho."])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert "pho" in entry.dossier_text
        assert entry.last_fact_id == 1

    @pytest.mark.asyncio
    async def test_skips_subject_with_unchanged_watermark(self) -> None:
        store = HistoryStore(":memory:")
        _seed_subject_fact(store, text="A.", canonical_key="discord:1", display="C")
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="prior",
        )
        llm = _ScriptedLLM(replies=["should not be used"])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        assert llm.calls == []  # no LLM call when nothing new
        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert entry.dossier_text == "prior"  # unchanged

    @pytest.mark.asyncio
    async def test_compounds_prior_dossier_with_new_facts(self) -> None:
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store, text="Cass likes pho.", canonical_key="discord:1", display="Cass"
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=1,
            dossier_text="Cass likes pho.",
        )
        # New evidence appears.
        _seed_subject_fact(
            store,
            text="Cass moved to Toronto.",
            canonical_key="discord:1",
            display="Cass",
        )
        llm = _ScriptedLLM(replies=["Cass enjoys pho and recently moved to Toronto."])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        # Prior dossier was fed into the prompt.
        assert len(llm.calls) == 1
        joined = "\n".join(m.content for m in llm.calls[0])
        assert "Cass likes pho." in joined  # the prior dossier
        assert "Toronto" in joined  # the new fact

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert "Toronto" in entry.dossier_text
        assert entry.last_fact_id == 2

    @pytest.mark.asyncio
    async def test_handles_multiple_subjects_in_one_tick(self) -> None:
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store, text="Cass fact.", canonical_key="discord:1", display="Cass"
        )
        _seed_subject_fact(
            store, text="Aria fact.", canonical_key="discord:2", display="Aria"
        )
        llm = _ScriptedLLM(replies=["Cass dossier.", "Aria dossier."])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        c = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        a = store.get_people_dossier(familiar_id="fam", canonical_key="discord:2")
        assert c is not None
        assert a is not None
        assert c.dossier_text != a.dossier_text

    @pytest.mark.asyncio
    async def test_empty_llm_reply_does_not_overwrite(self) -> None:
        """A blank LLM reply must not blow away an existing dossier."""
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store, text="Cass likes pho.", canonical_key="discord:1", display="Cass"
        )
        store.put_people_dossier(
            familiar_id="fam",
            canonical_key="discord:1",
            last_fact_id=0,
            dossier_text="keep me",
        )
        # Force refresh by lowering the watermark below the latest fact.
        llm = _ScriptedLLM(replies=["   "])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert entry.dossier_text == "keep me"

    @pytest.mark.asyncio
    async def test_no_subjects_means_no_llm_call(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(  # subject-less fact
            familiar_id="fam",
            channel_id=1,
            text="A subject-less fact.",
            source_turn_ids=[1],
        )
        llm = _ScriptedLLM(replies=["unused"])

        worker = PeopleDossierWorker(store=store, llm_client=llm, familiar_id="fam")
        await worker.tick()

        assert llm.calls == []
