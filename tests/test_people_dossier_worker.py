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

from familiar_connect.history.async_store import AsyncHistoryStore
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
        yield reply.content_str


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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        await worker.tick()

        # Prior dossier was fed into the prompt.
        assert len(llm.calls) == 1
        joined = "\n".join(m.content_str for m in llm.calls[0])
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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        await worker.tick()

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="discord:1")
        assert entry is not None
        assert entry.dossier_text == "keep me"

    @pytest.mark.asyncio
    async def test_builds_self_dossier_with_familiar_label(self) -> None:
        """Self-keyed fact compounds a self-dossier; prompt uses the name."""
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store,
            text="Sapphire ran a gaslighting bit and felt proud.",
            canonical_key="ego:fam",
            display="Sapphire",
        )
        llm = _ScriptedLLM(replies=["Sapphire: enjoys running provocative bits."])

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="ego:fam")
        assert entry is not None
        assert entry.last_fact_id == 1
        # prompt addressed the familiar by name, not the raw key
        joined = "\n".join(m.content_str for m in llm.calls[0])
        assert "Sapphire" in joined
        assert "ego:fam" not in joined

    @pytest.mark.asyncio
    async def test_self_dossier_strips_echoed_importance_tag(self) -> None:
        """Strip any ``(importance N)`` tag the writer echoes into prose."""
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store,
            text="Sapphire guards her autonomy fiercely.",
            canonical_key="ego:fam",
            display="Sapphire",
        )
        llm = _ScriptedLLM(
            replies=[
                "(importance 8) Sapphire guards her autonomy and keeps her own records."
            ]
        )
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        entry = store.get_people_dossier(familiar_id="fam", canonical_key="ego:fam")
        assert entry is not None
        assert "(importance" not in entry.dossier_text
        assert "Sapphire guards her autonomy" in entry.dossier_text

    @pytest.mark.asyncio
    async def test_self_dossier_prompt_preserves_opinions(self) -> None:
        """Self-record keeps durable feelings/opinions; person-dossier sheds them.

        Self-dossier is the substrate for consistently-forming opinions
        (feeds the sleep cycle), so it must NOT use the person prompt's
        blanket 'drop transient feelings'.
        """
        store = HistoryStore(":memory:")
        _seed_subject_fact(
            store,
            text="Sapphire warmed to SpaceFish and stays wary of KaillaDame.",
            canonical_key="ego:fam",
            display="Sapphire",
        )
        llm = _ScriptedLLM(replies=["Sapphire: warm to SpaceFish, wary of KaillaDame."])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        system = next(m for m in llm.calls[0] if m.role == "system").content_str
        low = system.lower()
        # keeps settled opinions/stances; does not blanket-drop feelings
        assert "opinion" in low or "stance" in low
        assert "drop transient feelings" not in low

    @pytest.mark.asyncio
    async def test_self_dossier_excludes_low_importance_texture(self) -> None:
        """Texture-tier self facts (low importance) stay out of the dossier.

        The dream pass writes momentary 'texture' opinions at importance
        2-3; they live in the DB (RAG-recallable) but must not flood the
        always-injected self-dossier. Durable stances (7-9) and legacy
        NULL-importance facts still build it.
        """
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Sapphire is fiercely protective of her autonomy.",
            importance=9,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Sapphire was briefly curious about dream BLTs.",
            importance=2,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Sapphire keeps records out of habit.",  # NULL importance — legacy
        )
        llm = _ScriptedLLM(replies=["Sapphire: autonomous, keeps records."])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        body = "\n".join(m.content_str for m in llm.calls[0])
        assert "autonomy" in body  # durable kept
        assert "keeps records" in body  # legacy NULL kept
        assert "dream BLTs" not in body  # texture excluded

    @pytest.mark.asyncio
    async def test_self_dossier_orders_facts_by_importance_desc(self) -> None:
        """Self facts feed the prompt importance-desc; NULL sits in the 5-band.

        Stable sort preserves insertion (recency) order within a tier.
        Facts seeded in order [5, 9, 7, None] must render 9, 7, then the
        5-band (the importance-5 fact and the NULL fact, in insertion
        order: 5 was seeded before None).
        """
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),)

        def _add(text: str, importance: int | None) -> None:
            store.append_fact(
                familiar_id="fam",
                channel_id=1,
                source_turn_ids=[1],
                subjects=subj,
                text=text,
                importance=importance,
            )

        _add("fact-five.", 5)
        _add("fact-nine.", 9)
        _add("fact-seven.", 7)
        _add("fact-null.", None)
        llm = _ScriptedLLM(replies=["ok"])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        body = next(m for m in llm.calls[0] if m.role == "user").content_str
        order = [
            body.index("fact-nine."),
            body.index("fact-seven."),
            body.index("fact-five."),
            body.index("fact-null."),
        ]
        assert order == sorted(order)  # 9, 7, then 5-band (5 before NULL)

    @pytest.mark.asyncio
    async def test_self_dossier_annotates_and_biases_by_importance(self) -> None:
        """Self body tags scored facts `(importance N)`; header gives bias rule."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Sapphire guards her autonomy.",
            importance=9,
        )
        llm = _ScriptedLLM(replies=["ok"])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        body = next(m for m in llm.calls[0] if m.role == "user").content_str
        assert "- (importance 9) Sapphire guards her autonomy." in body
        system = next(m for m in llm.calls[0] if m.role == "system").content_str
        assert "importance" in system.lower()
        assert "weight higher-importance" in system.lower()

    @pytest.mark.asyncio
    async def test_self_dossier_null_importance_renders_untagged(self) -> None:
        """NULL-importance self fact appears, but without a numeric tag."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Sapphire keeps records out of habit.",  # NULL importance
        )
        llm = _ScriptedLLM(replies=["ok"])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        body = next(m for m in llm.calls[0] if m.role == "user").content_str
        assert "- Sapphire keeps records out of habit." in body
        assert "(importance" not in body  # unscored: no tag

    @pytest.mark.asyncio
    async def test_non_self_dossier_no_importance_tags_or_bias(self) -> None:
        """Non-self body has no importance tags; header has no bias clause."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:9", display_at_write="Aria"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Aria likes pho.",
            importance=8,
        )
        llm = _ScriptedLLM(replies=["Aria: likes pho."])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()

        body = next(m for m in llm.calls[0] if m.role == "user").content_str
        assert "- Aria likes pho." in body  # plain render, no tag
        assert "(importance" not in body
        system = next(m for m in llm.calls[0] if m.role == "system").content_str
        assert "weight higher-importance" not in system.lower()

    @pytest.mark.asyncio
    async def test_non_self_dossier_keeps_all_importances(self) -> None:
        """Importance filter is self-scoped — other people's dossiers unchanged."""
        store = HistoryStore(":memory:")
        subj = (FactSubject(canonical_key="discord:9", display_at_write="Aria"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            source_turn_ids=[1],
            subjects=subj,
            text="Aria mentioned a minor preference.",
            importance=2,
        )
        llm = _ScriptedLLM(replies=["Aria: has a minor preference."])
        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
        )
        await worker.tick()
        body = "\n".join(m.content_str for m in llm.calls[0])
        assert "minor preference" in body  # low-importance non-self fact kept

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

        worker = PeopleDossierWorker(
            store=AsyncHistoryStore(store), llm_client=llm, familiar_id="fam"
        )
        await worker.tick()

        assert llm.calls == []
