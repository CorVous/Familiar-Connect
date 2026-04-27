"""Tests for :class:`FactExtractor` — watermark-driven fact extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.fact_extractor import FactExtractor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _ScriptedLLM(LLMClient):
    """LLM stub returning canned JSON fact replies."""

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


def _facts_json(items: list[dict[str, object]]) -> str:
    return json.dumps(items)


def _seed_turns(store: HistoryStore, count: int, channel_id: int = 1) -> list[int]:
    ids = []
    for i in range(count):
        t = store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"turn {i}",
            author=None,
        )
        ids.append(t.id)
    return ids


class TestFactExtractorTick:
    @pytest.mark.asyncio
    async def test_extracts_facts_from_new_turns(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 12)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {"text": "Aria likes strawberries.", "source_turn_ids": [1, 3]},
                    {
                        "text": "Boris works nights on Tuesdays.",
                        "source_turn_ids": [5, 7],
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=store,
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 2
        texts = {f.text for f in facts}
        assert "Aria likes strawberries." in texts
        assert "Boris works nights on Tuesdays." in texts

    @pytest.mark.asyncio
    async def test_advances_watermark_after_extract(self) -> None:
        store = HistoryStore(":memory:")
        ids = _seed_turns(store, 12)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        wm = store.get_writer_watermark(familiar_id="fam")
        assert wm is not None
        # Processed the first batch_size=10 turns; watermark = id of the 10th.
        assert wm.last_written_id == ids[9]

    @pytest.mark.asyncio
    async def test_second_tick_processes_only_new_turns(self) -> None:
        """Each tick sees only turns past the watermark.

        With ``batch_size=5``, a first tick of 10 turns processes 5
        and advances; a second tick processes the next 5.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([]), _facts_json([])])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=5
        )
        await extractor.tick()
        await extractor.tick()

        assert _turn_count_in_prompt(llm.calls[0]) == 5
        assert _turn_count_in_prompt(llm.calls[1]) == 5

    @pytest.mark.asyncio
    async def test_noop_below_batch_size(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 2)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        assert llm.calls == []
        # watermark not advanced
        assert store.get_writer_watermark(familiar_id="fam") is None

    @pytest.mark.asyncio
    async def test_invalid_json_reply_is_tolerated(self) -> None:
        """Malformed LLM output should not crash the worker.

        Watermark should still advance — otherwise we'd loop forever
        on the same bad turns.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=["not json at all, sorry"])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        assert store.recent_facts(familiar_id="fam", limit=10) == []
        wm = store.get_writer_watermark(familiar_id="fam")
        assert wm is not None  # still advanced

    @pytest.mark.asyncio
    async def test_self_capability_facts_dropped(self) -> None:
        """Self-capability statements aren't observations about the world.

        The store holds facts about people and events, not about the
        familiar's own abilities. Such "facts" expire the moment the
        capability changes; they shouldn't be persisted at all.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": (
                            "I cannot remember the names or faces of people "
                            "from before the current conversation."
                        ),
                        "source_turn_ids": [1],
                    },
                    {
                        "text": "The assistant does not have access to the internet.",
                        "source_turn_ids": [2],
                    },
                    {
                        "text": "As an AI, I have no personal preferences.",
                        "source_turn_ids": [3],
                    },
                    {
                        "text": "Aria likes strawberries.",
                        "source_turn_ids": [5],
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in facts}
        assert texts == {"Aria likes strawberries."}, texts

    @pytest.mark.asyncio
    async def test_extract_prompt_warns_off_self_capability(self) -> None:
        """Extractor's instruction must explicitly forbid self-capability.

        Best-effort line of defence before the post-filter.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        assert "self-capability" in system_msg.content.lower() or (
            "your own" in system_msg.content.lower()
        ), system_msg.content


class TestFactExtractorSubjects:
    """Participants manifest in prompt + ``subject_keys`` parsed back into facts.

    The extractor is the *only* place that knows both the canonical
    keys (from author rows) and the display names the LLM is using.
    Soft-link those by giving the LLM a manifest and asking it to
    optionally tag each fact with the canonical_keys it's about.

    Identity hints from the LLM are advisory — mic-sharing, relays,
    and ambiguity all break clean subject mapping. The extractor
    stores whatever the LLM emits without claiming authority.
    """

    @pytest.mark.asyncio
    async def test_prompt_includes_participants_manifest(self) -> None:
        """LLM must receive ``canonical_key → current display name`` pairs."""
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat from cass {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content for m in llm.calls[0])
        # The manifest must list canonical_key → display_name
        assert "discord:111" in prompt_text
        assert "Cass" in prompt_text

    @pytest.mark.asyncio
    async def test_extracts_subject_keys_into_fact_subjects(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cass likes pho.",
                        "source_turn_ids": [1],
                        "subject_keys": ["discord:111"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].subjects == (
            FactSubject(canonical_key="discord:111", display_at_write="Cass"),
        )

    @pytest.mark.asyncio
    async def test_subject_keys_outside_manifest_are_dropped(self) -> None:
        """Soft validation: the LLM may hallucinate keys; drop unknowns silently."""
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cass and a stranger ate pho.",
                        "source_turn_ids": [1],
                        "subject_keys": ["discord:111", "discord:does-not-exist"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert facts[0].subjects == (
            FactSubject(canonical_key="discord:111", display_at_write="Cass"),
        )

    @pytest.mark.asyncio
    async def test_facts_without_subject_keys_still_stored(self) -> None:
        """Subject_keys is optional — facts without it default to empty subjects."""
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    # No subject_keys field at all
                    {"text": "It rained on Tuesday.", "source_turn_ids": [1]},
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].subjects == ()


class TestFactExtractorMirrorsMentions:
    """Resolved subjects are mirrored into ``turn_mentions``.

    A bare-text reference ("what about Aria?") never lands a Discord
    ping — ``TextResponder`` only fills ``turn_mentions`` from
    ``message.mentions``. The fact extractor's name-resolution
    bridges that gap: when it identifies a subject, it records the
    canonical key against every source turn so
    :class:`PeopleDossierLayer` picks the person up at its next
    assemble — same cadence as Discord pings.
    """

    @pytest.mark.asyncio
    async def test_mirrors_subjects_into_turn_mentions(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        # Cass speaks; the LLM extracts a fact about her (Aria-style
        # cross-reference would land the same way once Aria is in
        # the participants manifest).
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cass likes pho.",
                        "source_turn_ids": [3, 5],
                        "subject_keys": ["discord:111"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        # Every source turn now lists the resolved subject.
        assert store.mentions_for_turn(turn_id=3) == ("discord:111",)
        assert store.mentions_for_turn(turn_id=5) == ("discord:111",)
        # Untouched turns get nothing.
        assert store.mentions_for_turn(turn_id=4) == ()

    @pytest.mark.asyncio
    async def test_no_subjects_means_no_mention_writes(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    # No subject_keys ⇒ nothing to mirror.
                    {"text": "It rained on Tuesday.", "source_turn_ids": [3]},
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        assert store.mentions_for_turn(turn_id=3) == ()

    @pytest.mark.asyncio
    async def test_mirror_does_not_clobber_prior_pings(self) -> None:
        """Idempotent: an existing pinged-mention coexists with mirrored subjects."""
        store = HistoryStore(":memory:")
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        # An earlier Discord @ ping already recorded Aria.
        store.record_mentions(turn_id=3, canonical_keys=["discord:222"])
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cass likes pho.",
                        "source_turn_ids": [3],
                        "subject_keys": ["discord:111"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=store, llm_client=llm, familiar_id="fam", batch_size=10
        )
        await extractor.tick()

        # Both keys present, sorted by canonical_key as the read API guarantees.
        assert store.mentions_for_turn(turn_id=3) == ("discord:111", "discord:222")


def _turn_count_in_prompt(messages: list[Message]) -> int:
    """Count how many ``- [role] ...`` lines appear in the user message.

    Mirrors the worker's prompt layout; a test-level proxy for "how
    many turns did this extraction see?".
    """
    user_msg = next((m for m in messages if m.role == "user"), None)
    if user_msg is None:
        return 0
    return sum(1 for line in user_msg.content.splitlines() if line.startswith("- "))
