"""Tests for :class:`FactExtractor` — watermark-driven fact extraction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from familiar_connect.activities import (
    ACTIVITY_RETURN_MODE,
    RETURN_TURN_MARKER_PREFIX,
    SLEEP_RETURN_MODE,
)
from familiar_connect.config import load_character_config
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.fact_extractor import FactExtractor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Real dream-framing clause = the merged ``_default`` config value (single
# source of truth); no in-code copy. Mirrors production wiring.
_DEFAULT_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "familiars"
    / "_default"
    / "character.toml"
)


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
        yield reply.content_str


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
            store=AsyncHistoryStore(store),
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=5,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        assert "self-capability" in system_msg.content_str.lower() or (
            "your own" in system_msg.content_str.lower()
        ), system_msg.content_str

    @pytest.mark.asyncio
    async def test_extract_prompt_distinguishes_claims_and_fiction(self) -> None:
        """Extractor's instruction must demand claim attribution + fiction handling.

        One speaker's assertions about another person: stored attributed,
        never flat. Roleplay events: recorded as bits, never real events.
        Guards the recontamination path — extractor processes every turn
        regardless of whether the assistant engaged.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        text = system_msg.content_str.lower()
        assert "claim" in text, system_msg.content_str
        assert "fiction" in text, system_msg.content_str
        assert "running joke" in text, system_msg.content_str

    @pytest.mark.asyncio
    async def test_extract_prompt_guards_identity_impersonation(self) -> None:
        """Prompt must forbid minting identity facts from impersonation bits.

        A member play-acting as another person ("No I am Cor") must not
        become an identity fact or merge two participants — the pants↔Cor
        dossier bleed. Identity ties to canonical_key, never an adopted name.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        text = system_msg.content_str.lower()
        assert "impersonat" in text, system_msg.content_str
        assert "distinct" in text, system_msg.content_str

    @pytest.mark.asyncio
    async def test_extract_prompt_guards_world_trivia(self) -> None:
        """Generic trivia a speaker mentions isn't a fact ABOUT them.

        Helios typing Pokémon lore became his only 'fact', yielding a
        junk dossier. Trivia/game-lore must be skipped or left subjectless.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        text = system_msg.content_str.lower()
        assert "trivia" in text, system_msg.content_str
        assert "subjectless" in text, system_msg.content_str


class TestFactExtractorActivityReturnSkip:
    """Activity-return turns never enter extraction (v1 provenance).

    Experience text persisted on return is self-generated fiction;
    only the engine's mechanical event-fact records the activity.
    Skip keyed on ``turns.mode == ACTIVITY_RETURN_MODE`` — the
    content prefix is display-only.
    """

    @pytest.mark.asyncio
    async def test_user_message_with_marker_prefix_not_skipped(self) -> None:
        """Display prefix in ordinary user content must still extract."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 9)
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="[returned from vacation] anyway, ask me about Lisbon",
            author=None,
        )
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        assert len(llm.calls) == 1
        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        # no mode tag ⇒ regular user turn ⇒ shown to LLM
        assert "ask me about Lisbon" in prompt_text

    @pytest.mark.asyncio
    async def test_return_turn_excluded_from_extraction_batch(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 9)
        marker = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="system",
            content=f"{RETURN_TURN_MARKER_PREFIX}creek walk] watched a heron fish",
            author=None,
            mode=ACTIVITY_RETURN_MODE,
        )
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        assert len(llm.calls) == 1
        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        # normal turns still extracted; return turn never shown to LLM
        assert "turn 0" in prompt_text
        assert "heron" not in prompt_text
        # watermark still advances over the marker turn — no loop
        wm = store.get_writer_watermark(familiar_id="fam")
        assert wm is not None
        assert wm.last_written_id == marker.id

    @pytest.mark.asyncio
    async def test_fallback_sources_exclude_return_turn(self) -> None:
        """Whole-batch source fallback must not cite the return turn."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 9)
        marker = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="system",
            content=f"{RETURN_TURN_MARKER_PREFIX}creek walk] watched a heron fish",
            author=None,
            mode=ACTIVITY_RETURN_MODE,
        )
        llm = _ScriptedLLM(
            replies=[
                # no source_turn_ids ⇒ extractor falls back to whole batch
                _facts_json([{"text": "Aria likes strawberries."}])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == "Aria likes strawberries."
        assert marker.id not in facts[0].source_turn_ids

    @pytest.mark.asyncio
    async def test_all_return_batch_skips_llm_but_advances_watermark(self) -> None:
        store = HistoryStore(":memory:")
        last_id = 0
        for i in range(10):
            t = store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="system",
                content=f"{RETURN_TURN_MARKER_PREFIX}walk {i}] saw things",
                author=None,
                mode=ACTIVITY_RETURN_MODE,
            )
            last_id = t.id
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        assert llm.calls == []
        wm = store.get_writer_watermark(familiar_id="fam")
        assert wm is not None
        assert wm.last_written_id == last_id


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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].subjects == ()


class TestFactExtractorSelfSubject:
    """Familiar's OWN narrative routes to the reserved ``self:`` subject.

    The familiar's performances/bits, choices, and relational
    stances/feelings get a home (the self-dossier) instead of
    poisoning whichever person the bit was about. Self-CAPABILITY
    statements stay dropped — capabilities/limits aren't narrative.
    """

    @pytest.mark.asyncio
    async def test_prompt_teaches_self_key_and_name(self) -> None:
        """Manifest carries self key + name; routes narrative, bans capability."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        assert "ego:fam" in prompt_text
        assert "Sapphire" in prompt_text
        lower = prompt_text.lower()
        # routes narrative/feelings to self key
        assert "ego:fam" in prompt_text
        # still forbids self-capability
        assert "self-capability" in lower or "your own" in lower

    @pytest.mark.asyncio
    async def test_self_narrative_routed_to_self_subject(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Sapphire ran a bit and privately felt proud.",
                        "source_turn_ids": [1],
                        "subject_keys": ["ego:fam"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].subjects == (
            FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),
        )

    @pytest.mark.asyncio
    async def test_self_key_not_mirrored_into_turn_mentions(self) -> None:
        """Self key is always-injected — must not enter ``turn_mentions``.

        Mirroring it would pollute the index and consume a ``max_people``
        slot in ``PeopleDossierLayer``, breaking the self cap-exemption.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Sapphire chose to disengage.",
                        "source_turn_ids": [3],
                        "subject_keys": ["ego:fam"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        # fact still carries the self subject, but the turn is not mention-tagged
        assert store.recent_facts(familiar_id="fam", limit=10)[0].subjects == (
            FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),
        )
        assert store.mentions_for_turn(turn_id=3) == ()

    @pytest.mark.asyncio
    async def test_self_capability_still_dropped_with_self_key(self) -> None:
        """Self-key routing must NOT regress the capability post-filter."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "I cannot remember names.",
                        "source_turn_ids": [1],
                        "subject_keys": ["ego:fam"],
                    },
                    {
                        "text": "Sapphire chose to walk away from the argument.",
                        "source_turn_ids": [2],
                        "subject_keys": ["ego:fam"],
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in facts}
        assert texts == {"Sapphire chose to walk away from the argument."}, texts

    @pytest.mark.asyncio
    async def test_display_name_capability_dropped_narrative_kept(self) -> None:
        """Third-person self-naming may phrase a capability with the name.

        "Sapphire cannot remember names" is a capability, not narrative —
        must be dropped even though it isn't first-person.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Sapphire cannot remember names.",
                        "source_turn_ids": [1],
                        "subject_keys": ["ego:fam"],
                    },
                    {
                        "text": "Sapphire chose to walk away.",
                        "source_turn_ids": [2],
                        "subject_keys": ["ego:fam"],
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in facts}
        assert texts == {"Sapphire chose to walk away."}, texts

    @pytest.mark.asyncio
    async def test_display_name_capability_no_false_positives(self) -> None:
        """Name-capability filter is inability-only — must not eat narrative.

        Word-prefix collisions ("cancelled" ⊃ "can"), copula/dynamic
        negation ("is not fond", "doesn't trust"), and positive ability
        are NARRATIVE/stance — the self-dossier's payload — and must stay.
        Only genuine inability ("cannot", "has no", "is unable") drops.
        """
        keep = [
            "Sapphire cancelled the movie night.",  # 'can' prefix
            "Sapphire candidly admitted she was wrong.",  # 'can' prefix
            "Sapphire is not fond of KaillaDame.",  # relational stance
            "Sapphire doesn't trust easily.",  # trait/stance
            "Sapphire can sing surprisingly well.",  # positive ability
        ]
        drop = [
            "Sapphire cannot remember names.",
            "Sapphire has no internet access.",
        ]
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {"text": t, "source_turn_ids": [1], "subject_keys": ["ego:fam"]}
                    for t in keep + drop
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            familiar_display_name="Sapphire",
            batch_size=10,
        )
        await extractor.tick()

        texts = {f.text for f in store.recent_facts(familiar_id="fam", limit=20)}
        assert texts == set(keep), texts


class TestFactExtractorParticipantsWidening:
    """Manifest extends past batch authors to recent channel participants.

    Without widening, a batch of 10 turns where only Cass speaks
    forecloses on linking any other name in the turn text to a
    canonical key — the manifest would only carry Cass. ``recent_distinct_authors``
    on the channel (capped at 30) widens the manifest to people who
    spoke in the recent past, so the LLM can resolve "what about Aria?"
    to ``discord:aria_id`` even when Aria didn't speak in this batch.
    """

    @pytest.mark.asyncio
    async def test_manifest_includes_prior_channel_authors(self) -> None:
        store = HistoryStore(":memory:")
        # Aria spoke earlier (now outside the batch window).
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi from aria",
            author=Author(
                platform="discord",
                user_id="222",
                username="aria_login",
                display_name="Aria",
            ),
        )
        # 10 turns from Cass — fills the batch.
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
        # Pre-advance the watermark past Aria's turn so she's outside the batch.
        store.put_writer_watermark(familiar_id="fam", last_written_id=1)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        # Both speakers must be in the manifest, despite Aria being outside the batch.
        assert "discord:111" in prompt_text  # Cass (batch author)
        assert "discord:222" in prompt_text  # Aria (recent prior author)
        assert "Aria" in prompt_text

    @pytest.mark.asyncio
    async def test_subject_keys_resolve_against_widened_manifest(self) -> None:
        """An LLM-emitted subject_key for a non-batch author is now accepted."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi from aria",
            author=Author(
                platform="discord",
                user_id="222",
                username="aria_login",
                display_name="Aria",
            ),
        )
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"chat {i} mentioning Aria",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        store.put_writer_watermark(familiar_id="fam", last_written_id=1)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cass talked about Aria's bakery.",
                        "source_turn_ids": [3],
                        # Aria's key would have been dropped without widening.
                        "subject_keys": ["discord:111", "discord:222"],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        keys = {s.canonical_key for s in facts[0].subjects}
        assert keys == {"discord:111", "discord:222"}
        # And turn_mentions picks up Aria too (no Discord ping needed).
        assert "discord:222" in store.mentions_for_turn(turn_id=3)

    @pytest.mark.asyncio
    async def test_manifest_capped_at_total_limit(self) -> None:
        """Bounded by ``participants_max`` so the prompt doesn't bloat unboundedly."""
        store = HistoryStore(":memory:")
        # Seed 50 distinct prior authors in channel 1.
        for i in range(50):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content="x",
                author=Author(
                    platform="discord",
                    user_id=f"prior-{i}",
                    username=f"u{i}",
                    display_name=f"U{i}",
                ),
            )
        # Then a batch of 10 from Cass.
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"batch {i}",
                author=Author(
                    platform="discord",
                    user_id="111",
                    username="cass_login",
                    display_name="Cass",
                ),
            )
        store.put_writer_watermark(familiar_id="fam", last_written_id=50)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
            participants_max=30,
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        # Manifest lines look like ``- canonical_key — display``; count them.
        manifest_count = sum(
            1
            for line in prompt_text.splitlines()
            if line.startswith("- discord:") and " — " in line
        )
        assert manifest_count == 30

    @pytest.mark.asyncio
    async def test_widening_scoped_per_channel(self) -> None:
        """Authors from other channels don't leak into the manifest."""
        store = HistoryStore(":memory:")
        # Aria spoke in channel 99, never in channel 1.
        store.append_turn(
            familiar_id="fam",
            channel_id=99,
            role="user",
            content="other channel",
            author=Author(
                platform="discord",
                user_id="222",
                username="aria_login",
                display_name="Aria",
            ),
        )
        # Batch of 10 from Cass on channel 1.
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
        store.put_writer_watermark(familiar_id="fam", last_written_id=1)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        prompt_text = "\n".join(m.content_str for m in llm.calls[0])
        assert "discord:111" in prompt_text  # Cass present
        assert "discord:222" not in prompt_text  # Aria isolated to her channel


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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
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
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        # Both keys present, sorted by canonical_key as the read API guarantees.
        assert store.mentions_for_turn(turn_id=3) == ("discord:111", "discord:222")


class TestFactExtractorBiTemporal:
    """Bi-temporal seed values come from the source turn (M1).

    The extractor's default ``valid_from`` is the timestamp of the
    first source turn — the moment the world is observed to be in this
    state. The LLM may override with an explicit ``valid_from`` ISO-
    timestamp string when it spotted an "as of …" phrase.
    """

    @pytest.mark.asyncio
    async def test_default_valid_from_matches_source_turn_timestamp(self) -> None:
        store = HistoryStore(":memory:")
        ids = _seed_turns(store, 10)
        recents_turns = store.recent(familiar_id="fam", channel_id=1, limit=20)
        ts_by_id = {t.id: t.timestamp for t in recents_turns}

        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Aria likes strawberries.",
                        "source_turn_ids": [ids[0], ids[2]],
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].valid_from == ts_by_id[ids[0]]
        assert facts[0].valid_to is None

    @pytest.mark.asyncio
    async def test_llm_valid_from_override_parsed(self) -> None:
        """Explicit ``valid_from`` in the LLM reply overrides the turn timestamp."""
        store = HistoryStore(":memory:")
        ids = _seed_turns(store, 10)
        override = datetime(2024, 1, 15, tzinfo=UTC)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Aria moved to Berlin in early 2024.",
                        "source_turn_ids": [ids[0]],
                        "valid_from": override.isoformat(),
                    }
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].valid_from == override

    @pytest.mark.asyncio
    async def test_extract_prompt_documents_valid_from_field(self) -> None:
        """Prompt must teach the LLM about the optional ``valid_from`` field."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        assert "valid_from" in system_msg.content

    @pytest.mark.asyncio
    async def test_extract_prompt_warns_off_retirement_use_of_valid_to(self) -> None:
        """``valid_to`` is world-time, not a retirement marker.

        Conflating the two leaks bookkeeping into the validity column
        and bypasses the supersession path. The prompt must steer the
        LLM away from setting ``valid_to`` just because a fact looks
        outdated.
        """
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        content_lower = system_msg.content_str.lower()
        # Must mention valid_to is bounded to a known end (world-time).
        assert "valid_to" in system_msg.content_str
        # And must explicitly warn off retirement / replacement use.
        assert "outdated" in content_lower or "replaced" in content_lower
        assert "supersed" in content_lower


class TestFactExtractorImportance:
    """M2 — extractor emits a 1-10 importance hint per fact."""

    @pytest.mark.asyncio
    async def test_extract_prompt_documents_importance_field(self) -> None:
        """Prompt must teach the LLM the 1-10 importance scale."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()

        system_msg = next(m for m in llm.calls[0] if m.role == "system")
        assert "importance" in system_msg.content
        # 1-10 scale referenced explicitly so the model doesn't invent
        # a 0-1 or 0-100 scale.
        assert "1" in system_msg.content
        assert "10" in system_msg.content

    @pytest.mark.asyncio
    async def test_persists_importance_when_emitted(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Aria is allergic to peanuts.",
                        "source_turn_ids": [1],
                        "importance": 9,
                    },
                    {
                        "text": "Boris had cereal for breakfast.",
                        "source_turn_ids": [3],
                        "importance": 2,
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()
        facts = {f.text: f for f in store.recent_facts(familiar_id="fam", limit=10)}
        assert facts["Aria is allergic to peanuts."].importance == 9
        assert facts["Boris had cereal for breakfast."].importance == 2

    @pytest.mark.asyncio
    async def test_missing_importance_persists_as_none(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {"text": "A fact.", "source_turn_ids": [1]},
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert facts[0].importance is None

    @pytest.mark.asyncio
    async def test_invalid_importance_clamps_or_drops(self) -> None:
        """Out-of-range / non-integer values reach the store, which clamps."""
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Too high.",
                        "source_turn_ids": [1],
                        "importance": 99,
                    },
                    {
                        "text": "Negative.",
                        "source_turn_ids": [3],
                        "importance": -3,
                    },
                    {
                        "text": "Garbage.",
                        "source_turn_ids": [5],
                        "importance": "very important",
                    },
                ])
            ]
        )
        extractor = FactExtractor(
            store=AsyncHistoryStore(store),
            llm_client=llm,
            familiar_id="fam",
            batch_size=10,
        )
        await extractor.tick()
        facts = {f.text: f for f in store.recent_facts(familiar_id="fam", limit=10)}
        assert facts["Too high."].importance == 10
        assert facts["Negative."].importance == 1
        # Non-numeric input drops silently to None — no poison ranking.
        assert facts["Garbage."].importance is None


def _turn_count_in_prompt(messages: list[Message]) -> int:
    """Count how many ``- [role] ...`` lines appear in the user message.

    Mirrors the worker's prompt layout; a test-level proxy for "how
    many turns did this extraction see?".
    """
    user_msg = next((m for m in messages if m.role == "user"), None)
    if user_msg is None:
        return 0
    # ``- id=`` distinguishes turn lines from manifest entries
    # (``- <key> — <name>``), now always non-empty via the self-subject.
    return sum(
        1 for line in user_msg.content_str.splitlines() if line.startswith("- id=")
    )


def _seed_with_dream_turn(store: HistoryStore) -> tuple[list[int], int]:
    """9 authored user turns + 1 sleep_return dream turn (batch of 10)."""
    author = Author(platform="discord", user_id="1", username="cor", display_name="Cor")
    normal_ids = [
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content=f"turn {i}",
            author=author,
        ).id
        for i in range(9)
    ]
    dream = store.append_turn(
        familiar_id="fam",
        channel_id=1,
        role="assistant",
        content=f"{RETURN_TURN_MARKER_PREFIX}asleep] The archive sang to me.",
        author=None,
        mode=SLEEP_RETURN_MODE,
    )
    return normal_ids, dream.id


def _default_dream_clause() -> str:
    """Real clause sourced from the merged ``_default`` config (no copy)."""
    cfg = load_character_config(_DEFAULT_PROFILE, defaults_path=_DEFAULT_PROFILE)
    return cfg.dream_extraction_clause


def _dream_extractor(
    store: HistoryStore,
    llm: _ScriptedLLM,
    *,
    dream_extraction_clause: str | None = None,
) -> FactExtractor:
    return FactExtractor(
        store=AsyncHistoryStore(store),
        llm_client=llm,
        familiar_id="fam",
        familiar_display_name="Sapphire",
        batch_size=10,
        dream_extraction_clause=(
            dream_extraction_clause
            if dream_extraction_clause is not None
            else _default_dream_clause()
        ),
    )


class TestSleepReturnDreamExtraction:
    """``sleep_return`` turns are PROCESSED with dream framing.

    Claim-discipline rail enforced in code: any fact grounded in a
    dream turn lands under ``self:`` ONLY, dream-framed — never under
    a person's key.
    """

    @pytest.mark.asyncio
    async def test_dream_turn_shown_to_llm_with_dream_rule(self) -> None:
        store = HistoryStore(":memory:")
        _seed_with_dream_turn(store)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        await _dream_extractor(store, llm).tick()
        assert len(llm.calls) == 1
        system = llm.calls[0][0].content_str
        user = llm.calls[0][1].content_str
        assert "The archive sang to me." in user
        assert "dream" in system.lower()
        assert "dreamed" in system

    @pytest.mark.asyncio
    async def test_no_dream_clause_without_dream_turns(self) -> None:
        store = HistoryStore(":memory:")
        _seed_turns(store, 10)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        await _dream_extractor(store, llm).tick()
        assert "dream" not in llm.calls[0][0].content_str.lower()

    @pytest.mark.asyncio
    async def test_configured_dream_clause_reaches_llm(self) -> None:
        """Caller-supplied clause template (with placeholders) is interpolated."""
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        clause = "DREAM-MARKER {self_name} keyed {self_key} ids {ids}"
        await _dream_extractor(store, llm, dream_extraction_clause=clause).tick()
        system = llm.calls[0][0].content_str
        assert "DREAM-MARKER Sapphire keyed ego:fam" in system
        assert str(dream_id) in system

    @pytest.mark.asyncio
    async def test_dream_clause_with_stray_brace_does_not_crash(self) -> None:
        """A clause override with a stray brace / missing placeholder degrades.

        An override changes phrasing, never crashes the pass. A literal
        ``{`` and an unknown ``{please}`` token pass through verbatim; a
        valid ``{self_name}`` placeholder still fills.
        """
        store = HistoryStore(":memory:")
        _seed_with_dream_turn(store)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        clause = "tidy up {please} for {self_name} a { brace"
        await _dream_extractor(store, llm, dream_extraction_clause=clause).tick()
        system = llm.calls[0][0].content_str
        # brace / unknown token pass through literally; valid one fills
        assert "tidy up {please} for Sapphire a { brace" in system

    @pytest.mark.asyncio
    async def test_claim_discipline_rail_fires_with_config_clause(self) -> None:
        """Code rail forces self-subject + framing even when clause is config-sourced.

        A config override changes the dream-clause phrasing; it can't stop
        the code rail from re-keying a dream-grounded fact to ``self:`` and
        dream-framing it.
        """
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cor fought a dragon in the rafters.",
                        "source_turn_ids": [dream_id],
                        "subject_keys": ["discord:1"],
                    }
                ])
            ]
        )
        await _dream_extractor(
            store, llm, dream_extraction_clause="say anything about {self_name}"
        ).tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == (
            "Sapphire dreamed that Cor fought a dragon in the rafters."
        )
        assert facts[0].subjects == (
            FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),
        )

    @pytest.mark.asyncio
    async def test_dream_fact_forced_to_self_subject_and_framed(self) -> None:
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        # model misbehaves: flat claim, person-subject
                        "text": "Cor fought a dragon in the rafters.",
                        "source_turn_ids": [dream_id],
                        "subject_keys": ["discord:1"],
                    }
                ])
            ]
        )
        await _dream_extractor(store, llm).tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == (
            "Sapphire dreamed that Cor fought a dragon in the rafters."
        )
        assert facts[0].subjects == (
            FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),
        )
        # forced-self facts never enter turn_mentions
        assert store.mentions_for_turn(turn_id=dream_id) == ()

    @pytest.mark.asyncio
    async def test_already_dream_framed_text_kept_verbatim(self) -> None:
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Sapphire dreamed the archive sang to her.",
                        "source_turn_ids": [dream_id],
                        "subject_keys": ["ego:fam"],
                    }
                ])
            ]
        )
        await _dream_extractor(store, llm).tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert facts[0].text == "Sapphire dreamed the archive sang to her."

    @pytest.mark.asyncio
    async def test_mixed_sources_count_as_dream(self) -> None:
        store = HistoryStore(":memory:")
        normal_ids, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(
            replies=[
                _facts_json([
                    {
                        "text": "Cor was in the dream too.",
                        "source_turn_ids": [normal_ids[0], dream_id],
                        "subject_keys": ["discord:1"],
                    }
                ])
            ]
        )
        await _dream_extractor(store, llm).tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert facts[0].subjects == (
            FactSubject(canonical_key="ego:fam", display_at_write="Sapphire"),
        )

    @pytest.mark.asyncio
    async def test_fallback_sources_exclude_dream_turn(self) -> None:
        """Unsourced facts fall back to NON-dream batch ids.

        Return-turn precedent — real facts about people stay
        person-attributable.
        """
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(replies=[_facts_json([{"text": "Cor likes strawberries."}])])
        await _dream_extractor(store, llm).tick()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == "Cor likes strawberries."
        assert dream_id not in facts[0].source_turn_ids

    @pytest.mark.asyncio
    async def test_watermark_advances_over_dream_turn(self) -> None:
        store = HistoryStore(":memory:")
        _, dream_id = _seed_with_dream_turn(store)
        llm = _ScriptedLLM(replies=[_facts_json([])])
        await _dream_extractor(store, llm).tick()
        wm = store.get_writer_watermark(familiar_id="fam")
        assert wm is not None
        assert wm.last_written_id == dream_id
