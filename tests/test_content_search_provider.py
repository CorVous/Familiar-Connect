"""End-to-end tests for the ContentSearchProvider orchestrator.

The provider stacks three tiers: deterministic people lookup →
optional embedding retrieval → single-shot filter. These tests
exercise the composition with scripted LLM stubs and verify the
never-forget-a-person invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.content_search import (
    FILTER_PRIORITY,
    FILTER_SOURCE,
    PEOPLE_LOOKUP_SOURCE,
    ContentSearchProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.llm import LLMClient, Message
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 100
_FAMILIAR = "aria"


def _request(utterance: str = "what do you know about Alice?") -> ContextRequest:
    return ContextRequest(
        familiar_id=_FAMILIAR,
        channel_id=_CHANNEL,
        guild_id=1,
        speaker="Alice",
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=2048,
        deadline_s=10.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Return a fresh MemoryStore for each test."""
    return MemoryStore(tmp_path / "memory")


class _ScriptedLLMClient(LLMClient):
    """LLMClient stub returning a queue of canned responses."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="scripted-test-key", model="scripted/test-model")
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._responses:
            msg = (
                "scripted LLM client ran out of responses; "
                f"chats so far: {len(self.calls)}"
            )
            raise RuntimeError(msg)
        return Message(role="assistant", content=self._responses.pop(0))

    async def close(self) -> None:  # pragma: no cover
        return

    def prompt_at(self, index: int) -> str:
        """Return the user-message text of the chat call at *index*."""
        return self.calls[index][0].content


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id_and_deadline(self, store: MemoryStore) -> None:
        provider = ContentSearchProvider(store=store, llm_client=_ScriptedLLMClient([]))
        assert provider.id == "content_search"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, store: MemoryStore) -> None:
        provider = ContentSearchProvider(store=store, llm_client=_ScriptedLLMClient([]))
        assert isinstance(provider, ContextProvider)


# ---------------------------------------------------------------------------
# Filter straight-through (no retriever, no people files)
# ---------------------------------------------------------------------------


class TestFilterStraightThrough:
    @pytest.mark.asyncio
    async def test_answer_yields_contribution(self, store: MemoryStore) -> None:
        side = _ScriptedLLMClient(["ANSWER: Aria knows Alice from last Tuesday."])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contribs = await provider.contribute(_request())

        assert len(contribs) == 1
        c = contribs[0]
        assert isinstance(c, Contribution)
        assert c.layer is Layer.content
        assert c.priority == FILTER_PRIORITY
        assert c.source == FILTER_SOURCE
        assert "Aria knows Alice" in c.text
        assert len(side.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_answer_no_contribution(self, store: MemoryStore) -> None:
        side = _ScriptedLLMClient(["ANSWER:"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        assert await provider.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_malformed_response_treated_as_answer(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedLLMClient(["Just text without a prefix."])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contribs = await provider.contribute(_request())
        assert len(contribs) == 1
        assert "Just text" in contribs[0].text


# ---------------------------------------------------------------------------
# Deterministic people-lookup + filter composition
# ---------------------------------------------------------------------------


class TestPeopleAndFilter:
    @pytest.mark.asyncio
    async def test_speaker_file_plus_filter_two_contributions(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/alice.md", "Alice is a ska fan from York.")
        side = _ScriptedLLMClient(["ANSWER: She's into ska."])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contribs = await provider.contribute(_request())

        people = [c for c in contribs if c.source == PEOPLE_LOOKUP_SOURCE]
        rag = [c for c in contribs if c.source == FILTER_SOURCE]
        assert len(people) == 1
        assert "ska fan from York" in people[0].text
        assert len(rag) == 1
        assert "into ska" in rag[0].text

    @pytest.mark.asyncio
    async def test_filter_prompt_mentions_utterance(self, store: MemoryStore) -> None:
        side = _ScriptedLLMClient(["ANSWER:"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        await provider.contribute(_request(utterance="tell me about ska"))
        assert "tell me about ska" in side.prompt_at(0)


# ---------------------------------------------------------------------------
# People-lookup invariant — the forgetting-bug fix
# ---------------------------------------------------------------------------


class TestPeopleLookupInvariant:
    """Deterministic-tier guarantees the forgetting-bug fix.

    Speaker and mentioned-name files are surfaced regardless of
    filter behaviour — empty ANSWER, exception, etc.
    """

    @pytest.mark.asyncio
    async def test_speaker_file_included_when_filter_returns_empty(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/alice.md", "Alice is a ska fan from York.")
        side = _ScriptedLLMClient(["ANSWER:"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contribs = await provider.contribute(_request(utterance="hi"))

        assert len(contribs) == 1
        assert contribs[0].source == PEOPLE_LOOKUP_SOURCE
        assert "ska fan from York" in contribs[0].text

    @pytest.mark.asyncio
    async def test_speaker_file_included_when_filter_raises(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/alice.md", "Alice's notes.")

        class _ExplodingClient(LLMClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="scripted-test-key", model="scripted/test-model"
                )

            async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
                msg = "simulated LLM failure"
                raise RuntimeError(msg)

            async def close(self) -> None:  # pragma: no cover
                return

        provider = ContentSearchProvider(store=store, llm_client=_ExplodingClient())
        contribs = await provider.contribute(_request())

        assert len(contribs) == 1
        assert contribs[0].source == PEOPLE_LOOKUP_SOURCE
        assert "Alice's notes" in contribs[0].text


# ---------------------------------------------------------------------------
# ESCALATE path through orchestrator
# ---------------------------------------------------------------------------


class TestEscalateEndToEnd:
    @pytest.mark.asyncio
    async def test_escalate_grep_then_answer(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "Alice mentioned she plays trombone.")
        side = _ScriptedLLMClient([
            "ESCALATE: need to check instrument; GREP: trombone",
            "ANSWER: Alice plays trombone.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contribs = await provider.contribute(_request())

        rag = [c for c in contribs if c.source == FILTER_SOURCE]
        assert len(rag) == 1
        assert "trombone" in rag[0].text.lower()
        assert len(side.calls) == 2
