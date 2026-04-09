"""Red-first tests for the ContentSearchProvider.

Step 8 of future-features/context-management.md. The interesting one:
a cheap tool-using model with grep / glob / read_file / list_dir
tools scoped to a single familiar's MemoryStore. Each contribute()
call runs a small loop — the model decides which tools to call, the
provider executes them against the store, the model eventually
returns an ``ANSWER:`` and the provider wraps it as a Contribution.

The first cut uses **structured prompting** rather than a real
tool-call API: the model is asked to emit either ``TOOL: {...}`` or
``ANSWER: ...`` lines and the provider parses by line prefix. This
keeps the SideModel Protocol surface tiny and lets the loop logic
be tested with scripted stubs.

Covers familiar_connect.context.providers.content_search, which
doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.content_search import (
    CONTENT_SEARCH_PRIORITY,
    ContentSearchProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
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


class _ScriptedSideModel:
    """SideModel stub that returns a queue of canned responses.

    Each call to complete() pops the next canned string from the
    front of the queue. Captured prompts are kept for assertion.
    """

    id = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,  # noqa: ARG002
    ) -> str:
        self.calls.append(prompt)
        if not self._responses:
            msg = (
                "scripted side-model ran out of responses; "
                f"prompts so far: {len(self.calls)}"
            )
            raise RuntimeError(msg)
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id_and_deadline(self, store: MemoryStore) -> None:
        provider = ContentSearchProvider(store=store, side_model=_ScriptedSideModel([]))
        assert provider.id == "content_search"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, store: MemoryStore) -> None:
        provider = ContentSearchProvider(store=store, side_model=_ScriptedSideModel([]))
        assert isinstance(provider, ContextProvider)


# ---------------------------------------------------------------------------
# Trivial answer paths
# ---------------------------------------------------------------------------


class TestImmediateAnswer:
    @pytest.mark.asyncio
    async def test_immediate_answer_yields_contribution(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedSideModel(["ANSWER: Aria knows Alice from last Tuesday."])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        c = contributions[0]
        assert isinstance(c, Contribution)
        assert c.layer is Layer.content
        assert c.priority == CONTENT_SEARCH_PRIORITY
        assert "Aria knows Alice" in c.text
        # Single side-model call total — straight to ANSWER, no tool loop.
        assert len(side.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_answer_yields_no_contribution(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedSideModel(["ANSWER:"])
        provider = ContentSearchProvider(store=store, side_model=side)
        contributions = await provider.contribute(_request())
        assert contributions == []

    @pytest.mark.asyncio
    async def test_whitespace_only_answer_yields_no_contribution(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedSideModel(["ANSWER:    \n   "])
        provider = ContentSearchProvider(store=store, side_model=side)
        assert await provider.contribute(_request()) == []


# ---------------------------------------------------------------------------
# Single tool call → answer
# ---------------------------------------------------------------------------


class TestSingleToolCall:
    @pytest.mark.asyncio
    async def test_grep_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "She likes ska music and old citadels.")
        side = _ScriptedSideModel([
            'TOOL: {"tool": "grep", "args": {"pattern": "ska"}}',
            "ANSWER: Alice likes ska music.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "Alice likes ska music" in contributions[0].text
        assert len(side.calls) == 2
        # The second prompt fed back the grep result, so it should mention
        # the file the hit came from.
        assert "people/alice.md" in side.calls[1]

    @pytest.mark.asyncio
    async def test_read_file_then_answer(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "Aria's favourite tea is jasmine.")
        side = _ScriptedSideModel([
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            "ANSWER: Aria likes jasmine tea.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "jasmine tea" in contributions[0].text
        # Second prompt should contain the file content from the tool result.
        assert "jasmine" in side.calls[1]

    @pytest.mark.asyncio
    async def test_list_dir_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        store.write_file("people/bob.md", "y")
        side = _ScriptedSideModel([
            'TOOL: {"tool": "list_dir", "args": {"path": "people"}}',
            "ANSWER: Aria has files on Alice and Bob.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        # The list_dir result fed back to the model includes both names.
        assert "alice.md" in side.calls[1]
        assert "bob.md" in side.calls[1]

    @pytest.mark.asyncio
    async def test_glob_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        store.write_file("topics/ska.md", "y")
        side = _ScriptedSideModel([
            'TOOL: {"tool": "glob", "args": {"pattern": "**/*.md"}}',
            "ANSWER: Aria has people and topics directories.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)
        contributions = await provider.contribute(_request())
        assert len(contributions) == 1
        assert "people/alice.md" in side.calls[1]
        assert "topics/ska.md" in side.calls[1]


# ---------------------------------------------------------------------------
# Multi-step loops
# ---------------------------------------------------------------------------


class TestMultiStepLoop:
    @pytest.mark.asyncio
    async def test_list_then_read_then_answer(self, store: MemoryStore) -> None:
        store.write_file(
            "people/alice.md",
            "She loves ska. Last Tuesday we argued about Madness vs the Specials.",
        )
        side = _ScriptedSideModel([
            'TOOL: {"tool": "list_dir", "args": {"path": "people"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "people/alice.md"}}',
            "ANSWER: Alice argued about ska bands last Tuesday.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "argued about ska bands" in contributions[0].text
        assert len(side.calls) == 3


# ---------------------------------------------------------------------------
# Iteration cap and graceful failure modes
# ---------------------------------------------------------------------------


class TestIterationCap:
    @pytest.mark.asyncio
    async def test_max_iterations_returns_no_contribution(
        self, store: MemoryStore
    ) -> None:
        """A model that never emits ANSWER is cut off cleanly."""
        store.write_file("notes.md", "x")
        forever = ['TOOL: {"tool": "list_dir", "args": {}}'] * 10
        side = _ScriptedSideModel(forever)
        provider = ContentSearchProvider(store=store, side_model=side, max_iterations=3)

        contributions = await provider.contribute(_request())

        assert contributions == []
        # Exactly max_iterations side-model calls.
        assert len(side.calls) == 3


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_path_traversal_attempt_returns_error_result(
        self, store: MemoryStore
    ) -> None:
        store.write_file("notes.md", "kept")
        side = _ScriptedSideModel([
            'TOOL: {"tool": "read_file", "args": {"path": "../escape.md"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            "ANSWER: Recovered after the bad path.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "Recovered" in contributions[0].text
        # The second prompt should reflect the error from the first call.
        assert "error" in side.calls[1].lower()
        # The third prompt should contain the recovered file content.
        assert "kept" in side.calls[2]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self, store: MemoryStore) -> None:
        side = _ScriptedSideModel([
            'TOOL: {"tool": "delete_universe", "args": {}}',
            "ANSWER: I gave up on that one.",
        ])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "error" in side.calls[1].lower()
        assert "delete_universe" in side.calls[1]

    @pytest.mark.asyncio
    async def test_malformed_tool_call_treated_as_final_answer(
        self, store: MemoryStore
    ) -> None:
        """A garbage response is treated as the final answer text.

        Rather than failing hard — the model said something, we
        surface it.
        """
        side = _ScriptedSideModel(["I'm not following the format. Just text."])
        provider = ContentSearchProvider(store=store, side_model=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "Just text" in contributions[0].text


# ---------------------------------------------------------------------------
# Initial prompt content
# ---------------------------------------------------------------------------


class TestInitialPromptContent:
    @pytest.mark.asyncio
    async def test_user_utterance_is_in_initial_prompt(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedSideModel(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, side_model=side)
        await provider.contribute(_request(utterance="tell me about ska"))
        assert "tell me about ska" in side.calls[0]

    @pytest.mark.asyncio
    async def test_speaker_name_is_in_initial_prompt(self, store: MemoryStore) -> None:
        side = _ScriptedSideModel(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, side_model=side)
        await provider.contribute(_request())
        # Speaker is "Alice" by default in _request().
        assert "Alice" in side.calls[0]

    @pytest.mark.asyncio
    async def test_tool_descriptions_appear_in_initial_prompt(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedSideModel(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, side_model=side)
        await provider.contribute(_request())
        prompt = side.calls[0]
        for tool in ("list_dir", "glob", "grep", "read_file"):
            assert tool in prompt, f"{tool} missing from initial prompt"
