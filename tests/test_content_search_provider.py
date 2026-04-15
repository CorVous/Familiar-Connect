"""Red-first tests for the ContentSearchProvider.

Step 8 of docs/architecture/context-pipeline.md. The interesting one:
a cheap tool-using model with grep / glob / read_file / list_dir
tools scoped to a single familiar's MemoryStore. Each contribute()
call runs a small loop — the model decides which tools to call, the
provider executes them against the store, the model eventually
returns an ``ANSWER:`` and the provider wraps it as a Contribution.

The first cut uses **structured prompting** rather than a real
tool-call API: the model is asked to emit either ``TOOL: {...}`` or
``ANSWER: ...`` lines and the provider parses by line prefix. This
keeps the LLM interface surface tiny and lets the loop logic be
tested with scripted stubs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.content_search import (
    CONTENT_SEARCH_PRIORITY,
    DEFAULT_MAX_ITERATIONS,
    FORCED_ANSWER_MARKER,
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
    """LLMClient stub that returns a queue of canned responses.

    Each call to :meth:`chat` pops the next canned string from the
    front of the queue and wraps it in an assistant :class:`Message`.
    Captured message lists are kept for assertion. Inherits from
    :class:`LLMClient` so it is structurally and nominally a drop-in
    replacement at call sites typed as ``LLMClient``.
    """

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

    async def close(self) -> None:  # pragma: no cover — no resources
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
# Trivial answer paths
# ---------------------------------------------------------------------------


class TestImmediateAnswer:
    @pytest.mark.asyncio
    async def test_immediate_answer_yields_contribution(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedLLMClient(["ANSWER: Aria knows Alice from last Tuesday."])
        provider = ContentSearchProvider(store=store, llm_client=side)

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
        side = _ScriptedLLMClient(["ANSWER:"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contributions = await provider.contribute(_request())
        assert contributions == []

    @pytest.mark.asyncio
    async def test_whitespace_only_answer_yields_no_contribution(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedLLMClient(["ANSWER:    \n   "])
        provider = ContentSearchProvider(store=store, llm_client=side)
        assert await provider.contribute(_request()) == []


# ---------------------------------------------------------------------------
# Single tool call → answer
# ---------------------------------------------------------------------------


class TestSingleToolCall:
    @pytest.mark.asyncio
    async def test_grep_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "She likes ska music and old citadels.")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "grep", "args": {"pattern": "ska"}}',
            "ANSWER: Alice likes ska music.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        # Two contributions: deterministic people-lookup (Alice is speaker)
        # + agent-loop ANSWER.
        assert len(contributions) == 2
        agent = next(c for c in contributions if c.source == "content_search")
        assert "Alice likes ska music" in agent.text
        assert len(side.calls) == 2
        # The second prompt fed back the grep result, so it should mention
        # the file the hit came from.
        assert "people/alice.md" in side.prompt_at(1)

    @pytest.mark.asyncio
    async def test_read_file_then_answer(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "Aria's favourite tea is jasmine.")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            "ANSWER: Aria likes jasmine tea.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "jasmine tea" in contributions[0].text
        # Second prompt should contain the file content from the tool result.
        assert "jasmine" in side.prompt_at(1)

    @pytest.mark.asyncio
    async def test_list_dir_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        store.write_file("people/bob.md", "y")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "list_dir", "args": {"path": "people"}}',
            "ANSWER: Aria has files on Alice and Bob.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        # Deterministic tier picks up alice.md (speaker) — agent loop
        # emits its ANSWER alongside.
        agent = next(c for c in contributions if c.source == "content_search")
        assert "Alice and Bob" in agent.text
        # The list_dir result fed back to the model includes both names.
        assert "alice.md" in side.prompt_at(1)
        assert "bob.md" in side.prompt_at(1)

    @pytest.mark.asyncio
    async def test_glob_then_answer(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        store.write_file("topics/ska.md", "y")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "glob", "args": {"pattern": "**/*.md"}}',
            "ANSWER: Aria has people and topics directories.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)
        contributions = await provider.contribute(_request())
        # Deterministic (alice.md) + agent-loop ANSWER.
        agent = next(c for c in contributions if c.source == "content_search")
        assert "people and topics" in agent.text
        assert "people/alice.md" in side.prompt_at(1)
        assert "topics/ska.md" in side.prompt_at(1)


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
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "list_dir", "args": {"path": "people"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "people/alice.md"}}',
            "ANSWER: Alice argued about ska bands last Tuesday.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        # Deterministic (alice.md) + agent-loop ANSWER.
        agent = next(c for c in contributions if c.source == "content_search")
        assert "argued about ska bands" in agent.text
        assert len(side.calls) == 3


# ---------------------------------------------------------------------------
# Iteration cap and graceful failure modes
# ---------------------------------------------------------------------------


class TestIterationCap:
    @pytest.mark.asyncio
    async def test_max_iterations_returns_no_contribution(
        self, store: MemoryStore
    ) -> None:
        """A model that never emits ANSWER is cut off cleanly.

        Even with forced-answer on the final iteration, a model that
        stubbornly keeps emitting TOOL returns the empty contribution
        and the warning log fires.
        """
        store.write_file("notes.md", "x")
        forever = ['TOOL: {"tool": "list_dir", "args": {}}'] * 10
        side = _ScriptedLLMClient(forever)
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=3)

        contributions = await provider.contribute(_request())

        assert contributions == []
        # Exactly max_iterations side-model calls.
        assert len(side.calls) == 3


class TestDefaultMaxIterations:
    def test_default_lowered_to_three(self) -> None:
        """Default iteration cap lowered to 3.

        Three flailing tool calls don't blow the budget before the
        forced-answer iteration fires.
        """
        assert DEFAULT_MAX_ITERATIONS == 3


class TestForcedAnswerOnFinalIteration:
    @pytest.mark.asyncio
    async def test_final_iteration_prompt_forbids_more_tools(
        self, store: MemoryStore
    ) -> None:
        """On the last iteration the prompt is swapped to 'emit ANSWER now'."""
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "list_dir", "args": {}}',
            'TOOL: {"tool": "glob", "args": {"pattern": "*"}}',
            "ANSWER: best guess from what I saw.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=3)
        contribs = await provider.contribute(_request())
        # Final prompt (index 2 since 3 calls) uses the forced-answer marker.
        final_prompt = side.prompt_at(2)
        assert FORCED_ANSWER_MARKER in final_prompt, (
            "final iteration should use the forced-answer prompt"
        )
        # First and second prompts use the normal prompt (not forced).
        assert FORCED_ANSWER_MARKER not in side.prompt_at(0)
        assert FORCED_ANSWER_MARKER not in side.prompt_at(1)
        assert len(contribs) == 1
        assert "best guess" in contribs[0].text

    @pytest.mark.asyncio
    async def test_forced_answer_surfaces_best_effort_when_model_cooperates(
        self, store: MemoryStore
    ) -> None:
        """With forced-answer, the model's last reply becomes the contribution."""
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "list_dir", "args": {}}',
            'TOOL: {"tool": "glob", "args": {"pattern": "**/*.md"}}',
            "ANSWER: Alice mentioned ska, couldn't find anything more specific.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=3)
        contribs = await provider.contribute(_request())
        assert len(contribs) == 1
        assert "ska" in contribs[0].text


class TestRedundantToolCallBailOut:
    @pytest.mark.asyncio
    async def test_repeated_tool_args_jumps_to_forced_answer(
        self, store: MemoryStore
    ) -> None:
        """Repeating a (tool, args) pair bails out early to the forced prompt.

        Saves an LLM call vs running the iteration cap down the full
        length.
        """
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "grep", "args": {"pattern": "foo"}}',
            # identical call — redundant
            'TOOL: {"tool": "grep", "args": {"pattern": "foo"}}',
            "ANSWER: found nothing interesting.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=5)
        contribs = await provider.contribute(_request())
        assert len(contribs) == 1
        # 3 calls total: normal, redundant detection, forced answer.
        assert len(side.calls) == 3
        # Third call uses forced-answer prompt.
        assert FORCED_ANSWER_MARKER in side.prompt_at(2)
        # Second call was still the normal prompt (redundancy is
        # detected on parse, not on prompt build).
        assert FORCED_ANSWER_MARKER not in side.prompt_at(1)

    @pytest.mark.asyncio
    async def test_same_tool_different_args_not_redundant(
        self, store: MemoryStore
    ) -> None:
        """Same tool name but different args is a legitimate different call."""
        store.write_file("a.md", "apple")
        store.write_file("b.md", "banana")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "read_file", "args": {"path": "a.md"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "b.md"}}',
            "ANSWER: saw both fruits.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=5)
        contribs = await provider.contribute(_request())
        # All three calls ran; no early bail-out.
        assert len(contribs) == 1
        assert len(side.calls) == 3
        # Third prompt is the normal one — max_iterations=5 means iter 2
        # is not the final iteration and no redundancy was detected.
        assert FORCED_ANSWER_MARKER not in side.prompt_at(2)
        # Both file contents appear in the third prompt (proving both tools ran).
        third = side.prompt_at(2)
        assert "apple" in third
        assert "banana" in third

    @pytest.mark.asyncio
    async def test_redundancy_does_not_re_execute_tool(
        self, store: MemoryStore
    ) -> None:
        """On redundancy detection, the tool is NOT re-executed.

        If the model keeps asking for the same grep, we shouldn't keep
        running it — the result would be identical.
        """
        store.write_file("notes.md", "content")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            "ANSWER: fine.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side, max_iterations=5)
        await provider.contribute(_request())
        # The second prompt should contain the file content once (from iter 0);
        # the third prompt (forced) should not have duplicated tool results.
        second = side.prompt_at(1)
        third = side.prompt_at(2)
        # 'content' should appear exactly once in the scratchpad of the
        # second prompt, and the third prompt's scratchpad should NOT
        # have grown (no extra TOOL_RESULT block from the redundant call).
        assert second.count("TOOL_RESULT") == 1
        assert third.count("TOOL_RESULT") == 1


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_path_traversal_attempt_returns_error_result(
        self, store: MemoryStore
    ) -> None:
        store.write_file("notes.md", "kept")
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "read_file", "args": {"path": "../escape.md"}}',
            'TOOL: {"tool": "read_file", "args": {"path": "notes.md"}}',
            "ANSWER: Recovered after the bad path.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "Recovered" in contributions[0].text
        # The second prompt should reflect the error from the first call.
        assert "error" in side.prompt_at(1).lower()
        # The third prompt should contain the recovered file content.
        assert "kept" in side.prompt_at(2)

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self, store: MemoryStore) -> None:
        side = _ScriptedLLMClient([
            'TOOL: {"tool": "delete_universe", "args": {}}',
            "ANSWER: I gave up on that one.",
        ])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "error" in side.prompt_at(1).lower()
        assert "delete_universe" in side.prompt_at(1)

    @pytest.mark.asyncio
    async def test_malformed_tool_call_treated_as_final_answer(
        self, store: MemoryStore
    ) -> None:
        """A garbage response is treated as the final answer text.

        Rather than failing hard — the model said something, we
        surface it.
        """
        side = _ScriptedLLMClient(["I'm not following the format. Just text."])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contributions = await provider.contribute(_request())

        assert len(contributions) == 1
        assert "Just text" in contributions[0].text


# ---------------------------------------------------------------------------
# Initial prompt content
# ---------------------------------------------------------------------------


class TestPeopleLookupInvariant:
    """Deterministic-tier guarantees the forgetting-bug fix.

    Speaker and mentioned-name files are surfaced regardless of
    agent-loop behaviour — empty ANSWER, exception, etc.
    """

    @pytest.mark.asyncio
    async def test_speaker_file_included_when_agent_returns_empty(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/alice.md", "Alice is a ska fan from York.")
        # Agent loop emits empty ANSWER — would drop any contribution on
        # its own. Deterministic tier must still surface Alice's file.
        side = _ScriptedLLMClient(["ANSWER:"])
        provider = ContentSearchProvider(store=store, llm_client=side)

        contribs = await provider.contribute(_request(utterance="hi"))

        assert len(contribs) == 1
        assert contribs[0].source == PEOPLE_LOOKUP_SOURCE
        assert "ska fan from York" in contribs[0].text

    @pytest.mark.asyncio
    async def test_speaker_file_included_when_agent_raises(
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


class TestInitialPromptContent:
    @pytest.mark.asyncio
    async def test_user_utterance_is_in_initial_prompt(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedLLMClient(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        await provider.contribute(_request(utterance="tell me about ska"))
        assert "tell me about ska" in side.prompt_at(0)

    @pytest.mark.asyncio
    async def test_speaker_name_is_in_initial_prompt(self, store: MemoryStore) -> None:
        side = _ScriptedLLMClient(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        await provider.contribute(_request())
        # Speaker is "Alice" by default in _request().
        assert "Alice" in side.prompt_at(0)

    @pytest.mark.asyncio
    async def test_tool_descriptions_appear_in_initial_prompt(
        self, store: MemoryStore
    ) -> None:
        side = _ScriptedLLMClient(["ANSWER: noted"])
        provider = ContentSearchProvider(store=store, llm_client=side)
        await provider.contribute(_request())
        prompt = side.prompt_at(0)
        for tool in ("list_dir", "glob", "grep", "read_file"):
            assert tool in prompt, f"{tool} missing from initial prompt"
