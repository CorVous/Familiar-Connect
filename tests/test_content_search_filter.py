"""Red-first tests for the single-shot filter tier.

Tier 3 of ContentSearchProvider. Replaces the tool-using agent loop:
one LLM call with embedding-retrieved chunks in hand, optionally one
follow-up call after a single ``GREP:`` escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.providers.content_search.filter import (
    FILTER_MAX_ITERATIONS,
    FILTER_PRIORITY,
    FILTER_SOURCE,
    run,
)
from familiar_connect.context.providers.content_search.retrieval import (
    RetrievedChunk,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")


def _request(utterance: str = "tell me about Alice") -> ContextRequest:
    return ContextRequest(
        familiar_id="aria",
        channel_id=100,
        guild_id=1,
        author=_ALICE,
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=2048,
        deadline_s=10.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


class _ScriptedLLMClient(LLMClient):
    """Canned-response LLM stub. Mirrors the one in test_content_search_provider."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="scripted-test-key", model="scripted/test-model")
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._responses:
            msg = f"scripted LLM ran out of responses after {len(self.calls)} calls"
            raise RuntimeError(msg)
        return Message(role="assistant", content=self._responses.pop(0))

    async def close(self) -> None:  # pragma: no cover
        return

    def prompt_at(self, i: int) -> str:
        return self.calls[i][0].content


def _chunk(
    rel_path: str,
    heading_path: str,
    text: str,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        rel_path=rel_path, heading_path=heading_path, text=text, score=score
    )


# ---------------------------------------------------------------------------
# Straight-through ANSWER paths
# ---------------------------------------------------------------------------


class TestImmediateAnswer:
    @pytest.mark.asyncio
    async def test_answer_yields_contribution(self, store: MemoryStore) -> None:
        client = _ScriptedLLMClient(["ANSWER: Alice likes ska music."])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[_chunk("people/alice.md", "Alice", "Alice loves ska")],
            deterministic=[],
        )
        assert len(contribs) == 1
        c = contribs[0]
        assert c.layer is Layer.content
        assert c.priority == FILTER_PRIORITY
        assert c.source == FILTER_SOURCE
        assert "Alice likes ska" in c.text
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_answer_yields_no_contribution(
        self, store: MemoryStore
    ) -> None:
        client = _ScriptedLLMClient(["ANSWER:"])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert contribs == []
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_whitespace_answer_yields_no_contribution(
        self, store: MemoryStore
    ) -> None:
        client = _ScriptedLLMClient(["ANSWER:    \n   "])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert contribs == []


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


class TestPromptContent:
    @pytest.mark.asyncio
    async def test_utterance_in_prompt(self, store: MemoryStore) -> None:
        client = _ScriptedLLMClient(["ANSWER:"])
        await run(
            llm_client=client,
            store=store,
            request=_request(utterance="what about jazz?"),
            retrieved=[],
            deterministic=[],
        )
        assert "what about jazz?" in client.prompt_at(0)

    @pytest.mark.asyncio
    async def test_retrieved_chunks_in_prompt(self, store: MemoryStore) -> None:
        client = _ScriptedLLMClient(["ANSWER:"])
        await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[
                _chunk("topics/ska.md", "Ska", "Two-tone ska emerged in 1979.", 0.9),
                _chunk("people/alice.md", "Alice", "Alice loves ska", 0.7),
            ],
            deterministic=[],
        )
        prompt = client.prompt_at(0)
        assert "topics/ska.md" in prompt
        assert "1979" in prompt
        assert "people/alice.md" in prompt

    @pytest.mark.asyncio
    async def test_deterministic_paths_hidden_from_prompt_snippets(
        self, store: MemoryStore
    ) -> None:
        """Filter is told about deterministic tier so it doesn't re-emit.

        The deterministic tier's loaded rel_paths are passed as an
        "already included" hint — the filter shouldn't duplicate them.
        """
        det = [
            Contribution(
                layer=Layer.content,
                priority=85,
                text="Alice's verbatim notes file",
                estimated_tokens=estimate_tokens("Alice's verbatim notes file"),
                source="content_search.people",
            )
        ]
        client = _ScriptedLLMClient(["ANSWER:"])
        await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=det,
        )
        prompt = client.prompt_at(0)
        # deterministic contribution mentioned so the filter knows
        # not to re-emit Alice's file text
        assert "verbatim notes" in prompt or "already" in prompt.lower()


# ---------------------------------------------------------------------------
# ESCALATE → grep → forced second call
# ---------------------------------------------------------------------------


class TestEscalation:
    @pytest.mark.asyncio
    async def test_escalate_runs_one_grep_and_second_call(
        self, store: MemoryStore
    ) -> None:
        store.write_file("notes.md", "Alice mentioned she plays the trombone.")
        client = _ScriptedLLMClient([
            "ESCALATE: need to check instrument; GREP: trombone",
            "ANSWER: Alice plays trombone.",
        ])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert len(contribs) == 1
        assert "trombone" in contribs[0].text.lower()
        assert len(client.calls) == 2
        # grep result present in second prompt
        second = client.prompt_at(1)
        assert "trombone" in second
        assert "notes.md" in second

    @pytest.mark.asyncio
    async def test_escalate_with_no_grep_match_still_answers(
        self, store: MemoryStore
    ) -> None:
        """Grep returning no hits still produces a forced-answer turn."""
        client = _ScriptedLLMClient([
            "ESCALATE: dunno; GREP: nonexistentpattern",
            "ANSWER: I couldn't find it.",
        ])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert len(contribs) == 1
        assert "couldn't find" in contribs[0].text
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_escalate_cap_is_two_calls(self, store: MemoryStore) -> None:
        """Second call is forced — even if it tries another ESCALATE it's ignored."""
        client = _ScriptedLLMClient([
            "ESCALATE: first; GREP: foo",
            "ESCALATE: second; GREP: bar",  # ignored — forced turn
        ])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        # second ESCALATE is not treated as answer and no third call is made
        assert contribs == []
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_escalate_without_grep_pattern_still_second_call(
        self, store: MemoryStore
    ) -> None:
        """Malformed ESCALATE (no GREP:) → still runs a second forced call."""
        client = _ScriptedLLMClient([
            "ESCALATE: thinking more",
            "ANSWER: best I can do.",
        ])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert len(contribs) == 1
        assert "best I can do" in contribs[0].text
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Malformed / graceful paths
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    @pytest.mark.asyncio
    async def test_malformed_first_response_treated_as_answer(
        self, store: MemoryStore
    ) -> None:
        client = _ScriptedLLMClient(["I forgot the format. Just some text."])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert len(contribs) == 1
        assert "Just some text" in contribs[0].text

    @pytest.mark.asyncio
    async def test_malformed_second_response_surfaced(self, store: MemoryStore) -> None:
        """After ESCALATE, a malformed second response is still surfaced."""
        client = _ScriptedLLMClient([
            "ESCALATE: lookup; GREP: foo",
            "prose without prefix",
        ])
        contribs = await run(
            llm_client=client,
            store=store,
            request=_request(),
            retrieved=[],
            deterministic=[],
        )
        assert len(contribs) == 1
        assert "prose without prefix" in contribs[0].text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_iterations_is_two(self) -> None:
        """Plan pins the filter to at most 2 LLM calls."""
        assert FILTER_MAX_ITERATIONS == 2

    def test_priority_and_source_exported(self) -> None:
        assert FILTER_PRIORITY == 70
        assert FILTER_SOURCE == "content_search.rag"
