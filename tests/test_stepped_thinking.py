"""Red-first tests for SteppedThinkingPreProcessor.

Step 10 of docs/architecture/context-pipeline.md. Pre-processor that
calls the ``reasoning_context`` slot's :class:`LLMClient` with a
focused "think step by step about what the user is really asking"
prompt and stashes the result on the request as a Contribution at
``Layer.depth_inject`` so the budgeter picks it up alongside
provider contributions. Inspired by SillyTavern's
``st-stepped-thinking``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from familiar_connect.context.processors.stepped_thinking import (
    STEPPED_THINKING_PRIORITY,
    SteppedThinkingPreProcessor,
)
from familiar_connect.context.protocols import PreProcessor
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
    PendingTurn,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")
_BOB = Author(platform="discord", user_id="2", username="bob", display_name="Bob")

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "author": _ALICE,
        "utterance": "what do you think about ska music?",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 10.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


class _ScriptedLLMClient(LLMClient):
    """LLMClient stub that returns canned responses with optional delay/exc.

    Inherits from :class:`LLMClient` so it is a drop-in replacement
    at call sites typed as ``LLMClient`` — the processor touches
    ``self._llm_client.chat(messages)`` and reads ``.content``.
    """

    def __init__(
        self,
        response: str = "I should think about ska bands and Alice's preferences.",
        *,
        delay_s: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        super().__init__(api_key="scripted-test-key", model="scripted/test-model")
        self._response = response
        self._delay_s = delay_s
        self._exc = exc
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._exc is not None:
            raise self._exc
        return Message(role="assistant", content=self._response)

    async def close(self) -> None:  # pragma: no cover — no resources
        return


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id(self) -> None:
        proc = SteppedThinkingPreProcessor(llm_client=_ScriptedLLMClient())
        assert proc.id == "stepped_thinking"

    def test_conforms_to_pre_processor_protocol(self) -> None:
        proc = SteppedThinkingPreProcessor(llm_client=_ScriptedLLMClient())
        assert isinstance(proc, PreProcessor)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_request_with_new_contribution(self) -> None:
        side = _ScriptedLLMClient(response="Alice asked about ska. I should mention.")
        proc = SteppedThinkingPreProcessor(llm_client=side)

        result = await proc.process(_request())

        assert len(result.preprocessor_contributions) == 1
        c = result.preprocessor_contributions[0]
        assert isinstance(c, Contribution)
        assert c.layer is Layer.depth_inject
        assert c.priority == STEPPED_THINKING_PRIORITY
        assert "Alice asked about ska" in c.text
        assert c.source == "stepped_thinking"

    @pytest.mark.asyncio
    async def test_preserves_other_request_fields(self) -> None:
        proc = SteppedThinkingPreProcessor(llm_client=_ScriptedLLMClient())
        original = _request()
        result = await proc.process(original)

        assert result.familiar_id == original.familiar_id
        assert result.channel_id == original.channel_id
        assert result.utterance == original.utterance

    @pytest.mark.asyncio
    async def test_appends_to_existing_contributions(self) -> None:
        existing = Contribution(
            layer=Layer.depth_inject,
            priority=50,
            text="from a prior pre-processor",
            estimated_tokens=10,
            source="other",
        )
        original = _request()
        seeded = ContextRequest(
            familiar_id=original.familiar_id,
            channel_id=original.channel_id,
            guild_id=original.guild_id,
            author=original.author,
            utterance=original.utterance,
            modality=original.modality,
            budget_tokens=original.budget_tokens,
            deadline_s=original.deadline_s,
            preprocessor_contributions=(existing,),
        )

        proc = SteppedThinkingPreProcessor(llm_client=_ScriptedLLMClient())
        result = await proc.process(seeded)

        assert len(result.preprocessor_contributions) == 2
        assert result.preprocessor_contributions[0] == existing
        assert result.preprocessor_contributions[1].source == "stepped_thinking"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


class TestPromptContent:
    @pytest.mark.asyncio
    async def test_prompt_includes_utterance(self) -> None:
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        await proc.process(_request(utterance="tell me about Madness"))
        assert "tell me about Madness" in side.calls[0][0].content

    @pytest.mark.asyncio
    async def test_prompt_includes_speaker(self) -> None:
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        await proc.process(_request())
        assert "Alice" in side.calls[0][0].content


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_side_model_exception_returns_request_unchanged(self) -> None:
        side = _ScriptedLLMClient(exc=RuntimeError("kaboom"))
        proc = SteppedThinkingPreProcessor(llm_client=side)
        original = _request()
        result = await proc.process(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_side_model_timeout_returns_request_unchanged(self) -> None:
        side = _ScriptedLLMClient(delay_s=2.0)
        proc = SteppedThinkingPreProcessor(llm_client=side, processor_timeout_s=0.05)
        original = _request()
        result = await proc.process(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_empty_response_returns_request_unchanged(self) -> None:
        side = _ScriptedLLMClient(response="")
        proc = SteppedThinkingPreProcessor(llm_client=side)
        original = _request()
        result = await proc.process(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_request_unchanged(
        self,
    ) -> None:
        side = _ScriptedLLMClient(response="   \n   ")
        proc = SteppedThinkingPreProcessor(llm_client=side)
        original = _request()
        result = await proc.process(original)
        assert result == original


# ---------------------------------------------------------------------------
# Context awareness: history + buffered pending turns
# ---------------------------------------------------------------------------


class TestHistoryInPrompt:
    """When a HistoryStore is wired in, prior turns appear in the prompt.

    Without history the reasoning model only sees one isolated utterance
    and cannot contextualise. See ``docs/architecture/context-pipeline.md``.
    """

    @pytest.mark.asyncio
    async def test_recent_history_turns_appear_in_prompt(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id="aria",
            channel_id=100,
            role="user",
            content="I've been listening to 2-tone ska lately",
            author=_ALICE,
        )
        store.append_turn(
            familiar_id="aria",
            channel_id=100,
            role="assistant",
            content="Madness or The Specials?",
        )

        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side, history_store=store)
        await proc.process(_request(utterance="what do you think about ska music?"))

        prompt = side.calls[0][0].content
        assert "2-tone ska" in prompt
        assert "Madness or The Specials?" in prompt

    @pytest.mark.asyncio
    async def test_history_is_channel_scoped(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id="aria",
            channel_id=999,
            role="user",
            content="unrelated other-channel chatter",
            author=_BOB,
        )

        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side, history_store=store)
        await proc.process(_request(channel_id=100))

        prompt = side.calls[0][0].content
        assert "unrelated other-channel" not in prompt

    @pytest.mark.asyncio
    async def test_history_store_optional_preserves_prior_behaviour(self) -> None:
        """No history_store ⇒ no history section, processor still runs."""
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        result = await proc.process(_request(utterance="hi"))
        assert len(result.preprocessor_contributions) == 1
        assert "hi" in side.calls[0][0].content

    @pytest.mark.asyncio
    async def test_history_respects_window_size(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "history.db")
        for i in range(30):
            store.append_turn(
                familiar_id="aria",
                channel_id=100,
                role="user",
                content=f"message number {i}",
                author=_ALICE,
            )

        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(
            llm_client=side, history_store=store, history_window=5
        )
        await proc.process(_request())

        prompt = side.calls[0][0].content
        # last 5 should appear, earliest 20 should not
        assert "message number 29" in prompt
        assert "message number 25" in prompt
        assert "message number 0" not in prompt
        assert "message number 10" not in prompt


class TestPendingTurnsInPrompt:
    """All buffered pending turns flow into the prompt, not just trigger."""

    @pytest.mark.asyncio
    async def test_all_pending_turns_appear_in_prompt(self) -> None:
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        pending = (
            PendingTurn(author=_ALICE, text="hey so"),
            PendingTurn(author=_ALICE, text="about the ska thing"),
            PendingTurn(author=_ALICE, text="what do you think?"),
        )
        await proc.process(
            _request(utterance="what do you think?", pending_turns=pending),
        )

        prompt = side.calls[0][0].content
        assert "hey so" in prompt
        assert "about the ska thing" in prompt
        assert "what do you think?" in prompt

    @pytest.mark.asyncio
    async def test_pending_turns_override_single_utterance(self) -> None:
        """When pending_turns is non-empty, only buffered content is used.

        Last pending turn should match utterance per the contract, but
        the prompt shouldn't duplicate it.
        """
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        pending = (
            PendingTurn(author=_ALICE, text="first half"),
            PendingTurn(author=_ALICE, text="final bit"),
        )
        await proc.process(_request(utterance="final bit", pending_turns=pending))

        prompt = side.calls[0][0].content
        assert prompt.count("final bit") == 1
        assert "first half" in prompt

    @pytest.mark.asyncio
    async def test_empty_pending_turns_falls_back_to_utterance(self) -> None:
        side = _ScriptedLLMClient()
        proc = SteppedThinkingPreProcessor(llm_client=side)
        await proc.process(_request(utterance="single trigger"))
        assert "single trigger" in side.calls[0][0].content
