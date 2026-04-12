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
from typing import Any

import pytest

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
)
from familiar_connect.llm import LLMClient, Message

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "speaker": "Alice",
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
            speaker=original.speaker,
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
