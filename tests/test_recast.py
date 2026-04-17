"""Red-first tests for RecastPostProcessor.

Step 10 of docs/architecture/context-pipeline.md. Post-processor that
takes the main LLM's reply, runs the ``post_process_style`` slot's
:class:`LLMClient` with a focused cleanup prompt, and returns the
rewritten text. Inspired by SillyTavern's ``recast-post-processing``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from familiar_connect.context.processors.recast import RecastPostProcessor
from familiar_connect.context.protocols import PostProcessor
from familiar_connect.context.types import ContextRequest, Modality
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "author": _ALICE,
        "utterance": "hello",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 10.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


class _ScriptedLLMClient(LLMClient):
    """LLMClient stub that returns canned responses with optional delay/exc."""

    def __init__(
        self,
        response: str = "Cleaner reply.",
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
        proc = RecastPostProcessor(llm_client=_ScriptedLLMClient())
        assert proc.id == "recast"

    def test_conforms_to_post_processor_protocol(self) -> None:
        proc = RecastPostProcessor(llm_client=_ScriptedLLMClient())
        assert isinstance(proc, PostProcessor)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_cleaned_text(self) -> None:
        side = _ScriptedLLMClient(response="Tighter, voice-friendly version.")
        proc = RecastPostProcessor(llm_client=side)
        result = await proc.process("Original wordy reply.", _request())
        assert result == "Tighter, voice-friendly version."

    @pytest.mark.asyncio
    async def test_strips_response_whitespace(self) -> None:
        side = _ScriptedLLMClient(response="   trimmed.   \n")
        proc = RecastPostProcessor(llm_client=side)
        result = await proc.process("Original.", _request())
        assert result == "trimmed."

    @pytest.mark.asyncio
    async def test_voice_modality_uses_voice_oriented_prompt(self) -> None:
        side = _ScriptedLLMClient()
        proc = RecastPostProcessor(llm_client=side)
        await proc.process("hello", _request(modality=Modality.voice))
        # Voice prompt mentions speech / TTS / out-loud somewhere.
        prompt = side.calls[0][0].content.lower()
        voice_tokens = ("speech", "spoken", "voice", "out loud")
        assert any(t in prompt for t in voice_tokens)

    @pytest.mark.asyncio
    async def test_text_modality_uses_general_cleanup_prompt(self) -> None:
        side = _ScriptedLLMClient()
        proc = RecastPostProcessor(llm_client=side)
        await proc.process("hello", _request(modality=Modality.text))
        prompt = side.calls[0][0].content
        # Original reply is in the prompt no matter the modality.
        assert "hello" in prompt


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_side_model_exception_returns_original(self) -> None:
        side = _ScriptedLLMClient(exc=RuntimeError("kaboom"))
        proc = RecastPostProcessor(llm_client=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_side_model_timeout_returns_original(self) -> None:
        side = _ScriptedLLMClient(delay_s=2.0)
        proc = RecastPostProcessor(llm_client=side, processor_timeout_s=0.05)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_empty_response_returns_original(self) -> None:
        side = _ScriptedLLMClient(response="")
        proc = RecastPostProcessor(llm_client=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_original(self) -> None:
        side = _ScriptedLLMClient(response="   \n   ")
        proc = RecastPostProcessor(llm_client=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."
