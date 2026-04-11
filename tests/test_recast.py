"""Red-first tests for RecastPostProcessor.

Step 10 of docs/architecture/context-pipeline.md. Post-processor that
takes the main LLM's reply, runs a cheap SideModel with a focused
cleanup prompt, and returns the rewritten text. Inspired by
SillyTavern's ``recast-post-processing``.

The pipeline doesn't yet invoke post-processors (that lands in step
7's wiring), so these tests exercise the processor in isolation
against the protocol surface defined in
``familiar_connect.context.protocols``.

Covers familiar_connect.context.processors.recast, which doesn't
exist yet.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from familiar_connect.context.processors.recast import RecastPostProcessor
from familiar_connect.context.protocols import PostProcessor
from familiar_connect.context.types import ContextRequest, Modality

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _request(**overrides: object) -> ContextRequest:
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "speaker": "Alice",
        "utterance": "hello",
        "modality": Modality.text,
        "budget_tokens": 2048,
        "deadline_s": 10.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)  # type: ignore[arg-type]


class _ScriptedSideModel:
    id = "scripted"

    def __init__(
        self,
        response: str = "Cleaner reply.",
        *,
        delay_s: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._delay_s = delay_s
        self._exc = exc
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,  # noqa: ARG002
    ) -> str:
        self.calls.append(prompt)
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._exc is not None:
            raise self._exc
        return self._response


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id(self) -> None:
        proc = RecastPostProcessor(side_model=_ScriptedSideModel())
        assert proc.id == "recast"

    def test_conforms_to_post_processor_protocol(self) -> None:
        proc = RecastPostProcessor(side_model=_ScriptedSideModel())
        assert isinstance(proc, PostProcessor)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_cleaned_text(self) -> None:
        side = _ScriptedSideModel(response="Tighter, voice-friendly version.")
        proc = RecastPostProcessor(side_model=side)
        result = await proc.process("Original wordy reply.", _request())
        assert result == "Tighter, voice-friendly version."

    @pytest.mark.asyncio
    async def test_strips_response_whitespace(self) -> None:
        side = _ScriptedSideModel(response="   trimmed.   \n")
        proc = RecastPostProcessor(side_model=side)
        result = await proc.process("Original.", _request())
        assert result == "trimmed."

    @pytest.mark.asyncio
    async def test_voice_modality_uses_voice_oriented_prompt(self) -> None:
        side = _ScriptedSideModel()
        proc = RecastPostProcessor(side_model=side)
        await proc.process("hello", _request(modality=Modality.voice))
        # Voice prompt mentions speech / TTS / out-loud somewhere.
        prompt = side.calls[0].lower()
        voice_tokens = ("speech", "spoken", "voice", "out loud")
        assert any(t in prompt for t in voice_tokens)

    @pytest.mark.asyncio
    async def test_text_modality_uses_general_cleanup_prompt(self) -> None:
        side = _ScriptedSideModel()
        proc = RecastPostProcessor(side_model=side)
        await proc.process("hello", _request(modality=Modality.text))
        prompt = side.calls[0]
        # Original reply is in the prompt no matter the modality.
        assert "hello" in prompt


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_side_model_exception_returns_original(self) -> None:
        side = _ScriptedSideModel(exc=RuntimeError("kaboom"))
        proc = RecastPostProcessor(side_model=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_side_model_timeout_returns_original(self) -> None:
        side = _ScriptedSideModel(delay_s=2.0)
        proc = RecastPostProcessor(side_model=side, processor_timeout_s=0.05)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_empty_response_returns_original(self) -> None:
        side = _ScriptedSideModel(response="")
        proc = RecastPostProcessor(side_model=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_original(self) -> None:
        side = _ScriptedSideModel(response="   \n   ")
        proc = RecastPostProcessor(side_model=side)
        result = await proc.process("Original reply.", _request())
        assert result == "Original reply."
