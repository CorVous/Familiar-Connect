"""Red-first tests for the SideModel Protocol.

The cheap-side-model surface used by context providers and processors
for focused sub-tasks (summarisation, lorebook management, stepped
thinking, recast cleanup). Deliberately tiny — one async method
``complete(prompt, *, max_tokens) -> str`` plus an ``id`` attribute —
so providers can be tested with scripted stubs and the production
implementation can swap without touching them.

Covers familiar_connect.context.side_model, which doesn't exist yet.
"""

from __future__ import annotations

import pytest

from familiar_connect.context.side_model import SideModel


class _GoodStub:
    id = "good"

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        return f"got {len(prompt)} chars, max {max_tokens}"


class _MissingId:
    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,  # noqa: ARG002
    ) -> str:
        return prompt


class _MissingComplete:
    id = "no_complete"


class TestSideModelProtocol:
    def test_runtime_checkable_accepts_conforming_stub(self) -> None:
        assert isinstance(_GoodStub(), SideModel)

    def test_runtime_checkable_rejects_missing_complete(self) -> None:
        assert not isinstance(_MissingComplete(), SideModel)

    @pytest.mark.asyncio
    async def test_complete_signature_works_through_protocol(self) -> None:
        """A SideModel-typed reference can be awaited normally."""
        model: SideModel = _GoodStub()
        result = await model.complete("hello world", max_tokens=64)
        assert "11 chars" in result
        assert "max 64" in result
