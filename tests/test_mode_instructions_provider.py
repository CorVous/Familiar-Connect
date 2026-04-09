"""Red-first tests for ModeInstructionProvider.

Per-mode static instruction loader. Reads
``data/familiars/<id>/modes/<mode_value>.md`` and emits its contents
as a :class:`Layer.author_note` Contribution so a channel in
``text_conversation_rp`` mode can be told "keep replies short and
chat-room-ish" without editing Python.

The provider is constructed per turn inside
:meth:`Familiar.build_pipeline`, so it receives the active
:class:`ChannelMode` as a constructor argument instead of looking it
up off the request.

Covers ``familiar_connect.context.providers.mode_instructions``,
which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from familiar_connect.config import ChannelMode
from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.mode_instructions import (
    MODE_INSTRUCTION_PRIORITY,
    ModeInstructionProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Layer,
    Modality,
)

if TYPE_CHECKING:
    from pathlib import Path


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


# ---------------------------------------------------------------------------
# Construction & protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_id(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert proc.id == "mode_instructions"

    def test_deadline_is_positive(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert proc.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert isinstance(proc, ContextProvider)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_reads_matching_mode_file(self, tmp_path: Path) -> None:
        (tmp_path / "text_conversation_rp.md").write_text(
            "Keep it short. Reply like a chat-room message.",
        )
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )

        contributions = await proc.contribute(_request())

        assert len(contributions) == 1
        c = contributions[0]
        assert c.layer is Layer.author_note
        assert c.priority == MODE_INSTRUCTION_PRIORITY
        assert c.text == "Keep it short. Reply like a chat-room message."
        assert c.source == "mode_instructions:text_conversation_rp"

    @pytest.mark.asyncio
    async def test_different_modes_pick_different_files(self, tmp_path: Path) -> None:
        (tmp_path / "text_conversation_rp.md").write_text("chat-style")
        (tmp_path / "full_rp.md").write_text("novel-style")

        text_proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        rp_proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
        )

        (text_c,) = await text_proc.contribute(_request())
        (rp_c,) = await rp_proc.contribute(_request())

        assert text_c.text == "chat-style"
        assert rp_c.text == "novel-style"

    @pytest.mark.asyncio
    async def test_trims_surrounding_whitespace(self, tmp_path: Path) -> None:
        (tmp_path / "text_conversation_rp.md").write_text(
            "\n\n  leading and trailing blank lines  \n\n",
        )
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )

        (c,) = await proc.contribute(_request())
        assert c.text == "leading and trailing blank lines"

    @pytest.mark.asyncio
    async def test_estimated_tokens_nonzero_for_nonempty_text(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "full_rp.md").write_text("a" * 40)  # ~10 tokens at 4 chars each
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
        )
        (c,) = await proc.contribute(_request())
        assert c.estimated_tokens >= 1


# ---------------------------------------------------------------------------
# Missing / empty files
# ---------------------------------------------------------------------------


class TestMissingOrEmpty:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert await proc.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_missing_modes_root_returns_empty_list(self, tmp_path: Path) -> None:
        missing_root = tmp_path / "does-not-exist"
        proc = ModeInstructionProvider(
            modes_root=missing_root,
            mode=ChannelMode.text_conversation_rp,
        )
        assert await proc.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        (tmp_path / "text_conversation_rp.md").write_text("")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert await proc.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_whitespace_only_file_returns_empty_list(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "text_conversation_rp.md").write_text("   \n\n   \t\n")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert await proc.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_nonmatching_files_are_ignored(self, tmp_path: Path) -> None:
        """A file for a *different* mode must not leak through."""
        (tmp_path / "full_rp.md").write_text("novel-style prose")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.text_conversation_rp,
        )
        assert await proc.contribute(_request()) == []
