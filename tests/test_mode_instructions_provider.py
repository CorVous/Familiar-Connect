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


# ---------------------------------------------------------------------------
# Channel backdrop override
# ---------------------------------------------------------------------------


class TestChannelBackdropOverride:
    @pytest.mark.asyncio
    async def test_backdrop_override_emitted_when_set(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="Custom channel text.",
        )
        contributions = await proc.contribute(_request())
        assert len(contributions) == 1
        assert contributions[0].text == "Custom channel text."

    @pytest.mark.asyncio
    async def test_backdrop_override_source_prefix(self, tmp_path: Path) -> None:
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="Custom.",
        )
        (c,) = await proc.contribute(_request())
        assert c.source == "channel_backdrop:full_rp"

    @pytest.mark.asyncio
    async def test_backdrop_override_beats_mode_file(self, tmp_path: Path) -> None:
        (tmp_path / "full_rp.md").write_text("Mode default text.")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="Channel override wins.",
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Channel override wins."

    @pytest.mark.asyncio
    async def test_whitespace_only_backdrop_falls_back_to_mode_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "full_rp.md").write_text("Mode default text.")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="   ",
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Mode default text."

    @pytest.mark.asyncio
    async def test_empty_backdrop_falls_back_to_mode_file(self, tmp_path: Path) -> None:
        (tmp_path / "full_rp.md").write_text("Mode default text.")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="",
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Mode default text."

    @pytest.mark.asyncio
    async def test_none_backdrop_falls_back_to_mode_file(self, tmp_path: Path) -> None:
        (tmp_path / "full_rp.md").write_text("Mode default text.")
        proc = ModeInstructionProvider(
            modes_root=tmp_path,
            mode=ChannelMode.full_rp,
            channel_backdrop_override=None,
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Mode default text."


# ---------------------------------------------------------------------------
# Defaults fallback (defaults_modes_root)
# ---------------------------------------------------------------------------


class TestDefaultsFallback:
    @pytest.mark.asyncio
    async def test_defaults_used_when_familiar_file_absent(
        self, tmp_path: Path
    ) -> None:
        familiar_modes = tmp_path / "familiar" / "modes"
        familiar_modes.mkdir(parents=True)
        default_modes = tmp_path / "default" / "modes"
        default_modes.mkdir(parents=True)
        (default_modes / "full_rp.md").write_text("Default fallback text.")

        proc = ModeInstructionProvider(
            modes_root=familiar_modes,
            mode=ChannelMode.full_rp,
            defaults_modes_root=default_modes,
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Default fallback text."
        assert c.source == "mode_instructions_default:full_rp"

    @pytest.mark.asyncio
    async def test_familiar_file_beats_defaults(self, tmp_path: Path) -> None:
        familiar_modes = tmp_path / "familiar" / "modes"
        familiar_modes.mkdir(parents=True)
        default_modes = tmp_path / "default" / "modes"
        default_modes.mkdir(parents=True)
        (familiar_modes / "full_rp.md").write_text("Familiar wins.")
        (default_modes / "full_rp.md").write_text("Default text.")

        proc = ModeInstructionProvider(
            modes_root=familiar_modes,
            mode=ChannelMode.full_rp,
            defaults_modes_root=default_modes,
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Familiar wins."
        assert c.source == "mode_instructions:full_rp"

    @pytest.mark.asyncio
    async def test_neither_file_nor_defaults_returns_empty(
        self, tmp_path: Path
    ) -> None:
        familiar_modes = tmp_path / "familiar" / "modes"
        familiar_modes.mkdir(parents=True)
        default_modes = tmp_path / "default" / "modes"
        default_modes.mkdir(parents=True)

        proc = ModeInstructionProvider(
            modes_root=familiar_modes,
            mode=ChannelMode.full_rp,
            defaults_modes_root=default_modes,
        )
        assert await proc.contribute(_request()) == []

    @pytest.mark.asyncio
    async def test_backdrop_beats_defaults(self, tmp_path: Path) -> None:
        familiar_modes = tmp_path / "familiar" / "modes"
        familiar_modes.mkdir(parents=True)
        default_modes = tmp_path / "default" / "modes"
        default_modes.mkdir(parents=True)
        (default_modes / "full_rp.md").write_text("Default text.")

        proc = ModeInstructionProvider(
            modes_root=familiar_modes,
            mode=ChannelMode.full_rp,
            channel_backdrop_override="Channel wins over default.",
            defaults_modes_root=default_modes,
        )
        (c,) = await proc.contribute(_request())
        assert c.text == "Channel wins over default."
        assert c.source == "channel_backdrop:full_rp"
