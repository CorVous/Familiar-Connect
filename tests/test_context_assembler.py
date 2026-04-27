"""Tests for :mod:`familiar_connect.context.assembler` and its layers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from familiar_connect.context import (
    Assembler,
    AssemblyContext,
    CharacterCardLayer,
    CoreInstructionsLayer,
    OperatingModeLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(
    *,
    channel_id: int = 1,
    viewer_mode: str = "voice",
    familiar_id: str = "fam",
    guild_id: int | None = None,
) -> AssemblyContext:
    return AssemblyContext(
        familiar_id=familiar_id,
        channel_id=channel_id,
        viewer_mode=viewer_mode,
        guild_id=guild_id,
    )


class TestCoreInstructionsLayer:
    @pytest.mark.asyncio
    async def test_reads_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "core.md"
        p.write_text("You are a helpful familiar.\nBe concise.\n")
        layer = CoreInstructionsLayer(path=p)
        out = await layer.build(_ctx())
        assert "helpful familiar" in out
        assert "Be concise" in out

    @pytest.mark.asyncio
    async def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        layer = CoreInstructionsLayer(path=tmp_path / "missing.md")
        assert not await layer.build(_ctx())

    def test_invalidation_key_changes_on_edit(self, tmp_path: Path) -> None:
        p = tmp_path / "core.md"
        p.write_text("v1")
        layer = CoreInstructionsLayer(path=p)
        key_1 = layer.invalidation_key(_ctx())
        p.write_text("v2")
        key_2 = layer.invalidation_key(_ctx())
        assert key_1 != key_2


class TestCharacterCardLayer:
    @pytest.mark.asyncio
    async def test_reads_card_from_sidecar(self, tmp_path: Path) -> None:
        card = tmp_path / "character.md"
        card.write_text("## Persona\n\nA playful familiar named Aria.\n")
        layer = CharacterCardLayer(card_path=card)
        out = await layer.build(_ctx())
        assert "Aria" in out

    @pytest.mark.asyncio
    async def test_empty_when_no_sidecar(self, tmp_path: Path) -> None:
        layer = CharacterCardLayer(card_path=tmp_path / "no-card.md")
        assert not await layer.build(_ctx())


class TestOperatingModeLayer:
    @pytest.mark.asyncio
    async def test_voice_mode_renders_voice_instructions(self) -> None:
        layer = OperatingModeLayer(
            modes={
                "voice": "Speak in short sentences.",
                "text": "You may use markdown.",
            }
        )
        out = await layer.build(_ctx(viewer_mode="voice"))
        assert "short sentences" in out

    @pytest.mark.asyncio
    async def test_text_mode_renders_text_instructions(self) -> None:
        layer = OperatingModeLayer(
            modes={
                "voice": "Speak in short sentences.",
                "text": "You may use markdown.",
            }
        )
        out = await layer.build(_ctx(viewer_mode="text"))
        assert "markdown" in out

    @pytest.mark.asyncio
    async def test_unknown_mode_returns_empty(self) -> None:
        layer = OperatingModeLayer(modes={"voice": "x"})
        out = await layer.build(_ctx(viewer_mode="weird"))
        assert not out


class TestRecentHistoryLayer:
    @pytest.mark.asyncio
    async def test_pulls_recent_turns_from_store(self) -> None:
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=10,
            role="user",
            content="hello",
            author=alice,
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=10,
            role="assistant",
            content="hi back",
            author=None,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)

        messages = await layer.recent_messages(_ctx(channel_id=10))
        contents = [m.content for m in messages]
        assert any("hello" in c for c in contents)
        assert any("hi back" in c for c in contents)

    @pytest.mark.asyncio
    async def test_user_messages_get_author_name_and_prefix(self) -> None:
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="42", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=10,
            role="user",
            content="hey",
            author=alice,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=10))
        user_msg = next(m for m in messages if m.role == "user")
        # name is derived from platform:user_id, sanitized
        assert user_msg.name is not None
        # content is prefixed with display name so the model sees the
        # attribution even if it drops `name`
        assert "Alice]" in user_msg.content
        assert "hey" in user_msg.content

    @pytest.mark.asyncio
    async def test_user_messages_include_utc_timestamp_in_prefix(self) -> None:
        """Format ``[HH:MM Display] content``.

        Gives the model a sense of message rhythm so a rapid
        back-and-forth between two humans doesn't read as a question
        into the void.
        """
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="42", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=10,
            role="user",
            content="hey",
            author=alice,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=10))
        user_msg = next(m for m in messages if m.role == "user")
        assert re.match(r"^\[\d{2}:\d{2} Alice\] hey$", user_msg.content), (
            user_msg.content
        )

    @pytest.mark.asyncio
    async def test_respects_window_size(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(50):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"m{i}",
                author=None,
            )
        layer = RecentHistoryLayer(store=store, window_size=5)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_build_returns_empty_system_contribution(self) -> None:
        """Return ``""`` so the layer opts out of the system prompt.

        ``RecentHistoryLayer`` contributes to ``recent_history``, not
        the system prompt.
        """
        store = HistoryStore(":memory:")
        layer = RecentHistoryLayer(store=store, window_size=20)
        assert not await layer.build(_ctx(channel_id=1))


class TestRecentHistoryGuildAwareNames:
    """Speaker prefix uses ``resolve_label(canonical_key, guild_id)``.

    Per-guild nicknames must beat the bare ``Author.display_name``
    that's snapshotted on each turn — a user who changed their
    guild-nick mid-conversation should appear under the new name in
    the prompt.
    """

    @pytest.mark.asyncio
    async def test_uses_guild_nick_from_accounts_table(self) -> None:
        store = HistoryStore(":memory:")
        cass = Author(
            platform="discord",
            user_id="111",
            username="cass_login",
            display_name="Cass",  # snapshot at write time
            global_name="Cassidy",
        )
        store.upsert_account(cass)
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=cass,
            guild_id=42,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=1, guild_id=42))
        user_msg = next(m for m in messages if m.role == "user")
        assert "Aria" in user_msg.content
        # Snapshot 'Cass' is no longer the rendered label (modulo it
        # might appear if it's part of the content, but here content="hi")
        assert "Cass" not in user_msg.content


class TestRecentHistoryReplyMarkers:
    """Reply linkage: child turns surface marker + parent context.

    Depth depends on whether the parent is in the same recent window:
    in-window → marker + ≤80 char snippet; out-of-window → marker +
    full parent content (soft-cap 400).
    """

    @pytest.mark.asyncio
    async def test_marker_only_when_parent_in_window(self) -> None:
        store = HistoryStore(":memory:")
        bob = Author(
            platform="discord",
            user_id="2",
            username="bob",
            display_name="Bob",
            global_name="Bob",
        )
        store.upsert_account(bob)
        alice = Author(
            platform="discord",
            user_id="1",
            username="alice",
            display_name="Alice",
            global_name="Alice",
        )
        store.upsert_account(alice)
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="parent message",
            author=bob,
            platform_message_id="msg-1",
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="child message",
            author=alice,
            platform_message_id="msg-2",
            reply_to_message_id="msg-1",
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        child = next(m for m in messages if "child message" in m.content)
        # Marker present, includes Bob.
        assert "↩" in child.content
        assert "Bob" in child.content
        # Snippet kept short (parent content present in some form).
        assert "parent message" in child.content

    @pytest.mark.asyncio
    async def test_inlines_full_parent_when_outside_window(self) -> None:
        """Parent older than recent window: child carries full parent text."""
        store = HistoryStore(":memory:")
        bob = Author(
            platform="discord",
            user_id="2",
            username="bob",
            display_name="Bob",
            global_name="Bob",
        )
        store.upsert_account(bob)
        alice = Author(
            platform="discord",
            user_id="1",
            username="alice",
            display_name="Alice",
            global_name="Alice",
        )
        store.upsert_account(alice)
        # Parent at id 1.
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="parent text far away",
            author=bob,
            platform_message_id="msg-1",
        )
        # Filler so parent falls outside the window (window_size=2).
        for i in range(5):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"filler {i}",
                author=alice,
            )
        # Child reply (will be in the window).
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="child reply",
            author=alice,
            platform_message_id="msg-child",
            reply_to_message_id="msg-1",
        )
        layer = RecentHistoryLayer(store=store, window_size=2)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        child = next(m for m in messages if "child reply" in m.content)
        # Marker AND full parent text appears.
        assert "↩" in child.content
        assert "parent text far away" in child.content

    @pytest.mark.asyncio
    async def test_drops_marker_when_parent_unknown(self) -> None:
        """Parent message_id we've never seen → no marker, render plain."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="1", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="orphan reply",
            author=alice,
            platform_message_id="msg-x",
            reply_to_message_id="msg-ghost",
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        m = next(msg for msg in messages if "orphan reply" in msg.content)
        assert "↩" not in m.content


class TestRecentHistoryMentionRewriting:
    """Discord ``<@USER_ID>`` mention markers become ``[@DisplayName]``."""

    @pytest.mark.asyncio
    async def test_rewrites_user_mention_to_display_name(self) -> None:
        store = HistoryStore(":memory:")
        bob = Author(
            platform="discord",
            user_id="222",
            username="bob",
            display_name="Bob",
            global_name="Bob",
        )
        store.upsert_account(bob)
        alice = Author(
            platform="discord", user_id="111", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey <@222>, look at this",
            author=alice,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        user_msg = next(m for m in messages if "look at this" in m.content)
        assert "<@222>" not in user_msg.content
        assert "[@Bob]" in user_msg.content

    @pytest.mark.asyncio
    async def test_unknown_mention_id_falls_back_to_id(self) -> None:
        """If we don't have the account row, render bare id rather than fail."""
        store = HistoryStore(":memory:")
        alice = Author(
            platform="discord", user_id="111", username="alice", display_name="Alice"
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey <@999>",
            author=alice,
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        messages = await layer.recent_messages(_ctx(channel_id=1))
        user_msg = next(m for m in messages if "hey" in m.content)
        # `[@999]` is the resolved label for unknown account
        # (resolve_label falls back to user_id).
        assert "[@999]" in user_msg.content


class TestAssembler:
    @pytest.mark.asyncio
    async def test_composes_non_empty_layers_in_order(self, tmp_path: Path) -> None:
        core = tmp_path / "core.md"
        core.write_text("CORE\n")
        card = tmp_path / "card.md"
        card.write_text("CARD\n")

        asm = Assembler(
            layers=[
                CoreInstructionsLayer(path=core),
                CharacterCardLayer(card_path=card),
                OperatingModeLayer(modes={"voice": "BE_TERSE"}),
            ],
        )
        prompt = await asm.assemble(_ctx(viewer_mode="voice"))
        assert "CORE" in prompt.system_prompt
        assert "CARD" in prompt.system_prompt
        assert "BE_TERSE" in prompt.system_prompt
        # Empty layers are dropped
        asm2 = Assembler(
            layers=[
                CoreInstructionsLayer(path=tmp_path / "missing.md"),
                CharacterCardLayer(card_path=card),
            ],
        )
        prompt2 = await asm2.assemble(_ctx())
        assert "CARD" in prompt2.system_prompt
        # Only CARD, no leading blank lines
        assert prompt2.system_prompt.strip().startswith("CARD")

    @pytest.mark.asyncio
    async def test_recent_history_layer_contributes_messages(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey",
            author=None,
        )
        asm = Assembler(
            layers=[
                RecentHistoryLayer(store=store, window_size=20),
            ],
        )
        prompt = await asm.assemble(_ctx(channel_id=1))
        assert not prompt.system_prompt
        assert len(prompt.recent_history) == 1
        assert prompt.recent_history[0].content.endswith("hey")

    @pytest.mark.asyncio
    async def test_cache_reuses_layer_output_on_same_key(self) -> None:
        calls = 0

        class _CountingLayer:
            name = "counter"

            def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
                return "stable"

            async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
                nonlocal calls
                calls += 1
                return "X"

        asm = Assembler(layers=[_CountingLayer()])
        await asm.assemble(_ctx())
        await asm.assemble(_ctx())
        assert calls == 1  # second call hit the cache

    @pytest.mark.asyncio
    async def test_cache_invalidates_on_key_change(self) -> None:
        calls = 0
        current_key = "a"

        class _DynamicLayer:
            name = "dyn"

            def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
                return current_key

            async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
                nonlocal calls
                calls += 1
                return f"v={current_key}"

        asm = Assembler(layers=[_DynamicLayer()])
        out1 = await asm.assemble(_ctx())
        assert calls == 1
        current_key = "b"
        out2 = await asm.assemble(_ctx())
        assert calls == 2
        assert out1.system_prompt != out2.system_prompt
