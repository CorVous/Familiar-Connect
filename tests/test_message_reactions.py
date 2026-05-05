"""Discord reactions on persisted text messages.

Reactions arrive after the original message via gateway events. We
persist them keyed by ``platform_message_id`` and surface them in
:class:`RecentHistoryLayer`'s output so the LLM sees the social
signal.

Performance: reactions for an entire recent-history window must
resolve in a single SQL roundtrip — Discord publishes reaction
events frequently and the assembler runs on every reply.
"""

from __future__ import annotations

import pytest
from discord import PartialEmoji

from familiar_connect.bot import (
    _emoji_repr,
    apply_reaction_clear,
    apply_reaction_delta,
)
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import RecentHistoryLayer
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _ctx(*, channel_id: int = 1, familiar_id: str = "fam") -> AssemblyContext:
    return AssemblyContext(
        familiar_id=familiar_id, channel_id=channel_id, viewer_mode="text"
    )


def _author(user_id: str = "111", name: str = "Alice") -> Author:
    return Author(
        platform="discord",
        user_id=user_id,
        username=name.lower(),
        display_name=name,
    )


class TestReactionsStore:
    def test_set_and_lookup_single(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam",
            platform_message_id="m1",
            emoji="👍",
            count=3,
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {"m1": (("👍", 3),)}

    def test_set_reaction_is_upsert(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=1
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=5
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {"m1": (("👍", 5),)}

    def test_count_zero_removes_row(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=0
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {}

    def test_batch_fetch_orders_by_count_desc_then_emoji(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="🎉", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=5
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="❤️", count=2
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        # 👍 wins on count; 🎉 / ❤️ tie at 2 — emoji-string asc breaks ties.
        assert out["m1"][0] == ("👍", 5)
        assert {e for e, _ in out["m1"][1:]} == {"🎉", "❤️"}

    def test_batch_fetch_scopes_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="famA", platform_message_id="m1", emoji="👍", count=1
        )
        store.set_reaction(
            familiar_id="famB", platform_message_id="m1", emoji="❤️", count=1
        )
        out_a = store.reactions_for_messages(
            familiar_id="famA", platform_message_ids=["m1"]
        )
        assert out_a == {"m1": (("👍", 1),)}

    def test_batch_fetch_unknown_ids_omitted(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=1
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1", "nope"]
        )
        assert "nope" not in out
        assert out["m1"] == (("👍", 1),)

    def test_bump_reaction_increments(self) -> None:
        store = HistoryStore(":memory:")
        store.bump_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", delta=1
        )
        store.bump_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", delta=1
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {"m1": (("👍", 2),)}

    def test_bump_reaction_floor_at_zero_drops_row(self) -> None:
        store = HistoryStore(":memory:")
        store.bump_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", delta=1
        )
        store.bump_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", delta=-2
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {}

    def test_bump_reaction_negative_with_no_row_is_noop(self) -> None:
        """Stray remove without a matching add (bot offline) does nothing."""
        store = HistoryStore(":memory:")
        store.bump_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", delta=-1
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {}

    def test_clear_reactions_all_emojis(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="❤️", count=1
        )
        store.clear_reactions(familiar_id="fam", platform_message_id="m1")
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {}

    def test_clear_reactions_one_emoji(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="❤️", count=1
        )
        store.clear_reactions(familiar_id="fam", platform_message_id="m1", emoji="👍")
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["m1"]
        )
        assert out == {"m1": (("❤️", 1),)}

    def test_batch_fetch_uses_single_query(self) -> None:
        """Reactions for an entire window resolve in O(1) SQL roundtrips.

        Performance contract — Discord pushes reaction events often
        and the assembler runs per reply. Counting with a SQL trace
        keeps us honest if someone refactors into a per-message loop.
        """
        store = HistoryStore(":memory:")
        for i in range(5):
            store.set_reaction(
                familiar_id="fam",
                platform_message_id=f"m{i}",
                emoji="👍",
                count=1,
            )
        seen: list[str] = []
        store._conn.set_trace_callback(seen.append)  # type: ignore[attr-defined]
        try:
            store.reactions_for_messages(
                familiar_id="fam",
                platform_message_ids=[f"m{i}" for i in range(5)],
            )
        finally:
            store._conn.set_trace_callback(None)  # type: ignore[attr-defined]
        select_stmts = [s for s in seen if s.lstrip().upper().startswith("SELECT")]
        assert len(select_stmts) == 1, select_stmts


class TestRecentHistoryReactions:
    @pytest.mark.asyncio
    async def test_user_message_renders_reactions(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author(),
            platform_message_id="m1",
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="m1", emoji="❤️", count=1
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert msgs[0].role == "user"
        assert "👍 x2" in msgs[0].content
        assert "❤️ x1" in msgs[0].content

    @pytest.mark.asyncio
    async def test_assistant_message_renders_reactions(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="hi back",
            author=None,
            platform_message_id="bot1",
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="bot1", emoji="🎉", count=1
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert msgs[0].role == "assistant"
        assert "🎉 x1" in msgs[0].content

    @pytest.mark.asyncio
    async def test_no_reactions_no_suffix(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author(),
            platform_message_id="m1",
        )
        layer = RecentHistoryLayer(store=store, window_size=20)
        msgs = await layer.recent_messages(_ctx(channel_id=1))
        assert "reactions" not in msgs[0].content.lower()


class TestEmojiRepr:
    def test_unicode_emoji_returns_name(self) -> None:
        assert _emoji_repr(PartialEmoji(name="👍")) == "👍"

    def test_custom_emoji_returns_tagged_form(self) -> None:
        out = _emoji_repr(PartialEmoji(name="party_blob", id=12345))
        assert out == "<:party_blob:12345>"

    def test_animated_custom_emoji_uses_a_prefix(self) -> None:
        out = _emoji_repr(PartialEmoji(name="dance", id=999, animated=True))
        assert out == "<a:dance:999>"

    def test_empty_name_returns_empty_string(self) -> None:
        assert not _emoji_repr(PartialEmoji(name=None))


class TestBotReactionDispatch:
    def test_add_then_remove_returns_to_zero(self) -> None:
        store = HistoryStore(":memory:")
        emoji = PartialEmoji(name="👍")
        apply_reaction_delta(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: True,
            channel_id=1,
            message_id=42,
            emoji=emoji,
            delta=1,
        )
        apply_reaction_delta(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: True,
            channel_id=1,
            message_id=42,
            emoji=emoji,
            delta=1,
        )
        apply_reaction_delta(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: True,
            channel_id=1,
            message_id=42,
            emoji=emoji,
            delta=-1,
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["42"]
        )
        assert out == {"42": (("👍", 1),)}

    def test_unsubscribed_channel_writes_nothing(self) -> None:
        store = HistoryStore(":memory:")
        apply_reaction_delta(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: False,
            channel_id=1,
            message_id=42,
            emoji=PartialEmoji(name="👍"),
            delta=1,
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["42"]
        )
        assert out == {}

    def test_clear_drops_everything(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="42", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="42", emoji="❤️", count=1
        )
        apply_reaction_clear(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: True,
            channel_id=1,
            message_id=42,
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["42"]
        )
        assert out == {}

    def test_clear_one_emoji_keeps_others(self) -> None:
        store = HistoryStore(":memory:")
        store.set_reaction(
            familiar_id="fam", platform_message_id="42", emoji="👍", count=2
        )
        store.set_reaction(
            familiar_id="fam", platform_message_id="42", emoji="❤️", count=1
        )
        apply_reaction_clear(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _ch: True,
            channel_id=1,
            message_id=42,
            emoji=PartialEmoji(name="👍"),
        )
        out = store.reactions_for_messages(
            familiar_id="fam", platform_message_ids=["42"]
        )
        assert out == {"42": (("❤️", 1),)}


class TestRecentHistoryReactionsBatch:
    @pytest.mark.asyncio
    async def test_window_resolves_reactions_in_one_query(self) -> None:
        """One SELECT for reactions across the entire window.

        Mirrors the store-level perf test but exercises the call path
        the responder actually uses.
        """
        store = HistoryStore(":memory:")
        for i in range(4):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"msg {i}",
                author=_author(),
                platform_message_id=f"m{i}",
            )
            store.set_reaction(
                familiar_id="fam",
                platform_message_id=f"m{i}",
                emoji="👍",
                count=1,
            )
        layer = RecentHistoryLayer(store=store, window_size=20)
        seen: list[str] = []
        store._conn.set_trace_callback(seen.append)  # type: ignore[attr-defined]
        try:
            await layer.recent_messages(_ctx(channel_id=1))
        finally:
            store._conn.set_trace_callback(None)  # type: ignore[attr-defined]
        reaction_selects = [
            s
            for s in seen
            if "message_reactions" in s and s.lstrip().upper().startswith("SELECT")
        ]
        assert len(reaction_selects) == 1, reaction_selects
