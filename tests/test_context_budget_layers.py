"""Per-layer ``max_tokens`` self-truncation tests.

Each dynamic layer accepts a ``max_tokens`` cap and should drop
trailing items / truncate body text rather than blow past the cap.
The Budgeter applies a final overall trim, but per-layer caps keep
each section's contribution proportional to its allocation.
"""

from __future__ import annotations

import pytest

from familiar_connect.budget import estimate_tokens
from familiar_connect.context import (
    AssemblyContext,
    ConversationSummaryLayer,
    CrossChannelContextLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _ctx(
    *,
    channel_id: int = 1,
    viewer_mode: str = "voice",
    familiar_id: str = "fam",
) -> AssemblyContext:
    return AssemblyContext(
        familiar_id=familiar_id,
        channel_id=channel_id,
        viewer_mode=viewer_mode,
    )


class TestRecentHistoryLayerMaxTokens:
    @pytest.mark.asyncio
    async def test_drops_oldest_to_fit_token_cap(self) -> None:
        store = HistoryStore(":memory:")
        # 10 turns, each ~25 chars of content
        for i in range(10):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"message number {i:02d} blah",
                author=None,
            )
        layer = RecentHistoryLayer(store=store, window_size=20, max_tokens=40)
        msgs = await layer.recent_messages(_ctx())
        # Newest survived; cap honoured.
        assert msgs[-1].content.endswith("09 blah")
        # Token usage stays under cap (allow 1-msg overflow safety).
        total = sum(estimate_tokens(m.content) for m in msgs)
        assert total <= 60

    @pytest.mark.asyncio
    async def test_no_cap_means_full_window(self) -> None:
        store = HistoryStore(":memory:")
        for i in range(5):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"m{i}",
                author=None,
            )
        layer = RecentHistoryLayer(store=store, window_size=20)
        msgs = await layer.recent_messages(_ctx())
        assert len(msgs) == 5


class TestRagContextLayerMaxTokens:
    @pytest.mark.asyncio
    async def test_caps_total_section_tokens(self) -> None:
        """Adding more results stops once tokens cross the cap."""
        store = HistoryStore(":memory:")
        # Seed turns with substantial content so RAG can find them.
        for i in range(20):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"discussion of widgets and gadgets number {i}",
                author=None,
            )
        layer = RagContextLayer(
            store=store,
            max_results=10,
            max_facts=0,
            max_tokens=20,
        )
        layer.set_current_cue("widgets")
        out = await layer.build(_ctx())
        # Either layer returned something tiny or empty; never way over cap.
        assert estimate_tokens(out) <= 40


class TestPeopleDossierLayerMaxTokens:
    @pytest.mark.asyncio
    async def test_drops_trailing_dossiers_past_cap(self) -> None:
        store = HistoryStore(":memory:")
        # Author + dossier rows for several people; turns let them surface.
        for i in range(5):
            uid = f"u{i}"
            author = Author(
                platform="discord",
                user_id=uid,
                username=f"user{i}",
                display_name=f"User{i}",
            )
            store.upsert_account(author)
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"hello from user{i}",
                author=author,
            )
            store.put_people_dossier(
                familiar_id="fam",
                canonical_key=f"discord:{uid}",
                dossier_text="X" * 400,  # ~100 tokens each
                last_fact_id=i + 1,
            )

        # Cap roughly to fit 1-2 dossiers.
        layer = PeopleDossierLayer(
            store=store,
            window_size=20,
            max_people=10,
            max_tokens=200,
        )
        out = await layer.build(_ctx())
        # Bounded; below 2x the cap with safety margin.
        assert estimate_tokens(out) <= 400


class TestConversationSummaryLayerMaxTokens:
    @pytest.mark.asyncio
    async def test_truncates_long_summary(self) -> None:
        store = HistoryStore(":memory:")
        long_text = "summary content blah " * 200  # very long
        # Need at least one turn so summary makes sense; just write summary.
        store.put_summary(
            familiar_id="fam",
            channel_id=1,
            summary_text=long_text,
            last_summarised_id=1,
        )
        layer = ConversationSummaryLayer(store=store, max_tokens=50)
        out = await layer.build(_ctx())
        assert estimate_tokens(out) <= 80  # cap + header overhead


class TestCrossChannelContextLayerMaxTokens:
    @pytest.mark.asyncio
    async def test_truncates_total_cross_channel_block(self) -> None:
        store = HistoryStore(":memory:")
        viewer_channel = 1
        for src in (10, 20, 30):
            store.put_cross_context(
                familiar_id="fam",
                viewer_mode="voice:1",
                source_channel_id=src,
                summary_text="cross summary text " * 100,
                source_last_id=1,
            )
        layer = CrossChannelContextLayer(
            store=store,
            viewer_map={viewer_channel: [10, 20, 30]},
            ttl_seconds=600,
            max_tokens=80,
        )
        out = await layer.build(_ctx(channel_id=viewer_channel))
        assert estimate_tokens(out) <= 120  # cap + section header overhead
