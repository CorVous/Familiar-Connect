"""Tests for Phase-3 context layers.

Covers :class:`ConversationSummaryLayer`,
:class:`CrossChannelContextLayer`, and :class:`RagContextLayer`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import (
    ConversationSummaryLayer,
    CrossChannelContextLayer,
    RagContextLayer,
)
from familiar_connect.history.store import HistoryStore


def _ctx(*, channel_id: int = 1, viewer_mode: str = "voice") -> AssemblyContext:
    return AssemblyContext(
        familiar_id="fam", channel_id=channel_id, viewer_mode=viewer_mode
    )


class TestConversationSummaryLayer:
    @pytest.mark.asyncio
    async def test_returns_summary_text(self) -> None:
        store = HistoryStore(":memory:")
        store.put_summary(
            familiar_id="fam",
            channel_id=1,
            last_summarised_id=5,
            summary_text="Earlier they talked about foxes.",
        )
        layer = ConversationSummaryLayer(store=store)
        out = await layer.build(_ctx())
        assert "foxes" in out

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_summary(self) -> None:
        store = HistoryStore(":memory:")
        layer = ConversationSummaryLayer(store=store)
        assert not await layer.build(_ctx())

    def test_invalidation_key_tracks_watermark(self) -> None:
        store = HistoryStore(":memory:")
        store.put_summary(
            familiar_id="fam",
            channel_id=1,
            last_summarised_id=5,
            summary_text="v1",
        )
        layer = ConversationSummaryLayer(store=store)
        k1 = layer.invalidation_key(_ctx())
        store.put_summary(
            familiar_id="fam",
            channel_id=1,
            last_summarised_id=10,
            summary_text="v2",
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2


class TestCrossChannelContextLayer:
    @pytest.mark.asyncio
    async def test_renders_configured_sources(self) -> None:
        store = HistoryStore(":memory:")
        store.put_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
            source_last_id=5,
            summary_text="In #general: banter about pumpkins.",
        )
        layer = CrossChannelContextLayer(
            store=store, viewer_map={1: [100]}, ttl_seconds=300
        )
        out = await layer.build(_ctx(channel_id=1, viewer_mode="voice"))
        assert "pumpkins" in out

    @pytest.mark.asyncio
    async def test_suppresses_stale_summaries_past_ttl(self) -> None:
        store = HistoryStore(":memory:")
        store.put_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
            source_last_id=5,
            summary_text="Stale content.",
        )
        # Manually backdate created_at.
        old_ts = (datetime.now(tz=UTC) - timedelta(minutes=30)).isoformat()
        store._conn.execute(
            "UPDATE cross_context_summaries SET created_at = ?",
            (old_ts,),
        )
        store._conn.commit()

        layer = CrossChannelContextLayer(
            store=store, viewer_map={1: [100]}, ttl_seconds=60
        )
        out = await layer.build(_ctx(channel_id=1, viewer_mode="voice"))
        # Stale content should not be rendered; layer opts out.
        assert not out

    @pytest.mark.asyncio
    async def test_no_config_for_viewer_returns_empty(self) -> None:
        store = HistoryStore(":memory:")
        store.put_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
            source_last_id=5,
            summary_text="Something.",
        )
        layer = CrossChannelContextLayer(store=store, viewer_map={}, ttl_seconds=300)
        assert not await layer.build(_ctx(channel_id=1))

    @pytest.mark.asyncio
    async def test_multiple_sources_concatenated(self) -> None:
        store = HistoryStore(":memory:")
        store.put_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=100,
            source_last_id=5,
            summary_text="From #general: apples.",
        )
        store.put_cross_context(
            familiar_id="fam",
            viewer_mode="voice:1",
            source_channel_id=200,
            source_last_id=8,
            summary_text="From #alerts: a service restart.",
        )
        layer = CrossChannelContextLayer(
            store=store, viewer_map={1: [100, 200]}, ttl_seconds=300
        )
        out = await layer.build(_ctx(channel_id=1))
        assert "apples" in out
        assert "service restart" in out


class TestRagContextLayer:
    @pytest.mark.asyncio
    async def test_empty_when_no_cues(self) -> None:
        store = HistoryStore(":memory:")
        layer = RagContextLayer(store=store, max_results=5)
        # No cues ⇒ empty contribution.
        assert not await layer.build(_ctx())

    @pytest.mark.asyncio
    async def test_returns_matches_for_current_cue(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="Let's discuss the fox plan tomorrow at noon.",
            author=None,
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="Sure thing.",
            author=None,
        )
        layer = RagContextLayer(store=store, max_results=5)
        layer.set_current_cue("fox")
        out = await layer.build(_ctx(channel_id=1))
        assert "fox plan" in out

    @pytest.mark.asyncio
    async def test_scoped_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="other",
            channel_id=1,
            role="user",
            content="Fox on a different familiar.",
            author=None,
        )
        layer = RagContextLayer(store=store, max_results=5)
        layer.set_current_cue("fox")
        out = await layer.build(_ctx())
        assert not out

    def test_invalidation_key_reflects_cue_and_watermark(self) -> None:
        store = HistoryStore(":memory:")
        layer = RagContextLayer(store=store, max_results=5)
        layer.set_current_cue("fox")
        k1 = layer.invalidation_key(_ctx())
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="new turn",
            author=None,
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2
        layer.set_current_cue("otter")
        k3 = layer.invalidation_key(_ctx())
        assert k3 not in {k1, k2}
