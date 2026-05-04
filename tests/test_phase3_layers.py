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
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import Author


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

    @pytest.mark.asyncio
    async def test_excludes_turns_within_recent_window(self) -> None:
        """RAG must not re-surface turns already in ``RecentHistoryLayer``.

        Otherwise the user's own most-recent message (which is in
        ``fts_turns`` and matches its own cue perfectly) shows up
        twice in the prompt — once verbatim, once as "possibly
        relevant earlier turns".
        """
        store = HistoryStore(":memory:")
        # 30 turns total; the ones inside the recent-window (the
        # last 20) must NOT appear in RAG output.
        for i in range(30):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"strawberry observation number {i}",
                author=None,
            )
        layer = RagContextLayer(store=store, max_results=10, recent_window_size=20)
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        # Every numeric mention should be from id 1..10 (older than
        # the 20-turn window's id 11..30).
        for older_idx in range(10):
            assert f"number {older_idx}" in out, (older_idx, out)
        for recent_idx in range(10, 30):
            assert f"number {recent_idx}" not in out, (recent_idx, out)

    @pytest.mark.asyncio
    async def test_fact_with_renamed_subject_gets_annotation(self) -> None:
        """When a fact's subject has since renamed, annotate the rendered fact.

        Render preserves the original fact text (what was actually
        said) and appends a soft hint linking the stale display name
        to the current one. The link is advisory — a real-world mic
        share or relay can break the canonical-key correspondence —
        but it's enough to keep the model from treating "Cass" and
        "peeks" as different people.
        """
        store = HistoryStore(":memory:")
        # User started as "Cass"; an earlier turn from before rename.
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hello from cass",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cass",
            ),
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:111", display_at_write="Cass"),
            ),
        )
        # The same user has since renamed to "peeks".
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="still here under a new name",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="peeks",
            ),
        )
        layer = RagContextLayer(store=store, max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "Cass likes pho." in out  # original text preserved
        assert "peeks" in out  # current display name surfaced
        assert "Cass" in out  # old name still visible (it's in the text)

    @pytest.mark.asyncio
    async def test_fact_without_rename_renders_without_annotation(self) -> None:
        """If display name hasn't changed, no annotation is added."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hello from cass",
            author=Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cass",
            ),
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:111", display_at_write="Cass"),
            ),
        )
        layer = RagContextLayer(store=store, max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "Cass likes pho." in out
        # No "now known as" / "formerly" / "→" annotation when names match.
        assert "now known as" not in out.lower()
        assert "formerly" not in out.lower()

    @pytest.mark.asyncio
    async def test_legacy_fact_without_subjects_renders_unchanged(self) -> None:
        """Facts written before the subjects feature have no metadata.

        Forward-only fix — they render as plain text, no annotation.
        """
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="An old fact about pho.",
            source_turn_ids=[],
        )
        layer = RagContextLayer(store=store, max_facts=3)
        layer.set_current_cue("pho")
        out = await layer.build(_ctx(channel_id=1))

        assert "An old fact about pho." in out
        assert "now known as" not in out.lower()

    @pytest.mark.asyncio
    async def test_renders_with_date_header_and_12h_clock(self) -> None:
        """Earlier turns group under ``YYYY-MM-DD:`` with ``H:MMpm`` lines."""
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="my brain's dying",
            author=Author(
                platform="discord",
                user_id="111",
                username="peebo",
                display_name="Peebo",
            ),
        )
        # pin the timestamp so the assertion is stable
        ts = datetime(2026, 5, 3, 14, 29, tzinfo=UTC).isoformat()
        store._conn.execute("UPDATE turns SET timestamp = ? WHERE id = 1", (ts,))
        store._conn.commit()
        layer = RagContextLayer(store=store, max_results=5, context_window=0)
        layer.set_current_cue("brain")
        out = await layer.build(_ctx(channel_id=1))
        assert "## Possibly relevant earlier turns" in out
        assert "2026-05-03:" in out
        assert "> [2:29PM Peebo]: my brain's dying" in out

    @pytest.mark.asyncio
    async def test_includes_neighbour_context_per_hit(self) -> None:
        """Each hit pulls ±context_window neighbours, dedup'd, in order."""
        store = HistoryStore(":memory:")
        for content in [
            "warmup chatter A",
            "marker turn that mentions strawberry",
            "follow up reply B",
            "unrelated middle",
            "later mention of strawberry too",
        ]:
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=content,
                author=None,
            )
        layer = RagContextLayer(store=store, max_results=5, context_window=1)
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        # hit + immediate neighbours rendered; unrelated middle still
        # appears because it's a neighbour of the second hit too.
        assert "warmup chatter A" in out
        assert "marker turn that mentions strawberry" in out
        assert "follow up reply B" in out
        assert "later mention of strawberry too" in out

    @pytest.mark.asyncio
    async def test_zero_recent_window_keeps_old_behavior(self) -> None:
        """Default ``recent_window_size=0`` disables exclusion.

        Existing tests and any caller that doesn't opt into the
        window-aware mode get the unfiltered RAG semantics.
        """
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="strawberry observation",
            author=None,
        )
        layer = RagContextLayer(store=store, max_results=5)  # default = 0
        layer.set_current_cue("strawberry")
        out = await layer.build(_ctx(channel_id=1))
        assert "strawberry observation" in out
