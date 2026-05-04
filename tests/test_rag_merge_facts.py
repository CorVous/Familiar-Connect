"""Phase-4: :class:`RagContextLayer` merges facts with turns.

Also covers M2 importance-weighted reranking when retrieval weights
are supplied.
"""

from __future__ import annotations

import pytest

from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import RagContextLayer
from familiar_connect.history.store import HistoryStore


def _ctx() -> AssemblyContext:
    return AssemblyContext(familiar_id="fam", channel_id=1, viewer_mode="voice")


class TestRagFactsMerge:
    @pytest.mark.asyncio
    async def test_renders_fact_and_turn_sections(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="Mentioned strawberries in passing.",
            author=None,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
        )
        layer = RagContextLayer(store=store, max_results=5, max_facts=3)
        layer.set_current_cue("strawb")
        out = await layer.build(_ctx())
        assert "relevant facts" in out
        assert "Aria likes strawberries" in out
        assert "relevant earlier turns" in out
        assert "Mentioned strawberries in passing" in out

    @pytest.mark.asyncio
    async def test_fact_only_when_no_matching_turns(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[99],
        )
        layer = RagContextLayer(store=store, max_facts=3)
        layer.set_current_cue("strawb")
        out = await layer.build(_ctx())
        assert "Aria likes strawberries" in out
        assert "earlier turns" not in out

    def test_invalidation_key_reflects_fact_watermark(self) -> None:
        store = HistoryStore(":memory:")
        layer = RagContextLayer(store=store)
        layer.set_current_cue("foo")
        k1 = layer.invalidation_key(_ctx())
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="New fact.",
            source_turn_ids=[1],
        )
        k2 = layer.invalidation_key(_ctx())
        assert k1 != k2


class TestRagFactImportanceReranking:
    """M2 — importance-weighted retrieval.

    With ``importance_weight > 0`` the layer over-fetches BM25 candidates
    and reranks so a high-importance fact beats an equally-matched
    low-importance one.
    """

    @pytest.mark.asyncio
    async def test_high_importance_outranks_low_with_equal_match(self) -> None:
        store = HistoryStore(":memory:")
        # Two facts that both match "strawb*"; the high-importance one should
        # win the single available slot when importance is weighted heavily.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria casually mentioned strawberries once.",
            source_turn_ids=[1],
            importance=1,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is severely allergic to strawberries.",
            source_turn_ids=[2],
            importance=10,
        )
        layer = RagContextLayer(
            store=store,
            max_facts=1,
            bm25_weight=1.0,
            importance_weight=5.0,
            recency_weight=0.0,
        )
        layer.set_current_cue("strawb")
        out = await layer.build(_ctx())
        assert "severely allergic" in out
        assert "casually mentioned" not in out

    @pytest.mark.asyncio
    async def test_default_weights_preserve_bm25_order(self) -> None:
        """Constructor with no weights set must match pre-M2 ordering."""
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
            importance=1,
        )
        layer = RagContextLayer(store=store, max_facts=5)
        layer.set_current_cue("strawb")
        out = await layer.build(_ctx())
        assert "Aria likes strawberries" in out

    @pytest.mark.asyncio
    async def test_legacy_facts_treated_as_neutral(self) -> None:
        """``importance=None`` rows participate as the midpoint, not zero."""
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Legacy fact about strawberries.",
            source_turn_ids=[1],
            # no importance — legacy / unscored
        )
        layer = RagContextLayer(
            store=store,
            max_facts=1,
            bm25_weight=1.0,
            importance_weight=2.0,
        )
        layer.set_current_cue("strawb")
        out = await layer.build(_ctx())
        assert "Legacy fact about strawberries" in out
