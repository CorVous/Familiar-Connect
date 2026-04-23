"""Phase-4: :class:`RagContextLayer` merges facts with turns."""

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
