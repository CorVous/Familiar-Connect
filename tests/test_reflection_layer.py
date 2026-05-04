"""Tests for :class:`ReflectionLayer` (M3).

Renders recent reflections as a system-prompt block with citation
breadcrumbs ``[T#id, F#id]`` and a ``(stale)`` flag on rows that cite
superseded facts.
"""

from __future__ import annotations

import pytest

from familiar_connect.context import ReflectionLayer
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.history.store import HistoryStore


def _ctx(channel_id: int | None = 1) -> AssemblyContext:
    return AssemblyContext(
        familiar_id="fam",
        channel_id=channel_id,
        viewer_mode="text",
    )


@pytest.mark.asyncio
async def test_empty_when_no_reflections() -> None:
    store = HistoryStore(":memory:")
    layer = ReflectionLayer(store=store, max_reflections=3)
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_renders_text_with_citation_breadcrumbs() -> None:
    store = HistoryStore(":memory:")
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="Crew morale dipped after Friday.",
        cited_turn_ids=[42],
        cited_fact_ids=[7],
        last_turn_id=42,
        last_fact_id=7,
    )
    layer = ReflectionLayer(store=store, max_reflections=3)
    out = await layer.build(_ctx())
    assert "## Recent reflections" in out
    assert "Crew morale dipped after Friday." in out
    assert "[T#42, F#7]" in out
    assert "(stale)" not in out


@pytest.mark.asyncio
async def test_flags_stale_when_any_cited_fact_superseded() -> None:
    store = HistoryStore(":memory:")
    f1 = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria lives in Paris.",
        source_turn_ids=[1],
    )
    f2 = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria lives in Berlin.",
        source_turn_ids=[2],
    )
    store.supersede_fact(familiar_id="fam", old_id=f1.id, new_id=f2.id)
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="Aria's location keeps shifting.",
        cited_turn_ids=[1],
        cited_fact_ids=[f1.id, f2.id],
        last_turn_id=2,
        last_fact_id=f2.id,
    )
    layer = ReflectionLayer(store=store, max_reflections=3)
    out = await layer.build(_ctx())
    assert "(stale)" in out


@pytest.mark.asyncio
async def test_does_not_flag_stale_when_no_facts_superseded() -> None:
    store = HistoryStore(":memory:")
    f = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria lives in Berlin.",
        source_turn_ids=[1],
    )
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes the cold.",
        cited_turn_ids=[1],
        cited_fact_ids=[f.id],
        last_turn_id=1,
        last_fact_id=f.id,
    )
    layer = ReflectionLayer(store=store, max_reflections=3)
    out = await layer.build(_ctx())
    assert "(stale)" not in out


@pytest.mark.asyncio
async def test_max_reflections_caps_output() -> None:
    store = HistoryStore(":memory:")
    for i in range(5):
        store.append_reflection(
            familiar_id="fam",
            channel_id=1,
            text=f"reflection {i}",
            cited_turn_ids=[i + 1],
            cited_fact_ids=[],
            last_turn_id=i + 1,
            last_fact_id=0,
        )
    layer = ReflectionLayer(store=store, max_reflections=2)
    out = await layer.build(_ctx())
    assert out.count("- reflection") == 2
    # Newest first.
    assert "reflection 4" in out
    assert "reflection 3" in out
    assert "reflection 2" not in out


@pytest.mark.asyncio
async def test_max_reflections_zero_opts_out() -> None:
    store = HistoryStore(":memory:")
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="something",
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=1,
        last_fact_id=0,
    )
    layer = ReflectionLayer(store=store, max_reflections=0)
    out = await layer.build(_ctx())
    assert not out


def test_invalidation_key_changes_on_new_reflection() -> None:
    store = HistoryStore(":memory:")
    layer = ReflectionLayer(store=store, max_reflections=3)
    k0 = layer.invalidation_key(_ctx())
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="r1",
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=1,
        last_fact_id=0,
    )
    k1 = layer.invalidation_key(_ctx())
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="r2",
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=2,
        last_fact_id=0,
    )
    k2 = layer.invalidation_key(_ctx())
    assert k0 != k1
    assert k1 != k2


@pytest.mark.asyncio
async def test_channel_scope_excludes_other_channels() -> None:
    store = HistoryStore(":memory:")
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text="ch1 reflection",
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=1,
        last_fact_id=0,
    )
    store.append_reflection(
        familiar_id="fam",
        channel_id=2,
        text="ch2 reflection",
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=1,
        last_fact_id=0,
    )
    layer = ReflectionLayer(store=store, max_reflections=5)
    out = await layer.build(_ctx(channel_id=1))
    assert "ch1 reflection" in out
    assert "ch2 reflection" not in out


@pytest.mark.asyncio
async def test_max_tokens_truncates_block() -> None:
    store = HistoryStore(":memory:")
    long = "x" * 4000
    store.append_reflection(
        familiar_id="fam",
        channel_id=1,
        text=long,
        cited_turn_ids=[1],
        cited_fact_ids=[],
        last_turn_id=1,
        last_fact_id=0,
    )
    layer = ReflectionLayer(store=store, max_reflections=3, max_tokens=20)
    out = await layer.build(_ctx())
    # 20 tokens ≈ 80 chars; the rendered line should be much shorter
    # than the raw 4000-char text.
    assert len(out) < 200
