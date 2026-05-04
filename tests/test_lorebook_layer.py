"""Tests for :class:`LorebookLayer` (M4).

Hand-authored, keyword-activated canon. The layer reads
``data/familiars/<id>/lorebook.toml`` and matches entry ``keys``
against the recent-history window; hits render at declared
``priority`` order, sorted descending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context import LorebookLayer
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.history.store import HistoryStore

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(channel_id: int | None = 1) -> AssemblyContext:
    return AssemblyContext(
        familiar_id="fam",
        channel_id=channel_id,
        viewer_mode="text",
    )


def _seed(store: HistoryStore, *texts: str, channel_id: int = 1) -> None:
    for text in texts:
        store.append_turn(
            familiar_id="fam",
            channel_id=channel_id,
            role="user",
            content=text,
            author=None,
        )


def _write_lorebook(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_empty_when_file_missing(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "anything")
    layer = LorebookLayer(store=store, path=tmp_path / "missing.toml")
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_empty_when_no_entries_match(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "hello there")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["dragon"]\ncontent = "Dragons breathe fire."\n',
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_single_entry_triggers_on_key_match(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "Tell me about Paris.")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "Paris is the capital of France."\n',
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert "## Lorebook" in out
    assert "Paris is the capital of France." in out


@pytest.mark.asyncio
async def test_match_is_case_insensitive(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "PARIS is lovely")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "City info."\n',
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert "City info." in out


@pytest.mark.asyncio
async def test_multiple_entries_sorted_by_priority_desc(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "paris and dragons")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        (
            '[[entries]]\nkeys = ["paris"]\ncontent = "low-pri Paris"\n'
            "priority = 1\n\n"
            '[[entries]]\nkeys = ["dragon"]\ncontent = "high-pri dragon"\n'
            "priority = 100\n"
        ),
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    high = out.index("high-pri dragon")
    low = out.index("low-pri Paris")
    assert high < low


@pytest.mark.asyncio
async def test_selective_requires_all_keys(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "We met in Paris. No mention of the second key.")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        (
            '[[entries]]\nkeys = ["paris", "berlin"]\nselective = true\n'
            'content = "Both cities matter."\n'
        ),
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_selective_fires_when_all_keys_present(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "We met in Paris and later moved to Berlin.")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        (
            '[[entries]]\nkeys = ["paris", "berlin"]\nselective = true\n'
            'content = "Both cities matter."\n'
        ),
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert "Both cities matter." in out


@pytest.mark.asyncio
async def test_max_entries_caps_output(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "alpha beta gamma delta")
    body = "".join(
        f'[[entries]]\nkeys = ["{k}"]\ncontent = "{k}-content"\npriority = {p}\n\n'
        for k, p in [("alpha", 4), ("beta", 3), ("gamma", 2), ("delta", 1)]
    )
    book = _write_lorebook(tmp_path / "lorebook.toml", body)
    layer = LorebookLayer(store=store, path=book, max_entries=2)
    out = await layer.build(_ctx())
    assert "alpha-content" in out
    assert "beta-content" in out
    assert "gamma-content" not in out
    assert "delta-content" not in out


@pytest.mark.asyncio
async def test_max_tokens_truncates_block(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "alpha beta")
    long = "x" * 4000
    body = (
        f'[[entries]]\nkeys = ["alpha"]\ncontent = "{long}"\npriority = 2\n\n'
        f'[[entries]]\nkeys = ["beta"]\ncontent = "second entry"\npriority = 1\n'
    )
    book = _write_lorebook(tmp_path / "lorebook.toml", body)
    layer = LorebookLayer(store=store, path=book, max_tokens=20)
    out = await layer.build(_ctx())
    # 20 tokens ≈ 80 chars; the rendered block should be much shorter
    # than the raw 4000-char first entry.
    assert len(out) < 200


@pytest.mark.asyncio
async def test_only_recent_window_scanned(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    # Old turn far behind the window — should NOT count.
    _seed(store, "I love Paris")
    for i in range(50):
        _seed(store, f"filler turn {i}")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "Paris info."\n',
    )
    layer = LorebookLayer(store=store, path=book, recent_window=10)
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_empty_keys_list_never_fires(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "anything goes")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = []\ncontent = "should not appear"\n',
    )
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert not out


def test_invalidation_key_changes_when_match_set_changes(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "city"\n',
    )
    layer = LorebookLayer(store=store, path=book)
    k0 = layer.invalidation_key(_ctx())
    _seed(store, "tell me about paris")
    k1 = layer.invalidation_key(_ctx())
    assert k0 != k1


def test_invalidation_key_changes_when_file_changes(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "paris")
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "v1"\n',
    )
    layer = LorebookLayer(store=store, path=book)
    k0 = layer.invalidation_key(_ctx())
    _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "v2"\n',
    )
    k1 = layer.invalidation_key(_ctx())
    assert k0 != k1


@pytest.mark.asyncio
async def test_malformed_toml_yields_empty_block(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "anything")
    book = _write_lorebook(tmp_path / "lorebook.toml", "this is = not valid [[ toml")
    layer = LorebookLayer(store=store, path=book)
    out = await layer.build(_ctx())
    assert not out


@pytest.mark.asyncio
async def test_channel_scope_uses_active_channel(tmp_path: Path) -> None:
    store = HistoryStore(":memory:")
    _seed(store, "paris", channel_id=1)
    _seed(store, "no key here", channel_id=2)
    book = _write_lorebook(
        tmp_path / "lorebook.toml",
        '[[entries]]\nkeys = ["paris"]\ncontent = "city"\n',
    )
    layer = LorebookLayer(store=store, path=book)
    # Channel 2 has no matching turn — layer should opt out.
    out = await layer.build(_ctx(channel_id=2))
    assert not out
    # Channel 1 has the match — layer fires.
    out = await layer.build(_ctx(channel_id=1))
    assert "city" in out
