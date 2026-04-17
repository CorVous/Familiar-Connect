"""Tests for LastContextCache and render_markdown."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect.context.last_context import (
    LastContextCache,
    LastContextEntry,
    render_markdown,
)
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from pathlib import Path


def _entry(modality: str = "text") -> LastContextEntry:
    return LastContextEntry(
        messages=(
            Message(role="system", content="You are a cat."),
            Message(role="user", content="hi", name="Alice"),
            Message(role="assistant", content="meow"),
        ),
        captured_at=datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC),
        modality=modality,
    )


class TestLastContextCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        entry = _entry()
        cache.put(channel_id=1, messages=entry.messages, modality=entry.modality)

        # Fresh instance — no in-memory state
        cache2 = LastContextCache(channels_root=tmp_path)
        got = cache2.get(channel_id=1)
        assert got is not None
        assert got.messages == entry.messages
        assert got.modality == "text"

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        assert cache.get(channel_id=999) is None

    def test_per_channel_isolation(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        msgs_a = (Message(role="system", content="channel A"),)
        msgs_b = (Message(role="system", content="channel B"),)
        cache.put(channel_id=1, messages=msgs_a, modality="text")
        cache.put(channel_id=2, messages=msgs_b, modality="voice")

        got_a = cache.get(channel_id=1)
        got_b = cache.get(channel_id=2)
        assert got_a is not None
        assert got_b is not None
        assert got_a.messages == msgs_a
        assert got_b.messages == msgs_b

    def test_file_is_sibling_json(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        cache.put(channel_id=42, messages=(), modality="text")
        assert (tmp_path / "42.last-context.json").exists()


class TestRenderMarkdown:
    def test_headings_and_separators(self) -> None:
        md = render_markdown(_entry())
        assert "## [0] system" in md
        assert "## [1] user (Alice)" in md
        assert "## [2] assistant" in md
        assert "---" in md

    def test_content_is_unredacted(self) -> None:
        md = render_markdown(_entry())
        assert "You are a cat." in md
        assert "hi" in md
        assert "meow" in md

    def test_header_contains_modality_and_count(self) -> None:
        md = render_markdown(_entry("voice-regen"))
        assert "modality=voice-regen" in md
        assert "3 messages" in md
