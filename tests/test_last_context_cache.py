"""Tests for LastContextCache markdown-on-disk cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.context.last_context import LastContextCache
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from pathlib import Path


_MESSAGES = (
    Message(role="system", content="You are a cat."),
    Message(role="user", content="hi", name="Alice"),
    Message(role="assistant", content="meow"),
)


class TestLastContextCache:
    def test_round_trip_bytes(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        cache.put(channel_id=1, messages=_MESSAGES, modality="text")

        # Fresh instance — no in-memory state
        cache2 = LastContextCache(channels_root=tmp_path)
        raw = cache2.get(channel_id=1)
        assert raw is not None
        text = raw.decode("utf-8")
        assert "## [0] system" in text
        assert "## [1] user (Alice)" in text
        assert "## [2] assistant" in text
        assert "---" in text
        assert "modality=text" in text
        assert "3 messages" in text
        # Content unredacted
        assert "You are a cat." in text
        assert "meow" in text

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
        assert b"channel A" in got_a
        assert b"channel B" in got_b
        assert b"modality=text" in got_a
        assert b"modality=voice" in got_b

    def test_file_is_sibling_markdown(self, tmp_path: Path) -> None:
        cache = LastContextCache(channels_root=tmp_path)
        cache.put(channel_id=42, messages=_MESSAGES, modality="text")
        assert (tmp_path / "42.last-context.md").exists()
