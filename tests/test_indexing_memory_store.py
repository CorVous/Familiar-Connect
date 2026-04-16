"""Red-first tests for IndexingMemoryStore — write-hook decorator.

Wraps a plain MemoryStore, forwards all reads, and emits a rel_path
event on each write so a background worker can keep the embedding
index fresh. memory/store.py itself stays index-agnostic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.providers.content_search.index.maintenance import (
    IndexingMemoryStore,
)
from familiar_connect.memory.store import MemoryStore, MemoryStorePathError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def inner(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


class TestReadPassthrough:
    def test_read_file_delegates(self, inner: MemoryStore) -> None:
        inner.write_file("x.md", "hello")
        wrapped = IndexingMemoryStore(inner)
        assert wrapped.read_file("x.md") == "hello"

    def test_list_dir_delegates(self, inner: MemoryStore) -> None:
        inner.write_file("a.md", "x")
        wrapped = IndexingMemoryStore(inner)
        names = {e.name for e in wrapped.list_dir("")}
        assert "a.md" in names

    def test_glob_delegates(self, inner: MemoryStore) -> None:
        inner.write_file("people/alice.md", "x")
        wrapped = IndexingMemoryStore(inner)
        assert wrapped.glob("people/*.md") == ["people/alice.md"]

    def test_grep_delegates(self, inner: MemoryStore) -> None:
        inner.write_file("notes.md", "ska is a genre")
        wrapped = IndexingMemoryStore(inner)
        hits = wrapped.grep("ska")
        assert len(hits) == 1
        assert hits[0].rel_path == "notes.md"

    def test_root_delegates(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)
        assert wrapped.root == inner.root


class TestWriteHook:
    def test_write_file_forwards_and_enqueues(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)
        wrapped.write_file("people/alice.md", "hello")
        # actually written
        assert inner.read_file("people/alice.md") == "hello"
        # enqueued for indexing
        assert wrapped.drain_pending() == ["people/alice.md"]

    def test_append_file_forwards_and_enqueues(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)
        wrapped.write_file("note.md", "a")
        wrapped.drain_pending()  # clear
        wrapped.append_file("note.md", "b")
        assert inner.read_file("note.md") == "ab"
        assert wrapped.drain_pending() == ["note.md"]

    def test_drain_pending_is_fifo(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)
        wrapped.write_file("a.md", "1")
        wrapped.write_file("b.md", "2")
        wrapped.write_file("a.md", "3")  # re-write
        drained = wrapped.drain_pending()
        assert drained == ["a.md", "b.md", "a.md"]

    def test_drain_empty_returns_empty(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)
        assert wrapped.drain_pending() == []

    def test_failed_write_is_not_enqueued(self, inner: MemoryStore) -> None:
        """If the inner write raises, the path must NOT be enqueued."""
        wrapped = IndexingMemoryStore(inner)
        # path traversal → MemoryStorePathError
        with pytest.raises(MemoryStorePathError):
            wrapped.write_file("../escape.md", "bad")
        assert wrapped.drain_pending() == []


class TestAsyncWaitForWrites:
    @pytest.mark.asyncio
    async def test_wait_resolves_on_write(self, inner: MemoryStore) -> None:
        wrapped = IndexingMemoryStore(inner)

        async def writer() -> None:
            await asyncio.sleep(0)  # yield so the waiter registers
            wrapped.write_file("x.md", "hello")

        # wait_for_writes resolves as soon as the first write lands
        waiter_task = asyncio.create_task(wrapped.wait_for_writes())
        writer_task = asyncio.create_task(writer())
        await asyncio.wait_for(waiter_task, timeout=1.0)
        await writer_task
        assert wrapped.drain_pending() == ["x.md"]
