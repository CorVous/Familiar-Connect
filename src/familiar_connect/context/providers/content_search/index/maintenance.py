"""Write-hook decorator around ``MemoryStore``.

Keeps ``memory/store.py`` index-agnostic while still emitting
events a background indexer can consume. ``IndexingMemoryStore``
wraps a store and records ``rel_path`` in a FIFO queue on every
successful ``write_file`` / ``append_file`` call. The companion
async worker drains the queue and calls
``EmbeddingIndex.build_if_stale`` (or its per-file variant) to keep
the cache fresh.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.memory.store import (
        GrepHit,
        MemoryEntry,
        MemoryStore,
    )


class IndexingMemoryStore:
    """MemoryStore decorator that enqueues rel_paths after writes.

    Reads are forwarded verbatim; writes go through the inner store
    first (so failures preserve the "no enqueue without commit"
    invariant) and are then recorded.
    """

    def __init__(self, inner: MemoryStore) -> None:
        self._inner = inner
        self._pending: deque[str] = deque()
        # Lazily-created on first wait — asyncio.Event wants an
        # ambient loop, which we don't always have at construction time.
        self._event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Read delegates
    # ------------------------------------------------------------------

    @property
    def root(self):  # noqa: ANN201 — mirrors MemoryStore.root property
        return self._inner.root

    def list_dir(self, rel_path: str = "") -> list[MemoryEntry]:
        return self._inner.list_dir(rel_path)

    def read_file(self, rel_path: str) -> str:
        return self._inner.read_file(rel_path)

    def glob(self, pattern: str) -> list[str]:
        return self._inner.glob(pattern)

    def grep(
        self,
        pattern: str,
        rel_path: str = "",
        *,
        case_insensitive: bool = True,
    ) -> list[GrepHit]:
        return self._inner.grep(pattern, rel_path, case_insensitive=case_insensitive)

    @property
    def audit_entries(self):  # noqa: ANN201 — mirrors MemoryStore
        return self._inner.audit_entries

    # ------------------------------------------------------------------
    # Write intercepts
    # ------------------------------------------------------------------

    def write_file(
        self,
        rel_path: str,
        content: str,
        *,
        source: str = "unknown",
    ) -> None:
        self._inner.write_file(rel_path, content, source=source)
        self._enqueue(rel_path)

    def append_file(
        self,
        rel_path: str,
        content: str,
        *,
        source: str = "unknown",
    ) -> None:
        self._inner.append_file(rel_path, content, source=source)
        self._enqueue(rel_path)

    # ------------------------------------------------------------------
    # Pending-write queue API (consumed by the background indexer)
    # ------------------------------------------------------------------

    def drain_pending(self) -> list[str]:
        """Pop all pending rel_paths in FIFO order."""
        out = list(self._pending)
        self._pending.clear()
        # clear event too so the next wait blocks until a fresh write
        if self._event is not None and not self._pending:
            self._event.clear()
        return out

    async def wait_for_writes(self) -> None:
        """Block until at least one write is pending, then return."""
        if self._event is None:
            self._event = asyncio.Event()
            if self._pending:
                self._event.set()
        await self._event.wait()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, rel_path: str) -> None:
        self._pending.append(rel_path)
        if self._event is not None:
            self._event.set()

    # ------------------------------------------------------------------
    # Inner store accessor (for code that genuinely needs the raw store)
    # ------------------------------------------------------------------

    @property
    def inner(self) -> MemoryStore:
        return self._inner
