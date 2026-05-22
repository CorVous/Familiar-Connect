"""Async proxy for :class:`HistoryStore`.

Calls dispatch to store's executor — Turso/tantivy never run on
event-loop thread. ``max_workers=4``: Turso supports concurrent
connections (MVCC, one per worker via :class:`TursoConnection`),
tantivy is thread-safe, so queries run in parallel instead of
queueing behind slow FTS.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore


class AsyncHistoryStore:
    """Async proxy for :class:`HistoryStore`.

    Dispatches every call to store's executor — DB IO never blocks
    event loop. 4-worker pool + Turso MVCC lets concurrent callers
    (live conversation + embedding/reflection workers) run without
    queueing.
    """

    def __init__(self, store: HistoryStore) -> None:
        self._inner = store

    @property
    def sync(self) -> HistoryStore:
        """Raw sync store — for callers that can't await."""
        return self._inner

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr  # type: ignore[return-value]

        @functools.wraps(attr)
        async def _call(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._inner._executor,  # noqa: SLF001
                functools.partial(attr, *args, **kwargs),
            )

        return _call

    def close(self) -> None:
        self._inner.close()
