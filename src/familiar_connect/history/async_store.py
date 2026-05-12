"""Async proxy for :class:`HistoryStore`.

All method calls are dispatched to the store's executor so Turso/tantivy
never run on the event-loop thread. The store uses ``max_workers=4`` —
Turso supports concurrent connections (MVCC, one per worker thread via
:class:`TursoConnection`) and tantivy is internally thread-safe, so
queries can run in parallel instead of queueing behind whichever request
just hit a slow FTS path.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore


class AsyncHistoryStore:
    """Async proxy for :class:`HistoryStore`.

    Dispatches every call to the store's executor so DB IO never blocks
    the event loop. The 4-worker pool plus Turso MVCC means concurrent
    callers (live conversation + embedding/reflection workers) stop
    queueing behind each other.
    """

    def __init__(self, store: HistoryStore) -> None:
        self._inner = store

    @property
    def sync(self) -> HistoryStore:
        """Raw synchronous store — for callers that cannot await."""
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
