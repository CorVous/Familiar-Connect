"""Async proxy for :class:`HistoryStore`.

All method calls are dispatched to the store's single-threaded executor so
SQLite never runs on the event-loop thread.  ``max_workers=1`` on the store's
own executor guarantees serial access without an explicit Python lock.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore


class AsyncHistoryStore:
    """Async proxy for :class:`HistoryStore`.

    Dispatches every call to a dedicated background thread, keeping SQLite off
    the event loop.  ``max_workers=1`` on the store's own executor serialises
    all access so no extra Python lock is needed.
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
