"""Lock-serialised Turso connection wrapper.

pyturso 0.5.1 declares ``threadsafety=1`` (connections must not be
shared across threads) *and* has a class of cross-connection schema
cache bugs on Windows: a worker thread's freshly-opened
``turso.Connection`` doesn't see tables / indexes the main thread
just committed, surfacing as ``Parse error: no such table: …``
inside :class:`AsyncHistoryStore`'s executor.

To sidestep both, every :class:`TursoConnection` instance funnels
*all* calls through one shared ``turso.Connection`` guarded by a
lock — for file-backed DBs as well as ``:memory:``. The lock keeps
us within Turso's ``threadsafety=1`` contract (only one thread
touches the connection at a time) while guaranteeing every caller
sees the same schema cache. Throughput is fine: DB calls are short
and infrequent, and :class:`AsyncHistoryStore` still hops them off
the event loop.

Callers get a ``sqlite3.Connection``-compatible API so
:class:`HistoryStore` keeps its existing call sites
(``self._conn.execute(...)`` etc.). All connections use
``conn.row_factory = turso.Row`` so columns are accessible by name
(``row["fact_id"]``) — drop-in for the prior ``sqlite3.Row`` setup.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import turso

PathLike = str | Path
SqlParams = Sequence[Any] | Mapping[str, Any]
TraceCallback = Callable[[str], object]

_EXPERIMENTAL_FEATURES = "index_method"


class TursoConnection:
    """One shared Turso connection per instance, serialised by a lock."""

    def __init__(self, path: PathLike) -> None:
        raw = str(path)
        self._path = raw
        self._lock = threading.Lock()
        self._closed = False
        self._trace_callback: TraceCallback | None = None
        self._shared: turso.Connection = self._open_new()

    def _open_new(self) -> turso.Connection:
        conn = turso.connect(self._path, experimental_features=_EXPERIMENTAL_FEATURES)
        conn.row_factory = turso.Row
        return conn

    def _conn(self) -> turso.Connection:
        if self._closed:
            msg = "TursoConnection is closed"
            raise RuntimeError(msg)
        return self._shared

    # ------------------------------------------------------------------
    # sqlite3.Connection passthrough surface
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: SqlParams = ()) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        with self._lock:
            return self._conn().execute(sql, params)

    def executemany(self, sql: str, rows: Sequence[SqlParams]) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        with self._lock:
            return self._conn().executemany(sql, rows)

    def executescript(self, script: str) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(script)
        with self._lock:
            return self._conn().executescript(script)

    def set_trace_callback(self, callback: TraceCallback | None) -> None:
        """Install a SQL trace hook for ``execute``/``executemany``/``executescript``.

        Each call forwards its raw SQL string to *callback* before
        issuing it. ``None`` disables. Mirrors
        ``sqlite3.Connection.set_trace_callback`` enough to satisfy
        existing query-count tests.
        """
        self._trace_callback = callback

    def commit(self) -> None:
        with self._lock:
            self._conn().commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn().rollback()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                self._shared.close()
