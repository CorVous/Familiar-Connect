"""Thread-local Turso connection wrapper.

Turso (pyturso) declares ``threadsafety=1``: connections must not be
shared across threads. With :class:`AsyncHistoryStore` dispatching
work to a multi-worker executor, each worker thread needs its own
connection. This wrapper handles that transparently behind a
``sqlite3.Connection``-compatible API so :class:`HistoryStore` keeps
its existing call sites (``self._conn.execute(...)`` etc.).

Special case: ``:memory:`` databases are per-connection — opening a
fresh connection from a second thread yields a *different* empty DB,
which would break tests. For ``:memory:`` we fall back to one shared
connection guarded by a lock. Tests typically run with
``max_workers=1`` so contention is irrelevant; production paths use
file-backed DBs and get true per-thread connections.

All connections use ``conn.row_factory = turso.Row`` so callers can
access columns by name (``row["fact_id"]``) — drop-in for the prior
``sqlite3.Row`` setup.
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
    """Per-thread Turso connections under one Connection-like facade.

    Pass ``":memory:"`` for an in-process DB shared across threads via
    a lock (test-only); pass a file path for one connection per
    worker thread.
    """

    def __init__(self, path: PathLike) -> None:
        raw = str(path)
        self._path = raw
        self._is_memory = raw == ":memory:"
        self._local = threading.local()
        self._all: list[turso.Connection] = []
        self._lock = threading.Lock()
        self._closed = False
        self._trace_callback: TraceCallback | None = None
        if self._is_memory:
            # one shared in-memory DB, locked on every call
            self._shared: turso.Connection | None = self._open_new()
        else:
            self._shared = None

    def _open_new(self) -> turso.Connection:
        conn = turso.connect(self._path, experimental_features=_EXPERIMENTAL_FEATURES)
        conn.row_factory = turso.Row
        with self._lock:
            self._all.append(conn)
        return conn

    def _conn(self) -> turso.Connection:
        if self._closed:
            msg = "TursoConnection is closed"
            raise RuntimeError(msg)
        if self._is_memory:
            shared = self._shared
            if shared is None:
                msg = "shared in-memory connection was None"
                raise RuntimeError(msg)
            return shared
        existing = getattr(self._local, "conn", None)
        if existing is None:
            existing = self._open_new()
            self._local.conn = existing
        return existing

    # ------------------------------------------------------------------
    # sqlite3.Connection passthrough surface
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: SqlParams = ()) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        if self._is_memory:
            with self._lock:
                return self._conn().execute(sql, params)
        return self._conn().execute(sql, params)

    def executemany(self, sql: str, rows: Sequence[SqlParams]) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        if self._is_memory:
            with self._lock:
                return self._conn().executemany(sql, rows)
        return self._conn().executemany(sql, rows)

    def executescript(self, script: str) -> turso.Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(script)
        if self._is_memory:
            with self._lock:
                return self._conn().executescript(script)
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
        if self._is_memory:
            with self._lock:
                self._conn().commit()
        else:
            self._conn().commit()

    def rollback(self) -> None:
        if self._is_memory:
            with self._lock:
                self._conn().rollback()
        else:
            self._conn().rollback()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for conn in self._all:
                with contextlib.suppress(Exception):
                    conn.close()
            self._all.clear()
            self._shared = None
