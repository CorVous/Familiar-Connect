"""Dedicated-thread Turso connection wrapper.

pyturso 0.5.1 declares ``threadsafety=1`` (connections must not be
shared across threads). On Windows, even *serialised* cross-thread
access surfaces internal schema-cache bugs: a worker thread's call
into a connection opened on the main thread can fail with
``Parse error: no such table: …`` after that table was just created
and observed via ``sqlite_master`` on the opening thread. We treat
this as evidence of pyturso state that's affine to the OS thread that
first touched the connection.

Sidestep the entire class of bugs by routing every turso call —
``connect`` itself, ``execute`` family, cursor fetches, ``commit``,
``close`` — through one dedicated ``ThreadPoolExecutor`` with
``max_workers=1``. The connection is opened on that thread and never
seen from any other. Callers from any thread can use the wrapper
safely; calls are serialised onto the executor.

The wrapper keeps a ``sqlite3.Connection``-compatible API so
:class:`HistoryStore` keeps its existing call sites
(``self._conn.execute(...).fetchall()`` etc.). All connections use
``conn.row_factory = turso.Row`` so columns are accessible by name
(``row["fact_id"]``) — drop-in for the prior ``sqlite3.Row`` setup.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import turso

PathLike = str | Path
SqlParams = Sequence[Any] | Mapping[str, Any]
TraceCallback = Callable[[str], object]

_EXPERIMENTAL_FEATURES = "index_method"


class TursoConnection:
    """Single-thread Turso connection facade.

    Every turso call lands on one dedicated OS thread, owned by an
    internal ``max_workers=1`` ``ThreadPoolExecutor``. Pass
    ``":memory:"`` for an ephemeral DB or a file path for an
    on-disk DB; behaviour is identical from the caller's side.
    """

    def __init__(self, path: PathLike) -> None:
        self._path = str(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turso")
        self._closed = False
        self._trace_callback: TraceCallback | None = None
        self._shared: turso.Connection = self._run(self._open_new)

    def _open_new(self) -> turso.Connection:
        conn = turso.connect(self._path, experimental_features=_EXPERIMENTAL_FEATURES)
        conn.row_factory = turso.Row
        return conn

    def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if self._closed:
            msg = "TursoConnection is closed"
            raise RuntimeError(msg)
        return self._executor.submit(fn, *args, **kwargs).result()

    def _conn(self) -> turso.Connection:
        """Return the underlying ``turso.Connection`` (test/diagnostic only).

        Production callers should use the ``execute`` family — direct
        access bypasses the single-thread guarantee.
        """
        if self._closed:
            msg = "TursoConnection is closed"
            raise RuntimeError(msg)
        return self._shared

    # ------------------------------------------------------------------
    # sqlite3.Connection passthrough surface (executor-dispatched)
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: SqlParams = ()) -> _Cursor:
        return self._run(self._do_execute, sql, params)

    def _do_execute(self, sql: str, params: SqlParams) -> _Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        return _Cursor(self, self._shared.execute(sql, params))

    def executemany(self, sql: str, rows: Sequence[SqlParams]) -> _Cursor:
        return self._run(self._do_executemany, sql, rows)

    def _do_executemany(self, sql: str, rows: Sequence[SqlParams]) -> _Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(sql)
        return _Cursor(self, self._shared.executemany(sql, rows))

    def executescript(self, script: str) -> _Cursor:
        return self._run(self._do_executescript, script)

    def _do_executescript(self, script: str) -> _Cursor:
        cb = self._trace_callback
        if cb is not None:
            cb(script)
        return _Cursor(self, self._shared.executescript(script))

    def set_trace_callback(self, callback: TraceCallback | None) -> None:
        """Install a SQL trace hook for ``execute``/``executemany``/``executescript``.

        The callback is invoked from the executor thread, just before
        the underlying pyturso call. ``None`` disables. Mirrors
        ``sqlite3.Connection.set_trace_callback`` enough to satisfy
        existing query-count tests.
        """
        self._trace_callback = callback

    def commit(self) -> None:
        self._run(self._shared.commit)

    def rollback(self) -> None:
        self._run(self._shared.rollback)

    def reopen(self) -> None:
        """Close the underlying connection and open a fresh one.

        pyturso 0.5.1 on Windows can hold a stale schema cache even
        on the same connection — a ``SELECT`` from a just-created
        table raises ``Parse error: no such table: …``. Reopening
        forces pyturso to re-read ``sqlite_master`` from disk into a
        fresh cache. Both close + open happen on the executor thread
        so the new connection is also thread-affine to it. Use after
        schema setup / migrations; never mid-transaction.
        """
        self._run(self._do_reopen)

    def _do_reopen(self) -> None:
        with contextlib.suppress(Exception):
            self._shared.close()
        self._shared = self._open_new()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._do_close).result()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True

    def _do_close(self) -> None:
        with contextlib.suppress(Exception):
            self._shared.close()


class _Cursor:
    """Cursor proxy — fetches dispatch onto the owner's executor thread."""

    def __init__(self, owner: TursoConnection, raw: turso.Cursor) -> None:
        self._owner = owner
        self._raw = raw

    def fetchone(self) -> Any:  # noqa: ANN401
        return self._owner._run(self._raw.fetchone)  # noqa: SLF001

    def fetchall(self) -> list[Any]:
        return self._owner._run(self._raw.fetchall)  # noqa: SLF001

    def fetchmany(self, size: int | None = None) -> list[Any]:
        if size is None:
            return self._owner._run(self._raw.fetchmany)  # noqa: SLF001
        return self._owner._run(self._raw.fetchmany, size)  # noqa: SLF001

    @property
    def lastrowid(self) -> int | None:
        return self._owner._run(lambda: self._raw.lastrowid)  # noqa: SLF001

    @property
    def rowcount(self) -> int:
        return self._owner._run(lambda: self._raw.rowcount)  # noqa: SLF001
