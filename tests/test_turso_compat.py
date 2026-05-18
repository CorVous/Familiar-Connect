"""Cross-thread invariants for :class:`TursoConnection`.

pyturso 0.5.1 has cross-connection schema cache bugs (a worker's
fresh connection doesn't see DDL committed by the main thread).
``TursoConnection`` sidesteps that by funnelling every call through
one shared ``turso.Connection`` guarded by a lock — for file-backed
DBs as well as ``:memory:``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from familiar_connect.history.turso_compat import TursoConnection

if TYPE_CHECKING:
    from pathlib import Path


def test_file_backed_shares_one_connection_across_threads(tmp_path: Path) -> None:
    """Worker threads must see DDL committed on the main thread."""
    db_path = tmp_path / "shared.db"
    tc = TursoConnection(db_path)
    try:
        tc.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        tc.execute("INSERT INTO t VALUES (1)")
        tc.commit()

        observed: list[int] = []

        def worker() -> None:
            row = tc.execute("SELECT id FROM t").fetchone()
            observed.append(int(row["id"]))

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        assert observed == [1]
    finally:
        tc.close()


def test_in_memory_shares_one_connection_across_threads() -> None:
    """``:memory:`` keeps its existing single-shared-connection behaviour."""
    tc = TursoConnection(":memory:")
    try:
        tc.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        tc.execute("INSERT INTO t VALUES (1)")
        tc.commit()

        observed: list[int] = []

        def worker() -> None:
            row = tc.execute("SELECT id FROM t").fetchone()
            observed.append(int(row["id"]))

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        assert observed == [1]
    finally:
        tc.close()


def test_file_backed_returns_same_connection_object(tmp_path: Path) -> None:
    """Structural guard: only one underlying ``turso.Connection`` per DB.

    Cross-connection schema cache staleness on pyturso 0.5.1 means we
    can't safely open a second connection — every call must route
    through the same object so DDL is always visible.
    """
    db_path = tmp_path / "single.db"
    tc = TursoConnection(db_path)
    try:
        main_conn = tc._conn()

        captured: list[object] = []

        def worker() -> None:
            captured.append(tc._conn())

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        assert captured == [main_conn]
    finally:
        tc.close()


def test_all_turso_calls_route_through_one_os_thread(tmp_path: Path) -> None:
    """Pin every turso call to one dedicated OS thread.

    pyturso 0.5.1 has thread-affine internal state; route everything
    through one OS thread regardless of caller. The trace callback
    fires inside the executor, so its observed ``threading.get_ident()``
    reveals which thread actually invoked pyturso. Calls from the main
    thread *and* a worker thread must surface the same identifier.
    """
    tc = TursoConnection(tmp_path / "thread.db")
    seen: set[int] = set()
    tc.set_trace_callback(lambda _sql: seen.add(threading.get_ident()))
    try:
        tc.execute("CREATE TABLE t (id INTEGER)")

        def worker() -> None:
            tc.execute("INSERT INTO t VALUES (1)")
            tc.execute("SELECT id FROM t").fetchall()

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        assert len(seen) == 1
        assert threading.get_ident() not in seen
    finally:
        tc.close()


def test_cursor_methods_work_from_worker_thread(tmp_path: Path) -> None:
    """Cursor proxies must dispatch fetchone/fetchall/lastrowid/rowcount."""
    tc = TursoConnection(tmp_path / "cur.db")
    try:
        tc.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
        cur = tc.execute("INSERT INTO t (v) VALUES ('a')")
        assert int(cur.lastrowid or 0) > 0

        results: dict[str, Any] = {}

        def worker() -> None:
            results["one"] = tc.execute("SELECT v FROM t").fetchone()
            results["all"] = tc.execute("SELECT v FROM t").fetchall()
            upd = tc.execute("UPDATE t SET v='b' WHERE v='a'")
            results["rowcount"] = upd.rowcount

        th = threading.Thread(target=worker)
        th.start()
        th.join()

        assert results["one"]["v"] == "a"
        assert [r["v"] for r in results["all"]] == ["a"]
        assert results["rowcount"] == 1
    finally:
        tc.close()


def test_reopen_swaps_underlying_connection_and_preserves_data(
    tmp_path: Path,
) -> None:
    """``reopen()`` discards stale schema cache by opening a fresh connection.

    pyturso 0.5.1 on Windows can hold a stale schema cache *on the same
    connection* even after ``CREATE TABLE`` + ``commit``. Closing and
    re-opening forces a fresh read of ``sqlite_master`` from disk.
    """
    db_path = tmp_path / "reopen.db"
    tc = TursoConnection(db_path)
    try:
        tc.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        tc.execute("INSERT INTO t VALUES (42)")
        tc.commit()

        before = tc._conn()
        tc.reopen()
        after = tc._conn()

        assert before is not after
        row = tc.execute("SELECT id FROM t").fetchone()
        assert int(row["id"]) == 42
    finally:
        tc.close()
