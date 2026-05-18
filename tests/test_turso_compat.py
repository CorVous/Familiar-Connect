"""Cross-thread invariants for :class:`TursoConnection`.

pyturso 0.5.1 has cross-connection schema cache bugs (a worker's
fresh connection doesn't see DDL committed by the main thread).
``TursoConnection`` sidesteps that by funnelling every call through
one shared ``turso.Connection`` guarded by a lock — for file-backed
DBs as well as ``:memory:``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

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
