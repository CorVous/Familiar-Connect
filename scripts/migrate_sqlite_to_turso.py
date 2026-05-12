r"""One-shot migration: stdlib sqlite3 ``history.db`` → Turso ``history.db``.

Iterates ``data/familiars/*/`` (or paths passed on the CLI), detects
real SQLite files (magic header ``SQLite format 3\000``), copies the
seven source-of-truth tables into a fresh Turso DB, and atomically
swaps the file:

    history.db                → history.db.legacy-<UTC-ISO>
    history.db.turso (staged) → history.db

Projections (summaries, reflections, dossiers,
cross_context_summaries, people_dossiers, memory_writer_watermark)
are intentionally not copied — they rebuild from ``turns``/``facts``
on next :class:`HistoryStore.__init__` via existing watermark workers.

Tantivy FTS indexes under ``fts/`` rebuild on next store init too —
:meth:`HistoryStore._reindex_if_empty` handles that.

Usage:
    uv run python scripts/migrate_sqlite_to_turso.py [path ...]

Default scan root: ``data/familiars/`` next to the repo root.
Idempotent: rerun safely; skips already-migrated files (turso header
or legacy backup present).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

# Tables to copy verbatim. Order matters where FKs-by-convention exist
# (facts before fact_embeddings).
SOURCE_TABLES: tuple[str, ...] = (
    "turns",
    "facts",
    "accounts",
    "account_guild_nicks",
    "message_reactions",
    "turn_mentions",
    "fact_embeddings",
)

SQLITE_MAGIC = b"SQLite format 3\x00"


_SQLITE_HEADER_LEN = 16


def _is_sqlite_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < _SQLITE_HEADER_LEN:
        return False
    with path.open("rb") as fh:
        return fh.read(_SQLITE_HEADER_LEN) == SQLITE_MAGIC


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


def _migrate_one(db_path: Path, *, dry_run: bool) -> bool:
    """Migrate one familiar DB; return True if anything changed."""
    if not _is_sqlite_file(db_path):
        print(f"  skip {db_path} (not stdlib sqlite — already turso?)")
        return False

    legacy_glob = sorted(db_path.parent.glob(f"{db_path.name}.legacy-*"))
    if legacy_glob:
        print(f"  skip {db_path} (legacy backup exists: {legacy_glob[0].name})")
        return False

    # late import — only needed when actually migrating; keeps the
    # script invocable on hosts that have stdlib sqlite3 but not turso
    from familiar_connect.history.store import HistoryStore

    staging = db_path.with_suffix(db_path.suffix + ".turso")
    if staging.exists():
        staging.unlink()

    counts: dict[str, tuple[int, int]] = {}
    old = sqlite3.connect(db_path)
    old.row_factory = sqlite3.Row

    # HistoryStore creates the new file, applies _SCHEMA, runs migrations
    store = HistoryStore(staging)
    new_conn = store._conn
    try:
        for table in SOURCE_TABLES:
            if not _table_exists(old, table):
                continue
            cols = _table_columns(old, table)
            placeholders = ",".join("?" for _ in cols)
            col_list = ",".join(cols)
            rows = old.execute(f"SELECT {col_list} FROM {table}").fetchall()  # noqa: S608
            n_src = len(rows)
            if n_src == 0:
                counts[table] = (0, 0)
                continue
            for row in rows:
                new_conn.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",  # noqa: S608
                    tuple(row),
                )
            new_conn.commit()
            n_dst = new_conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
            counts[table] = (n_src, n_dst)
            if n_src != n_dst:
                msg = f"row count mismatch in {table}: src={n_src} dst={n_dst}"
                raise RuntimeError(msg)
    finally:
        store.close()
        old.close()

    print(f"  {db_path}")
    for tbl, (src, dst) in counts.items():
        print(f"    {tbl}: {src} → {dst}")

    if dry_run:
        print(f"  DRY RUN: leaving {staging} in place; original untouched")
        return False

    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.legacy-{ts}")
    shutil.move(str(db_path), str(backup))
    shutil.move(str(staging), str(db_path))
    # Tidy up the WAL file that lived alongside the staging DB — its
    # name no longer matches now that the DB is renamed. Turso will
    # create a fresh ``<db>-wal`` on the next open.
    stale_wal = staging.with_name(staging.name + "-wal")
    if stale_wal.exists():
        stale_wal.unlink()
    print(f"  → {backup.name} (backup), {db_path.name} (now turso)")
    return True


def _discover(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        if not root.is_dir():
            continue
        # familiar layout: data/familiars/<id>/history.db
        for db in root.glob("*/history.db"):
            out.append(db)  # noqa: PERF402
        # also accept a single familiar dir passed directly
        direct = root / "history.db"
        if direct.exists() and direct not in out:
            out.append(direct)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="DB files or familiar root dirs (default: ./data/familiars)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="copy + verify counts, but don't replace the original",
    )
    args = ap.parse_args(argv)

    roots: list[Path] = list(args.paths) if args.paths else [Path("data/familiars")]
    targets = _discover(roots)
    if not targets:
        print(f"no history.db files found under {[str(r) for r in roots]}")
        return 1

    print(f"found {len(targets)} candidate DB(s)")
    migrated = 0
    for db in targets:
        try:
            if _migrate_one(db, dry_run=args.dry_run):
                migrated += 1
        except Exception as exc:  # noqa: BLE001 — single-DB failure shouldn't abort the batch
            print(f"  FAILED {db}: {exc}")

    print(f"\ndone: migrated {migrated} / {len(targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
