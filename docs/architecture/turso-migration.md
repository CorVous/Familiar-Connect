# Turso migration

In May 2026 the history store moved from stdlib `sqlite3` to
[Turso](https://github.com/tursodatabase/turso) (Rust rewrite, beta)
plus [tantivy](https://github.com/quickwit-oss/tantivy-py) for
full-text search. This page captures what changed, why, and how to
roll back.

## Why

A bug in late April had a long FTS5 query hold the only `HistoryStore`
connection for 10+ seconds, starving the Discord heartbeat and every
other DB-touching coroutine. The point-fix was to dispatch SQLite
calls to a `ThreadPoolExecutor(max_workers=1)` so the event loop
stayed alive; the underlying contention — every DB op queueing
behind one connection — didn't go away.

Turso supports `BEGIN CONCURRENT` (MVCC) and is written in Rust, so:

- Multiple worker threads can hold concurrent connections without
  the "one connection, one lock" pattern.
- The Rust core releases the GIL during query execution, so true
  parallelism across worker threads becomes possible.
- Tantivy queries — which sat on the SQLite FTS5 hot path — now run
  outside the database entirely.

## What changed

**Relational storage:** stdlib `sqlite3` → `pyturso` (`turso.connect`).
The file at `data/familiars/<id>/history.db` is now a Turso database
(SQLite-compatible format).

**FTS:** SQLite FTS5 virtual tables + triggers → on-disk tantivy
indexes under `data/familiars/<id>/fts/turns/` and
`data/familiars/<id>/fts/facts/`. Writes go through
`HistoryStore.append_turn` / `append_fact` and synchronously upsert
into tantivy after the relational commit.

**Connection model:** single `sqlite3.Connection` shared across
threads via `check_same_thread=False` → per-thread Turso connections
managed by `TursoConnection` (`history/turso_compat.py`).
`:memory:` mode still uses one connection guarded by a lock because
each new Turso connection to `:memory:` gets a fresh empty DB.

**Executor pool:** `AsyncHistoryStore` bumped from `max_workers=1`
to `max_workers=4`. With per-thread connections + MVCC, queries
actually run in parallel instead of serialising.

**Search semantics:** the old `_FTS_STOPWORDS` Python-side filter
and `_build_fts_match` OR-joiner are gone. Tantivy's English
analyzer chain — lowercase → ascii-fold → custom stopword filter →
english stemmer — applies the same intent at both index and query
time. `café` / `cafe` still match; `fox` matches `foxes` via stem
instead of the old prefix-`*` hack. BM25 scores are now positive
(higher = better) instead of negative (lower = better); the fusion
code in `context/layers.py:_rerank_fact_candidates` was updated to
match.

## What didn't change

- Schema for `turns`, `facts`, `accounts`, etc. (`_SCHEMA` in
  `history/store.py`).
- Watermark-driven projections (`SummaryWorker`,
  `FactExtractor`, `ReflectionWorker`, `PeopleDossierWorker`,
  `FactSupersedeWorker`, `FactEmbeddingWorker`).
- BLOB-packed float32 embeddings in `fact_embeddings`. Cosine
  similarity is still computed in Python (`context/layers.py`).
- The public `HistoryStore` / `AsyncHistoryStore` API surface.

## Migrating an existing familiar

```bash
uv run python scripts/migrate_sqlite_to_turso.py
# or scope to specific dirs / files:
uv run python scripts/migrate_sqlite_to_turso.py data/familiars/aria
uv run python scripts/migrate_sqlite_to_turso.py --dry-run data/familiars
```

The script:

1. Detects real SQLite files by magic header (skips already-migrated
   Turso files and dirs with a `*.legacy-*` backup already present).
2. Opens the old DB read-only, copies 7 tables (`turns`, `facts`,
   `accounts`, `account_guild_nicks`, `message_reactions`,
   `turn_mentions`, `fact_embeddings`) row-for-row into a staging
   Turso DB.
3. Asserts row counts match per table; bails on mismatch.
4. Renames `history.db` → `history.db.legacy-<UTC-ISO>`, then
   renames the staging file into place.

Tantivy indexes auto-rebuild from `turns` / `facts` on the next
`HistoryStore.__init__` (see `_reindex_if_empty`).

Projection tables (`summaries`, `cross_context_summaries`,
`people_dossiers`, `reflections`, `memory_writer_watermark`) are
*not* copied — the watermark workers reconstruct them from the
source-of-truth rows on next run, which is by design.

## Rolling back

The migration leaves `history.db.legacy-<ts>` next to the new file.
To roll back:

1. Stop the bot.
2. Delete the new `history.db` and `data/familiars/<id>/fts/`.
3. Rename `history.db.legacy-<ts>` → `history.db`.
4. Downgrade the package or check out a pre-migration commit.

You'll lose any turns / facts written since the migration. (If
that matters, dump the post-migration Turso file's `turns` /
`facts` rows first.)

## Known beta caveats

- **pyturso 0.5.1 ships without FTS.** That's why FTS lives in
  tantivy. If a future wheel ships with `--features fts`, we can
  revisit consolidating.
- **No read-your-writes inside a transaction for Turso FTS** — not
  relevant here since we don't use Turso FTS.
- **`threadsafety=1`** — connections must not be shared across
  threads. `TursoConnection` handles per-thread connections; do
  not reach inside it.
- **`ALTER TABLE` can spuriously report "no such table"** on Windows
  even when `sqlite_master` and `PRAGMA table_info` agree the table
  exists. `HistoryStore._safe_add_column` swallows that parse error
  (and `duplicate column`); the post-migration `_SCHEMA` pass's
  `CREATE TABLE IF NOT EXISTS` is the safety net for genuinely
  missing tables.

## Recovering corrupt tantivy indexes

If FTS goes weird (parser errors, stale results), the index dir is
disposable:

```bash
rm -rf data/familiars/<id>/fts/
# next HistoryStore.__init__ rebuilds from turns/facts
```

The relational data is untouched and remains the source of truth.
