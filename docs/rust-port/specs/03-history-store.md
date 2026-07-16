# 03-history-store — port spec

Source modules: `src/familiar_connect/history/__init__.py` (docstring only, no
re-exports), `store.py` (3860 loc), `async_store.py`, `fts.py`,
`turso_compat.py`.
Reference docs: `docs/architecture/memory-strategies.md`,
`docs/architecture/overview.md` § "Turso history store".
Conformance oracle: `tests/test_history_store.py`, `test_attentional_store.py`,
`test_facts_store.py`, `test_history_fts.py`, `test_history_identity.py`,
`test_history_replies.py`, `test_history_alarms.py`,
`test_fact_embeddings_store.py`, `test_people_dossiers_store.py`,
`test_reflections_store.py`, `test_message_reactions.py`,
`test_turso_compat.py` (~5900 test loc total).

## Role

The one durable-state subsystem: a per-familiar SQLite-format database
(written through pyturso, Turso's SQLite-compatible Rust engine) holding the
append-only `turns` log — the source of truth — plus every watermarked
side-index projection (summaries, facts, fact embeddings, people dossiers,
reflections, alarms, activities, identity cache, reactions, focus/attention
state). Full-text search lives OUTSIDE the DB in two tantivy indexes
(`fts/turns/`, `fts/facts/`) kept write-synchronous with their tables. The
governing rule (memory-strategies.md): `turns` is truth; every side-index can
be deleted and rebuilt from it. The sync `HistoryStore` is wrapped by
`AsyncHistoryStore` (thread-pool facade) and every SQL statement is funneled
onto one dedicated OS thread by `TursoConnection`.

## Public API surface

None of these are Protocol/ABC seams — `HistoryStore` is a concrete class
consumed directly (tests build it with `":memory:"`). The only swap seams are
(a) `AsyncHistoryStore` duck-proxies *any* store-shaped object, and (b) tests
monkeypatch `FtsIndex._commit_writer` / `FtsIndex.add` / `turso.Cursor.execute`
— see Rust port notes.

### `TursoConnection(path: str | Path)` — turso_compat.py

`sqlite3.Connection`-compatible facade; every underlying call is submitted to
an internal `ThreadPoolExecutor(max_workers=1, thread_name_prefix="turso")`
and awaited synchronously (`submit(...).result()`).

- `execute(sql, params=()) -> _Cursor`, `executemany(sql, rows)`,
  `executescript(script)` — trace callback (if set) invoked with the SQL text
  on the executor thread *before* the pyturso call.
- `_Cursor` proxy: `fetchone()`, `fetchall()`, `fetchmany(size=None)`,
  `.lastrowid` (`int | None`), `.rowcount` (`int`) — all dispatched back onto
  the owner's executor thread.
- `commit()`, `rollback()`.
- `reopen()` — close + reopen the underlying connection on the executor thread
  (works around pyturso 0.5.1 stale-schema-cache bugs; call only between
  transactions). Data must survive; the connection object must be replaced.
- `set_trace_callback(cb | None)` — used by query-count tests.
- `close()` — idempotent; submits close, then `shutdown(wait=True)`. Any call
  after close raises `RuntimeError("TursoConnection is closed")`.
- Connection opened with `turso.connect(path, experimental_features="index_method")`
  and `row_factory = turso.Row` (name-addressable rows).
- `_conn()` — escape hatch returning the raw connection (tests only).

### `FtsIndex(path: PathLike | None)` — fts.py

One tantivy index over `(row_id: i64 stored/indexed/fast, content: text
unstored)` with a custom analyzer registered as `"familiar_en"`. `path=None`
→ in-memory (tests); otherwise directory is `mkdir -p`'d and opened with
`reuse=True`. One persistent writer (`heap_size=15_000_000, num_threads=1`).

- `add(row_id, content)` — upsert (delete-by-row_id then add), commit
  immediately, reload index so reads see it synchronously.
- `add_many(rows: list[(int, str)])` — bulk upsert, ONE commit (used by
  rebuild); empty list is a no-op.
- `delete(row_id)`, `clear()` (delete_all_documents) — both commit + reload.
- `search(query, *, limit) -> list[(row_id, bm25_score)]` — see behaviors
  17–21.
- `is_empty() -> bool` — `searcher.num_docs == 0`. (Currently no production
  caller; keep for migration tooling.)
- `close()` — best-effort final commit (exceptions suppressed), then
  `wait_merging_threads()`.
- Commit retry: transient-lock ValueErrors (message contains
  `"PermissionDenied"`, `"Access is denied"`, or `"os error 5"`) retried with
  sleeps `(0.05, 0.2, 0.5)` s, then one final attempt that propagates. Non-lock
  errors propagate immediately (no retry).

### `HistoryStore(db_path: str | Path)` — store.py

Sync API. `":memory:"` → in-memory turso + two in-memory FtsIndexes; file path
→ parent dirs created, FTS at `<db_dir>/fts/turns` and `<db_dir>/fts/facts`.
Constructor: `executescript(_SCHEMA)` + commit, then `_migrate()`, then open
both FTS indexes. Also owns `ThreadPoolExecutor(max_workers=4,
thread_name_prefix="db")` used only by `AsyncHistoryStore`.
`close()` — `executor.shutdown(wait=False)`, then fts_turns.close, finally
fts_facts.close, finally conn.close (nested try/finally: later closers run
even if earlier ones raise).

Module constant: `FOCUS_STREAM_CHANNEL_ID = -1` (reserved channel id for the
per-familiar focus-stream summary; real channels are large positive ids;
`0` is the channel-less bucket). Imported by 05/06/07.

Method groups (all keyword-only args unless noted; full query shapes in
Behaviors / Data formats):

Turns: `append_turn(...) -> HistoryTurn`, `stage_turn(...) -> HistoryTurn`
(= append_turn with `consumed=False`; note it has NO `pings_bot` param —
staged ping turns are written via `append_turn(consumed=False, pings_bot=True)`),
`lookup_turn_by_platform_message_id -> HistoryTurn | None`,
`update_turn_content_by_message_id -> None` (silent no-op when unmatched;
re-indexes FTS for each matched row id),
`turns_by_ids -> list[HistoryTurn]`, `recent(familiar_id, channel_id, limit,
mode=None, before_id=None)`, `turns_around(familiar_id, channel_id, turn_id,
before=5, after=5)`, `recent_distinct_authors -> list[Author]`,
`older_than(familiar_id, max_id, channel_id=None, limit=10_000)`,
`latest_id(familiar_id, channel_id=None) -> int | None`,
`count(familiar_id, channel_id=None) -> int`,
`turns_in_id_range(familiar_id, min_id_exclusive, max_id_inclusive,
channel_id=None)`, `all_channel_ids -> set[int]`,
`distinct_other_channels(familiar_id, exclude_channel_id) ->
list[OtherChannelInfo]`, `latest_id_at_or_before(familiar_id, ts) -> int | None`.

Mentions: `record_mentions(turn_id, canonical_keys)` (idempotent),
`mentions_for_turn(turn_id) -> tuple[str, ...]` (sorted asc).

Reactions: `set_reaction`, `bump_reaction`, `clear_reactions(emoji=None)`,
`reactions_for_messages -> dict[str, tuple[(emoji, count), ...]]`.

Summaries: `get_summary(familiar_id, channel_id=0) -> SummaryEntry | None`,
`put_summary(familiar_id, last_summarised_id, summary_text, channel_id=0,
last_consumed_at=None)`.

Watermarks: `get_writer_watermark / put_writer_watermark` (memory writer),
`turns_since_watermark(familiar_id, limit=10_000)`,
`get_sleep_watermark / advance_sleep_watermark(last_fact_id=None,
last_turn_id=None)` (partial-axis update),
`latest_reflection_watermarks -> (int, int)` / `set_reflection_watermark`,
`get_digest_watermark / set_digest_watermark`,
`get_archive_watermark / set_archive_watermark / set_archive_watermark_all`.

Dossiers/identity: `get_people_dossier / put_people_dossier`,
`subjects_with_facts -> dict[str, int]`, `facts_for_subject(...,
min_id_exclusive=0, include_superseded=False, as_of=None)`,
`upsert_account(author: Author)` (positional arg), `get_account_profile ->
AccountProfile | None`, `upsert_guild_nick`, `resolve_label(canonical_key,
guild_id, familiar_id=None) -> str` (never empty), `latest_author_for ->
Author | None`.

FTS-backed reads: `search_turns(familiar_id, query, limit, channel_id=None,
max_id=None)`, `rebuild_fts()`, `latest_fts_id(familiar_id) -> int` (actually
`MAX(turns.id)` from SQL, 0 when none — relies on write-synchronous indexing),
`search_facts`, `search_facts_scored -> list[(Fact, f64)]`.

Facts: `append_fact(familiar_id, channel_id, text, source_turn_ids,
subjects=(), valid_from=None, valid_to=None, importance=None, dedup=True) ->
Fact`, `facts_by_ids` (includes superseded), `recent_facts(familiar_id, limit,
include_superseded=False, as_of=None)`, `latest_fact_id -> int` (0 when none),
`all_fact_ids -> set[int]`, `supersede(familiar_id, obsolete_facts,
new_fact: FactDraft | Fact | int | None) -> SupersedeResult`,
`ancestors_of(familiar_id, fact_id) -> list[Fact]`,
`superseded_fact_ids(familiar_id, fact_ids) -> set[int]`.

Fact embeddings: `set_fact_embedding(fact_id, model, vector: list[float])`
(raises ValueError on empty vector), `get_fact_embeddings(fact_ids, model) ->
dict[int, list[float]]`, `unembedded_facts(familiar_id, model, limit)`,
`latest_embedded_fact_id(familiar_id, model) -> int`.

Reflections: `append_reflection(...) -> Reflection`,
`recent_reflections(familiar_id, channel_id=None, limit)`.

Alarms: `insert_alarm(...) -> str` (id = `uuid4().hex`),
`list_pending_alarms -> list[dict]` (raw dict rows, ordered scheduled_at asc),
`mark_alarm_fired(alarm_id, fired_at) -> bool`,
`cancel_alarm(alarm_id, cancelled_at) -> bool`.

Attentional stream: `promote_staged_turns(familiar_id, channel_id,
catch_up_limit=20) -> Promotion`, `promote_staged_turns_since(familiar_id,
after_turn_id, catch_up_limit=20) -> Promotion`, `count_staged -> int`,
`staged_channels -> dict[int, ChannelUnread]`,
`recent_cross_channel(familiar_id, limit, respect_archive=False)`,
`consumed_turns_after(familiar_id, after_consumed_at: str, after_id: int,
limit)`, `get_focus_pointers(familiar_id) -> FocusPointers | None` /
`set_focus_pointers(familiar_id, *, text_channel_id, voice_channel_id)`
(note: `familiar_id` positional on both focus and digest-watermark methods).

Activities: `create_activity -> int`, `finish_activity` (ValueError unless
status in {"completed","cut_short"}), `set_activity_experience`,
`active_activity -> ActivityRecord | None`, `latest_activity(familiar_id,
type_id) -> ActivityRecord | None`.

### `AsyncHistoryStore(store: HistoryStore)` — async_store.py

Reflection proxy: `__getattr__` fetches the attribute from the inner store;
non-callables are returned as-is (NOT awaitable); callables are wrapped into
an async fn that runs `functools.partial(attr, *args, **kwargs)` on the
store's 4-worker executor via `loop.run_in_executor`. `sync` property exposes
the raw store (used by layers that need sync access, e.g. invalidation keys).
`close()` is synchronous. No caching of wrappers; a new coroutine-producing
closure is minted per attribute access.

### Value types (all frozen dataclasses unless noted)

- `HistoryTurn`: `id, timestamp: datetime, role: str, author: Author | None,
  content: str, channel_id: int = 0, mode: str | None, platform_message_id:
  str | None, reply_to_message_id: str | None, guild_id: int | None,
  arrived_at: datetime | None, consumed_at: datetime | None,
  pings_bot: bool = False`.
- `ChannelUnread(NamedTuple)`: `(unread: int, pings: int)` — consumers
  destructure it as a plain 2-tuple (digest renderer); the Rust type must
  stay structurally tuple-like or the consumer contract updated.
- `Promotion(NamedTuple)`: `(consumed: int, missed: int)`.
- `SummaryEntry`: `last_summarised_id, summary_text, created_at,
  last_consumed_at: str | None` (composite-watermark cursor kept as a STRING,
  not parsed).
- `OtherChannelInfo`: `channel_id, mode, latest_id, latest_timestamp`.
- `WatermarkEntry`: `last_written_id, created_at`.
- `SleepWatermark`: `last_fact_id, last_turn_id, updated_at`.
- `FocusPointers`: `text_channel_id: int | None, voice_channel_id: int | None,
  updated_at`.
- `AccountProfile`: `canonical_key, username, global_name, pronouns, bio`.
- `PeopleDossierEntry`: `canonical_key, last_fact_id, dossier_text, created_at`.
- `FactSubject`: `canonical_key, display_at_write`.
- `FactDraft`: `channel_id: int | None, text, subjects: tuple[FactSubject,...]`
  — deliberately carries NO turn ids (store owns provenance).
- `SupersedeResult`: `minted: Fact | None, superseded: tuple[int,...],
  skipped: tuple[(int, str),...]`.
- `Fact`: `id, familiar_id, channel_id: int | None, text, source_turn_ids:
  tuple[int,...], created_at, superseded_at: datetime | None, superseded_by:
  int | None, subjects, valid_from, valid_to, importance: int | None`.
- `Reflection`: `id, familiar_id, channel_id: int | None, text,
  cited_turn_ids, cited_fact_ids, created_at, last_turn_id, last_fact_id`.
- `ActivityRecord`: `id, familiar_id, type_id, label, started_at,
  planned_return_at, note, status, actual_return_at, experience_text`.

## Behaviors & invariants

### Threading / async semantics

1. **Single-owning-thread contract**: every pyturso call (connect, execute*,
   fetch*, lastrowid, rowcount, commit, rollback, close, reopen) executes on
   the ONE OS thread owned by `TursoConnection`'s 1-worker executor. Callers
   on any thread block on `.result()`. Test pins this by asserting the trace
   callback observes exactly one thread id, which is NOT the caller's
   (test_turso_compat.py).
2. Statements from concurrent logical operations serialize per-statement, not
   per-operation: with `AsyncHistoryStore`'s 4 workers, two multi-statement
   ops (e.g. two `supersede` calls) can interleave their
   execute/execute/commit sequences on the turso thread. All store methods
   commit at the end of the method; there is no explicit BEGIN. This is
   benign today only because interleaved commits commit each other's work.
   The Rust port SHOULD strengthen this (whole-operation dispatch) but MUST
   NOT weaken visible semantics (see port notes).
3. `AsyncHistoryStore` guarantees DB/FTS work never runs on the event loop.
   Parallelism claim: up to 4 concurrent calls; tantivy searches genuinely
   parallel, SQL serializes onto the turso thread.
4. No locks are held across store calls; `FtsIndex` holds an `RLock` only
   around writer mutations (add/add_many/delete/clear/close). Search takes no
   lock. `HistoryStore` itself has no locks.
5. No timeouts, no retry policy anywhere except the tantivy commit retry
   (behavior 22) — a hung DB call blocks its caller forever.
6. Nothing in this subsystem spawns asyncio tasks or is cancellable
   mid-operation; cancellation of the awaiting coroutine leaves the executor
   job running to completion (standard run_in_executor semantics). Port must
   preserve at-most-once execution of each dispatched call.
7. After `HistoryStore.close()`, further async calls fail (executor is shut
   down); further direct sync calls fail at the `TursoConnection` layer with
   `RuntimeError("TursoConnection is closed")`.

### Construction & migration

8. Constructor order: mkdir parents → executescript(_SCHEMA) → commit →
   `_migrate()` → open FTS indexes. `_SCHEMA` uses `IF NOT EXISTS`
   throughout, so it is a repair pass on every open.
9. `_migrate()` steps, in order, each followed by commit:
   a. For `arrived_at` then `consumed_at`: try `ALTER TABLE turns ADD COLUMN
      <col> TEXT`; on success immediately backfill
      `UPDATE turns SET <col> = timestamp WHERE <col> IS NULL`. Any exception
      (column exists, or pyturso's spurious "Parse error: no such table") is
      swallowed — backfill runs ONLY when the ALTER succeeded. This one-time
      scoping is load-bearing: a deliberately staged turn
      (`consumed_at IS NULL`) must survive restart un-promoted
      (TestMigrationBackfillScope).
   b. `ALTER TABLE turns ADD COLUMN pings_bot INTEGER NOT NULL DEFAULT 0`
      (swallow failure; DEFAULT backfills legacy rows to 0).
   c. `ALTER TABLE summaries ADD COLUMN last_consumed_at TEXT` (swallow).
   d. `ALTER TABLE turns ADD COLUMN missed_at TEXT` (swallow; NO backfill —
      legacy rows are never "missed").
   e. `CREATE INDEX IF NOT EXISTS idx_turns_consumed ON turns (familiar_id,
      consumed_at, arrived_at, id)`.
   f. Ego-key rewrite (issue #154), idempotent, runs every open:
      `UPDATE people_dossiers SET canonical_key = 'ego:' ||
      substr(canonical_key, 6) WHERE canonical_key LIKE 'self:%'` and
      `UPDATE facts SET subjects_json = replace(subjects_json,
      '"canonical_key": "self:', '"canonical_key": "ego:') WHERE subjects_json
      LIKE '%"canonical_key": "self:%'`. NOTE: the replace pattern contains a
      space after the colon — it matches Python `json.dumps` default
      formatting exactly (see Data formats).
10. Opening a legacy DB (turns without new columns, accounts without
    pronouns/bio) must not raise, even if the engine throws parse errors from
    ALTER (pinned by test with monkeypatched `turso.Cursor.execute`). There
    is NO accounts-table column migration in current code.

### Turn writes & reads

11. `append_turn`: `timestamp` = now(UTC) always; `arrived_at` defaults to
    `timestamp` when omitted; `consumed_at = arrived_at` when `consumed=True`
    (default), else NULL. All three stored as ISO-8601 strings. `pings_bot`
    stored as 1/0. After INSERT+commit, the turn's content is indexed into
    fts_turns via `_safe_fts_add` (behavior 23). Returns a `HistoryTurn`
    built from inputs — QUIRK: the returned value leaves `guild_id`,
    `platform_message_id`, `reply_to_message_id` at their defaults (None)
    even when persisted; callers that need them re-read. Preserve or
    consciously fix with test updates.
12. Turn ids are engine-assigned AUTOINCREMENT, strictly increasing, global
    across channels within a DB file, persistent across reopen.
13. Read scoping: every turn query filters `familiar_id`; per-channel queries
    additionally filter `channel_id`. Order guarantees pinned by tests:
    `recent` returns oldest-first (fetch `ORDER BY id DESC LIMIT n`, then
    reverse); `mode` filter is exact string match; `before_id` is strict
    `id < before_id`. `older_than` is INCLUSIVE `id <= max_id`, asc.
    `turns_in_id_range` is `(min_id_exclusive, max_id_inclusive]`, asc.
    `turns_by_ids` dedupes + sorts input ids, returns asc.
    `turns_around` = up to `before` rows with `id < anchor` (desc, reversed)
    + up to `after`+1 rows with `id >= anchor` (asc) — anchor included, clips
    at edges, per-channel partition. Negative before/after clamp to 0.
14. `recent_distinct_authors`: GROUP BY (author_platform, author_user_id),
    skip NULL-author rows, order by MAX(id) desc, limit; `limit <= 0` → [].
    All `limit <= 0` guards across the store return empty without querying.
15. `lookup_turn_by_platform_message_id` picks the HIGHEST id on duplicates
    (`ORDER BY id DESC LIMIT 1`).
16. `latest_id_at_or_before` compares `timestamp <= ts.isoformat()` as TEXT —
    correctness depends on the fixed isoformat-UTC storage format being
    lexicographically chronological.

### FTS

17. Analyzer (both indexes): simple tokenizer (split on whitespace +
    punctuation) → remove_long(64) → lowercase → ascii_fold →
    custom_stopword(88-word list in fts.py — copy verbatim; includes chat
    fillers "hey", "hi", "lol", "ok", "know", "yes") → stemmer("english").
    Match this chain exactly in Rust tantivy or BM25 rankings drift.
18. Query path: `limit <= 0`, empty, or whitespace-only query → []. Query is
    sanitized by regex `[^\w\s]+` → space, then lowercased (kills tantivy
    query syntax; defuses AND/OR/NOT as operators; pinned: `"FOX AND
    BEAR"` matches docs containing either word, apostrophes/colons/quotes
    tolerated). If sanitize leaves only whitespace → []. Residual
    `parse_query` ValueError → [] (never raises). Parse is disjunctive (OR)
    over `content` only. Stopword-only queries → [] (zero parsed terms).
19. `search_turns`: overfetch FTS `max(limit*4, limit)` hits, join ids back
    to `turns` with `familiar_id` + optional `channel_id` + optional
    `id <= max_id` filters, then re-rank in memory by (-bm25, -id) — BM25
    desc, tie newer-first — truncate to limit. (In-code comment says "cap at
    10x"; actual factor is 4 — port the code, not the comment.)
20. `search_facts` / `search_facts_scored` share the same shape: overfetch
    4x, join back with the facts-validity filter (behavior 28), re-rank by
    (-score, -id), truncate. Scores are tantivy BM25: POSITIVE,
    higher-is-better (prior SQLite FTS5 was negative lower-is-better —
    downstream fusion in 05 assumes positive).
21. `rebuild_fts`: clear turns index, re-add ALL rows of `turns` (all
    familiars) ordered by id asc via one bulk commit. Facts index has no
    rebuild method today.
22. Commit retry (FtsIndex): only ValueErrors whose message contains a lock
    signature retry (3 backoffs then a final propagate); other ValueErrors
    propagate on first failure. Pinned by 3 monkeypatch tests.
23. `_safe_fts_add` (store-side guard): catches ONLY ValueError from
    `FtsIndex.add`, logs a warning, never raises — SQL row is already
    committed; an fts write failure must not fail `append_turn`/`append_fact`
    (pinned). Non-ValueError exceptions would propagate (not pinned; keep the
    narrow catch).
24. `update_turn_content_by_message_id`: SELECT matching ids first, UPDATE
    content, commit, then re-index each id with the new content (upsert
    replaces the old doc).

### Facts

25. `append_fact` stores `source_turn_ids` as a JSON int array,
    `subjects_json` as a JSON array of `{"canonical_key", "display_at_write"}`
    objects or NULL when subjects empty. `valid_from` defaults to
    `created_at` (now) when omitted; `valid_to` NULL means still applies.
    `importance` clamps to [1,10] when non-None; None preserved verbatim.
26. Near-duplicate suppression (`dedup=True`, only when `valid_to is None`):
    normalize text = lowercase → whitespace-collapse (`" ".join(split())`) →
    remove ALL `'` and `"` chars → strip surrounding chars ``.,!?;:()[]{}``
    plus whitespace. If any CURRENT fact (same familiar, validity filter of
    behavior 28 default) has equal normalized text AND set-equal subject
    canonical keys, return that existing Fact — no insert, no FTS write.
    Scope rules pinned: NULL-subject vs keyed-subject same-text are NOT dups;
    per-familiar; a superseded match does NOT block insert; `valid_to` set
    bypasses dedup entirely. Implementation scans all current facts in
    Python — O(n) per insert; port may index but must keep exact-match
    semantics (no fuzzy/semantic dedup).
27. `Fact` row parsing is tolerant: missing columns in a SELECT (older query
    shapes), unparseable JSON in source_turn_ids/subjects → empty tuples;
    malformed subject items (non-dict, missing keys) skipped.
28. Validity filter (`_facts_validity_where`): default "current truth" =
    `superseded_at IS NULL AND (valid_to IS NULL OR valid_to > <now-iso>)`
    (now captured per call; `include_superseded=True` drops the first
    conjunct). `as_of=<dt>` OVERRIDES include_superseded and switches to the
    bi-temporal slice `(valid_from IS NULL OR valid_from <= as_of) AND
    (valid_to IS NULL OR valid_to > as_of)` — superseded rows INCLUDED so
    audits reconstruct prior beliefs. Comparisons are TEXT comparisons on
    isoformat strings.
29. `recent_facts` newest-first (`ORDER BY id DESC LIMIT ?`).
    `facts_for_subject`: pre-filter `subjects_json LIKE '%"<key>"%'` (quoted
    key prevents discord:1 / discord:11 collision), `id > min_id_exclusive`,
    validity filter, asc — then a Python-side exact membership check on
    parsed subjects (LIKE is only an optimization).
30. `subjects_with_facts`: scans all non-superseded facts with non-NULL
    subjects_json asc, maps canonical_key → max fact id seen (last write
    wins under asc order). Tolerant JSON parsing (skip malformed).
31. `supersede` — three forms keyed on `new_fact` type:
    - `None` (retire): per-id — unknown id or already-superseded id is
      recorded in `skipped` with reason strings exactly
      `"unknown fact id={fid}"` / `"fact id={fid} already superseded"`; the
      rest get `superseded_at = now`, `superseded_by = NULL`. One commit.
    - `FactDraft` (merge): ATOMIC. Pre-flight every obsolete id; if list is
      empty OR any id unknown/already-superseded → decline whole: mint
      nothing, supersede nothing, all stale ids in `skipped`. Otherwise mint
      the replacement via `append_fact(dedup=False)` with
      `source_turn_ids` = order-preserving union of the obsolete rows'
      provenance (caller-supplied draft has none), then point every obsolete
      row's `superseded_by` at the minted id. Invariant: a CURRENT minted
      fact with zero ancestors ("phantom merge") is impossible;
      provenance-union == ancestry.
    - `Fact` or `int` (repoint): per-id skip-and-record like retire, but
      `superseded_by = <given id>`; mints nothing.
32. Every row actually superseded (all three forms) triggers dossier
    invalidation: `DELETE FROM people_dossiers WHERE familiar_id = ? AND
    canonical_key = ?` for each subject key of THAT row. Deletion — not
    watermark reset — is load-bearing (a surviving row would re-compound
    stale prose via the worker's "Previous dossier" path). Subject-less
    facts drop nothing; other subjects' and other familiars' dossiers are
    spared (all pinned).
33. `ancestors_of`: one-hop reverse walk — rows with `superseded_by =
    fact_id`, asc. `superseded_fact_ids`: subset filter, empty input short-
    circuits without a query.

### Identity, mentions, reactions

34. `upsert_account`: PK canonical_key; `username`/`global_name` last-write-
    win (even to NULL); `pronouns`/`bio` only overwrite when the new value is
    non-NULL (`COALESCE(excluded.x, accounts.x)`); `last_seen_at` always
    stamps now.
35. `upsert_guild_nick`: PK (canonical_key, guild_id); NULL nick is a
    meaningful "no override" record, distinct from no row.
36. `resolve_label` preference: (1) non-empty guild nick for (key, guild_id)
    when guild_id given → (2) accounts.global_name → (3) accounts.username →
    (4) when familiar_id given, `latest_author_for(...)`.label (freshest turn
    snapshot) → (5) user_id part of `platform:user_id`, or the raw key if no
    colon / empty tail. Always non-empty.
37. `latest_author_for`: returns None on malformed canonical_key (no colon,
    empty platform or user_id); else newest matching turn's Author.
38. `record_mentions`: dedupes input preserving first-seen order, executemany
    `INSERT OR IGNORE`, empty input no-ops (no SQL). Reads sorted by key asc.
39. Reactions: `set_reaction` with `count <= 0` deletes the row; else upsert.
    `bump_reaction(delta)`: `delta == 0` no-op (no SQL); UPDATE
    `count = count + delta`; if rowcount==0 AND delta>0, INSERT with
    count=delta; then DELETE any row with `count <= 0` (floor-at-zero:
    stray remove leaves no row, never a negative). `clear_reactions`: all
    emojis or one. `reactions_for_messages`: filters empty/falsy ids, ONE
    query (pinned by trace-callback test), result maps message id → tuples
    ordered count desc then emoji asc; no-reaction messages absent; empty
    input → {} without SQL.

### Summaries & watermarks

40. Summaries keyed (familiar_id, channel_id); channel_id=0 is the legacy
    channel-less bucket; `FOCUS_STREAM_CHANNEL_ID` (-1) holds the focus-
    stream summary. `put_summary` full-row upsert (created_at restamped;
    `last_consumed_at` overwritten even to None). `last_consumed_at` is an
    opaque cursor STRING (isoformat) round-tripped verbatim; NULL on legacy
    rows.
41. `advance_sleep_watermark`: both-None → no-op (no SQL). Upsert where
    omitted axes default to 0 on first insert and are preserved on update via
    `COALESCE(?, col)` — hygiene owns last_fact_id, dream owns last_turn_id;
    neither clobbers the other (pinned).
42. `latest_reflection_watermarks`: prefer the `reflection_watermark` row;
    fall back to the newest `reflections` row's (last_turn_id, last_fact_id);
    else (0, 0). `set_reflection_watermark` upserts unconditionally (called
    even on no-op ticks — see memory-strategies.md for why).
43. `turns_since_watermark`: reads writer watermark (0 when unset), returns
    turns `id > watermark` asc, limit default 10_000.
44. Archive watermarks: `set_archive_watermark_all` upserts one row per
    DISTINCT channel with turns (single INSERT…SELECT…ON CONFLICT).
    `get_archive_watermark` → int | None.

### Attentional stream

45. Staged = `consumed_at IS NULL AND missed_at IS NULL`. Missed
    (`missed_at` set, `consumed_at` stays NULL) is TERMINAL: excluded from
    `count_staged`, `staged_channels`, `recent_cross_channel`,
    `consumed_turns_after`, and never re-promoted (both promotion queries
    filter `missed_at IS NULL`).
46. `promote_staged_turns` (focus swap, one channel): select staged rows
    ordered `arrived_at DESC, id DESC`; the first `catch_up_limit` rows PLUS
    any row with pings_bot=1 (regardless of rank) get
    `consumed_at = now-iso`; the rest get `missed_at = now-iso`. Rank
    counting includes ping rows (window is positional over the newest-first
    list). Default `catch_up_limit = 20` (class constant
    `_DEFAULT_CATCH_UP_LIMIT`; the canonical knob `[focus].catch_up_limit`
    lives in subsystem 02 and is threaded by FocusManager/ActivityEngine).
47. `promote_staged_turns_since` (activity return, all channels): scope
    `id > after_turn_id` only — pre-absence staged turns untouched; consumed
    turns untouched. Rows ordered `channel_id ASC, arrived_at DESC, id DESC`;
    window ranked PER CHANNEL; pings always caught. Both promoters stamp via
    chunked UPDATEs (`WHERE id IN (...)`, 500 ids per statement), one commit
    for the whole promotion, and return `Promotion(consumed, missed)` counts.
    Selection runs in application code, not SQL (engine lacks window
    functions) — port may use SQL if results are bit-identical.
48. Promotion sets `consumed_at`/`missed_at` to NOW, not arrived_at — this is
    the load-bearing detail behind the composite cursor (behavior 50).
49. `recent_cross_channel`: inner query = newest `limit` consumed turns
    (`consumed_at IS NOT NULL`) ordered `arrived_at DESC, id DESC`; with
    `respect_archive=True` the archive filter
    (`t.id > COALESCE(<channel watermark>, 0)`) applies OUTSIDE the window —
    the window SHRINKS rather than backfilling past the watermark (pinned).
    Result reversed to oldest-first.
50. `consumed_turns_after`: composite exclusive cursor
    `consumed_at > after OR (consumed_at = after AND id > after_id)`, ordered
    `consumed_at ASC, id ASC`, limit. Empty-string cursor matches everything
    (cold start). Watermarking on consumed_at (not id) is the headline
    guarantee: a late-promoted old-id turn has fresh consumed_at and must be
    picked up (pinned: test_includes_late_promoted_low_id). Cursor is passed
    around as the raw TEXT value, never parsed.
51. Focus pointers / digest watermark: single-row-per-familiar upserts;
    getters return None when unset; NULL channel ids allowed; updated_at
    restamped each write.

### Alarms & activities

52. Alarms: TEXT PK (uuid4 hex); pending = `fired_at IS NULL AND cancelled_at
    IS NULL`; `mark_alarm_fired`/`cancel_alarm` guard on both-NULL and return
    whether a row changed (False for unknown/already-terminal ids).
    `channel_kind` CHECK ('text','voice') — violating insert must error
    (pinned). `scheduled_at`/timestamps are caller-supplied strings.
53. Activities: append-only; active row = `actual_return_at IS NULL`, newest
    by id; `finish_activity` validates status in application code
    (ValueError) despite the CHECK; finished rows remain (append-only
    history pinned); `latest_activity` filters by type_id, includes finished,
    newest-first.

## Data formats

### SQLite schema (post-migration, authoritative)

All timestamps are TEXT: Python `datetime.now(tz=UTC).isoformat()` →
`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`. CAVEAT: Python omits `.ffffff` when
microseconds are exactly 0 (~1e-6 of writes); lexicographic comparisons
(`timestamp <= ?`, `consumed_at > ?`, validity filters) assume
chronological==lexicographic. Rust MUST emit fixed-width microseconds AND
the `+00:00` suffix (not `Z`) to interoperate with existing rows, and must
parse both variants.

```sql
CREATE TABLE turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id TEXT NOT NULL,
    channel_id INTEGER NOT NULL,          -- 0 = channel-less, -1 = focus stream
    guild_id INTEGER,                     -- NULL for DMs/non-Discord/legacy
    role TEXT NOT NULL,                   -- "user" | "assistant" | "tool" | free-form
    author_platform TEXT, author_user_id TEXT,
    author_username TEXT, author_display_name TEXT,   -- all NULL for authorless
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,              -- write time (now), NOT arrival time
    mode TEXT,                            -- free-form tag e.g. "activity_return"
    platform_message_id TEXT, reply_to_message_id TEXT,  -- platform-native ids as TEXT
    tool_calls_json TEXT, tool_call_id TEXT,
    pings_bot INTEGER NOT NULL DEFAULT 0,
    -- migration-added:
    arrived_at TEXT,                      -- immutable ingest time; NULL only pre-migration
    consumed_at TEXT,                     -- NULL = staged (or missed)
    missed_at TEXT                        -- terminal; consumed_at stays NULL
);
CREATE INDEX idx_turns_channel      ON turns (familiar_id, channel_id, id);
CREATE INDEX idx_turns_global       ON turns (familiar_id, id);
CREATE INDEX idx_turns_channel_mode ON turns (familiar_id, channel_id, mode, id);
CREATE INDEX idx_turns_platform_msg ON turns (familiar_id, platform_message_id);
CREATE INDEX idx_turns_consumed     ON turns (familiar_id, consumed_at, arrived_at, id);

CREATE TABLE message_reactions (
    familiar_id TEXT NOT NULL, platform_message_id TEXT NOT NULL,
    emoji TEXT NOT NULL, count INTEGER NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (familiar_id, platform_message_id, emoji)
);
CREATE INDEX idx_message_reactions_lookup ON message_reactions (familiar_id, platform_message_id);

CREATE TABLE turn_mentions (
    turn_id INTEGER NOT NULL, canonical_key TEXT NOT NULL,
    PRIMARY KEY (turn_id, canonical_key)
);
CREATE INDEX idx_turn_mentions_canonical ON turn_mentions (canonical_key, turn_id);

CREATE TABLE summaries (
    familiar_id TEXT NOT NULL, channel_id INTEGER NOT NULL DEFAULT 0,
    last_summarised_id INTEGER NOT NULL, summary_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_consumed_at TEXT,                -- migration-added; opaque cursor string
    PRIMARY KEY (familiar_id, channel_id)
);

CREATE TABLE memory_writer_watermark (
    familiar_id TEXT PRIMARY KEY, last_written_id INTEGER NOT NULL, created_at TEXT NOT NULL
);

CREATE TABLE people_dossiers (
    familiar_id TEXT NOT NULL, canonical_key TEXT NOT NULL,
    last_fact_id INTEGER NOT NULL, dossier_text TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (familiar_id, canonical_key)
);

CREATE TABLE reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id TEXT NOT NULL, channel_id INTEGER,   -- NULL = channel-agnostic
    text TEXT NOT NULL,
    cited_turn_ids TEXT NOT NULL, cited_fact_ids TEXT NOT NULL,  -- JSON int arrays
    created_at TEXT NOT NULL,
    last_turn_id INTEGER NOT NULL, last_fact_id INTEGER NOT NULL
);
CREATE INDEX idx_reflections_familiar         ON reflections (familiar_id, id);
CREATE INDEX idx_reflections_familiar_channel ON reflections (familiar_id, channel_id, id);

CREATE TABLE reflection_watermark (
    familiar_id TEXT PRIMARY KEY,
    last_turn_id INTEGER NOT NULL, last_fact_id INTEGER NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE sleep_watermark (
    familiar_id TEXT PRIMARY KEY,
    last_fact_id INTEGER NOT NULL, last_turn_id INTEGER NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE accounts (
    canonical_key TEXT PRIMARY KEY,       -- "discord:123" / "twitch:456" / "ego:<id>"
    platform TEXT NOT NULL, user_id TEXT NOT NULL,
    username TEXT, global_name TEXT, pronouns TEXT, bio TEXT,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE account_guild_nicks (
    canonical_key TEXT NOT NULL, guild_id INTEGER NOT NULL,
    nick TEXT,                            -- NULL = explicit "no override"
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (canonical_key, guild_id)
);

CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id TEXT NOT NULL, channel_id INTEGER,
    text TEXT NOT NULL,
    source_turn_ids TEXT NOT NULL,        -- JSON int array, e.g. "[1, 2]"
    created_at TEXT NOT NULL,
    superseded_at TEXT, superseded_by INTEGER,   -- NULL = current; FK by convention
    subjects_json TEXT,                   -- JSON array or NULL (legacy / no subjects)
    valid_from TEXT, valid_to TEXT,       -- world-time; NULL = legacy / still applies
    importance INTEGER                    -- 1-10 or NULL
);
CREATE INDEX idx_facts_familiar          ON facts (familiar_id, id);
CREATE INDEX idx_facts_familiar_current  ON facts (familiar_id, superseded_at, id);
CREATE INDEX idx_facts_familiar_validity ON facts (familiar_id, valid_from, valid_to);

CREATE TABLE fact_embeddings (
    fact_id INTEGER NOT NULL, model TEXT NOT NULL,
    dim INTEGER NOT NULL, vector BLOB NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (fact_id, model)
);
CREATE INDEX idx_fact_embeddings_model ON fact_embeddings (model, fact_id);

CREATE TABLE alarms (
    id TEXT PRIMARY KEY,                  -- uuid4 hex
    familiar_id TEXT NOT NULL, channel_id INTEGER NOT NULL,
    channel_kind TEXT NOT NULL CHECK(channel_kind IN ('text','voice')),
    scheduled_at TEXT NOT NULL, reason TEXT NOT NULL,
    originating_turn_id TEXT,
    fired_at TEXT, cancelled_at TEXT,     -- pending = both NULL
    created_at TEXT NOT NULL
);
CREATE INDEX idx_alarms_pending ON alarms (familiar_id, fired_at, cancelled_at, scheduled_at);

CREATE TABLE focus_pointers (
    familiar_id TEXT PRIMARY KEY,
    text_channel_id INTEGER, voice_channel_id INTEGER, updated_at TEXT NOT NULL
);

CREATE TABLE unread_digest_watermark (
    familiar_id TEXT PRIMARY KEY, watermark_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE channel_archive_watermark (
    familiar_id TEXT NOT NULL, channel_id INTEGER NOT NULL,
    turn_id INTEGER NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (familiar_id, channel_id)
);

CREATE TABLE activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    familiar_id TEXT NOT NULL, type_id TEXT NOT NULL, label TEXT NOT NULL,
    started_at TEXT NOT NULL, planned_return_at TEXT NOT NULL,
    note TEXT,
    status TEXT CHECK(status IN ('completed','cut_short')),
    actual_return_at TEXT, experience_text TEXT
);
CREATE INDEX idx_activities_active ON activities (familiar_id, actual_return_at, id);
```

### JSON column encodings (exact)

Written with Python `json.dumps` DEFAULT formatting — separators `", "` and
`": "` (spaces after both):

- `facts.source_turn_ids` / `reflections.cited_*_ids`: `[1, 2, 3]` or `[]`.
- `facts.subjects_json`:
  `[{"canonical_key": "discord:1", "display_at_write": "Alice"}, ...]`
  or SQL NULL when no subjects. Two consumers depend on the exact text:
  (1) the ego migration's `replace(..., '"canonical_key": "self:', ...)`
  needs the space after the colon; (2) `facts_for_subject`'s LIKE pre-filter
  needs the key quoted (any spacing works). A Rust port using serde_json
  default (compact, no spaces) will still satisfy (2), and (1) only matters
  for rows written by the old Python build — but mixed-format DBs are the
  norm after a port, so either emit Python-style spacing or make the ego
  migration whitespace-insensitive.
- `turns.tool_calls_json` / `tool_call_id`: opaque strings supplied by
  subsystem 08 (round-tripped verbatim; never parsed here).

### Fact-embedding vector encoding

`vector` BLOB = packed little-endian IEEE-754 float32, `dim` elements
(`struct.pack("<{n}f")` / `unpack`). Round-trip pinned to f32 precision only.
Reads return missing (fact_id, model) pairs as absent keys.

### Tantivy indexes (on disk)

`<db_dir>/fts/turns/` and `<db_dir>/fts/facts/` — standard tantivy segment
directories. Doc schema: `row_id` (i64, stored, indexed, fast) + `content`
(text, NOT stored, tokenizer `familiar_en`). Upserts implemented as
delete-by-term(row_id) + add. Index directories are disposable —
`rebuild_fts` regenerates turns from SQL (facts index would need equivalent
tooling in the port).

## Config knobs

This subsystem reads NO TOML keys and NO env vars directly. Everything is
constructor-injected:

- DB path: wiring (subsystem 02, `Familiar.from_root`) passes
  `data/familiars/<id>/history.db`; FTS lives beside it at
  `data/familiars/<id>/fts/{turns,facts}/`. `":memory:"` → fully in-memory
  (tests).
- `catch_up_limit`: parameter with hardcoded fallback 20
  (`HistoryStore._DEFAULT_CATCH_UP_LIMIT`); the canonical knob is
  `[focus].catch_up_limit` (subsystem 02 `FocusConfig`), threaded by
  FocusManager / ActivityEngine callers.
- Hardcoded internals worth surfacing as constants: turso
  `experimental_features="index_method"`; executor sizes (1 turso thread, 4
  db workers); tantivy writer heap 15 MB / 1 thread; FTS overfetch factor 4;
  commit retry delays (0.05, 0.2, 0.5); promotion UPDATE chunk size 500;
  `older_than`/`turns_since_watermark` default limit 10 000; importance
  clamp [1, 10]; token max length 64; `FOCUS_STREAM_CHANNEL_ID = -1`.

## Dependency edges

Imports (this subsystem depends on):

- `familiar_connect.identity` — `Author`, `Platform` (subsystem 02).
- `familiar_connect.log_style` — log formatting helpers (subsystem 01).
- Third-party: `turso` (pyturso 0.5.1), `tantivy` (Python bindings). No
  network services; everything is local files.

Imported by (consumers → subsystem):

- 02 config+identity: `familiar.py` constructs
  `AsyncHistoryStore(HistoryStore(root / "history.db"))`.
- 05 context-assembly: `context/layers.py` (HistoryTurn,
  FOCUS_STREAM_CHANNEL_ID, search_turns/search_facts_scored/latest_fts_id/
  recent/reactions/resolve_label/…).
- 06 responders: `processors/text_responder.py`, `voice_responder.py`
  (append/stage turns, FOCUS_STREAM_CHANNEL_ID); `focus.py` (FocusManager —
  promotions, staged_channels, focus pointers; grouped with 06 in the plan).
- 07 background-workers: `processors/summary_worker.py` (consumed_turns_after,
  put_summary), `fact_extractor.py` (turns_since_watermark, append_fact),
  `fact_supersede_worker.py` (supersede), `people_dossier_worker.py`
  (subjects_with_facts, facts_for_subject, dossier CRUD),
  `reflection_worker.py` (reflection append + watermark),
  `fact_embedding_worker.py` (unembedded_facts, set_fact_embedding),
  `history_writer.py`, `projectors.py`.
- 04 embedding+sleep: `sleep/apply.py`, `consolidation.py`, `maintenance.py`,
  `opinion_formation.py` (facts_by_ids, supersede with FactDraft,
  sleep watermark, latest_fts_id).
- 08 llm+tools: `tools/registry.py`, `scheduler.py` (alarms),
  `channel_view.py`, `read_channel` (turns_around).
- 10 discord-shell: `bot.py` (append_turn with pings_bot, reactions,
  upsert_account/upsert_guild_nick, update_turn_content_by_message_id).
- 11 twitch+activities: `activities/engine.py` (promote_staged_turns_since,
  archive watermarks, activities CRUD, turns_around).

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| test_history_store.py (2021 loc) | construction/paths/:memory:; append/read round-trips incl. author, guild_id, mode, pings_bot; id monotonicity + reopen persistence; recent ordering/limits/per-channel/per-familiar isolation; distinct authors; older_than/latest_id/count scoping; summary CRUD + last_consumed_at round-trip + legacy NULL; consumed_turns_after composite cursor incl. late-promoted-low-id headline case; one-time backfill scope; ego-key migration + idempotence; legacy-DB open never raises (monkeypatched ALTER failure); mode filters; writer watermark; before_id paging; recent_cross_channel respect_archive (window shrinks); promote_staged_turns_since (per-channel window, pings caught, scope > after_id); turns_around; archive watermark CRUD; activities lifecycle incl. status ValueError; staged_channels ping tally; async wrappers | logic-portable (2 tests monkeypatch turso.Cursor / use raw turso to hand-build legacy DBs → needs-Rust-mock or fixture DB files) |
| test_attentional_store.py (564) | arrived_at/consumed_at semantics of append vs stage; consumed defaults; count_staged / staged_channels scoping; promote_staged_turns window + missed terminality (excluded from every read path); pings always caught; focus pointers + digest watermark CRUD; migration backfill (hand-built legacy rows via `store._conn`); HistoryTurn arrived_at default None; AsyncHistoryStore dispatch | logic-portable (migration tests poke `_conn` directly → rewrite against file fixtures) |
| test_facts_store.py (1241) | `_placeholders` / subjects-json parsing helpers; fact append/read round-trips; supersession all three forms incl. atomic-merge decline, provenance union, ancestry, per-id skip reasons (exact strings), dossier invalidation matrix; sleep watermark partial-axis; bi-temporal defaults/as_of/audit; importance clamp; scored search positivity; dedup matrix (normalization, subject scoping, valid_to bypass, superseded non-blocking, per-familiar) | logic-portable (helper tests become unit tests of Rust equivalents) |
| test_history_fts.py (281) | search semantics: scoping, limit, determinism, rebuild, empty/unknown/stopword-only queries, punctuation + uppercase-boolean tolerance, stemming recall, max_id bound; commit retry/exhaustion/fail-fast; append_turn survives FTS failure | mixed: search semantics logic-portable; retry tests monkeypatch `_commit_writer`/`add` → needs-Rust-mock (inject failing commit via trait/test hook) |
| test_history_identity.py (207) | accounts upsert (COALESCE pronouns/bio, last-write username), guild nicks incl. NULL-nick override, resolve_label 5-step chain | logic-portable |
| test_history_replies.py (165) | platform/reply id round-trip via recent; lookup scoped per-familiar, None on miss; turn_mentions idempotence/dedupe/empty-noop | logic-portable |
| test_history_alarms.py (205) | alarm CRUD, pending filter, fired/cancel guards + bool returns, CHECK enforcement, tool_calls_json/tool_call_id round-trip | logic-portable |
| test_fact_embeddings_store.py (135) | f32 round-trip, empty-vector ValueError, upsert, per-model isolation, unembedded_facts (superseded excluded, asc order, limit), latest_embedded_fact_id | logic-portable |
| test_people_dossiers_store.py (231) | dossier CRUD/overwrite/familiar isolation; subjects_with_facts max-id map, skips subject-less; facts_for_subject membership + min_id + superseded default | logic-portable |
| test_reflections_store.py (259) | append round-trip, NULL channel, newest-first, channel filter includes NULL-channel rows, limits, watermark fallback chain, superseded_fact_ids subset | logic-portable |
| test_message_reactions.py (425) | set/bump/clear semantics incl. floor-at-zero, negative-no-row no-op; batch ordering; SINGLE-QUERY pin via set_trace_callback; higher-layer rendering (belongs to 05/10) | store parts logic-portable; single-query pin needs a Rust trace/statement-count hook |
| test_turso_compat.py (174) | one connection shared across threads; every call on one OS thread ≠ caller thread; cursor proxies from worker threads; reopen swaps connection + preserves data | Python-specific-skip if the Rust design drops the thread-affinity shim (rusqlite/turso-rs have no such constraint); keep only if an actor thread is retained, re-pinned as "all statements execute on the owner task" |

Also touching the store as a fixture (not conformance for 03): the worker
tests (test_summary_worker, test_fact_extractor, test_fact_supersede_worker,
test_people_dossier_worker, test_reflection_worker,
test_fact_embedding_worker), sleep tests (test_sleep_apply,
test_consolidation, test_maintenance_passes, test_opinion_formation),
layer/responder/tool/activity tests. They exercise cross-module contracts
(watermark discipline, dossier invalidation, FactDraft merges) and should
pass unchanged against the ported store.

## Rust port notes

- **Engine choice**: files are standard SQLite format. Options: `rusqlite`
  (mature, sync, `Connection: Send + !Sync`) or the `turso` Rust crate
  (the same engine pyturso wraps, async-native). Either removes pyturso
  0.5.1's pathologies — `reopen()`, the stale-schema-cache workaround, and
  the swallow-parse-error-on-ALTER hack exist ONLY for pyturso; port the
  tolerant migration (attempt ALTER, ignore duplicate-column errors) but the
  reopen/schema-cache machinery can be dropped. Keep
  `PRAGMA`-free/`AUTOINCREMENT` semantics identical (ids must never be
  reused — AUTOINCREMENT, not bare rowid).
- **Collapse the three-layer threading sandwich**. Python has: caller →
  AsyncHistoryStore (4-thread pool) → HistoryStore (sync) → TursoConnection
  (1-thread pool, per-STATEMENT dispatch). In Rust, prefer one dedicated DB
  actor (thread or `spawn_blocking` + `Mutex<Connection>`) receiving
  whole-operation closures/messages over an mpsc channel. This upgrade makes
  multi-statement operations (supersede, `_merge_atomically`, promotions,
  bump_reaction) genuinely atomic — wrap them in explicit transactions,
  which the Python version never had (per-statement interleaving was
  possible; see behavior 2). No test pins the non-atomicity; strengthening
  is safe. Preserve: DB work off the async runtime, at-most-once execution,
  calls after close() fail with an explicit error.
- **AsyncHistoryStore's `__getattr__` duck-proxy cannot be transliterated.**
  Write explicit `async fn` wrappers (or generate with a macro). Callers use
  `.sync` for a handful of sync paths (invalidation keys in 05) — decide
  whether to expose a sync view or make those async in the port.
- **tantivy is native Rust** — the Python bindings wrap the real crate, so
  analyzer parity is achievable exactly: `SimpleTokenizer` +
  `RemoveLongFilter(64)` + `LowerCaser` + `AsciiFoldingFilter` +
  `StopWordFilter(custom 88-word list)` + `Stemmer(English)`. Copy the
  stopword list verbatim from fts.py. Field schema: `row_id` i64
  STORED|INDEXED|FAST, `content` text unstored with the custom tokenizer.
  Query path: sanitize with `[^\w\s]+` → space + lowercase (note Python `\w`
  is Unicode-aware; use a Unicode-aware regex), parse against `content`
  only, disjunctive. Existing on-disk indexes may not be
  version-compatible with a newer tantivy — plan to `rebuild_fts` (and add
  the missing facts-rebuild) on first open rather than promising index
  reuse. The Windows AV commit-retry (lock-signature sniffing on error
  strings) is worth keeping but match on `std::io::ErrorKind::PermissionDenied`
  instead of substrings.
- **Timestamp discipline**: store as ISO-8601 UTC with `+00:00` offset and
  ALWAYS-present 6-digit microseconds (chrono
  `format("%Y-%m-%dT%H:%M:%S%.6f+00:00")` or jiff equivalent); parse
  tolerantly (existing Python rows may lack microseconds). Lexicographic
  ordering of these strings is a correctness dependency in five query paths.
- **JSON columns**: use serde_json but see Data formats — decide explicitly
  on separator compatibility for `subjects_json` (recommend: emit compact,
  make the ego migration match both spaced and compact forms, keep the
  LIKE pre-filters which are whitespace-agnostic).
- **Row tolerance**: the Python `_row_to_*` helpers catch IndexError/KeyError
  for columns absent from older SELECT shapes. In Rust, make every SELECT
  list explicit and total (always select all needed columns) and drop the
  fallback machinery; the only load-bearing fallback is `channel_id -> 0`
  for the summary-worker's legacy shape — audit call sites instead.
- **Typed replacements**: `list_pending_alarms` returns raw dicts — define
  an `AlarmRow` struct. `ChannelUnread`/`Promotion` are NamedTuples consumed
  structurally — plain Rust tuples or structs with accessors; update the
  digest renderer contract in 05/06 accordingly. `supersede`'s
  `FactDraft | Fact | int | None` union → a proper enum
  (`Retire | Merge(FactDraft) | Repoint(FactId)`).
- **Error policy**: Python leans on exceptions only for
  `finish_activity` status, empty embedding vector, closed connection, and
  CHECK violations; everything else returns Option/empty. Mirror with
  `Result<T, StoreError>` but keep the "reads never fail on malformed
  stored JSON — degrade to empty" behavior.
- **Monkeypatch seams to redesign**: tests patch `FtsIndex._commit_writer`
  (commit-failure injection), `FtsIndex.add` (append_turn resilience), and
  `turso.Cursor.execute` (migration failure injection),
  `set_trace_callback` (query counting). Provide equivalent injection
  points: a commit hook or fallible-writer trait on the FTS wrapper, and a
  statement-trace hook on the connection actor.
- **Suggested crates**: rusqlite (or turso), tantivy, tokio, serde/serde_json,
  chrono or jiff, uuid (v4, `simple()` hex format to match `uuid4().hex` —
  32 lowercase hex chars, no dashes), thiserror, bytemuck or byteorder (f32
  blob packing), regex.
- **Do not port**: `reopen()`, `_conn()` escape hatch, `is_empty()` (no
  caller), the Row `Any` typing, `latest_fts_id`'s indirection can stay as a
  plain MAX(id) query.
