# 07-background-workers — port spec

Source files: `processors/fact_extractor.py`, `processors/summary_worker.py`,
`processors/people_dossier_worker.py`, `processors/reflection_worker.py`,
`processors/fact_embedding_worker.py`, `processors/fact_supersede_worker.py`
(~1,705 loc). The projector registry/factories (`processors/projectors.py`)
that instantiates these is specced in 06; this spec covers the six workers it
constructs.

## Role

Six watermark-driven background LLM workers ("memory projectors") that
project the append-only `turns` table (03) into side-indices: atomic facts,
the focus-stream rolling summary, per-person dossiers, higher-order
reflections, fact embeddings, and fact retirement (supersession). Every
worker is a forever-loop asyncio task spawned by the wiring layer's
TaskGroup, ticking on a fixed interval, idempotent enough that any
side-index can be deleted and rebuilt from `turns`. All LLM traffic is
non-streaming `chat` on the `"background"` LLM slot — nothing here is on the
reply hot path; the read path (05 layers) never waits on these workers.

## Public API surface

All six classes structurally implement the `MemoryProjector` Protocol
(defined in `processors/projectors.py`, subsystem 06):

```python
class MemoryProjector(Protocol):   # duck-typed seam; third parties plug in
    name: str                      # log/task label
    async def run(self) -> None: ...   # forever loop; cancel to stop
```

`tick()` is public on every worker (tests drive it directly, `run()` merely
loops it). Constructors are keyword-only throughout.

### FactExtractor (`name = "fact-extractor"`, registry name `rich_note`)

```python
FactExtractor(*, store: AsyncHistoryStore, llm_client: LLMClient,
    familiar_id: str, familiar_display_name: str | None = None,
    batch_size: int = 10, tick_interval_s: float = 15.0,
    participants_max: int = 30, dream_extraction_clause: str = "")
async def run(self) -> None
async def tick(self) -> None        # @span("facts.tick")
```

- `familiar_display_name=None` → defaults to `familiar_id.title()`.
- `batch_size` and `participants_max` clamped to `max(1, x)` at construction.
- Holds `self._sync = store.sync` — participant-manifest building uses the
  **sync** store directly (see B-F13).
- Module-level pure helpers worth keeping as free functions (unit-visible via
  behavior): `_is_self_capability(text, name_re)`, `_build_participants(...)`,
  `_resolve_subjects(raw, participants)`, `_build_extract_prompt(...)`,
  `_normalize_fact_items(parsed)`, `_parse_importance(raw)`,
  `_parse_iso_dt(raw)`.

### SummaryWorker (`name = "summary-worker"`, registry `rolling_summary`)

```python
SummaryWorker(*, store, llm_client, familiar_id: str,
    turns_threshold: int = 10, backfill_cap: int = 200,
    tick_interval_s: float = 5.0)
async def run(self) -> None
async def tick(self) -> None        # @span("summary.tick")
```

`turns_threshold`, `backfill_cap` clamped `max(1, x)`. Note: `backfill_cap`
is NOT threaded by the 06 factory (factory passes only `turns_threshold`,
`tick_interval_s`; cap stays the constructor default 200).

### PeopleDossierWorker (`name = "people-dossier-worker"`, registry `people_dossier`)

```python
PeopleDossierWorker(*, store, llm_client, familiar_id: str,
    familiar_display_name: str | None = None, tick_interval_s: float = 20.0)
async def run(self) -> None
async def tick(self) -> None        # @span("people_dossier.tick")
```

Module constant `SELF_DOSSIER_MIN_IMPORTANCE = 5` and pure helper
`_dossier_facts(facts, *, is_self)` (filter+sort, see B-D4/D5).

### ReflectionWorker (`name = "reflection-worker"`, registry `reflection`)

```python
ReflectionWorker(*, store, llm_client, familiar_id: str,
    turns_threshold: int = 20, max_reflections_per_tick: int = 3,
    max_turns_per_tick: int = 50, recent_facts_limit: int = 20,
    tick_interval_s: float = 60.0)
async def run(self) -> None
async def tick(self) -> None        # @span("reflection.tick")
```

Clamps: `max(1,…)` on threshold / max_reflections / max_turns;
`max(0, recent_facts_limit)`. Pure helper `_dominant_channel(turns)`.

### FactEmbeddingWorker (`name = "fact-embedding-worker"`, registry `fact_embedding`)

```python
FactEmbeddingWorker(*, store, embedder: Embedder, familiar_id: str,
    batch_size: int = 32, tick_interval_s: float = 15.0)
async def run(self) -> None
async def tick(self) -> int         # @span("fact_embedding.tick"); count written
```

The only worker with no LLM client. Consumes the `Embedder` protocol (04):
`name: str` (storage model key), `dim: int`, `async embed(texts: list[str])
-> list[list[float]]`. Tests substitute plain duck-typed stubs — the Rust
trait must accept non-registered implementations.

### FactSupersedeWorker (`name = "fact-supersede-worker"`, registry `fact_supersede`)

```python
FactSupersedeWorker(*, store, llm_client, familiar_id: str,
    batch_size: int = 5, tick_interval_s: float = 60.0, priors_max: int = 20)
def prime_watermark(self) -> None   # SYNC; store.sync.latest_fact_id
async def run(self) -> None         # calls prime_watermark() first
async def tick(self) -> int         # @span("fact_supersede.tick"); count retired
```

`batch_size`, `priors_max` clamped `max(1, x)`. Watermark
`_last_seen_fact_id` is **in-memory only** (starts 0; primed at run start).

## Behaviors & invariants

### Shared loop contract (all six workers)

1. **run() loop shape**: `while True: try tick() / except CancelledError:
   raise / except Exception: log WARNING "<Tag> tick_error=<repr>" ; then
   `await asyncio.sleep(tick_interval_s)` — the sleep runs after *every*
   iteration, success or failure. There is no immediate re-tick after a full
   batch: backlog drains at one batch per interval. Workers must never die
   from a tick exception (transport errors, DB errors — anything but
   cancellation).
2. **Cancellation**: `CancelledError` re-raised from `run()`; the wiring
   layer (`commands/run.py`, 10) spawns one task per projector inside an
   `asyncio.TaskGroup` (`tg.create_task(proj.run(), name=proj.name)`), so
   shutdown or any sibling task failure cancels all workers. Cancellation can
   land at any await inside `tick()` — partial-tick writes are tolerated by
   design (see per-worker idempotency notes).
3. **LLM discipline**: every LLM call is `chat` (non-streaming) on the
   client wired as `llm_clients["background"]` by the 06 factories.
   Structured-reply workers (extractor, reflection, supersede) go through
   `request_structured` (08): shape failures get ONE corrective re-ask
   (`DEFAULT_MAX_RETRIES = 1`, so ≤2 LLM calls per request), then degrade to
   `value=None` → the worker's empty container. **Transport errors propagate
   unchanged** out of `request_structured`/`chat` and are absorbed by the
   run-loop's catch — the distinction between "malformed reply" (degrade,
   usually advance watermark) and "transport error" (raise, usually retry
   same window next tick) is load-bearing and called out per worker below.
4. **Spans**: each `tick` is wrapped `@span("<area>.tick")` — DEBUG-level
   duration log with ok/error status; must re-raise, never swallow.
5. **Watermark idempotency doctrine** (docs/architecture/memory-strategies.md):
   `turns` is source of truth; each side-index is a projection rebuildable by
   deleting its table. Duplicate-write protection on replay comes from the
   store layer: `append_fact` dedups on normalized text + subject-key set
   against current facts; `record_mentions` is PK-deduped; `put_summary` /
   `put_people_dossier` / embeddings are upserts.

### FactExtractor (B-F)

1. **Gate**: `turns_since_watermark(familiar_id, limit=batch_size)` (oldest
   first, after `memory_writer_watermark.last_written_id`, 0 when unset).
   If fewer than `batch_size` turns returned → return: no LLM call, **no
   watermark write**. `batch_size` is both the trigger threshold and the
   per-tick cap; a sub-batch tail waits indefinitely (no timeout flush).
2. **Activity-return skip**: turns with `mode == "activity_return"` are
   filtered out of the LLM batch entirely (self-generated fiction; the
   activity engine already recorded a mechanical event-fact). Skip keys on
   `turns.mode`, never on the `"[returned from "` content prefix — ordinary
   user text carrying that prefix still extracts. If the whole batch is
   activity-return, no LLM call is made but the watermark still advances.
3. **Dream turns**: `mode == "sleep_return"` turns stay IN the batch;
   their ids form `dream_ids` used by the dream clause + rail below.
4. **Participants manifest** (`_build_participants`): map canonical_key →
   display, batch-first: (a) every batch turn's author, resolved via
   `HistoryStore.resolve_label(canonical_key, guild_id=turn.guild_id,
   familiar_id)`; per-channel guild = the LAST turn's guild_id seen for that
   channel in the batch; (b) widened per batch channel with
   `recent_distinct_authors(familiar_id, channel_id, limit=participants_max)`,
   skipping keys already present, hard-capped at `participants_max` TOTAL
   entries (assert: exactly 30 manifest lines with 50 priors + 1 batch
   author at the default cap). Widening is per-channel — authors from
   channels not in the batch never leak in.
5. **Self subject**: after building the manifest, the reserved key
   `ego:<familiar_id>` (`identity.ego_canonical_key`) is inserted mapping to
   `familiar_display_name` — so the manifest may hold `participants_max + 1`
   entries. The system prompt teaches the model to file the familiar's OWN
   narrative/choices/stances under that key ("only exception to 'not about
   you'") while still banning self-capability statements.
6. **Prompt shape**: system = intro + optional self clause + optional dream
   clause + `render_contract(_FACT_SCHEMA)` + claims/fiction guidance; user =
   `Participants (canonical_key — current display name):` block of
   `- <key> — <display>` lines, blank line, `Turns (id prefixed):` then
   `- id=<id> [<who>] <content>` where `<who>` = `author.label` (not
   display_name) or `role` for author-less turns. Tests pin prompt content:
   the words "self-capability"/"your own", "claim", "fiction",
   "running joke", "impersonat…", "distinct", "trivia", "subjectless",
   "valid_from", "valid_to" + ("outdated" or "replaced") + "supersed…",
   "importance" with "1" and "10".
7. **Dream clause**: rendered only when `dream_ids` nonempty. The
   config-sourced template (`[prompt].dream_extraction_clause`) is filled via
   `fill_placeholders` (02) with `self_name`, `self_key`, `ids` (sorted,
   comma-joined) — unknown `{tokens}` and stray braces pass through
   verbatim, never raising. Empty template → no clause.
8. **Reply parsing**: `request_structured` with `_FACT_SCHEMA` (root=array;
   fields text, source_turn_ids, optional subject_keys / valid_from /
   valid_to / importance). `_normalize_fact_items`: non-list → `[]`;
   non-dict items skipped; source ids accepted from ints and digit-strings
   (leading `-` tolerated by the digit test; bools rejected because
   `str(True)` isn't a digit string); subject_keys keep only strings.
9. **Per-fact source validation**: keep only ids present in the batch;
   empty result → fallback to the WHOLE batch id set minus `dream_ids`
   (or all batch ids if every turn was a dream). Fallback comes from a
   Python `set` → ordering is unspecified; the Rust port should pick a
   deterministic order (sorted ascending is safe — only membership is
   test-pinned). `channel_id` for the fact = channel of `source_ids[0]`;
   default `valid_from` = timestamp of `source_ids[0]`.
10. **Self-capability post-filter**: drop (count + DEBUG log) any fact whose
    stripped text matches, case-insensitively, either (a) the first-person /
    third-person-generic prefix regex — `i can(not|'t| not)?`, `i do(n't|
    not)`, `i am (not|unable)`, `i'm (not|unable)`, `i have no`, `as a(n)?
    (ai|assistant|language model|llm)`, `the (assistant|ai|familiar|model|
    bot)` — or (b) the display-name inability regex `^\s*<escaped display
    name>` followed by `cannot\b | can't\b | can not\b | is unable\b |
    has no\b`. The name regex is inability-ONLY: "cancelled", "candidly",
    "is not fond", "doesn't trust", "can sing" must all survive
    (word-boundary + narrow verb list; regression-pinned).
11. **Subject resolution**: soft-validate `subject_keys` against the
    manifest — unknown keys dropped silently, duplicates deduped preserving
    first occurrence, `display_at_write` taken from the manifest (never from
    LLM echo). Empty/missing subject_keys → fact stored with empty subjects.
12. **Dream claim-discipline rail (code, not prompt)**: if ANY source id is
    in `dream_ids`, subjects are FORCED to `[FactSubject(ego_key,
    display_name)]` regardless of what the model tagged, and unless the text
    already contains "dream" (case-insensitive substring) it is prefixed
    `"<DisplayName> dreamed that <text>"`. A config clause override can
    change phrasing but never bypasses this rail.
13. **Persist + mention mirror**: `append_fact(familiar_id, channel_id,
    text, source_turn_ids, subjects, valid_from, valid_to, importance)` —
    importance passed through un-clamped (store clamps to [1,10]; the
    parse layer maps bool→None, float→trunc-int, numeric string→int,
    garbage→None). Then for the fact's NON-ego subject keys only, call
    `record_mentions(turn_id=tid, canonical_keys=keys)` for every source
    turn — bridges bare-text name references into the same `turn_mentions`
    index Discord pings populate (PeopleDossierLayer read path, 05). Ego
    key is never mirrored (it is always-injected by the layer; mirroring
    would burn a `max_people` slot). Mirror is idempotent and coexists with
    prior ping rows (reads come back sorted by canonical_key).
14. **Watermark**: at the end of a successful tick, always
    `put_writer_watermark(last_written_id = new_turns[-1].id)` — including
    the malformed-JSON / zero-fact / all-skipped cases, and including
    filtered activity-return turns (their ids advance past too). NOT reached
    when an exception escapes mid-tick (LLM transport error, DB error) —
    that batch retries next tick. Partial replay after such a crash is safe
    via `append_fact` text-dedup and idempotent `record_mentions`.
15. **Manifest/turn ordering**: dict insertion order is meaningful (batch
    authors first, then widened, self key last); turns render oldest-first.

### SummaryWorker (B-S)

1. **Single index**: writes ONLY the focus-stream rolling summary, stored
   in `summaries` under `channel_id = FOCUS_STREAM_CHANNEL_ID = -1`. It no
   longer writes per-channel summaries (regression-pinned).
2. **Composite watermark**: cursor = `(last_consumed_at, last_summarised_id)`
   from the prior `SummaryEntry` (`("", 0)` cold-start / `last_consumed_at
   or ""` when prior row is legacy-NULL). Fetch via
   `consumed_turns_after(after_consumed_at, after_id,
   limit=max(turns_threshold, backfill_cap))`. Watermarking on
   `consumed_at`, not id, is load-bearing: a staged turn promoted late
   carries an old id but fresh `consumed_at`, and an id cursor would skip
   it forever. Cursor is exclusive: `consumed_at > after` OR (equal AND
   `id > after_id`). Staged (unconsumed) turns are invisible.
3. **Gate**: fewer than `turns_threshold` new consumed turns → no-op (no
   LLM call, no write). First run bounded by `backfill_cap` per tick (500
   seeded turns → tick 1 summarises ids 1..200, tick 2 → 201..400, …).
4. **Compounding prompt**: system asks for a 3-5 sentence retrieval-friendly
   summary preserving proper nouns / commitments / open questions; user body
   is `Previous summary:\n<prior>` + `\nNew turns:` when a prior exists
   (else `Turns:`), then `- [#<channel_id> <who>] <content>` lines with
   `<who>` = `author.display_name` or role.
5. **Empty-reply guard**: `reply.content_str.strip()` empty → return WITHOUT
   writing — watermark not advanced, so the same (growing) window retries
   next tick. (Contrast with extractor/reflection: summaries compound, so a
   dropped window would lose content.)
6. **Write**: `put_summary(channel_id=-1, last_summarised_id=last.id,
   summary_text=text, last_consumed_at=last.consumed_at.isoformat() or
   None)` — upsert, one row per familiar.

### PeopleDossierWorker (B-D)

1. **Scan**: `subjects_with_facts(familiar_id)` → `{canonical_key:
   max(current fact id)}` (superseded facts excluded — a subject whose only
   facts are retired stops being a refresh candidate). Subjects are
   processed sequentially within the tick, in map iteration order.
2. **Per-subject gate**: skip when `latest_fact_id <= prior.last_fact_id`
   (0 when no dossier row). Then fetch only the NEW window:
   `facts_for_subject(canonical_key, min_id_exclusive=prior_wm)` (ASC by
   id, current facts only). Empty window → return (no write).
3. **Compounding**: prior dossier text is fed back in the prompt
   (`Previous dossier:\n<prior>` + `\nNew facts:`; cold start `Facts:`),
   i.e. same shape as SummaryWorker; dossier cost stays bounded because
   only new facts ride each refresh.
4. **Self-dossier importance filter** (`is_ego_key(subject)`): keep facts
   with `importance is None or >= SELF_DOSSIER_MIN_IMPORTANCE (5)`; drop
   explicitly-low "texture" facts (dream pass writes those at 2-3; they
   stay in the DB, RAG-recallable, but must not flood the always-injected
   self-record). Non-self subjects get NO importance filtering.
5. **Self-dossier ordering + annotation**: kept facts stable-sorted
   importance-descending with NULL ranked at the keep threshold (5), so
   seeding [5, 9, 7, None] renders 9, 7, 5, None (insertion order preserved
   within a tier). Scored facts render `- (importance N) <text>`; NULL
   renders plain. Non-self bodies always render plain (no tags).
6. **All-texture window**: when the self filter empties a nonempty window
   AND a prior dossier exists, advance the watermark
   (`put_people_dossier(last_fact_id=latest_fact_id,
   dossier_text=prior.dossier_text)`) keeping the prior text — otherwise
   every tick would re-filter the same window. No prior → do nothing.
7. **Display resolution**: self subject → `familiar_display_name` directly
   (there is no accounts row for `ego:<id>`; `resolve_label` would degrade
   to the raw id tail); others → `await store.resolve_label(canonical_key,
   guild_id=None, familiar_id)`. The self prompt must address the familiar
   by name and never leak the `ego:` key (test-pinned).
8. **Prompt split**: self header = "evolving self-record", keeps settled
   opinions/stances/feelings, drops only momentary reactions, explains the
   importance weighting ("weight higher-importance stances more heavily")
   and reconciles contradictions toward newer evidence; person header =
   short retrieval-friendly dossier, "Drop transient feelings", no
   importance/bias language. Tests grep for "opinion|stance" present and
   "drop transient feelings" ABSENT in the self header, and
   "weight higher-importance" ABSENT in the person header.
9. **Reply handling**: strip whitespace; for self, additionally delete every
   `(importance \d+)\s*` occurrence (writer echo) then re-strip. Empty
   result → return WITHOUT writing (a blank reply must not clobber an
   existing dossier; watermark also stays put, so the window retries).
10. **Write**: `put_people_dossier(canonical_key, last_fact_id =
    latest_fact_id, dossier_text=text)` — watermark is the subject's max
    fact id from the scan (not the max of the filtered window).
11. **Cross-worker contract**: `HistoryStore.supersede` (03) DELETES the
    retired facts' subjects' dossier rows; this worker's `prior=None` path
    is what rebuilds them clean on the next tick. Do not "optimise" by
    caching dossiers in worker memory.

### ReflectionWorker (B-R)

1. **Gate is id-delta arithmetic, not row count**: fire when
   `latest_id(familiar_id) - prior_turn_wm >= turns_threshold`, where the
   prior watermark comes from `latest_reflection_watermarks` (explicit
   `reflection_watermark` table row, falling back to the newest reflection
   row's snapshot for pre-watermark DBs, then `(0, 0)`). `latest_id` None
   or <= 0 → no-op.
2. **Watermark ALWAYS advances — try/finally**: `set_reflection_watermark
   (last_turn_id=latest_turn, last_fact_id=latest_fact)` runs in a
   `finally` around the tick body. Consequence (deliberate): empty replies,
   malformed JSON, all-filtered items, AND mid-tick transport errors or
   cancellation all advance the watermark; skipped windows are skipped
   forever. Reflections are best-effort synthesis — the guardrail exists so
   a no-substance reply can't pin the worker to an ever-growing window
   (100k-token prompt balloon).
3. **Window**: `turns_in_id_range(min_id_exclusive=prior_wm,
   max_id_inclusive=latest_turn)`; empty → return (watermark still
   advances via finally). If larger than `max_turns_per_tick`, keep the
   most-recent TAIL (`[-cap:]`); older turns are skipped, not deferred
   (second runaway-cost guardrail; pinned: 500 seeded turns → exactly 50
   `- id=` lines, newest 50).
4. **Context facts**: `recent_facts(limit=recent_facts_limit)` (current
   only, newest first) render after the turns as `Recent facts (id
   prefixed):` `- id=<id> <text>` (section omitted when empty). Turn lines
   are `- id=<id> [<who>] <content>` with `<who>` = display_name or role.
5. **Schema per call**: array of `{text, cited_turn_ids, cited_fact_ids?}`
   with constraints `"Reply with at most <max_reflections_per_tick>
   items."`, `"Cite at least one turn id or fact id per reflection."` and
   empty note `"If nothing of substance is happening, reply with []."`;
   requested via `request_structured` (retry-once).
6. **Citation validation**: cited turn ids must be in the CAPPED window's id
   set; cited fact ids are validated against ALL known fact ids for the
   familiar (including superseded — reflections may lean on older facts
   surfaced via dossiers) EXCEPT when the recent-facts context was empty, in
   which case the valid fact set is empty (no `all_fact_ids` query; all
   fact citations drop). Normalization here accepts `int` citations only
   (no string coercion — stricter than the extractor). Invalid ids are
   dropped from the row; a row with empty text OR zero surviving citations
   is skipped (uncited reflection = free-floating opinion). The `max
   items` constraint is prompt-only — the worker does NOT truncate extra
   items the model returns.
7. **Channel scoping**: `_dominant_channel` — single distinct channel → it;
   empty → None; mixed → most-frequent (Python `max` over dict items:
   first-encountered wins ties). Rust should preserve "most frequent,
   deterministic tiebreak".
8. **Write**: `append_reflection(channel_id, text, cited_turn_ids,
   cited_fact_ids, last_turn_id=latest_turn, last_fact_id=latest_fact)` —
   the snapshot ids are stored per row (render-time anchor for the 05
   ReflectionLayer) in addition to the separate watermark table.

### FactEmbeddingWorker (B-E)

1. **Implicit watermark**: no watermark table — `unembedded_facts(model=
   embedder.name, limit=batch_size)` (current facts lacking a
   `fact_embeddings` row for this model, ascending id) IS the watermark.
   Model swap / deleted side-index / new facts all converge on the same
   idempotent backfill. Empty → tick returns 0 with no embedder call.
2. **One embedder call per tick** covering the whole batch
   (`embed([f.text ...])`), then one `set_fact_embedding(fact_id, model=
   embedder.name, vector)` upsert per row — per-row persistence means an
   interrupted batch resumes where it stopped.
3. **Mismatch guard**: `len(vectors) != len(pending)` → WARNING log, return
   0, write NOTHING (a buggy backend must not corrupt rows or advance
   state; the same batch retries next tick).
4. **Return value**: count written (== batch size processed). Pinned batch
   progression for 5 facts at batch 2: ticks return (2, 2, 1, 0).
5. Cadence (15 s) intentionally matches the extractor so RAG sees fresh
   vectors quickly; embedding is cheap relative to the extraction LLM call.

### FactSupersedeWorker (B-P)

1. **In-memory watermark**: `_last_seen_fact_id` starts 0; `run()` calls
   `prime_watermark()` before the first tick, which SYNCHRONOUSLY reads
   `store.sync.latest_fact_id(familiar_id)` — a fresh deploy never burns
   LLM calls re-evaluating historical facts. State is lost on restart and
   re-primed to "latest", so facts appended while the process was down are
   never supersede-evaluated (accepted: system-time bookkeeping is
   best-effort).
2. **Candidate window**: `recent_facts(limit=batch_size,
   include_superseded=False)` (newest first), filtered to `id >
   _last_seen_fact_id`, then sorted ASCENDING so cascading retirements
   settle deterministically (oldest new fact evaluated first). Corollary:
   if more than `batch_size` new facts accumulate between ticks, the older
   ones fall outside the candidate window and are permanently skipped when
   the watermark advances.
3. **Watermark advance**: after evaluating, set `_last_seen_fact_id =
   max(candidate ids)` — even when the LLM output was garbage (shape
   degrade → no retirement) so a consistently-unparseable fact can't loop.
   Not reached if an exception escapes mid-tick (transport error →
   same candidates retry next tick). No-new-facts tick returns 0 with no
   LLM call and no watermark change.
4. **Per-fact evaluation** (`_evaluate`): subject-less facts are skipped
   outright (no priors to pair with, no LLM call). For each subject of the
   new fact: `facts_for_subject(canonical_key, include_superseded=False)`
   (ASC), exclude the new fact itself and any prior already shown for an
   earlier subject of this fact, keep the LAST `priors_max` (most recent);
   empty → next subject. One LLM call per (new fact × subject with
   nonempty unique priors).
5. **Reply contract**: object root `{"superseded_ids": [<id>...]}` with
   constraint "Only include ids from the Prior facts list below — do not
   invent ids." and empty note "Empty list when nothing is retired."
   Parsed via `coerce_positive_int_list` (02: bools rejected, digit strings
   accepted, non-positive and duplicates dropped) then intersected with the
   shown prior ids — hallucinated ids and the new fact's own id are thereby
   filtered (self-supersede impossible; test-pinned).
6. **Retirement**: `store.supersede(obsolete_facts=ids, new_fact=f_new)` —
   existing-id form (repoints old rows at the new fact; mints nothing).
   Per-id skip-and-record: priors already retired by a concurrent writer
   land in `result.skipped`, never raise. The tick's return counts only
   `len(result.superseded)` per call. Side effect owned by 03: supersede
   deletes affected subjects' `people_dossiers` rows (see B-D11).
7. **Prompt**: system persona defines "retired" = contradicted or directly
   replaced, NOT merely older/differently-worded; user lists
   `New fact (id=<id>): <text>`, blank, `Prior facts:` with
   `- id=<id>: <text>` lines.

### Cross-worker temporal contract

- **Two time axes, two writers** (memory-strategies.md): world-time
  `valid_from`/`valid_to` is set ONLY by the extractor (source-turn default
  + explicit LLM override on "as of …" phrasing); system-time
  `superseded_at`/`superseded_by` is set ONLY by the supersede worker via
  `store.supersede`. The extractor prompt actively steers the model away
  from using `valid_to` as a retirement marker. Never conflate in the port.
- Every fact reader in the system filters both axes for "current truth";
  these workers rely on `recent_facts` / `facts_for_subject` defaults doing
  that filtering.

## Data formats

No new tables — all storage shapes belong to 03 (`facts`, `summaries`,
`people_dossiers`, `reflections`, `fact_embeddings`,
`memory_writer_watermark`, `reflection_watermark`, `turn_mentions`). What
this subsystem OWNS is the LLM wire shapes:

- **Fact extraction reply** (root=array, rendered by `render_contract`):
  `[{"text": "<one sentence>", "source_turn_ids": [<id>...], "subject_keys":
  [<key>...], "valid_from": "<ISO-8601 timestamp>", "valid_to": "<ISO-8601
  timestamp>", "importance": <1-10>}, ...]` — followed by "Each item's
  fields:" bullets for every field with a description; optional fields
  carry an `(optional)` marker. `valid_from`/`valid_to` accepted as any
  `datetime.fromisoformat`-parseable string (date-only allowed; naive
  parses are assumed UTC). Unparseable → None (→ source-turn timestamp for
  valid_from).
- **Reflection reply** (root=array): `[{"text": "<one or two sentences>",
  "cited_turn_ids": [<id>...], "cited_fact_ids": [<id>...]}, ...]` plus the
  two constraints and empty-note from B-R5.
- **Supersede reply** (root=object): `{"superseded_ids": [<id>...]}`.
- **Summary / dossier replies**: free prose (no contract); consumed via
  `content_str.strip()`.
- Prompt line grammars (test-parsed, keep byte-exact): manifest
  `- <key> — <display>` (em-dash U+2014 with single spaces); turn lines
  `- id=<id> [<who>] <content>` (extractor, reflection) vs
  `- [#<channel> <who>] <content>` (summary); fact lines `- <text>` /
  `- (importance <N>) <text>` (dossier) / `- id=<id> <text>` (reflection) /
  `- id=<id>: <text>` (supersede priors).
- Sentinels: `FOCUS_STREAM_CHANNEL_ID = -1`; `turns.mode` tags
  `"activity_return"` / `"sleep_return"`; content display prefix
  `"[returned from "` (never used for logic here); ego subject key
  `"ego:<familiar_id>"` (`is_ego_key`: platform == "ego" and non-empty
  tail).

## Config knobs

All under `[providers.memory]` in `character.toml` (parsed by 02 into
frozen dataclasses; knob tables are accepted whether or not the projector
is enabled, so toggling keeps tuning):

```toml
[providers.memory]
projectors = ["rolling_summary", "rich_note", "people_dossier",
              "reflection", "fact_supersede"]      # default; order = spawn order
# "fact_embedding" is registered but opt-in (requires
# [providers.embedding].backend != "off"; factory raises ValueError otherwise)

[providers.memory.rolling_summary]
turns_threshold = 10
tick_interval_s = 5.0
# backfill_cap (worker default 200) is NOT exposed in TOML / factory

[providers.memory.rich_note]
batch_size       = 10     # threshold AND per-tick cap
tick_interval_s  = 15.0
participants_max = 30

[providers.memory.people_dossier]
tick_interval_s = 20.0

[providers.memory.reflection]
turns_threshold          = 20
max_reflections_per_tick = 3
max_turns_per_tick       = 50
recent_facts_limit       = 20
tick_interval_s          = 60.0

[providers.memory.fact_supersede]
batch_size      = 5
tick_interval_s = 60.0
priors_max      = 20
```

Also consumed (via `ProjectorContext`, 06): `[prompt].dream_extraction_clause`
— static prose template with `{self_name}`, `{self_key}`, `{ids}`
placeholders; the shipped default (data/familiars/_default/character.toml)
frames dream turns as fiction requiring the self key. Empty string → no
dream clause; overrides change phrasing only (code rail B-F12 unaffected).
Unknown projector names fail loudly at CONFIG LOAD (02 validates against
the registry), not at spawn. No env vars are read by this subsystem.

## Dependency edges

Imports (what 07 consumes):

| module | subsystem | what's used |
|---|---|---|
| `history.async_store.AsyncHistoryStore` | 03 | duck-typed async proxy (`__getattr__` → `run_in_executor`, 4-thread pool); `.sync` escape hatch (extractor manifest, supersede prime) |
| `history.store` | 03 | `HistoryTurn`, `Fact`, `FactSubject`, `SummaryEntry`, `PeopleDossierEntry`, `Reflection`, `SupersedeResult`, `FOCUS_STREAM_CHANNEL_ID`; methods: turns_since_watermark, put/get_writer_watermark, append_fact (text-dedup!), record_mentions, resolve_label, recent_distinct_authors, consumed_turns_after, get/put_summary, subjects_with_facts, facts_for_subject, get/put_people_dossier, latest_id, latest_fact_id, latest_reflection_watermarks, set_reflection_watermark, turns_in_id_range, recent_facts, all_fact_ids, append_reflection, unembedded_facts, set_fact_embedding, latest_embedded_fact_id, supersede |
| `llm` | 08 | `LLMClient.chat`, `Message` (role/content, `content_str`) |
| `structured_request` | 08 | `Schema`, `Field`, `render_contract`, `request_structured` (retry-once, transport errors propagate) |
| `structured_output` | 02 | `coerce_positive_int_list` (supersede); `coerce_json` indirectly via 08 |
| `identity` | 02 | `ego_canonical_key`, `is_ego_key` |
| `prompt_fill` | 02 | `fill_placeholders` (crash-safe dream clause) |
| `activities` | 11 | `ACTIVITY_RETURN_MODE`, `SLEEP_RETURN_MODE` constants (`RETURN_TURN_MARKER_PREFIX` only in tests) |
| `embedding.protocol.Embedder` | 04 | batch `embed`, `name` as storage key, `dim` (log only) |
| `log_style`, `diagnostics.spans` | 01 | colored kv logging, `@span` timing |
| `config.MemoryProvidersConfig` + per-worker knob dataclasses | 02 | constructor knobs (threaded by 06 factories) |

Imported by: `processors/projectors.py` (06 — registry/factories, the only
production constructor call sites); `commands/run.py` (10) spawns via the
registry. Read-side consumers of the side-indices (05 layers, 04 sleep) have
no import edge into this subsystem.

## Test inventory

| test file | behaviors pinned | portability |
|---|---|---|
| `tests/test_fact_extractor.py` (1657 loc) | full-batch gate; watermark advance incl. bad-JSON; per-tick batch slicing; self-capability drop (first-person + named-inability, false-positive keep-list); prompt content rails (claims/fiction/impersonation/trivia/valid_to/importance wording); activity-return skip by mode (prefix-only content still extracts; all-return batch → no LLM call + watermark advance; fallback sources exclude return turn); participants manifest (key+display present, widening from prior channel authors, per-channel isolation, cap=30 counted by `- discord:` lines); subject_keys → FactSubject with manifest display, unknown-key drop, missing → empty; ego manifest entry + self-narrative routing + no ego turn_mentions; mention mirroring per source turn, idempotent w/ prior pings, sorted reads; bi-temporal defaults (valid_from = first-source timestamp, valid_to None, ISO override); importance persist/None/store-clamp(99→10, -3→1, garbage→None); dream suite (clause interpolation from real default config, stray-brace degrade, forced self-subject + "dreamed that" framing, already-framed verbatim, mixed-source counts as dream, fallback excludes dream id, watermark over dream turn, no clause without dream turns) | logic-portable (scripted LLM stub + in-memory HistoryStore; dream tests load `data/familiars/_default/character.toml` — keep that fixture path working) |
| `tests/test_summary_worker.py` | focus-stream write under channel -1 with last_consumed_at; threshold no-op; staged turns invisible; prior summary compounds into second prompt; backfill_cap slices first runs (200/tick over 500); late-promoted turns picked up via consumed_at cursor; per-channel summaries no longer written | logic-portable (needs store with `promote_staged_turns` + `consumed` flag) |
| `tests/test_people_dossier_worker.py` | create-on-first-fact (watermark=fact id); unchanged watermark → zero LLM calls; compounding (prior text + only-new facts in prompt); multi-subject tick; blank reply keeps prior text; self-dossier via display name (no `ego:` leak); echoed `(importance N)` stripped; self header keeps opinions / no "drop transient feelings"; low-importance texture excluded, NULL kept; importance-desc stable ordering incl. NULL-at-5 band; `- (importance 9) …` annotation + "weight higher-importance" header; NULL renders untagged; non-self: no tags, no bias clause, low importance kept; subject-less facts → no LLM call | logic-portable |
| `tests/test_reflection_worker.py` | threshold no-op; write w/ citations + snapshot ids; no refire until id-delta re-crossed; multiple rows per reply; invalid citation dropped but row kept; empty-text row skipped; malformed reply tolerated; watermark advances on `[]` AND on malformed (note: 4 scripted bad replies = 2 structured attempts × retry); window cap = exactly 50 newest `- id=` lines | logic-portable |
| `tests/test_fact_supersede_worker.py` | prime → no-new no-op (zero LLM calls); flagged prior gets superseded_by/superseded_at; empty list untouched; hallucinated id ignored; bad JSON swallowed; new-fact self-id filtered (both stay current); watermark makes second tick a no-op; subject-less fact → no LLM call | logic-portable |
| `tests/test_fact_embedding_worker.py` | batch embed in one call; batch_size caps writes; idempotent re-tick (0, no embedder call); resume progression (2,2,1,0); real HashEmbedder round-trip (dim 64 vectors readable); empty store no-op; short-vector backend → 0 written, nothing persisted | logic-portable (duck-typed `_CountingEmbedder` stub → Rust needs a plain trait impl) |
| `tests/test_memory_projectors.py` | registry defaults/order/unknown-raises/third-party registration; per-worker knob threading asserted via PRIVATE fields (`w._batch_size` etc.); fact_embedding opt-in + embedder-required error; config-load projector validation | 06's remit; knob asserts need Rust-visible getters or behavior-level asserts |
| `tests/test_config.py` (subset) | dream_extraction_clause placeholders present in default, override, "no in-code copy" (`DREAM_EXTRACTION_CLAUSE_DEFAULT` must not exist on the module); knob dataclass defaults + TOML round-trip | 02's remit; the "no module constant" assert is Python-specific-skip |

Conformance nuance for the Rust harness: scripted-LLM stubs pop one canned
reply per `chat` call and return a benign default (`"[]"` /
`'{"superseded_ids": []}'` / prose) when exhausted; structured-request
retries consume extra replies — tests count `llm.calls`, so the port's
retry budget must match (1 corrective re-ask).

## Rust port notes

- **Task model**: each worker = one `tokio::task` running
  `loop { if let Err(e) = tick().await { warn!(...) } sleep(interval).await }`
  with cancellation via `CancellationToken`/`JoinSet` abort inside the
  supervisor (10). Python's `except CancelledError: raise` maps to "let
  abort/cancel propagate; never catch-all a cancellation". Reflection's
  finally-advances-watermark MUST survive cancellation semantics you choose
  — if you use `tokio::select!` with a token, run the watermark write in
  the drop/cleanup path or accept the documented skip; simplest faithful
  mapping is abort-only-at-await + a `scopeguard`-style async finally
  (i.e., write the watermark before returning on every path, and on error
  paths, mirroring the Python `try/finally`).
- **Error taxonomy**: split worker errors into `Transport(anyhow)` (raise →
  retry same window) vs `Shape` (degrade → empty + advance watermark). The
  Python code encodes this implicitly (exceptions vs `StructuredReply.ok`);
  make it explicit in the Rust `structured_request` return type.
- **AsyncHistoryStore is duck typing at its worst**: `__getattr__` proxies
  ANY sync method through a 4-thread executor. In Rust this disappears —
  give 03 an async store trait (or `spawn_blocking` internally) and have
  workers call it uniformly. The two `.sync` escape hatches (extractor's
  manifest building, supersede's `prime_watermark`) currently run blocking
  SQL on the event loop / caller thread — a known wart; port them as normal
  async calls, not as a faithful blocking transliteration.
- **Regexes** (`regex` crate, case-insensitive flags): the self-capability
  prefix regex is `re.match` (anchored at start, after `^\s*`) — use
  anchored `Regex::is_match` with `^`; the display-name variant embeds
  `regex::escape(display_name)`; the importance-tag strip is a global
  `replace_all`. Port the alternations exactly — the false-positive
  keep-list in tests is the oracle. `str.title()` for the display-name
  default: Python title-cases per word boundary; a familiar_id like
  `"my-fam"` becomes `"My-Fam"` — replicate or restrict ids (02 slugs are
  lowercase alnum+dash, so a simple per-segment capitalize suffices).
- **Datetime**: `datetime.fromisoformat` accepts date-only and offset
  forms; use `chrono`'s RFC3339 parse with a date-only fallback
  (`NaiveDate` → midnight UTC) and assume-UTC for naive values.
  `consumed_at` watermark comparisons are STRING comparisons over ISO
  timestamps in SQL — keep the stored format lexicographically ordered
  (RFC3339 UTC with fixed precision).
- **Ordering determinism**: three spots rely on Python dict insertion order
  or set iteration — participants manifest (insertion order: batch authors,
  widened, self last; keep an `IndexMap`), the whole-batch source-id
  fallback (unordered `set`; pick sorted-ascending in Rust and note the
  Python behavior was unspecified), and `_dominant_channel` tie-break
  (make it deterministic, e.g. lowest channel id on tie).
- **Numeric coercion**: Python `bool` is an `int` subclass — the guards
  (`isinstance(raw, bool)` before `int`) exist to reject `true`→1. With
  `serde_json::Value` this is free (Bool vs Number are distinct); just
  don't add bool→int conversions. Float importance truncates (`int(9.7)` →
  9, toward zero).
- **Suggested crates**: `tokio` (tasks, sleep), `tokio-util`
  (CancellationToken), `serde_json` (reply parsing), `regex`, `chrono`,
  `indexmap` (manifest), `tracing` (spans — `@span("facts.tick")` maps to
  `#[tracing::instrument]` or a manual timed span at DEBUG), `scopeguard`
  or explicit match-all-paths for the reflection finally.
- **Redesign candidates** (behavior-preserving simplifications): (a) the
  `MemoryProjector` Protocol becomes `#[async_trait] trait MemoryProjector
  { fn name(&self) -> &'static str; async fn run(&self, cancel: Token); }`
  with `tick` exposed for tests; (b) prompt builders are pure
  string-assembly — keep them as free functions returning `Vec<Message>` so
  the conformance tests can snapshot them; (c) supersede's in-memory
  watermark could be persisted, but DON'T — restart-skip is documented
  behavior and tests pin `prime_watermark`; (d) do not merge the extractor's
  two watermark semantics ("advance only on success") with reflection's
  ("advance always") — they differ on purpose.
- **Test seams**: every LLM test double subclasses `LLMClient` but only
  `chat` matters; the embedding tests use a bare struct with `name`/`dim`/
  `embed`. Design the 08 `LlmClient` and 04 `Embedder` traits so a 5-line
  scripted stub satisfies them (no builder ceremony), or the ~90 ported
  tests in this subsystem get painful.
