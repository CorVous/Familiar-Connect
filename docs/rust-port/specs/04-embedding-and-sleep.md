# 04 embedding-and-sleep — port spec

Source files: `src/familiar_connect/embedding/{__init__,protocol,factory,hash,fastembed}.py`,
`src/familiar_connect/sleep/{__init__,consolidation,apply,opinion_formation,maintenance}.py`
(~1.9k LOC total). Architecture docs: `docs/architecture/sleep.md` (normative for rails,
watermark axes, concurrency story), `docs/architecture/memory-strategies.md` §"Embeddings (M6)".

## Role

Two loosely-related halves that share the "background LLM/compute behind a seam" shape:

1. **Embedding** — the `Embedder` protocol (text → fixed-dim float vector) plus a factory
   registry with three built-ins: `off` (returns no embedder), `hash` (deterministic no-deps
   locality-hash baseline), `fastembed` (ONNX sentence embedder, optional extra). Consumers are
   `FactEmbeddingWorker` (07, writes `fact_embeddings`) and `RagContextLayer` (05, cue embedding
   at rerank).
2. **Sleep passes** — the nightly maintenance run: a *consolidation* pass (LLM proposes fact
   retire/rewrite over the whole day-window; code-enforced rails validate; supersede-only apply)
   and an *opinion-formation* pass (two-stage per-day-stance → synthesis, grounded-by-construction,
   minting `ego:` facts). A small registry (`MaintenancePass`) sequences them and threads
   consolidation's retirements into the opinion pass's known-bits deny-list. The activities
   engine (11) fires the whole run as a background task on sleep departure.

The governing design rule everywhere: **the model proposes, code decides**. Prompt text is
config; every safety rail runs in code after the reply, so no prompt override can weaken one.

## Public API surface

### embedding/protocol.py — `Embedder` (Protocol; THE swappable seam → Rust trait)

```
name: str          # backend label persisted with each vector ((fact_id, model) row key)
dim: int           # fixed dimensionality; storage assumes one fixed-size vector per row
async embed(texts: list[str]) -> list[list[float]]
```

Contract: output order matches input; `len(out) == len(texts)` always; empty input → empty list
without side effects; blank strings map to a stable sentinel (zero vector), never raise. Must be
safe to call from a background async task; CPU-bound work belongs off the reactor
(`asyncio.to_thread` → `tokio::task::spawn_blocking`).

### embedding/hash.py — `HashEmbedder`

- `HashEmbedder(*, dim: int = 256)`; `dim < 8` → `ValueError` (message contains `>= 8`).
- `name = "hash-v1"` (class-level constant — version it if the algorithm ever changes).
- Fully deterministic across processes/runs/platforms. Algorithm (pinned by tests, must be
  bit-for-bit portable):
  1. tokenize with regex `\w+` (Unicode word chars), `casefold()` each token;
  2. per token: `digest = BLAKE2b(token_utf8, digest_size=4)`;
     `idx = u16::from_be_bytes(digest[0..2]) % dim`;
     `sign = +1.0 if digest[2] & 1 == 0 else -1.0`; `vec[idx] += sign`;
  3. L2-normalize; if norm is exactly 0 (no tokens / all-cancelling) return the raw zero vector.

### embedding/fastembed.py — `FastEmbedEmbedder`

- `FastEmbedEmbedder(*, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None)`.
- `name = f"fastembed:{model_name}"` — model identity in the name splits storage rows per model.
- `dim` from a static known-dims table at construction: bge-small/base/large = 384/768/1024,
  `sentence-transformers/all-MiniLM-L6-v2` = 384, `intfloat/e5-small-v2` = 384,
  `intfloat/multilingual-e5-small` = 384; unknown model → `dim = 0` until the first real vector
  probes it. A nonzero pre-known dim is **never** overwritten by the probe (guard is `if vectors
  and not self.dim`).
- Lazy model load on first non-empty `embed()`, double-checked under an `asyncio.Lock`; exactly
  one model construction even under concurrent first calls. Both the import and model
  construction happen in a worker thread. `embed([])` returns `[]` without loading.
- Missing `fastembed` package at load time → `RuntimeError` whose message contains
  `local-embed` (install hint). Model kwargs: always `model_name`; `cache_dir` only when set
  (unset must fall through to the library default `~/.cache/fastembed` — tests pin kwarg absence).
- `embed` runs the synchronous `model.embed(texts)` (yields ndarray per text) in a worker thread
  and converts to plain float lists.

### embedding/factory.py

- `register_embedder(name: str, factory: Fn(EmbeddingConfig) -> Option<Embedder>)` — module-global
  registry; re-registration overwrites silently.
- `known_embedders() -> set[str]` — consulted by config parsing (02) at load time.
- `create_embedder(config: EmbeddingConfig) -> Embedder | None` — unknown backend →
  `ValueError` `unknown embedding backend {backend!r}; valid: <sorted, comma-joined or "(none)">`.
- Built-ins registered at import: `"off"` → `None`; `"hash"` → `HashEmbedder(dim=config.dim)`;
  `"fastembed"` → probe that the `fastembed` module is importable (import probe only, NOT model
  load) and raise `RuntimeError` mentioning `local-embed` if absent — deliberate fail-fast at
  startup vs. crashing mid-turn — then `FastEmbedEmbedder(model_name=config.fastembed_model,
  cache_dir=config.fastembed_cache_dir)`.

### sleep/consolidation.py

Data types (all frozen/immutable):

- `ConsolidationWindow { familiar_id, facts: (Fact,...), turns: (HistoryTurn,...),
  prior_watermark: SleepWatermark|None, max_fact_id, max_turn_id, facts_truncated,
  turns_truncated }`
- `RetireAction { fact_ids: (int,...), reason: str }`
- `RewriteAction { old_fact_ids: (int,...), new_text: str, subject_keys: (str,...), reason: str }`
- `RejectedAction { kind: "retire"|"rewrite", payload: dict, rail: str, detail: str }`
- `ConsolidationPlan { familiar_id, retire, rewrite, rejected, new_last_fact_id,
  new_last_turn_id, facts_considered=0, facts_truncated=0, turns_considered=0,
  turns_truncated=0, notes: (str,...) }` + property `mutated_count` = Σ retire ids + Σ rewrite
  old ids.

Functions:

- `async gather_window(store, *, familiar_id, facts_max=500, turns_max=400) -> ConsolidationWindow`
- `build_prompt(window, *, self_key, system="") -> list[Message]`
- `parse_actions(reply: str) -> (retire_raw: list[dict], rewrite_raw: list[dict])`
- `reply_parse_failed(reply: str) -> bool`
- `validate(window, *, retire_raw, rewrite_raw, self_key, cap=50, notes=()) -> ConsolidationPlan`
  — pure, no I/O
- `async plan_consolidation(store, llm, *, familiar_id, facts_max=500, turns_max=400, cap=50,
  system="") -> ConsolidationPlan` — read-only end to end
- Constants: `DEFAULT_FACTS_MAX=500`, `DEFAULT_TURNS_MAX=400`, `DEFAULT_RETIRE_CAP=50`.

### sleep/apply.py

- `ApplyReport { retired_fact_ids: (int,...), rewritten: ((old_ids, new_id),...),
  watermark: (last_fact_id, last_turn_id), skipped: ((kind, fact_id, reason),...) }`
- `async apply_consolidation(store, plan, *, familiar_display_name=None) -> ApplyReport` — the
  only DB-mutating consolidation entry point.

### sleep/opinion_formation.py

- `DayBatch { date: "YYYY-MM-DD", turns }` + `turn_ids: frozenset[int]`,
  `self_turn_ids: frozenset[int]` (turns with `role == "assistant"`).
- `OpinionWindow { familiar_id, days: (DayBatch,...), prior_watermark, max_turn_id }`
- `StanceMoment { text, date, turn_ids: (int,...) }`
- `OpinionFact { text, source_turn_ids, valid_from_date: "YYYY-MM-DD", self_grounded: bool,
  importance: int }`
- `RejectedOpinion { payload, rail, detail }`
- `OpinionPlan { familiar_id, opinions, rejected, flags: (str,...), new_last_turn_id,
  days_considered=0, stance_moments_considered=0, notes=() }`
- `OpinionApplyReport { recorded: ((text, fact_id),...), watermark: int }`
- `bucket_by_day(turns, tz_name) -> list[DayBatch]` (pure)
- `async gather_days(store, *, familiar_id, display_tz) -> OpinionWindow`
- `async extract_stance_moments(llm, day, *, self_name, denylist=(), system="") -> list[StanceMoment]`
- `async synthesize(llm, stance_moments, *, self_name, prior_self_dossier=None, system="") -> list[dict]`
- `validate_opinions(raw, *, stance_moments, window, cap=60, notes=()) -> OpinionPlan` (pure)
- `async plan_opinions(store, llm, *, familiar_id, display_tz, self_name, denylist=(),
  prior_self_dossier=None, cap=60, stance_system="", synthesis_system="") -> OpinionPlan`
- `async apply_opinions(store, plan, *, familiar_display_name=None) -> OpinionApplyReport`
- Constants: `DEFAULT_OPINION_CAP=60`; `DEFAULT_TURNS_MAX_PER_DAY=600` is **declared but unused**
  (dead — no per-day turn cap is implemented; decide deliberately whether to port or drop it).
- Test-touched internals (keep as crate-visible for tests): `_render_turn`,
  `_build_stance_prompt`, `_build_synthesis_prompt`, `_coerce_importance`.

### sleep/maintenance.py — pass registry + orchestrators

- Pass names: `CONSOLIDATION_PASS = "consolidation"`, `OPINION_PASS = "opinion"`,
  `DEFAULT_PASSES = ("consolidation", "opinion")` (order is a contract).
- `SleepPromptText { consolidation_system="", stance_system="", synthesis_system="" }` (frozen)
  + `from_config(...)` relaying `[prompt]` strings verbatim. Empty defaults are intentional: the
  production prose lives ONLY in `data/familiars/_default/character.toml` — no in-code copy.
- `MaintenanceContext { store, llm, familiar_id, display_name: str|None, display_tz: str,
  apply: bool, facts_max=500, turns_max=400, retire_cap=50, opinion_cap=60,
  prompts: SleepPromptText }` (frozen; built once per sleep).
- `MaintenanceRun { denylist_fact_ids: Vec<int>, opinion_plan: Option<OpinionPlan> }` (mutable
  accumulator threaded pass→pass).
- `MaintenancePass` (Protocol seam → Rust trait): `name: str`,
  `async run(run: &mut MaintenanceRun) -> object`.
- `register_pass(name, factory)` / `known_passes()` / `create_passes(*, names, context)`
  (instantiates in `names` order; unknown name → `ValueError` naming the bad name and listing
  valid ones) / `async run_passes(passes, run=None) -> MaintenanceRun` (strictly sequential; NO
  error guard — exceptions propagate to the engine's guard).
- Free orchestrators (used by passes and available to ad-hoc callers):
  - `async execute_consolidation(*, store, llm, familiar_id, familiar_display_name, apply,
    facts_max=500, turns_max=400, cap=50, system="") -> ConsolidationPlan`
  - `consolidation_denylist_ids(plan) -> list[int]` — **accepted** retire ids + rewrite old ids,
    in that order (rejected proposals excluded).
  - `async execute_opinion_formation(*, store, llm, familiar_id, familiar_display_name,
    display_tz, apply, denylist=(), cap=60, stance_system="", synthesis_system="") -> OpinionPlan`
- Built-in passes `ConsolidationPass` / `OpinionFormationPass` registered at import.

## Behaviors & invariants

### Embedding

1. `HashEmbedder` output is deterministic and process/machine-independent: same `(dim, texts)` →
   identical vectors forever ("hash-v1" pins the algorithm). Casefold makes `"Café Latte"` and
   `"café LATTE"` identical vectors. Tests also pin: blank text → exact zero vector; output count
   and per-vector length always match input count and `dim`; disjoint-token texts have cosine
   < 0.1 at dim 256; overlapping-token pair scores above disjoint pair.
2. Note Python `\w+` with `re.UNICODE` matches `[letter, digit, underscore]` including marks —
   Rust `regex` crate `\w` with Unicode on is equivalent enough for tests, but the fixture
   vectors themselves must be regenerated only if you accept breaking stored `hash-v1` vectors;
   safer to bump the name to `hash-v2` if tokenization differs at all (stored vectors are keyed
   by `model` name, so a rename triggers a clean idempotent backfill instead of a corrupt mixed
   similarity space).
3. `FastEmbedEmbedder` lazy-load invariants: no load on construction or on empty input; exactly
   one underlying model built regardless of call concurrency (double-checked lock); a load
   failure surfaces as `RuntimeError` naming `local-embed` on every embed attempt (the model
   stays unloaded, retryable).
4. `dim` may legitimately be `0` (unknown model, pre-first-embed) — downstream code
   (`FactEmbeddingWorker` logging) tolerates it; nothing may assume `dim > 0` before first embed.
5. Factory selection is startup-fail-fast: config parsing (02) validates
   `[providers.embedding].backend` against `known_embedders()` at load; the `fastembed` factory
   additionally probes the import so a deploy without the extra refuses to start rather than
   crashing on the first message. Model download/load (~130 MB BGE-small) still happens lazily
   on first embed.
6. `"off"` backend returns `None`/`Option::None`: the wiring layer passes `Option<Embedder>`
   everywhere; consumers (05, 07) treat "no embedder" and "seam off" identically. Call sites are
   never conditional on backend names.

### Consolidation — window gathering

7. `gather_window` semantics: facts = ALL current (non-superseded, non-expired) facts (fetched
   with limit 100_000, newest-first), keep newest `facts_max`, then **reverse to oldest-first**
   for stable prompt ordering; `facts_truncated = max(0, total - facts_max)`. Turns = every turn
   with `prior_watermark.last_turn_id < id ≤ max_turn_id` (0 lower bound when no watermark);
   keep the newest `turns_max` (tail); `turns_max ≤ 0` → empty turns. `max_fact_id` /
   `max_turn_id` are the TRUE uncapped high-water marks (`latest_fact_id` counts superseded rows
   too; `latest_fts_id` = max turn id) — the watermark advances to these, not to the capped
   window edge, so truncation never re-shows old data next night.
8. The fact window is intentionally NOT id-bounded by the watermark — consolidation reasons over
   the whole live fact base every night; only the turns axis is windowed.

### Consolidation — prompt / parse

9. `build_prompt` produces exactly two messages: `system` = caller-supplied text **verbatim**
   (may be empty string — never substituted in code), and one `user` message:
   `Self subject key (the character): {self_key}` + blank line + `Facts (current):` +
   one line per fact `- id={id} subjects=[{", ".join(subject keys) or "—"}]: {text}` and, when
   turns exist, blank line + `Recent conversation (for attribution context):` + one line per
   turn `- {author.label if author else role}: {content}`.
10. `plan_consolidation` makes ONE raw `llm.chat` call — no `request_structured` retry loop
    (unlike the opinion pass). `parse_actions` is maximally permissive: fenced/prose-wrapped
    JSON extracted via `coerce_json(expect="object")`; anything that isn't an object → `([], [])`;
    `retire`/`rewrite` values that aren't lists → `[]`; non-dict items dropped. Garbage never
    raises — it degrades to an empty plan.
11. `reply_parse_failed`: empty/whitespace reply → `false` (a silent model is a clean empty
    plan); a non-empty reply that fails to parse to a JSON object → `true`, which appends the
    exact note `"llm reply did not parse to a plan object — treated as empty"` to `plan.notes`
    and logs a WARNING. Distinguishing "model fumbled JSON" from "clean empty plan" is a pinned
    behavior.

### Consolidation — validation rails (order and strings are contracts)

12. `validate` is pure and deterministic. Processing order: all retire proposals in input order,
    THEN all rewrite proposals in input order. Shared mutable state across the loop: `claimed`
    id-set and `mutated` counter — so an earlier action reserves ids and cap budget from later
    ones.
13. Retire rails in check order: (a) `coerce_positive_int_list(payload["fact_ids"])` — bools
    rejected, numeric strings accepted, non-positive and duplicate values dropped, non-list →
    `[]`; empty → rail `no_ids`, detail `"no fact_ids"`; (b) `_check_ids`: any id not in window →
    `unknown_id`; any referenced fact having a subject whose `canonical_key == self_key` →
    `self_subject` (consolidation NEVER touches opinions — the pass boundary invariant); any id
    already claimed → `duplicate_target`; (c) `mutated + len(ids) > cap` → `cap`, detail
    `"cap={cap} reached"`. Detail for id rails is `str(ids)` (Python tuple repr).
14. Rewrite rails in check order: `no_ids` (on `old_fact_ids`) → `_check_ids` (same three) →
    `empty_text` (blank `new_text` after strip; detail `"blank new_text"`) → subjects:
    `subject_keys` via `coerce_str_list`; `subject_lost` when the union of source facts' subject
    keys is non-empty but proposal keys are empty (detail = `str(sorted(source_keys))`);
    `subject_introduced` when any proposed key is neither in the source-key union nor equal to
    `self_key` (detail = `str(introduced_list)`) — the ego-key exception lets a bit about a
    person be re-attributed to the familiar's own narrative, never minted under a new person →
    `noop` only for single-source rewrites where `_normalize_fact_text(old) ==
    _normalize_fact_text(new)` AND source subject-key frozenset == proposed key frozenset
    (detail `"restatement"`) → `cap` last. Accepted actions record
    `reason = str(payload.get("reason","")).strip()`.
15. `_normalize_fact_text` (imported from 03; the noop rail and opinion dedup both depend on it):
    lowercase → split/join whitespace collapse → remove ALL `'` and `"` chars → strip the
    surrounding character set `.,!?;:()[]{} \t\n`. Internal non-quote punctuation is kept.
16. The plan always carries `new_last_fact_id = window.max_fact_id`,
    `new_last_turn_id = window.max_turn_id` and the considered/truncated counts, even when
    everything was rejected.

### Consolidation — apply

17. `apply_consolidation` is the ONLY mutator; planning is dry-run-safe by construction
    (`apply=False` orchestration writes nothing, not even the watermark — pinned by tests).
18. Apply order: snapshot all rewrite source facts up front via `facts_by_ids` (exact-id fetch,
    INCLUDES superseded rows — needed for display resolution even if a row races); execute all
    retires, then all rewrites, then advance the watermark once.
19. Retire → `store.supersede(obsolete_facts=ids, new_fact=None)`: per-id skip-and-record —
    already-retired ids land in `result.skipped (id, reason)` and are recorded on the report as
    `("retire", id, reason)`, never raised. Rewrite → `store.supersede(obsolete_facts,
    new_fact=FactDraft{channel_id = channel of FIRST old fact, text, subjects})`: the store owns
    the merge atomically — it pre-flights every source and declines the WHOLE merge (mints
    nothing, `minted=None`) if any source is no longer current; apply records all skips for that
    action and continues. This makes apply idempotent and race-tolerant: re-running a
    partially-applied plan skips what already happened; a concurrent writer superseding a
    planned fact between plan and apply can't crash the pass or strand a half-merge. The minted
    fact's `source_turn_ids` = union of the obsolete rows' provenance — computed by the store,
    never supplied by this subsystem (`FactDraft` deliberately carries no turn ids).
20. `_subjects_for_rewrite` display resolution, per proposed key: first-seen
    `display_at_write` from source-fact subjects (iteration: old_fact_ids order, then subject
    order within each fact; `setdefault` = first wins); else if the key is an ego key
    (`ego:<nonempty>`), `familiar_display_name` or the key's part after the first `:`; else the
    key's part after the first `:`.
21. Watermark: apply advances **only** `last_fact_id` (to `plan.new_last_fact_id`). The turn
    axis belongs to the opinion pass; `advance_sleep_watermark` is a COALESCE partial update so
    neither pass clobbers the other, even within one run (pinned: after consolidation-only apply,
    `(last_fact_id, last_turn_id) == (N, 0)`). The watermark advances even when every action was
    skipped/rejected — coverage, not success, is what it records. `ApplyReport.watermark` echoes
    `(plan.new_last_fact_id, plan.new_last_turn_id)` — the turn element is informational only,
    not written.

### Opinion formation

22. `bucket_by_day`: sort turns by id ascending; bucket key = the turn's UTC-aware timestamp
    converted to `tz_name` (IANA, via ZoneInfo → `chrono-tz`), formatted `%Y-%m-%d`; day batches
    sorted by date string ascending; within a day turns stay id-ordered. Empty input → empty vec.
    Tests pin cross-midnight splitting (02:00Z/10:00Z in America/Los_Angeles → two days).
23. `gather_days` windows the TURN axis only: `prior.last_turn_id < id ≤ latest_fts_id`; no cap
    (see the dead `DEFAULT_TURNS_MAX_PER_DAY` note). First run = whole-backlog catch-up; a
    missed night just widens the window.
24. Turn rendering (`_render_turn`) is an anti-impersonation contract:
    `role == "assistant"` → `[{id}] {self_name} (you): {content}`; a non-assistant turn whose
    author label EQUALS `self_name` → label becomes `{label} (a user, not you)`; otherwise
    `[{id}] {label}: {content}` with `label = author.label` or the role when authorless.
25. Stage 1 (per-day) runs through `request_structured` (08) with `_STANCE_SCHEMA` — i.e. one
    corrective re-ask on shape failure (DEFAULT_MAX_RETRIES=1), degrade to `[]` after that.
    Code-enforced grounding: keep only cited turn ids that belong to the day
    (`⊆ day.turn_ids`, order-preserving after `coerce_positive_int_list`); a stance-moment with
    text blank or zero surviving ids is dropped silently. Days are processed **sequentially** in
    date order (the docstring says "parallel-friendly" but the implementation is a plain loop —
    keep sequential for LLM budget/ordering parity unless deliberately redesigned).
26. Stage 2 (`synthesize`): empty stance-moment list short-circuits to `[]` WITHOUT an LLM call.
    Otherwise one `request_structured` call with `_SYNTHESIS_SCHEMA`; returns raw dict items from
    the `"opinions"` key (non-dict/non-list degrade to `[]`).
27. Prompt assembly: config-sourced `system` text is filled via `fill_placeholders` — literal
    `{self_name}` substitution only, single pass, unknown tokens and stray braces pass through
    verbatim, never raises (crash-safe config override is a pinned test). The deny-list block
    (stance stage) / prior-self-dossier block (synthesis stage) and the JSON reply-shape
    contract (`render_contract(schema)`) are appended IN CODE after the persona text, joined
    into the single system message: `f"{instruction}{deny}\n\n{contract}"`. Stance user message:
    `Day {date}:\n` + rendered turns. Synthesis user message: `Stance-moments:\n` + lines
    `- ({date}) ids={[ids]}: {text}`.
28. `validate_opinions` rails, per raw item in order: `empty_text` (blank text, detail
    `"blank text"`) → `ungrounded` when ids are empty OR any id falls outside the union of ALL
    stance-moments' ids (detail `f"ids={list} bad={bad_list}"`) — the synthesis can never invent
    grounding → `duplicate` (normalized-text dedup across the plan; first occurrence wins;
    detail = the text) → `cap` when `len(accepted) >= cap` (detail `f"cap={cap}"`; NOTE: counts
    accepted opinions, unlike consolidation's cap which counts mutated facts). Accepted opinions
    compute: `self_grounded` = any source id authored by the familiar (assistant-role turns in
    the window); NOT self-grounded → accepted but flagged `"no_self_authored: {text}"` in
    `plan.flags` (the room's stance vs. hers — flag, never reject); `valid_from_date` = MIN date
    over cited ids present in the window's turn→day map (honest timeline for backlog catch-up);
    `importance` via `_coerce_importance`: bool → 5, int → clamp to [1,10], digit-string
    (optional leading `-`) → parse+clamp, anything else/absent → 5. Importance is NEVER a
    rejection reason.
29. `apply_opinions`: each opinion becomes `store.append_fact` with `channel_id=None` (opinions
    are global stances, not channel-bound), subjects = exactly one
    `FactSubject{canonical_key: ego_canonical_key(fam) = "ego:{familiar_id}",
    display_at_write: familiar_display_name or familiar_id}`, `source_turn_ids` as cited,
    `valid_from = {valid_from_date}T00:00:00+00:00` (midnight UTC of the earliest grounding
    day, even though `created_at` is tonight), `importance` as coerced. Beware: the store's
    insert-time near-duplicate suppression can return an EXISTING fact — `recorded` then carries
    that existing id; this is fine and expected. Then advance **only** `last_turn_id` to
    `plan.new_last_turn_id` (pinned: a pre-existing `last_fact_id=77` survives).
30. The two passes' watermark-axis ownership is the core cross-pass invariant: consolidation
    writes `last_fact_id` only, opinions write `last_turn_id` only; `advance_sleep_watermark`
    with both `None` is a no-op; on first insert the omitted axis defaults to 0.

### Maintenance registry / orchestration

31. `execute_*` orchestrators: plan ALWAYS runs; each rail-blocked proposal is logged as an
    individual WARNING containing the pass name, the literal rail name, detail truncated to 80
    chars, and `str(payload)` truncated to 160 (the sleep audit JSON was removed; this log IS
    the audit trail — a test greps WARNING records for the rail name, e.g. `unknown_id`). Apply
    runs only when `apply=true`. Dry-run writes nothing, including watermarks.
32. `execute_opinion_formation` fetches the prior self-dossier
    (`store.get_people_dossier(familiar_id, canonical_key=ego key)`) and threads
    `prior.dossier_text` into the synthesis prompt ("refine/extend, do not blindly repeat").
33. Denylist data-flow (the one thing this registry has that the projector registry lacks):
    `ConsolidationPass.run` extends `run.denylist_fact_ids` with the plan's ACCEPTED retire +
    rewrite-source ids; `OpinionFormationPass.run` resolves those ids to fact TEXTS via
    `facts_by_ids` (works even though apply already superseded them — exact-id fetch includes
    superseded rows) and passes them as the stance prompt's known-bits deny block: header
    `KNOWN BITS / NOISE (already judged not-real — {self_name} may have a TAKE on these, ... but
    must never treat them as true events):` + `- {text}` lines. Pinned end-to-end: fact "noise"
    retired by consolidation arrives as `denylist == ("noise",)` in the opinion call.
34. `run_passes` is strictly sequential and unguarded — a pass exception aborts the remaining
    passes and propagates. The engine (11) wraps the whole run in its own catch-everything guard.
35. `OpinionFormationPass.run` sets `run.opinion_plan = plan` so the engine's dream-prose step
    (NOT this subsystem) can read the night's freshly minted opinions.

### Engine coupling (context for the port; owned by 11)

36. The engine spawns `_run_sleep_passes` as a detached background task (named
    `sleep-passes-{familiar_id}`) at sleep DEPARTURE, using the `"background"` LLM slot and
    `apply=True`; the task handle is cancelled on engine stop. Cancellation/failure mid-run is
    safe by design: watermarks only advance on apply, so an interrupted window is simply
    re-covered next sleep; per-action skip handling makes a re-run of overlapping work
    idempotent. A `sleep_passes_enabled=false` flag short-circuits the task (tests/minimal
    deployments). Pass failures degrade wake prose to seed-only; they never crash the engine or
    block the activity return.

## Data formats

### LLM reply shapes (parsed permissively; these are the shapes the prompts request)

Consolidation (single object, hand-parsed via `parse_actions`, no schema/retry):

```json
{"retire":  [{"fact_ids": [1, 2], "reason": "why"}],
 "rewrite": [{"old_fact_ids": [3, 4], "new_text": "…", "subject_keys": ["discord:A"], "reason": "why"}]}
```

Stance stage (via `request_structured`, root=object, container `candidates`):

```json
{"candidates": [{"text": "<stance>", "turn_ids": [<id>, ...]}]}
```

Schema declaration pins the contract text: fields `text` placeholder `"<stance>"`, `turn_ids`
placeholder `[<id>...]`; `empty_note = "Empty list when nothing stands out."`.

Synthesis stage (root=object, container `opinions`):

```json
{"opinions": [{"text": "<their stance>", "source_turn_ids": [<id>...], "importance": <1-10>, "reason": "<why>"}]}
```

The rendered contract line (from 08's `render_contract`) is
`Reply with JSON only, no prose or code fences: {skeleton}` — treat the renderer as 08's
contract; this subsystem only declares the two `Schema` values above.

### Store rows touched (schema owned by 03; listed for cross-checking)

- `sleep_watermark(familiar_id PK, last_fact_id INT, last_turn_id INT, updated_at TEXT ISO-8601)`
  — one row per familiar; COALESCE partial upsert per axis.
- `facts(... source_turn_ids TEXT JSON array, created_at, superseded_at TEXT|NULL,
  superseded_by INT|NULL, valid_from TEXT|NULL, valid_to TEXT|NULL, importance INT|NULL)` —
  consolidation retires/mints via `supersede`; opinions mint via `append_fact` with
  `valid_from = <date>T00:00:00+00:00` and the ego subject.
- `fact_embeddings(fact_id, model, dim, vector BLOB, created_at)` keyed `(fact_id, model)`;
  `vector` is packed little-endian float32; `model` = `Embedder.name` — a backend/model swap
  accumulates new rows beside old (audit stays queryable, rank filters on the live name). The
  worker/`LEFT JOIN` live in 03/07; this subsystem only supplies `name`/`dim`/vectors.
- Ego subject key format: `ego:{familiar_id}` (`ego_canonical_key` in 02);
  `is_ego_key` = platform `ego` + non-empty id.

### Prompt wire formats (exact templates)

- Consolidation user msg: see invariant 9.
- Stance system msg: `{filled instruction}{optional deny block}\n\n{contract}`; deny block =
  `\n\nKNOWN BITS / NOISE (already judged not-real — {self_name} may have a TAKE on these, e.g.
  finding a running joke tedious, but must never treat them as true events):\n- {t1}\n- {t2}…`
- Synthesis system msg: `{filled instruction}{optional prior block}\n\n{contract}`; prior block =
  `\n\n{self_name}'s existing self-understanding (refine/extend, do not blindly repeat):\n{dossier}`
- Turn line / stance-moment line formats: invariants 24 and 27.

## Config knobs

| Key | Default | Read by | Notes |
|---|---|---|---|
| `[providers.embedding].backend` | `"off"` | factory via `EmbeddingConfig` | must be in `known_embedders()`; validated at config parse (02 imports this subsystem's registry) |
| `[providers.embedding].dim` | `256` | `hash` backend only | positive int; real backends ignore it |
| `[providers.embedding].fastembed_model` | `"BAAI/bge-small-en-v1.5"` | `fastembed` backend | non-empty string |
| `[providers.embedding].fastembed_cache_dir` | `None` | `fastembed` backend | empty string coerces to `None`; unset → library default `~/.cache/fastembed` |
| `[prompt].sleep_consolidation_system` | `""` | consolidation system msg | prose ships only in `data/familiars/_default/character.toml`; `_default`→override deep-merge |
| `[prompt].sleep_stance_system` | `""` | stance stage (`{self_name}`) | same |
| `[prompt].sleep_synthesis_system` | `""` | synthesis stage (`{self_name}`) | same |

Unknown keys in `[providers.embedding]` are a config error (02). The `[sleep]` window/grace
schedule and `dream_extraction_clause` belong to the activities engine / extractor (11 / 07),
not here. In-code tunables (not TOML today): `facts_max=500`, `turns_max=400`, `retire_cap=50`,
`opinion_cap=60` — exposed as `MaintenanceContext` fields/orchestrator kwargs.

## Dependency edges

Imports FROM (this subsystem → others):

| Module | Subsystem | What is used |
|---|---|---|
| `history.store` / `history.async_store` | 03 | `Fact`, `FactDraft`, `FactSubject`, `HistoryTurn`, `SleepWatermark`, `SupersedeResult`, `_normalize_fact_text`, `_subject_key_set`; store methods `get_sleep_watermark`, `advance_sleep_watermark` (partial per-axis), `latest_fact_id`, `latest_fts_id`, `recent_facts`, `turns_in_id_range`, `facts_by_ids`, `supersede`, `append_fact`, `get_people_dossier` |
| `identity` | 02 | `ego_canonical_key`, `is_ego_key` |
| `config` | 02 | `EmbeddingConfig` (type only; 02 also imports our `known_embedders` at parse — a deliberate 02→04 edge) |
| `llm` | 08 | `LLMClient.chat`, `Message` (`content_str`) |
| `structured_output` | 08 | `coerce_json`, `coerce_positive_int_list`, `coerce_str_list` |
| `structured_request` | 08 | `Schema`, `Field`, `render_contract`, `request_structured` (retry-once) |
| `prompt_fill` | 08 | `fill_placeholders` (crash-safe `{key}` substitution) |
| `log_style` | 01 | log formatting only |

Imported BY (others → this subsystem):

| Importer | Subsystem | What it uses |
|---|---|---|
| `processors/fact_embedding_worker.py`, `processors/projectors.py` | 07 | `Embedder` protocol (batch embed; `name` as storage model key; tolerates `dim==0`; treats a `len(vectors)!=len(texts)` mismatch as a backend bug → skip batch, retry next tick) |
| `context/layers.py` (`RagContextLayer`) | 05 | `Embedder` for cue embedding at rerank |
| `activities/engine.py` | 11 | `MaintenanceContext`, `MaintenanceRun`, `SleepPromptText`, `create_passes`, `run_passes`, `DEFAULT_PASSES`; reads `OpinionPlan.opinions` for dream prose |
| `commands/run.py` (wiring) | 10 | `create_embedder(config.embedding)` once at startup, shares the instance across assemblers + projector context; `SleepPromptText.from_config` |
| `config.py` | 02 | `known_embedders()` (deferred import inside `_parse_embedding_config`) |

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `tests/test_embedding.py` | HashEmbedder: dim≥8 ValueError, `hash-v1` name, dim propagation, empty→empty, blank→zero-vec, length preservation, determinism, casefold invariance, cosine separation of disjoint vs overlapping token sets; factory: builtin names, off→None, hash threading, unknown→ValueError; config defaults | logic-portable (two factory tests monkeypatch `importlib.util.find_spec` → in Rust make the import probe an injectable closure/feature flag) |
| `tests/test_fastembed_embedder.py` | name carries model; known-dims table (384/768); dim 0 for unknown model; distinct names per model; empty input skips load; missing dep → RuntimeError w/ `local-embed`; single load across calls; dim probed from first vector; pre-known dim never clobbered; `cache_dir` kwarg passed only when set | needs-Rust-mock (stubs `sys.modules["fastembed"]`; port by trait-mocking the model loader/backend) |
| `tests/test_consolidation.py` | every retire/rewrite rail incl. rail names, cap-defers-excess, duplicate_target, self_subject (retire+rewrite+non-blocking for ordinary facts), subject_introduced + ego-key exception, noop, empty_text, subject_lost vs subjectless-OK; build_prompt verbatim system + `_default` config prose thread-through; gather_window current-facts/max-ids/superseded-exclusion/truncation; plan e2e with fake LLM incl. garbage-reply note, clean-empty no-note, system-reaches-LLM, rail-beats-prompt | logic-portable (uses in-memory store + scripted LLM; `_default/character.toml` fixture read from repo data dir) |
| `tests/test_sleep_apply.py` | retire marks superseded; rewrite mints with union provenance + `superseded_by` backlink; ego-subject rewrite display; concurrent-supersede → skip-not-raise, valid sibling action still applies, watermark still advances; watermark fact-axis-only `(3,0)`; full plan→apply e2e | logic-portable |
| `tests/test_opinion_formation.py` | `_render_turn` you/not-you/plain; day bucketing tz + ordering + self_turn_ids; gather_days watermark; stance extraction id-filtering + drop-no-valid-ids; stance/synthesis prompt: `{self_name}` fill, stray-brace degrade, importance rubric present from `_default` prose; validate rails (ungrounded superset + zero, duplicate, cap, empty via blanks), no_self_authored flag, earliest valid_from, importance clamp/default matrix (8→8, 99→10, 0→1, missing→5, "very"→5); plan e2e incl. configured prompts reach LLM + rail-beats-prompt; apply mints ego fact w/ provenance + valid_from midnight UTC + importance; turn-axis-only watermark preserving fact axis 77 | logic-portable |
| `tests/test_maintenance_passes.py` | registry order/unknown-raises/known set/DEFAULT_PASSES tuple; denylist threading (retired text reaches opinion call); default run applies both passes producing exactly one ego opinion | mostly logic-portable; the denylist spy monkeypatches a module fn — port as a recording mock of the orchestrator seam or assert via prompt content |
| `tests/test_sleep_pass_orchestrators.py` | dry-run never mutates (facts + watermark) for both passes; apply mutates + advances; rail rejection logged at WARNING containing rail name | logic-portable (log assertion → `tracing` capture) |
| `tests/test_fact_embedding_worker.py`, `tests/test_memory_projectors.py` | consumer-side contract on `Embedder` (07's remit; keep in view for the trait design) | see 07's spec |
| `tests/test_config.py` (embedding portions) | `[providers.embedding]` parse/validation incl. backend-must-be-registered | 02's remit |

## Rust port notes

- **Crate seams.** `Embedder` → `#[async_trait] trait Embedder { fn name(&self) -> &str; fn dim
  (&self) -> usize; async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> }`
  — but note the Python contract is "never raise for content reasons"; reserve `Err` for backend
  faults (matching FastEmbed load failure). `dim` is mutable state in FastEmbed (0 → probed):
  use interior mutability (`AtomicUsize`) or return `Option<usize>` — flag the API change to the
  reviewer. `MaintenancePass` → trait with `&mut MaintenanceRun`.
- **Registries.** Both registries (embedder factories, maintenance passes) are import-time
  global dicts with overwrite-on-reregister. In Rust prefer explicit construction: a
  `HashMap<&str, Factory>` built by a `builtins()` fn + insert for third parties. The dynamic
  extension point matters (config validation lists registered names) — keep `known_*()` exact,
  including the sorted-comma error strings tests match on (`match="nope"`, `"unknown embedding
  backend"`).
- **Float parity.** HashEmbedder must reproduce vectors exactly if `hash-v1` rows are to be
  reused: `blake2b` with 4-byte digest (`blake2` crate supports variable output), big-endian u16
  index, f32 vs f64 accumulation — Python accumulates in f64 and stores f64; the storage format
  is f32 LE. Normalize in f64 then cast, or accept last-ulp drift and bump to `hash-v2`
  (cleanest: the model-keyed side-index makes re-backfill free and automatic).
- **`asyncio.to_thread` / load lock** → `tokio::sync::OnceCell` or `Mutex<Option<Model>>` +
  `spawn_blocking`. `fastembed` has a native Rust crate (`fastembed` on crates.io, ONNX via ort)
  — use it instead of shelling to Python; keep `name` format `fastembed:{model}` and the
  known-dims table so stored rows stay compatible.
- **Timezone bucketing**: `chrono` + `chrono-tz`; the bucket key is a LOCAL calendar date string
  from a UTC timestamp — DST-transition days must match Python `astimezone` semantics (chrono-tz
  does). Invalid tz name currently raises out of `gather_days` (engine guard catches it); keep
  that behavior rather than defaulting.
- **Permissive JSON coercers** (`coerce_json` fence-stripping/balanced-blob extraction,
  `coerce_positive_int_list` bool-rejection quirk — port faithfully; `true` must not become 1)
  live in 08; this subsystem only consumes them, but its garbage-degradation tests will
  transitively pin them.
- **Concurrency semantics to preserve**: the whole sleep run is ONE background task; passes are
  sequential; per-day stance calls are sequential; there is no internal timeout, retry (beyond
  `request_structured`'s single shape re-ask), or cancellation handling in this subsystem —
  cancellation safety comes entirely from "watermark advances only on apply" + store-level
  skip-and-record. Do not add parallelism to stance extraction without checking LLM-slot
  ordering assumptions in tests (FakeLLMClient pops scripted replies in call order).
- **Ordering is behavior**: retire-before-rewrite validation (claimed-set + shared cap budget),
  input-order processing, first-wins dedup, first-fact channel for merges, first-seen display
  per subject key. Use order-preserving structures (`Vec` + `HashSet` guard, or `IndexMap`),
  not `HashMap` iteration.
- **Redesign candidates** (flag, don't silently do): dead `DEFAULT_TURNS_MAX_PER_DAY`; the
  consolidation pass calling raw `llm.chat` instead of `request_structured` (no shape retry —
  upgrading it would change call counts that tests script); `ApplyReport.watermark` carrying an
  unwritten turn axis; `RejectedAction.payload: dict[str, Any]` → `serde_json::Value`.
- **Suggested crates**: `tokio`, `async-trait`, `serde`/`serde_json`, `blake2`, `chrono` +
  `chrono-tz`, `regex` (Unicode `\w`), `fastembed` (optional feature `local-embed`), `tracing`
  (WARNING-level rejection audit is a tested contract), `thiserror` for the factory errors.
