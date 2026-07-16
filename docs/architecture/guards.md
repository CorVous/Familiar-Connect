# Pipeline guards

Familiar-Connect's pipelines are studded with small **guards**: local checks that
drop, skip, suppress, or scrub a unit of work before it reaches the next stage —
a near-duplicate fact that should not be re-inserted, a hallucinated citation
that should not be persisted, a leaked `<invoke>` block that should never be
spoken aloud. Issue [#132](https://github.com/CorVous/Familiar-Connect/issues/132)
observed that these guards had grown "jagged": every one was expressed ad-hoc,
with no shared vocabulary and inconsistent observability (the reply-gate guards
logged a structured skip line; the memory-path guards dropped silently). This
page is the written decision that issue asked for.

## Decision

**Adopt (1) a shared taxonomy and (2) a shared audit-log convention. Explicitly
decline (3) a guard interface / registry.**

- **We adopt a taxonomy** — a common vocabulary that names each guard, the
  pipeline step it sits at, its shape, and the failure mode it prevents — so the
  question "what guards apply at step X?" has one answer, enumerated below.
- **We adopt an audit-log convention** — one structured `tracing::debug!` line,
  built from the existing `log_style` primitives, emitted wherever a guard drops
  work. This makes every guard's activity observable and greppable without
  changing what the guard does.
- **We decline a guard interface or registry.** A `trait Guard` with a central
  registry that every drop point implements was considered and rejected.

### Why not a guard interface / registry

The guards are genuinely heterogeneous by **altitude and shape**, and
forcing them under one interface would be over-engineering that buys nothing the
non-goals of #132 allow us to spend:

- They operate on **different data at different stages**: a streamed-token
  sentinel (reply gate), a normalize-then-compare over a SQLite row (DB-insert
  path), a post-parse id-set intersection (extraction filter), a bare
  empty-string check (reply gate). There is no honest common input or output
  type; a `Guard<In, Out>` would degenerate into `Guard<Anything, Anything>`.
- Their **control flow differs**: some `continue` a loop, some `return` an
  enum variant, some rewrite a string in place, some latch a three-state
  decision across delta boundaries. A uniform "returns pass/reject" signature
  would misrepresent the ones that transform rather than veto.
- #132 is explicitly **behaviour-preserving** (non-goal: "no behavior change").
  A registry earns its keep when guards must be enumerated, reordered, or
  toggled at runtime — none of which this codebase does or wants. Adding the
  indirection now would be cost without a consumer.

A taxonomy plus a logging convention gives us the thing we actually lacked —
discoverability and consistent observability — at documentation-plus-one-log-line
cost, and leaves each guard expressed in the idiom that fits it.

## Taxonomy

Each guard is classified on three axes.

### Axis 1 — pipeline step / altitude

Where the guard sits determines what it can see and what a drop costs.

| Step | `step=` value | What flows through | Cost of a drop |
| --- | --- | --- | --- |
| Pre-LLM turn filter | `turn_filter` | raw turns before a background LLM batch | a turn is excluded from extraction |
| Post-parse extraction filter | `extraction_filter` | parsed items from an LLM reply | an extracted item is discarded |
| DB-insert path | `db_insert` | a row about to be written | the write is skipped, existing row kept |
| Post-stream reply gate | `reply_gate` | the assistant's streamed/finished reply | the reply is suppressed or scrubbed |

### Axis 2 — guard shape

- **Sentinel** — inspects a streaming prefix and latches a decision
  (`SilentDetector` / `StreamGate`).
- **Normalize-and-compare** — canonicalises then looks for an existing match
  (`append_fact` dedup, `fact_supersede` priors).
- **Allowlist / id-set filter** — keeps only members of a known-valid set
  (participants manifest, reflection citations).
- **Predicate veto** — a boolean test that drops on match (self-capability,
  activity-return, empty-reply).
- **Rewrite** — strips a matched span rather than vetoing the whole unit
  (leaked-metadata scrub).

### Axis 3 — failure mode prevented

The concrete bad outcome each guard exists to stop — e.g. compounding duplicate
facts, persisting hallucinated provenance, or speaking tool-call XML aloud. Named
per-guard in the enumeration below.

## The guards

Locations verified against the current tree. Paths are under
`familiar-connect/src/`.

1. **Near-duplicate fact suppression** — `db_insert`, normalize-and-compare.
   `history/store.rs` `append_fact` gates on `find_current_dup` (normalising via
   `normalize_fact_text`) and returns `FactInsert::Existing` instead of minting a
   row. *Prevents:* the fact store compounding paraphrase/near-duplicate rows.
   **Expressed in the audit convention** (`guard=append_fact_dedup`).
2. **Self-capability drop** — `extraction_filter`, predicate veto.
   `processors/fact_extractor.rs` `is_self_capability` (definition), applied in
   `tick_inner` (the `if is_self_capability(...) { continue; }` site). *Prevents:*
   filing the model's own "as an AI I can't…" boilerplate as a fact.
   **Expressed in the audit convention** (`guard=self_capability`).
3. **Activity-return turn skip** — `turn_filter`, predicate veto.
   `processors/fact_extractor.rs` `tick_inner`, the
   `.filter(|t| t.mode != Some(ACTIVITY_RETURN_MODE))` on the batch. *Prevents:*
   re-extracting facts from self-generated activity fiction the activity engine
   already recorded.
4. **Participants-manifest subject gating** — `extraction_filter`, allowlist.
   `processors/fact_extractor.rs` `resolve_subjects` soft-validates each
   LLM-supplied subject key against the `build_participants` allowlist before a
   fact's subjects are accepted. *Prevents:* facts being filed under invented or
   out-of-scope subject keys.
5. **Reflection citation validation** — `extraction_filter`, id-set filter.
   `processors/reflection_worker.rs` `build` intersects the LLM's
   `cited_turn_ids` / `cited_fact_ids` against `valid_turn_ids` /
   `valid_fact_ids` (widened to all-known facts only when the recent set is
   empty). *Prevents:* persisting hallucinated provenance ids on a reflection.
6. **Supersede priors dedup** — `extraction_filter`, normalize-and-compare.
   `processors/fact_supersede_worker.rs` `evaluate` excludes `f_new` itself and
   de-dupes candidate priors across subjects (`seen_priors`) before asking the
   LLM which to retire. *Prevents:* evaluating the same prior twice and
   redundant supersede work.
7. **Silent / stream reply gate** — `reply_gate`, sentinel. `silence.rs`
   `SilentDetector` recognises the leading `<silent>` sentinel for the text path
   (`processors/text_responder.rs`); `StreamGate` widens it for voice
   (`processors/voice_responder.rs`), additionally latching `Suppress` on a
   leaked tool-call prefix (`classify_leading_leak`). *Prevents:* emitting a
   deliberate silence as prose, and (voice) speaking a leaked `<invoke>` /
   `silent(` / `<tool_call>` block aloud.
8. **Leaked-metadata prefix scrub** — `reply_gate`, rewrite.
   `processors/text_responder.rs` `strip_leaked_metadata_prefix`
   (`LEAKED_META_PREFIX_RE`) strips a leaked `[HH:MMxM]` / `[↩ …]` transcript
   prefix the model sometimes echoes. *Prevents:* the bot parroting its own
   context-framing metadata back into the channel.
9. **Empty-reply guard** — `reply_gate`, predicate veto.
   `processors/text_responder.rs` and `processors/voice_responder.rs` both drop a
   turn whose finished reply is whitespace-only (`reply.trim().is_empty()`).
   *Prevents:* sending an empty message / speaking silence.
10. **Wake shift-or-silent gate** — `reply_gate`, predicate veto.
   `processors/text_responder.rs` suppresses a wake turn that produced prose but
   did not `shift_focus` this turn (`is_wake && shifted_to.is_none()`), logging
   `guard=wake_shift_or_silent action=suppress` (#170). *Prevents:* a nudge-driven
   wake turn misrouting its reply to the stale focus channel instead of going
   silent.

## The audit-log convention

Wherever a guard drops work, emit **one** structured line at `debug` level, built
from `log_style` (`ls`) primitives so it shares the console styling and stays
parseable by the `diagnose` CLI. The field vocabulary:

```
[<Tag>] guard=<name> step=<step> action=<action> reason=<reason> [<context…>]
```

- `[<Tag>]` — the surrounding module's `ls::tag(...)` label (e.g. `[Facts]`).
- `guard=<name>` — stable snake_case guard identifier from the enumeration above.
- `step=<step>` — one of `turn_filter` / `extraction_filter` / `db_insert` /
  `reply_gate` (Axis 1).
- `action=<action>` — `skip` / `drop` / `suppress` / `scrub`.
- `reason=<reason>` — snake_case cause (e.g. `near_duplicate`,
  `self_capability_claim`).
- optional trailing `key=value` context (an id, a truncated text sample).

**Level is `debug`, always.** Several of these guards sit on hot paths (the
fact-insert path fires on every extracted fact); an `info` line there would spam
the console. The convention deliberately reuses the same `ls::kv_styled` shape
the reply-gate guards already emit for `skip=empty_reply`, so no new formatter or
parser is needed.

### Adoption

The convention is demonstrated on exactly two guards today — the two #132 named
as its worst silent offenders:

- **Near-duplicate fact suppression** (guard 1) — `append_fact` now emits
  `guard=append_fact_dedup step=db_insert action=skip reason=near_duplicate`
  after the DB actor returns (so the line lands on the caller's thread and is
  test-observable), where it previously dropped silently.
- **Self-capability drop** (guard 2) — the existing debug line in
  `fact_extractor` was rewritten from the ad-hoc `drop=self_capability` shape
  into the convention:
  `guard=self_capability step=extraction_filter action=drop reason=self_capability_claim`.

Both are covered by `LogCapture` assertion tests
(`tests/history_facts.rs::dedup_skip_emits_guard_audit_line`,
`tests/workers_fact_extractor.rs::self_capability_drop_emits_guard_audit_line`).
The remaining guards can be migrated to the convention incrementally as their
modules are next touched; nothing forces a big-bang change, and — per the
decision above — no guard changes what it filters.
