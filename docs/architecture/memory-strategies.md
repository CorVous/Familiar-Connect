# Memory strategies

The rule: **`turns` is source of truth; everything else is a
projection that can be dropped and rebuilt.** This page maps the
strategies that fit inside that rule, says which one ships today,
and names the seams alternatives plug into.

Forward-looking work lives in [Roadmap](roadmap.md). Implementation
details for what already ships live in
[Context pipeline](context-pipeline.md).

## The four families

1. **Tiered virtual memory with self-edited blocks** (Letta /
   MemGPT). LLM mutates its own context via tools. **Rejected** —
   destructive; see
   [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).

2. **Bi-temporal append-only knowledge graphs** (Zep / Graphiti,
   Cognee). `t_valid` / `t_invalid` per edge; new facts mark old
   ones invalid; nothing deleted. **Adopted incrementally** —
   bi-temporal facts shipped (M1; see [Bi-temporal facts](#bi-temporal-facts)),
   M5 plans Graphiti as a swappable projector.

3. **Self-evolving atomic notes with emergent linking** (A-MEM,
   generative-agents memory stream). One observation per row,
   LLM-generated metadata, links decided at write-time.
   **Familiar-Connect's `facts` and `people_dossiers` are this
   family today.** M2 (importance) and M3 (citation-bearing
   reflections) close the gaps.

4. **Hand-authored, rule-activated character context**
   (RisuAI / SillyTavern lorebooks). Keyword-activated entries.
   `core_instructions.md` + `character.md` are a degenerate case
   today (always-on, no keys); M4 adds a real lorebook layer.

## What's wired today

| Side-index | Writer | Read by |
|---|---|---|
| `fts_turns` (FTS5) | SQLite triggers on `turns` | `RagContextLayer` |
| `fts_facts` (FTS5) | SQLite triggers on `facts` | `RagContextLayer` |
| `summaries` | `SummaryWorker` | `ConversationSummaryLayer` |
| `cross_context_summaries` | `SummaryWorker` | `CrossChannelContextLayer` |
| `facts` | `FactExtractor` | `RagContextLayer` (via FTS) |
| `people_dossiers` | `PeopleDossierWorker` | `PeopleDossierLayer` |

Every writer is watermark-driven and idempotent; deleting a
side-index table rebuilds it from `turns`. See
[Context pipeline](context-pipeline.md) for watermark semantics.

Reads are multi-signal: BM25 + recent-window exclusion today; M2
adds importance, M6 adds embeddings.

## Swap points

### 1. Layer order and selection (`Layer` Protocol)

Per-channel `prompt_layers` in `character.toml` swaps order or
disables layers per Discord channel. New layer = one class
implementing `build(ctx)` + `invalidation_key(ctx)`, registered in
`commands/run.py::_default_assembler`.

This is how M3 (`ReflectionLayer`) and M4 (`LorebookLayer`) plug
in. It's also how a candidate replacement layer can run
side-by-side: register both, switch via TOML on a test channel.

### 2. Projection writer (planned: `MemoryProjector` Protocol)

Today's writers collectively implement one strategy: rich-note +
per-channel summary + per-person dossier. M5 lifts them behind a
`MemoryProjector` Protocol. Default stays. A `GraphitiProjector`
or `CogneeProjector` runs alongside or replaces. Selected via
`[providers.memory].projectors`.

### 3. Retrieval ranking (`RagContextLayer`)

Today's ranking combines BM25 score and recency. M2 widens to
four signals (BM25 × recency × importance × embedding once M6
ships), TOML-driven:

```toml
[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.4
importance_weight = 0.6
embedding_weight  = 0.0   # 0 disables until M6
```

See [Tuning](tuning.md#forward-looking-schema).

## Bi-temporal facts

Each row in `facts` carries two independent time axes:

- **System-time** (`created_at`, `superseded_at`, `superseded_by`) —
  when *we* recorded it, and when we retired it. Supersession keeps
  the old row; the new row points back via `superseded_by`.
- **World-time** (`valid_from`, `valid_to`) — when the fact was
  observed to *apply in the world*. `valid_from` defaults to the
  source turn's timestamp; the LLM may override with an explicit
  ISO-8601 string when it spots an "as of …" phrase ("Aria moved to
  Berlin in early 2024"). `valid_to` is `NULL` while the fact still
  applies.

Default reads stay "current truth": `superseded_at IS NULL` and
`valid_to IS NULL OR valid_to > now`. Audit queries pass
`as_of=<datetime>` to `recent_facts` / `search_facts` /
`facts_for_subject`; that switches to a bi-temporal slice
(`valid_from <= as_of < valid_to`) and includes superseded rows so
prior beliefs remain reconstructable.

Legacy rows (pre-M1) carry `valid_from = valid_to = NULL` and read
as "always valid"; no backfill — the feature is forward-only.

## Why rich-note + bi-temporal, not graph-only

Live disagreement in the field: knowledge graph (Zep, Cognee,
GraphRAG) vs richly-attributed flat notes with emergent links
(A-MEM, generative agents). Graphs win on multi-hop relational
queries, lose on token cost. Rich-note systems are simpler, cheaper,
and degrade on multi-hop.

For a Discord familiar, multi-hop queries are rare ("what's
Alice's drink?", "what were we talking about yesterday?"
dominate). Rich-note + bi-temporal handles them at lower cost.
Graphiti-style graph stays on the roadmap (M5) as an *additional*
projector, not a replacement.

## Why authored canon stays separate from experiential memory

- **Authored canon** (`character.md`, future `lorebook.toml`) —
  human-edited, never evolves on its own.
- **Experiential memory** (`turns` and projections) — bot-generated,
  evolves every interaction.

Mixing them lets the agent rewrite its own character description.
Keeping them split makes the trust boundary inspectable.

## Operator playbook

### Rebuild a side-index

```bash
sqlite3 data/familiars/<id>/history.db "DELETE FROM facts;"
# next FactExtractor tick rebuilds from turns
```

### Force a dossier re-fold

```sql
UPDATE people_dossiers
SET dossier_text = '...', last_fact_id = 0
WHERE canonical_key = 'discord:1234';
```

`last_fact_id = 0` forces the worker to refold every prior fact
on its next tick.

### A/B a layer on one channel

`[channels.<id>].prompt_layers` overrides default layer order on
one Discord channel. Compare a test channel (with the candidate
layer) against a control. Once A1 lands, the same mechanism
extends to STT, turn detection, and voice pipeline mode.
