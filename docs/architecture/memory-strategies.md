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
   `core_instructions.md` + `character.md` ship as the always-on
   degenerate case; M4 added a real keyword-activated lorebook
   alongside them — see [Lorebook](#lorebook-m4).

## What's wired today

| Side-index | Writer | Read by |
|---|---|---|
| `fts_turns` (FTS5) | SQLite triggers on `turns` | `RagContextLayer` |
| `fts_facts` (FTS5) | SQLite triggers on `facts` | `RagContextLayer` |
| `summaries` | `SummaryWorker` | `ConversationSummaryLayer` |
| `cross_context_summaries` | `SummaryWorker` | `CrossChannelContextLayer` |
| `facts` | `FactExtractor` | `RagContextLayer` (via FTS) |
| `people_dossiers` | `PeopleDossierWorker` | `PeopleDossierLayer` |
| `reflections` | `ReflectionWorker` | `ReflectionLayer` |

Every writer is watermark-driven and idempotent; deleting a
side-index table rebuilds it from `turns`. See
[Context pipeline](context-pipeline.md) for watermark semantics.

Reads are multi-signal: BM25 + recent-window exclusion + a 1-10
importance hint per fact (M2). M6 adds embeddings.

## Swap points

### 1. Layer order and selection (`Layer` Protocol)

Per-channel `prompt_layers` in `character.toml` swaps order or
disables layers per Discord channel. New layer = one class
implementing `build(ctx)` + `invalidation_key(ctx)`, registered in
`commands/run.py::_default_assembler`.

This is how M3 (`ReflectionLayer`) and M4 (`LorebookLayer`) plug
in. It's also how a candidate replacement layer can run
side-by-side: register both, switch via TOML on a test channel.

### 2. Projection writer (`MemoryProjector` Protocol)

The watermark-driven writers — `SummaryWorker`, `FactExtractor`,
`PeopleDossierWorker`, `ReflectionWorker` — implement a thin
:class:`MemoryProjector` Protocol (M5):

```python
class MemoryProjector(Protocol):
    name: str
    async def run(self) -> None: ...
```

Names registered today (built-ins):

| Name | Class | Side-index |
|---|---|---|
| `rolling_summary` | `SummaryWorker` | `summaries`, `cross_context_summaries` |
| `rich_note` | `FactExtractor` | `facts` |
| `people_dossier` | `PeopleDossierWorker` | `people_dossiers` |
| `reflection` | `ReflectionWorker` | `reflections` |

Operators pick the active set in `character.toml`:

```toml
[providers.memory]
projectors = ["rolling_summary", "rich_note", "people_dossier", "reflection"]
```

Default lists all four. Drop a name to disable that writer. Add a
third-party projector (Graphiti, Cognee, …) by calling
`register_projector("graphiti", factory)` at import time and listing
it here. Unknown names fail loudly at config load — a typo never
silently drops a writer. Empty list disables all projection (the
side-indices stop refreshing; reads still work against whatever
exists).

### 3. Retrieval ranking (`RagContextLayer`)

Three signals fuse at rank time, TOML-driven:

```toml
[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.0
importance_weight = 0.6   # M2 — fact's 1-10 hint
embedding_weight  = 0.0   # M6 — disabled until embeddings ship
```

The layer over-fetches BM25 candidates (up to 4× `max_rag_facts`),
normalises each signal to `[0, 1]` within the candidate batch, and
keeps the top N by weighted sum:

- **BM25 quality** — best score in the batch maps to 1.0.
- **Recency** — newest fact id in the batch = 1.0.
- **Importance** — `importance/10`. Legacy / unscored rows
  (`importance IS NULL`) get the neutral midpoint, never zero.

`importance_weight = 0` reproduces pre-M2 ordering. See
[Tuning — retrieval ranking](tuning.md#retrieval-ranking-m2).

## Lorebook (M4)

Hand-authored, keyword-activated canon. Lives at
`data/familiars/<id>/lorebook.toml`:

```toml
[[entries]]
keys      = ["paris", "france"]
content   = "Paris is the capital of France. Founded ~250 BCE…"
priority  = 100
selective = false       # any key matches; default

[[entries]]
keys      = ["dragon", "wyvern"]
content   = "Dragons in this world breathe radiant light, not fire."
priority  = 50
```

`LorebookLayer` reads the file on every assemble (cheap; cached by
the `Assembler`'s invalidation key), scans the active channel's
last `recent_window` turns case-insensitively against each entry's
keys, and renders the matching subset newest-priority-first under a
`## Lorebook` block. `selective = true` flips the per-entry match
from any-key (OR) to all-keys (AND) — useful when a generic key
("dragon") shouldn't fire on its own without a disambiguator
("dragon" + "Cassidy").

No worker; the file is the sole source of truth. Operators edit it
in place, no migration. The cache key combines a content hash of
the file with the matched entry indices, so the layer only flips
when the file or the activation set actually changes.

The relevant knobs live in `[budget.<tier>]`:

```toml
[budget.text]
lorebook_tokens      = 800   # cap on the rendered block
max_lorebook_entries = 10    # hard cap on rendered entries
```

Authored canon stays separate from experiential memory (see
[below](#why-authored-canon-stays-separate-from-experiential-memory))
— the lorebook never mutates on its own.

## Reflections (M3)

Higher-order syntheses over recent turns + facts. Each reflection
row carries:

- `text` — one or two sentences naming a pattern, recurring tension,
  open question, or theme.
- `cited_turn_ids` / `cited_fact_ids` — forever-provenance JSON
  arrays. Citations render as breadcrumbs `[T#42, F#7]` so the
  reading model can map a synthesis back to its source.
- `last_turn_id` / `last_fact_id` — watermark snapshotting the
  worker's view at write time. The next tick uses the newest row's
  watermark as its lower bound; no separate watermark table.

`ReflectionWorker` ticks every 60 s; fires when at least
`turns_threshold` (default 20) new turns have arrived since the
newest reflection. It builds a prompt over the new turns plus the
20 most recent facts, asks for at most 3 reflections per tick, and
persists each answer that cites at least one valid turn or fact
id. Uncited answers are dropped — a free-floating opinion isn't a
synthesis.

`ReflectionLayer` reads the most recent rows scoped to the active
channel (channel-agnostic rows surface in every channel), renders
citation breadcrumbs, and flags `(stale)` on rows that cite at
least one superseded fact. Stale rows are never deleted; the audit
trail is the point.

The relevant knobs live in `[budget.<tier>]`:

```toml
[budget.text]
reflection_tokens     = 800   # cap on the rendered block
max_reflections       = 5     # hard cap on rendered rows
```

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

- **Authored canon** (`character.md`, `lorebook.toml`) —
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
