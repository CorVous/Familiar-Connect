# Memory strategies

Familiar-Connect's memory subsystem is built around one rule: **the
event log is the single source of truth, everything else is a
projection that can be dropped and rebuilt.** Within that rule, the
field has converged on a small set of strategies that each project the
log differently. This page maps the strategies, says which one
Familiar-Connect uses today, and describes the seams where alternative
strategies plug in.

For the *forward-looking* tasks, see [Roadmap](roadmap.md). For the
implementation details of the strategies that already ship, see
[Context pipeline](context-pipeline.md).

## Source of truth

The `turns` table in `data/familiars/<id>/history.db` is append-only.
Every voice utterance (after STT), every Discord message, every
assistant reply is appended with author, channel, content, and the
wall-clock timestamp at which the bot saw it. **Nothing is ever
deleted from `turns`.** Side-indices (FTS5, summaries, facts, people
dossiers) are regenerated from `turns` whenever they go stale or get
nuked for debugging.

This places Familiar-Connect in the same family as Park et al.'s
generative-agents memory stream, MemoriesDB's append-only architecture,
and Graphiti's episode subgraph. It is **explicitly not** in the
Letta / MemGPT family of self-edited destructive memory blocks.

## The four families

Across rigorous open-source memory work, four distinct families
emerge. Familiar-Connect's current implementation draws from (3) and
(4) and has roadmap items to incorporate (2).

1. **Tiered "OS-style" virtual memory with self-edited blocks**
   (Letta / MemGPT). The agent calls tools like `core_memory_replace`
   to mutate its own context. **Rejected for Familiar-Connect** — see
   [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).

2. **Bi-temporal append-only knowledge graphs that supersede rather
   than overwrite** (Zep / Graphiti, Cognee). Every edge carries
   `t_valid` / `t_invalid` timestamps. New facts mark old facts
   invalid; nothing is deleted. **Roadmap item M1 (bi-temporal facts)
   adopts this directly**; the longer-term M5 makes Graphiti-style
   graph projection a swappable backend.

3. **Self-evolving atomic notes with emergent linking** (A-MEM,
   generative-agents memory stream). Each observation is one row with
   LLM-generated metadata (importance, keywords, contextual
   description). Links between rows are LLM-decided at write-time,
   not just embedding-cosine. **Familiar-Connect's `facts` and
   `people_dossiers` tables are this family today**; M2
   (importance-weighted retrieval) and M3 (citation-bearing
   reflections) close the remaining gaps.

4. **Hand-authored, rule-activated character context**
   (RisuAI / SillyTavern lorebooks). Keyword-activated entries
   inserted into the prompt deterministically. **Familiar-Connect's
   `core_instructions.md` and `character.md` files are a degenerate
   case of this** — always-on, no keys. M4 adds a real lorebook layer
   for keyword-activated worldbuilding.

## What's wired today

The current projection pipeline writes:

| Side-index | Writer | Read by |
|---|---|---|
| `fts_turns` (FTS5) | SQLite triggers on `turns` | `RagContextLayer` |
| `fts_facts` (FTS5) | SQLite triggers on `facts` | `RagContextLayer` |
| `summaries` | `SummaryWorker` | `ConversationSummaryLayer` |
| `cross_context_summaries` | `SummaryWorker` | `CrossChannelContextLayer` |
| `facts` | `FactExtractor` | `RagContextLayer` (via FTS) |
| `people_dossiers` | `PeopleDossierWorker` | `PeopleDossierLayer` |

Each writer is watermark-driven and idempotent over its watermark, so
deleting the side-index table and restarting rebuilds it from `turns`.
See [Context pipeline](context-pipeline.md) for the precise watermark
semantics.

Reads are multi-signal: the `RagContextLayer` already combines BM25 +
recent-window exclusion + (M2 will add) importance + (M6 will add)
embedding similarity. The recurring lesson from the field is that
pure-cosine retrieval is insufficient; multi-signal ranking is
state-of-practice.

## Swap points

The architecture exposes three places to swap memory strategies. Each
is a Protocol seam already in the codebase (or a small refactor away).

### 1. Layer order and selection (`Layer` Protocol)

Per-channel `prompt_layers` in `character.toml` is the per-prompt
swap. A new layer is one class implementing
`build(ctx)` + `invalidation_key(ctx)`; new layers register in
`commands/run.py::_default_assembler` and are selected by name in
TOML.

This is how M3 (`ReflectionLayer`) and M4 (`LorebookLayer`) plug in
without touching any other layer. It's also how an experiment that
wants to *replace* `ConversationSummaryLayer` with a graph-walk
summary can run side-by-side: register both, switch via TOML on a
test channel.

### 2. Projection writer (planned: `MemoryProjector` Protocol)

The current writers (`SummaryWorker`, `FactExtractor`,
`PeopleDossierWorker`, FTS triggers) collectively implement one
projection strategy: **rich-note + per-channel summary + per-person
dossier**.

M5 in [Roadmap](roadmap.md) lifts this into a `MemoryProjector`
Protocol — given a stream of new turns, write to whichever
side-indices the projector maintains. The default stays today's
implementation. A second projector (Graphiti-style bi-temporal graph,
or A-MEM-style emergent-link notes) can run alongside or replace it.
Selected via `[providers.memory].projectors` in TOML.

### 3. Retrieval ranking (`RagContextLayer` ranking function)

Today's ranking is a small private function inside `RagContextLayer`
that combines BM25 score and recency. M2 widens this to a four-signal
score (BM25 × recency × importance × embedding_similarity once M6
ships). The weights are TOML-driven so an operator can experiment
without touching code:

```toml
[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.4
importance_weight = 0.6
embedding_weight  = 0.0   # 0 disables (default until M6)
```

See [Tuning — forward-looking schema](tuning.md#forward-looking-schema).

## Why we prefer rich-note + bi-temporal over graph-only

A live disagreement in the field is whether the projection should be
a knowledge graph (Zep, Cognee, GraphRAG) or a richly-attributed flat
note collection with emergent links (A-MEM, generative agents).
Graphs are better for multi-hop relational queries ("who introduced
Alice to Bob and where?") but cost more tokens to maintain.
Rich-note systems are simpler and cheaper but degrade on multi-hop.

For a Discord familiar, multi-hop relational queries are rare —
"what's Alice's favourite drink?" and "what were we talking about
yesterday?" are the dominant patterns. Rich-note + bi-temporal facts
handles these natively at lower cost. The Graphiti-style graph stays
on the roadmap (M5) as an *additional* projector for installations
that want it, not as a replacement.

## Why we keep authored canon separate from experiential memory

Two different masters demand two different stores:

- **Authored canon** (`character.md`, future `lorebook.toml`) is
  hand-edited by the operator. It must never evolve on its own. The
  agent's memory subsystem must not write to it.
- **Experiential memory** (`turns` and its projections) is
  bot-generated. It evolves on every interaction.

Mixing them lets the agent rewrite its own character description,
which is exactly the failure mode Letta's destructive `core_memory_*`
tools enable. Keeping them separate makes the trust boundary
inspectable: every line of `character.md` was written by a human;
every line of `dossier_text` was written by an LLM.

## Operator playbook

### Wipe a side-index without losing data

```bash
sqlite3 data/familiars/<id>/history.db "DELETE FROM facts;"
# next FactExtractor tick rebuilds from turns
```

The `turns` table is the source of truth; any side-index can be
rebuilt by deleting its rows. This is the recommended fallback when a
prompt looks wrong: you can always nuke the projection and let the
worker re-run. See [Context pipeline — single-writer pattern](context-pipeline.md#single-writer-pattern).

### Replace `dossier_text` for one person

```sql
UPDATE people_dossiers
SET dossier_text = '...', last_fact_id = 0
WHERE canonical_key = 'discord:1234';
```

Setting `last_fact_id = 0` forces the worker to re-fold every fact on
its next tick. Use this if a hand-fixed dossier needs to incorporate
prior facts immediately.

### A/B a prompt layer on one channel

`[channels.<id>].prompt_layers` overrides the default order on a
single Discord channel — perfect for a quiet test channel where you
can compare with-and-without a candidate layer side by side. See
[Configuration model — channel overrides](configuration-model.md).
