# Context Pipeline

The context pipeline is the single path between "something happened" and "call the LLM" for both text and voice. Providers assemble contributions concurrently, a budgeter packs them into layered system-prompt slots, the LLM is called, and post-processors run over the reply before it reaches TTS or Discord.

!!! success "Status: Implemented"
    Steps 1–10 below have all shipped. The pipeline, memory store, character unpacker, lorebook importer, providers (`CharacterProvider`, `HistoryProvider`, `ContentSearchProvider`), and both initial processors (`SteppedThinkingPreProcessor`, `RecastPostProcessor`) are in place and individually toggleable per familiar and per modality.

    **Still deferred:** the per-turn monitoring dashboard, a `/context` slash command that shows the last assembled context, and a `familiar init --from-card` subcommand. See [Roadmap](../roadmap/index.md).

## Shape of one reply turn

```mermaid
flowchart TB
    req([ContextRequest<br/>author + utterance + modality + deadline])
    req --> tg

    subgraph tg[asyncio.TaskGroup - providers run in parallel]
        direction LR
        char[CharacterProvider<br/>self/*.md]
        hist[HistoryProvider<br/>recent + summary]
        content[ContentSearchProvider<br/>agentic grep/read]
    end

    tg --> contribs([list of Contribution<br/>layer + priority + text + tokens])
    contribs --> budget[Budgeter.fill<br/>pack by priority, truncate tail]
    budget --> pre[Pre-processors<br/>SteppedThinkingPreProcessor, ...]
    pre --> llm[OpenRouter<br/>streaming completion]
    llm --> post[Post-processors<br/>RecastPostProcessor, ...]
    post --> out([Reply → TTS / Discord<br/>+ HistoryStore write])

    classDef phase fill:#f4ecff,stroke:#6a1b9a;
    class budget,pre,llm,post phase;
```

Every provider runs with its own `asyncio.timeout`; a straggler past
its deadline is dropped and logged, not awaited, so no single slow
provider can stall the reply. A failing provider is caught and logged
too — the rest of the pipeline carries on with whatever contributions
did return.

## Working assumptions

- **Concurrency: `asyncio` + `asyncio.TaskGroup`** throughout.
- **No new long-running dependencies** (no vector DB, no memory daemon, no MCP sidecar for our own memory).
- **No third-party state services** (no mem0, no Zep, no hosted vector DB, no OpenAI embeddings in the first pass).
- **Memory source of truth is the per-familiar plain-text directory.** See [Memory](memory.md).
- **Providers and processors are registered in code.** No plugin discovery, no dynamic loading.
- **Single operator, one active familiar per process.** Multiple character folders may coexist under `data/familiars/<id>/`, but exactly one is active at a time (selected by `FAMILIAR_ID`). See [Configuration model](configuration-model.md).

---

## 1. Protocols, dataclasses, and the pipeline

Package: `familiar_connect.context`.

- `context/types.py` — `ContextRequest`, `Contribution`, `Layer` enum, `Modality` enum.
- `context/protocols.py` — `ContextProvider`, `PreProcessor`, `PostProcessor`.
- `context/pipeline.py` — `ContextPipeline` orchestrator.

Shapes:

- **`ContextRequest`** — triggering event, `familiar_id`, originating `channel_id`, originating `guild_id` (observability only), `author: Author | None` (see [Identity model](configuration-model.md#identity-model)), utterance, `Modality` (`"voice"` or `"text"`), target token budget, and a deadline.
- **`Contribution`** — `layer: Layer`, `priority: int`, `text: str`, `estimated_tokens: int`, `source: str`.
- **`Layer` enum** — `core`, `character`, `content`, `history_summary`, `recent_history`, `author_note`, `depth_inject`.
- **`ContextProvider`** — `Protocol` with `async def contribute(request: ContextRequest) -> list[Contribution]`.
- **`PreProcessor` / `PostProcessor`** — `Protocol`s with `async def process(...)`.
- **`ContextPipeline.run(request)`** — opens an `asyncio.TaskGroup`, spawns every enabled provider with the per-provider deadline via `asyncio.timeout`, collects contributions, hands them to the budgeter, calls the LLM, and then runs post-processors over the reply.

Guarantees:

- Slow providers past their deadline are dropped with a logged warning; the pipeline never stalls on one straggler.
- A failing provider does not poison other providers — the pipeline catches, logs, and continues.
- Pre-processors run in registration order; post-processors run in reverse registration order (so they wrap symmetrically).
- Modality on the request reaches each provider, so providers can branch on it.

## 2. Token budgeter

Module: `familiar_connect.context.budget`.

- Wraps `tiktoken` for OpenAI-family models and falls back to a character-count heuristic for non-OpenAI models.
- `Budgeter.fill(layers, contributions, budget_by_layer)` walks contributions in priority order, assigns them to layers, truncates from the lowest-priority end when a layer's budget is exceeded, and emits a structured log entry for any dropped or truncated content.
- Truncation prefers sentence boundaries, falls back to whitespace, and finally to a hard slice.

A typical allocation for an 8k-token request budget, with the default
providers enabled, looks like this (numbers illustrative — the
per-layer budget is configurable per familiar):

```mermaid
---
config:
  sankey:
    showValues: false
---
sankey-beta
Request budget,Character,1500
Request budget,Recent history,2500
Request budget,History summary,800
Request budget,Content search,1500
Request budget,Stepped thinking,500
Request budget,Reserved for completion,1200
```

Layers on the right are filled highest-priority first. If a layer
overflows its slot, the lowest-priority contribution in that layer is
truncated or dropped, and the drop is logged with enough metadata that
`/context` (when it ships) can show "this turn dropped the rolling
summary because recent-history overflowed."

## 3. `MemoryStore`

Module: `familiar_connect.memory.store`. Covered in detail on the [Memory](memory.md) page — this is the single piece of new infrastructure the agentic-search design adds.

## 4. Character-card unpacker

Module: `familiar_connect.bootstrap.unpack_character`.

On familiar creation, the unpacker reads a Character Card V3 and writes one Markdown file per field into `memory/self/` (`description.md`, `personality.md`, `scenario.md`, `first_mes.md`, `mes_example.md`, `system_prompt.md`, `post_history_instructions.md`). The original card bytes are preserved alongside as `self/.original.png` so a future unpacker revision can re-run against the source.

Idempotent: re-unpacking the same card is a no-op; re-unpacking a *different* card errors unless `overwrite=True` is passed.

See the [Bootstrapping guide](../guides/bootstrapping.md) for usage.

## 5. `CharacterProvider`

Module: `familiar_connect.context.providers.character`.

Reads `self/*.md` from the familiar's `MemoryStore` and emits one or more `Contribution(layer=Layer.character, priority=HIGH)` entries. Always on — a familiar with no character is not a bot. No LLM calls; pure filesystem read.

## 6. `HistoryProvider`

Module: `familiar_connect.context.providers.history`.

Reads from the existing text/voice history store and emits two contributions per call:

- The last N turns verbatim (`Layer.recent_history`, high priority).
- A `Layer.history_summary` contribution built from older turns via the `history_summary` LLM slot.

Summaries are cached in SQLite keyed by `(familiar_id, last_summarised_id)` — global per familiar, regardless of which channel each older turn happened in — so they are only regenerated when new turns have actually been added to the familiar's history. The recent rolling window is partitioned per channel; the rolling summary is global per familiar. Compression target is roughly 10:1.

**Background refresh.** The summariser runs off the critical path. On a cache miss (or stale cache), `contribute()` returns the current cache contents immediately — empty on the very first turn, or the previous summary if one exists — and schedules a background `asyncio.Task` to rebuild the summary. The fresh value lands in the cache and is picked up on the next turn. Concurrent refreshes for the same target watermark dedupe to a single task. Same pattern applies to cross-context "meanwhile elsewhere" summaries. A first-turn user sees only the recent-history layer; from turn two onwards the summary layer is populated. This trades one turn of freshness for keeping the 8+ s summariser cost out of every reply's latency.

## 7. Wiring: single-character install, subscriptions, channel configs

Delivered in one commit on top of step 10:

- **`familiar_connect.config`** — `CharacterConfig`, `ChannelConfig`, `ChannelMode` enum, `channel_config_for_mode` defaults table, TOML loaders via stdlib `tomllib`.
- **`familiar_connect.subscriptions`** — `SubscriptionRegistry` persisted to `data/familiars/<id>/subscriptions.toml`. Replaced the old single-slot `text_session` registry.
- **`familiar_connect.channel_config`** — `ChannelConfigStore` with lazy per-channel TOML sidecars under `data/familiars/<id>/channels/`.
- **`familiar_connect.context.render`** — `assemble_chat_messages` owns the SillyTavern-accurate `Layer.depth_inject` placement (insert at position-from-end of the full chat list, clamped to after the system prompt).
- **`familiar_connect.familiar`** — `Familiar` dataclass bundles config, memory store, history store, per-slot `LLMClient`s (keyed by slot name), providers, processors, subscriptions, and channel configs. `Familiar.load_from_disk` is the sole constructor. `Familiar.build_pipeline(channel_config)` filters registered providers/processors per channel mode.
- **`bot.py`** — `/awaken` and `/sleep` are gone; `/subscribe-text`, `/unsubscribe-text`, `/subscribe-my-voice`, `/unsubscribe-voice`, `/channel-full-rp`, `/channel-text-conversation-rp`, and `/channel-imitate-voice` replaced them. `on_message` routes every subscribed message through the pipeline, runs post-processors, persists both turns to `HistoryStore`, and fans out to TTS when a voice subscription exists in the same guild.
- **`commands/run.py`** — selects the active familiar via `FAMILIAR_ID` env var (or `--familiar` flag), builds the `Familiar` bundle from disk, and hands it to `create_bot`.
- **`ContextPipeline`** gained a `post_processors` parameter and a `run_post_processors` method; the bot calls it on the main LLM reply before TTS/history. Processors run in reverse registration order; a processor that raises is caught and its stage skipped.
- **Schema break** — `owner_user_id` has been dropped from `ContextRequest`, `HistoryStore`, and every call site. Existing `history.db` files need to be deleted before running this branch.
- **`text_session.py`** was deleted. History lives in `HistoryStore`; session state lives in `SubscriptionRegistry`.

## 8. `ContentSearchProvider`

Module: `familiar_connect.context.providers.content_search`.

A multi-tier provider scoped to a single familiar's `MemoryStore`. Each `contribute()` call runs the tiers in order and returns all emitted Contributions.

### Tier 1 — Deterministic people lookup

No LLM involvement. Runs first. For every `contribute()` call:

- If `people/<request.author.slug>.md` exists, load it verbatim (exact lookup by canonical platform-id slug).
- Tokenize the utterance (single words, capitalized mid-sentence words, hyphenated phrases) and resolve each candidate against `people/_aliases.json` — the writer-maintained `{ "lowercased-name": "author-slug" }` index — then load the mapped canonical file.
- Each loaded file is truncated to ~800 tokens. Emitted at priority 85 with source `content_search.people`.

This tier is the correctness floor — the people-file guarantee documented in [Memory → People lookup guarantee](memory.md#people-lookup-guarantee). It runs regardless of whether the agent-loop tier succeeds, fails, or returns empty.

### Tier 2 — Embedding retrieval

Optional semantic retrieval over the SQLite embedding cache (see [Memory → Derived indices](memory.md#derived-indices)). Embeds the utterance via `fastembed` / ONNX and returns the top-K chunks by cosine similarity. Chunks whose `rel_path` was already loaded by tier 1 are excluded so the filter doesn't see duplicates.

The embedding index is built (or refreshed) as a fire-and-forget background task on the first `contribute()` call. A stale or missing index returns `[]` — tier 1 is the correctness floor.

### Tier 3 — Single-shot filter

One cheap-LLM call with the utterance, the top-K retrieved snippets (≤150 tokens each), and a note about which people files tier 1 already included. The model emits one of:

- `ANSWER: <merged context>` — forwarded to the main model.
- `ANSWER:` (empty) — nothing worth forwarding.
- `ESCALATE: <reason>; GREP: <pattern>` — the filter runs one `store.grep(pattern)` call, feeds the results back in a second forced-answer prompt, and surfaces whatever the model says. 2 LLM calls max.

**Output** — up to one `Contribution(layer=Layer.content, ...)` at priority 70 and source `content_search.rag`.

**Graceful failure** — if the filter or the retriever raises, the deterministic tier's contributions are still returned.

**Deterministic mode for tests** — the cheap model client is injectable so tests can substitute a scripted responder.

## 9. SillyTavern lorebook / world-info importer

Module: `familiar_connect.bootstrap.import_silly_tavern`.

Reads a SillyTavern lorebook or world-info JSON file and writes one Markdown file per entry into a subdirectory of the memory store. Each output file is plain Markdown: the entry title as the H1, the content as the body, and the trigger keywords as a short bulleted list at the top (kept for human reference; the runtime does not use them). See the [Bootstrapping guide](../guides/bootstrapping.md) for usage.

## 10. Processors

Modules:

- `familiar_connect.context.processors.stepped_thinking`
- `familiar_connect.context.processors.recast`

**`SteppedThinkingPreProcessor`** runs a cheap model with a focused "think step by step about what the user is really asking" prompt, appends the result as a hidden assistant-visible note in the outgoing context, and marks it so it is never surfaced to the user. Inspired by SillyTavern's `st-stepped-thinking`.

**`RecastPostProcessor`** takes the main LLM reply and runs a focused cleanup pass with a cheap model: strip formatting artefacts, tighten tone, optionally rewrite for speech (since the reply is headed to TTS). Inspired by SillyTavern's `recast-post-processing`.

Both are **off by default for voice** (to protect TTFB) and **on by default for text**.

---

## Non-goals for the first pass

Tracked so they don't sneak back in mid-implementation.

- **Any third-party vector store or embedding API.** Vector retrieval ships locally via `fastembed`/ONNX and a SQLite `.index/embeddings.sqlite` — see [Memory → Derived indices](memory.md#derived-indices).
- **Any SillyTavern keyword/World Info runtime.** Imports flatten to Markdown; there is no keyword walker.
- **Plugin discovery / dynamic loading.** Providers and processors are registered in code at startup.
- **Cross-familiar shared memory.** Each familiar's memory directory is isolated; two familiars never see each other's memories.
- **Per-guild config or per-guild familiar overrides.** A familiar's behaviour is identical regardless of which guild it's invoked in. `guild_id` is carried on `ContextRequest` as observability only.
- **LLM-driven memory housekeeping** (duplicate detection, conflict reconciliation, stale-entry flagging). Planned as a future add-on.
- **The voice "fast path + elaboration path" parallel-generation strategy.** The pipeline shape doesn't preclude it, but we don't build it now.
- **Any third-party state service.** See [Design decisions](decisions.md).
- **Bridging to a running SillyTavern instance** in any form. Rejected — see [Design decisions](decisions.md).
