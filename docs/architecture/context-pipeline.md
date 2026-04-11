# Context Pipeline

The context pipeline is the single path between "something happened" and "call the LLM" for both the text and voice code paths. Providers assemble contributions concurrently, a budgeter packs them into layered system-prompt slots, the LLM is called, and post-processors run over the reply before it reaches TTS or Discord.

!!! success "Status: Implemented"
    Steps 1–10 below have all shipped. The pipeline, memory store, character unpacker, lorebook importer, providers (`CharacterProvider`, `HistoryProvider`, `ContentSearchProvider`), and both initial processors (`SteppedThinkingPreProcessor`, `RecastPostProcessor`) are in place and individually toggleable per familiar and per modality.

    **Still deferred:** the per-turn monitoring dashboard, a `/context` slash command that shows the last assembled context, wiring STT input into the pipeline, and a `familiar init --from-card` subcommand. See [Roadmap](../roadmap/index.md).

## Working assumptions

- **Concurrency stack: `asyncio` + `asyncio.TaskGroup`.** The codebase is asyncio-native top to bottom.
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

- **`ContextRequest`** — triggering event, `familiar_id`, originating `channel_id`, originating `guild_id` (observability only), speaker, utterance, `Modality` (`"voice"` or `"text"`), target token budget, and a deadline.
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
- A `Layer.history_summary` contribution built from older turns via a cheap side-model.

Summaries are cached in SQLite keyed by `(familiar_id, last_summarised_id)` — global per familiar, regardless of which channel each older turn happened in — so they are only regenerated when new turns have actually been added to the familiar's history. The recent rolling window is partitioned per channel; the rolling summary is global per familiar. Compression target is roughly 10:1.

Respects the deadline: if the summariser hasn't returned in time, the provider emits only the verbatim window and flags the cached summary as stale for the next run.

## 7. Wiring: single-character install, subscriptions, channel configs

Delivered in one commit on top of step 10:

- **`familiar_connect.config`** — `CharacterConfig`, `ChannelConfig`, `ChannelMode` enum, `channel_config_for_mode` defaults table, TOML loaders via stdlib `tomllib`.
- **`familiar_connect.subscriptions`** — `SubscriptionRegistry` persisted to `data/familiars/<id>/subscriptions.toml`. Replaced the old single-slot `text_session` registry.
- **`familiar_connect.channel_config`** — `ChannelConfigStore` with lazy per-channel TOML sidecars under `data/familiars/<id>/channels/`.
- **`familiar_connect.context.render`** — `assemble_chat_messages` owns the SillyTavern-accurate `Layer.depth_inject` placement (insert at position-from-end of the full chat list, clamped to after the system prompt).
- **`familiar_connect.familiar`** — `Familiar` dataclass bundles config, memory store, history store, side model, providers, processors, subscriptions, and channel configs. `Familiar.load_from_disk` is the sole constructor. `Familiar.build_pipeline(channel_config)` filters registered providers/processors per channel mode.
- **`bot.py`** — `/awaken` and `/sleep` are gone; `/subscribe-text`, `/unsubscribe-text`, `/subscribe-my-voice`, `/unsubscribe-voice`, `/channel-full-rp`, `/channel-text-conversation-rp`, and `/channel-imitate-voice` replaced them. `on_message` routes every subscribed message through the pipeline, runs post-processors, persists both turns to `HistoryStore`, and fans out to TTS when a voice subscription exists in the same guild.
- **`commands/run.py`** — selects the active familiar via `FAMILIAR_ID` env var (or `--familiar` flag), builds the `Familiar` bundle from disk, and hands it to `create_bot`.
- **`ContextPipeline`** gained a `post_processors` parameter and a `run_post_processors` method; the bot calls it on the main LLM reply before TTS/history. Processors run in reverse registration order; a processor that raises is caught and its stage skipped.
- **Schema break** — `owner_user_id` has been dropped from `ContextRequest`, `HistoryStore`, and every call site. Existing `history.db` files need to be deleted before running this branch.
- **`text_session.py`** was deleted. History lives in `HistoryStore`; session state lives in `SubscriptionRegistry`.

## 8. `ContentSearchProvider`

Module: `familiar_connect.context.providers.content_search`.

A small tool-using cheap-model loop scoped to a single familiar's `MemoryStore`. Tools registered with the cheap model on each call:

- `list_dir(path)` → list of files and subdirectories.
- `glob(pattern)` → paths matching a glob.
- `grep(pattern, path="")` → matches with surrounding context.
- `read_file(path)` → file contents.

**Loop** — up to K tool-call turns (default 5), hard deadline. The final turn is a "return the relevant snippets" message; anything before that is tool calls.

**Output** — a single `Contribution(layer=Layer.content, ...)` with the concatenated snippets and the file paths they came from as `source`.

**Logging** — every tool call is written to a per-turn trace log so "why did the bot say X" has a reproducible answer.

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

- **Vector retrieval of any kind.** No `sqlite-vec`, no embedding API calls, no chunking strategy. Vector search becomes a *tool* the same `ContentSearchProvider` agent can call, added later, only if measurements show `grep` getting too slow.
- **Any SillyTavern keyword/World Info runtime.** Imports flatten to Markdown; there is no keyword walker.
- **Plugin discovery / dynamic loading.** Providers and processors are registered in code at startup.
- **Cross-familiar shared memory.** Each familiar's memory directory is isolated; two familiars never see each other's memories.
- **Per-guild config or per-guild familiar overrides.** A familiar's behaviour is identical regardless of which guild it's invoked in. `guild_id` is carried on `ContextRequest` as observability only.
- **LLM-driven memory housekeeping** (duplicate detection, conflict reconciliation, stale-entry flagging). Planned as a future add-on.
- **The voice "fast path + elaboration path" parallel-generation strategy.** The pipeline shape doesn't preclude it, but we don't build it now.
- **Any third-party state service.** See [Design decisions](decisions.md).
- **Bridging to a running SillyTavern instance** in any form. Rejected — see [Design decisions](decisions.md).
