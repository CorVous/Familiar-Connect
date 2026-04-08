# Context Management Implementation Plan

This document is the concrete implementation roadmap for the Context Management architecture described in `plan.md`. Work on branch `claude/add-context-management-XpnLJ` should land roughly in the order below.

The deliberate theme is **architecture before providers**: the pipeline frame, the protocols, and the budgeter must be in place and tested *before* any real provider is wired up, so that every provider lands into a stable shape instead of driving the shape.

This document and `plan.md` are expected to stay in sync. When a design choice changes here, it changes there.

---

## Order of work

### 1. Protocols, dataclasses, and the empty pipeline

**New package:** `familiar_connect.context`.

- `ContextRequest` — dataclass holding the triggering event, the recent turn history, the active character card, the active preset, per-guild config, the target token budget, and a deadline.
- `Layer` — enum covering the layers declared in `SystemPromptLayers` (`core`, `character`, `world_info`, `rag`, `history_summary`, `recent_history`, `author_note`, `depth_inject`).
- `Contribution` — dataclass of `layer: Layer`, `priority: int`, `text: str`, `estimated_tokens: int`, `source: str`.
- `ContextProvider` — `Protocol` with `async def contribute(request: ContextRequest) -> list[Contribution]`.
- `PreProcessor` / `PostProcessor` — `Protocol`s with `async def process(...)`. Pre-processors mutate the outgoing `ContextRequest`; post-processors mutate the LLM reply before TTS.
- `ContextPipeline` — holds the registered providers and processors. Its `run(request)` method opens a `trio.open_nursery`, spawns every enabled provider with the deadline, collects the resulting contributions, hands them to the budgeter, calls the LLM, and then runs post-processors over the reply.

**Tests** — `tests/test_context_pipeline.py`:

- An empty pipeline returns the core instructions unchanged.
- Stub providers run concurrently and their contributions are collected in deterministic order.
- A slow stub provider past the deadline is dropped with a logged warning rather than stalling the pipeline.
- A failing provider does not poison other providers — the pipeline catches, logs, and continues.
- Pre-processors run in registration order; post-processors run in reverse registration order (so they wrap symmetrically).

### 2. Token budgeter

**New module:** `familiar_connect.context.budget`.

- Wrap `tiktoken` for OpenAI-family models; fall back to a character-count heuristic for non-OpenAI models. The heuristic is documented as approximate so callers know when to be conservative.
- `Budgeter.fill(layers: SystemPromptLayers, contributions, budget_by_layer)` walks contributions in priority order, assigns them to layers, truncates from the lowest-priority end when a layer's budget is exceeded, and emits a structured log entry for any dropped or truncated content.
- Truncation prefers sentence boundaries, falls back to whitespace, and finally to a hard slice.

**Tests** — `tests/test_budget.py`:

- Under-budget: everything fits, nothing is dropped.
- Over-budget at one layer: lowest-priority contributions in that layer are truncated first; higher-priority layers are untouched.
- Over-budget overall: lowest-priority layers are truncated first.
- Truncation snaps to sentence boundaries when possible.
- Dropped or truncated content is logged with `source`, `layer`, and `tokens_dropped`.

### 3. History provider + rolling summary

**New module:** `familiar_connect.context.providers.history`.

- `HistoryProvider` reads from the existing text/voice history store and emits two contributions per call: the last N turns verbatim (`Layer.recent_history`, high priority), and a `Layer.history_summary` contribution built from older turns via a cheap side-model.
- Summaries are cached in SQLite keyed by `(guild_id, channel_id, last_summarised_message_id)` so they are only regenerated when new turns age out of the sliding window. Compression target is roughly 10:1.
- Respects the deadline: if the summariser hasn't returned in time, the provider emits only the verbatim window and flags the cached summary as stale for the next run.

**Tests** — `tests/test_history_provider.py`:

- Sliding window with fewer than N turns returns everything in the recent layer and no summary.
- Sliding window with more than N turns returns the latest N plus a summary contribution.
- Cached summary is reused when no new turns have aged out (assert no extra LLM calls).
- Summariser timeout falls back gracefully (recent layer present, summary contribution absent, log entry written).

### 4. Per-guild configuration and wiring

- Extend the guild settings store with a `context` section: which providers are enabled, which model each cheap side-call uses, and per-layer token budgets.
- Wire `ContextPipeline` into `bot.py` and `text_session.py` so every reply goes through it. Replace the ad-hoc history construction currently in `text_session.py` (and remove its TODO referring to `plan.md`'s context-management design).
- Voice replies use the same pipeline. Streaming still happens at the LLM → TTS boundary; the pipeline runs to completion before the LLM call starts.

**Tests** — extend `tests/test_text_session.py` and add a voice-path equivalent using a stub LLM client:

- A reply request with no providers enabled still produces a working call (defaults to core instructions only).
- Toggling a provider via guild config changes the request body the stub LLM receives.

### 5. World Info / Lorebook provider

**New module:** `familiar_connect.context.providers.world_info`.

- Parse SillyTavern World Info / Lorebook JSON.
- Implement the keyword trigger walker: `constant`, `selective` with primary + secondary keys, `position`, `depth`, `order`, and `probability`. Recursive scan is explicitly out of scope for the first cut and listed below in *Non-goals*.
- Emit contributions tagged `Layer.world_info` with the entry's declared `order` mapped to a Familiar-Connect priority.

**Tests** — `tests/test_world_info_provider.py`:

- A `constant` entry always fires.
- A `selective` entry fires only when both its primary and secondary keys match the recent turn buffer.
- `depth` and `position` are respected in the output ordering.
- `probability < 1` is sampled deterministically when seeded.
- A malformed entry is skipped with a warning rather than crashing the pipeline.

### 6. RAG provider

**New module:** `familiar_connect.context.providers.rag`.

- `sqlite-vec` vector store over message chunks and persona facts, embedded with `text-embedding-3-small`.
- Cosine top-K with a recency decay multiplier and an optional guild / channel filter.
- Emits contributions tagged `Layer.rag` with a snippet plus a citation back to the original message id.

**Tests** — `tests/test_rag_provider.py` (with a deterministic stub embedding model):

- Query returns the top-K semantically nearest chunks.
- Recency decay boosts recent results over older ones with the same base score.
- Filtering by guild / channel excludes other sources.
- An empty store returns no contributions (no error).

### 7. Lorebook manager provider

**New module:** `familiar_connect.context.providers.lorebook_manager`.

- Implements `future-features/lorebook.md`. A cheap side-model is given a flat index of lorebook entry titles + one-line summaries and asked to choose which entries to surface for the current turn.
- Emits contributions tagged `Layer.world_info` with the manager's choices, lower priority than `WorldInfoProvider`'s constant entries so the explicit lorebook always wins ties.

**Tests** — `tests/test_lorebook_manager_provider.py`:

- With a stub cheap model returning a known subset, the correct entries are surfaced.
- Manager errors / timeouts degrade gracefully to no contribution.
- Manager returning an unknown entry id is logged and ignored.

### 8. Processor surface and two initial processors

**New modules:**

- `familiar_connect.context.processors.stepped_thinking`
- `familiar_connect.context.processors.recast`

- `SteppedThinkingPreProcessor` runs a cheap model with a focused "think step by step about what the user is really asking" prompt, appends the result as a hidden assistant-visible note in the outgoing context, and marks it so it is never surfaced to the user. Inspired by SillyTavern's `st-stepped-thinking`.
- `RecastPostProcessor` takes the main LLM reply and runs a focused cleanup pass with a cheap model: strip formatting artefacts, tighten tone, optionally rewrite for speech (since the reply is headed to TTS). Inspired by SillyTavern's `recast-post-processing`.

**Tests** — one file per processor, covering:

- Happy path (stub cheap model returns expected text).
- Timeout / failure path (processor degrades gracefully and the original message is preserved).
- Per-guild toggle takes effect.

---

## Non-goals for the first pass

Tracked here so they don't sneak back in mid-implementation.

- **Recursive World Info scanning.** Single-pass keyword matching only.
- **Cross-guild shared memory.** Each guild's lorebook, RAG store, and history are isolated.
- **A UI for editing lorebooks, world info, or provider config.** These live as files plus slash commands for now.
- **Auto-detection of same-person-under-different-usernames.** Tracked in `future-features/lorebook.md`.
- **Streaming providers.** The pipeline runs to completion before the main LLM call starts. Streaming still happens at the LLM → TTS boundary as today.
- **Plugin discovery / dynamic loading.** Providers and processors are registered in code at startup. No entry points, no folder scans.
- **Bridging to a running SillyTavern instance** (and its more aggressive variants). Rejected design — see `plan.md` § *Design Decisions Considered and Rejected* for the rationale.

---

## Definition of done for the branch

- `ContextPipeline` is the **only** path from "something happened" to "call the LLM" in both the text and voice code paths.
- All seven providers and both processors above exist, are individually toggleable per guild, and have tests.
- `plan.md` and this document stay in sync.
- `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, and `uv run pytest` all pass on the branch before pushing.
