# Context Management Implementation Plan

This document is the concrete implementation roadmap for the Context Management architecture described in `plan.md`. Work on branch `claude/add-context-management-XpnLJ` should land roughly in the order below.

The deliberate theme is **architecture before providers**: the pipeline frame, the protocols, the budgeter, and the memory directory must be in place and tested *before* any real provider is wired up, so that every provider lands into a stable shape instead of driving the shape.

This document and `plan.md` are expected to stay in sync. When a design choice changes here, it changes there.

---

## Working assumptions

- **Concurrency stack: `asyncio` + `asyncio.TaskGroup`.** The concurrency migration off `trio` has landed — the codebase is now asyncio-native top to bottom.
- **No new long-running dependencies** (no vector DB, no memory daemon, no MCP sidecar for our own memory).
- **No third-party state services** (no mem0, no Zep, no hosted vector DB, no OpenAI embeddings in the first pass).
- **Memory source of truth is the per-familiar plain-text directory described in `plan.md` § Memory Directory.**
- **Providers and processors are registered in code.** No plugin discovery, no dynamic loading.
- **One familiar per install.** Per `future-features/configuration-levels.md`, each checkout runs exactly one character. Multi-character setups spin up multiple checkouts. `owner_user_id` has been removed from the data model.

---

## Order of work

### 1. Protocols, dataclasses, and the empty pipeline

**New package:** `familiar_connect.context`.

Files:

- `context/__init__.py`
- `context/types.py` — `ContextRequest`, `Contribution`, `Layer` enum, `Modality` enum.
- `context/protocols.py` — `ContextProvider`, `PreProcessor`, `PostProcessor`.
- `context/pipeline.py` — `ContextPipeline` orchestrator.

Shapes:

- `ContextRequest` — triggering event, `familiar_id`, originating `channel_id`, originating `guild_id` (observability only), speaker, utterance, `Modality` (`"voice"` or `"text"`), target token budget, and a deadline. See `future-features/configuration-levels.md` for the ownership model (one familiar per install).
- `Contribution` — `layer: Layer`, `priority: int`, `text: str`, `estimated_tokens: int`, `source: str`.
- `Layer` enum — `core`, `character`, `content`, `history_summary`, `recent_history`, `author_note`, `depth_inject`.
- `ContextProvider` — `Protocol` with `async def contribute(request: ContextRequest) -> list[Contribution]`.
- `PreProcessor` / `PostProcessor` — `Protocol`s with `async def process(...)`.
- `ContextPipeline.run(request)` — opens an `asyncio.TaskGroup`, spawns every enabled provider with the per-provider deadline via `asyncio.timeout`, collects the resulting contributions, hands them to the budgeter, calls the LLM, and then runs post-processors over the reply.

**Tests** — `tests/test_context_pipeline.py`:

- An empty pipeline returns the core instructions unchanged.
- Stub providers run concurrently and their contributions are collected in deterministic order.
- A slow stub provider past the deadline is dropped with a logged warning rather than stalling the pipeline.
- A failing provider does not poison other providers — the pipeline catches, logs, and continues.
- Pre-processors run in registration order; post-processors run in reverse registration order (so they wrap symmetrically).
- Modality on the request reaches each provider, so providers can branch on it if they want.

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

### 3. `MemoryStore` — the on-disk memory directory

**New module:** `familiar_connect.memory.store`.

This is the single piece of new infrastructure the agentic-search design adds. Everything else is pipeline glue.

- `MemoryStore` owns a per-familiar directory, scoped by `familiar_id`. Default path: `data/familiars/<familiar_id>/memory/`.
- API (all synchronous file I/O — these are small text files on local disk):
  - `list_dir(rel_path: str = "") -> list[MemoryEntry]`
  - `read_file(rel_path: str) -> str`
  - `write_file(rel_path: str, content: str)` — writes via temp-file + rename.
  - `append_file(rel_path: str, content: str)`
  - `grep(pattern: str, rel_path: str = "", case_insensitive: bool = True) -> list[GrepHit]` — uses Python's `re` over the file tree. No shell-out.
  - `glob(pattern: str) -> list[str]`
- **Path-traversal safety** — every operation resolves against the store's root with `Path.resolve()` and rejects anything outside it. No `..`, no absolute paths, no symlinks out.
- **Sanity limits** — per-file size cap (configurable, default 256 KB), per-operation result cap, per-directory file count cap. Exceeding a cap raises a typed exception the search agent can observe.
- **Audit log** — every write is logged (file, length, source) so we can reconstruct "when did the bot's beliefs about Alice change."

**Tests** — `tests/test_memory_store.py`:

- Happy paths for each API method against a temporary directory fixture.
- Path-traversal attempts raise `MemoryStoreError`.
- Symlinks pointing outside the store raise.
- Size caps fire at the boundary.
- Audit log records every write.

### 4. Character-card unpacker

**New module:** `familiar_connect.memory.unpack_character`.

- On familiar creation, read the loaded `CharacterCard` (already provided by `character.py`) and write one Markdown file per field into `memory/self/`:
  - `self/description.md`
  - `self/personality.md`
  - `self/scenario.md`
  - `self/first_mes.md`
  - `self/mes_example.md`
  - `self/system_prompt.md`
  - `self/post_history_instructions.md`
- Preserves the original card bytes alongside the unpacked files as `self/.original.png` so a future unpacker revision can re-run against the original source.
- Idempotent. Re-unpacking the same card produces the same files; re-unpacking a *different* card errors unless `overwrite=True` is passed.

**Tests** — `tests/test_unpack_character.py`:

- Unpacking writes the expected files with the expected content.
- Re-unpacking the same card is a no-op.
- Re-unpacking a different card errors without `overwrite=True`.
- Unpacking an empty-field card simply omits the empty files (no empty files on disk).

### 5. `CharacterProvider`

**New module:** `familiar_connect.context.providers.character`.

- Reads `self/*.md` from the familiar's `MemoryStore` and emits one or more `Contribution(layer=Layer.character, priority=HIGH)` entries.
- Always on (not per-modality toggleable — if the familiar has no character, there's no bot).
- No LLM calls; pure filesystem read.

**Tests** — `tests/test_character_provider.py`:

- With a fully populated `self/`, all fields appear in the contributions in the declared order.
- Missing files are silently skipped.
- A familiar with no `self/` at all returns an empty list and logs a warning.

### 6. `HistoryProvider` (sliding window + rolling summary)

**New module:** `familiar_connect.context.providers.history`.

- Reads from the existing text/voice history store and emits two contributions per call: the last N turns verbatim (`Layer.recent_history`, high priority), and a `Layer.history_summary` contribution built from older turns via a cheap side-model.
- Summaries are cached in SQLite keyed by `(familiar_id, last_summarised_id)` — global per familiar, regardless of which channel each older turn happened in — so they are only regenerated when new turns have actually been added to the familiar's history. The recent rolling window is partitioned per channel; the rolling summary is global per familiar (see `plan.md` § Context Management for the rationale). Compression target is roughly 10:1.
- Respects the deadline: if the summariser hasn't returned in time, the provider emits only the verbatim window and flags the cached summary as stale for the next run.

**Tests** — `tests/test_history_provider.py`:

- Sliding window with fewer than N turns returns everything in the recent layer and no summary.
- Sliding window with more than N turns returns the latest N plus a summary contribution.
- Cached summary is reused when no new turns have aged out (assert no extra LLM calls).
- Summariser timeout falls back gracefully (recent layer present, summary contribution absent, log entry written).

### 7. Wiring, single-character install, subscriptions, channel configs ✅

**Done.** Delivered in one commit on top of step 10.

- New module `familiar_connect.config` — `CharacterConfig`, `ChannelConfig`, `ChannelMode` enum, `channel_config_for_mode` defaults table, and TOML loaders via stdlib `tomllib`.
- New module `familiar_connect.subscriptions` — `SubscriptionRegistry` persisted to `data/familiars/<id>/subscriptions.toml`. Replaces the single-slot `text_session` registry.
- New module `familiar_connect.channel_config` — `ChannelConfigStore` with lazy per-channel TOML sidecars under `data/familiars/<id>/channels/`.
- New module `familiar_connect.context.render` — `assemble_chat_messages` owns the SillyTavern-accurate `Layer.depth_inject` placement (insert at position-from-end of the full chat list, clamped to after the system prompt).
- New module `familiar_connect.familiar` — `Familiar` dataclass bundles config, memory store, history store, side model, providers, processors, subscriptions, and channel configs. `Familiar.load_from_disk` is the sole constructor. `Familiar.build_pipeline(channel_config)` filters registered providers/processors per channel mode.
- Rewrote `bot.py`: deleted `/awaken` and `/sleep`, added `/subscribe-text`, `/unsubscribe-text`, `/subscribe-my-voice`, `/unsubscribe-voice`, `/channel-full-rp`, `/channel-text-conversation-rp`, `/channel-imitate-voice`. `on_message` routes every subscribed message through the pipeline, runs post-processors, persists both turns to `HistoryStore`, and fans out to TTS when a voice subscription exists in the same guild.
- Rewrote `commands/run.py`: selects the active familiar via `FAMILIAR_ID` env var (or `--familiar` flag), builds the `Familiar` bundle from disk, and hands it to `create_bot`.
- `ContextPipeline` gained a `post_processors` parameter and a `run_post_processors` method; the bot calls it on the main LLM reply before TTS/history. Processors run in reverse registration order so later-registered processors wrap earlier ones symmetrically; a processor that raises is caught and its stage is skipped.
- Dropped `owner_user_id` from `ContextRequest`, `HistoryStore`, and every call site. Schema break — existing `history.db` files need to be deleted before running this branch.
- Deleted `text_session.py`; history lives in `HistoryStore`, session state lives in `SubscriptionRegistry`.

**Tests** — `tests/test_config.py`, `tests/test_subscriptions.py`, `tests/test_channel_config_store.py`, `tests/test_render.py`, `tests/test_familiar.py`, `tests/test_bot_message_loop.py` (replacing the old `test_text_session.py` / `test_bot_text.py` pair). Every first-party provider/processor/renderer wire path is exercised end-to-end with a stub `LLMClient` subclass.

**Deferred to later:**

- Per-turn provider / processor latency dashboard — the `PipelineOutput.outcomes` already carries the data, and `bot.py` logs a structured line per outcome; the actual web dashboard is a separate work item.
- A `/context` slash command showing the assembled context for the last reply.
- Voice-side input: the STT-to-pipeline path still isn't wired. `/subscribe-my-voice` joins the channel and keeps the PCM sink open for TTS replies, but incoming audio is a later roadmap step.
- A `familiar init --from-card` subcommand that unpacks a Character Card V3 PNG into the `data/familiars/<id>/` layout. For this branch users run the existing `unpack_character` module manually.

### 8. `ContentSearchProvider` — the memory search agent

**New module:** `familiar_connect.context.providers.content_search`.

This is the interesting one. It is a small tool-using cheap-model loop scoped to a single familiar's `MemoryStore`.

- Tools registered with the cheap model on each call:
  - `list_dir(path)` → list of files and subdirectories.
  - `glob(pattern)` → paths matching a glob.
  - `grep(pattern, path="")` → matches with surrounding context.
  - `read_file(path)` → file contents.
- Loop: up to K tool-call turns (default 5), hard deadline. The final turn is a "return the relevant snippets" message; anything before that is tool calls.
- Output: a single `Contribution(layer=Layer.content, ...)` with the concatenated snippets and the file paths they came from as `source`.
- **Logging:** every tool call is written to a per-turn trace log so "why did the bot say X" has a reproducible answer.
- **Deterministic mode for tests** — the cheap model client is injectable so tests can substitute a scripted responder.

**Tests** — `tests/test_content_search_provider.py`:

- Scripted cheap-model responses drive the loop through a list/grep/read sequence and the provider returns the expected snippets.
- Deadline exceeded mid-loop returns whatever was gathered so far, plus a log entry.
- Tool-call loop that tries a path-traversal attack is rejected by `MemoryStore` and the loop continues safely.
- Empty memory directory returns an empty list of contributions (no error).

### 9. SillyTavern lorebook / world-info importer

**New module:** `familiar_connect.memory.import_silly_tavern`.

- Reads a SillyTavern lorebook or world-info JSON file and writes one Markdown file per entry into a subdirectory of the memory store (e.g. `lore/imported/` or similar).
- Each output file is plain Markdown: the entry title as the H1, the content as the body, and the trigger keywords as a short bulleted list at the top (kept for human reference; the runtime does not use them).
- Importer prints a summary of what was written and refuses to overwrite existing files without `--force`.

**Tests** — `tests/test_import_silly_tavern.py`:

- Importing a known-shape lorebook JSON produces the expected files.
- Importing a malformed JSON errors cleanly.
- Re-importing the same file without force errors.
- Importing with force overwrites.

### 10. Processor surface and two initial processors

**New modules:**

- `familiar_connect.context.processors.stepped_thinking`
- `familiar_connect.context.processors.recast`

- `SteppedThinkingPreProcessor` runs a cheap model with a focused "think step by step about what the user is really asking" prompt, appends the result as a hidden assistant-visible note in the outgoing context, and marks it so it is never surfaced to the user. Inspired by SillyTavern's `st-stepped-thinking`.
- `RecastPostProcessor` takes the main LLM reply and runs a focused cleanup pass with a cheap model: strip formatting artefacts, tighten tone, optionally rewrite for speech (since the reply is headed to TTS). Inspired by SillyTavern's `recast-post-processing`.
- Both are off by default for voice (to protect TTFB) and on by default for text.

**Tests** — one file per processor, covering:

- Happy path (stub cheap model returns expected text).
- Timeout / failure path (processor degrades gracefully and the original message is preserved).
- Per-modality toggle takes effect.

---

## Non-goals for the first pass

Tracked here so they don't sneak back in mid-implementation.

- **Vector retrieval of any kind.** No `sqlite-vec`, no embedding API calls, no chunking strategy. Vector search becomes a *tool* the same `ContentSearchProvider` agent can call, added later, only if measurements show `grep` getting too slow.
- **Any SillyTavern keyword/World Info runtime.** Imports flatten to Markdown; there is no keyword walker.
- **Plugin discovery / dynamic loading.** Providers and processors are registered in code at startup. No entry points, no folder scans.
- **Cross-familiar shared memory.** Each familiar's memory directory is isolated under its owner's user directory. Two familiars (even owned by the same user) never see each other's memories.
- **Per-guild config or per-guild familiar overrides.** A familiar's behaviour is identical regardless of which guild it's invoked in. `guild_id` is carried on `ContextRequest` as observability only; per-guild scaling is a future extension noted in `future-features/configuration-levels.md`.
- **LLM-driven memory housekeeping** (duplicate detection, conflict reconciliation, stale-entry flagging). Planned as a future add-on; not first-cut.
- **The voice "fast path + elaboration path"** parallel-generation strategy. The pipeline shape doesn't preclude it, but we don't build it now.
- **Any third-party state service.** See `plan.md` § Design Decisions Considered and Rejected.
- **Bridging to a running SillyTavern instance** in any form. Rejected design — see `plan.md` § Design Decisions Considered and Rejected.

---

## Definition of done for the branch

- `ContextPipeline` is the **only** path from "something happened" to "call the LLM" in both the text and voice code paths.
- The memory directory exists, character cards are unpacked into it on familiar creation, and `ContentSearchProvider` can search it via a cheap tool-using model with a deterministic-mode test harness.
- `CharacterProvider`, `HistoryProvider`, `ContentSearchProvider`, and both processors exist, are individually toggleable per character *and* per modality, and have tests.
- A SillyTavern lorebook importer exists and is documented.
- The monitoring dashboard shows per-provider, per-turn latency and token usage.
- `plan.md`, `future-features/context-management.md`, and `future-features/memory.md` stay in sync.
- `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, and `uv run pytest` all pass on the branch before pushing.
