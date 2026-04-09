# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

---

## Goals

- **Single runtime**: The entire backend runs as one Python process using `asyncio`. No separate worker scripts, no external message broker.
- **Unified entry point**: One `python main.py` starts everything — Discord gateway, voice capture, transcription, LLM, TTS, and Twitch listener all run as concurrent tasks under a single `asyncio.TaskGroup`.

---

## Target Architecture

All components run as coroutines within a single `asyncio` event loop, scoped by `asyncio.TaskGroup` for structured concurrency and clean cancellation (Python 3.13+):

```
Discord Voice → audio capture → asyncio.Queue
                                      ↓
                            Transcription (Deepgram streaming)
                                      ↓
                  asyncio.Queue (text) ← Twitch Events
                                      ↓
                          Message Processor + Chattiness
                                      ↓
               Context Management pipeline (see § Context Management)
                                      ↓
                           OpenRouter (streaming)
                                      ↓
                              TTS (Cartesia / Azure) → Audio
                                      ↓
                    asyncio.Queue → Discord Voice Playback
```

Use red/green A/B TDD.

> **Concurrency-stack migration:** earlier drafts of this plan used `trio` + `trio.Nursery` + `trio.MemorySendChannel`. The current code is a thin `trio` shepherd on top of an asyncio bot (py-cord is asyncio-native, `twitchAPI` is asyncio-native, `commands/run.py` already runs py-cord inside an asyncio worker thread), and every library the context layer wants — OpenAI/OpenRouter SDK, httpx, Starlette/Hypercorn, agent tool loops, local embedding models — is asyncio-first. Python 3.13's `asyncio.TaskGroup`, `asyncio.timeout`, and native `ExceptionGroup` cover most of what `trio.Nursery` gave us. The migration (drop `trio`, `trio-asyncio`, `pytest-trio`; `MemorySendChannel` → `asyncio.Queue`; delete the worker-thread bridge) is tracked as a separate work item. All new code on the context-management branch is written asyncio-native so we don't have to rewrite it later.

---

## Core Features

### Discord Bot
Built with **py-cord**. Voice send/receive uses **davey** to handle Discord's DAVE (Audio/Video E2E Encryption) protocol.

- **`/awaken`** — Joins your voice channel, captures audio in real-time from all speakers
- **`/sleep`** — Gracefully disconnects
- **`/setup`** — Configuration wizard (UI-driven via Discord modals) to set:
  - Familiar name, personality prompt, chattiness level (0–100)
  - Which transcription/LLM/TTS provider to use
  - Model selection, temperature

Bot token in `.env` as `DISCORD_BOT`.

Bot token in `.env` as `DISCORD_BOT`.

### Transcription

**Primary: Deepgram (Nova-2, streaming)**
- Native WebSocket streaming API, ~300ms latency, strong accuracy
- ~$0.0043/min pay-as-you-go, $200 free credit on signup
- Handles raw PCM streams directly — maps well to Discord's audio pipeline
- Good Python SDK (`deepgram-sdk`)

**Fallback: faster-whisper (local)**
- Zero cost, no rate limits, no external dependency
- Requires a GPU for real-time performance (large-v3 model, ~1-2s per 5s chunk on RTX 3060+)
- Good offline / privacy-preserving option

Pipeline: Discord 48kHz Opus → decode to PCM → resample to 16kHz → stream to Deepgram WebSocket (or feed chunks to faster-whisper).

### Message Processing & Chattiness

The bot evaluates each incoming event (transcription, text message, Twitch event) against a decision pipeline to determine whether to respond.

**Response decision heuristics (evaluated in priority order):**
1. **Direct address** (name mention, @mention, "hey familiar-name"): Always respond.
2. **Direct question to nobody specific** ("does anyone know..."): Roll against chattiness threshold.
3. **Silence detection**: If nobody speaks for N seconds (scaled by chattiness), the bot may interject.
4. **Topic relevance**: If the conversation touches the familiar's domain knowledge, increase response probability by ~20-30%.
5. **Twitch events**: Always acknowledge subs/bits/raids. Follows only at chattiness > 50.

**Chattiness slider mapping (0-100):**

| Range | Behavior |
|-------|----------|
| 0–10 | Only respond when directly addressed by name |
| 11–30 | + Respond to direct questions aimed at nobody specific |
| 31–60 | + Interject after prolonged silence (threshold = `12 - (chattiness * 0.1)` seconds). React to Twitch events. |
| 61–85 | + Probabilistic response to general conversation (`(chattiness - 60) / 100` chance per utterance) |
| 86–100 | + Comment on most topics, shorter silence threshold, react to almost everything |

**Turn-taking rules:**
- Wait 1.5–2s after the last speaker finishes (VAD silence) before starting a response
- If someone starts speaking while the bot is generating but hasn't started outputting audio, abort
- If already speaking, finish the current sentence then yield
- If interrupted twice in 60s, double the silence threshold temporarily

**Rate limiting:**
- Minimum 15–30s between unprompted responses (scales inversely with chattiness)
- Hard cap: max 3 unprompted responses per minute
- If 3+ humans are actively talking (multiple speakers in last 10s), raise the response threshold — talk less in fast-moving conversations, not more

### AI Response (OpenRouter)

The LLM call is the core of the bot's reply path. Its inputs — system prompt, retrieved knowledge, conversation history, per-user notes — are assembled by the Context Management pipeline described in the next section, *not* inline in the bot loop. The LLM client (`familiar_connect.llm`) only speaks to OpenRouter; it is deliberately unaware of where its messages came from so the pipeline can be tested and extended in isolation.

- **Provider:** OpenRouter. Default model `openai/gpt-4o`, overridable per familiar via `/setup` and via `OPENROUTER_MODEL`.
- **Streaming:** Responses are streamed so the TTS path can start speaking before the full reply arrives.
- **Cheap side-model slot:** A smaller, faster model (e.g. `openai/gpt-4o-mini`) is made available to providers and processors for focused sub-tasks — summarisation, lorebook management, stepped thinking, recast-style cleanup — without inflating the main call.

### Context Management

Everything upstream of the OpenRouter call — character cards, system prompt assembly, memory retrieval, conversation history, cheap side-model calls — is assembled by a single **context pipeline** that runs as a scoped `asyncio.TaskGroup` on every reply. The pipeline is the architectural backbone for all "AI behaviour knobs" in the bot and is deliberately boring: a pure-ish function graph over small, typed dataclasses, with pluggable providers and processors registered once at startup.

#### Core principles

1. **The source of truth is a directory of plain-text files, per familiar.** Everything else is an optimisation on top. No special file formats, no schemas, no required fields. Markdown preferred but not enforced. A human can read the whole memory with `grep -r` or a text editor; a model can read it with the same tools plus `glob` / `read`. *See § Memory Directory below.*
2. **Local-first.** The context layer makes no calls to third-party state stores. All context state lives in-process, in the filesystem next to the bot, or in the bot's own SQLite. The only network calls in the context layer are to the LLM endpoints we're already using for generation. No mem0, no Zep, no hosted vector DB, no MCP daemon for our own memory. (MCP stays on the table as a way to later *expose* Familiar-Connect's memory to other tools — not as the way Familiar-Connect consumes its own.)
3. **Single operator, one active familiar per process.** Familiar-Connect is run by a single admin on their own machine — there is no multi-user / multi-tenant ambition. Multiple character folders may coexist under `data/familiars/` (so the same install can hold Aria *and* Bob), but exactly one is active at a time: `FAMILIAR_ID` picks it at startup. Memory and the rolling history summary are global per familiar; only the *recent conversation window* is partitioned per channel so two simultaneous conversations don't bleed into each other. A user who wants two characters live at once runs two bot processes. See `future-features/configuration-levels.md` for the full configuration model and `§ Memory Directory` for the on-disk layout.
4. **Swappable content, stable frame.** Character cards, preset harnesses, content sources, and pre/post-processors all conform to a small set of protocols. Swapping any of them is a config change, never a refactor.
5. **Per-character, per-modality toggles.** Every provider and processor is individually toggleable. The enabled set can differ for voice and text because the latency budgets differ — voice will often disable slow providers that text can happily use. See § Modality Profiles.
6. **Explicit token budgets.** Every section the pipeline assembles has a declared priority and token budget. Nothing is dropped by accident; nothing bloats by accident.
7. **No giant framework as runtime.** We do not build this layer inside LangChain, LlamaIndex, mem0, Zep, or Haystack. Importing a *specific utility* from one of them (e.g. a chunker, a token counter, a document loader) is allowed when it's a net simplification. Adopting any of them as the orchestration layer is not. See *Design Decisions Considered and Rejected*.
8. **Format-level SillyTavern interop, runtime independence.** SillyTavern Character Cards (V3), presets, lorebooks, and world info can be *imported*. On import they are unpacked into plain-text files in the memory directory and never touched by a special runtime again. Familiar-Connect does not embed, bridge, or RPC into a running SillyTavern instance.

#### Memory directory

Every familiar owns a directory of plain-text files, rooted in the character folder. Default layout:

```
data/familiars/<familiar_id>/memory/
    self/
        description.md        # unpacked from character card on init
        personality.md
        scenario.md
        first_mes.md
        system_prompt.md
    people/
        alice.md
        bob.md
    topics/
        elden-ring.md
        last-tuesdays-argument-about-ska.md
    sessions/
        2026-04-07-evening.md
        2026-04-08-afternoon.md
    lore/
        house-rules.md
        backstory.md
    notes.md                  # free-form scratchpad
```

Everything about this layout is conventional, not enforced. Subdirectories are just for human ergonomics; the search agent treats the whole tree as one pile of text.

**Rules of the directory:**

- **Plain text (Markdown preferred).** No JSON, no YAML frontmatter requirements, no proprietary formats.
- **Per-install isolation.** One checkout = one familiar. Multi-character setups spin up multiple checkouts, and two familiars never share a directory. See `future-features/configuration-levels.md` for the full ownership model.
- **Markdown cross-links encouraged.** Files can reference each other with relative links like `[alice](../people/alice.md)`. This is free today (a reader just sees the link text) and sets up graph-style traversal tools later without changing the storage format.
- **Character cards are unpacked into `self/` on familiar creation.** Each field of a loaded Character Card V3 becomes a file. Editing the character is then just editing those files, and the search agent finds them the same way it finds everything else.
- **SillyTavern lorebook / world-info JSON imports are flattened to Markdown** in the appropriate subdirectory at import time. We never maintain a runtime keyword walker over ST's trigger format.
- **Session summaries are written by a post-session writer pass** (cheap side-model), one file per session, under `sessions/`.
- **Optional housekeeping passes** (duplicate detection, conflict reconciliation, stale-entry flagging) are future add-ons. They run as cheap-model jobs that read the directory and propose edits; a human reviews. Not first-cut work.

Everything fancy — a vector index, a graph walker, a pre-computed people/topic summary, a tag index — is an *optional cache built on top of this directory*. If the cache is lost, it can be rebuilt from the text. If the cache and the text ever disagree, the text wins.

#### Pipeline shape

```
incoming event → [pre-processors] → ContextRequest
                                          ↓
                             [context providers (fan-out)]
                                ├─ HistoryProvider   (recent turns + rolling summary)
                                ├─ ContentSearchProvider  (memory dir, tool-using cheap model)
                                ├─ CharacterProvider (self/ files, high priority, always on)
                                └─ AuthorsNoteProvider (depth injections, optional)
                                          ↓
                             [budgeter] → SystemPromptLayers
                                          ↓
                             main LLM call (streaming)
                                          ↓
                             [post-processors] → final text → TTS
```

#### Core types

All in a new `familiar_connect.context` package.

- **`ContextRequest`** — the triggering event, `familiar_id`, originating `channel_id`, originating `guild_id` (observability only), `speaker`, `utterance`, `modality` (`"voice"` or `"text"`), target token budget, deadline, and any contributions pre-processors have accumulated. The active `Familiar` bundle is held separately by the bot layer and looked up per turn.
- **`Contribution`** — a typed bundle of `layer`, `priority`, `text`, `estimated_tokens`, and `source`. Providers return lists of these.
- **`ContextProvider`** — `Protocol` with one method, `async def contribute(request) -> list[Contribution]`.
- **`PreProcessor` / `PostProcessor`** — `Protocol`s with `async def process(...)`. Pre-processors mutate the outgoing request (e.g. inject a hidden chain-of-thought). Post-processors mutate the reply before it reaches TTS (e.g. tone cleanup, rewriting for speech).
- **`Budgeter`** — walks contributions in priority order, counts tokens (`tiktoken` when the model's tokenizer is known, character-count heuristic otherwise), fills the existing `SystemPromptLayers` in `familiar_connect.llm` up to per-layer and total budgets, and emits a structured log entry for any content it has to truncate or drop. Nothing is silently dropped.
- **`ContextPipeline`** — the orchestrator. Opens an `asyncio.TaskGroup`, spawns enabled providers concurrently with the per-provider deadline, collects contributions, hands them to the budgeter, calls the LLM, then runs post-processors.

#### Initial provider catalogue

| Provider | Role | Backing store / model | Deadline fit |
|---|---|---|---|
| `CharacterProvider` | Reads `self/*.md` (unpacked character card) and injects it at high priority. Always on. | Filesystem | Instant |
| `HistoryProvider` | Sliding window of recent turns in the current conversation + rolling summary of older turns | SQLite + cheap side-model | Instant for window; cheap call for summary (cached) |
| `ContentSearchProvider` | The memory search agent. A cheap tool-using model with `grep` / `glob` / `read_file` tools scoped to the familiar's memory directory, run under a hard deadline, returning a bundle of relevant snippets. | Filesystem + cheap side-model | 1–3 cheap-model round trips; parallelisable with transcription (see § Modality profiles) |
| `AuthorsNoteProvider` | Per-familiar text injected at a configurable message depth | Config | Instant |

**Not in the initial catalogue (explicitly deferred):**

- **`WorldInfoProvider`** — the SillyTavern keyword walker. Dropped. Imports flatten to Markdown.
- **`LorebookManagerProvider`** — subsumed by `ContentSearchProvider`, which *is* the cheap-model manager, just with a more general toolset.
- **`VectorSearchProvider`** — not a provider at first. If/when memory directories grow past what `grep` handles comfortably, we add a `semantic_search` *tool* that the same `ContentSearchProvider` agent can call. The switch from grep to vector is a tool-registration change, not a new provider. Embeddings at that point will come from a local model (e.g. `sentence-transformers`), not a third-party API.

#### Initial processor catalogue

| Processor | Kind | Role | Inspired by |
|---|---|---|---|
| `SteppedThinkingPreProcessor` | pre | Cheap-model hidden chain-of-thought appended to the outgoing context, never shown to users | SillyTavern `st-stepped-thinking` |
| `RecastPostProcessor` | post | Cheap-model focused cleanup pass on the main reply — tighten tone, strip artefacts, optionally rewrite for speech | SillyTavern `recast-post-processing` |

Each of these is a ~100-line Python module implementing the shared protocols. Adding a new provider or processor is the same shape — there is no plugin-discovery magic; everything is registered in code at startup.

#### System prompt layering

The budgeter fills the existing `SystemPromptLayers` dataclass in `familiar_connect.llm`. Priority order:

1. Core instructions and safety rails
2. Character (from `self/` files via `CharacterProvider`)
3. Content search results (from `ContentSearchProvider`)
4. Conversation summary (from `HistoryProvider`)
5. Author's note / depth injections
6. Recent message history (passed as discrete `Message` objects, not concatenated into the system prompt)

#### Modality profiles

Voice and text have different latency budgets; the same familiar may want different context in each.

- **Text profile.** Default includes `CharacterProvider`, `HistoryProvider`, `ContentSearchProvider`, optional `SteppedThinkingPreProcessor` and `RecastPostProcessor`. Higher deadlines, more side-model calls, more tokens.
- **Voice profile.** Default includes `CharacterProvider`, `HistoryProvider`, and a stricter-deadline `ContentSearchProvider`. Pre/post processors are off by default because they cost TTFB. The voice profile can also run `ContentSearchProvider` **speculatively during transcription**: the moment Deepgram emits a stable partial transcript, the search agent starts; if the final transcript diverges too far, the speculative result is discarded.

Every provider and processor has an `enabled_for: {"text", "voice"}` field in per-character config, plus per-modality token-budget overrides. The dashboard (see § Monitoring Dashboard) surfaces per-turn latency breakdowns per provider so we can *measure* which components are worth their deadline cost in each modality.

#### Long-run voice strategy (not first pass)

A future direction worth designing the shape of, but not building yet:

- **Fast path + elaboration path.** For voice turns, run two generations in parallel: a tiny immediate-response model producing a short acknowledgment ("mm, let me think about that..."), and the full pipeline producing the real reply. The acknowledgment buys 1–2 seconds of TTS runway while the real reply finishes. When the real reply is ready, it takes over.
- This only makes sense for voice and probably only for long, context-heavy turns. It's noted here so the pipeline shape doesn't preclude it — the `ContextPipeline` is already a function, so calling it twice in parallel is trivial.

#### Persistence and data-loss safety

Per `future-features/persistence.md` and the memory-directory rules above:

- Raw transcripts of every conversation are stored verbatim in SQLite.
- The memory directory contains the distilled, human-readable form of everything the familiar "knows." It is the *model's* view of the world.
- Derived artefacts — rolling summaries, future vector indices, tag caches — are rebuildable from the raw transcript store and the memory directory. Losing them is annoying but not destructive.
- Original imported character cards are kept verbatim alongside the unpacked `self/` files, so a future change to the unpacking logic can re-run against the originals.

#### What we borrow from SillyTavern — files only

| From ST | What we do with it |
|---|---|
| Character Card V3 (`character.py`, already implemented) | Unpacked into `self/*.md` on familiar creation via `memory/unpack_character.py`; the `CharacterProvider` then surfaces them per turn |
| SillyTavern World Info / Lorebook JSON | One-shot import → flattened to Markdown files in the memory directory |
| Macro vocabulary (`{{char}}`, `{{user}}`) via `macros.py` | Still supported at prompt-assembly time |

We deliberately do **not** borrow SillyTavern's preset / `prompt_order` format. An earlier prototype on this branch did, via a now-deleted `preset.py`. It's been replaced by the layered pipeline: `Layer` ordering in `context/render.py` drives the top-to-bottom assembly, `channel_config_for_mode()` in `config.py` drives which providers run per channel mode, and per-mode prose instructions live in `data/familiars/<id>/modes/<mode>.md`. A SillyTavern preset could still be imported *into* Markdown later, the same way lorebooks are — but the runtime's prompt ordering is not preset-shaped.

We do **not** embed SillyTavern, run it as a side-car, RPC into it, or host its extensions. See *Design Decisions Considered and Rejected* for the long version.

#### Terminology cleanup

- **"Lorebook"** is a SillyTavern import-path term only. In our docs and code the concept is "the memory directory." We keep "lorebook" in the importer name so users can find it.
- **"RAG"** without qualification is banned from the docs. Use **"content retrieval"** for the general idea, **"content search"** for the agentic grep/read path, and **"vector retrieval"** for the specific future tool that uses embeddings. *"Retrieval-augmented generation"* as a pattern is what we're doing; we just shouldn't pretend it implies vectors.

#### Where the roadmap lives

The concrete "architecture before providers" implementation roadmap is `future-features/context-management.md`. The design of the memory directory and its per-person / per-topic / per-session conventions is `future-features/memory.md`.

### Text-to-Speech

**Primary: Cartesia Sonic**
- Purpose-built for real-time conversational AI
- Sub-100ms time-to-first-byte — best-in-class latency
- Native WebSocket streaming, quality rivaling ElevenLabs at lower cost
- Outputs 44.1kHz PCM natively
- Voice cloning support

**Secondary: Azure Speech (Neural)**
- Keep the 9 original Azure voices for nostalgia
- ~$16/M chars, ~100-200ms latency, rock-solid reliability
- Mature Python SDK, good fallback if Cartesia has downtime

**Budget fallback: Fish Audio**
- Generous free tier for development/testing
- Community voice models for variety

Pipeline: LLM text → stream to Cartesia/Azure WebSocket → receive PCM audio → resample to 48kHz Opus → feed to Discord voice playback.

### Twitch Integration
- Connects to Twitch EventSub WebSocket as a task in the root `asyncio.TaskGroup`
- Feeds channel events directly into the internal text queue:
  - Channel point redemptions, subscriptions, gift subs, cheers (bits), follows, ad breaks

#### Slash Commands

| Command | Options | Description |
|---------|---------|-------------|
| `/twitch connect` | `channel` (string) | Connect the familiar to a Twitch channel and begin watching for events |
| `/twitch disconnect` | — | Stop watching the current Twitch channel |
| `/twitch status` | — | Show the currently connected channel and which event types are enabled |
| `/twitch events` | `subscriptions` (bool) `cheers` (bool) `follows` (bool) `ads` (bool) | Toggle which event categories produce messages; omitted options are unchanged |
| `/twitch ads-immediate` | `enabled` (bool) | Toggle whether ad break events are sent to the LLM immediately rather than batched with the normal cycle |
| `/twitch redemptions add` | `name` (string) | Add a channel point redemption name to the allow-list |
| `/twitch redemptions remove` | `name` (string) | Remove a channel point redemption name from the allow-list |
| `/twitch redemptions list` | — | Show all redemption names currently on the allow-list |
| `/twitch redemptions clear` | — | Remove all redemption names from the allow-list |

**Notes:**
- All `/twitch` commands require a role that has the "Manage Server" permission or a configured admin role
- Twitch credentials (OAuth token, client ID) are set via `/setup` or in `.env` as `TWITCH_CLIENT_ID` and `TWITCH_ACCESS_TOKEN`; they are never accepted as slash command arguments
- Settings are persisted per Discord guild so they survive bot restarts

### Monitoring Dashboard

**Starlette + Hypercorn** (asyncio-native web dashboard):
- Hypercorn runs on asyncio and mounts as a task in the bot's root `asyncio.TaskGroup`
- Starlette is ASGI-compatible, lightweight
- Routes:
  - `/health` — JSON status of each service (Discord, Twitch, transcription, TTS, LLM)
  - `/events` — Recent event log via SSE or WebSocket
  - `/context` — Per-turn, per-provider latency and token metrics from the context pipeline, so provider/processor enable/disable decisions can be made from real measurements
  - Simple HTML page consuming these endpoints
- Controls can be added as POST endpoints
- Dependencies: `hypercorn`, `starlette`

---

## Security Guidelines

This project handles user-provided API keys and tokens (Discord bot token, Twitch OAuth, Deepgram, Cartesia, Azure, OpenAI). Treat all credentials as secrets.

### Credential Storage
- **Never hardcode tokens or API keys** in source code, config files checked into git, or log output
- Store secrets in environment variables or a `.env` file that is **gitignored**
- If persisting user-configured keys (e.g. from `/setup`), encrypt at rest using a machine-local key (e.g. via `cryptography.Fernet` with a key derived from a master secret in an env var)
- SQLite database files containing user data should not be committed to the repo

### Transport & Network
- All external API calls (Deepgram, Cartesia, Azure, Claude, Twitch) must use TLS (HTTPS / WSS) — never downgrade to plaintext
- The monitoring dashboard should bind to `127.0.0.1` by default, not `0.0.0.0`, to avoid exposing it to the network
- If the dashboard is exposed externally, require authentication (even a simple shared secret or token header)

### Logging & Error Handling
- **Never log secrets** — sanitize tokens, API keys, and auth headers from log output and error messages
- Avoid logging full request/response bodies from API calls that may contain keys
- Use structured logging so sensitive fields can be filtered consistently

### Input Validation
- Sanitize user input from Discord commands and Twitch events before passing to the LLM or storing in the database
- Treat all text from external sources (transcription output, Twitch chat, Discord messages) as untrusted
- Apply length limits on user-provided configuration values (personality prompts, familiar names) to prevent abuse

### Dependency Hygiene
- Pin dependency versions in `requirements.txt` to avoid supply-chain surprises
- Review new dependencies before adding them — prefer well-maintained packages with active security response
- Keep dependencies updated for security patches

### Principle of Least Privilege
- Discord bot should only request the permissions it actually needs (voice connect, send messages, use slash commands)
- Twitch token scopes should be minimal — only what EventSub subscriptions require
- API keys for third-party services should use the most restrictive tier/role available

---

## Design Decisions Considered and Rejected

These are ideas that were seriously considered during planning and deliberately turned down. They are recorded here so future contributors (including future maintainers revisiting the codebase) don't rediscover them from scratch without the rationale.

### Bridging to a running SillyTavern instance

**The idea:** Run SillyTavern as a side-car process and route Familiar-Connect's context assembly and/or generation through it. The appeal was obvious — SillyTavern has a large, mature extension ecosystem (World Info, stepped thinking, recast post-processing, TunnelVision RAG, and many more) and reusing it directly would short-circuit a great deal of work.

**Why it's rejected:**

- **SillyTavern is a single-user local web app, not a library.** Its extensions are browser-side JavaScript hooked into the chat UI's event bus (`eventSource`, `getContext()`, generation interceptors). None of that is reachable from an external process.
- **SillyTavern's HTTP server is a thin LLM proxy.** It exists so the browser can dodge CORS for the upstream model API. It does not run the extension pipeline. Calling it from Familiar-Connect would buy us nothing we don't already get by talking to OpenRouter directly.
- **Running SillyTavern extensions outside SillyTavern requires a headless browser** driving a real ST tab (Playwright / CDP), intercepting `generate` calls, and marshaling chat state in and out. This is high-latency (the bot picked Cartesia for sub-100ms TTFB; a Chromium round-trip blows that budget), fragile across ST versions, and painful to test alongside the bot's `asyncio.TaskGroup`-scoped concurrency.
- **SillyTavern's architecture assumes one user, one active chat.** Familiar-Connect is multi-guild and concurrent. Forcing every guild through a single ST session serialises the bot; spawning one ST instance per guild is a deployment nightmare.

**What we take from it instead:** file formats only — Character Card V3, presets, World Info / lorebooks, and the macro vocabulary. These give SillyTavern users a painless on-ramp without making SillyTavern a runtime dependency. See the Context Management section for the detailed split.

### Adopting a large LLM orchestration framework as runtime (LangChain, LlamaIndex, Haystack, etc.)

**The idea:** Build context management *inside* an existing framework's abstractions — register our providers as its components, let its runtime drive the bot's event loop, use its memory / retriever / agent classes as our primary building blocks.

**Why it's rejected as a runtime:**

- **These frameworks assume a synchronous request/response chat app and bury the prompt-assembly step we specifically want to be visible and testable.** Adopting one would mean either fighting its opinions or shoehorning our pipeline inside it.
- **LangChain's abstractions in particular have been rewritten repeatedly.** Production users commonly end up wrapping their own pipelines on top rather than depending on the framework's own memory/agent layers. We'd be writing the same glue and paying for the dependency anyway.
- **Each of them wants to own the event loop and the request lifecycle.** Familiar-Connect already has opinions about both (single-process, structured concurrency under `asyncio.TaskGroup`, per-turn deadlines, multi-modality). The framework would be working against us at the layer we care about most.

**What we do allow:** importing a *specific utility* from one of these libraries when it's a net simplification. Rule of thumb: if the import gives you a single function you call once (e.g. `llama-index`'s text splitters, a document loader for PDFs/HTML, a tokenizer helper), it's a utility — fine. If it wants to own your event loop, your prompt structure, or your retrieval flow, it's a runtime — not fine.

**What we take from them in spirit:** the "provider / retriever / processor" vocabulary these frameworks popularised. We implement it in a few hundred lines of project-local Python against our own protocols.

### Third-party managed memory services (mem0, Zep, etc.)

**The idea:** Outsource long-term memory and user-fact tracking to a managed or self-hosted memory service.

**Why it's rejected:** The project commits to a **local-first principle** (see Context Management § Core principles): all context state lives in-process, in the filesystem, or in SQLite on the same host. Sending conversation transcripts to a third-party memory service violates that principle, and the scale Familiar-Connect targets (one bot, N guilds, a single host, a learning project not headed for wide adoption) does not justify the operational or privacy cost. A plain-text memory directory plus a cheap search agent is strictly simpler and has the additional property that a human can `grep` and edit the familiar's memory directly.

This rejection also applies to **running a memory MCP server we own as a sidecar** for the bot's own internal use. MCP is useful when multiple separate agents need to share a tool surface; when both ends of the wire are inside the same Python process, in-process function calls are simpler on every axis (latency, debuggability, no socket lifecycle to manage). MCP stays on the table as a way to later *expose* Familiar-Connect's memory to external tools; it is not how Familiar-Connect consumes its own.

### Embedding a SillyTavern extension runtime via headless browser

**The idea:** A more aggressive variant of the bridge — actually load each SillyTavern extension we want by spinning up a headless Chromium with ST inside it, intercept the LLM call, and pass results back over CDP.

**Why it's rejected:** Listed separately because it's tempting in its own right. It still falls to all the latency, fragility, multi-guild, and concurrency-fit problems above, *and* it adds Chromium to the runtime. The cost of maintaining the glue layer would dwarf the cost of porting the two or three extensions we actually want.

---

## Getting Started

### Prerequisites
- Python 3.11+
- SQLite (bundled with Python)
- GPU recommended for local Whisper fallback
