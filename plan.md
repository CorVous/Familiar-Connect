# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

---

## Goals

- **Single runtime**: The entire backend runs as one Python process using `trio`. No separate worker scripts, no external message broker.
- **Unified entry point**: One `python main.py` starts everything — Discord gateway, voice capture, transcription, LLM, TTS, and Twitch listener all run as concurrent tasks in a `trio.Nursery`.

---

## Target Architecture

All components run as coroutines within a single `trio` event loop:

```
Discord Voice → audio capture → trio.MemoryChannel
                                      ↓
                            Transcription (Deepgram streaming)
                                      ↓
                  trio.MemoryChannel (text) ← Twitch Events
                                      ↓
                          Message Processor + Chattiness
                                      ↓
                           OpenRouter + Conversation History
                                      ↓
                              TTS (Cartesia / Azure) → Audio
                                      ↓
                    trio.MemoryChannel → Discord Voice Playback
```

Use red/green A/B TDD.

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

- **Provider:** OpenRouter. Default model `openai/gpt-4o`, overridable per guild via `/setup` and via `OPENROUTER_MODEL`.
- **Streaming:** Responses are streamed so the TTS path can start speaking before the full reply arrives.
- **Cheap side-model slot:** A smaller, faster model (e.g. `openai/gpt-4o-mini`) is made available to providers and processors for focused sub-tasks — summarisation, lorebook management, stepped thinking, recast-style cleanup — without inflating the main call.

### Context Management

Everything upstream of the OpenRouter call — character cards, system prompt assembly, world info, retrieved knowledge, conversation history, cheap side-model calls — is assembled by a single **context pipeline** that runs inside the root `trio.Nursery` on every reply. The pipeline is the architectural backbone for all "AI behaviour knobs" in the bot and is deliberately boring: a pure-ish function graph over small, typed dataclasses, with pluggable providers and processors registered once at startup.

**Architectural goals:**

1. **Swappable content, stable frame.** Character cards, preset harnesses, world info, RAG backends, and pre/post-processors all conform to a small set of protocols. Swapping any of them is a config change, never a refactor.
2. **Format-level SillyTavern interop, runtime independence.** Users can bring their SillyTavern Character Card V3 files, presets, lorebooks, and world info directly. Familiar-Connect re-implements the *behaviour* of those files in Python — it does not embed, bridge, or RPC into a running SillyTavern instance. See *Design Decisions Considered and Rejected* for the full rationale.
3. **Explicit token budgets.** Every section the pipeline assembles has a declared priority and token budget. Nothing is dropped by accident; nothing bloats by accident.
4. **Fits `trio` and low-latency voice.** Providers run concurrently in a scoped nursery with a deadline. Anything slow enough to miss the deadline is logged and dropped rather than blocking the reply.
5. **No giant framework.** No LangChain, LlamaIndex, mem0, Zep, or Haystack. See *Design Decisions Considered and Rejected*.

**Pipeline shape:**

```
incoming event → [pre-processors] → ContextRequest
                                          ↓
                             [context providers (fan-out)]
                                ├─ history (sliding window + rolling summary)
                                ├─ world info / lorebook walker
                                ├─ lorebook manager (cheap side-model)
                                ├─ RAG (sqlite-vec)
                                └─ author's note / depth injections
                                          ↓
                             [budgeter] → SystemPromptLayers
                                          ↓
                             main LLM call (streaming)
                                          ↓
                             [post-processors] → final text → TTS
```

**Core types** (live in a new `familiar_connect.context` package):

- **`ContextRequest`** — the triggering event, recent turns, active character card, active preset, per-guild config, target token budget, and deadline.
- **`Contribution`** — a typed bundle of `layer`, `priority`, `text`, `estimated_tokens`, and `source`. Providers return lists of these.
- **`ContextProvider`** — `Protocol` with one method, `async def contribute(request) -> list[Contribution]`.
- **`PreProcessor` / `PostProcessor`** — `Protocol`s with `async def process(...)`. Pre-processors mutate the outgoing request (e.g. inject a hidden chain-of-thought). Post-processors mutate the reply before it reaches TTS (e.g. tone cleanup, rewriting for speech).
- **`Budgeter`** — walks contributions in priority order, counts tokens (`tiktoken` when the model's tokenizer is known, character-count heuristic otherwise), fills the existing `SystemPromptLayers` in `familiar_connect.llm` up to per-layer and total budgets, and emits a structured log entry for any content it has to truncate or drop. Nothing is silently dropped.
- **`ContextPipeline`** — the orchestrator. Runs providers in a scoped nursery with the deadline, collects contributions, hands them to the budgeter, calls the LLM, then runs post-processors.

All of this is per-guild. A guild's config picks which providers and processors are enabled, which model each cheap side-call uses, and the per-layer token budgets.

**Context management strategy (as realised by the providers):**

- Keep the last ~20 turns verbatim in the `recent_history` layer (`HistoryProvider`).
- Older turns are compressed into a rolling summary (~10:1 ratio) by a cheap side-model, cached in SQLite keyed by the last-summarised message id so summaries are only regenerated when turns age out of the window (`HistoryProvider`).
- All raw messages are stored verbatim in SQLite with timestamps, per `future-features/persistence.md` — summaries, embeddings, and extracted people/topic entries are derived artefacts and can be regenerated from the raw store.

**Provider catalogue (initial set):**

| Provider | Role | Backing store / model |
|---|---|---|
| `HistoryProvider` | Sliding window of recent turns + rolling summary of older turns | SQLite + cheap side-model |
| `WorldInfoProvider` | Keyword-triggered lore entries (SillyTavern World Info / Lorebook format) | JSON files on disk |
| `LorebookManagerProvider` | Cheap-model pass that picks which entries in a large lorebook to surface; see `future-features/lorebook.md` | SQLite + cheap side-model |
| `RagProvider` | Cosine-similarity retrieval over embeddings of past messages and persona facts | `sqlite-vec` + `text-embedding-3-small` |
| `AuthorsNoteProvider` | Per-guild text injected at a configurable message depth | Guild config |

**Processor catalogue (initial set):**

| Processor | Kind | Role | Inspired by |
|---|---|---|---|
| `SteppedThinkingPreProcessor` | pre | Cheap-model hidden chain-of-thought appended to the outgoing context, never shown to users | SillyTavern `st-stepped-thinking` |
| `RecastPostProcessor` | post | Cheap-model focused cleanup pass on the main reply — tighten tone, strip artefacts, optionally rewrite for speech | SillyTavern `recast-post-processing` |

Each of these is a ~100-line Python module implementing the shared protocols. Adding a new provider or processor is the same shape — there is no plugin-discovery magic; everything is registered in code at startup.

**System prompt layering.** The budgeter fills the existing `SystemPromptLayers` dataclass in `familiar_connect.llm`. Priority order:

1. Core instructions and safety rails
2. Character card (personality, speech patterns, backstory)
3. World info / lorebook entries surfaced this turn
4. Retrieved RAG context
5. Conversation summary
6. Recent message history (passed as discrete `Message` objects, not concatenated into the system prompt)

**What we borrow from SillyTavern** — file formats only:

- **Character Card V3** via `character.py` (already implemented).
- **Preset JSON / `prompt_order`** via `preset.py` (already implemented).
- **World Info / Lorebook JSON** — the `WorldInfoProvider` reads ST-exported files directly.
- **Macro vocabulary** (`{{char}}`, `{{user}}`, etc.) via `macros.py`.

We do **not** embed SillyTavern, run it as a side-car, RPC into it, or host its extensions. See *Design Decisions Considered and Rejected* below for why.

The concrete "architecture before providers" implementation roadmap lives in `future-features/context-management.md`.

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
- Connects to Twitch EventSub WebSocket as a task in the root nursery
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

**Starlette + Hypercorn** (trio-native web dashboard):
- Hypercorn natively supports trio via `hypercorn.trio.serve` — runs as a nursery task alongside the bot
- Starlette is ASGI-compatible, lightweight
- Routes:
  - `/health` — JSON status of each service (Discord, Twitch, transcription, TTS, LLM)
  - `/events` — Recent event log via SSE or WebSocket
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
- **Running SillyTavern extensions outside SillyTavern requires a headless browser** driving a real ST tab (Playwright / CDP), intercepting `generate` calls, and marshaling chat state in and out. This is high-latency (the bot picked Cartesia for sub-100ms TTFB; a Chromium round-trip blows that budget), fragile across ST versions, and painful to test from inside a `trio` nursery.
- **SillyTavern's architecture assumes one user, one active chat.** Familiar-Connect is multi-guild and concurrent. Forcing every guild through a single ST session serialises the bot; spawning one ST instance per guild is a deployment nightmare.

**What we take from it instead:** file formats only — Character Card V3, presets, World Info / lorebooks, and the macro vocabulary. These give SillyTavern users a painless on-ramp without making SillyTavern a runtime dependency. See the Context Management section for the detailed split.

### Adopting a large LLM orchestration framework (LangChain, LlamaIndex, Haystack, etc.)

**The idea:** Build context management on top of an existing framework rather than rolling our own pipeline.

**Why it's rejected:**

- **All of the serious candidates are `asyncio`-first.** The bot is committed to `trio`, and the `trio-asyncio` bridge — while real — adds friction at exactly the layer (cancellation, deadlines, nursery scoping) where we care most about correctness.
- **Their memory and retrieval abstractions are less flexible than the `SystemPromptLayers` + provider pipeline already drafted in `familiar_connect.llm`.** Adopting one would mean either fighting its opinions or shoehorning our pipeline inside it.
- **Dependency surface is large.** Familiar-Connect is a single `trio` process with a tight dependency budget; a framework pulls in a lot of code we wouldn't use.

**What we take from them instead:** ideas, not code. The pipeline design borrows the "provider / retriever / processor" vocabulary these frameworks popularised, but implements it in a few hundred lines of project-local Python against our own protocols.

### Third-party managed memory services (mem0, Zep, etc.)

**The idea:** Outsource long-term memory and user-fact tracking to a managed or self-hosted memory service.

**Why it's rejected:** Same architectural friction as the frameworks above (asyncio-first, heavy, opinionated), plus data-locality concerns around sending user conversations to an external service. For the scale Familiar-Connect targets — one bot, N guilds, a single host — `sqlite-vec` plus a summariser side-model is strictly simpler and avoids adding another long-running dependency to the process tree.

### Embedding a SillyTavern extension runtime via headless browser

**The idea:** A more aggressive variant of the bridge — actually load each SillyTavern extension we want by spinning up a headless Chromium with ST inside it, intercept the LLM call, and pass results back over CDP.

**Why it's rejected:** Listed separately because it's tempting in its own right. It still falls to all the latency, fragility, multi-guild, and trio-fit problems above, *and* it adds Chromium to the runtime. The cost of maintaining the glue layer would dwarf the cost of porting the two or three extensions we actually want.

---

## Getting Started

### Prerequisites
- Python 3.11+
- SQLite (bundled with Python)
- GPU recommended for local Whisper fallback
