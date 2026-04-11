# Architecture overview

Familiar-Connect is an AI familiar that joins Discord voice channels,
listens to users, understands speech, and talks back using real AI
voices.

## Goals

- **Single runtime.** The entire backend runs as one Python process
  using `asyncio`. No separate worker scripts, no external message
  broker.
- **Unified entry point.** One `familiar-connect run` starts
  everything — Discord gateway, voice capture, transcription, LLM,
  TTS, and Twitch listener all run as concurrent tasks under a single
  `asyncio.TaskGroup`.
- **Local-first.** The context layer makes no calls to third-party
  state stores. All context state lives in-process, in the filesystem
  next to the bot, or in the bot's own SQLite. The only network calls
  in the context layer are to the LLM endpoints we're already using
  for generation.
- **Single operator, one active familiar per process.**
  Familiar-Connect is run by a single admin on their own machine —
  there is no multi-user / multi-tenant ambition. Multiple character
  folders may coexist under `data/familiars/`, but exactly one is
  active at a time. See
  [Configuration model](configuration-model.md) for the detailed
  ownership rules.

## Target architecture

All components run as coroutines within a single `asyncio` event
loop, scoped by `asyncio.TaskGroup` for structured concurrency and
clean cancellation (Python 3.13+):

```
Discord Voice → audio capture → asyncio.Queue
                                      ↓
                            Transcription (Deepgram streaming)
                                      ↓
                  asyncio.Queue (text) ← Twitch Events
                                      ↓
                    ConversationMonitor (chattiness + interjection)
                                      ↓
               Context pipeline (see Context pipeline page)
                                      ↓
                           OpenRouter (streaming)
                                      ↓
                              TTS (Cartesia / Azure) → Audio
                                      ↓
                    asyncio.Queue → Discord Voice Playback
```

Development uses red/green TDD throughout.

## Core components

### Discord bot
Built with **py-cord**. Voice send/receive uses **davey** to handle
Discord's DAVE (Audio/Video E2E Encryption) protocol. The subscription
surface and channel-mode slash commands are documented in
[Slash commands](../getting-started/slash-commands.md).

### Transcription

**Primary: Deepgram (Nova-2, streaming)**

- Native WebSocket streaming API, ~300ms latency, strong accuracy
- Handles raw PCM streams directly — maps well to Discord's audio
  pipeline
- Good Python SDK (`deepgram-sdk`)

**Fallback: faster-whisper (local)**

- Zero cost, no rate limits, no external dependency
- Requires a GPU for real-time performance
- Good offline / privacy-preserving option

Pipeline: Discord 48kHz Opus → decode to PCM → resample to 16kHz →
stream to Deepgram WebSocket (or feed chunks to faster-whisper).

!!! warning "STT not yet wired into the reply path"
    The transcription and voice-pipeline modules exist, but incoming
    voice audio is not yet fed into the context pipeline. See
    [Voice input](../roadmap/voice-input.md) for the roadmap entry.

### AI response (OpenRouter)

The LLM call is the core of the bot's reply path. Its inputs — system
prompt, retrieved knowledge, conversation history, per-user notes —
are assembled by the [Context pipeline](context-pipeline.md), *not*
inline in the bot loop. The LLM client (`familiar_connect.llm`) only
speaks to OpenRouter; it is deliberately unaware of where its messages
came from so the pipeline can be tested and extended in isolation.

- **Provider:** OpenRouter. Default model `openai/gpt-4o`, overridable
  per familiar via config and via `OPENROUTER_MODEL`.
- **Streaming:** Responses are streamed so the TTS path can start
  speaking before the full reply arrives.
- **Cheap side-model slot:** A smaller, faster model (e.g.
  `openai/gpt-4o-mini`) is made available to providers and processors
  for focused sub-tasks — summarisation, lorebook management, stepped
  thinking, recast-style cleanup — without inflating the main call.

### Context pipeline

Everything upstream of the OpenRouter call — character cards, system
prompt assembly, memory retrieval, conversation history, cheap
side-model calls — is assembled by a single **context pipeline** that
runs as a scoped `asyncio.TaskGroup` on every reply. The pipeline is
the architectural backbone for all "AI behaviour knobs" in the bot.

See [Context pipeline](context-pipeline.md) for the full design and
step-by-step implementation history, and [Memory](memory.md) for the
on-disk memory directory the pipeline reads and writes.

### Text-to-speech

**Primary: Cartesia Sonic**

- Purpose-built for real-time conversational AI
- Sub-100ms time-to-first-byte — best-in-class latency
- Native WebSocket streaming, quality rivalling ElevenLabs at lower
  cost
- Outputs 44.1kHz PCM natively
- Voice cloning support

**Secondary: Azure Speech (Neural)**

- Keep the 9 original Azure voices for nostalgia
- Mature Python SDK, good fallback if Cartesia has downtime

**Budget fallback: Fish Audio**

- Generous free tier for development/testing
- Community voice models for variety

Pipeline: LLM text → stream to Cartesia/Azure WebSocket → receive PCM
audio → resample to 48kHz Opus → feed to Discord voice playback.

### Twitch integration

Connects to Twitch EventSub WebSocket as a task in the root
`asyncio.TaskGroup`. See the [Twitch guide](../guides/twitch.md) for
the event catalogue and slash command surface.

### Monitoring dashboard

**Starlette + Hypercorn** (asyncio-native web dashboard):

- Hypercorn runs on asyncio and mounts as a task in the bot's root
  `asyncio.TaskGroup`
- Routes:
    - `/health` — JSON status of each service (Discord, Twitch,
      transcription, TTS, LLM)
    - `/events` — Recent event log via SSE or WebSocket
    - `/context` — Per-turn, per-provider latency and token metrics
      from the context pipeline, so provider/processor enable/disable
      decisions can be made from real measurements

!!! warning "Dashboard not yet shipped"
    The `PipelineOutput.outcomes` data is already captured per turn
    and `bot.py` logs a structured line per outcome; the web
    dashboard itself is a separate work item. See the
    [Context pipeline deferred list](context-pipeline.md#deferred).

## Persistence

- Raw transcripts of every conversation are stored verbatim in
  SQLite (`familiar_connect.history.store.HistoryStore`).
- The [memory directory](memory.md) contains the distilled,
  human-readable form of everything the familiar "knows." It is the
  *model's* view of the world.
- Derived artefacts — rolling summaries, future vector indices, tag
  caches — are rebuildable from the raw transcript store and the
  memory directory. Losing them is annoying but not destructive.
- Original imported character cards are kept verbatim alongside the
  unpacked `self/` files (`memory/self/.original.png`), so a future
  change to the unpacking logic can re-run against the originals.
