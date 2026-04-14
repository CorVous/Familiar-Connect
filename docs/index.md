# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users,
understands speech, and talks back using real AI voices.

## What it does today

- Joins Discord text and voice channels via per-channel subscription
  commands (`/subscribe-text`, `/subscribe-my-voice`).
- Runs every reply through a **context pipeline** of pluggable providers
  (character card, recent history + rolling summary, agentic memory
  search) and processors (stepped-thinking pre-processor, recast
  post-processor).
- Replies via OpenRouter (main model + a cheap side-model slot for
  sub-tasks) and Cartesia TTS for voice playback.
- Watches Twitch channels via EventSub and feeds events (subs, bits,
  cheers, follows, ad breaks, channel-point redemptions) into the same
  conversation pipeline.
- Stores every transcript verbatim in SQLite; stores every familiar's
  long-term memory as plain Markdown files under
  `data/familiars/<id>/memory/`.

## Where to look next

- **Running the bot:** [Installation](getting-started/installation.md)
  → [On-disk layout](getting-started/on-disk-layout.md)
  → [Slash commands](getting-started/slash-commands.md).
- **Seeding a familiar from a character card or SillyTavern lorebook:**
  [Bootstrapping guide](guides/bootstrapping.md).
- **Understanding the architecture:**
  [Architecture overview](architecture/overview.md) is the entry point;
  [Context pipeline](architecture/context-pipeline.md) is the biggest
  design document.
- **What's still planned:** [Roadmap](roadmap/index.md) — carries a
  status banner on every page so you can tell at a glance what's
  shipped, what's partially wired, and what's still paper.

## Status at a glance

| Area | Status |
|---|---|
| Context pipeline, providers, processors, budgeter | **Shipped** |
| Memory directory + `MemoryStore` + `ContentSearchProvider` | **Shipped** |
| Character card unpacker, SillyTavern lorebook importer | **Shipped** |
| Per-channel subscriptions + channel modes + configuration model | **Shipped** |
| Twitch EventSub integration | **Shipped** |
| [Metrics and profiling](guides/metrics.md) (per-turn traces + CLI report) | **Shipped** |
| Conversation monitor (chattiness / interjection / lull) | [Planned](roadmap/conversation-flow.md) |
| Voice speech-to-text wired into the reply path | [Partial](roadmap/voice-input.md) — STT modules exist, not yet wired |
| Discord-side session logging (threads + live embeds) | [Planned](roadmap/session-logging.md) |
| Web search as a provider tool | [Research](roadmap/web-search.md) |
