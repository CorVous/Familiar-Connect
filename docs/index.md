# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users,
understands speech, and talks back using real AI voices.

## What's here

A Discord bot that reads and speaks in Discord text and voice channels,
backed by an event bus, layered context assembly, and a per-familiar
memory store (Turso for relational data, tantivy for full-text search).
Events flow through processors that assemble a prompt, call an LLM,
and reply.

- CLI entry point: `familiar-connect run --familiar <id>`.
- Discord text + voice subscriptions (`/subscribe-text`,
  `/subscribe-voice`, plus their `unsubscribe-*` counterparts).
- Twitch EventSub client.
- Turso history store (`data/familiars/<id>/history.db`) with facts,
  summaries, dossiers, reflections, and tantivy FTS + embedding
  retrieval.
- OpenRouter `LLMClient`, Deepgram STT, Azure / Cartesia / Gemini TTS.

## Where to look next

- **Running the bot:** [Installation](getting-started/installation.md)
  → [On-disk layout](getting-started/on-disk-layout.md)
  → [Slash commands](getting-started/slash-commands.md).
- **Architecture:**
  [overview](architecture/overview.md)
  → [memory strategies](architecture/memory-strategies.md)
  → [voice pipeline](architecture/voice-pipeline.md).
- **Every knob, one page:** [Tuning](architecture/tuning.md).
- **What's next:** [Roadmap](architecture/roadmap.md).
