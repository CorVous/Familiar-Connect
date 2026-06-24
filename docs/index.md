# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens, understands speech, and talks back with real AI voices.

## What's here

A Discord bot that reads and speaks in text and voice channels, backed by an event bus, layered context assembly, and a per-familiar memory store (Turso for relational data, tantivy for full-text search). Events flow through processors that assemble a prompt, call an LLM, and reply.

- CLI entry: `familiar-connect run --familiar <id>`.
- Discord text + voice subscriptions (`/subscribe-text`, `/subscribe-voice`, plus the `unsubscribe-*` counterparts).
- Twitch EventSub client.
- Turso history store (`data/familiars/<id>/history.db`) with facts, summaries, dossiers, reflections, and tantivy FTS + embedding retrieval.
- OpenRouter `LLMClient`, Deepgram STT, Azure / Cartesia / Gemini TTS.

## Where to look next

- **Running the bot:** [Installation](getting-started/installation.md)
  → [On-disk layout](getting-started/on-disk-layout.md)
  → [Slash commands](getting-started/slash-commands.md).
- **Architecture:**
  [overview](architecture/overview.md)
  → [memory strategies](architecture/memory-strategies.md)
  → [voice pipeline](architecture/voice-pipeline.md).
- **Activities:** [she gets up from the screen](architecture/activities.md).
- **Every knob, one page:** [Tuning](architecture/tuning.md).
- **Prompting lessons:** [field findings](architecture/prompting.md).
- **What's next:** [Roadmap](architecture/roadmap.md).
