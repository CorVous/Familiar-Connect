# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users,
understands speech, and talks back using real AI voices.

## What's here

A Discord bot shell with working plumbing for text, voice, Twitch
EventSub, STT, TTS, OpenRouter, and SQLite history. Incoming events
are logged and dropped — there is no reply path.

- CLI entry point: `familiar-connect run --familiar <id>`.
- Discord text + voice subscriptions (`/subscribe-text`,
  `/subscribe-voice`, plus their `unsubscribe-*` counterparts).
- Twitch EventSub client.
- SQLite transcript store (`data/familiars/<id>/history.db`).
- OpenRouter `LLMClient`, Deepgram STT, Azure / Cartesia / Gemini TTS.

## Where to look next

- **Running the bot:** [Installation](getting-started/installation.md)
  → [On-disk layout](getting-started/on-disk-layout.md)
  → [Slash commands](getting-started/slash-commands.md).
- **Architecture:**
  [overview](architecture/overview.md)
  → [memory strategies](architecture/memory-strategies.md)
  → [voice pipeline](architecture/voice-pipeline.md).
- **Tuning every knob from one page:**
  [Tuning](architecture/tuning.md).
- **What's coming next:** [Roadmap](architecture/roadmap.md).
