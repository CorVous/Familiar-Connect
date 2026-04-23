# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users,
understands speech, and talks back using real AI voices.

!!! warning "Demolition in progress"
    The `claude/re-arch-*` branch has removed the reply orchestration
    layer. What remains is a Discord bot shell with working plumbing
    for text, voice, Twitch EventSub, STT, TTS, OpenRouter, and
    SQLite history — but every event currently log-and-drops. The
    next reply-path design will fill this back in.

## What still boots

- CLI entry point: `familiar-connect run --familiar <id>`.
- Discord text + voice subscriptions (`/subscribe-text`,
  `/subscribe-voice`, plus their `unsubscribe-*` counterparts).
- Twitch EventSub client.
- SQLite transcript store (`data/familiars/<id>/history.db`).
- OpenRouter `LLMClient`, Deepgram / faster-whisper STT, Azure /
  Cartesia / Gemini TTS — instantiated on startup, not yet called
  from a reply path.

## Where to look next

- **Running the bot:** [Installation](getting-started/installation.md)
  → [On-disk layout](getting-started/on-disk-layout.md)
  → [Slash commands](getting-started/slash-commands.md).
- **Understanding what remains:**
  [Architecture overview](architecture/overview.md).
