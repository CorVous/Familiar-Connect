# Configuration Model

Two config levels.

!!! warning "Demolition in progress"
    Many prior config knobs (channel modes, interjection, interrupt
    tolerance, typing simulation, memory writer, most LLM slots) have
    been cut along with the reply orchestration layer. This page
    reflects what still loads.

## 1. Bot instance config

Secrets and install selector the host needs to run the bot at all.
Set by the admin, never exposed through Discord.

- `DISCORD_BOT` — Discord bot token
- `OPENROUTER_API_KEY` — shared across every LLM call site
- `CARTESIA_API_KEY` — Cartesia TTS (required when `[tts].provider="cartesia"`)
- `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — Azure Speech
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) — Gemini TTS
- `DEEPGRAM_API_KEY` — Deepgram STT
- `DEEPGRAM_MODEL` / `DEEPGRAM_LANGUAGE` — STT model + language (defaults: `nova-3` / `en`)
- `FAMILIAR_ID` — selects which character folder under `data/familiars/` this process runs

Where it lives: environment variables and/or a `.env` file. Never
checked into git. Never editable from inside Discord.

## 2. Character config

Per-familiar, loaded once from `data/familiars/<familiar_id>/character.toml`,
deep-merged over `data/familiars/_default/character.toml`.

Surface today:

- `display_tz` — IANA timezone (default `"UTC"`).
- `aliases` — list of names the familiar answers to.
- `[providers.history].window_size` — SQLite transcript window (default 20).
- `[llm.main_prose]` — model + optional temperature.
- `[tts]` — provider (`azure` / `cartesia` / `gemini`) + provider-specific voice / model fields.

### Default profile

A reference familiar lives at `data/familiars/_default/` and is
checked into the repo. Two purposes:

1. **Fallback source.** Any field missing from the user's
   `character.toml` falls back to the corresponding value in
   `_default/character.toml`. No hardcoded defaults live in
   Python — the default profile is the single source of truth.
2. **Documentation-by-example.** A new operator copies `_default/`
   to `data/familiars/my-familiar/` and edits from there.

The leading underscore is a convention to keep `FAMILIAR_ID=_default`
from being a meaningful selection.

### TTS providers

| Provider | Env vars | Character fields |
|---|---|---|
| `azure` (default) | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | `azure_voice` |
| `cartesia` | `CARTESIA_API_KEY` | `cartesia_voice_id`, `cartesia_model` |
| `gemini` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `gemini_voice`, `gemini_model` (+ optional style / scene / pace / accent / context / audio-profile) |

### Subscriptions

`data/familiars/<id>/subscriptions.toml` — which Discord channels
the bot listens in. Written by `/subscribe-text` and
`/subscribe-voice`. Not editable by hand in practice; the slash
commands rewrite the whole file on every mutation.
