# Configuration Model

Two config levels. Every operator knob, organised by goal:
[Tuning](tuning.md).

## 1. Bot instance config

Secrets and install selector the host needs to run the bot at all.
Set by the admin, never exposed through Discord.

- `DISCORD_BOT` — Discord bot token
- `OPENROUTER_API_KEY` — shared across every LLM call site
- `CARTESIA_API_KEY` — Cartesia TTS (required when `[tts].provider="cartesia"`)
- `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — Azure Speech
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) — Gemini TTS
- `DEEPGRAM_API_KEY` — Deepgram STT credential. Other Deepgram knobs live in `[providers.stt.deepgram]`; the matching `DEEPGRAM_*` env vars override TOML at startup. See [Tuning — STT — Deepgram](tuning.md#stt-deepgram) for the full list.
- `FAMILIAR_ID` — selects which character folder under `data/familiars/` this process runs

Where it lives: environment variables and/or a `.env` file. Never
checked into git. Never editable from inside Discord.

## 2. Character config

Per-familiar, loaded once from `data/familiars/<familiar_id>/character.toml`,
deep-merged over `data/familiars/_default/character.toml`.

Surface today:

- `display_tz` — IANA timezone (default `"UTC"`).
- `aliases` — list of names the familiar answers to.
- `[providers.history].voice_window_size` / `.text_window_size` —
  recent-history layer windows, tiered by responder (defaults 20 / 30).
  Stopgap until a dynamic budgeter ships.
- `[providers.turn_detection].strategy` — `"deepgram"` (default) or
  `"ten+smart_turn"`. See [Tuning — local turn detection](tuning.md#local-turn-detection-v1).
- `[providers.stt]` + `[providers.stt.deepgram]` — STT backend
  selector + per-backend knobs (`endpointing_ms`, `keyterms`, …).
  Only `deepgram` today; V3 widens. Per-knob env override available.
  See [Tuning — STT — Deepgram](tuning.md#stt-deepgram).
- `[llm.fast]` / `[llm.prose]` / `[llm.background]` — tiered LLM slots
  (model, temperature, optional `provider_order`, `reasoning`,
  `tool_calling`). See [Tuning — LLM slots](tuning.md#llm-slots) for
  the schema and the call-site → slot mapping.
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
