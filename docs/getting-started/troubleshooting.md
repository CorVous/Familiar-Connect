# Troubleshooting

Common startup errors and what they mean:

- **`DISCORD_BOT environment variable is not set`** — missing bot
  token. Add it to `.env`.
- **`No familiar selected`** — neither `FAMILIAR_ID` nor `--familiar`
  was given.
- **`Familiar folder does not exist`** — create the folder under the
  familiars root (see [On-disk layout](on-disk-layout.md)), or set
  `FAMILIARS_ROOT`.
- **`OPENROUTER_API_KEY environment variable is required`** — missing
  OpenRouter key.
- **`Opus library not found — voice playback will not work`** — voice
  commands still run, but no audio; install libopus.

## Runtime symptoms

- **Bot joined voice but no audio plays** — confirm libopus loaded on
  startup (look for the `Loaded Opus from:` debug line). Without it
  `voice_client.play(...)` is silent. Also confirm a TTS provider in
  `[tts].provider` and the matching env var (`AZURE_SPEECH_KEY`,
  `CARTESIA_API_KEY`, or `GOOGLE_API_KEY` / `GEMINI_API_KEY`) is set;
  with no client the player falls back to `LoggingTTSPlayer`, which
  only logs.
- **`(playback only — no transcriber)` after `/subscribe-voice`** —
  `DEEPGRAM_API_KEY` is missing or invalid. The bot joined the channel
  and can speak, but incoming audio isn't transcribed.
- **No reply to a text message** — confirm `/subscribe-text` ran in
  that channel (subscriptions are per-channel) and that
  `subscriptions.toml` lists it.
