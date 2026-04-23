# Troubleshooting

Common startup errors and what they mean:

- **`DISCORD_BOT environment variable is not set`** — missing bot
  token. Add it to your `.env`.
- **`No familiar selected`** — neither `FAMILIAR_ID` nor `--familiar`
  was given.
- **`Familiar folder does not exist`** — create `data/familiars/<id>/`.
- **`OPENROUTER_API_KEY environment variable is required`** — missing
  OpenRouter key.
- **`Opus library not found — voice playback will not work`** — voice
  commands still run, but no audio; install libopus.
