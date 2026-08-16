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
- **`[llm.prose].image_tools = true needs …`** — **upgrade note.** A
  familiar that started fine before this release can exit 1 here. Enabling
  `image_tools` without either delivery path used to load and silently
  degrade — `view_image` was registered but could return nothing about the
  image — and is now rejected at load instead. Only configs that set
  `[llm.prose].image_tools = true` with **both** `[llm].image_description_model`
  empty **and** `[llm.prose].multimodal = false` are affected; nothing else
  about the familiar (text, voice, alarms) had stopped working, which is why
  the misconfiguration went unnoticed. Pick one:

    ```toml
    [llm]
    image_description_model = "openai/gpt-4o"  # describe images as text

    [llm.prose]
    multimodal = true                          # or send the image itself
    ```

    Setting `image_tools = false` also clears it, and matches what the
    familiar was effectively doing before. See
    [Tuning — Vision wiring checks](../architecture/tuning.md#vision-wiring-checks).

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
