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
- **`TTS provider '<name>' has no wired backend`** — `[tts].provider` is
  set to `azure` or `gemini`, neither of which has an implemented
  backend. TTS is disabled for the session; set
  `[tts].provider = "cartesia"` and `CARTESIA_API_KEY`.
- **`[llm.<slot>].tool_calling = false disables every tool call …`** —
  that surface can never call `shift_focus`, so its channel focus is
  fixed for the session. Set `tool_calling = true` on the slot.
- **`[Config] slot=… model=… capability=… fix=…`** — the detached
  startup audit compared the slot's declared capability flags against
  the model's OpenRouter metadata. `ERROR` means the model does not
  support what the config declares; `INFO` is the advisory inverse. See
  [Startup model diagnostics](../architecture/configuration-model.md#startup-model-diagnostics).

## Runtime symptoms

- **Bot joined voice but no audio plays** — confirm libopus loaded on
  startup (look for the `Loaded Opus from:` debug line). Without it
  `voice_client.play(...)` is silent. Also confirm a TTS provider in
  `[tts].provider` and the matching env var (`CARTESIA_API_KEY`) is set;
  with no client the player falls back to `LoggingTTSPlayer`, which
  only logs. `azure` and `gemini` have no wired backend: startup logs
  `TTS provider '<name>' has no wired backend` and disables TTS.
- **Voice transcripts come out anonymous** — every frame is unattributed
  when the SSRC → user map never fills. Look for the periodic
  `[🎙️  Voice] receive ticks=… speaking_frames=… unmapped_frames=…`
  line: `unmapped_frames` equal to `speaking_frames` means no op-5
  `Speaking` event was ever seen. See
  [Voice pipeline](../architecture/voice-pipeline.md#songbird-join-order-and-ssrc-attribution).
- **`RTCP decryption failed: Crypto(Error)` in the log** — songbird's UDP
  receive task, benign and non-fatal by design. It is filtered out by
  default (`songbird::driver::tasks::udp_rx=error`); `-vv` restores it.
- **`(playback only — no transcriber)` after `/subscribe-voice`** —
  `DEEPGRAM_API_KEY` is missing or invalid. The bot joined the channel
  and can speak, but incoming audio isn't transcribed.
- **No reply to a text message** — confirm `/subscribe-text` ran in
  that channel (subscriptions are per-channel) and that
  `subscriptions.toml` lists it.
