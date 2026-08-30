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
- **`[tts].provider '<name>' is no longer supported`** — the `azure` and
  `gemini` stubs never had a backend and were removed. Set
  `[tts].provider = "cartesia"` and `CARTESIA_API_KEY`.
- **`[llm.<slot>].tool_calling = false is unsupported …`** — silence and
  `shift_focus` are both tool calls, so that surface could neither
  decline to reply nor move. Remove the key (it defaults to `true`);
  config loading refuses the value outright, and the same text appears
  as a boot `ERROR` for a config built in process.
- **`[Config] slot=… model=… capability=… fix=…`** — the detached
  startup audit compared the slot's declared capability flags against
  the model's OpenRouter metadata. `ERROR` means the model does not
  support what the config declares — for `tool_calling` the only fix is
  a different model, since it cannot be turned off; `INFO` is the
  advisory inverse. See
  [Startup model diagnostics](../architecture/configuration-model.md#startup-model-diagnostics).

## Runtime symptoms

- **Bot joined voice but no audio plays** — confirm libopus loaded on
  startup (look for the `Loaded Opus from:` debug line). Without it
  `voice_client.play(...)` is silent. Also confirm a TTS provider in
  `[tts].provider` and the matching env var (`CARTESIA_API_KEY`) is set;
  with no client the player falls back to `LoggingTTSPlayer`, which
  only logs.
- **Voice transcripts come out anonymous** — every frame is unattributed
  when the SSRC → user map never fills. Look for the periodic
  `[🎙️  Voice] receive ticks=… speaking_frames=… unmapped_frames=…`
  line: `unmapped_frames` equal to `speaking_frames` means no op-5
  `Speaking` event was ever seen. See
  [Voice pipeline](../architecture/voice-pipeline.md#songbird-join-order-and-ssrc-attribution).
- **Voice turns are attributed to a bare user id, not a name** — the speaker
  is missing from the voice-member roster, so only the id survives. The roster
  is snapshotted from the gateway cache at join and maintained from
  voice-state updates; an occupant the cache cannot resolve (no
  `GUILD_MEMBERS` intent) stays nameless until they type or their voice state
  changes. See
  [Voice pipeline](../architecture/voice-pipeline.md#voice-member-roster).
- **`[Player] synthesize_error=…Cartesia TTS error (status=400)…`** — the
  provider rejected one chunk's transcript; the rest of the reply still
  plays, so the symptom is a missing sentence, usually the last one.
  Chunks with nothing to voice (whitespace, punctuation, or a lone
  trailing emoji) are skipped before the request goes out, so a 400 that
  survives that gate points at the request itself: check
  `[tts].cartesia_voice_id` and `[tts].cartesia_model` against the voices
  and models the account actually has. See
  [Voice pipeline](../architecture/voice-pipeline.md#sentence-streaming).
- **`RTCP decryption failed: Crypto(Error)` in the log** — songbird's UDP
  receive task, benign and non-fatal by design. It is filtered out by
  default (`songbird::driver::tasks::udp_rx=error`); `-vv` restores it.
- **`(playback only — no transcriber)` after `/subscribe-voice`** —
  `DEEPGRAM_API_KEY` is missing or invalid. The bot joined the channel
  and can speak, but incoming audio isn't transcribed.
- **No reply to a text message** — confirm `/subscribe-text` ran in
  that channel (subscriptions are per-channel) and that
  `subscriptions.toml` lists it.
