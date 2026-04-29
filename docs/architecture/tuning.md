# Tuning

Single-point reference for every operator-tunable knob. If a value
isn't on this page, open an issue or add it.

The split:

- **Secrets and install selectors live in env.** Tokens, API keys,
  active familiar id. Never in git.
- **Behaviour and tuning knobs live in `character.toml`.**
  Per-familiar, deep-merged over `_default/character.toml`.
  `[channels.<id>]` overrides per Discord channel.

Where a knob lives in env today but isn't a secret (STT thresholds),
this page documents both the current location and the planned TOML
home. [Roadmap A2](roadmap.md#a2-consolidate-stt-env-vars-into-toml)
tracks the migration.

## Where each knob lives

| Category | Today | Planned home |
|---|---|---|
| Discord / OpenRouter / TTS / STT credentials | env | env (unchanged) |
| Familiar id | env (`FAMILIAR_ID`) or `--familiar` | env (unchanged) |
| LLM model + temperature per slot | `[llm.<slot>]` | unchanged |
| TTS provider + voice | `[tts]` | unchanged |
| History window + prompt layer order | `[providers.history]`, `[channels.<id>]` | unchanged |
| Deepgram STT thresholds & key-terms | env (`DEEPGRAM_*`) | `[providers.stt.deepgram]` (A2) |
| Turn detection strategy | Deepgram-only | `[providers.turn_detection]` (V1) |
| Memory projector selection | hard-wired | `[providers.memory]` (M5) |
| Voice pipeline mode + sentence streaming | cascaded, full-buffer | `[providers.voice_pipeline]` (V2 / V5) |
| RAG / fact retrieval ranking | private constants | `[memory.retrieval]` (M2 / M6) |

## Environment variables

Set in `.env` or the host environment. Never log them.

### Required

| Var | Purpose |
|---|---|
| `DISCORD_BOT` | Discord bot token. |
| `OPENROUTER_API_KEY` | Shared across every LLM call site. |
| `DEEPGRAM_API_KEY` | STT credential. |
| `FAMILIAR_ID` | Character folder under `data/familiars/`. Overridable by `--familiar`. |

### TTS provider credentials (one set, depending on `[tts].provider`)

| Var | Provider |
|---|---|
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | Azure (default). |
| `CARTESIA_API_KEY` | Cartesia. |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini. |

## Character TOML — current schema

Source of truth: `data/familiars/_default/character.toml`. Every
field is overridable per familiar; `[channels.<id>]` overrides per
Discord channel.

```toml
display_tz = "UTC"
aliases    = []

[providers.history]
window_size = 20

[tts]
provider          = "azure"      # "azure" | "cartesia" | "gemini"
azure_voice       = "en-US-AmberNeural"
cartesia_voice_id = "..."
cartesia_model    = "sonic-3"
gemini_voice      = "Kore"
gemini_model      = "gemini-3.1-flash-tts-preview"
greetings         = []

[llm.main_prose]
model       = "z-ai/glm-5.1"
temperature = 0.7

[channels.123456789012345678]
history_window_size = 30
prompt_layers       = ["core", "card", "operating_mode", "recent_history"]
message_rendering   = "prefixed"  # "prefixed" | "name_only"
```

## Tuning by goal

### Lower voice latency

1. **`DEEPGRAM_ENDPOINTING_MS`** (default `500`). Drop to `300` for
   snappier finals; raise to `700` if it cuts mid-sentence. Right
   long-term answer: a semantic turn classifier (V1).
2. **`DEEPGRAM_UTTERANCE_END_MS`** (default `1500`). Speech-end
   grace. Lower = faster handoff, more risk of mid-sentence cuts.
3. **`[channels.<id>].history_window_size`**. Lower on busy
   channels — smaller prompt, lower LLM TTFT.
4. **`[tts].provider = "cartesia"`**. Lowest hosted
   time-to-first-audio of the three.

V1 (local VAD + Smart Turn) and V2 (sentence streaming) are where
the big wins live. Today's knobs are within a few hundred
milliseconds of each other.

### Better turn handling in busy channels

- **`[channels.<id>].history_window_size`** — bump to 30–40 for
  more "is this turn for me?" context.
- **`[channels.<id>].message_rendering = "prefixed"`** — keeps the
  `[HH:MM Display Name]` prefix; the rhythm of timestamps helps
  the model judge multi-party flow.
- **`<silent>` sentinel** — already wired (see
  [multi-party addressivity](context-pipeline.md#multi-party-addressivity)).
  Don't override the sentinel instruction in the character prompt.

### Better long-term memory

- **`[providers.history].window_size`** — recent-history layer
  default. Larger = more raw context, bigger prompt.
- **`SummaryWorker.turns_threshold`** (default `10`). New turns
  before the rolling summary regenerates. Constructor arg in
  `commands/run.py`; planned move to TOML.
- **`PeopleDossierLayer.max_people`** (default `8`). Hard cap on
  dossier rows per prompt.
- **Roadmap M2 / M6** — importance-weighted retrieval and
  embeddings are the bigger wins.

### A/B a strategy on one channel

`[channels.<id>].prompt_layers` overrides default layer order on
one Discord channel. Compare candidate vs control side by side.
Once A1 lands, the same per-channel mechanism extends to STT,
turn detection, and voice pipeline mode.

## STT — Deepgram

Today: env. Defaults bias toward fewer mid-sentence cuts during
thinking pauses; lower the silence thresholds for snappier finals.

| Var | Default | Purpose |
|---|---|---|
| `DEEPGRAM_MODEL` | `nova-3` | Model name. |
| `DEEPGRAM_LANGUAGE` | `en` | Language code. |
| `DEEPGRAM_ENDPOINTING_MS` | `500` | Silence ms before a segment finalizes. |
| `DEEPGRAM_UTTERANCE_END_MS` | `1500` | Speech-end grace window. |
| `DEEPGRAM_SMART_FORMAT` | `true` | Punctuation, number/date/unit normalization. |
| `DEEPGRAM_PUNCTUATE` | `true` | Explicit punctuation pass. |
| `DEEPGRAM_KEYTERMS` | _(empty)_ | Comma-separated jargon to bias nova-3 (e.g. `"lifecycle mesh, Tam"`). |
| `DEEPGRAM_REPLAY_BUFFER_S` | `5.0` | Seconds replayed after WebSocket reconnect. |
| `DEEPGRAM_KEEPALIVE_INTERVAL_S` | `3.0` | Keepalive ping cadence. |
| `DEEPGRAM_RECONNECT_MAX_ATTEMPTS` | `5` | Reconnect attempts before giving up. |
| `DEEPGRAM_RECONNECT_BACKOFF_CAP_S` | `16.0` | Reconnect backoff cap. |
| `DEEPGRAM_IDLE_CLOSE_S` | `30.0` | Per-user stream closed after this many silent seconds; reopened on next chunk. `0` disables. |

### Planned consolidation

[A2](roadmap.md#a2-consolidate-stt-env-vars-into-toml) moves
non-secret knobs to `[providers.stt.deepgram]`. Env continues to
override TOML (container-friendly).

```toml
# planned, not yet wired
[providers.stt]
backend = "deepgram"

[providers.stt.deepgram]
model               = "nova-3"
language            = "en"
endpointing_ms      = 500
utterance_end_ms    = 1500
smart_format        = true
punctuate           = true
keyterms            = ["lifecycle mesh", "Tam"]
replay_buffer_s     = 5.0
keepalive_interval_s = 3.0
reconnect_max_attempts = 5
reconnect_backoff_cap_s = 16.0
idle_close_s        = 30.0
```

## TTS

Already TOML-driven. `[tts]` selects provider + per-provider voice
/ model. Provider-specific keys are read only when that provider
is selected.

| Provider | Voice field | Model field | Extras |
|---|---|---|---|
| `azure` (default) | `azure_voice` | (built-in) | — |
| `cartesia` | `cartesia_voice_id` | `cartesia_model` | — |
| `gemini` | `gemini_voice` | `gemini_model` | `gemini_style`, `gemini_scene`, `gemini_pace`, `gemini_accent`, `gemini_context`, `gemini_audio_profile` |

`greetings = ["..."]` pre-synthesises greeting audio at startup so
first speech doesn't pay TTS cold-start.

## LLM slots

`[llm.main_prose]` is the only slot today. Schema is open; future
slots (`summary`, `fact_extraction`, `turn_classifier`) plug in
under `[llm.<slot>]`. Each picks its model independently — fast /
cheap for background work, stronger for user-facing reply. See
`familiar_connect.config.LLM_SLOT_NAMES`.

```toml
[llm.main_prose]
model       = "z-ai/glm-5.1"
temperature = 0.7
```

## History / context layers

Constants live in `commands/run.py`. Roadmap A1 moves them to TOML.

| Knob | Default | Source |
|---|---|---|
| `RecentHistoryLayer.window_size` | `20` | `[providers.history].window_size` |
| `CrossChannelContextLayer.ttl_seconds` | `600` | constructor arg |
| `SummaryWorker.turns_threshold` | `10` | constructor arg |
| `SummaryWorker.cross_k` | `5` | constructor arg |
| `SummaryWorker.tick_interval_s` | `5.0` | class default |
| `FactExtractor.batch_size` | `10` | constructor arg |
| `FactExtractor.tick_interval_s` | `15.0` | class default |
| `PeopleDossierWorker.tick_interval_s` | `20.0` | class default |
| `PeopleDossierLayer.window_size` | matches history window | constructor arg |
| `PeopleDossierLayer.max_people` | `8` | constructor arg |
| `RagContextLayer.max_results` | `5` | constructor arg |
| `RagContextLayer.recent_window_size` | matches history window | constructor arg |

### Per-channel overrides

Today's `[channels.<id>]` knobs:

- `history_window_size` — overrides the global default.
- `prompt_layers` — explicit ordered list of layer names.
- `message_rendering` — `"prefixed"` or `"name_only"`.

## Forward-looking schema

Documented now so the schema is settled before wiring lands. Not
read by today's code.

```toml
# planned (A1 / V1 / V2 / V5 / M5)

[providers.stt]
backend = "deepgram"             # | "faster_whisper" | "parakeet"

[providers.turn_detection]
strategy = "deepgram"            # | "silero+smart_turn" | "ten"

[providers.memory]
projectors = ["rich_note", "people_dossier"]

[providers.voice_pipeline]
mode               = "cascaded"  # | "s2s"
sentence_streaming = true

[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.4
importance_weight = 0.6          # M2
embedding_weight  = 0.0          # M6
```
