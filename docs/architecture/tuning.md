# Tuning

Single-point reference for every operator-tunable knob in
Familiar-Connect. If you find yourself reaching for a value that
isn't on this page, please open an issue or add it.

The split is deliberate:

- **Secrets and install selectors live in environment variables.**
  Tokens, API keys, and the active familiar id. Never checked into
  git.
- **Behaviour and tuning knobs live in `character.toml`.** Per-familiar,
  deep-merged over `data/familiars/_default/character.toml`. Hot
  paths (history window, prompt layers, rendering mode) are
  per-channel overridable via `[channels.<id>]`.

When a knob currently lives in env but is not a secret (e.g. STT
endpointing thresholds), this page documents both today's location and
the planned TOML home. The roadmap's
[A2 item](roadmap.md#a2-consolidate-stt-env-vars-into-toml) tracks
the migration.

## Where each knob lives — at a glance

| Category | Today | Planned home |
|---|---|---|
| Discord / OpenRouter / TTS / STT credentials | env | env (unchanged) |
| Familiar id selection | env (`FAMILIAR_ID`) or `--familiar` flag | env (unchanged) |
| LLM model + temperature per slot | `[llm.<slot>]` | unchanged |
| TTS provider + voice | `[tts]` | unchanged |
| History window + prompt layer order | `[providers.history]`, `[channels.<id>]` | unchanged |
| Deepgram STT thresholds & key-terms | env (`DEEPGRAM_*`) | `[providers.stt.deepgram]` (A2) |
| Turn detection strategy | (Deepgram-only today) | `[providers.turn_detection]` (V1) |
| Memory projector selection | (single hard-wired set today) | `[providers.memory]` (M5) |
| Voice pipeline mode + sentence streaming | (cascaded, full-buffer today) | `[providers.voice_pipeline]` (V2 / V5) |
| RAG / fact retrieval ranking weights | (private constants today) | `[memory.retrieval]` (M2 / M6) |

The "planned home" column is the target schema; today's values are the
authoritative source until the roadmap items ship.

## Environment variables (secrets and selectors)

These stay in env regardless of how the rest of the config evolves.
Set in `.env` or the host environment. Never log them.

### Required

| Var | Purpose |
|---|---|
| `DISCORD_BOT` | Discord bot token. |
| `OPENROUTER_API_KEY` | Shared across every LLM call site. |
| `DEEPGRAM_API_KEY` | STT credential. |
| `FAMILIAR_ID` | Which character folder under `data/familiars/`. Overridable by `--familiar`. |

### TTS provider credentials (one set, depending on `[tts].provider`)

| Var | Provider |
|---|---|
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | Azure (default). |
| `CARTESIA_API_KEY` | Cartesia. |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini. |

## Character TOML — current schema

The full schema lives in
`data/familiars/_default/character.toml`. Every field is overridable
per familiar; `[channels.<id>]` sub-tables override per Discord
channel.

```toml
display_tz = "UTC"
aliases    = []

[providers.history]
window_size = 20

[tts]
provider          = "azure"             # "azure" | "cartesia" | "gemini"
azure_voice       = "en-US-AmberNeural"
cartesia_voice_id = "..."
cartesia_model    = "sonic-3"
gemini_voice      = "Kore"
gemini_model      = "gemini-3.1-flash-tts-preview"
greetings         = []

[llm.main_prose]
model       = "z-ai/glm-5.1"
temperature = 0.7

[channels.123456789012345678]            # one block per Discord channel
history_window_size = 30
prompt_layers       = ["core", "card", "operating_mode", "recent_history"]
message_rendering   = "prefixed"         # "prefixed" | "name_only"
```

## Tuning by goal

### "I want lower voice latency"

The biggest wins, in order of leverage:

1. **`DEEPGRAM_ENDPOINTING_MS`** (default `500`). Drop to `300` for
   snappier finals if the bot is cutting people off too rarely; raise
   to `700` if it's cutting them off mid-sentence. See
   [Voice pipeline — turn detection](voice-pipeline.md#turn-detection)
   — the right long-term answer is a semantic turn classifier
   (roadmap V1), not just lowering the timeout.
2. **`DEEPGRAM_UTTERANCE_END_MS`** (default `1500`). Speech-end grace
   window. Lower = faster turn handoff, more risk of mid-sentence
   cuts.
3. **`[channels.<id>].history_window_size`**. Lower this on
   high-traffic channels — fewer recent turns means smaller prompts
   means lower LLM TTFT.
4. **`[tts].provider = "cartesia"`**. Cartesia's streaming TTS has
   the lowest hosted time-to-first-audio. Azure is fine; Gemini is
   currently the slowest of the three.

Roadmap items V1 (local VAD + Smart Turn) and V2 (sentence streaming)
are where the larger wins come from; today's knobs are within a few
hundred milliseconds of each other.

### "I want better turn handling in busy channels"

- **`[channels.<id>].history_window_size`**. Larger window means
  more context for "is this turn for me?" reasoning. The default 20
  is balanced; bump to 30–40 in high-traffic channels.
- **`[channels.<id>].message_rendering = "prefixed"`**. Always
  include the `[HH:MM Display Name]` prefix; the rhythm of timestamps
  helps the model judge multi-party flow. The model also relies on
  this to decide whether to emit the `<silent>` sentinel.
- **`<silent>` sentinel.** Already wired; see
  [multi-party addressivity](context-pipeline.md#multi-party-addressivity).
  Make sure the character's system prompt doesn't override the
  sentinel instruction.

### "I want better long-term memory"

- **`[providers.history].window_size`**. The recent-history layer's
  default. Larger = more raw context but bigger prompt.
- **`SummaryWorker.turns_threshold`** (default `10`). New turns before
  the rolling summary regenerates. Currently a constructor argument
  in `commands/run.py`; planned move to TOML.
- **`PeopleDossierLayer.max_people`** (default `8`). Hard cap on
  dossier rows rendered per prompt. Lower this if dossiers are
  blowing your prompt budget.
- **Roadmap M2 / M6** (importance-weighted retrieval, embeddings)
  are the larger wins for long-term recall.

### "I want to A/B a strategy on one channel"

`[channels.<id>].prompt_layers` overrides the default layer order on
one Discord channel. Use a quiet test channel with the candidate
layer enabled and a control channel without it; compare side by side.
Once the providers config spine (A1) lands, the same per-channel
mechanism extends to STT, turn detection, and voice pipeline mode.

## STT — Deepgram

Today these live in env. Defaults bias toward fewer mid-sentence cuts
during thinking pauses; lower the silence thresholds for snappier
finals.

| Var | Default | Purpose |
|---|---|---|
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model name. |
| `DEEPGRAM_LANGUAGE` | `en` | Language code. |
| `DEEPGRAM_ENDPOINTING_MS` | `500` | Silence ms before a segment finalizes. |
| `DEEPGRAM_UTTERANCE_END_MS` | `1500` | Speech-end grace window. |
| `DEEPGRAM_SMART_FORMAT` | `true` | Punctuation, number/date/unit normalization. |
| `DEEPGRAM_PUNCTUATE` | `true` | Explicit punctuation pass. |
| `DEEPGRAM_KEYTERMS` | _(empty)_ | Comma-separated jargon / proper nouns to bias nova-3 toward (e.g. `"rebasing, lifecycle mesh, Tam"`). |
| `DEEPGRAM_REPLAY_BUFFER_S` | `5.0` | Seconds of audio replayed after WebSocket reconnect. |
| `DEEPGRAM_KEEPALIVE_INTERVAL_S` | `3.0` | Keepalive ping cadence. |
| `DEEPGRAM_RECONNECT_MAX_ATTEMPTS` | `5` | Reconnect attempts before giving up. |
| `DEEPGRAM_RECONNECT_BACKOFF_CAP_S` | `16.0` | Reconnect backoff cap. |
| `DEEPGRAM_IDLE_CLOSE_S` | `30.0` | Per-user stream is closed after this many seconds of silence; reopened on next chunk. `0` disables. |

### Planned consolidation

Roadmap [A2](roadmap.md#a2-consolidate-stt-env-vars-into-toml) moves
these into `[providers.stt.deepgram]` in `character.toml`. Env vars
will remain as overrides for container deployments where mounting a
TOML file is awkward, with TOML as the declarative source.

```toml
# Planned schema (not yet wired)
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

Already TOML-driven. The `[tts]` section selects the provider and per-
provider voice / model. Provider-specific keys are read only when that
provider is selected.

| Provider | Voice field | Model field | Extras |
|---|---|---|---|
| `azure` (default) | `azure_voice` | (built-in) | — |
| `cartesia` | `cartesia_voice_id` | `cartesia_model` | — |
| `gemini` | `gemini_voice` | `gemini_model` | `gemini_style`, `gemini_scene`, `gemini_pace`, `gemini_accent`, `gemini_context`, `gemini_audio_profile` |

`greetings = ["..."]` pre-synthesises greeting audio at startup so the
first speech doesn't pay TTS cold-start.

## LLM slots

`[llm.main_prose]` is the only slot today. The schema is open: future
slots (e.g. `summary`, `fact_extraction`, `turn_classifier`) plug in
under `[llm.<slot>]`. Each slot picks its model independently, which
lets a fast/cheap model handle background work while a stronger model
handles the user-facing reply. See
`familiar_connect.config.LLM_SLOT_NAMES`.

```toml
[llm.main_prose]
model       = "z-ai/glm-5.1"
temperature = 0.7
```

## History / context layers

Constants live in `commands/run.py` today; the table below records the
defaults so you don't have to hunt for them. Roadmap A1 moves them to
TOML.

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

Today's overrides under `[channels.<id>]`:

- `history_window_size` — overrides the global default for this
  channel's `RecentHistoryLayer`.
- `prompt_layers` — explicit ordered list of layer names.
- `message_rendering` — `"prefixed"` (always include
  `[HH:MM display_name]` content prefix; UTC) or `"name_only"`
  (rely on the OpenAI `name` field alone — save tokens in DMs).

## Forward-looking schema

The sections below are documented now so the schema is settled before
the wiring lands. None of these are read by today's code; they will
become live as the corresponding [roadmap](roadmap.md) items ship.

```toml
# Planned (roadmap A1 / V1 / V2 / V5 / M5).

[providers.stt]
backend = "deepgram"             # | "faster_whisper" | "parakeet"

[providers.turn_detection]
strategy = "deepgram"            # | "silero+smart_turn" | "ten"

[providers.memory]
projectors = ["rich_note", "people_dossier"]
# future: ["graphiti"], ["a_mem"]

[providers.voice_pipeline]
mode               = "cascaded"  # | "s2s" (research, V5)
sentence_streaming = true        # V2

[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.4
importance_weight = 0.6          # M2
embedding_weight  = 0.0          # M6 (0 disables until embeddings land)
```
