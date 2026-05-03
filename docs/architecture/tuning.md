# Tuning

Single-point reference for every operator-tunable knob. If a value
isn't on this page, open an issue or add it.

The split:

- **Secrets and install selectors live in env.** Tokens, API keys,
  active familiar id. Never in git.
- **Behaviour and tuning knobs live in `character.toml`.**
  Per-familiar, deep-merged over `_default/character.toml`.
  `[channels.<id>]` overrides per Discord channel.

Non-secret operator knobs live in TOML (`character.toml` fields and
their `[channels.<id>]` overrides). The corresponding env var, where
one exists, overrides TOML at startup so containers can keep the
toml baked into the image and tune per host without a rebuild.

## Where each knob lives

| Category | Today | Planned home |
|---|---|---|
| Discord / OpenRouter / TTS / STT credentials | env | env (unchanged) |
| Familiar id | env (`FAMILIAR_ID`) or `--familiar` | env (unchanged) |
| LLM model + temperature per slot | `[llm.<slot>]` | unchanged |
| TTS provider + voice | `[tts]` | unchanged |
| History window + prompt layer order | `[providers.history]`, `[channels.<id>]` | unchanged |
| Deepgram STT thresholds & key-terms | `[providers.stt.deepgram]` (env override per knob) | unchanged |
| Parakeet local STT (V3 phase 2) | `[providers.stt.parakeet]` (env override per knob) | unchanged |
| FasterWhisper local STT (V3 phase 3) | `[providers.stt.faster_whisper]` (env override per knob) | unchanged |
| STT backend selector | `[providers.stt].backend` (`STT_BACKEND` env override) | unchanged |
| Turn detection strategy | `[providers.turn_detection]` | unchanged |
| Memory projector selection | hard-wired | `[providers.memory]` (M5) |
| Voice pipeline mode | cascaded + sentence streaming | `[providers.voice_pipeline]` (V5 only — sentence streaming shipped) |
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

[providers.turn_detection]
strategy = "deepgram"    # "deepgram" | "ten+smart_turn"

[providers.stt]
backend = "deepgram"     # V3 widens to "faster_whisper" | "parakeet"

[providers.stt.deepgram]
model           = "nova-3"
language        = "en"
endpointing_ms  = 500
keyterms        = []     # see § STT — Deepgram for the full set

[tts]
provider          = "azure"      # "azure" | "cartesia" | "gemini"
azure_voice       = "en-US-AmberNeural"
cartesia_voice_id = "..."
cartesia_model    = "sonic-3"
gemini_voice      = "Kore"
gemini_model      = "gemini-3.1-flash-tts-preview"
greetings         = []

[llm.main_prose]
model                    = "z-ai/glm-5.1"
temperature              = 0.7
provider_order           = ["z-ai"]   # optional — pin OpenRouter routing
provider_allow_fallbacks = true       # optional — default true

[channels.123456789012345678]
history_window_size = 30
prompt_layers       = ["core", "card", "operating_mode", "recent_history"]
message_rendering   = "prefixed"  # "prefixed" | "name_only"
```

## Tuning by goal

### Lower voice latency

1. **`[llm.<slot>].provider_order`** — pin OpenRouter to a single
   stable provider so prompt caching survives across turns (see
   [provider pinning](#provider-pinning) below).
2. **`[providers.stt.deepgram].endpointing_ms`** (default `500`).
   Drop to `300` for snappier finals; raise to `700` if it cuts
   mid-sentence. Right long-term answer: a semantic turn classifier
   (V1).
3. **`[providers.stt.deepgram].utterance_end_ms`** (default `1500`).
   Speech-end grace. Lower = faster handoff, more risk of
   mid-sentence cuts.
4. **`[channels.<id>].history_window_size`**. Lower on busy
   channels — smaller prompt, lower LLM TTFT.
5. **`[tts].provider = "cartesia"`**. Lowest hosted
   time-to-first-audio of the three.

Sentence streaming (formerly V2) shipped — TTS first audio fires
on the first sentence boundary, not after the LLM finishes.
Remaining big win is V1 (local VAD + Smart Turn). Today's knobs
above are within a few hundred milliseconds of each other.

#### Provider pinning

OpenRouter load-balances each call across whichever providers are
available. Diagnostics in production showed ten different providers
across sixteen calls within a few minutes — each one a cold prompt
cache, so input tokens stayed at `cached=0` even with identical
system prompts turn-over-turn.

Pin a provider in `[llm.<slot>]`:

```toml
[llm.main_prose]
model                    = "z-ai/glm-5.1"
provider_order           = ["z-ai"]      # first-party — best caching
provider_allow_fallbacks = true          # default: fall back if pinned is down
```

For GLM family models, `z-ai` is the first-party provider and the
most reliable caching path. For other model families, check the
upstream's OpenRouter listing for the canonical provider; second
choice is usually `deepinfra` or `together` (large stable infra).

`provider_allow_fallbacks=true` (the default) lets OpenRouter route
elsewhere when the pinned provider is unavailable — necessary so a
flaky provider doesn't black out the bot. Set it to `false` only
when you want hard failures rather than a cache-cold call.

**This is a stopgap.** OpenRouter's default routing improves
periodically and the `provider_order` line should be revisited
whenever you change models. The `[LLM call]` log line shows
`provider=...` and `cached=...` per call — use them to verify the
pin is working and to decide when it's no longer needed.

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

`backend = "deepgram"` is the only implementation today; V3 phases
2/3 add Parakeet and FasterWhisper. Selector + env override are
already wired:

- `[providers.stt].backend` in `character.toml` picks the backend.
- `STT_BACKEND` env var overrides TOML at process start (mirrors
  `LOCAL_TURN_DETECTION` — container-friendly).
- Unknown backend → `ValueError`, caught in `commands/run.py` and
  logged as "Transcriber unavailable"; bot still starts, voice path
  degrades to no-op.

Backend-specific knobs live in `[providers.stt.<backend>]`. For
Deepgram, defaults bias toward fewer mid-sentence cuts during
thinking pauses; lower the silence thresholds for snappier finals.

```toml
[providers.stt]
backend = "deepgram"            # V3: | "faster_whisper" | "parakeet"

[providers.stt.deepgram]
model                   = "nova-3"
language                = "en"
endpointing_ms          = 500
utterance_end_ms        = 1500
smart_format            = true
punctuate               = true
keyterms                = ["lifecycle mesh", "Tam"]
replay_buffer_s         = 5.0
keepalive_interval_s    = 3.0
reconnect_max_attempts  = 5
reconnect_backoff_cap_s = 16.0
idle_close_s            = 30.0
```

| Field | Default | Purpose |
|---|---|---|
| `model` | `nova-3` | Model name. |
| `language` | `en` | Language code. |
| `endpointing_ms` | `500` | Silence ms before a segment finalizes. |
| `utterance_end_ms` | `1500` | Speech-end grace window. |
| `smart_format` | `true` | Punctuation, number/date/unit normalization. |
| `punctuate` | `true` | Explicit punctuation pass. |
| `keyterms` | `[]` | List of jargon / proper nouns to bias nova-3 toward. |
| `replay_buffer_s` | `5.0` | Seconds replayed after WebSocket reconnect. |
| `keepalive_interval_s` | `3.0` | Keepalive ping cadence. |
| `reconnect_max_attempts` | `5` | Reconnect attempts before giving up. |
| `reconnect_backoff_cap_s` | `16.0` | Reconnect backoff cap. |
| `idle_close_s` | `30.0` | Per-user stream closed after this many silent seconds; reopened on next chunk. `0` disables. |

### Env overrides

Each TOML field above has a matching `DEEPGRAM_*` env var that wins
at startup — same precedence model as `LOCAL_TURN_DETECTION`. Useful
when a baked-in container image needs per-host knobs without a
rebuild. `DEEPGRAM_API_KEY` (the secret) only lives in env.

| Env var | Overrides |
|---|---|
| `DEEPGRAM_MODEL` | `model` |
| `DEEPGRAM_LANGUAGE` | `language` |
| `DEEPGRAM_ENDPOINTING_MS` | `endpointing_ms` |
| `DEEPGRAM_UTTERANCE_END_MS` | `utterance_end_ms` |
| `DEEPGRAM_SMART_FORMAT` | `smart_format` (`0/1`) |
| `DEEPGRAM_PUNCTUATE` | `punctuate` (`0/1`) |
| `DEEPGRAM_KEYTERMS` | `keyterms` (comma-separated; empty string clears) |
| `DEEPGRAM_REPLAY_BUFFER_S` | `replay_buffer_s` |
| `DEEPGRAM_KEEPALIVE_INTERVAL_S` | `keepalive_interval_s` |
| `DEEPGRAM_RECONNECT_MAX_ATTEMPTS` | `reconnect_max_attempts` |
| `DEEPGRAM_RECONNECT_BACKOFF_CAP_S` | `reconnect_backoff_cap_s` |
| `DEEPGRAM_IDLE_CLOSE_S` | `idle_close_s` |

## STT — Parakeet (V3 phase 2)

Local NeMo Parakeet-TDT 0.6B v3 backend (Apache 2.0 toolkit, CC-BY-4.0
weights). No API key — model loads on first turn (~600 MB; cached in
the HuggingFace cache thereafter). Buffer-and-finalize semantics:
audio accumulates per user, the local turn detector fires
`finalize()` on turn-complete, NeMo runs once and emits one final
result.

**Requirements:**

- `uv sync --extra local-turn --extra local-stt-parakeet` — pulls
  TEN-VAD, Smart Turn, NeMo, torch.
- `[providers.turn_detection].strategy = "ten+smart_turn"` (or env
  `LOCAL_TURN_DETECTION=1`). Without a local turn detector nothing
  drives `finalize()`, so transcripts never surface.

```toml
[providers.stt]
backend = "parakeet"

[providers.stt.parakeet]
model_name   = "nvidia/parakeet-tdt-0.6b-v3"
device       = "auto"     # "auto" | "cuda" | "cpu"
idle_close_s = 30.0
```

| Field | Default | Purpose |
|---|---|---|
| `model_name` | `nvidia/parakeet-tdt-0.6b-v3` | HuggingFace ID passed to `nemo.collections.asr.models.ASRModel.from_pretrained`. |
| `device` | `auto` | `auto` defers to NeMo (CUDA if available, else CPU); `cuda` / `cpu` force. |
| `idle_close_s` | `30.0` | Per-user buffer reset after silence; matches Deepgram parity. |

| Env var | Overrides |
|---|---|
| `PARAKEET_MODEL_NAME` | `model_name` |
| `PARAKEET_DEVICE` | `device` |
| `PARAKEET_IDLE_CLOSE_S` | `idle_close_s` |

## STT — FasterWhisper (V3 phase 3)

Local CTranslate2-backed Whisper. Lighter than Parakeet — no torch,
~150 MB for the `small` model. Same buffer-and-finalize shape:
audio accumulates per user, the local turn detector fires
`finalize()` on turn-complete, Whisper runs once and emits one
final result.

**Requirements:**

- `uv sync --extra local-turn --extra local-stt-whisper` — pulls
  TEN-VAD, Smart Turn, faster-whisper.
- `[providers.turn_detection].strategy = "ten+smart_turn"` (or env
  `LOCAL_TURN_DETECTION=1`). Without a local turn detector nothing
  drives `finalize()`.

```toml
[providers.stt]
backend = "faster_whisper"

[providers.stt.faster_whisper]
model_size   = "small"          # "tiny" | "base" | "small" | "medium" | "large-v3"
device       = "auto"           # "auto" | "cuda" | "cpu"
compute_type = "auto"           # "auto" | "int8" | "float16" | "float32"
language     = "en"
idle_close_s = 30.0
```

| Field | Default | Purpose |
|---|---|---|
| `model_size` | `small` | CT2 model size; tradeoffs latency vs. accuracy. |
| `device` | `auto` | `auto` defers to faster-whisper; `cuda` / `cpu` force. |
| `compute_type` | `auto` | Quantisation; `int8` is the CPU sweet spot. |
| `language` | `en` | Pinned avoids per-turn language detection latency. |
| `idle_close_s` | `30.0` | Per-user buffer reset after silence. |

| Env var | Overrides |
|---|---|
| `FASTER_WHISPER_MODEL_SIZE` | `model_size` |
| `FASTER_WHISPER_DEVICE` | `device` |
| `FASTER_WHISPER_COMPUTE_TYPE` | `compute_type` |
| `FASTER_WHISPER_LANGUAGE` | `language` |
| `FASTER_WHISPER_IDLE_CLOSE_S` | `idle_close_s` |

## Local turn detection (V1)

V1 fork of the audio path: TEN-VAD + Smart Turn v3 own endpointing
locally, Deepgram becomes pure STT. Saves 150–200 ms vs. remote
endpointing. Requires the `local-turn` extra (`uv sync --extra
local-turn`) and a Smart Turn ONNX model file at
`data/models/smart-turn-v3.onnx`.

When active, per-user Deepgram clones are spawned with
`endpointing_ms=10` so they wait on `Finalize` from the local chain
rather than firing on their own silence timer.

### Enabling via TOML (recommended)

```toml
[providers.turn_detection]
strategy = "ten+smart_turn"   # "deepgram" (default) | "ten+smart_turn"
```

### Enabling via env var (legacy / container override)

`LOCAL_TURN_DETECTION=1` overrides the TOML setting — useful in
container deployments where `character.toml` is baked into the image
but the feature flag lives in the environment.

### Knobs (env)

| Var | Default | Purpose |
|---|---|---|
| `LOCAL_TURN_DETECTION` | _(unset)_ | Env override: `1/true/yes/on` enables the local chain. |
| `SMART_TURN_MODEL_PATH` | `data/models/smart-turn-v3.onnx` | Pipecat Smart Turn v3 ONNX file. |
| `LOCAL_TURN_SILENCE_MS` | `200` | Silence after speech before SmartTurn classifies. |
| `LOCAL_TURN_SPEECH_START_MS` | `100` | Consecutive speech before "speaking" latches. |
| `LOCAL_TURN_VAD_THRESHOLD` | `0.5` | TEN-VAD `is_speech` cutoff. |
| `LOCAL_TURN_SMART_TURN_THRESHOLD` | `0.5` | SmartTurn `is_complete` cutoff. |
| `TEN_VAD_HOP_SIZE` | `256` | TEN-VAD frame size in samples at 16 kHz; `256` (16 ms) or `160` (10 ms). |

A missing Smart Turn ONNX file turns the feature off with a warning
— the bot falls back to Deepgram endpointing rather than failing to
start.

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
# shipped
[providers.turn_detection]
strategy = "deepgram"            # | "ten+smart_turn"

[providers.stt]
backend = "deepgram"             # | "parakeet" | "faster_whisper" (V3 closed)
# env: STT_BACKEND overrides this at startup (mirrors LOCAL_TURN_DETECTION)

[providers.stt.parakeet]         # V3 phase 2 (shipped)
model_name   = "nvidia/parakeet-tdt-0.6b-v3"
device       = "auto"
idle_close_s = 30.0

[providers.stt.faster_whisper]   # V3 phase 3 (shipped)
model_size   = "small"
device       = "auto"
compute_type = "auto"
language     = "en"
idle_close_s = 30.0

# planned (V5, M5)

[providers.memory]
projectors = ["rich_note", "people_dossier"]                        # (M5)

[providers.voice_pipeline]
mode = "cascaded"                # | "s2s" (V5)

[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.4
importance_weight = 0.6          # M2
embedding_weight  = 0.0          # M6
```
