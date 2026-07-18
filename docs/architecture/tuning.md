# Tuning

Single-point reference for every operator-tunable knob. If a value
isn't here, open an issue or add it.

The split:

- **Secrets and install selectors live in env.** Tokens, API keys,
  active familiar id. Never in git.
- **Behaviour and tuning knobs live in `character.toml`.**
  Per-familiar, deep-merged over `_default/character.toml`.
  `[channels.<id>]` overrides per Discord channel.

Non-secret knobs live in TOML (`character.toml` fields and their
`[channels.<id>]` overrides). Where an env var exists, it overrides
TOML at startup so containers can bake the toml into the image and
tune per host without a rebuild.

## Where each knob lives

| Category | Today | Planned home |
|---|---|---|
| Discord / OpenRouter / TTS / STT credentials | env | env (unchanged) |
| Familiar id | env (`FAMILIAR_ID`) or `--familiar` | env (unchanged) |
| LLM model + temperature per slot | `[llm.<slot>]` | unchanged |
| TTS provider + voice | `[tts]` | unchanged |
| Per-tier prompt token budget | `[budget.<tier>]` | unchanged |
| History turn safety cap + prompt layer order | `[providers.history]`, `[channels.<id>]` | unchanged |
| Deepgram STT thresholds & key-terms | `[providers.stt.deepgram]` | unchanged |
| Parakeet local STT (V3 phase 2) | `[providers.stt.parakeet]` | unchanged |
| FasterWhisper local STT (V3 phase 3) | `[providers.stt.faster_whisper]` | unchanged |
| STT backend selector | `[providers.stt].backend` | unchanged |
| Turn detection strategy + tuning | `[providers.turn_detection]` + `[providers.turn_detection.local]` | unchanged |
| Memory projector selection | `[providers.memory]` (M5) | unchanged |
| Memory worker cadences / batch sizes | `[providers.memory.<name>]` | unchanged |
| Voice pipeline mode | cascaded + sentence streaming | `[providers.voice_pipeline]` (V5 only — sentence streaming shipped) |
| Embedding backend (M6) | `[providers.embedding]` | unchanged |
| RAG / fact retrieval ranking | `[memory.retrieval]` (M2 + M6) | unchanged |
| Attentional unread-nudge controls | `[focus]` | unchanged |
| Agentic tool-loop cap | `[tools]` | unchanged |
| LLM request concurrency | `[llm].max_concurrent_requests` | unchanged |
| Activities catalog + cadence | `data/familiars/<id>/activities.toml` | unchanged |

## Environment variables

Set in `.env` or the host environment. Never log them.

### Required

| Var | Purpose |
|---|---|
| `DISCORD_BOT` | Discord bot token. |
| `OPENROUTER_API_KEY` | Shared across every LLM call site. |
| `DEEPGRAM_API_KEY` | STT credential. |
| `FAMILIAR_ID` | Character folder under the familiars root. Overridable by `--familiar`. |

### TTS provider credentials (one set, depending on `[tts].provider`)

| Var | Provider |
|---|---|
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | Azure (default). |
| `CARTESIA_API_KEY` | Cartesia. |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini. |

### Optional path overrides

| Var | Purpose |
|---|---|
| `FAMILIARS_ROOT` | Override the per-user familiars root (default: platform data dir; issue #201). |
| `FAMILIAR_DEFAULTS_ROOT` | Override where the tracked `_default` skeleton resolves (default: `data/familiars`). |

See [On-disk layout](../getting-started/on-disk-layout.md#where-the-familiars-root-lives).

## Character TOML — current schema

Source of truth: `data/familiars/_default/character.toml`. Every
field overridable per familiar; `[channels.<id>]` overrides per
channel.

```toml
display_tz = "UTC"
aliases    = []

[providers.history]
voice_window_size = 100   # safety net — per-section token caps trim first
text_window_size  = 200

# Per-section caps (each enforced independently; no combined cap).
# The whole-prompt total is the sum of these — see § Prompt assembly budget.
[budget.voice]
recent_history_tokens = 3000
rag_tokens            = 900   # ...plus the other per-section caps
[budget.text]
recent_history_tokens = 8000
rag_tokens            = 2400
[budget.background]
recent_history_tokens = 24000
rag_tokens            = 8000

[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.0
importance_weight = 0.6   # M2 — see § Retrieval ranking
embedding_weight  = 0.0   # M6 — needs [providers.embedding] + projector

[providers.embedding]
backend = "off"           # "off" | "hash"
dim     = 256             # for backends that accept one

[providers.memory]        # M5 — see § Memory projectors
projectors = [
    "rolling_summary", "rich_note", "people_dossier",
    "reflection", "fact_supersede",
]

[providers.memory.rich_note]   # per-worker tuning — one table per projector
batch_size      = 10
tick_interval_s = 15.0

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

[llm]
max_concurrent_requests = 4  # process-wide cap on in-flight LLM requests

[llm.fast]
model        = "anthropic/claude-haiku-4.5"
temperature  = 0.7
reasoning    = "off"        # "off" | "low" | "medium" | "high" | "default" | omit
                            # "default" = model default; overrides a level merged
                            # in from _default/character.toml (TOML has no null)
tool_calling = false

[llm.prose]
model                    = "z-ai/glm-5.2"
temperature              = 0.7
provider_order           = ["z-ai"]   # optional — pin OpenRouter routing
provider_allow_fallbacks = true       # optional — default true
reasoning                = "medium"
tool_calling             = false

[llm.background]
model          = "z-ai/glm-5.2"
temperature    = 0.7
provider_order = ["z-ai"]
reasoning      = "medium"
tool_calling   = true

[channels.123456789012345678]
history_window_size = 30
prompt_layers       = ["core", "card", "operating_mode", "recent_history"]
message_rendering   = "prefixed"  # "prefixed" | "name_only"

[discord.text]
respond_to_typing        = true   # cancel in-flight reply on user typing
typing_backoff_initial_s = 1.0    # first pause when another bot is typing
typing_backoff_max_s     = 30.0   # ceiling after exponential doubling

[focus]
unread_nudge_enabled   = true     # nudge the model when unreads arrive
nudge_debounce_seconds = 30.0     # rapid arrivals share one nudge

[tools]
loop_max_iterations = 5           # hard cap on agentic-loop rounds per turn
```

## Tuning by goal

### Lower voice latency

1. **`[llm.<slot>].provider_order`** — pin OpenRouter to one stable
   provider so prompt caching survives across turns (see
   [provider pinning](#provider-pinning) below).
2. **`[providers.stt.deepgram].endpointing_ms`** (default `500`).
   Drop to `300` for snappier finals; raise to `700` if it cuts
   mid-sentence. Long-term fix: a semantic turn classifier (V1).
3. **`[providers.stt.deepgram].utterance_end_ms`** (default `1500`).
   Speech-end grace. Lower = faster handoff, more risk of
   mid-sentence cuts.
4. **`[channels.<id>].history_window_size`**. Lower on busy
   channels — smaller prompt, lower LLM TTFT.
5. **`[tts].provider = "cartesia"`**. Lowest hosted
   time-to-first-audio of the three.

Sentence streaming (formerly V2) shipped — TTS first audio fires
on the first sentence boundary, not after the LLM finishes. The
remaining big win is V1 (local VAD + Smart Turn); knobs above sit
within a few hundred milliseconds of each other.

#### Provider pinning

OpenRouter load-balances each call across available providers.
Production diagnostics showed ten different providers across
sixteen calls within minutes — each a cold prompt cache, so input
tokens stayed at `cached=0` despite identical system prompts
turn-over-turn.

Pin a provider in `[llm.<slot>]`:

```toml
[llm.prose]
model                    = "z-ai/glm-5.2"
provider_order           = ["z-ai"]      # first-party — best caching
provider_allow_fallbacks = true          # default: fall back if pinned is down
```

For GLM models, `z-ai` is the first-party provider and most
reliable caching path. For other model families, check the upstream
OpenRouter listing for the canonical provider; second choice is
usually `deepinfra` or `together` (large stable infra).

`provider_allow_fallbacks=true` (default) lets OpenRouter route
elsewhere when the pinned provider is unavailable, so a flaky
provider doesn't black out the bot. Set `false` only when hard
failures beat cache-cold calls.

**This is a stopgap.** OpenRouter's default routing improves
periodically; revisit `provider_order` whenever you change models.
The `[LLM call]` log line shows `provider=...` and `cached=...` per
call — use them to verify the pin works and decide when it's no
longer needed.

### Better turn handling in busy channels

- **`[channels.<id>].history_window_size`** — bump to 30–40 for
  more "is this turn for me?" context.
- **`[channels.<id>].message_rendering = "prefixed"`** — keeps the
  `[HH:MM Display Name]` prefix; timestamp rhythm helps the model
  judge multi-party flow.
- **`<silent>` sentinel** — already wired (see
  [multi-party addressivity](context-pipeline.md#multi-party-addressivity)).
  Don't override the sentinel instruction in the character prompt.
  Under tool calling the `silent(reasoning)` tool is the equivalent
  agentic-path gate.

### Attentional focus

The familiar attends to one text + one voice channel
at a time; unfocused channels' messages are **staged** (stored, no
reply) until the model shifts focus. Focus is model-driven through
the `shift_focus(channel_id)` tool, so it only moves on slots with
[`tool_calling = true`](#tool_calling) — otherwise focus stays on its
startup default. On startup focus defaults to the first text and
first voice subscription; thereafter it persists in the
`focus_pointers` table across restarts. The
`read_channel(limit?, before_id?, around_id?)` tool lets the familiar
peek at the focused text channel without consuming staged turns,
paging back or jumping to a turn id. Inspect current focus + per-channel unread
counts via `/diagnostics` (`Focus: text=#… voice=#…`,
`Unreads: #… (N)`). See
[Attentional stream](context-pipeline.md#attentional-stream).

Unread-nudge and catch-up controls live in `[focus]`:

```toml
[focus]
unread_nudge_enabled   = true
nudge_debounce_seconds = 30.0
catch_up_limit         = 20
```

| Field | Default | Purpose |
|---|---|---|
| `unread_nudge_enabled` | `true` | When true, an unfocused-channel arrival nudges the model so unreads surface promptly. The nudge never moves focus — only the model's `shift_focus` does. Set false to disable. |
| `nudge_debounce_seconds` | `30.0` | Rapid arrivals within this window share one nudge; the next unread after the window fires again. |
| `catch_up_limit` | `20` | How many staged turns she actually catches up on when attention lands on a channel — the `shift_focus` preview size and the per-channel cap on activity-return promotion. Staged backlog beyond this is **missed** (dropped from her window and rolling summary), not silently consumed. Direct @-mentions/replies-to-her are always caught regardless of age. |

Set `unread_nudge_enabled = false` to keep attention pinned to the
focused channel and never nudge for backgrounded traffic; when enabled,
`nudge_debounce_seconds` is the sole throttle.

`catch_up_limit` makes perception match consumption: she only takes in
the last *N* staged turns she actually previews on a focus swap (or per
channel on activity return), so a long backlog she was away for is
genuinely **missed** rather than folded into her summary as if read.
Raise it so she catches up on more; lower it so she misses more of what
piled up while away.

## Discord text channel knobs

`[discord.text]` controls how the bot reacts to Discord's typing
events while a reply is mid-flight.

```toml
[discord.text]
respond_to_typing        = true
typing_backoff_initial_s = 1.0
typing_backoff_max_s     = 30.0
```

- **`respond_to_typing`** — when `true` (default), a typing-start
  event from another user in a subscribed channel cancels the active
  `TurnScope` so the bot stops streaming instead of talking over
  someone. Same path as voice barge-in. Set `false` to ignore typing
  entirely; the bot still finishes replies mid-message.
- **`typing_backoff_initial_s`** / **`typing_backoff_max_s`** —
  exponential backoff envelope when *another bot* (e.g. another
  familiar-connect instance) is typing in the same channel. Each bot
  typing event installs a `now + window` deadline; the responder
  waits past it before generating, then doubles the next window up
  to the cap. A real user message resets the ladder. Prevents
  pingpong: two bots mirroring each other's typing indicators would
  otherwise reply in lockstep forever.

While generating, the bot surfaces Discord's "Bot is typing…"
indicator (via `BotHandle.trigger_typing`) so users see the in-flight
signal — including on regenerated replies after a barge-in cancel.
The indicator opens lazily, only after `SilentDetector` rules out the
`<silent>` sentinel, so reasoning resolving to silence never flickers
it on. Stops cleanly when the streaming context exits.

## Activities

Sidecar `data/familiars/<id>/activities.toml`, not `character.toml`.
Missing file or empty catalog = feature off (zero behavior change);
invalid content fails loudly at startup. Full lifecycle and catalog
entry schema: [Activities](activities.md).

```toml
archive_after_minutes = 45
idle_nudge_minutes    = 20
min_gap_minutes       = 90
active_hours          = "10:00-23:00"
```

| Knob | Default | Purpose |
|---|---|---|
| `archive_after_minutes` | `45` | Absence at/above this sets the per-channel archive watermark at the departure turn — prompt window resets there; `read_channel` scrollback doesn't. |
| `idle_nudge_minutes` | `20` | Focused-channel quiet time before an idle nudge may offer `start_activity`; also the nudge debounce window. |
| `min_gap_minutes` | `90` | Minimum gap after a return before the next nudge. |
| `active_hours` | unset (always) | `"HH:MM-HH:MM"` in `display_tz`; may wrap midnight. Nudges only fire inside this window. |

Per-activity behavior (duration range, reachability while out,
experience seed) lives on the `[[catalog]]` entries — see the
[catalog entry schema](activities.md#configuration).

### Better long-term memory

- **`[budget.<tier>].recent_history_tokens` and the other per-section
  caps** — primary knobs. Each section (recent history, RAG, dossiers,
  summary, cross-channel, reflections, lorebook) has its own
  independent cap; lift the ones you want to give the model more room.
  There is no combined cap — the prompt total is just their sum. See
  [Prompt assembly budget](#prompt-assembly-budget).
- **`[providers.history].voice_window_size` / `.text_window_size`** —
  hard upper bound on history turns per tier. Safety net: the
  per-section token caps usually bite first. Lower only to force a
  tighter absolute cap on prompt size.
- **`SummaryWorker.turns_threshold`** (default `10`). New turns
  before the rolling summary regenerates. Constructor arg in
  `commands/run.rs`; planned move to TOML.
- **`[budget.<tier>].max_dossier_people`** — was
  `PeopleDossierLayer.max_people`. Hard cap on dossier rows per
  prompt; combined with `dossier_tokens` so count or byte size
  (whichever bites first) drops trailing rows.
- **`[memory.retrieval].importance_weight`** — bias retrieval
  toward safety-critical facts (allergies, names, life events).
  See [Retrieval ranking](#retrieval-ranking-m2).
- **`data/familiars/<id>/lorebook.toml`** — keyword-activated
  authored canon (M4). Hand-written world / setting / lore entries
  surfaced only when a key appears in recent turns. See
  [Memory strategies — lorebook](memory-strategies.md#lorebook-m4).
- **`[providers.embedding].backend`** + `embedding_weight` —
  semantic recall (M6, opt-in seam). Pick a backend, add
  `"fact_embedding"` to `[providers.memory].projectors`, and raise
  `[memory.retrieval].embedding_weight` once the side-index
  populates. See [Embeddings (M6)](#embeddings-m6).

### A/B a strategy on one channel

`[channels.<id>].prompt_layers` overrides default layer order on
one channel. Compare candidate vs control side by side. Once A1
lands, the same per-channel mechanism extends to STT, turn
detection, and voice pipeline mode.

## STT — Deepgram

`backend = "deepgram"` is the default. Selector lives in
`[providers.stt].backend`; an unknown value (or a local backend whose
engine isn't wired) → `SttError`, caught in `commands/run.rs` and
logged as "Transcriber unavailable" — bot still starts, voice path
degrades to no-op. `DEEPGRAM_API_KEY` is the only env input.

Defaults bias toward fewer mid-sentence cuts during thinking pauses;
lower silence thresholds for snappier finals.

```toml
[providers.stt]
backend = "deepgram"            # | "parakeet" | "faster_whisper"

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
| `keyterms` | `[]` | List of jargon / proper nouns to bias nova-3 toward. Voice-channel member proper nouns (display names, usernames, aliases, nicknames) are auto-appended per speaker at connect time on top of this list, then deduped and capped — so this field is only for jargon the members' names don't already cover. |
| `replay_buffer_s` | `5.0` | Seconds replayed after WebSocket reconnect. |
| `keepalive_interval_s` | `3.0` | Keepalive ping cadence. |
| `reconnect_max_attempts` | `5` | Reconnect attempts before giving up. |
| `reconnect_backoff_cap_s` | `16.0` | Reconnect backoff cap. |
| `idle_close_s` | `30.0` | Per-user stream closed after this many silent seconds; reopened on next chunk. `0` disables. |

## STT — Parakeet (V3 phase 2)

Local NeMo Parakeet-TDT 0.6B v3 backend (Apache 2.0 toolkit,
CC-BY-4.0 weights). No API key — model loads on first turn
(~600 MB; cached in the HuggingFace cache thereafter).
Buffer-and-finalize: audio accumulates per user, the local turn
detector fires `finalize()` on turn-complete, NeMo runs once and
emits one final result.

**Requirements:**

- `cargo build --features local-turn,local-stt` — pulls
  TEN-VAD, Smart Turn, NeMo, torch.
- `[providers.turn_detection].strategy = "ten+smart_turn"`. Without
  a local turn detector nothing drives `finalize()`, so transcripts
  never surface.

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

## STT — FasterWhisper (V3 phase 3)

Local CTranslate2-backed Whisper. Lighter than Parakeet — no torch,
~150 MB for the `small` model. Same buffer-and-finalize shape:
audio accumulates per user, the local turn detector fires
`finalize()` on turn-complete, Whisper runs once, emits one final
result.

**Requirements:**

- `cargo build --features local-turn,local-stt` — pulls
  TEN-VAD, Smart Turn, faster-whisper.
- `[providers.turn_detection].strategy = "ten+smart_turn"`. Without
  a local turn detector nothing drives `finalize()`.

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

## Local turn detection (V1)

V1 fork of the audio path: TEN-VAD + Smart Turn v3 own endpointing
locally, Deepgram becomes pure STT. Saves 150–200 ms vs. remote
endpointing. Also required when the STT backend is local (Parakeet
or FasterWhisper) since neither has an internal endpointer.

Requires the `local-turn` feature (`cargo build --features local-turn`).
Smart Turn ONNX weights pull from HuggingFace on first use (cached
under `~/.cache/huggingface`); subsequent runs are filesystem-only.
`HF_HUB_OFFLINE=1` forces cache-only mode for air-gapped
deployments.

Default `smart_turn_filename` is the **CPU ONNX export**, matching
the CPU ONNX Runtime (`ort`) the `local-turn` feature pulls. To run
the GPU export, build `ort` with a GPU execution provider:

```toml
[providers.turn_detection.local]
smart_turn_filename = "smart-turn-v3.2-gpu.onnx"
```

When active, per-user Deepgram clones spawn with
`endpointing_ms=10` so they wait on `Finalize` from the local chain
rather than firing on their own silence timer.

```toml
[providers.turn_detection]
strategy = "ten+smart_turn"   # "deepgram" (default) | "ten+smart_turn"

[providers.turn_detection.local]
smart_turn_repo_id    = "pipecat-ai/smart-turn-v3"
smart_turn_filename   = "smart-turn-v3.2-cpu.onnx"
silence_ms            = 200
speech_start_ms       = 100
vad_threshold         = 0.5
smart_turn_threshold  = 0.5
vad_hop_size          = 256
idle_fallback_s       = 1.5
```

| Field | Default | Purpose |
|---|---|---|
| `smart_turn_repo_id` | `pipecat-ai/smart-turn-v3` | HuggingFace repo holding the ONNX exports. |
| `smart_turn_filename` | `smart-turn-v3.2-cpu.onnx` | Specific export. Switch to `smart-turn-v3.2-gpu.onnx` when `ort` is built with a GPU execution provider. |
| `silence_ms` | `200` | Silence after speech before SmartTurn classifies. |
| `speech_start_ms` | `100` | Consecutive speech before "speaking" latches. |
| `vad_threshold` | `0.5` | TEN-VAD `is_speech` cutoff. |
| `smart_turn_threshold` | `0.5` | SmartTurn `is_complete` cutoff. |
| `vad_hop_size` | `256` | TEN-VAD frame size in samples at 16 kHz; `256` (16 ms) or `160` (10 ms). |
| `idle_fallback_s` | `1.5` | Idle gap (no audio) before the pump force-completes a turn stranded in the endpointer. Longer than the plain-Deepgram idle-finalize (0.5 s) so a natural pause doesn't defeat Smart Turn's hold-through-pause. Tune down for snappier recovery from misfires, up if slow speakers get cut off. |

A missing Smart Turn ONNX file disables the feature with a warning
— the bot falls back to Deepgram endpointing rather than failing to
start.

## TTS

Already TOML-driven. `[tts]` selects provider + per-provider voice
/ model. Provider-specific keys read only when that provider is
selected.

| Provider | Voice field | Model field | Extras |
|---|---|---|---|
| `azure` (default) | `azure_voice` | (built-in) | — |
| `cartesia` | `cartesia_voice_id` | `cartesia_model` | — |
| `gemini` | `gemini_voice` | `gemini_model` | `gemini_style`, `gemini_scene`, `gemini_pace`, `gemini_accent`, `gemini_context`, `gemini_audio_profile` |

`greetings = ["..."]` pre-synthesises greeting audio at startup so
first speech doesn't pay TTS cold-start.

## LLM slots

Three tiered slots, by latency / quality:

| Slot | Call sites | Defaults |
|---|---|---|
| `fast` | voice replies (`VoiceResponder`) | low-latency model, reasoning off, tools off |
| `prose` | text-channel replies (`TextResponder`) | quality model, reasoning on, tools off |
| `background` | summaries, fact extraction, dossiers (`SummaryWorker`, `FactExtractor`, `PeopleDossierWorker`) | quality model, reasoning on, tools on |

Each slot picks its model independently. Slot names are canonical;
unknown slots fail loudly at config load. See
`familiar_connect::config::LLM_SLOT_NAMES`.

### Schema

```toml
[llm]
image_description_model  = ""              # shared; empty = disabled
max_concurrent_requests  = 4               # shared; process-wide cap

[llm.<slot>]
model                    = "z-ai/glm-5.2"   # required
temperature              = 0.7              # optional, [0, 2]
top_p                    = 0.95             # optional, [0, 1]
top_k                    = 20               # optional, positive int
presence_penalty         = 1.5              # optional, [-2, 2]
provider_order           = ["z-ai"]         # optional, OpenRouter pin
provider_allow_fallbacks = true             # optional, default true
reasoning                = "medium"         # "off"|"none"|"low"|"medium"|"high"|"default"|omit
think_prepend            = false            # optional, default false
tool_calling             = false            # optional, default false
image_tools              = false            # optional, default false
multimodal               = false            # optional, default false
```

### Sampling knobs (`top_p` / `top_k` / `presence_penalty`)

Optional pass-throughs to the OpenRouter payload; omitted = provider
default. Set them when a model card prescribes specific values — e.g.
Qwen3.6 requires `presence_penalty = 1.5` and mode-specific
temperature/top_p, because near-greedy decoding (low temperature,
default penalties) sends that family into endless repetition loops:
runaway multi-minute thinking turns in thinking mode, degenerate text
otherwise.

### `reasoning`

Maps to OpenRouter's `reasoning` parameter:

- `"off"` → `reasoning.exclude = true` (suppress thinking even on
  models that reason by default, like GLM 5.1).
- `"none"` → `reasoning.effort = "none"` (disable thinking generation
  entirely — the no-think mode for hybrid-reasoning models like
  Qwen3.6; pair with `think_prepend`).
- `"low"` / `"medium"` / `"high"` → `reasoning.effort = <level>`.
- `"default"` → no `reasoning` field; reclaims the model default over
  a level merged in from `_default/character.toml`.
- omitted → no `reasoning` field; defer to model default. Haiku 4.5
  never reasons regardless; GLM 5.1 reasons by default.

### `think_prepend`

Appends a fake closed think block (`<think>\n\n</think>`) as an
assistant prefill message on every request from this slot's client.
Qwen3.6 no-think stabiliser: with `reasoning = "none"` and no prefill,
the model leaks thinking as plain text. Useless on other models —
leave `false`.

### `tool_calling`

Runs the slot's agentic loop with the full tool registry:
`set_alarm` / `cancel_alarm`, `silent`, `shift_focus`, and (text
only) `read_channel` plus `start_activity` (the latter only when the
[activities catalog](activities.md#configuration) is non-empty).
With it `false` the registry never installs, so
the model can't shift focus or stay silent *via tools* — the
`<silent>` text sentinel still works on the bare streaming path, but
focus stays pinned to its startup default. Enable it on `prose` /
`fast` to make the attentional stream model-driven.

The loop's per-turn iteration cap (model call → tool execution →
re-call) is `[tools].loop_max_iterations` (default `5`, shared by
voice and text responders). Raise it for deeper multi-tool tasks at
the cost of latency and spend.

### `max_concurrent_requests`

Shared key at `[llm]` level. Sizes the process-wide semaphore that
caps in-flight LLM requests across every slot — the bottleneck is
the OpenRouter key's rate limit, not any single call site. Default
`4`. Lower it when hitting 429s; raise it when background workers
queue behind each other.

### `image_tools`

When `true`, registers the `view_image` tool in the text tool registry
for this slot. The agentic loop runs when either `tool_calling` or
`image_tools` is set. `view_image` is never registered in the voice
registry. Requires `[llm].image_description_model` for descriptions.
The describe prompt is neutral by default; append per-familiar persona
constraints with `[prompt].image_description_constraints` (see below).

### `[prompt].image_description_constraints`

Text appended to the neutral image-description base prompt. Per-familiar
persona tuning: a character not set in the present can ban naming
specific characters, people, franchises, or brands so it doesn't acquire
modern pop-culture knowledge that would break immersion. Empty (default)
= base prompt only. Bound into `view_image` at tool construction, so it
is static for the familiar's lifetime — not carried per turn.

### `multimodal`

When `true`, `ImageResult` tool-result messages include JPEG
`image_url` content blocks so vision-capable models can see the image.
When `false` (default), only the text description is sent.
Set this only for slots backed by vision-capable models.

## Prompt assembly budget

Each assembly layer self-truncates to its own per-section
`max_tokens` cap while the prompt is built — recent history, RAG,
dossiers, summary, cross-channel, reflections, lorebook. There is no
separate combined cap and no post-assembly trim step: the assembled
prompt is bounded by the **sum** of the per-section caps. That sum is
exposed in code as the derived `TierBudget.total_tokens` for
reporting only — nothing trims against it. Token estimates use a fast
`len(text) / 4` heuristic — no real tokenizer on the hot path;
sub-microsecond per message.

Every cap is a hard number. No "auto-fill from a total" — the source
of truth is `data/familiars/_default/character.toml`, which spells
out each value per tier. Per-familiar overrides deep-merge over
those defaults, so changing one knob leaves the rest in place.

```toml
[budget.voice]
recent_history_tokens = 3000   # cap on recent-history layer
rag_tokens            = 900
dossier_tokens        = 900
summary_tokens        = 600
reflection_tokens     = 600
lorebook_tokens       = 600
max_history_turns     = 200    # safety net behind recent_history_tokens
max_rag_turns         = 10
max_rag_facts         = 6
max_dossier_people    = 16
max_reflections       = 6
max_lorebook_entries  = 12

[budget.text]      # same shape, larger envelope
recent_history_tokens = 8000
# …

[budget.background]
recent_history_tokens = 24000
# …
```

| Tier | Derived total (sum of caps) | Sized for |
|---|---|---|
| `voice` | ~`7200` | Voice replies; tightest tier to protect first-token latency. |
| `text` | ~`19200` | Thoughtful replies; raise further for very long-context models. |
| `background` | ~`56000` | Offline summary / fact / dossier workers. |

Override one knob in your familiar's `character.toml`:

```toml
# data/familiars/aria/character.toml
[budget.voice]
recent_history_tokens = 4000   # rest of the voice envelope inherits from _default
```

### Per-model overrides

`[budget.model_curves."<model-name>"]` registers per-section float
multipliers for a model; all per-section `TierBudget` caps are valid
keys, unset fields default to `1.0`. `CharacterConfig.budget_for()`
applies the curve when the tier's active slot uses that model
(tier→slot: `voice→fast`, `text→prose`, `background→background`).
There is no `total_tokens` multiplier — the derived total scales
automatically when the per-section caps scale.

```toml
[budget.model_curves."claude-opus-4-7"]
recent_history_tokens = 2.5
rag_tokens            = 1.5
```

## History / context layers

| Knob | Default | Source |
|---|---|---|
| `RecentHistoryLayer.window_size` (voice tier) | `100` | `[providers.history].voice_window_size` |
| `RecentHistoryLayer.window_size` (text tier) | `200` | `[providers.history].text_window_size` |
| `RecentHistoryLayer.coalesce_max_gap_seconds` | `45.0` | `[providers.history].coalesce_max_gap_seconds` |
| `RecentHistoryLayer.silence_gap_fold_seconds` (text tier) | `0` (disabled) | `[providers.history].text_silence_gap_fold_seconds` |
| `RecentHistoryLayer.max_tokens` | `1500` (voice) | `[budget.<tier>].recent_history_tokens` |
| `RagContextLayer.max_results` | `5` (voice) | `[budget.<tier>].max_rag_turns` |
| `RagContextLayer.max_facts` | `3` (voice) | `[budget.<tier>].max_rag_facts` |
| `RagContextLayer.max_tokens` | `450` (voice) | `[budget.<tier>].rag_tokens` |
| `RagContextLayer.recent_window_size` | matches history window | constructor arg |
| `PeopleDossierLayer.max_people` | `8` (voice) | `[budget.<tier>].max_dossier_people` |
| `PeopleDossierLayer.max_tokens` | `450` (voice) | `[budget.<tier>].dossier_tokens` |
| `ConversationSummaryLayer.max_tokens` | `300` (voice) | `[budget.<tier>].summary_tokens` |
| `ReflectionLayer.max_reflections` | `3` (voice) | `[budget.<tier>].max_reflections` |
| `ReflectionLayer.max_tokens` | `300` (voice) | `[budget.<tier>].reflection_tokens` |
| `LorebookLayer.max_entries` | `6` (voice) | `[budget.<tier>].max_lorebook_entries` |
| `LorebookLayer.max_tokens` | `300` (voice) | `[budget.<tier>].lorebook_tokens` |
| `LorebookLayer.recent_window` | matches history window | constructor arg |
| `SummaryWorker.turns_threshold` | `10` | constructor arg |
| `SummaryWorker.tick_interval_s` | `5.0` | class default |
| `FactExtractor.batch_size` | `10` | constructor arg |
| `FactExtractor.tick_interval_s` | `15.0` | class default |
| `PeopleDossierWorker.tick_interval_s` | `20.0` | class default |
| `ReflectionWorker.turns_threshold` | `20` | constructor arg |
| `ReflectionWorker.tick_interval_s` | `60.0` | class default |

### Per-channel overrides

Today's `[channels.<id>]` knobs:

- `history_window_size` — overrides the global default.
- `prompt_layers` — explicit ordered list of layer names.
- `message_rendering` — `"prefixed"` or `"name_only"`.

## Retrieval ranking (M2)

```toml
[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.0
importance_weight = 0.6   # M2 — fact's 1-10 importance hint
embedding_weight  = 0.0   # M6 — needs an embedder + populated index
```

`RagContextLayer` over-fetches BM25 candidates (up to 4×
`max_rag_facts`), normalises each signal to `[0, 1]` within the
candidate batch, then keeps the top N by weighted sum.

| Field | Default | Purpose |
|---|---|---|
| `bm25_weight` | `1.0` | FTS5 BM25 quality. Best in batch = 1.0. |
| `recency_weight` | `0.0` | Newer fact id in batch = 1.0. |
| `importance_weight` | `0.6` | `importance/10`. NULL = neutral 0.5. |
| `embedding_weight` | `0.0` | Cosine sim to cue embedding. Needs M6 wired. |

`importance_weight = 0` reproduces pre-M2 BM25-only ordering. Raise
it to bias toward safety-critical facts (allergies, names, life
events); raise `recency_weight` to anchor retrieval to recent
conversation. Negative weights rejected at load time.

Importance is set per-fact by `FactExtractor`: the prompt asks the
LLM for a 1–10 integer (1 = throwaway, 5 = ordinary, 10 =
identity-defining / safety-critical). Out-of-range values clamp on
the store side; non-numeric input drops to NULL.

## Embeddings (M6)

```toml
[providers.embedding]
backend          = "off"   # "off" | "hash" | "fastembed"
dim              = 256     # hash only — vector size
fastembed_model  = "BAAI/bge-small-en-v1.5"
fastembed_cache_dir = ""   # blank = ~/.cache/fastembed
```

Three knobs gate the seam — flip all three to turn it on:

1. **Backend** — `[providers.embedding].backend`. `off` (default)
   short-circuits creation; the projector raises if listed without
   one, and the RAG layer skips the embedding signal even when its
   weight is positive (warned once at startup).
2. **Projector** — add `"fact_embedding"` to
   `[providers.memory].projectors`. The watermark-driven worker
   embeds every current fact missing a vector for the active model.
3. **Weight** — `[memory.retrieval].embedding_weight > 0`.

Built-in backends:

| Backend | Cost | Quality | When to use |
|---|---|---|---|
| `off` | none | none | default; semantic recall not wanted |
| `hash` | none | weak (token-overlap baseline) | tests, smoke checks, cold-start without ONNX |
| `fastembed` | ONNX runtime + ~130 MB model on first load | strong (BGE-small default) | production semantic recall |

Third-party backends register at startup (same pattern as the
STT factory); the seam is stable so `register_embedder` drops in
without touching `RagContextLayer`.

### FastEmbed install + model selection

```bash
cargo build --features local-embed
```

Brings in the `fastembed` crate (ONNX via `ort`). Model downloads on
first use (cached under `~/.cache/fastembed`). Common choices:

If `backend = "fastembed"` is selected but the feature isn't compiled
in, the bot **refuses to start** — the built-in `fastembed` factory
fails fast with the `--features local-embed` hint (the real ONNX
backend is injected only when the feature is present). Fail-fast at
boot beats a misconfigured deploy silently crashing on its first
message. (Only the backend wiring is checked, not the model download —
startup stays fast; the ~130 MB model still loads lazily on first
embed.)

| `fastembed_model` | Dim | Approx size | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~130 MB | Default. Best speed/quality tradeoff. |
| `BAAI/bge-base-en-v1.5` | 768 | ~440 MB | Higher quality, ~2× slower. |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90 MB | Smallest; older but well-tested. |

Vectors tag with the embedder's `name` (`fastembed:<model>`), so
upgrading from BGE-small to BGE-base accumulates new vectors beside
the old. The next `FactEmbeddingWorker` tick backfills under the
new model name; old rows stay queryable for audit but don't leak
into the active rank.

Operator playbook:

```toml
[providers.embedding]
backend         = "fastembed"
fastembed_model = "BAAI/bge-small-en-v1.5"

[providers.memory]
projectors = [
    "rolling_summary", "rich_note", "people_dossier", "reflection",
    "fact_supersede", "fact_embedding",
]

[memory.retrieval]
embedding_weight = 0.6
```

Side-index lives at `fact_embeddings` keyed `(fact_id, model)`. To
reclaim space after a model swap, drop rows tagged with the old
model:

```bash
sqlite3 data/familiars/<id>/history.db \\
    "DELETE FROM fact_embeddings WHERE model = 'fastembed:BAAI/bge-small-en-v1.5';"
# next FactEmbeddingWorker tick rebuilds under the new model
```

Or wipe the whole table to force a full re-embed under the active
model:

```bash
sqlite3 data/familiars/<id>/history.db "DELETE FROM fact_embeddings;"
```

## Memory projectors (M5)

Each watermark-driven writer implements the `MemoryProjector` trait —
`fn name(&self) -> &str` plus `async fn run(self: Box<Self>)`. TOML selector
picks which run; unknown names raise at config load.

```toml
[providers.memory]
projectors = [
    "rolling_summary", "rich_note", "people_dossier",
    "reflection", "fact_supersede",
]
```

| Name | Class | Side-index produced |
|---|---|---|
| `rolling_summary` | `SummaryWorker` | `summaries` |
| `rich_note` | `FactExtractor` | `facts` |
| `people_dossier` | `PeopleDossierWorker` | `people_dossiers` |
| `reflection` | `ReflectionWorker` | `reflections` |
| `fact_supersede` | `FactSupersedeWorker` | retires replaced rows in `facts` |
| `fact_embedding` | `FactEmbeddingWorker` | `fact_embeddings` (M6, opt-in) |

Default keeps the five above; `fact_embedding` is registered but
must be added explicitly since it depends on a configured embedder
backend (see [Embeddings (M6)](#embeddings-m6)). Drop a name to
disable that writer. Empty list disables every memory projector
(read paths still work — they just see stale side-indices).

### Worker tuning

Each built-in projector reads a `[providers.memory.<name>]` knob
table. Cadences trade memory freshness against LLM spend — every
tick that finds work costs background LLM calls. Knob tables are
accepted whether or not the projector is listed in `projectors`, so
toggling a projector keeps its tuning.

```toml
[providers.memory.rolling_summary]
turns_threshold = 10    # new turns per channel before summary refreshes
tick_interval_s = 5.0

[providers.memory.rich_note]
batch_size       = 10   # turns per extraction batch (also the trigger)
tick_interval_s  = 15.0
participants_max = 30   # cap on participant manifest rows in the prompt

[providers.memory.people_dossier]
tick_interval_s = 20.0

[providers.memory.reflection]
turns_threshold          = 20   # new turns before a reflection pass
max_reflections_per_tick = 3
max_turns_per_tick       = 50   # window cap on turns fed to the prompt
recent_facts_limit       = 20   # recent facts included in the prompt
tick_interval_s          = 60.0

[providers.memory.fact_supersede]
batch_size      = 5     # new facts evaluated per tick (one LLM call each)
tick_interval_s = 60.0
priors_max      = 20    # prior facts shown to the LLM per subject
```

All knobs are positive numbers; unknown keys fail at config load.
For experiments, drop `tick_interval_s` and the thresholds to make
side-indices converge fast; for production, raise them to cut
background spend.

Third-party projectors (Graphiti / Cognee / external memory
service) plug in by calling
`familiar_connect::processors::projectors::register_projector(name, factory)`
at startup; once registered, the same selector picks them up.
Each side-index remains regenerable from `turns`, so swapping
projectors mid-deployment doesn't lose ground-truth — restart the
new projector and let it backfill.

## Forward-looking schema

Documented now so the schema settles before wiring lands. Not read
by today's code.

```toml
# shipped
[providers.turn_detection]
strategy = "deepgram"            # | "ten+smart_turn"

[providers.stt]
backend = "deepgram"             # | "parakeet" | "faster_whisper" (V3 closed)

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

# shipped (M5) — see § Memory projectors

[providers.memory]
projectors = [
    "rolling_summary", "rich_note", "people_dossier",
    "reflection", "fact_supersede",
]

# planned (V5)

[providers.voice_pipeline]
mode = "cascaded"                # | "s2s" (V5)
```
