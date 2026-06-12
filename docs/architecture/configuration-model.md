# Configuration Model

Two config levels. Every operator knob, organised by goal:
[Tuning](tuning.md).

## 1. Bot instance config

Secrets and install selector the host needs to run the bot at all. Set
by the admin, never exposed through Discord.

- `DISCORD_BOT` — Discord bot token
- `OPENROUTER_API_KEY` — shared across every LLM call site
- `CARTESIA_API_KEY` — Cartesia TTS (required when `[tts].provider="cartesia"`)
- `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — Azure Speech
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) — Gemini TTS
- `DEEPGRAM_API_KEY` — Deepgram STT credential. Every other Deepgram knob lives in `[providers.stt.deepgram]`. Full list: [Tuning — STT — Deepgram](tuning.md#stt-deepgram).
- `FAMILIAR_ID` — picks the character folder under `data/familiars/` this process runs.

Lives in environment variables or a `.env` file. Never checked into
git. Never editable from inside Discord.

## 2. Character config

Per-familiar, loaded once from
`data/familiars/<familiar_id>/character.toml`, deep-merged over
`data/familiars/_default/character.toml`.

Surface today:

- `display_tz` — IANA timezone (default `"UTC"`) the final-reminder
  clock renders in (e.g. `"It is now: … 2:30PM PDT"`). Invalid names
  (e.g. `"PST"`) fail fast at config load.
- `aliases` — names the familiar answers to.
- `[providers.history].voice_window_size` / `.text_window_size` —
  recent-history layer windows, tiered by responder (defaults
  100 / 200). Safety nets behind the token-aware `[budget.<tier>]`
  caps.
- `[providers.history].coalesce_max_gap_seconds` — at prompt-render
  time, collapse consecutive same-speaker voice fragments when the
  gap between them is within this many seconds. Default `45.0`; `0`
  disables. Discord text turns are unaffected (they carry
  `platform_message_id`, which suppresses coalescing).
- `[providers.turn_detection].strategy` — `"deepgram"` (default) or
  `"ten+smart_turn"`. See
  [Tuning — local turn detection](tuning.md#local-turn-detection-v1).
- `[providers.stt]` + `[providers.stt.deepgram]` — STT backend
  selector + per-backend knobs (`endpointing_ms`, `keyterms`, …). Only
  `deepgram` today; V3 widens. Per-knob env override available. See
  [Tuning — STT — Deepgram](tuning.md#stt-deepgram).
- `[providers.memory]` — memory projector selection (`projectors`
  list) plus per-worker tuning tables
  (`[providers.memory.<name>]` — cadences, batch sizes,
  thresholds). See
  [Tuning — Memory projectors](tuning.md#memory-projectors-m5).
- `[llm].image_description_model` — model name for vision-based image
  descriptions (e.g. `"openai/gpt-4o"`). Shared across all slots; empty
  string (default) disables the description step. When set, `create_llm_clients`
  builds a reserved `"__image_description__"` client.
- `[llm].max_concurrent_requests` — process-wide cap on in-flight
  LLM requests across every slot (default `4`).
- `[llm.fast]` / `[llm.prose]` / `[llm.background]` — tiered LLM slots
  (model, temperature, optional `provider_order`, `reasoning`,
  `tool_calling`, `image_tools`, `multimodal`). Schema and call-site →
  slot mapping at [Tuning — LLM slots](tuning.md#llm-slots).
  `tool_calling` is wired end-to-end: when `true`, the responder for
  that slot installs the in-process `ToolRegistry` (today: `set_alarm`,
  `cancel_alarm`, and optionally `view_image`) and runs the agentic loop.
  `image_tools` (default `false`) independently gates `view_image`
  registration — the loop runs when either flag is set. `multimodal`
  (default `false`) controls whether `ImageResult` tool-result messages
  include JPEG content blocks (`true`) or text description only (`false`).
  See [Tool calling](overview.md#tool-calling) and
  [Image viewing](overview.md#image-viewing).
- `[tts]` — provider (`azure` / `cartesia` / `gemini`) + provider-specific voice / model fields.
- `[focus]` — attentional idle-nudge timing (`idle_wake_seconds`,
  `nudge_debounce_seconds`). See
  [Tuning — Attentional focus](tuning.md#attentional-focus).
- `[tools]` — agentic loop bounds (`loop_max_iterations`, default
  `5`), shared by voice and text responders.
- `[prompt].post_history_instructions` — free-text block appended to
  the *trailing* reminder, the system message that sits after recent
  history (right before the model's next turn). The deepest,
  most recency-biased slot, so behavioral nudges land hardest here.
  Rendered verbatim (markdown fine); empty string omits the block.
  The shipped default is a short roleplay-etiquette note nudging the
  familiar to lean on `<silent>`. See
  [Context pipeline — Final reminder](context-pipeline.md#final-reminder).

### Default profile

Reference familiar at `data/familiars/_default/`, checked into the
repo. Two purposes:

1. **Fallback source.** Any field missing from the user's
   `character.toml` falls back to the corresponding value in
   `_default/character.toml`. No hardcoded defaults live in Python —
   the default profile is the single source of truth.
2. **Documentation-by-example.** A new operator copies `_default/` to
   `data/familiars/my-familiar/` and edits from there.

The leading underscore keeps `FAMILIAR_ID=_default` from being a
meaningful selection.

### TTS providers

| Provider | Env vars | Character fields |
|---|---|---|
| `azure` (default) | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | `azure_voice` |
| `cartesia` | `CARTESIA_API_KEY` | `cartesia_voice_id`, `cartesia_model` |
| `gemini` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `gemini_voice`, `gemini_model` (+ optional style / scene / pace / accent / context / audio-profile) |

### Subscriptions

`data/familiars/<id>/subscriptions.toml` — which Discord channels the
bot listens in. Written by `/subscribe-text` and `/subscribe-voice`.
Not editable by hand in practice; the slash commands rewrite the whole
file on every mutation.
