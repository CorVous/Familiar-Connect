# Configuration Model

Two config levels. Every operator knob, organised by goal:
[Tuning](tuning.md).

## 1. Bot instance config

Secrets and install selector the host needs to run the bot at all. Set
by the admin, never exposed through Discord.

- `DISCORD_BOT` — Discord bot token
- `OPENROUTER_API_KEY` — shared across every LLM call site
- `CARTESIA_API_KEY` — Cartesia TTS (required when `[tts].provider="cartesia"`, the default)
- `DEEPGRAM_API_KEY` — Deepgram STT credential. Every other Deepgram knob lives in `[providers.stt.deepgram]`. Full list: [Tuning — STT — Deepgram](tuning.md#stt-deepgram).
- `FAMILIAR_ID` — picks the character folder (under the familiars root) this process runs.
- `FAMILIARS_ROOT` — overrides the per-user familiars root (default: platform data dir). `FAMILIAR_DEFAULTS_ROOT` overrides where the tracked `_default` skeleton resolves (default: `data/familiars`). See [On-disk layout](../getting-started/on-disk-layout.md#where-the-familiars-root-lives).

Lives in environment variables or a `.env` file. Never checked into
git. Never editable from inside Discord.

## 2. Character config

Per-familiar, loaded once from
`<familiars-root>/<familiar_id>/character.toml`, deep-merged over
`data/familiars/_default/character.toml` (the tracked `_default`
skeleton stays CWD-relative).

Surface today:

- `display_tz` — IANA timezone (default `"UTC"`) every model-facing
  clock renders in. Invalid names (e.g. `"PST"`) fail fast at config
  load. The store stays UTC-only; `display_tz` is the translation
  layer at the edges:
    - The final-reminder line is the single clock anchor and carries
      the numeric offset —
      `"It is now: 2026-05-04 2:30PM PDT (-07:00)"`. Everything else
      the model reads is local to the same zone and omits the offset,
      so `set_alarm`'s RFC-3339 `when` is constructible without
      guessing "PDT → -07:00".
    - Recent-history prefixes stay bare `HH:MM`; a `YYYY-MM-DD:`
      marker is interleaved only when the window spans more than one
      local day (see
      [Context pipeline](context-pipeline.md#dynamic)).
    - Schedule windows shown to the model (`start_activity`'s hours,
      the bedtime and schedule refusals) name the zone.
    - A date-only `valid_from` from the fact extractor anchors to
      local midnight rather than 00:00 UTC.
- `[sleep]` — sleep schedule, character-domain wall-clock config
  localized via `display_tz`. `window = "HH:MM-HH:MM"` (may wrap
  midnight; bad format fails fast) and `grace_minutes` (default 30)
  drive the reserved `sleep` activity (catalog entry in
  `activities.toml`). Omit the table to leave the schedule disarmed.
  See [Sleep § The window](sleep.md#the-window).
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
- `[providers.history].llm_mirror_calls` — rows the LLM call mirror
  keeps per familiar in the `llm_calls` table (default `1000`). Every
  LLM request/response — assembled system prompt, message array,
  reply, tool calls and results, timings, token counts — is written
  there for troubleshooting and analytics; the newest N rows survive
  and older ones are pruned on write. `0` switches mirroring off
  entirely: no sink is installed and the transport skips capturing
  prompts, so the feature costs nothing. Negative values are rejected
  at load (`[providers.history].llm_mirror_calls must be >= 0, got -1`)
  — an unbounded mirror is deliberately not offered. Budget ~30 KB per
  row. These rows contain every participant's messages in full; see
  [Privacy policy](../legal/privacy-policy.md#what-is-stored-on-disk)
  before raising the cap, and
  [Memory strategies — `llm_calls`](memory-strategies.md#the-one-table-that-is-not-a-projection-llm_calls)
  for the schema.
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
- `[providers.memory.rich_note].self_capability_filter` /
  `.self_capability_pattern` — the fact extractor's self-capability
  post-filter. The flag (default `true`) turns the whole filter off when
  `false`, including the display-name inability rail. The pattern
  (default `""`) replaces the built-in matcher when non-empty; it is
  compiled during config load, so an invalid regex fails startup with
  `[providers.memory.rich_note].self_capability_pattern must be a valid
  regex, got '<pattern>'` rather than blowing up mid-run. See
  [Context pipeline — No self-capability statements](context-pipeline.md#no-self-capability-statements).
- `[llm].image_description_model` — the **substitution** model: it
  describes an image for a calling slot that cannot see it (e.g.
  `"openai/gpt-4o"`). Shared across all slots; empty string (default)
  disables it. When set, `create_llm_clients` builds a reserved
  `"__image_description__"` client.
- `[llm].image_caption_model` — the **persistence** model: the durable
  caption written to history, and therefore the only trace of an image
  that ever reaches fact extraction, summaries, and dossiers. Used
  whenever the calling slot *can* see the image, since the image itself
  never survives into the turn store. Pick something cheap. Empty
  (default) falls back to `image_description_model`. When set,
  `create_llm_clients` builds a reserved `"__image_caption__"` client.

  The two keys name one call, never two. `view_image` runs exactly one
  describe leg per image and picks the model by role: a text-only slot
  runs the substitution model (whose output doubles as the caption,
  because the tool result *is* the description); a vision-capable slot
  runs the caption model and skips substitution entirely. Either key
  stands in for a missing other. See
  [Image viewing](overview.md#image-viewing).
- `[llm].max_concurrent_requests` — process-wide cap on in-flight
  LLM requests across every slot (default `4`).
- `[llm.fast]` / `[llm.prose]` / `[llm.background]` — tiered LLM slots
  (model, temperature, optional `top_p` / `top_k` / `presence_penalty`,
  `provider_order`, `reasoning`, `think_prepend`, `tool_calling`,
  `image_tools`, `multimodal`). Schema and call-site →
  slot mapping at [Tuning — LLM slots](tuning.md#llm-slots).
  `tool_calling` is wired end-to-end: the responder for that slot
  installs the in-process `ToolRegistry` (today: `set_alarm`,
  `cancel_alarm`, `silent`, and optionally `view_image`) and runs the
  agentic loop. It defaults to `true` and `false` is refused at load —
  silence is a tool call, so a tool-less slot could never decline to
  reply. `image_tools` (default `false`) independently gates
  `view_image` registration. `multimodal`
  controls whether `ImageResult` tool-result messages include JPEG
  content blocks (`true`) or the text description only (`false`).
  `think_prepend = true` is refused at load for the same reason — see
  [`think_prepend` and tools](#think_prepend-and-tools).
  See [Tool calling](overview.md#tool-calling) and
  [Image viewing](overview.md#image-viewing). Both flags are cross-checked
  against the model at startup — see
  [Startup model diagnostics](#startup-model-diagnostics).

#### `multimodal` is tri-state

`multimodal` has three states, not two:

| Value | Meaning |
|---|---|
| omitted | **Auto-detect** from the OpenRouter catalog: on when the model's `architecture.input_modalities` include `image`, off otherwise. Off when no catalog is available. |
| `true` | Explicit override — always on, whatever the catalog says. |
| `false` | Explicit override — always off, whatever the catalog says. |

Explicit always beats detected, in both directions. An operator who
writes `multimodal = false` against a vision model has made a deliberate
cost choice and detection must never quietly reverse it; that is why the
field is `Option<bool>` in `LLMSlotConfig` rather than a `bool` where an
explicit `false` would be indistinguishable from silence. A profile that
already sets the key keeps working unchanged.

`image_tools` is deliberately **not** tri-state. It is an intent knob,
not a capability fact: `view_image` works on a text-only model too — the
substitution description stands in for the image — and an operator may
not want image fetches at all on a vision-capable one. Neither direction
is inferable from the catalog, so it stays an explicit opt-in defaulting
to `false`.

`[llm.<slot>]` does not run the unknown-key check, so adding
`multimodal` (or omitting it) never fails load.
- `[tts]` — provider (`cartesia`, the default and only implemented backend) + its voice / model fields.
- `[focus]` — attentional unread-nudge controls (`unread_nudge_enabled`,
  `nudge_debounce_seconds`). See
  [Tuning — Attentional focus](tuning.md#attentional-focus).
- `[tools]` — agentic loop bounds (`loop_max_iterations`, default
  `5`), shared by voice and text responders.
- `[tools].trusted_image_hosts` — hosts `view_image` may fetch.
  Default-deny: exact hostnames, or a `*.suffix` pattern matching any
  subdomain of `suffix`. Ships with Discord's CDNs plus a short list of
  image-only CDNs. Entries must be bare hostnames — a scheme, path, or
  port is rejected at load.
- `[tools].allow_untrusted_image_urls` — default `false`. `true` drops
  the host list. Private, loopback, link-local, and other reserved
  addresses stay refused either way, as do non-http(s) schemes; that
  rule has no config escape. See
  [Security — outbound image fetches](security.md#outbound-image-fetches-view_image).
- `[voice].roster_event_window_seconds` — how long a voice-call
  join/leave stays narrated in the prompt (`"Tam just joined."`),
  default `120.0`. Must be positive; a non-positive or non-numeric
  value fails fast at load. The `In the call: …` roster line itself
  never decays, and neither line is rendered on text turns. See
  [Context pipeline — Voice call roster](context-pipeline.md#voice-call-roster).
- `[prompt].post_history_instructions` — free-text block appended to
  the *trailing* reminder, the system message that sits after recent
  history (right before the model's next turn). The deepest,
  most recency-biased slot, so behavioral nudges land hardest here.
  Rendered verbatim (markdown fine); empty string omits the block.
  The shipped default is a short roleplay-etiquette note nudging the
  familiar to stay quiet unless it means to speak. See
  [Context pipeline — Final reminder](context-pipeline.md#final-reminder).
- `[prompt].operating_mode_voice` / `.operating_mode_text` — the per-mode
  operating directive. **One source for two consumers**: the
  `operating_mode` system-prompt layer and the trailing final reminder,
  which restates it for recency. No in-code copy exists, so the two
  cannot drift; a blank value drops the directive from both.
- `[prompt].voice_tool_ack` — voice-tier nudge to speak before calling a
  tool. Rendered only when the voice slot actually has tool calling; blank
  omits it. See [Overview — voice tool ordering](overview.md#tool-calling).
- `[prompt].shift_focus_coaching` — the clause spliced onto the unread
  digest after an em dash (`There are new messages in #x (id N) (3) —
  <this>`). One line, no leading dash; rendered only when the text slot
  has tool calling, and blank leaves the digest a plain statement of
  fact. See [Context pipeline — Final
  reminder](context-pipeline.md#final-reminder).
- `[prompt].start_activity_description` — the when-to-go policy carried by
  the `start_activity` tool description. Roleplay guidance rather than API
  contract, so it is config; the activity enum and its availability hints
  stay code-built from the catalog. Keep it under ~450 characters — it
  rides in every text-tier tool schema.
- `[prompt].rolling_summary_system`, `reflection_system`,
  `dossier_self_system`, `dossier_other_system` — static instruction text
  for the memory projectors (rolling summary, reflection, self-record and
  other-person dossiers). The window data (turns, facts, prior text,
  importance annotations) and the reflection reply contract are assembled
  in code; only the wording is configurable. Placeholders:
  `dossier_self_system` takes `{self_name}`, `dossier_other_system` takes
  `{display_name}`.
- `[prompt].sleep_consolidation_system`, `sleep_stance_system`,
  `sleep_synthesis_system`, `dream_extraction_clause` — static
  instruction text for the sleep passes and the fact-extractor's
  dream-framing clause. Dynamic window data is interpolated in code;
  only the wording is configurable. Placeholders: the stance / synthesis
  fields take `{self_name}`; `dream_extraction_clause` takes
  `{self_name}`, `{self_key}`, `{ids}`. Validation rails stay
  code-enforced regardless of this text. See
  [Sleep — Prompt text is config, rails are code](sleep.md#prompt-text-is-config-rails-are-code).

### Default profile

Reference familiar at `data/familiars/_default/`, checked into the
repo. Two purposes:

1. **Fallback source.** Any field missing from the user's
   `character.toml` falls back to the corresponding value in
   `_default/character.toml`. No hardcoded defaults live in code —
   the default profile is the single source of truth.
2. **Documentation-by-example.** A new operator copies `_default/`
   into the familiars root and edits from there.

The leading underscore keeps `FAMILIAR_ID=_default` from being a
meaningful selection.

### TTS providers

| Provider | Status | Env vars | Character fields |
|---|---|---|---|
| `cartesia` (default) | wired | `CARTESIA_API_KEY` | `cartesia_voice_id`, `cartesia_model` |

`cartesia` is the only accepted value. The `azure` and `gemini` stubs were
removed; a profile still naming one fails config validation with a message
saying the provider is no longer supported. The `TtsClient` /
`StreamingTtsClient` seam remains the extension point for further backends.

### `think_prepend` and tools

`think_prepend` appends a fake closed think block (`<think>\n\n</think>`)
as a trailing **assistant** message on every request from that slot — genuine,
opt-in prefix completion, and a Qwen3.6 no-think stabiliser. Providers that
implement prefix mode refuse to combine it with a `tools` array (DeepSeek
answers `Function call should not be used with prefix`). Since `tool_calling`
is mandatory, every slot sends tools, so `think_prepend` is not merely risky —
it is unusable. Writing `think_prepend = true` fails the load with:

```text
[llm.<slot>].think_prepend = true is unsupported: it appends a trailing
assistant message, which providers read as prefix completion, and prefix
completion cannot be combined with the tools array every slot now sends —
remove the key (it defaults to false) or set [llm.<slot>].think_prepend = false
```

Remove the key rather than flipping it; the default is the only supported
value.

### Startup model diagnostics

Capability flags on an LLM slot are free-form assertions about a free-form
OpenRouter model string, so config loading cannot validate them — and must not
try, because that would make loading depend on the network. Two checks run
after loading instead (`src/model_diagnostics.rs`).

**Tool-calling reachability** — immediate, no network. Config loading already
refuses `tool_calling = false`; this boot check is the backstop for a config
built in process. Both responder surfaces get a focus manager wired
unconditionally, so a tool-less slot could neither decline to reply nor reach
`shift_focus`. Logged at `ERROR` at boot with the slot named, phrased as
unsupported rather than merely limited.

**Capability audit** — detached, best-effort. After the bus starts, a
fire-and-forget task fetches `GET https://openrouter.ai/api/v1/models` and
compares each slot's declared flags against the model's metadata:

| Declared | Model metadata | Level |
|---|---|---|
| `tool_calling = true` (always) | `supported_parameters` lacks `tools` | `ERROR` — the only fix is a different model |
| `multimodal = true` (explicit) | `architecture.input_modalities` lacks `image` | `ERROR` |
| `image_tools = true` | `input_modalities` lacks `image` | `ERROR` |
| `image_tools = false` | `input_modalities` has `image` | `INFO` (advisory) |

An omitted `multimodal` is never flagged in either direction — auto-detection
already answers it, so there is no operator claim to contradict.

Every line names the slot, the model, and the remediation. An unknown model id,
an unreachable catalog, or an unparseable response produces one line and then
silence — the audit is advisory and never blocks, delays, or fails startup.
Variant suffixes (`:free`, `:nitro`, …) fall back to the base model id.

#### The catalog cache

Auto-detection needs the catalog at *composition* time — before the LLM clients
exist — while boot must stay offline. The two are reconciled by a disk cache
holding the last known good `GET /models` response:

- **Read** — synchronously at boot, before `create_llm_clients`. A local file
  read; there is no URL in the signature. Whatever it returns fills the unset
  `multimodal` flags for this process.
- **Refresh** — on the same detached task as the capability audit, after the
  bus starts. It fetches, rewrites the cache, and audits. What it writes takes
  effect on the **next** start; nothing in this process waits on it. A cache
  still inside its TTL skips the fetch entirely and audits from disk.

| | |
|---|---|
| Location | The platform per-user **cache** directory: `~/.cache/familiar-connect/openrouter-models.json` on Linux (honours `XDG_CACHE_HOME`), the OS-correct analog elsewhere. Falls back to a CWD-relative `data/cache/` when no home directory resolves. |
| Why there | The file is entirely regenerable from the network. It is not state, so it must not sit under the [familiars root](../getting-started/on-disk-layout.md#where-the-familiars-root-lives) beside `history.db` — clearing caches has to stay safe, and a backup of the state tree should not carry a stale model catalog. Same `ProjectDirs` qualifier the familiars root uses. |
| Staleness | 24 hours. OpenRouter adds models and corrects metadata on the order of days, and the worst a stale entry can do is one wrong auto-detection that an explicit `multimodal` always beats. A day also keeps a restart loop off the `/models` endpoint. |
| Age vs. use | Age gates the **refresh**, never the **read**. A month-old cache is still used at boot — last known good beats nothing. |

Failure is never fatal. A missing cache, an unreadable one, a truncated one, or
one holding something that is not a `CachedCatalog` all read as *absent*: every
slot keeps its configured value, which is the behaviour that predates the cache.
A first-ever run with no cache and no network therefore behaves exactly as
before. The next successful refresh overwrites a corrupt file; a failed refresh
leaves the previous good one intact.

### Subscriptions

`<familiars-root>/<id>/subscriptions.toml` — which Discord channels the
bot listens in. Written by `/subscribe-text` and `/subscribe-voice`.
Not meant for hand edits — the slash commands rewrite the whole file on
every mutation.
