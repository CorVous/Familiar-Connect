# Issue triage — `august-bugfixes`

Working notes for the sweep over the 25 open GitHub issues. Classification,
what landed on this branch, and what deliberately did not. Newly discovered
defects (not tracked on GitHub) are in the second half.

No comments were posted to any GitHub issue.

## Classification

| # | Title | Class | Disposition |
|---|---|---|---|
| 217 | Incorrect cargo.toml repository link | trivial | |
| 219 | `dim` contradicting `fastembed_model` fails silently | bounded bug | |
| 220 | silent reported as cancelled | bounded bug | |
| 221 | Tool calling is no longer optional | bounded bug | Focus switching was reachable only through the `shift_focus` tool, so a slot with `tool_calling = false` could never change context again. Three changes, all confined to the tool-less configuration. (1) A **non-tool focus fallback** in `text_responder.rs`: a message that directly pings the bot in a subscribed-but-unfocused channel now calls `FocusManagerApi::shift_now` and answers there, instead of staging silently. Gated on the responder having no agentic loop at all (`tool_mode` false — no registry/factory, or neither `tool_calling` nor `image_tools`); with tools on, `shift_focus` is her deliberate control and an automatic shift would fight it, so today's staging behavior is byte-identical. Non-ping traffic still stages either way. (2) The unread digest's "— use shift_focus if it pulls your attention" clause in `final_reminder.rs` is now gated on `tools_enabled`, which the text responder threads in (voice already did): a model that cannot comply is no longer coached to. (3) The tool-less text path ran only `SilentDetector`, not the `StreamGate` leak guard voice uses, so a model coached toward `shift_focus` could post the literal call syntax to Discord; `stream_bare_inner` now runs the same `StreamGate` and suppresses a leaked `shift_focus(…)` / `<invoke …>` imitation. |
| 222 | Familiars don't report focused channel correctly | bounded bug | |
| 203 | Wrong channel replies | investigation | |
| 199 | High volume of decryption failures | investigation | Reported symptom (`RTCP decryption failed`) is benign upstream songbird noise — quieted via a `songbird::driver::tasks::udp_rx=error` filter directive, restored at `-vv`. Found alongside it: `join_voice` registered its songbird handlers *after* the join resolved, so every one-shot op-5 Speaking event fired during the handshake was lost and 100% of voice frames went unattributed. Fixed with songbird's two-stage join. |
| 205 | Bad multi-party identity disambiguation | investigation | |
| 208 | Close out familiar location migration | bounded chore | |
| 214 | Add Privacy Policy and Terms of Use | bounded docs | |
| 218 | Detect issues with model configuration via diagnostics | bounded feature | New `src/model_diagnostics.rs`. Two checks, both outside config loading so loading stays offline. (1) A detached post-boot task fetches `GET /models` and compares each slot's `tool_calling` / `multimodal` / `image_tools` against `supported_parameters` + `architecture.input_modalities`: declared-but-unsupported → `ERROR`, the inverse → `INFO` advisory; unknown id, unreachable catalog, or unparseable body → one line then silence. Never gates readiness. (2) The motivating case from #221 — `tool_calling = false` on a focus-managed slot — needs no network, so it is checked unconditionally at composition time and logged at `ERROR`. Comparison + parsing are pure and unit-tested without network. |
| 183 | Improve token-counting heuristic | bounded feature | Collection and reporting landed; enforcement on the assembly path deliberately did not. `budget::TokenCalibration` is a process-wide, in-memory singleton (same shape as `SpanCollector`: poison-safe `Mutex<Option<Arc<…>>>`, `reset_token_calibration` test seam) keyed by **model**, not `model.slot` — chars-per-token is a tokenizer property, so slots sharing a model share a rate and pooling their samples converges faster, and readers only know a model anyway. It holds `Σ estimated` / `Σ actual` and reports the ratio of those totals rather than an EWMA: no decay constant to tune and exact from the first sample, where an EWMA seeded at `1.0` would spend its early calls biased toward the seed. Fed from `CallMetrics::emit`, where the heuristic's guess and OpenRouter's `prompt_tokens` are both already in hand; the running ratio is then emitted on the `[LLM call]` line as the additive `cal_ratio` key (existing keys and formats untouched; the line carries no `span=`, so `diagnose` does not scrape it). `budget::estimate_tokens_calibrated(text, model)` is an additive wrapper — `estimate_tokens` and `CHARS_PER_TOKEN` are unchanged and every pinned estimator test stands. **Calibration may only ever revise an estimate upward**: the estimate gates client-side trimming before the request is sent, so over-counting merely drops extra context while under-counting risks an API rejection; a ratio at or below `1.0` is discarded and a ratio above it is capped at `4.0` (image blocks contribute 0 chars but real tokens, so an uncapped degenerate sample would over-trim; `4.0` still covers CJK density). **Not wired into `src/context/layers.rs`:** no call site there has a model in scope — `AssemblyContext` carries none — and threading one through would mean changing `AssemblyContext` plus both responders in `processors/`, well past this change's blast radius. Layers still trim on the raw estimator, noted in the module header. |
| 153 | Reduce strength of Regex rules for prose formatting | bounded feature | |
| 151 | Relocate hard-coded context to default/instance pattern | bounded feature | Eight prompt strings moved out of Rust into `[prompt]` in `_default/character.toml`, joining the six already there: `operating_mode_voice` / `operating_mode_text` (the per-mode directive — this collapses N9's verbatim duplication between `final_reminder.rs` and `run.rs::operating_modes()` into one config pair feeding both consumers), `voice_tool_ack` (the speak-before-you-call nudge), `start_activity_description` (the tool description's roleplay policy — persona guidance that happened to live in a schema field), `rolling_summary_system`, `reflection_system`, `dossier_self_system`, `dossier_other_system` (the memory projectors' instruction text). Each shipped default is byte-identical to the string it replaced, so a default-config run produces the same prompts; blanking the profile now yields no text at all rather than a stale in-code copy, which is what the new tests pin. Doctrine held: only phrasing moved — the reflection JSON contract, the `start_activity` enum + availability hints, the dossier importance ordering, and every validation rail stay in code, so a bad override degrades tone and can never break parsing. Deliberately **not** relocated: `fact_extractor.rs`'s `self_clause` / `intro` / `guidance` (~2400 chars welded to `render_contract(&FACT_SCHEMA)`, several rules with no code-level rail — needs its own override-boundary tests), the remaining `Tool::new` schema text (API surface, not persona), `structured_request.rs`'s contract renderer, and `layers.rs`'s markdown section headers. |
| 180 | Time format hygiene | medium feature | |
| 196 | ten-vad-sys FFI crate | large | |
| 200 | Break out character.toml by file | large | |
| 184 | Catch-up profiling for default config tuning | large | |
| 181 | Tool call context requirements | architectural | |
| 155 | Adhere to design guidelines from eval trials | architectural | |
| 204 | Decide on image processing strategy | design decision | |
| 206 | Research staged passes and cache optimization | research | |
| 207 | Simplify vectorization strategy (Turso) | research | |
| 96 | Parallel database accesses | blocked upstream | |
| 130 | Memory-hardening follow-ups | deferred by author | |

## Newly discovered issues

Defects and papercuts found incidentally during the sweep. Not filed on
GitHub. `[fixed]` items landed on this branch; `[open]` items are recorded
here only.

### N1 — shipped default selects an unimplemented TTS backend `[fixed]`

`data/familiars/_default/character.toml:331` and the Rust default
(`src/config.rs:351`) both set `[tts].provider = "azure"`. Both the Azure and
Gemini backends are stubs that return an error at synthesis time
(`src/tts.rs:1112`, `src/tts.rs:1329` — "backend not wired (deferred to
wiring layer 10)"). Cartesia is the only functional provider. `.env.example`
and `docs/getting-started/on-disk-layout.md` describe Azure as the default and
tell the user to set `AZURE_SPEECH_KEY`.

Net effect: a fresh install that follows the docs has no working TTS, and the
failure surfaces mid-conversation on the first synthesis rather than at
startup. Enabling the `azure-tts` feature does not help — the backend is
unwired regardless of the flag.

Fixed on this branch: the shipped default is now `cartesia` in both places
(`TTSConfig::default()` and the `parse_tts_config` fallback), matching the
`cartesia_voice_id` / `cartesia_model` the profile already carried. `azure` and
`gemini` stay valid config values — they are placeholders for real work — but
`tts::unwired_provider_reason` names them, and the composition root refuses to
build a client for either, logging at `ERROR` and running without TTS instead
of failing on the first synthesis. `.env.example`, the `_default` profile, and
the five docs that called Azure the default were updated in the same change.

### N2 — Deepgram connect URL with member names is logged at INFO

Up to 100 voice-member display names, usernames, aliases, and guild nicks are
appended to the Deepgram websocket URL as `keyterm=` recognition hints
(`src/bot.rs:916-927` → `src/bot.rs:2400-2404` → `src/stt/deepgram.rs:461-497`).
That full URL is then logged at INFO (`src/stt/deepgram.rs:546-551`), so every
voice session writes its participant roster into the logs. The query string
should be redacted.

### N3 — `view_image` fetches arbitrary URLs

`src/tools/image.rs:59-86,113-117` issues an unrestricted GET to any URL the
model passes, including URLs a user pastes into chat (`src/bot.rs:632-641`).
No scheme/host/private-range restriction, so a pasted link discloses the
operator's IP to an arbitrary host, and internal addresses are reachable.

### N4 — `cosine` silently returns 0.0 on dimension mismatch

`src/context/layers.rs:1081-1097` returns `0.0` when the query and stored
vectors differ in length rather than surfacing the mismatch. Currently
unreachable in normal operation (the embedder name keys storage), but it is
the silent-failure surface behind #219.

### N5 — `est_in_tokens` duplicates the estimator constant `[fixed]`

`src/llm.rs:510` hand-rolled `input_chars.div_ceil(4)` instead of calling
`budget::estimate_tokens`, duplicating `CHARS_PER_TOKEN` (`src/budget.rs:20`).
Recalibrating one silently desynchronizes the other. Relevant to #183.

Fixed with #183. The call site holds a scalar count, not the text, so
`estimate_tokens` itself was not callable; `budget::estimate_tokens_from_chars`
now owns the arithmetic and `estimate_tokens` delegates to it, leaving one home
for the constant.

### N6 — alarms never fire: the waker publishes an undowncastable payload `[fixed]`

`src/tools/waker.rs:100-120` republishes an alarm onto `discord.text` with a
raw `serde_json::Value` payload. `Payload` is a type-erased
`Arc<dyn Any + Send + Sync>` (`src/bus/envelope.rs:22-28`), and both subscribers
recover it with `downcast_ref::<DiscordTextPayload>()`, silently returning
`Ok(())` on mismatch (`src/processors/text_responder.rs:298-300`,
`src/processors/history_writer.rs:69-71`). A `Value` can never downcast to that
struct, so **every fired alarm is dropped** — no reply, no history write, in any
channel. Every other producer of this topic publishes the struct
(`src/sources/discord_text.rs:116`, `src/processors/text_responder.rs:633`,
`src/activities/engine.rs:1868`).

Root cause is visible at `src/activities/engine.rs:509-514`, whose comment
concedes `DiscordTextPayload` has no `alarm` field and that "alarm-piercing does
not reach the gate through the responder path yet". The waker worked around the
missing field with an untyped payload instead of adding it.

`tests/tools_alarm.rs` misses this because it asserts on the raw `Value`
payload and never round-trips through `TextResponder::handle`.

Fixed on this branch: `DiscordTextPayload` gained the `alarm: bool` field the
gate was waiting on, the waker publishes the real struct with `alarm = true`,
and `GatePayload::from_discord` reads the field instead of hardcoding `false`
(the piercing branch in `gate()` was already implemented and tested — only the
adaptation was stubbed). The waker tests now assert on the typed payload, and
a new `waker_event_drives_a_text_reply` feeds the waker's own published event
through `TextResponder::handle` and asserts a `send_text` on the right channel
— the round trip whose absence let this survive.

### N7 — `session_id` format differs across `discord.text` producers `[fixed]`

`TurnRouter` keys barge-in/cancel on the exact `session_id` string
(`src/bus/router.rs:27-35`), but producers disagree:
`format!("discord:{channel_id}")` (`src/sources/discord_text.rs:107`) vs. bare
`channel_id.to_string()` (`src/processors/text_responder.rs:628`,
`src/activities/engine.rs:1863`, `src/tools/waker.rs:115`). A real message and a
synthetic wake for the *same* channel are therefore different router sessions,
so neither preempts the other and both can reply concurrently. The bare form
also hid every synthetic turn from `TypingInterruptHandler`, which looks a
scope up by `format!("discord:{channel_id}")` (`src/typing_interrupt.rs:186`).

Fixed on this branch: all three synthetic producers now publish under
`discord:{channel_id}`, matching the real source and the `voice:{id}` /
`twitch:{id}` convention. Nothing parsed the bare form and no test contract
pinned it, so unification was a clean rename.

### N8 — voice turns lack the per-turn focus isolation text has

`shift_focus` is wired into the voice registry (`src/tools/builtins.rs:39`), but
`voice_responder.rs` never wraps `ctx.focus_manager` in the `TurnFocusRecorder`
that `text_responder.rs:878-883` uses. A mid-voice-turn shift commits straight
to global focus with no turn-local staging. It does not misroute audio, but it
lets a voice turn silently move the *text* focus pointer.

### N9 — operating-mode instructions duplicated in two places `[fixed]`

`src/context/final_reminder.rs:22-25` (`VOICE_INSTRUCTION` / `TEXT_INSTRUCTION`)
and `src/commands/run.rs:171-186` (`operating_modes()`) hold the same strings
verbatim. The latter's comment admits it: "Intentionally duplicates the strings
`OperatingModeLayer` is configured with — keep in sync." Folded into #151.

Fixed on this branch: both constants are gone. `operating_modes(&config)`
builds one map from `[prompt].operating_mode_voice` / `.operating_mode_text`
and hands it to both consumers — `OperatingModeLayer` and `FinalReminder`
(via each responder's `with_mode_instructions`). The reminder holds no
fallback copy, so an unconfigured or blank directive renders nothing rather
than resurrecting a stale duplicate.

### N10 — stale Python-era symbol name in docs

`docs/architecture/context-pipeline.md:289` refers to `_is_self_capability`.
The Rust symbol is `is_self_capability`
(`src/processors/fact_extractor.rs:115-120`).

### N11 — `intfloat/e5-small-v2` is listed but unusable

`fastembed_native_dim` knows the model (384), but `resolve_model`
(`src/embedding/fastembed.rs`) has no matching `fastembed::EmbeddingModel`
variant for it in the pinned 5.17.2 — only `MultilingualE5Small` exists. So the
name passes config validation and then fails at embedder construction.

### N12 — Twitch documented as live, implemented as a mock

`docs/index.md:11` and `docs/architecture/overview.md:58` describe an active
Twitch EventSub client. `src/twitch_watcher.rs` is feature-gated off and the
real EventSub session is deferred; only the mock exists.

### N13 — an alarm in a non-focused channel still produces no reply

Surfaced while fixing N6. `gate()` now correctly lets an alarm pierce an
activity absence, but the *focus* gate is separate: with a `FocusManager`
wired, an alarm firing outside the focused channel takes the responder's normal
staging path and never replies. Strictly better than before N6 (previously no
channel replied at all), but whether an alarm should also pierce focus is a
product decision, not a bug fix. Related to #221, where focus switching is
reachable only via a tool call.

### N14 — a channel rename is invisible until the next restart

Found while fixing #222. The focus name caches (`channel_names` / `guild_names`)
are write-only pushes from four discovery points — the `on_ready` snapshot, both
`/subscribe-*` handlers (added with the #222 fix), `register_dm_channel`, and
boot-time `rehydrate_dm_naming`. There is no `ChannelUpdate` / `GuildUpdate`
handler, so renaming a channel or server mid-session leaves presence, logs, and
the model's focus line showing the *old* name until the process restarts. Cheap
to close (`EventHandler::channel_update` → `fm.set_channel_name`), but it is a
separate gateway event from the one #222 reported.
