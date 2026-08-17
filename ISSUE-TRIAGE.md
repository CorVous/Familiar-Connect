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
| 221 | Tool calling is no longer optional | bounded bug | |
| 222 | Familiars don't report focused channel correctly | bounded bug | |
| 203 | Wrong channel replies | investigation | |
| 199 | High volume of decryption failures | investigation | Reported symptom (`RTCP decryption failed`) is benign upstream songbird noise — quieted via a `songbird::driver::tasks::udp_rx=error` filter directive, restored at `-vv`. Found alongside it: `join_voice` registered its songbird handlers *after* the join resolved, so every one-shot op-5 Speaking event fired during the handshake was lost and 100% of voice frames went unattributed. Fixed with songbird's two-stage join. |
| 205 | Bad multi-party identity disambiguation | investigation | |
| 208 | Close out familiar location migration | bounded chore | |
| 214 | Add Privacy Policy and Terms of Use | bounded docs | |
| 218 | Detect issues with model configuration via diagnostics | bounded feature | |
| 183 | Improve token-counting heuristic | bounded feature | |
| 153 | Reduce strength of Regex rules for prose formatting | bounded feature | |
| 151 | Relocate hard-coded context to default/instance pattern | bounded feature | |
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

### N1 — shipped default selects an unimplemented TTS backend

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

### N5 — `est_in_tokens` duplicates the estimator constant

`src/llm.rs:510` hand-rolls `input_chars.div_ceil(4)` instead of calling
`budget::estimate_tokens`, duplicating `CHARS_PER_TOKEN` (`src/budget.rs:20`).
Recalibrating one silently desynchronizes the other. Relevant to #183.

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

### N9 — operating-mode instructions duplicated in two places

`src/context/final_reminder.rs:22-25` (`VOICE_INSTRUCTION` / `TEXT_INSTRUCTION`)
and `src/commands/run.rs:171-186` (`operating_modes()`) hold the same strings
verbatim. The latter's comment admits it: "Intentionally duplicates the strings
`OperatingModeLayer` is configured with — keep in sync." Folded into #151.

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
