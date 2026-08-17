# Issue triage — `august-bugfixes`

A sweep over the 25 open GitHub issues: how each was classified, what landed on
this branch, and what deliberately did not. Defects found along the way that
are not tracked on GitHub are in the last section.

**No comments were posted to any GitHub issue.** Nothing here has been filed
upstream; this file is the whole record.

## Classification

| # | Title | Class | Outcome |
|---|---|---|---|
| 217 | Incorrect cargo.toml repository link | trivial | fixed |
| 208 | Close out familiar location migration | bounded chore | fixed |
| 219 | `dim` contradicting `fastembed_model` fails silently | bounded bug | fixed |
| 220 | silent reported as cancelled | bounded bug | fixed |
| 221 | Tool calling is no longer optional | bounded bug | fixed |
| 222 | Familiars don't report focused channel correctly | bounded bug | fixed |
| 199 | High volume of decryption failures | investigation | fixed — and a worse defect found beside it |
| 214 | Add Privacy Policy and Terms of Use | bounded docs | fixed |
| 218 | Detect issues with model configuration via diagnostics | bounded feature | fixed |
| 153 | Reduce strength of Regex rules for prose formatting | bounded feature | fixed |
| 151 | Relocate hard-coded context to default/instance pattern | bounded feature | fixed, minus two deliberate exclusions |
| 180 | Time format hygiene | medium feature | LLM-facing edges fixed; store was already clean |
| 183 | Improve token-counting heuristic | bounded feature | fixed — calibration learned, reported, and enforced on the assembly budget |
| 203 | Wrong channel replies | investigation | reported mechanism already fixed; two real causes found and fixed |
| 205 | Bad multi-party identity disambiguation | investigation | probably fixed via #199 — needs live confirmation |
| 196 | ten-vad-sys FFI crate | large | not attempted |
| 200 | Break out character.toml by file | large | not attempted; cheapest slice already taken by #151 |
| 184 | Catch-up profiling for default config tuning | large | not attempted; #183 supplies one of its inputs |
| 181 | Tool call context requirements | architectural | not attempted |
| 155 | Adhere to design guidelines from eval trials | architectural | not attempted |
| 204 | Decide on image processing strategy | design decision | not attempted — needs a product call |
| 206 | Research staged passes and cache optimization | research | not attempted |
| 207 | Simplify vectorization strategy (Turso) | research | not attempted |
| 96 | Parallel database accesses | blocked upstream | not attempted |
| 130 | Memory-hardening follow-ups | deferred by author | not attempted, per the issue itself |

## What landed

### 217 — repository link
Workspace `repository` pointed at a personal fork. The `authors` line is
attribution, not a link, and was left alone.

### 208 — familiars migration shim
Removed `migrate_legacy_familiars`, its call site, and its two tests. Home-dir
resolution itself stays. **Behavior change worth knowing:** anyone who never
ran a migrating build must now move `./data/familiars/<id>` by hand or set
`FAMILIARS_ROOT`; the "familiar folder does not exist" error already names the
absolute path it searched, so it is self-diagnosing. The issue says both
operators have run it.

### 219 — embedding `dim`
`dim` contradicting `fastembed_model` was accepted silently, and for the
fastembed backend `dim` is inert anyway — only `hash` reads it. The
model→native-dim table moved out from behind the `local-embed` feature gate
into `embedding::fastembed_native_dim`, collapsing a second hardcoded copy, and
is injected into `parse_embedding_config` rather than imported directly. An
explicit contradicting `dim` is now rejected at load; an unset one derives the
native value. The shipped `_default` profile carried the same contradiction
(`dim = 256` beside bge-small's 384) and was the trap a new user would hit
first.

### 220 — silent reported as cancelled
`Drop for SseDeltaStream` hardcoded `status=cancelled` for any early abandon,
so a deliberate `<silent>` decision was indistinguishable from a user barge-in.
`stream_completion` now returns an `LlmStream` wrapper exposing
`note_abandon_status`, and the two bare (no-tools) responder paths mark
`silent` / `suppressed`. Genuine barge-in still reports `cancelled`; the
tool-enabled paths were never affected because they drain the stream to
completion. `diagnose` does not scrape this line (no `span=` key), confirmed.

### 221 — tool calling is no longer optional
`shift_focus` was the only post-startup mutator of focus, so a slot with
`tool_calling = false` could never change channel again. Three changes, all
confined to the tool-less configuration:

1. A **non-tool focus fallback**: a message directly pinging the bot in a
   subscribed-but-unfocused channel now shifts focus and answers there instead
   of staging silently. Gated on the responder having no tool path at all, so
   with tools on the behavior is byte-identical — `shift_focus` stays the
   model's deliberate control and an automatic shift would fight it, and the
   #170 per-turn send-routing fix is untouched. Non-ping traffic still stages
   either way.
2. The unread digest no longer coaches a tools-off model to "use shift_focus".
3. The tool-less text path ran only `SilentDetector`, not the `StreamGate` leak
   guard voice uses — so a model coached toward a tool it could not call could
   post the literal call syntax to Discord. It now runs the same guard.

### 222 — channel reported as a raw snowflake
`/subscribe-text` and `/subscribe-voice` never recorded channel or guild names,
so any channel absent from the `on_ready` snapshot stayed nameless forever and
presence rendered a bare snowflake. Both handlers now record names from the
interaction (no REST added), a historyless DM falls back to its peer id, and
the remaining misses render `unnamed channel (id N)`. A gateway backfill at the
point of the miss was considered and rejected: `FocusManager` is
feature-agnostic core with no cache access, and resolving an uncached channel
is a REST round-trip on the presence path.

### 199 — decryption failures, and the real bug beside them
The reported `RTCP decryption failed: Crypto(Error)` spam is **benign upstream
songbird noise** — songbird's own source documents these UDP errors as
non-fatal by design and forwards the packet anyway. Quieted with a
target-scoped filter directive, restored at `-vv`.

The serious defect was next to it in the same log: `unmapped_frames` equalled
`speaking_frames`, i.e. 100% of voice frames unattributed. `join_voice`
registered its event handlers *after* `Songbird::join` had fully resolved, but
the SSRC→user Speaking event is one-shot and the events task has no replay — so
it was routinely lost during the handshake. Every frame then got a provisional
id equal to the raw SSRC, which can never resolve to a Discord user, making
transcripts anonymous. Handlers now register on the `Call` before stage-one
join.

### 214 — privacy policy and terms of use
Two original documents under `docs/legal/`, wired into the mkdocs nav and the
README, ready to paste into the Discord developer portal. Written against what
the code actually does rather than from a template — which is how N1, N2 and N3
below were found. The issue asked to "word-for-word appropriate existing
documents"; the intent (maximum disclaimer, no retention or compliance
promises, operator bears responsibility) is honored, but the text is original,
because another company's policy is itself copyrighted and describes a
different data-processing reality.

### 218 — model-configuration diagnostics
New `src/model_diagnostics.rs`, with both checks outside config loading so
loading stays offline. A detached post-boot task fetches `GET /models` and
compares each slot's declared `tool_calling` / `multimodal` / `image_tools`
against the model's real `supported_parameters` and
`architecture.input_modalities`: declared-but-unsupported logs `ERROR` with the
remediation, the inverse logs an `INFO` advisory. An unknown id, unreachable
catalog, or unparseable body logs once and is otherwise silent; readiness is
never gated. The response shape was verified against the live API, but the
parser is hand-rolled and defensive — a bad entry is skipped, never fatal.

Separately, the #221 case needs no network, so a focus-managed slot with tool
calling off is flagged unconditionally at composition time. The text and voice
wordings differ, because only text has the ping fallback.

### 153 — over-broad self-capability filter
The fact filter was a prefix match with no trailing anchor, and one branch —
`the (assistant|ai|familiar|model|bot)` — required no negation at all. "The
familiar loves rainy days" was discarded, in a product whose characters are
called familiars. Likewise a bare "I can juggle", "I have no siblings", "I
don't like Mondays", and "As an AI researcher, Cor works on alignment". Every
alternative now requires a capability tail or a capability object.

Judgement call worth reviewing: `i cannot|can't|can not` was left broad.
Negated first-person ability is the core disclaimer shape and the pinned "I
cannot remember the names of people" has no distinguishing object, so idioms
like "I can't wait" are still dropped. The clean fix is a negative lookahead,
which Rust's `regex` crate does not support. Operators who disagree now have
`[providers.memory.rich_note].self_capability_filter` and
`self_capability_pattern`, validated fail-fast at load.

### 151 — hard-coded prompts
Eight strings moved into the existing `[prompt]` table, joining the six sleep
prompts already there. The operating-mode directives were duplicated verbatim
between `final_reminder.rs` and `run.rs` with a comment asking readers to keep
them in sync; there is now one source feeding both. Doctrine held — only
phrasing moved; schemas, contract rendering, importance ordering, and every
validation rail stay in code, so an override can change tone but never break
parsing. Defaults are byte-identical to the strings they replaced.

Deliberately excluded: the fact extractor's `intro` / `guidance` /
`self_clause` (~2400 chars welded to `FACT_SCHEMA`, several rules with no
code-level rail — needs its own override-boundary tests), and the remaining
`Tool::new` schema descriptions, which are tool-calling API surface rather than
persona context.

### 180 — time format hygiene
The store was already clean: all 22 write paths go through `iso_utc`. The
damage was at the LLM-facing edges, and **every one of these bugs was invisible
under the default `display_tz = "UTC"`**, so the tests use a non-UTC zone.

- `set_alarm` demands RFC-3339 with a numeric offset, but the model was only
  ever shown a zone *abbreviation* — it had to guess that "PDT" means `-07:00`.
  The reminder clock now carries the offset.
- A date-only `valid_from` from the extractor was anchored to midnight **UTC**,
  though a bare `2026-08-16` from the model means local midnight — silently
  wrong by up to ±14h. Now anchored in `display_tz`, stepping past DST gaps.
- The recent-history window now marks local day changes, only when it actually
  spans more than one day, so token cost stays flat in the common case.
- Schedule refusals name their zone; one log line moved to `iso_utc`.

Not done: a window lying entirely on a previous local day still gets no date
marker, since there is only one distinct date to mark. Closing that needs `now`
plumbed into `recent_messages`.

### 183 — token-count calibration
True counts already arrived from OpenRouter and were already logged beside the
estimate. They now feed a per-model calibration store — keyed by model, since
chars-per-token is a tokenizer property — holding a ratio of running totals
rather than an EWMA, so it is exact from the first sample with no decay
constant to tune. The ratio is surfaced on the `[LLM call]` line and applied to
the prompt-assembly budget: `AssemblyContext` now carries the target model,
both responders set it from `LlmClient::model()`, and every layer trim
estimates through it. Truncation inverts the ratio, so a truncated section
still measures under its own cap.

`LlmClient::model()` defaults to `""` — a calibration miss, hence the raw
estimate — so every test double and every non-OpenRouter client keeps the
previous behavior untouched. The one decorator in the composition root
forwards it explicitly; inheriting the default there would have disabled
calibration for all responders without a symptom.

**The safety asymmetry is preserved and is the load-bearing design decision.**
The estimate drives client-side trimming *before* a request is sent, so
over-counting merely drops a little extra context while under-counting risks an
oversized request. Calibration may therefore only ever revise an estimate
upward: a ratio at or below 1.0 is discarded, and one above it is capped, since
image blocks contribute no characters but real billed tokens and a single
degenerate sample would otherwise over-trim everything.

### 203 / 205 — the two investigations
**#203 (wrong channel replies):** the mechanism the issue guesses at — replies
following global focus instead of the originating channel — was real, but is
already fixed and has a concurrency regression test. Two other causes of the
same user-visible symptom were found and fixed instead: N6 (alarms never
delivered at all) and N7 (a wake and a real message for one channel were
different router sessions, so neither preempted the other and both could
reply).

**#205 (multi-party identity confusion):** most likely the same root cause as
#199. With the SSRC→user binding lost, every speaker got a synthetic id, and a
mid-session SSRC change fragmented one person into several identities — exactly
the symptom. The issue also suspects "bad data being fed in by the app", and
that is what this was. **This needs live confirmation before closing**; the fix
is in the voice path, which cannot be exercised without a real Discord call.

## Not attempted, and why

These are recorded rather than half-done. Each is either genuinely large, or
turns on a decision that is the author's to make.

- **#196 (ten-vad-sys FFI crate)** — a new `-sys` crate with vendored C, the one
  place the workspace's `unsafe_code = "forbid"` would be relaxed. Local turn
  detection keeps degrading gracefully to Deepgram meanwhile. `TenVad::new`
  always returns `MissingBackend` today.
- **#200 (break out character.toml)** — the file is 546 lines. Investigated: the
  existing multi-file support is precedent-by-copy-paste, not a reusable
  abstraction (`character.toml`, `activities.toml` and `character.md` have three
  different loaders and three different fallback semantics, with `deep_merge`
  duplicated). Each new file needs its own loader, its own defaults-path wiring,
  and its own slice of a ~1200-line validator, and 150 tests assume one file.
  The cheapest real win — getting prose out of a config file — is largely what
  #151 just did.
- **#184 (catch-up profiling)** — wants real pilot data, not a code change.
  #183's `cal_ratio` supplies item 3 on its list.
- **#181 (tool call context requirements)** — explicitly framed by the author as
  an architectural/hygiene ticket not needed by current pilot bots.
- **#155 (design guidelines from eval trials)** — the issue is itself a design
  exploration, and says so ("big design task", "so much design work slated").
- **#204 (image processing strategy)** — a genuine fork in the road: keep the
  separate transcriber or use model multimodal capability. #218 now supplies the
  OpenRouter metadata that would drive the automatic selection it proposes, so
  the blocker is the decision, not the plumbing.
- **#206 (staged passes and cache optimization)** — the issue demands "detailed
  and comprehensive profiling to justify design decisions" before any change.
- **#207 (Turso vector search)** — a storage-backend migration.
- **#96 (parallel database access)** — the author is explicitly waiting on a
  Turso release.
- **#130 (memory-hardening follow-ups)** — the issue's own recommendation is
  "build none now — fix-if-it-bites", with item 4 to revisit once the
  self-dossier has run in production. Honoring that.

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

### N2 — Deepgram connect URL with member names is logged at INFO `[fixed]`

Up to 100 voice-member display names, usernames, aliases, and guild nicks are
appended to the Deepgram websocket URL as `keyterm=` recognition hints
(`src/bot.rs:916-927` → `src/bot.rs:2400-2404` → `src/stt/deepgram.rs:461-497`).
That full URL was then logged at INFO, so every voice session wrote its
participant roster into the logs.

Fixed on this branch: `DeepgramTranscriber::redacted_ws_url` renders the same
URL with the keyterm values replaced by a count, keeping the tunables that make
the line worth logging. The connect log uses it.

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

### N10 — stale Python-era symbol name in docs `[fixed]`

`docs/architecture/context-pipeline.md` referred to `_is_self_capability`; the
Rust symbol is `is_self_capability`. Corrected alongside #153.

Two other Python-era leftovers were cleared earlier on this branch: the tracked
`.audit-fixes.md` (a scratch brief for the Python ancestor's activities engine,
citing `uv run`, `pytest`, `waker.py`, and a dangling `AGENTS.md`, whose own
header said it was not meant to be committed) and the `ruff` / pydocstyle
asides in the Cargo.toml lint comments.

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
