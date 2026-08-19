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
| 222 | Familiars don't report focused channel correctly | bounded bug | fixed — first pass missed the root cause, see below |
| 199 | High volume of decryption failures | investigation | fixed — and a worse defect found beside it |
| 214 | Add Privacy Policy and Terms of Use | bounded docs | fixed |
| 218 | Detect issues with model configuration via diagnostics | bounded feature | fixed |
| 153 | Reduce strength of Regex rules for prose formatting | bounded feature | fixed |
| 151 | Relocate hard-coded context to default/instance pattern | bounded feature | fixed, minus two deliberate exclusions |
| 180 | Time format hygiene | medium feature | LLM-facing edges fixed; store was already clean |
| 183 | Improve token-counting heuristic | bounded feature | fixed — calibration learned, reported, and enforced on the assembly budget |
| 203 | Wrong channel replies | investigation | reported mechanism already fixed; two real causes found and fixed |
| 205 | Bad multi-party identity disambiguation | investigation | **confirmed fixed by live data** — see below |
| 196 | ten-vad-sys FFI crate | large | not attempted |
| 200 | Break out character.toml by file | large | not attempted; cheapest slice already taken by #151 |
| 184 | Catch-up profiling for default config tuning | large | not attempted; #183 supplies one of its inputs |
| 181 | Tool call context requirements | architectural | not attempted |
| 155 | Adhere to design guidelines from eval trials | architectural | not attempted |
| 204 | Decide on image processing strategy | design decision | decided (Option D) and implemented — and a live cost defect found beside it |
| 206 | Research staged passes and cache optimization | research | measurement capability landed; design change deliberately deferred pending data |
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
completion. `diagnose`'s span regex does not scrape this line (no `span=` key);
the #206 `[LLM call]` pass added later does, and counts the status vocabulary
open-endedly, so `silent` / `suppressed` appear in its status column without
further change.

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
**The first pass missed the root cause.** It fixed real gaps — `/subscribe-text`
and `/subscribe-voice` never recorded channel or guild names, a historyless DM
had no peer name, and a miss rendered a bare snowflake instead of
`unnamed channel (id N)` — but it left the reporter's actual symptom in place:
presence still read `unnamed channel (id …)` for an ordinary, long-subscribed
guild channel.

The real cause is that **the `on_ready` snapshot was always empty**. The
serenity `ready` handler built `channel_names` / `guild_names` from
`ctx.cache.guild(...)` for each entry in `ready.guilds`, but Discord's READY
payload lists guilds only as unavailable stubs (`{id, unavailable: true}`) —
full guild data follows as a burst of `GUILD_CREATE` events. Worse, serenity's
own cache update for READY runs *before* the handler is dispatched and
explicitly removes every guild in `ready.guilds` from `cache.guilds`
(`CacheUpdate for ReadyEvent`), so the lookup returned `None` for every guild,
on a cold start *and* on a reconnect. With no `guild_create`, `channel_update`,
or `guild_update` handler anywhere, nothing ever filled the caches, and every
guild channel rendered via the fallback for the whole session.

Fixed by adding `guild_create` (plus `channel_update` / `guild_update`, see
N14) feeding a new serenity-free seam, `BotEvents::on_guild_available(GuildInfo)`,
which reuses `record_channel_naming`. Because focus is seeded at boot — before
any name is known — and `sync_presence` runs then, a naming batch also re-syncs
presence, but only when the focused channel's rendered label actually changed
(one presence update per batch; Discord rate-limits presence). Without that
re-sync the status would stay wrong until the next focus shift and the bug
would look unfixed. The re-sync keeps the B-PR24 ordering: an in-flight
activity re-asserts its away presence afterwards, so a naming batch cannot
clobber an idle/dnd status with the online focus line.

The `ready` snapshot is kept, now sourced from `ctx.cache.guilds()` rather than
the READY stub list, so it covers a warm reconnect instead of being dead code.

A gateway backfill at the point of the miss was considered and rejected:
`FocusManager` is feature-agnostic core with no cache access, and resolving an
uncached channel is a REST round-trip on the presence path.

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

### 206 — prompt-cache measurement, not a redesign
The issue asks for profiling before any staged-pass or breakpoint change, so
only the profiling landed. `diagnose` gained a second aggregation over
`[LLM call]` lines (`diagnostics::llm_calls`), printed as extra tables after
the span table and only when such lines are present — a span-only log renders
byte-identically to before, which is pinned by a test. The hypothesis it is
built to test, and the reading guide, are under
[Not attempted, and why](#not-attempted-and-why) below and in
`docs/architecture/tuning.md` § Measuring prompt-cache behaviour. No prompt
assembly, layer order, cache breakpoint or worker cadence changed.

### 204 — image strategy: native multimodal, plus a persisted caption

The issue proposed replacing the separate transcriber with the model's own
multimodal capability, auto-enabled from OpenRouter metadata. The owner chose
**Option D**: do that, *and* keep a persisted caption, because the image itself
does not survive the turn.

That last point is the whole reason the transcriber cannot simply be deleted.
`turn_to_message_with_context` collapses every `role="tool"` turn to
`[tool result] <text>` on replay, unconditionally. So a `view_image` result is
seen as a picture exactly once, on the turn it was fetched, and the description
string is the only trace of it that the fact extractor, rolling summaries,
people dossiers, and RAG will ever read. Sending the image natively fixes
perception; it does nothing for memory.

**Four decisions, all implemented.**

1. **The two roles are split.** `[llm].image_description_model` was doing two
   unrelated jobs — *substitution* (letting a blind model perceive the image)
   and *persistence* (a durable artifact for memory). Substitution is needed
   only when the slot lacks vision; persistence is needed always. They are now
   two keys: `image_description_model` (substitution, quality matters — it is
   the model's only perception) and the new `[llm].image_caption_model`
   (persistence, pick something cheap). Two keys rather than one because an
   operator would genuinely choose *different models* for them; and each falls
   back to the other, so a profile setting only one keeps working. `view_image`
   still runs exactly **one** describe leg per image either way and picks the
   model by role.
2. **The caption comes from a dedicated model, never self-captioning.**
   Self-captioning by the primary was considered and rejected: it depends on
   prompt adherence and would need the caption fished back out of
   conversational prose — too fragile for something feeding fact extraction.
   The existing `describe_leg` machinery is reused unchanged.
3. **The catalog is cached to disk and refreshed in the background.** This is
   what makes metadata-driven selection affordable. Boot reads the last known
   good `GET /models` from `~/.cache/familiar-connect/openrouter-models.json`
   (platform cache dir, not the familiars root — the file is regenerable, not
   state) and resolves capabilities from it synchronously; there is no URL in
   that code path. The refresh rides the same detached task as the #218 audit
   and takes effect on the *next* start, so the "never gates readiness"
   invariant survives untouched. TTL 24 h, and age gates the refresh, never the
   read — a month-old cache still beats nothing. A missing, unreadable, or
   corrupt file reads as absent, so a first-ever run with no cache and no
   network behaves exactly as before.
4. **`multimodal` is tri-state.** It was `bool` defaulting `false`, so an
   explicit `false` was indistinguishable from silence and auto-detection would
   have quietly reversed a deliberate cost decision. It is now `Option<bool>`:
   omitted = auto-detect, `true`/`false` = an override detection never
   contradicts. `image_tools` deliberately stays a plain `bool` — it is an
   intent knob, not a capability fact (`view_image` works on a text-only model,
   and a vision model's operator may still not want image fetches), so neither
   direction is inferable from the catalog. `[llm.<slot>]` runs no unknown-key
   check, so the schema change breaks no profile.

**Net effect.** Vision-capable primary with a caption model configured: one
cheap describe call for memory, image sent natively, substitution model never
touched. Non-vision primary: one call, serving both roles — today's behaviour
exactly. The startup capability audit also stopped advising on `multimodal`
(auto-detection answers it now) and advises on `image_tools` instead.

Found and fixed on the way: **N15**, below.

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
- **#204 (image processing strategy)** — was listed here as "needs a product
  call". The call has since been made (Option D), so it moved to *What landed*.
- **#206 (staged passes and cache optimization)** — the issue demands "detailed
  and comprehensive profiling to justify design decisions" before any change,
  so the profiling landed and the design change did not. `diagnose` now
  aggregates `[LLM call]` lines alongside the span table: per `(slot, model)`
  cache-hit rate both by call and token-weighted, `ttfb_ms`/`ttft_ms`
  percentiles split by hit versus miss, mean prompt size, and observed
  token-estimator accuracy. Prompt assembly, layer order, cache breakpoints and
  worker cadences are untouched.

  The hypothesis it exists to test: the seven system-prompt layers (character
  card, operating mode, lorebook, conversation summary, reflections, people
  dossier, RAG context) join into one string and become a single system
  message, and for `anthropic/*` models `mark_system_cache_breakpoint` stamps
  `cache_control` on that message's last content block — the whole prompt.
  Anthropic gives no partial credit inside a marked block, and the tail of that
  block changes every turn: `rag_context` is the last layer and its cue is the
  literal current utterance, plus the voice head reminder carries a clock line.
  If so, `slot=fast` pays full price every turn. The text path (`slot=prose`,
  GLM) sets no `cache_control` at all and leans on the provider's automatic
  prefix caching, which does award partial credit — so the two slots side by
  side separate "the breakpoint mechanism is wrong" from "the layer order is
  wrong". Reading guide: `docs/architecture/tuning.md` § Measuring prompt-cache
  behaviour.
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

### N1 — shipped default selected an unimplemented TTS backend `[fixed]`

`data/familiars/_default/character.toml` and the Rust default both set
`[tts].provider = "azure"`. Both the Azure and Gemini backends were stubs that
returned "backend not wired (deferred to wiring layer 10)" at synthesis time.
Cartesia is the only functional provider. `.env.example` and the docs described
Azure as the default and told the user to set `AZURE_SPEECH_KEY`.

Net effect: a fresh install that followed the docs had no working TTS, and the
failure surfaced mid-conversation on the first synthesis rather than at
startup. Enabling the `azure-tts` feature did not help — the backend was
unwired regardless of the flag.

Fixed in two steps. First the shipped default moved to `cartesia` in both
places (`TTSConfig::default()` and the `parse_tts_config` fallback), matching
the `cartesia_voice_id` / `cartesia_model` the profile already carried, and a
startup refusal guarded the unwired providers.

Then the stubs were **removed outright** rather than left guarded: the
`AzureTTSClient` / `GeminiTTSClient` types, their backend seam traits, their
`TtsClientKind` variants, the `azure-tts` feature, the never-referenced
`azure-speech` dependency, and the `azure_voice` / `gemini_*` config keys are
all gone, and with them the startup refusal that only existed to guard them —
`[tts].provider` now accepts `cartesia` alone. A profile still naming a removed
provider fails config validation with `[tts].provider '<name>' is no longer
supported`, so the breakage is loud and names the fix. The `TtsClient` /
`StreamingTtsClient` seam is untouched: adding a real backend is still one new
type (roadmap V4).

### N2 — Deepgram connect URL with member names is logged at INFO `[fixed]`

Up to 100 voice-member display names, usernames, aliases, and guild nicks are
appended to the Deepgram websocket URL as `keyterm=` recognition hints
(`src/bot.rs:916-927` → `src/bot.rs:2400-2404` → `src/stt/deepgram.rs:461-497`).
That full URL was then logged at INFO, so every voice session wrote its
participant roster into the logs.

Fixed on this branch: `DeepgramTranscriber::redacted_ws_url` renders the same
URL with the keyterm values replaced by a count, keeping the tunables that make
the line worth logging. The connect log uses it.

### N3 — `view_image` fetches attacker-chosen hosts `[fixed]`

The original write-up here said the model passes a URL. It does not, and the
correction matters: `view_image_handler` takes an `image_id` (`img_0`, `img_1`,
…) and looks the URL up in `ctx.images`, a map the *bot* built. So the reachable
set is not "whatever the model invents" — it is whatever landed in that map.
`collect_images` (`src/bot.rs`) fills it from three sources: message
attachments (Discord-hosted), embeds preferring Discord's re-hosted `proxy_url`
(Discord-hosted), and `IMAGE_URL_RE.find_iter(content)` — a regex scrape of
inline image URLs out of the message text. **The third source is the hole**: any
user in the channel can paste `https://attacker.example/x.png`, it becomes
`img_N`, and if the model views it the bot GETs it. That discloses the
operator's IP and reaches whatever the host is, including addresses only the
operator's machine can route to. The content-type check was applied *after* the
response arrived, so loopback / link-local / RFC1918 could be probed by timing
even when the fetch was then rejected.

Measured against the 11,615-turn history DB before fixing: only 13 turns
contain an inline image URL at all, across exactly three hosts —
`cdn.discordapp.com` (16 URLs), `media.discordapp.net` (2), and
`64.media.tumblr.com` (2). Default-deny therefore costs essentially nothing in
practice.

Fixed on this branch. `tools::image_policy::UrlGuard` gates every fetch at the
fetch boundary — not at `collect_images` — so all three sources, and any fourth
added later, face one check. Two rules, applied before a socket opens:
non-http(s) schemes and non-public resolved addresses are refused with no
config escape (the check runs on resolved IPs, so a name pointing at
`127.0.0.1` is refused too); and `[tools].trusted_image_hosts` is a
default-deny allowlist, bypassable with `[tools].allow_untrusted_image_urls =
true` — which still cannot reach a private address. Redirects are followed by
hand so each hop re-enters the guard, and the connection is pinned to the
validated addresses against rebinding. See
[Security](docs/architecture/security.md#outbound-image-fetches-view_image).

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

### N14 — a channel rename is invisible until the next restart `[fixed]`

Found while fixing #222. The focus name caches (`channel_names` / `guild_names`)
were write-only pushes from discovery points that all fire once — the (empty)
`on_ready` snapshot, both `/subscribe-*` handlers, `register_dm_channel`, and
boot-time `rehydrate_dm_naming`. With no `ChannelUpdate` / `GuildUpdate`
handler, renaming a channel or server mid-session left presence, logs, and the
model's focus line showing the *old* name until the process restarted.

Closed with the second #222 pass: `EventHandler::channel_update` and
`guild_update` now feed the same `on_guild_available` seam as `guild_create`,
so a rename updates the caches and re-syncs presence when it touches the
focused channel.

### N15 — a vision-capable slot paid twice for every image `[fixed]`

Found while deciding #204. `view_image_handler` called the description model
whenever `ctx.description_llm.is_some()`, with no reference to the calling
model's capability. Nothing auto-disabled the transcriber.

So a slot configured with both `[llm].image_description_model` and
`multimodal = true` — the configuration the docs actively recommended for a
vision model — paid for a full second vision call *and* shipped the inline
image tokens, on every single view. The extra call bought nothing the model
could not already see. The two costs were invisible to each other: the tool
never knew the slot was multimodal, and the slot never knew a transcriber was
wired.

Fixed by the #204 role split. The handler now reads `ctx.multimodal` and runs
one leg: the caption model for a slot that can see (memory only), the
substitution model for one that cannot (perception, doubling as memory). The
substitution model is never invoked for a multimodal slot.

### N16 — the bot never sees who was already in the voice call `[fixed]`

Found in a live call. `BotHandle.voice_members` — the cache `resolve_member`
reads to name a speaker — was populated *only* by `on_voice_state_update`,
which by definition fires on changes. `/subscribe-voice` never enumerated the
channel, so every member already seated when the familiar joined was invisible
to it for the whole session: no name on their turns, and no contribution to the
STT keyterms that bias transcription toward the people actually present.

The call that surfaced it: 25% of user turns landed with both
`author_display_name` and `author_user_id` NULL, all from people who were in
the channel before the bot. (The SSRC layer was already fixed by #199/#205 —
`unmapped_frames=0` across 8,307 speaking frames — so this was the only
remaining cause.)

Fixed by snapshotting the channel's occupants from the gateway cache
(`Guild::voice_states`) into the roster at join, resolving each through
`VoiceState::member` → `Guild::members` → the user cache and skipping anyone
none of them knows (no `GUILD_MEMBERS` intent, and no REST on the audio path).

Two smaller pieces landed with it. The persisted turn now keeps the numeric
user id when name resolution misses, instead of dropping the author entirely
and fusing every unresolved speaker into one anonymous voice — rare after the
snapshot, but someone can still speak in the gap between joining and their
voice-state event landing. And the roster now carries the per-guild nickname,
which the keyterm list already expected.

### N17 — the voice-member roster only ever grew `[fixed]`

Same call. `on_voice_state_update` was insert-only: a leave
(`after_channel_id == None`) and a move to another channel both returned before
any removal, and nothing cleared the map on unsubscribe or shutdown. Members
therefore accumulated for the process's lifetime — the familiar could not
detect a departure, kept resolving people who had left, and biased STT keyterms
toward names nobody in the channel was going to say.

Fixed in the same handler: a join or move into the subscribed channel inserts,
anything else removes by member id (a member who was never in the roster makes
that a no-op, so no before-channel is needed), and `/unsubscribe-voice` clears
the roster outright. Bots are now excluded on update as well as at snapshot —
the guard previously covered only the familiar's own id.

Keyterms are baked into the Deepgram connect URL when a speaker's stream opens,
so a departed member's name lingers in streams already open; those close after
`idle_close_s` and reopen against the current roster.

## Live verification — 2026-08-18 voice call

Log archived at `~/.local/share/familiar-connect/logs/2026-08-18-voice-call.log`
(3,184 lines, 54 LLM calls, ~5 speakers). First real data on this branch.

**#199 / #205 — confirmed fixed.** `unmapped_frames=0` across 8,307 speaking
frames for the whole session; it was equal to `speaking_frames` (100% unmapped)
before. Zero `provisional-id` warnings, zero `RTCP decryption failed` lines. So
the `join_voice` registration race really was the root cause of #205's identity
confusion, and quieting the RTCP noise worked.

**#220 — confirmed live.** The `fast` slot reported `cancelled=8 ok=15
silent=5`: deliberate silences are now distinguishable from barge-ins in
production, which is exactly what the fix was for.

**#206 — first real measurement, and it changes the analysis.** The earlier
brief assumed the shipped `_default`, which puts Claude Haiku on `fast`. The
live profile runs `z-ai/glm-5v-turbo` and `z-ai/glm-5.2` — *both* z-ai — so the
Anthropic single-`cache_control`-block theory does not apply to this
deployment, and the proposed breakpoint split would have bought nothing.

The hypothesis still holds in a sharper form. Hit *rate* is high (fast 100%,
prose 88.5%) but token-weighted reuse is low (fast 23.3%, prose 17.5%) — around
four fifths of the prefix is invalidated every turn. Prose `cost_p50` is
1,333 ms between hits and misses, though over only 3 misses.

Separately and not about caching: `fast` TTFB p50 is 2,837 ms (p95 7,974 ms)
against a documented sub-second target. Worth its own investigation.

**#183 — the documented assumption was false.** Observed
`in_tokens / est_in_tokens` was **1.453** for glm-5.2 (n=26) and **1.197** for
glm-5v-turbo (n=15). `budget.rs` had documented `len/4` as deliberately
*over*-counting, "safer for budgets"; for both live models it under-counts by
20–45%, so `TierBudget` caps were effectively that much larger than configured
— a tuning-contract violation that would have silently invalidated #184's
numbers. The upward-only clamp added earlier is vindicated: this is precisely
the case it exists for. Calibration is now persisted across restarts so a cold
start is no longer blind, and the false claim is corrected in the docs.
