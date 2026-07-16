# Familiar-Connect Rust port — DESIGN (the porting contract)

This is the shared working contract for the ~11 porting agents. Each agent sees
its own subsystem spec (`rust/specs/NN-*.md`) plus this document. Read sections 4
(conventions) and 5 (decision log) before writing any code: they resolve the
choices that would otherwise be made divergently across subsystems.

Status of the scaffold this accompanies: `cargo build`, `cargo clippy
--all-targets`, and `cargo fmt --check` are clean on default features. Every
dependency version in the manifest resolves. `discord`, `discord-voice`,
`stt-deepgram`, `twitch`, and `azure-tts` also compile clean. Modules are stubs
(doc comment only) — no behavior is implemented.

---

## 1. Layout decision

**A Cargo workspace under `rust/` with one main crate `familiar-connect`
(lib + bin), whose modules mirror the Python package layout. FFI-bearing code
goes into separate `-sys` crates as it lands (only `ten-vad-sys` is foreseen).**

This confirms the stated prior. Rationale:

- **One crate dissolves the Python cycle-break hacks for free.** Rust permits
  item-level module cycles within a crate, so the two Python circular-import
  dodges evaporate as *structural* problems: `budget.py`'s placement at package
  root (so `config` could import `TierBudget` without dragging in `context`) and
  the `config → context` risk both disappear — `budget` is simply a leaf module
  that names `crate::llm::Message`. No crate-dependency ordering to satisfy.
- **We still inject the one genuinely upward edge, per spec 02's recommendation.**
  Config validation reaches *up* into registries owned by 04 (`known_embedders`)
  and 07 (`known_projectors`). Even though a single crate would let this be a
  plain module cycle, we make it an explicit parameter instead:
  `load_character_config(..., known_projectors: &BTreeSet<String>, known_embedders:
  &BTreeSet<String>)`. Injection is testable, keeps `config` a near-leaf, and
  matches the spec's stated preference. Do **not** transliterate the Python lazy
  import.
- **Disjoint edit surfaces for parallel agents.** Subsystem = module subtree.
  Agents edit non-overlapping files; a workspace-of-libs would force premature
  `pub` API boundaries between crates, but the specs deliberately defer many of
  those (dozens of `TYPE_CHECKING`/`Any` seams become trait objects). One crate
  lets a trait live anywhere and be referenced across modules without inter-crate
  version/feature juggling.
- **FFI isolation keeps `unsafe_code = "forbid"`.** The template forbids unsafe.
  The only FFI the port needs is TEN-VAD's C API (spec 09, `local-turn`). When it
  lands it becomes a `ten-vad-sys` workspace member that relaxes `[lints.rust]`
  to `unsafe_code = "deny"` with targeted `#[allow]`s at the FFI boundary; the
  main crate stays `forbid`. The workspace is set up now (`members =
  ["familiar-connect"]`) so adding `ten-vad-sys` is a one-line change.

Parity impact: none. This is a structural decision, invisible to behavior.

---

## 2. Module map

Every Python module → Rust module path → owning subsystem → port layer (§3).
`crate::` root is `familiar-connect/src/`. Directory modules use `mod.rs`.

| Python module | Rust module | Subsys | Layer |
|---|---|---|---|
| `log_style.py` | `log_style` | 01 | 0 |
| `macros.py` | `macros` | 01 | 0 |
| `bus/__init__.py` | `bus` (`bus/mod.rs`) | 01 | 0 |
| `bus/envelope.py` | `bus::envelope` | 01 | 0 |
| `bus/protocols.py` | `bus::protocols` | 01 | 0 |
| `bus/router.py` | `bus::router` | 01 | 0 |
| `bus/topics.py` | `bus::topics` | 01 | 0 |
| `bus/bus.py` | `bus::in_process` *(renamed — see D-R1)* | 01 | 0 |
| `diagnostics/__init__.py` | `diagnostics` | 01 | 0 |
| `diagnostics/spans.py` | `diagnostics::spans` | 01 | 0 |
| `diagnostics/collector.py` | `diagnostics::collector` | 01 | 0 |
| `diagnostics/voice_budget.py` | `diagnostics::voice_budget` | 01 | 0 |
| `diagnostics/cold_cache.py` | `diagnostics::cold_cache` | 01 | 0 |
| `diagnostics/report.py` | `diagnostics::report` | 01 | 0 |
| `structured_output.py` | `structured_output` | 02 | 0 |
| `prompt_fill.py` | `prompt_fill` | 02 | 0 |
| `subscriptions.py` | `subscriptions` | 02 | 0 |
| `identity.py` | `identity` | 02 | 1 |
| `config.py` | `config` | 02 | 1 |
| `familiar.py` | `familiar` | 02 | 3 |
| `history/__init__.py` | `history` | 03 | 1 |
| `history/turso_compat.py` | `history::db` *(reshaped to a DB actor — D-R2)* | 03 | 1 |
| `history/store.py` | `history::store` | 03 | 1 |
| `history/fts.py` | `history::fts` | 03 | 1 |
| `history/async_store.py` | `history::async_store` | 03 | 1 |
| `embedding/__init__.py` | `embedding` | 04 | 1 |
| `embedding/protocol.py` | `embedding::protocol` | 04 | 1 |
| `embedding/hash.py` | `embedding::hash` | 04 | 1 |
| `embedding/factory.py` | `embedding::factory` | 04 | 1 |
| `embedding/fastembed.py` | `embedding::fastembed` *(feat `local-embed`)* | 04 | 2 |
| `sleep/__init__.py` | `sleep` | 04 | 2 |
| `sleep/consolidation.py` | `sleep::consolidation` | 04 | 2 |
| `sleep/apply.py` | `sleep::apply` | 04 | 2 |
| `sleep/opinion_formation.py` | `sleep::opinion_formation` | 04 | 2 |
| `sleep/maintenance.py` | `sleep::maintenance` | 04 | 2 |
| `budget.py` | `budget` | 05 | 1 |
| `context/__init__.py` | `context` | 05 | 2 |
| `context/assembler.py` | `context::assembler` | 05 | 2 |
| `context/layers.py` | `context::layers` | 05 | 2 |
| `context/final_reminder.py` | `context::final_reminder` | 05 | 2 |
| `focus.py` | `focus` | 05 | 2 |
| `processors/__init__.py` | `processors` | 06 | 3 |
| `processors/text_responder.py` | `processors::text_responder` | 06 | 3 |
| `processors/voice_responder.py` | `processors::voice_responder` | 06 | 3 |
| `processors/projectors.py` | `processors::projectors` | 06 | 3 |
| `processors/history_writer.py` | `processors::history_writer` | 06 | 3 |
| `processors/debug_logger.py` | `processors::debug_logger` | 06 | 3 |
| `silence.py` | `silence` | 06 | 3 |
| `typing_interrupt.py` | `typing_interrupt` | 06 | 3 |
| `processors/fact_extractor.py` | `processors::fact_extractor` | 07 | 3 |
| `processors/summary_worker.py` | `processors::summary_worker` | 07 | 3 |
| `processors/people_dossier_worker.py` | `processors::people_dossier_worker` | 07 | 3 |
| `processors/reflection_worker.py` | `processors::reflection_worker` | 07 | 3 |
| `processors/fact_embedding_worker.py` | `processors::fact_embedding_worker` | 07 | 3 |
| `processors/fact_supersede_worker.py` | `processors::fact_supersede_worker` | 07 | 3 |
| `llm.py` | `llm` | 08 | 0*/2 (see §3) |
| `structured_request.py` | `structured_request` | 08 | 1 |
| `tools/__init__.py` | `tools` | 08 | 2 |
| `tools/registry.py` | `tools::registry` | 08 | 2 |
| `tools/loop.py` | `tools::agentic` *(renamed — `loop` is a keyword, D-R1)* | 08 | 2 |
| `tools/builtins.py` | `tools::builtins` | 08 | 2 |
| `tools/alarm.py` | `tools::alarm` | 08 | 2 |
| `tools/scheduler.py` | `tools::scheduler` | 08 | 2 |
| `tools/waker.py` | `tools::waker` | 08 | 2 |
| `tools/silent.py` | `tools::silent` | 08 | 2 |
| `tools/shift_focus.py` | `tools::shift_focus` | 08 | 2 |
| `tools/read_channel.py` | `tools::read_channel` | 08 | 2 |
| `tools/channel_view.py` | `tools::channel_view` | 08 | 2 |
| `tools/image.py` | `tools::image` *(feat `images`)* | 08 | 2 |
| `tools/image_describe.py` | `tools::image_describe` | 08 | 2 |
| `tools/image_compress.py` | `tools::image_compress` *(feat `images`)* | 08 | 2 |
| `tools/start_activity.py` | `tools::start_activity` | 08 | 2 |
| `sentence_streamer.py` | `sentence_streamer` | 09 | 1 |
| `voice/__init__.py` | `voice` | 09 | 1/2 |
| `voice/audio.py` | `voice::audio` | 09 | 1 |
| `voice/recording_sink.py` | `voice::recording_sink` | 09 | 2 |
| `voice/dave_client.py` | `voice::dave_client` *(see D8 — likely thinned/removed)* | 09 | 2 |
| `voice/dave_ws.py` | `voice::dave_ws` *(see D8)* | 09 | 2 |
| `voice/turn_detection/__init__.py` | `voice::turn_detection` | 09 | 2 |
| `voice/turn_detection/endpointer.py` | `voice::turn_detection::endpointer` | 09 | 2 |
| `voice/turn_detection/smart_turn.py` | `voice::turn_detection::smart_turn` *(feat `local-turn`)* | 09 | 2 |
| `voice/turn_detection/ten_vad.py` | `voice::turn_detection::ten_vad` *(feat `local-turn`, FFI)* | 09 | 2 |
| `voice/turn_detection/factory.py` | `voice::turn_detection::factory` | 09 | 2 |
| `stt/__init__.py` | `stt` | 09 | 2 |
| `stt/protocol.py` | `stt::protocol` | 09 | 2 |
| `stt/factory.py` | `stt::factory` | 09 | 2 |
| `stt/deepgram.py` | `stt::deepgram` *(feat `stt-deepgram`)* | 09 | 2 |
| `stt/parakeet.py` | `stt::parakeet` *(feat `local-stt`)* | 09 | 2 |
| `stt/faster_whisper.py` | `stt::faster_whisper` *(feat `local-stt`)* | 09 | 2 |
| `tts.py` | `tts` | 09 | 2 |
| `tts_player/__init__.py` | `tts_player` | 09 | 2 |
| `tts_player/protocol.py` | `tts_player::protocol` | 09 | 2 |
| `tts_player/discord_player.py` | `tts_player::discord_player` | 09 | 2 |
| `tts_player/logging_player.py` | `tts_player::logging_player` | 09 | 2 |
| `tts_player/mock.py` | `tts_player::mock` | 09 | 2 |
| `cli.py` | `cli` | 10 | 4 |
| `bot.py` | `bot` | 10 | 4 |
| `commands/__init__.py` | `commands` | 10 | 4 |
| `commands/run.py` | `commands::run` | 10 | 4 |
| `commands/diagnose.py` | `commands::diagnose` | 10 | 4 |
| `commands/version.py` | `commands::version` | 10 | 4 |
| `commands/example.py` | — *(not ported: template, D-R3)* | 10 | — |
| `sources/__init__.py` | `sources` | 10 | 4 |
| `sources/discord_text.py` | `sources::discord_text` | 10 | 4 |
| `sources/discord_embed_text.py` | `sources::discord_embed_text` | 10 | 4 |
| `sources/voice.py` | `sources::voice` | 09/10 | 3 |
| `sources/twitch.py` | `sources::twitch` *(feat `twitch`)* | 11/10 | 3 |
| `twitch.py` | `twitch` | 11 | 2 |
| `twitch_watcher.py` | `twitch_watcher` *(feat `twitch`)* | 11 | 3 |
| `activities/__init__.py` | `activities` | 11 | 3 |
| `activities/config.py` | `activities::config` | 11 | 3 |
| `activities/engine.py` | `activities::engine` | 11 | 3 |
| `__main__.py` | folded into `src/main.rs` | 10 | 4 |
| `__init__.py` (root) | `src/lib.rs` | — | — |
| *(no Python source)* | `support::{time,round,text}` | shared | 0 |

---

## 3. Port order (topological layers)

Derived from the specs' "Dependency edges". Modules **within a layer are
independently portable**; each layer may depend only on lower layers. This is a
build-order guide, not a hard partition — a stub of a higher layer's *type* can
be filled first when only a signature is needed.

- **Layer 0 — foundation (no internal subsystem deps).** `log_style`, `macros`,
  all of `bus::*`, all of `diagnostics::*` (dep: `log_style` only),
  `structured_output`, `prompt_fill`, `subscriptions`, `support::{time,round,text}`.
  Plus the **core value types** of `llm` — `Message`, `LLMDelta`,
  `sanitize_name` — which `identity`/`budget` need; the `LLMClient` transport
  half of `llm` is Layer 2. Split `llm.rs` accordingly (types first).
- **Layer 1.** `identity` (dep: `sanitize_name`), `budget` (dep: `Message`),
  `config` (dep: `budget` + injected validator sets), `embedding::{protocol,
  hash, factory}`, `structured_request` (dep: `structured_output` + `LlmClient`
  trait), the whole of `history::*` (dep: `identity`, `support::time`),
  `sentence_streamer`, `voice::audio` (both pure).
- **Layer 2.** `llm` client, `embedding::fastembed`, all of `tools::*`, all of
  `sleep::*`, `context::*` + `focus` + `final_reminder`, all of `stt::*`, `tts`,
  `tts_player::*`, `voice::turn_detection::*`, `voice::{dave_*, recording_sink}`,
  `twitch` (pure formatters).
- **Layer 3.** `processors::*` (06 responders + registry + 07 workers),
  `silence`, `typing_interrupt`, `activities::{config, engine}`, `twitch_watcher`,
  `sources::{voice, twitch}`, and `familiar` (the DI bundle — late because it
  aggregates handles to store/llm/tts/transcriber/bus/router).
- **Layer 4 — composition root.** `sources::{discord_text, discord_embed_text}`,
  `bot`, `cli`, `commands::{run, diagnose, version}`, `main.rs`.

Spec 01 confirms the bus subsystem is Layer 0 (imports nothing); spec 02
confirms `config` becomes Layer-1-clean once the two registry lookups are
injected.

---

## 4. Cross-cutting conventions (resolve ONCE; do not re-decide per subsystem)

### 4.1 Error handling
- One `thiserror::Error` enum per subsystem (e.g. `ConfigError`, `StoreError`,
  `EmbeddingError`, `LlmError`). Reserve `Err` for genuine faults; **reads that
  hit malformed stored data degrade to empty/`None`, never `Err`** (spec 03).
- **Byte-stable error messages are test contracts.** Reproduce these substrings
  exactly (via `#[error("…")]`):
  - `config` (02): `"default character profile"`, `"failed to parse TOML config
    at"`, `"unknown LLM slot"`, `"has unknown keys:"`, `"must be a table"`,
    `"must be a positive integer"`, `"display_tz"`, `"'HH:MM-HH:MM'"`, the
    `window_size` migration text, `"unknown memory projector"`, `"valid options:"`,
    `"typing_backoff_max_s must be >= typing_backoff_initial_s"`.
  - `embedding` (04): `"unknown embedding backend"`, `"local-embed"`.
  - `store` (03) supersede skips: `"unknown fact id={id}"`, `"fact id={id}
    already superseded"`.
  - `activities` (11): `"failed to parse TOML"`, the `"adapter"`-reserved message,
    availability strings (§ spec 11 B-18/20/27).
  - Factory errors (04/09) render the sorted valid-name list; keep the
    `", "`-joined-or-`"(none)"` shape.

### 4.2 Timestamps  (owner: `support::time`)
- Emit ISO-8601 UTC via `dt.format("%Y-%m-%dT%H:%M:%S%.6f+00:00")` —
  **fixed-width 6-digit microseconds, literal `+00:00` (never `Z`)**. Python
  omits `.ffffff` when micros are exactly 0; the Rust port always writes 6
  digits (this is *more* lexicographically stable, not less).
- Parse tolerantly: accept missing microseconds and `Z`. Provide
  `support::time::iso_utc(DateTime<Utc>) -> String` and `parse_iso(&str) ->
  Option<DateTime<Utc>>`. Lexicographic == chronological ordering of these
  strings is a correctness dependency in five `history` query paths (spec 03 §16,
  §28, §50) — every timestamp written to SQLite or JSON goes through `iso_utc`.
- Alarm past-skew, `valid_from = <date>T00:00:00+00:00`, activity daypart, and
  sleep-window math all use `chrono` + `chrono-tz`; reject naive datetimes in
  `set_alarm` (spec 08 §39).

### 4.3 Rounding & integer math  (owner: `support::round`)
- Everywhere Python uses `round()` (banker's / half-to-even): `TierBudget::
  apply_curve` (05/02), `VoiceBudgetRecorder` gap ms (01 §28), fact-importance
  paths — use `support::round::half_even(x) = x.round_ties_even()`. Divergence
  from `f64::round` (half-away-from-zero) is only at exact `.5`; no test hits it,
  but use `round_ties_even` for bit-parity.
- `@span` ms is **truncation toward zero**, not rounding: `(elapsed_s * 1000.0)
  as i64` (spec 01 §20). Keep these distinct.
- Floor division `//`: use `i32::div_euclid` for `stereo_to_mono`,
  `Resampler48to16`, Gemini `(a+b)//2` upsample (spec 09 §17 — byte-pinned).
- `int()` truncation in endpointer thresholds (spec 09 §25): explicit truncating
  cast, not round.
- Bool-is-not-int comes free: `serde_json::Value::Bool` ≠ `Number`, `toml::Value`
  distinguishes them — never add a `bool → int` coercion (specs 02 §8, 07, 08 §39).

### 4.4 Async architecture
- **Runtime:** `#[tokio::main(flavor = "multi_thread")]`. Everything shared
  across tasks is `Send + Sync`.
- **TurnScope = `tokio_util::sync::CancellationToken`** (idempotent `cancel()`,
  level-triggered `cancelled().await`, `is_cancelled()`). Cancellation is
  **cooperative**: check at loop tops / before each `speak` / at documented
  checkpoints. Do **not** blanket-`select!` a token around whole pipelines — the
  responders deliberately run specific sections (persist-after-send,
  `engine.end_turn()`) even when cancelled, and the checkpoint positions are
  test-observable (spec 06). Only the voice per-speaker final task is
  hard-cancelled (`JoinHandle::abort()`).
- **DB access = one dedicated actor** (spec 03). A single task owns the
  `rusqlite::Connection` and receives whole-operation closures/messages over an
  `mpsc` channel; `history::async_store` is the async facade. This honours the
  pyturso single-owning-thread contract, keeps DB work off the reactor, and lets
  multi-statement ops (`supersede`, promotions, `bump_reaction`) run in **explicit
  transactions** — strengthening atomicity the Python never had (safe; no test
  pins the non-atomicity). Calls after `close()` fail with an explicit error.
- **Bus fan-out stays sequential** in subscription-registration order
  (`for sub in subs { sub.put(ev).await }`) — do not parallelize; the BLOCK
  head-of-line coupling is test-pinned (spec 01 §6). Each subscription is its own
  channel: BLOCK → bounded `mpsc`; UNBOUNDED → `unbounded_channel`;
  DROP_OLDEST/DROP_NEWEST → a small `Mutex<VecDeque> + Notify` (or `try_send` +
  evict-retry). Add `Drop` on the subscription handle to fix the no-unsubscribe
  leak *without changing observable semantics* (spec 01 §4, D7).
- **Background workers:** each is one `tokio::task` running `loop { if let Err(e)
  = tick().await { warn!(…) } sleep(interval).await }`; cancellation via a
  `CancellationToken` / `JoinSet` abort in the supervisor. `ReflectionWorker`'s
  watermark-advances-in-`finally` must survive every exit path (write it before
  returning on all paths — mirror the Python `try/finally`, spec 07).
- **LLM rate limit:** an injected `Arc<tokio::sync::Semaphore>` (one per process,
  built at startup from config) — **not** a module global. Preserve the two
  release points exactly: `chat` holds a permit only across the POST (never
  across backoff sleeps); streaming drops the permit the instant response headers
  pass the status check (spec 08 §2–3).

### 4.5 Logging  (owner: `log_style` + a `tracing` layer)
- `tracing` + `tracing-subscriber`, but the emitted **log line is a wire format**
  regex-parsed by the `diagnose` CLI and by the Python build's logs
  (cross-version compat is desired). Implement a custom `FormatEvent` reproducing
  the `log_style` layout exactly:
  - `tag(text,color)` = `W + "[" + color + text + W + "]" + RS`;
    `kv(k,v)` = `kc + k + "=" + RS + vc + v + RS` (the `=` painted in key color,
    `RS` between `=` and value). **Single-parameter SGR only** (`ESC[<n>m`) — use
    raw escape strings, never a color crate that emits compound sequences
    (`ESC[1;33m` breaks both the formatter regex and the diagnose parser).
  - Span line (DEBUG): `[span] span=<name> ms=<int> status=<ok|error>` in that
    order (spec 01 §21, §44). Budget/cold-cache lines per spec 01 § Data formats.
- Two-tier visibility via an `EnvFilter` default `warn,familiar_connect=info`;
  `-vv` flips to `debug` (spec 01 §41). tracing **targets mirror the Python
  logger names** (`familiar_connect.diagnostics`,
  `familiar_connect.diagnostics.voice_budget`, `…cold_cache`, `familiar_connect.
  bus.bus`) so filter directives and the diagnose grep both keep working.
- `@span` → a `time_span!(name, expr)` helper (or `#[tracing::instrument]` +
  a `SpanCollector` layer) that emits the exact DEBUG line **and** records into
  the collector. Port the percentile/aggregation function once
  (`diagnostics::report`) and share it with `commands::diagnose` (Python
  duplicates it).

### 4.6 serde / JSON
- `serde_json` for all dynamic JSON (SSE chunks, tool args/params, reply
  parsing). `Message.content` → `enum Content { Text(String), Blocks(Vec<Value>)
  }` with a `content_str()` method; `to_dict` omits `None` optionals via
  `#[serde(skip_serializing_if = "Option::is_none")]` (spec 08).
- **`subjects_json` (03):** emit **compact** serde_json (no spaces). The ego
  migration `UPDATE` must match **both** the Python-spaced form (`"canonical_key":
  "self:`) and the compact form; the `LIKE` prefilters in `facts_for_subject` are
  whitespace-agnostic already (spec 03 § Data formats, D9). Mixed-format DBs
  (post-port) are expected and read correctly.
- Fact-embedding BLOB: little-endian packed f32 via `byteorder` (spec 03).

### 4.7 IDs / UUIDs
- `uuid::Uuid::new_v4()`. Alarm ids = `.simple()` (32 lowercase hex, no dashes —
  matches `uuid4().hex`). Event ids: `format!("discord-text-{}", &simple[..12])`,
  `format!("voice-{seq:08}")`, wake events use the full `.simple()` hex.
  Sequence numbers are per-source `u64` starting at 1 (source events),
  `sequence_number = 0` for synthetic/alarm/wake events.

### 4.8 Test conventions & injection seams
- Unit tests in-module `#[cfg(test)] mod tests`; integration tests in
  `familiar-connect/tests/`. Timing-sensitive tests use `tokio::time::pause`.
- **Replace every Python monkeypatch with trait-object injection.** Design traits
  minimal so a ~5-line scripted stub satisfies them (the Python doubles are tiny;
  ~90 worker/responder tests depend on this). Named seams the specs call out:
  - **08 `LlmClient` trait** — mocks touch only `chat`, `stream_completion`,
    `multimodal()`, `slot()`, `tool_calling_enabled()` (spec 08). A scripted stub
    pops canned replies per call; the structured-request retry budget must match
    (1 corrective re-ask) so `llm.calls` counts line up (spec 07).
  - **04 `Embedder` trait** — accepts non-registered impls (bare struct with
    `name`/`dim`/`embed`).
  - **03** — an FTS commit hook / fallible-writer seam (Python patches
    `FtsIndex._commit_writer` / `.add`) and a **statement-trace hook** on the DB
    actor for the single-query-count tests (Python `set_trace_callback`).
  - **09** — `Transcriber`, `TTSPlayer`, a VAD trait, the 4-method voice-client
    structural trait (`is_connected/is_playing/play/stop`), and the Deepgram
    `endpointing_ms` poke as a builder/setter (not a duck-typed setattr).
  - **06/05** — `FocusManager` behind a trait, a bus test double, injectable
    clock (`FocusManager` debounce clock, spec 05 §51).
  - **11** — `Clock` trait (`now()` + `sleep_until()`), injectable `rng`
    (`RngCore` + inclusive `gen_range(lo..=hi)`), a faulting store/bus impl for the
    "never raises" hardening tests, presence-cb recorder.
  - **02** — config validators injected as `&BTreeSet<String>` params.
- Process-wide singletons (`SpanCollector`, `VoiceBudgetRecorder`): `OnceLock` /
  `static Mutex<Option<Arc<…>>>`; expose `reset_*` behind
  `#[cfg(any(test, feature = "test-util"))]` (or prefer DI). The LLM semaphore is
  injected, never global.
- **Registries** (embedder factories, projectors, maintenance passes): explicit
  `with_builtins()` builder + `register(name, factory)` — not import-time global
  mutation. Keep `known_*()` sorted and the error strings byte-exact.

### 4.9 Unicode / text  (owner: `support::text`)
- All truncation caps count **Unicode scalars**, not bytes: use `chars().count()`
  / `char_indices` slicing, never byte slicing (emoji-heavy chat lands
  mid-codepoint otherwise). Ellipsis is U+2026, reply glyph U+21A9. Caps: reply
  400/80, bio 240, state-line name 40, wake-line 160, `trunc` default 200,
  chars-per-token 4 (spec 05, 06, 11).

---

## 5. Decision log

`Parity` column: **neutral** = invisible to behavior/tests; **adjacent** =
storage/format nuance, reads stay compatible; **BREAKING** = intended behavior
deviation reviewers must sign off.

| # | Decision | Rationale | Parity |
|---|---|---|---|
| D1 | **Single main crate + workspace** (§1) | Dissolves Python cycle hacks; disjoint agent edit surfaces; defers API boundaries | neutral |
| D2 | **FFI in `ten-vad-sys` later; main crate keeps `unsafe_code = "forbid"`** | Isolate the only foreseen unsafe; template ethos | neutral |
| D3 | **Config validators injected** (`known_projectors`/`known_embedders` as `&BTreeSet`) rather than module cycle or lazy import | Testable, keeps `config` a near-leaf, matches spec 02's stated preference | neutral |
| D4 | **`budget` is a leaf module** naming `crate::llm::Message` (no separate crate) | One crate makes the Python root-placement hack unnecessary | neutral |
| D5 | **Engine = `rusqlite` via a single DB actor** (not `turso`) with explicit transactions for multi-statement ops | Mature/low-risk vs turso beta; the actor gives async + honours single-owner contract; transactions strengthen atomicity (no test pins non-atomicity) | neutral (atomicity is a safe strengthening) |
| D6 | **Dispatcher error contract fixed: log-and-continue** (spec 01 §19) | Resolves the docstring-vs-wiring contradiction; a long-running bot must not unwind on one processor's escaped error | **BREAKING** (intended fix) |
| D7 | **`Drop`-based bus unsubscribe** on the subscription handle (spec 01 §4) | Fixes the unbounded no-unsubscribe leak while keeping observable semantics | neutral |
| D8 | **DAVE (dave_client/dave_ws) is owned by songbird, not ported 1:1** | Ecosystem report: `davey`/DAVE lives inside songbird ≥0.6; spec 09 says reimplement invariants A.1–A.14 as *verification targets* against songbird's hooks. Stub modules kept as a home for residual glue; the 09 agent likely thins/removes them | adjacent (see §6 tension) |
| D9 | **`subjects_json` emitted compact**, ego migration matches spaced+compact | serde_json default; `LIKE` prefilters whitespace-agnostic; mixed-format DBs read fine | adjacent |
| D10 | **`round_ties_even` for Python `round()`; truncation for `@span` ms** | Bit-parity at `.5`; distinct rounding vs truncation semantics preserved | neutral |
| D11 | **Timestamps always 6-digit micros + `+00:00`; parse tolerant** | Lexicographic ordering is a query-correctness dependency; Python's variable width is the wart | adjacent (more stable than Python) |
| D12 | **`TurnScope.started_at` = `Instant::now()` unconditionally** (drop the 0.0 no-loop fallback) | The fallback existed only for loop-less Python tests; adjust the one `started_at > 0` assertion | neutral-ish (one test assertion updated) |
| D13 | **Singletons via `OnceLock` + `cfg(test)` reset; LLM semaphore injected** | Kills global mutable state; keeps the reset seam tests need | neutral |
| D14 | **Registries = explicit builder, not import-time globals** (04/06/07 §factories) | Deterministic construction; `known_*()` + error strings preserved | neutral |
| D15 | **Assembler uses explicit slots, not `isinstance` downcasting** (spec 05) | Recent-history becomes a distinct slot; RAG cue routed via an explicit handle; layer-order pin applies to the system-prompt layer `Vec` | neutral |
| D16 | **`invalidation_key` does not block the reactor** (async or `spawn_blocking`) | Python's sync-store-on-event-loop is a wart, not a contract (spec 05) | neutral |
| D17 | **T15 empty-completion retry: preserve Python's library-default `max_iterations` for now**, flag the "use configured cap" fix for reviewer sign-off (spec 06 §T15) | Deliberate parity default; the fix is a one-line change if approved | neutral (candidate BREAKING if changed) |
| D18 | **Version corrections (see §6):** `rusqlite 0.40.1 → 0.37` (MSRV: bundled `libsqlite3-sys 0.38.1` needs unstable `cfg_select` on rustc 1.94.1); `rubato 4.0.0 → 0.16` (report version does not exist; 0.16.x is latest and unifies with songbird) | Keep the scaffold building on the pinned stable toolchain | neutral |
| D19 | **Discord = serenity 0.12.5 + songbird 0.6.0** (`driver,gateway,receive,serenity,rustls,tungstenite`); rustls everywhere (no OpenSSL) | Ecosystem report pick; both compile clean; songbird owns Opus + DAVE | neutral |
| D20 | **Twitch dormant:** port pure `twitch` (exact formatter strings); `twitch_watcher`/`sources::twitch` behind the `twitch` feature until a consumer exists | Matches production dormancy (spec 11 §12) | neutral |
| D21 | **Renames:** `bus::in_process` (was `bus/bus.py` — avoids `clippy::module_inception`), `tools::agentic` (was `tools/loop.py` — `loop` keyword), `history::db` (was `turso_compat.py` — now a DB actor); `commands/example.py` not ported | Rust keyword/lint/engine realities | neutral |

---

## 6. Dependency manifest

Full `[dependencies]` / `[dev-dependencies]` with the feature that gates each
optional crate. Versions are the ecosystem report's, except the two corrections
noted (D18). The DEFAULT feature set (`store`, `net`, `images`) compiles the
pure-Rust + safe-native foundation with **rustls everywhere (no OpenSSL)**;
native / local-ML crates stay off by default but their versions are still
resolved by `cargo build`.

### Always-compiled (regular deps)

| Crate | Version | Features | Used by |
|---|---|---|---|
| `tokio` | 1 | rt-multi-thread, macros, sync, time, io-util, net, fs, signal, test-util | all |
| `tokio-util` | 0.7.18 | rt, time | CancellationToken = TurnScope (01/06) |
| `futures` | 0.3 | — | Stream/StreamExt (06/08/09) |
| `async-trait` | 0.1 | — | seam traits |
| `serde` | 1 | derive | all serialized types |
| `serde_json` | 1 | — | dynamic JSON (08 etc.) |
| `thiserror` | 2 | — | per-subsystem errors |
| `anyhow` | 1 | — | backend-fault bubbling |
| `tracing` | 0.1 | — | logging/spans (01) |
| `tracing-subscriber` | 0.3.23 | env-filter | log formatter + two-tier filter (01/10) |
| `chrono` | 0.4 | serde | timestamps (all) |
| `chrono-tz` | 0.10 | — | IANA zone validation + local-date bucketing (02/04/05/11) |
| `regex` | 1 | — | many (Unicode `\w`) |
| `uuid` | 1 | v4 | ids (01/08/11) |
| `toml` | 1.1.2 | — | config/lorebook/activities/subscriptions (02/05/11) |
| `dotenvy` | 0.15.7 | — | `.env` (10) |
| `base64` | 0.22 | — | data URIs / TTS chunks (08/09) |
| `sha2` | 0.10 | — | greeting cache keys (09) |
| `blake2` | 0.10 | — | HashEmbedder (04) + layer keys (05) |
| `byteorder` | 1 | — | f32 embedding BLOB (03) |
| `clap` | 4 | derive | CLI (10) |

### Optional (feature-gated)

| Crate | Version | Feature | Features on crate | Notes |
|---|---|---|---|---|
| `rusqlite` | **0.37** (was 0.40.1, D18) | `store` (default) | bundled | SQLite engine (03) |
| `tantivy` | 0.26.1 | `store` (default) | — | FTS indexes (03) |
| `reqwest` | 0.12 | `net` (default) | json, stream, rustls-tls (no default) | LLM SSE + image fetch (08) |
| `eventsource-stream` | 0.2.3 | `net` (default) | — | SSE parse (08) |
| `tokio-tungstenite` | 0.24 | `net` (default) | connect, rustls-tls-webpki-roots (no default) | Deepgram/Cartesia/Twitch WS (09/11) |
| `image` | 0.25.10 | `images` (default) | — | view_image JPEG/GIF (08) |
| `serenity` | 0.12.5 | `discord` | builder, cache, client, gateway, http, model, rustls_backend, voice (no default) | gateway+REST (10) |
| `songbird` | 0.6.0 | `discord-voice` → `discord` | driver, gateway, receive, serenity, rustls, tungstenite (no default) | voice/DAVE (09); owns Opus+MLS |
| `deepgram` | 0.10.0 | `stt-deepgram` → `net` | — | streaming STT (09) |
| `fastembed` | 5.17.2 | `local-embed` | — | ONNX embeddings (04); pulls `ort` |
| `ort` | 2.0.0-rc.12 | `local-turn` | — | Smart Turn ONNX (09); rc — isolate behind the VAD/embedder seams |
| `hf-hub` | 0.3 | `local-turn` | — | weights download (09) |
| `twitch_api` | 0.8.0 | `twitch` | — | EventSub (11, dormant) |
| `azure-speech` | 0.10.0 | `azure-tts` | — | Azure TTS (09); alt path is REST over `net` |
| `rubato` | **0.16** (was 4.0.0, D18) | `audio-resample` | — | resampling (09) |
| `opus` | 0.3.1 | `audio-resample` | — | standalone Opus (09) — **needs system libopus** |

### dev-dependencies

| Crate | Version | Use |
|---|---|---|
| `tempfile` | 3 | tmp-dir fixtures (store/config) |
| `wiremock` | 0.6 | HTTP mock for the LLM client (08) |
| `assert_cmd` | 2 | CLI subprocess tests (10) |
| `predicates` | 3 | CLI assertions |

### Feature map (mirrors Python `[project.optional-dependencies]`)

```
default        = [store, net, images]
store          = rusqlite + tantivy                 # 03 (safe-native: cc/zstd)
net            = reqwest + eventsource-stream + tokio-tungstenite   # 08/09/11 (rustls)
images         = image                              # 08
discord        = serenity                           # 10 (gateway/REST)
discord-voice  = discord + songbird                 # 09 (voice/DAVE; symphonia/opus in-tree)
stt-deepgram   = net + deepgram                     # 09
local-embed    = fastembed                          # 04  (ONNX — needs onnxruntime; agent spike)
local-turn     = ort + hf-hub                       # 09  (ONNX + ten-vad-sys FFI later)
local-stt      = (whisper-rs / sherpa TBD)          # 09  (Parakeet/faster-whisper replacement)
twitch         = twitch_api                          # 11 (dormant)
azure-tts      = azure-speech                        # 09 (alt: REST over net)
audio-resample = rubato + opus                       # 09 (needs system libopus)
```

**Build validation performed on this scaffold (rustc 1.94.1, stable):**
`default`, `discord`, `discord-voice`, and `stt-deepgram,twitch,azure-tts` all
compile clean. `local-embed`/`local-turn` (onnxruntime) and `audio-resample`
(standalone libopus) are **resolution-validated only** — their native toolchain
(onnxruntime binary, libopus) is provisioned by the 04/09 agents during the
local-ML spike the ecosystem report calls for. `local-stt` has no crate chosen
yet (whisper-rs / sherpa-onnx candidates).
