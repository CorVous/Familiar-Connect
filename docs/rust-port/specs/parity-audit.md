# Parity / completeness audit — Python→Rust port of Familiar-Connect

Role: completeness critic. Scope: find what is MISSING (coverage holes, unported
tests nobody declared, dangling integration seams) — not correctness bugs.
Date: 2026-07-12. No code was edited; report only.

Inputs: 117 Python `tests/test_*.py` files (2,367 test functions) vs. the Rust
workspace (`rust/familiar-connect/`, ~2,056 tests over in-module `#[cfg(test)]`
blocks + 60 `tests/*.rs` integration files). Skip declarations sourced from
`rust/specs/*.md` census tables, module doc-comments, and in-code comments.

---

## 1. Test-file census — unaccounted / unpinned behaviors

**Headline:** every Python test *file* has a Rust home (ported tests) or a
declared skip, with ONE trivial exception. The material gap is a **wiring seam**,
not a test file — see §3. Method: 75 Python test files are cited by name in Rust
code; the remaining 42 were matched to their module homes and verified to contain
tests; all deliberate stubs and Python-specific skips were confirmed declared.

| Python file → what's unpinned | Status | Suggested disposition |
|---|---|---|
| `test_typing_interrupt.py` (Y1–Y6) — the **integration seam** is unpinned. The unit behaviors ARE ported (`src/typing_interrupt.rs`, 9 tests), but the handler is **never constructed in the composition root**, so no test proves the composed binary cancels replies on user typing. See §3. | **UNACCOUNTED (undeclared)** | Wire `TypingInterruptHandler` in `create_bot`/`run.rs` (is_subscribed closure over `bot_subs` + the router), OR add an explicit deferral note like the voice-runtime one and file it. |
| `test_docs.py` (6 tests) — docs-drift checks against `mkdocs.yml` / `docs/` via Python `ast`. No Rust analog; no mkdocs in the Rust tree; not referenced in any spec. | **UNACCOUNTED (undeclared) — but legitimately N/A** | Declare as Python-specific-skip (documentation-parity harness has no Rust equivalent). Low importance. |
| `test_integration.py` (11 tests) — not referenced in any spec. CLI help/version → covered by `tests/cli_smoke.rs`. `example` subcommand tests → moot (`commands/example.py` "intentionally not ported", `commands/mod.rs:2`). verbose-flag/invalid-subcommand/library-import → Python-argparse specifics. | Accounted (implicitly) | Add a one-line census note pointing `test_integration.py` → `cli_smoke.rs` + example-not-ported, so it isn't a silent orphan. |

### Declared skips confirmed (not unaccounted)
- `test_bus_protocols.py` — `isinstance` structural checks skipped (traits enforce at compile time); enum-variant-count test kept (`tests/bus_protocols.rs`). Declared in spec 01 §497.
- `test_turso_compat.py` — thread-affinity shim dropped (rusqlite/turso have no such constraint). Declared Python-specific-skip, spec 03 §805.
- `test_dave_client.py`, `test_dave_ws.py` — recorded **skip-with-reason** in-code (`src/voice/dave_client.rs`, `src/voice/dave_ws.rs`): songbird owns DAVE/MLS, so there is no `VoiceClient`/WS subclass seam to port. The docs are a verification checklist for the Layer-4 songbird glue.
- `test_parakeet_transcriber.py`, `test_faster_whisper_transcriber.py` — declared STUBs (`src/stt/parakeet.rs`, `src/stt/faster_whisper.rs`): no Rust NeMo/CTranslate2 runtime chosen; `local-stt` carries no engine; factory degrades with `SttError::LocalSttUnavailable` (parity of Python lazy-import failure). Contract + tests port when an engine lands.
- `test_recording_sink.py` — Sink-subclass assertion Python-specific-skip (spec 09 §797); the `(user, mono)` queue behavior is ported (`src/voice/recording_sink.rs`, 4 tests).
- Sub-test skips: `TestEstimatorPerf` (<1 ms wall-clock), `TestOpenRouterLive` (live API), two `test_config.py` module-reflection tests, the `DREAM_EXTRACTION_CLAUSE_DEFAULT` "no module constant" assert — all declared in their spec census rows.

### Coverage density note (not a gap)
Rust consolidates parametrized Python tests into fewer, denser tests. Steepest
ratios: history family (Python ~310 portable fns → Rust 162) and
`test_attentional_tools.py` (68 → spread across 30 in `tools_attentional.rs` +
in-module tool tests). Store CRUD for dossiers/reflections/fact-embeddings
(`test_people_dossiers_store.py`, `test_reflections_store.py`,
`test_fact_embeddings_store.py`) has **no dedicated Rust file** but IS covered —
consolidated into `tests/history_facts.rs` + the worker integration tests
(`workers_fact_embedding.rs`, `workers_reflection.rs`) and context-layer tests.
Behaviors present; only the file-to-file mapping is many-to-one.

---

## 2. Stub / missing-module list

**No undeclared code stubs.** `grep -rE 'todo!|unimplemented!'` over
`rust/familiar-connect/src/` returns **zero** hits. Every Python
`src/familiar_connect/**/*.py` module has a non-stub Rust counterpart except the
five deliberately-deferred stubs, all documented in-code:

| Rust module | Kind | Declaration |
|---|---|---|
| `src/stt/parakeet.rs` (19 lines) | STUB — intentionally unimplemented | module doc: no Rust NeMo runtime; `local-stt` engine unchosen; factory degrades. |
| `src/stt/faster_whisper.rs` (20 lines) | STUB — intentionally unimplemented | module doc: no CTranslate2 runtime; same degrade path. |
| `src/voice/dave_client.rs` (41 lines) | Documentation stub (checklist) | module doc: songbird owns the session; tests skip-with-reason. |
| `src/voice/dave_ws.rs` (91 lines) | Documentation stub (checklist) | module doc: songbird owns the voice gateway; tests skip-with-reason. |
| `src/voice/turn_detection/ten_vad.rs` (441 lines) | **Not a stub** — full wrapper; only the native FFI is deferred | module doc: `ten-vad-sys` FFI crate deferred (keeps `unsafe` out); native handle is an injected seam (`NativeTenVad`) that degrades to `VadError::MissingBackend`. Wrapper + thresholding + reset fully ported and tested. |

`commands/example.py` is "intentionally not ported" (`commands/mod.rs:1-3`) — a
template, not a subsystem. All four match the documented deliberate-stub set.

---

## 3. Seam audit — verdict

### 3a. SubscriptionRegistry shared-mutable seam — **FAIL (two divergent copies)**

The composition root does **NOT** supply one shared-mutable
`SubscriptionRegistry` consumed by both `BotEvents` and `FocusManager`. It builds
**two independent instances** loaded separately from the same sidecar file:

- `src/commands/run.rs:616` — `let focus_subs = Arc::new(SubscriptionRegistry::new(&subs_path)?);` → given to `FocusManager` (`run.rs:623`), read-only (`Arc<SubscriptionRegistry>`).
- `src/commands/run.rs:617` — `let bot_subs = Arc::new(Mutex::new(SubscriptionRegistry::new(&subs_path)?));` → given to the bot (`run.rs:650`, `CreateBotDeps.subscriptions`), shared-mutable.

The divergence is **declared in-code** (`run.rs:609-614`): "live runtime
`/subscribe` mutations reach the bot's copy (and disk) but not the focus snapshot
until the FocusManager shared-mutable-seam change lands (filed as a shared-file
request)." Runtime consequence: `/subscribe`, `/unsubscribe`, and ephemeral DM
subscriptions mutate `bot_subs` + disk (`bot.rs:975`, `1114`, `1602`), while the
`FocusManager` keeps reading a frozen `focus_subs` snapshot until process
restart. `is_focused`/`should_wake`/startup-default-focus (`run.rs:632-643`) all
read the stale copy.

Verdict: acknowledged deviation, but a **live dangling seam** — not the single
shared seam the target design specifies. Not present in the spec-10 port-notes;
only the `run.rs` in-code comment records it.

### 3b. typing-interrupt wiring — **FAIL (never wired; undeclared)**

Related seam, and the more serious one because it is **undeclared**:

- `TypingInterruptHandler::new` is called only inside a `#[cfg(test)]` block (`src/typing_interrupt.rs:223`). No production call site.
- `BotHandle::with_typing_interrupt` (`src/bot.rs:883`) is defined but **never called** anywhere in the crate.
- `handle.typing_interrupt` is initialized `None` (`bot.rs:855`) and never reassigned. Production `create_bot` (`bot.rs:1873`) wires the typing **indicator** (`trigger_typing = Some(SerenityTyping…)`, `bot.rs:1879-1882`) but **not** the typing **interrupt**.
- Therefore `run.rs:830-831` (`if let Some(handler) = &handle.typing_interrupt`) never fires — the text responder never receives a typing handler in the composed binary.

Consequence: `test_typing_interrupt.py`'s Y1–Y6 (cancel-on-user-typing, backoff
ladder) are unit-pinned but **dead in the running product**; the
`[discord.text].respond_to_typing` config switch has no consumer. Python's design
has `bot.py` own/construct the handler (spec 06 §647-650, matching the
`bot.py → typing_interrupt` import direction). No deferral note exists in specs or
code.

### 3c. known_projectors() / known_embedders() config validation — **PASS**

Correctly wired and load-bearing:

- `run.rs:375-376` computes `known_projectors()` / `known_embedders()` and threads them into `load_character_config(..., &known_proj, &known_emb)` (`run.rs:379-384`) — not computed-and-discarded.
- They flow through `parse_character_config` (`config.rs:804-805, 862-863, 936, 940`) to the real rejection arms: `config.rs:2044` `if !known_projectors.contains(name)` (unknown memory projector → error) and `config.rs:2090` `if !known_embedders.contains(&backend)` (unknown embedder backend → error).

### 3d. Other declared dangling seams (in-code, not flagged by the task)

- **Voice-runtime / TTS playback** (`run.rs:685-695`): the songbird `Call → VoiceClientLike` retrieval seam "is not populated by the landed gateway glue," so `DiscordVoicePlayer` gets a `|| None` closure and playback degrades to no-op until voice-runtime wiring lands. Declared. Consistent with the DAVE/songbird thinning; this is the integration side of `test_voice_intake.py`.

---

## 4. Feature-matrix build check

All four combos compile (fresh `cargo clean` was required mid-run: the 27 GB
target dir hit 100% disk and gave false "No space left on device" failures on the
last two — those are disk artifacts, not compile errors; all were rebuilt clean).

| Feature combo | Result | Notes |
|---|---|---|
| `--features twitch` | **PASS** (exit 0) | Compiles `twitch_api 0.8.0` + `reqwest`. |
| `--features stt-deepgram` | **PASS** (exit 0) | Compiles the official `deepgram` SDK path. |
| `--features azure-tts` | **PASS** (exit 0) | Compiles `azure-speech`; 2m53s from clean. |
| `--no-default-features --features store` | **PASS** (exit 0) | Compiles; **6 dead-code warnings** (net-gated fns unused when `net` off, e.g. `parse_word_timestamps` at `src/tts.rs:484`). Warnings only, no errors. |

---

## 5. Ready for DAVE session? — **YES, with two logged cleanup items**

The port is structurally complete for starting DAVE/voice work:

- DAVE scaffolding is in place as verification checklists (`dave_client.rs`, `dave_ws.rs`) with the opcode/lifecycle invariants the songbird glue is checked against; `test_dave_client.py`/`test_dave_ws.py` are properly recorded skip-with-reason.
- The 09↔10 voice seam (`(user_id, mono_pcm)` queue, `VoiceSource`, endpointer, `recording_sink`) is ported and tested; the only open piece is the declared voice-runtime wiring (`run.rs:687`), which is exactly what a DAVE/voice session would land.
- No undeclared code stubs; all feature combos build; config-validation seam is sound.

**Two seam gaps to log before/alongside the DAVE session (neither DAVE-blocking):**
1. **typing-interrupt handler is never constructed** (§3b) — undeclared, and it silently disables a shipped feature + its config switch. Highest-priority cleanup: wire it or declare-and-file it.
2. **SubscriptionRegistry two-copy divergence** (§3a) — declared but unresolved; live `/subscribe` mutations don't reach the FocusManager until restart. Resolve the "shared-file request" (FocusManager over `Arc<Mutex<…>>` or an `arc-swap` snapshot) so focus/wake logic sees runtime subscription changes.

Minor: add census notes for `test_integration.py` and `test_docs.py` so they stop
reading as orphans, and clear the 6 dead-code warnings under `--no-default-features`.
