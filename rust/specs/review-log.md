# Reviewer ablation log

Per 00-methodology-review-roi.md §Standing caveat: no public study measures
"fresh-context reviewer per module vs. same tokens on authorship" in a porting
pipeline with strong deterministic gates. This log records what OUR reviewers
uniquely catch — findings invisible to `cargo build` + `clippy -D warnings` +
the ported test suite — so the review stage can be re-costed or cut on
evidence after enough batches.

Format: one row per finding that survived fixer verification. "Gate-visible?"
= would any deterministic gate have caught it eventually (no reviewer needed)?

## Layer 0 pilot (3 packages: bus, diagnostics, foundation) — 2026-07-11

| Pkg | Kind | Sev | Finding | Gate-visible? | Outcome |
|---|---|---|---|---|---|
| bus | test-weakening | low | Ported `parent_event_ids` test dropped Python's hashability pin (`isinstance(..., tuple)`), leaving only value-equality that the static type trivially implies | No — test passed as written | Fixed: HashSet-insertion assertion restored the pin |
| diagnostics | async-hazard | high | `timed_async` failed to emit the `status=error` span when the future is dropped mid-flight (cancellation) — Python's `try/finally` emits it; the port only handled panic + success | No — no ported test covered cancellation-drop | Fixed: sync-Drop SpanGuard; new regression test (fails on old code) |
| diagnostics | semantic-divergence | med | `VoiceBudgetRecorder::record` panicked at `max_turns == 0` (turn evicted at creation), where Python degrades gracefully via a post-popitem local dict | No — no test hit the 0 configuration | Fixed: orphan-map arm; new regression test (panics on old code) |
| diagnostics | other (infra) | low | Singleton reset fns gated `#[cfg(test)]` instead of `#[cfg(any(test, feature = "test-util"))]` per DESIGN 4.8 — cross-subsystem integration tests (06/07/09) couldn't isolate singletons | Yes, later — Layer 3 tests would have hit it | Fixed by orchestrator (Cargo.toml is shared) |
| foundation | semantic-divergence | med | `coerce_json` parse boundary: serde_json rejects `NaN`/`Infinity`/`1e999` (Python `json.loads` accepts) and lossily floats >i64 ints (Python keeps exact) — undeclared divergence | No — compiles, passes all ported tests | Declared deviation + 2 pinning regression tests; runtime change infeasible within spec-approved types (serde_json::Value can't represent non-finite floats); `arbitrary_precision` declined as crate-wide blast radius |

**Batch verdict:** 5 findings / 3 packages; 4 of 5 invisible to every
deterministic gate, incl. one high-severity cancellation hazard — exactly the
class the methodology evidence predicted (fresh-context review catching what
compiler+tests structurally cannot). Reviewer stage retained for Layer 1.

**Fixer discipline note:** fixers correctly declined shared-file edits
(Cargo.toml) and routed them to the orchestrator; the foundation fixer
verified the finding empirically before choosing declare-and-pin over a
behavior change. No finding was rubber-stamped.

## Layer 1 (3 packages: config-identity, history, small-utils) — 2026-07-11

| Pkg | Kind | Sev | Finding | Gate-visible? | Outcome |
|---|---|---|---|---|---|
| config-identity | semantic-divergence | med | Error-message tails printed `got -1.0` where Python prints `got -1`: the sign check ran after int→f64 conversion, Python checks the raw TOML value first. Substring-matching tests (mirroring `pytest.raises(match=)`) let it through | No — substring asserts passed | Fixed: per-arm sign checks; tests strengthened to byte-exact whole-message match |
| config-identity | semantic-divergence | low | `parse_hhmm_range` length check counted bytes not chars (DESIGN 4.9); provably behavior-neutral behind the ASCII gate but convention-violating. Reviewer's deeper point (Python `isdigit()` accepts non-ASCII decimal digits) retained as blessed ASCII-only deviation | No | Fixed the count; deviation documented |
| history | test-weakening | med | `search_facts`/`search_facts_scored` (spec behavior 20) had zero test coverage despite live tantivy indexes in the harness — 8 Python fact-search tests unported and undeclared | No — nothing failed | Fixed: 8 FTS-backed integration tests added (stemming, scoping, supersede-exclusion, bi-temporal as_of, BM25 ordering) |
| history | semantic-divergence | low | `parse_subjects` dropped subject items with present-but-non-string fields; Python coerces via `str(...)` (True/False/None spellings included) | No | Fixed: coercion helper + parity unit tests |
| small-utils | semantic-divergence | med | Python `str.isspace()` counts U+001C–U+001F (FS/GS/RS/US) as whitespace; Rust `char::is_whitespace()` doesn't — sentence-boundary detection diverged on those separators (spec 09 §67). Exhaustive scalar diff confirmed those 4 codepoints are the only gap | No — compiled, passed ported tests | Fixed: `is_py_whitespace` helper; 3 parity tests cross-checked against live Python |

**Batch verdict:** 5 findings / 3 packages, all 5 invisible to the gates.
Running total: 10 findings / 6 packages, 9 gate-invisible. The two med-severity
semantic catches (whitespace scalars, error-message tails) are precisely the
"compiles and passes weak tests" class. Reviewer stage retained.

## Layer 2 Wave A (6 packages) — 2026-07-12

16 findings (llm 3, stt 2, context 3, tts/players 4, voice 4, twitch+embed 0);
15 applied, 1 skipped with cause. Highlights (all gate-invisible):

| Pkg | Kind | Sev | Finding (condensed) |
|---|---|---|---|
| llm | test-weakening | med | Retry-After wiring test dropped: Python pins that a 429 with `Retry-After: 2` sleeps exactly 2.0s |
| llm | async-hazard | low | Python `stream_completion` is lazy (semaphore acquire/POST run on first `__anext__`); Rust ran them eagerly at call time — semaphore hold-window semantics shifted |
| tts/players | test-weakening | med | Jitter-hint tests stopped verifying DiscordVoicePlayer actually forwards client jitter params |
| tts/players | async-hazard | low | Cartesia WS not closed gracefully on early consumer drop (barge-in path) — Python's async-generator `finally` sends close frame |
| voice | semantic-div | low | SmartTurn/TenVad error variants swallowed as `false` verdicts where Python propagates (UnsupportedLogitsShape, WrongChunkLength) |
| context | semantic-div | low | `str.title()` semantics, empty-string falsiness in channel markers, sub-ms gap precision — three small Python-ism drifts |
| stt | semantic-div | low | urlencode space-as-`+` divergence in Deepgram keyterms; diarization speaker coercion narrower than Python's `int(...)` |

Notable process event: the harness restarted mid-run, orphaning 2 fixers; the
workflow resumed from its journal with all completed agents cached — only the
in-flight fixers re-ran. Journal-based resume worked as designed.

**Batch verdict:** the async-hazard class keeps paying (lazy-vs-eager stream
semantics and WS close-on-drop are exactly what compiles clean and passes unit
tests). Running total: 26 findings / 12 packages, ~24 gate-invisible.

## Layer 2 Wave B (2 packages: tools, sleep) — 2026-07-12

7 findings (tools 4, sleep 3), all applied; gates green.

Notable fixer dispositions from the full Layer 2 fix stage:
- **Justified reviewer override (llm-client):** the reviewer's eager-vs-lazy
  `stream_completion` finding was verified REAL but SKIPPED with cause — the
  eager header/status check is the spec-prescribed generator→Stream+Drop
  translation (spec 08 port-notes) and is required for the permit-release-at-
  header contract; making it lazy would break three faithfully-ported error-
  path tests. First case of the decision-log/spec authority correctly
  overruling a technically-true finding.
- **tts-and-players jitter tests** were rewritten to assert the *observable*
  prebuffer/pad-underrun behavior of the built source rather than the stub's
  own fields — restoring guard value without a shared-file accessor.

**Layer 2 totals:** 23 findings / 8 packages, ~21 gate-invisible, 1 justified
skip. Tree at 1,484 tests / 0 failures after the layer. Reviewer retained.

## Layer 3 (3 packages: responders, workers, activities-watchers-familiar) — 2026-07-12

10 findings (responders 3, workers 4, activities 3), all applied. Highlights:

| Pkg | Kind | Sev | Finding (condensed) |
|---|---|---|---|
| responders | async-hazard | high | `spawn_final` inflight-map race: on a multi-thread runtime a fast task could complete and run its cleanup BEFORE the parent inserted its handle, stranding a completed entry. Impossible under Python's event loop (create_task can't run before the current coroutine awaits) — a true GIL-hidden race. Fixed by holding the map lock across spawn+insert (no await under lock) |
| responders | async-hazard | low | Per-delta cancellation check ran after the empty-content skip; Python checks cancellation FIRST unconditionally (spec V12, test-observable preempted log) |
| responders | unapproved-deviation | med | Projector registry `with_builtins()` was left empty with a stale deferral note; all six workers existed — populated with full knob threading, fallible factory |
| workers | semantic-divergence | low ×2 | Python f-string renders literal "None" for present-author/None-display-name; Rust rendered empty string. Classic `or`-truthiness: `familiar_display_name or familiar_id.title()` also catches "" |
| activities | semantic-divergence | med | `valid keys:` error lists used declaration order; Python `", ".join(sorted(...))` — byte-stable contract restored + tests upgraded to full-string equality |
| activities | semantic-divergence | low | Activity window end used absolute duration arithmetic; Python does wall-clock (DST-naive) addition |

**Batch verdict:** the headline find — the spawn_final race — is the single
strongest datapoint yet for the reviewer stage: a real concurrency bug that
Python's runtime made unwritable, invisible to every gate, in the most
concurrency-critical package of the codebase. Running total: 33 findings /
15 packages, ~30 gate-invisible. Reviewer retained through Layer 4.

## Layer 4 (discord-shell, 2-stage) — 2026-07-12

4 findings, 4 applied. Highlights:

| Kind | Sev | Finding (condensed) |
|---|---|---|
| semantic-divergence | med | diagnose `read_lines` used `.lines().map_while(Result::ok)` — BufRead yields Err on invalid UTF-8 and map_while STOPS, silently dropping every later span line; Python opens with `errors="replace"`. Fixed to lossy decode |
| unapproved-deviation | low | All four production bus subscriptions used Unbounded backpressure instead of the Python per-topic defaults |
| async-hazard | low | Composition root wired BotStore over the blocking sync store facade inside gateway handlers |
| semantic-divergence | low | serenity message_update glue dropped edits for messages absent from the cache |

**FINAL PHASE 3 TALLY:** 37 findings / 16 packages+stages across 5 layers,
~34 invisible to compiler+clippy+ported tests. Every layer produced at least
one finding that survived fixer verification; zero findings were
rubber-stamped and two were correctly overruled with cited authority. The
port completed at 2,056 tests / 0 failures, gates green on default and
discord,discord-voice feature sets.

## Phase 4: composition-root wiring round — 2026-07-12

Context: the DAVE-runbook verification pass and the parity audit (independent
fresh-context agents) both caught that per-package porting left three
composition-root gaps that no package-scoped gate could see: voice
subscribe/playback never dispatched, TypingInterruptHandler never constructed
in production, SubscriptionRegistry duplicated (bot mutations invisible to
FocusManager). All three closed by a wiring agent (+7 wiring tests), then
diff-reviewed: 4 findings, 3 applied, 1 skipped with cause.

| Kind | Sev | Finding (condensed) | Outcome |
|---|---|---|---|
| semantic-divergence | low | typing_start derived is_bot from the serenity cache (uncached bot → spurious cancel); TypingStartEvent's member field carries the bot flag on guild typing | Fixed |
| semantic-divergence | low | Voice-client getter used HashMap::values().next() (arbitrary order) vs Python's insertion-ordered first-joined channel | Fixed per journal |
| — | — | (remaining finding + skip detail in workflow journal wf_42f2381b-075) | 1 more fixed, 1 skipped with cited cause |

**Lesson recorded:** package-scoped porting + review leaves integration-level
holes by construction; the completeness-critic + runbook-verification pass
(two agents, ~380k tokens) caught all three before the live integration
session could burn an evening discovering them.
