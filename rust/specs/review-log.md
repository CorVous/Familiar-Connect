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
