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
