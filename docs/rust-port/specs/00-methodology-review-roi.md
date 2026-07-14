# Adversarial review vs. stronger authorship — evidence review

Research snapshot: 2026-07-11. This determines how Phase 3 (module porting)
spends verification tokens. Verdict first, then the evidence map, then the
pipeline policy derived from it.

## Verdict

**Keep fresh-context adversarial review — but not because "multi-agent beats
single-agent" (it doesn't, at matched budgets). Because of three narrower,
measured effects:**

1. Models are measurably worse at catching errors **in their own context**
   than identical errors presented fresh.
2. Extra authorship compute is bottlenecked by exactly this self-verification
   weakness, so it doesn't substitute for independent review.
3. In a port, the deterministic verifiers are **endogenous** — the test suite
   is itself ported by the same class of agent, and agents measurably overfit
   visible tests. Fresh review targets precisely this residual.

## What is measured (vs. popular intuition)

- **Budget-matched, generic multi-agent critique/debate loses or ties** vs.
  single-agent sampling/reasoning (arxiv 2604.02460; 2605.00914 "Cost of
  Consensus"; 2605.27559 "Detection Without Correction" — conditional
  miscorrection dominates at 53–94%). Headline multi-agent coding results are
  compute-confounded.
- **Self-correction without external signal is null-to-negative** (Huang et
  al. 2310.01798 line, held through 2026: Kamoi TACL-2024; RealCritic
  2501.14492). Self-Correction Bench (2507.02778): 64.5% blind-spot rate on
  own errors vs. identical external errors. Swapping self-feedback for human
  feedback raised repair success 1.58× (2306.09896) — feedback quality is the
  constraint, not revision ability.
- **Fresh context is the active ingredient** (Cross-Context Review,
  2603.12123): fresh-session review F1 28.6% vs. 24.6% same-session (d=0.52),
  and — key — vs. 23.8% for a subagent that *sees the transcript* (d=0.57).
  The transcript is the contaminant, not the session boundary. Corroborated by
  Self-Attribution Bias (2603.04582): correctness-monitor AUROC 0.99 → 0.89
  when the code is framed as the monitor's own. Effect sizes provisional
  (single-model studies).
- **Author–reviewer dialogue destroys reviewer precision** (0.30 → 0.20,
  2603.16244). Review must be single-pass and isolated.
- **Critique/selection is the best marginal spend of extra compute**:
  CriticGPT ~85% inserted-bug catch vs. ~25% human; CTRL critic +106%
  relative pass@1 over self-critique, weaker critic improves stronger
  generator; OpenHands critic-guided Best@8 73.8% vs. Random@8 57.9% —
  parallel sampling without a selector wastes most of its value.
- **Why compiler+clippy+ported tests aren't sufficient**: agents saturate
  visible test suites while held-out suites lag 43–48pp (SpecBench
  2605.21384); LLM-ported tests skew happy-path with weak assertions (large
  coverage-vs-mutation-score gaps, 2605.22175); compilation success ≠
  semantic equivalence (Lost in Translation ICSE-2024; Python vs. Rust `%`
  sign behavior and friends, 2605.02195); Rust's compiler kills data races
  but not deadlocks, async-ordering races, or anything behind FFI. Bun's
  reviewer catching a use-after-free that compiled clean is the existence
  proof.
- **Bun's "2 adversarial reviewers" is intuition, not measurement.** Primary
  source (recovered full text) confirms: the design rationale is a human
  code-review analogy, verbatim — no ablation, no false-positive rate, no
  cost attribution; evidence is three exemplar catches (a libuv-async
  use-after-free that compiled clean, a negative-timestamp trunc-vs-floor
  bug, an eager `unwrap_or` panic). Their reviewers got "the diff and
  nothing else." Their real safety net was the ~million-assertion
  language-independent test suite — an asset this port does NOT have, since
  our tests must be ported too (which is why our reviewer gets the Python
  original + ported tests, not diff-only).
- **Honest scoping of the critic evidence**: the cleanly-measured critic
  gains in coding harnesses are as SELECTORS over N candidates (OpenHands
  critic best@5 +5.8pp; CodeMonkeys selection recovers ~half the
  random-to-oracle gap; R2E-Gym/DeepSWE: hybrid tests+critic selection
  strictly dominates tests-alone, which saturates). No published result
  isolates "reviewer of a single attempt vs. same tokens on authorship."
  For our single-attempt-per-module pipeline, the reviewer case rests on
  the self-blind-spot/fresh-context evidence plus the endogenous-verifier
  problem above — plausible and cheap, but not leaderboard-proven; hence
  the ablation logging in §Standing caveat.

## Pipeline policy (Phase 3)

1. **One fresh-context DIFFERENTIAL reviewer per module.** Inputs: Python
   original + Rust diff + ported tests. Never the implementer's transcript.
   Explicit charges:
   - Where do the ported tests assert LESS than the originals (assertion
     count/strength, deleted edge cases)? — the test-weakening failure mode.
   - Where does Rust semantics silently diverge (integer overflow, `%` and
     integer division, float formatting, dict/HashMap ordering,
     exception→Result mapping, str/bytes)?
   - Any unsafe/FFI/async-ordering/cancellation hazards?
2. **Single pass, no dialogue.** Findings go to a separate fixer context;
   fixes re-verified by build+tests (execution is the only trusted arbiter;
   detection-without-correction is the dominant failure mode).
3. **Second reviewer only selectively**: modules with FFI, heavy concurrency
   (responders, audio path), or weak inherited coverage. Measure reviewer #2's
   unique-true-finding rate over the first batch; drop if <~10–20%.
4. **Don't reallocate reviewer budget to longer authorship trajectories.**
   A diff-scoped review is a small fraction of a port's cost; matched-budget
   evidence says unverified authorship compute is the weakest spend. If
   buying more authorship, buy best-of-N with selection against tests.
5. **Differential testing where cheap** (pure-logic modules): run Python and
   Rust on shared/property-based inputs; deterministic external signal beats
   any LLM critique where available. Reviewer covers only what differential
   tests can't reach.

## Standing caveat

No public study directly A/Bs "reviewer per module vs. same tokens on
authorship reasoning" inside a porting pipeline with strong CI. We log each
reviewer's unique findings (not caught by build/clippy/tests) during Phase 3
so this repo generates its own ablation data.

Full source list: see session research report (arxiv 2604.02460, 2605.00914,
2605.27559, 2310.01798, 2507.02778, 2501.14492, 2603.12123, 2603.16244,
2603.04582, 2404.13076, 2402.11436, 2605.08563, 2407.00215 CriticGPT,
2502.03492 CTRL, 2501.14723 CodeMonkeys, 2502.01839, 2606.21811, 2605.21384
SpecBench, 2606.15385, 2605.02195, ICSE-2024 Lost in Translation,
ACM 3729315, 2605.22175, 2410.21136; bun.com/blog/bun-in-rust;
metr.org 2025-07-10 RCT).
