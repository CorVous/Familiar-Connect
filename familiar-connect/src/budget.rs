//! Token estimator + `TierBudget` / `ModelBudgetCurve` (subsystem 05).
//!
//! Per-tier prompt-assembly budget. Each cap is a hard number — no proportional
//! derivation, no "auto-fill from total". The assembly layers consume the values
//! directly, each self-truncating to its own cap. The whole-prompt
//! [`TierBudget::total_tokens`] is a *derived* sum of the per-section token caps,
//! for reporting only — nothing trims against it.
//!
//! Token accounting uses the fast `len(text)/4` heuristic (chars-per-token 4) —
//! no real tokenizer on the hot path. `len` counts **Unicode scalars**, not
//! bytes.
//!
//! **Four chars per token is a rough default, not a safety margin.** It
//! over-counts for some tokenizers and under-counts materially for others.
//! Measured over a real 54-call session, `in_tokens / est_in_tokens` was
//! **1.453** for `z-ai/glm-5.2` (26 calls) and **1.197** for `z-ai/glm-5v-turbo`
//! (15 calls) — the heuristic 20-45% low, so every `TierBudget` cap was
//! effectively that much larger than configured. Per-model calibration exists
//! for exactly this.
//!
//! `CHARS_PER_TOKEN` deliberately stays at 4. Lowering it to bias safe would
//! over-trim every model the default already fits, silently discarding context
//! the operator configured; the targeted fix is calibration, not a smaller
//! global divisor. Settled — don't re-litigate.
//!
//! [`TokenCalibration`] does the correcting, from the *true* prompt-token counts
//! OpenRouter reports per call (#183), persisted across restarts by
//! [`cache`] so a cold start is not blind. Calibration is **upward-only**: the
//! estimate drives client-side trimming, so over-counting merely drops a little
//! extra context while under-counting risks an oversized request the API
//! rejects. See [`estimate_tokens_calibrated`]; the assembly layers trim
//! through it, and [`char_cap_for_tokens`] inverts it for truncation.
//!
//! This is a leaf module: it names [`crate::llm::Message`] but pulls in nothing
//! from `config`/`context`, so `config` can depend on it without a cycle.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use chrono::Utc;

use crate::budget::cache::CachedTotals;
use crate::llm::Message;
use crate::support::round::half_even;

pub mod cache;

/// OpenAI's well-known English heuristic — a rough default, *not* a safe upper
/// bound: it over-counts for some tokenizers and under-counts by 20-45% for
/// others (module docs carry the measured ratios). Stays at 4 by design;
/// [`TokenCalibration`] corrects per model instead.
const CHARS_PER_TOKEN: i64 = 4;

/// Per-message chat-format framing (role + delimiters).
const MESSAGE_OVERHEAD_TOKENS: i64 = 4;

/// Fast char-based token estimate: `ceil(len / 4)` over Unicode scalars, `0` for
/// the empty string.
///
/// Raw, and per-tokenizer accuracy varies either way (see module docs) — budget
/// call sites want [`estimate_tokens_calibrated`].
#[must_use]
pub fn estimate_tokens(text: &str) -> i64 {
    // Count Unicode scalars, not bytes.
    estimate_tokens_from_chars(text.chars().count())
}

/// [`estimate_tokens`] for callers holding a Unicode-scalar count, not the text.
///
/// Single home for `CHARS_PER_TOKEN` so no call site re-derives `/ 4` (N5).
#[must_use]
pub fn estimate_tokens_from_chars(char_count: usize) -> i64 {
    let n = i64::try_from(char_count).unwrap_or(i64::MAX);
    (n + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Chat-format estimate including role/name framing:
/// `estimate_tokens(content_str) + 4` (+ `estimate_tokens(name)` when `name` set).
#[must_use]
pub fn estimate_message_tokens(msg: &Message) -> i64 {
    let mut n = estimate_tokens(&msg.content_str()) + MESSAGE_OVERHEAD_TOKENS;
    if let Some(name) = &msg.name {
        // An empty name adds `estimate_tokens("") == 0`.
        n += estimate_tokens(name);
    }
    n
}

/// Sum [`estimate_message_tokens`] across a message list.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> i64 {
    messages.iter().map(estimate_message_tokens).sum()
}

// ---------------------------------------------------------------------------
// calibration (#183)
// ---------------------------------------------------------------------------

/// Upper bound on an applied calibration ratio.
///
/// Degenerate samples are reachable: multimodal `Content::Blocks` contribute 0
/// chars to the estimate while the provider bills real tokens for them, so one
/// image-heavy call implies an absurd ratio. Calibration only ever *raises* the
/// estimate, so an unbounded ratio would silently over-trim context. `4.0`
/// still covers the densest legitimate case (CJK text, ~1 token per char
/// against the heuristic's 0.25).
const MAX_CALIBRATION_RATIO: f64 = 4.0;

/// Updates tolerated between cache writes.
///
/// A write per LLM call would put file I/O on the hot path for a file that
/// changes by a fraction of a percent each time. Eight bounds what a hard kill
/// loses (seven samples) while keeping a busy voice session to a handful of
/// few-hundred-byte writes. Paired with [`FLUSH_MAX_AGE`] so a slow trickle of
/// calls still persists.
const FLUSH_EVERY_UPDATES: u32 = 8;

/// Wall-clock ceiling on unpersisted learning, checked when a sample lands.
const FLUSH_MAX_AGE: Duration = Duration::from_secs(60);

/// Running true-vs-estimated input-token ratio, keyed by model.
///
/// Keyed by **model, not `model.slot`**: chars-per-token is a property of the
/// model's tokenizer, so slots sharing a model share a rate, and pooling their
/// samples converges faster. Readers only know a model anyway.
///
/// The ratio is a running *ratio of totals* (`Σ actual / Σ estimated`) rather
/// than an EWMA: it needs no decay constant, it is exact from the very first
/// sample (an EWMA seeded at `1.0` would spend its early calls biased toward
/// the seed), and weighting by prompt size is what budget accuracy wants. The
/// trade-off is no adaptation to a mid-life tokenizer change; [`cache`]'s TTL
/// bounds how long persisted totals can outlive one.
///
/// Totals survive restarts through [`cache`], loaded lazily by
/// [`get_token_calibration`] and written back on a debounce
/// (`FLUSH_EVERY_UPDATES`). [`TokenCalibration::new`] keeps the old
/// memory-only behaviour.
#[derive(Debug, Default)]
pub struct TokenCalibration {
    state: Mutex<State>,
    /// Cache file backing the store; `None` keeps it purely in memory.
    path: Option<PathBuf>,
}

/// Accumulators plus the write-debounce bookkeeping, under one lock.
#[derive(Debug, Default)]
struct State {
    totals: HashMap<String, Totals>,
    /// Updates folded in since the last write attempt.
    pending: u32,
    /// When the last write was attempted; `None` until the first.
    last_flush: Option<Instant>,
}

/// Per-model accumulators; the ratio is `actual / estimated`.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    estimated: i64,
    actual: i64,
}

impl State {
    /// Snapshot to write, when the debounce says it is time; `None` otherwise.
    ///
    /// Marks the flush as taken, so a failed write costs one interval rather
    /// than retrying on every call.
    fn take_due_snapshot(&mut self, force: bool) -> Option<BTreeMap<String, CachedTotals>> {
        if self.pending == 0 {
            return None;
        }
        // `last_flush == None` fires on the very first sample of the process:
        // a one-call session must not learn nothing.
        let due = force
            || self.pending >= FLUSH_EVERY_UPDATES
            || self
                .last_flush
                .is_none_or(|then| then.elapsed() >= FLUSH_MAX_AGE);
        if !due {
            return None;
        }
        self.pending = 0;
        self.last_flush = Some(Instant::now());
        Some(
            self.totals
                .iter()
                .map(|(model, t)| {
                    (
                        model.clone(),
                        CachedTotals {
                            estimated: t.estimated,
                            actual: t.actual,
                        },
                    )
                })
                .collect(),
        )
    }
}

impl TokenCalibration {
    /// Empty store — no model has a ratio yet. Never touches disk.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store seeded from `path`'s persisted totals, and persisting back to it.
    ///
    /// A missing, corrupt, or stale cache seeds nothing: the store behaves
    /// exactly as [`Self::new`] and the next write replaces the file.
    #[must_use]
    pub fn with_cache_path(path: Option<PathBuf>) -> Self {
        let totals = path.as_deref().map(load_totals).unwrap_or_default();
        Self {
            state: Mutex::new(State {
                totals,
                ..State::default()
            }),
            path,
        }
    }

    /// Persist the accumulators now, bypassing the debounce.
    ///
    /// For a shutdown hook or a test; no-op when nothing is pending or the store
    /// has no path.
    pub fn flush(&self) {
        self.write_if_due(true);
    }

    /// Fold one observation (heuristic estimate vs provider-reported truth).
    ///
    /// Non-positive pairs carry no ratio and are dropped. Saturating adds keep a
    /// long-lived process from overflowing the accumulators. May persist — see
    /// `FLUSH_EVERY_UPDATES`.
    pub fn record(&self, model: &str, estimated: i64, actual: i64) {
        if estimated <= 0 || actual <= 0 {
            return;
        }
        {
            // The guard covers the whole update (lookup + both adds).
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let entry = guard.totals.entry(model.to_owned()).or_default();
            entry.estimated = entry.estimated.saturating_add(estimated);
            entry.actual = entry.actual.saturating_add(actual);
            guard.pending = guard.pending.saturating_add(1);
        }
        self.write_if_due(false);
    }

    /// Learned ratio for `model`; `None` until the model has a sample.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "token totals stay far below f64's exact-integer range"
    )]
    pub fn ratio(&self, model: &str) -> Option<f64> {
        // Copy out under the lock; release before doing arithmetic.
        let t = {
            let guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            guard.totals.get(model).copied()
        }?;
        (t.estimated > 0).then(|| t.actual as f64 / t.estimated as f64)
    }

    /// Write the snapshot the debounce releases, if any. Never holds the lock
    /// across the I/O.
    fn write_if_due(&self, force: bool) {
        let Some(path) = &self.path else {
            return;
        };
        let snapshot = {
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            guard.take_due_snapshot(force)
        };
        let Some(models) = snapshot else {
            return;
        };
        if let Err(err) = cache::write_cache(path, &models, Utc::now()) {
            tracing::debug!(
                target: "familiar_connect.budget",
                "token calibration cache write failed: {err}"
            );
        }
    }
}

/// Persisted totals for a fresh store: empty when the cache is absent, corrupt,
/// or past [`cache::CACHE_TTL_DAYS`].
fn load_totals(path: &Path) -> HashMap<String, Totals> {
    let Some(cached) = cache::read_cache(path) else {
        return HashMap::new();
    };
    if cache::is_stale(&cached.updated_at, Utc::now(), cache::CACHE_TTL_DAYS) {
        return HashMap::new();
    }
    cached
        .models
        .into_iter()
        .map(|(model, t)| {
            (
                model,
                Totals {
                    estimated: t.estimated,
                    actual: t.actual,
                },
            )
        })
        .collect()
}

static CALIBRATION: Mutex<Option<Arc<TokenCalibration>>> = Mutex::new(None);

/// Cache file the singleton loads from and persists to.
///
/// Production: the per-user cache file. Test builds: only what
/// `set_token_calibration_path` injected — `None` by default, so no unit test
/// can read or write the real operator's cache. `test-util` is a test-only
/// feature; a shipped build never enables it, so persistence is never off in
/// production. An integration test that drives the real client (the one path
/// that writes) must enable `test-util` and inject a tempdir.
#[allow(
    clippy::unnecessary_wraps,
    reason = "the None arm is the test build's; production always resolves a path"
)]
fn singleton_cache_path() -> Option<PathBuf> {
    #[cfg(any(test, feature = "test-util"))]
    {
        TEST_CACHE_PATH
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
    #[cfg(not(any(test, feature = "test-util")))]
    {
        Some(cache::default_cache_path())
    }
}

/// Return the process-wide [`TokenCalibration`], creating it on first use.
///
/// First use also loads the persisted totals (#183) — lazily, here, rather than
/// from a boot hook, so the store self-initialises wherever it is first touched
/// and `reset_token_calibration` keeps working. Fetched at call time (not import
/// time) so that reset takes effect immediately.
#[must_use]
pub fn get_token_calibration() -> Arc<TokenCalibration> {
    let mut guard = CALIBRATION.lock().unwrap_or_else(PoisonError::into_inner);
    guard
        .get_or_insert_with(|| Arc::new(TokenCalibration::with_cache_path(singleton_cache_path())))
        .clone()
}

/// Cache path the next singleton loads — tests only, always a tempdir.
#[cfg(any(test, feature = "test-util"))]
static TEST_CACHE_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Point the next singleton at `path` — tests only. Call *after*
/// [`reset_token_calibration`], which clears it.
#[cfg(any(test, feature = "test-util"))]
pub fn set_token_calibration_path(path: Option<PathBuf>) {
    *TEST_CACHE_PATH
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = path;
}

/// Reset the singleton so the next `get` creates a fresh instance — tests only.
///
/// Also clears the injected cache path, so no test inherits another's tempdir
/// and the default is a disk-free store.
#[cfg(any(test, feature = "test-util"))]
pub fn reset_token_calibration() {
    *CALIBRATION.lock().unwrap_or_else(PoisonError::into_inner) = None;
    set_token_calibration_path(None);
}

/// Calibration multiplier actually applied for `model`.
///
/// `1.0` (identity) for an unseen model or a learned ratio at or below `1.0`;
/// otherwise the learned ratio capped at `MAX_CALIBRATION_RATIO`. Single home
/// for the upward-only clamp — estimator and cap inverter share it.
fn applied_ratio(model: &str) -> f64 {
    get_token_calibration()
        .ratio(model)
        .filter(|r| *r > 1.0)
        .map_or(1.0, |r| r.min(MAX_CALIBRATION_RATIO))
}

/// [`estimate_tokens`] refined by what `model` actually charged on past calls.
///
/// **Upward-only, by design.** The estimate gates client-side trimming *before*
/// a request is sent, so the failure modes are asymmetric: over-counting drops
/// slightly more context than needed, under-counting ships an oversized request
/// the API rejects. A learned ratio at or below `1.0` is therefore discarded and
/// the raw heuristic stands; only ratios above `1.0` (capped at
/// `MAX_CALIBRATION_RATIO`) apply. An unseen model passes straight through.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "prompt-token counts are far below f64's exact-integer range; the i64::MAX guard covers truncation"
)]
pub fn estimate_tokens_calibrated(text: &str, model: &str) -> i64 {
    let raw = estimate_tokens(text);
    let ratio = applied_ratio(model);
    if ratio <= 1.0 {
        return raw;
    }
    let scaled = (raw as f64 * ratio).ceil();
    if scaled >= i64::MAX as f64 {
        return i64::MAX;
    }
    // Clamp: calibration may only ever revise upward.
    (scaled as i64).max(raw)
}

/// [`estimate_message_tokens`] under `model`'s calibration.
///
/// Content and name scale; the fixed chat framing does not.
#[must_use]
pub fn estimate_message_tokens_calibrated(msg: &Message, model: &str) -> i64 {
    let mut n = estimate_tokens_calibrated(&msg.content_str(), model) + MESSAGE_OVERHEAD_TOKENS;
    if let Some(name) = &msg.name {
        n += estimate_tokens_calibrated(name, model);
    }
    n
}

/// Most Unicode scalars whose calibrated estimate still fits `max_tokens` —
/// inverse of [`estimate_tokens_calibrated`]. `0` for a non-positive cap.
///
/// Truncation call sites need the *char* budget, so the calibration ratio
/// divides here where it multiplies in the estimator; without it a truncated
/// string would still measure over its own cap.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "token caps are small positive integers; the ratio is >= 1.0 so the quotient never exceeds max_tokens * 4"
)]
pub fn char_cap_for_tokens(max_tokens: i64, model: &str) -> usize {
    if max_tokens <= 0 {
        return 0;
    }
    let raw_cap = max_tokens.saturating_mul(CHARS_PER_TOKEN);
    let scaled = (raw_cap as f64 / applied_ratio(model)).floor();
    if scaled >= usize::MAX as f64 {
        return usize::MAX;
    }
    scaled as usize
}

/// Per-section multipliers for a specific model.
///
/// All fields default to `1.0` (identity — no change). Field names mirror
/// [`TierBudget`]'s 12 configurable caps exactly, so config parsing validates
/// keys via a simple set comparison. There is no `total_tokens` multiplier: the
/// whole-prompt total is derived from the per-section caps, so it scales
/// implicitly. Multipliers are validated positive (`> 0`) at config load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelBudgetCurve {
    /// Multiplier on the recent-history token cap.
    pub recent_history_tokens: f64,
    /// Multiplier on the RAG-context token cap.
    pub rag_tokens: f64,
    /// Multiplier on the people-dossier token cap.
    pub dossier_tokens: f64,
    /// Multiplier on the conversation-summary token cap.
    pub summary_tokens: f64,
    /// Multiplier on the reflections token cap.
    pub reflection_tokens: f64,
    /// Multiplier on the lorebook token cap.
    pub lorebook_tokens: f64,
    /// Multiplier on the max-history-turns count cap.
    pub max_history_turns: f64,
    /// Multiplier on the max-RAG-turns count cap.
    pub max_rag_turns: f64,
    /// Multiplier on the max-RAG-facts count cap.
    pub max_rag_facts: f64,
    /// Multiplier on the max-dossier-people count cap.
    pub max_dossier_people: f64,
    /// Multiplier on the max-reflections count cap.
    pub max_reflections: f64,
    /// Multiplier on the max-lorebook-entries count cap.
    pub max_lorebook_entries: f64,
}

impl Default for ModelBudgetCurve {
    fn default() -> Self {
        Self {
            recent_history_tokens: 1.0,
            rag_tokens: 1.0,
            dossier_tokens: 1.0,
            summary_tokens: 1.0,
            reflection_tokens: 1.0,
            lorebook_tokens: 1.0,
            max_history_turns: 1.0,
            max_rag_turns: 1.0,
            max_rag_facts: 1.0,
            max_dossier_people: 1.0,
            max_lorebook_entries: 1.0,
            max_reflections: 1.0,
        }
    }
}

/// Scale one integer cap by a curve multiplier: `max(1, round(base * mult))`,
/// with banker's rounding (half-to-even).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "budget caps are small positive ints; scaled values never approach i64/f64 precision limits"
)]
fn scale(base: i64, multiplier: f64) -> i64 {
    let scaled = half_even(base as f64 * multiplier) as i64;
    scaled.max(1)
}

/// Token budget for one assembly tier (voice / text / background).
///
/// Every cap is an explicit int enforced *independently*: each assembly layer
/// self-truncates to its own `*_tokens` cap. There is no combined cap — the
/// prompt's overall size is the sum of the section caps, surfaced as the derived
/// [`TierBudget::total_tokens`] for reporting.
///
/// The dataclass-level defaults below are the voice tier; production overlays the
/// shipped per-tier values from `_default/character.toml` at config load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierBudget {
    /// Cap on the recent-history block during build.
    pub recent_history_tokens: i64,
    /// Cap on the RAG-context block.
    pub rag_tokens: i64,
    /// Cap on the people-dossier block.
    pub dossier_tokens: i64,
    /// Cap on the conversation-summary block.
    pub summary_tokens: i64,
    /// Cap on the reflections block (M3).
    pub reflection_tokens: i64,
    /// Cap on the lorebook block (M4).
    pub lorebook_tokens: i64,
    /// Hard upper bound on recent-history turns (safety net before the token cap).
    pub max_history_turns: i64,
    /// Hard cap on RAG turn results.
    pub max_rag_turns: i64,
    /// Hard cap on RAG fact results.
    pub max_rag_facts: i64,
    /// Hard cap on dossier rows.
    pub max_dossier_people: i64,
    /// Hard cap on rendered reflection rows (M3).
    pub max_reflections: i64,
    /// Hard cap on rendered lorebook entries (M4).
    pub max_lorebook_entries: i64,
}

impl Default for TierBudget {
    fn default() -> Self {
        Self {
            recent_history_tokens: 3000,
            rag_tokens: 900,
            dossier_tokens: 900,
            summary_tokens: 600,
            reflection_tokens: 600,
            lorebook_tokens: 600,
            max_history_turns: 200,
            max_rag_turns: 10,
            max_rag_facts: 6,
            max_dossier_people: 16,
            max_reflections: 6,
            max_lorebook_entries: 12,
        }
    }
}

impl TierBudget {
    /// Derived sum of the six per-section **token** caps (count caps excluded).
    ///
    /// Not a configurable knob — nothing trims against it; it is the budgeted
    /// prompt ceiling exposed for reporting and headroom eyeballing.
    #[must_use]
    pub const fn total_tokens(&self) -> i64 {
        self.recent_history_tokens
            + self.rag_tokens
            + self.dossier_tokens
            + self.summary_tokens
            + self.reflection_tokens
            + self.lorebook_tokens
    }

    /// Return a new budget with each field scaled by the curve multiplier.
    ///
    /// `total_tokens` is derived, so it follows automatically once the
    /// constituent caps are scaled.
    #[must_use]
    pub fn apply_curve(&self, curve: &ModelBudgetCurve) -> Self {
        Self {
            recent_history_tokens: scale(self.recent_history_tokens, curve.recent_history_tokens),
            rag_tokens: scale(self.rag_tokens, curve.rag_tokens),
            dossier_tokens: scale(self.dossier_tokens, curve.dossier_tokens),
            summary_tokens: scale(self.summary_tokens, curve.summary_tokens),
            reflection_tokens: scale(self.reflection_tokens, curve.reflection_tokens),
            lorebook_tokens: scale(self.lorebook_tokens, curve.lorebook_tokens),
            max_history_turns: scale(self.max_history_turns, curve.max_history_turns),
            max_rag_turns: scale(self.max_rag_turns, curve.max_rag_turns),
            max_rag_facts: scale(self.max_rag_facts, curve.max_rag_facts),
            max_dossier_people: scale(self.max_dossier_people, curve.max_dossier_people),
            max_reflections: scale(self.max_reflections, curve.max_reflections),
            max_lorebook_entries: scale(self.max_lorebook_entries, curve.max_lorebook_entries),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ModelBudgetCurve, TierBudget, TokenCalibration, char_cap_for_tokens,
        estimate_message_tokens, estimate_message_tokens_calibrated, estimate_messages_tokens,
        estimate_tokens, estimate_tokens_calibrated, estimate_tokens_from_chars,
        get_token_calibration, reset_token_calibration,
    };
    use crate::diagnostics::testutil::singleton_guard;
    use crate::llm::Message;

    // --- estimate_tokens ---------------------------------------------------

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn short_string_rounds_up() {
        // ceil(5 / 4) = 2
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn overcount_safe() {
        // 44 chars -> 11 tokens (>= a real ~10-token tokenization).
        let text = "The quick brown fox jumps over the lazy dog.";
        assert!(estimate_tokens(text) >= 10);
    }

    #[test]
    fn message_overhead_added() {
        let m = Message::new("user", "x");
        // content = 1 token + overhead 4 = 5
        assert!(estimate_message_tokens(&m) >= 5);
    }

    #[test]
    fn message_with_name_costs_more() {
        let a = Message::new("user", "x");
        let b = Message::new("user", "x").with_name("alice_42");
        assert!(estimate_message_tokens(&b) > estimate_message_tokens(&a));
    }

    #[test]
    fn messages_sum() {
        let msgs = [
            Message::new("user", "abcd"),
            Message::new("assistant", "efgh"),
        ];
        let expected: i64 = msgs.iter().map(estimate_message_tokens).sum();
        assert_eq!(estimate_messages_tokens(&msgs), expected);
    }

    #[test]
    fn from_chars_matches_text_estimate() {
        assert_eq!(estimate_tokens_from_chars(0), 0);
        assert_eq!(estimate_tokens_from_chars(5), estimate_tokens("hello"));
        // Unicode scalars, not bytes: 3 CJK chars -> ceil(3/4) = 1.
        assert_eq!(estimate_tokens_from_chars(3), estimate_tokens("日本語"));
    }

    // --- TokenCalibration (direct; no singleton, parallel-safe) ------------

    #[test]
    fn first_sample_sets_the_ratio() {
        let cal = TokenCalibration::new();
        cal.record("m", 100, 120);
        let r = cal.ratio("m").expect("a ratio after one sample");
        assert!((r - 1.2).abs() < 1e-9, "ratio: {r}");
    }

    #[test]
    fn ratio_is_totals_weighted_across_samples() {
        let cal = TokenCalibration::new();
        cal.record("m", 100, 120);
        cal.record("m", 300, 280);
        // (120 + 280) / (100 + 300) = 1.0
        let r = cal.ratio("m").expect("a ratio");
        assert!((r - 1.0).abs() < 1e-9, "ratio: {r}");
    }

    #[test]
    fn unknown_model_has_no_ratio() {
        let cal = TokenCalibration::new();
        cal.record("m", 100, 120);
        assert!(cal.ratio("other").is_none());
    }

    #[test]
    fn models_do_not_share_ratios() {
        let cal = TokenCalibration::new();
        cal.record("a", 100, 200);
        cal.record("b", 100, 100);
        assert!((cal.ratio("a").unwrap() - 2.0).abs() < 1e-9);
        assert!((cal.ratio("b").unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn nonpositive_samples_are_ignored() {
        let cal = TokenCalibration::new();
        cal.record("m", 0, 500);
        cal.record("m", 100, 0);
        cal.record("m", -5, -5);
        assert!(cal.ratio("m").is_none());
    }

    #[test]
    fn concurrent_records_are_all_counted() {
        let cal = std::sync::Arc::new(TokenCalibration::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cal = std::sync::Arc::clone(&cal);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        cal.record("m", 10, 15);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread");
        }
        let r = cal.ratio("m").expect("a ratio");
        assert!((r - 1.5).abs() < 1e-9, "ratio: {r}");
    }

    // --- estimate_tokens_calibrated (singleton) ----------------------------

    #[test]
    fn calibration_unknown_model_passes_through() {
        let _g = singleton_guard();
        reset_token_calibration();
        let text = "a".repeat(400);
        assert_eq!(estimate_tokens_calibrated(&text, "never-seen"), 100);
    }

    #[test]
    fn calibration_raises_the_estimate() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 150);
        let text = "a".repeat(400);
        // ceil(100 * 1.5) = 150
        assert_eq!(estimate_tokens_calibrated(&text, "dense"), 150);
    }

    #[test]
    fn calibration_never_lowers_the_estimate() {
        let _g = singleton_guard();
        reset_token_calibration();
        // Model bills FEWER tokens than the heuristic guesses (ratio 0.5).
        get_token_calibration().record("sparse", 100, 50);
        let text = "a".repeat(400);
        assert_eq!(estimate_tokens_calibrated(&text, "sparse"), 100);
    }

    #[test]
    fn calibration_ratio_is_capped() {
        let _g = singleton_guard();
        reset_token_calibration();
        // Degenerate sample (e.g. image blocks count 0 chars): ratio 100.
        get_token_calibration().record("wild", 10, 1000);
        let text = "a".repeat(400);
        // 100 raw tokens * MAX_CALIBRATION_RATIO (4.0), not * 100.
        assert_eq!(estimate_tokens_calibrated(&text, "wild"), 400);
    }

    #[test]
    fn calibrated_estimate_of_empty_text_is_zero() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 400);
        assert_eq!(estimate_tokens_calibrated("", "dense"), 0);
    }

    #[test]
    fn calibrated_message_estimate_scales_content_not_framing() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 200);
        let msg = Message::new("user", "a".repeat(400)).with_name("Alice");
        // Content 100 -> 200, name "Alice" 2 -> 4, framing 4 flat.
        assert_eq!(estimate_message_tokens_calibrated(&msg, "dense"), 208);
        // Empty model misses the store: the raw estimate, byte for byte.
        assert_eq!(
            estimate_message_tokens_calibrated(&msg, ""),
            estimate_message_tokens(&msg)
        );
    }

    // --- char_cap_for_tokens -----------------------------------------------

    #[test]
    fn char_cap_shrinks_under_calibration() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 200);
        assert_eq!(char_cap_for_tokens(50, "dense"), 100);
        // Unseen and empty models keep the plain chars-per-token budget.
        assert_eq!(char_cap_for_tokens(50, "never-seen"), 200);
        assert_eq!(char_cap_for_tokens(50, ""), 200);
    }

    #[test]
    fn char_cap_of_nonpositive_is_zero() {
        let _g = singleton_guard();
        reset_token_calibration();
        assert_eq!(char_cap_for_tokens(0, ""), 0);
        assert_eq!(char_cap_for_tokens(-5, ""), 0);
    }

    #[test]
    fn char_cap_inverts_the_calibrated_estimate() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 150);
        let cap = char_cap_for_tokens(60, "dense");
        let text = "a".repeat(cap);
        assert!(estimate_tokens_calibrated(&text, "dense") <= 60);
    }

    #[test]
    fn reset_clears_learned_ratios() {
        let _g = singleton_guard();
        reset_token_calibration();
        get_token_calibration().record("dense", 100, 150);
        reset_token_calibration();
        assert!(get_token_calibration().ratio("dense").is_none());
    }

    // --- persistence across restarts (#183) --------------------------------

    mod persistence {
        use super::super::cache::{CACHE_FILE_NAME, CachedTotals, read_cache, write_cache};
        use super::super::{
            TokenCalibration, estimate_tokens_calibrated, get_token_calibration,
            reset_token_calibration, set_token_calibration_path,
        };
        use crate::diagnostics::testutil::singleton_guard;
        use chrono::{Duration, TimeZone as _, Utc};
        use std::collections::BTreeMap;
        use std::path::{Path, PathBuf};
        use std::sync::MutexGuard;
        use tempfile::TempDir;

        fn now() -> chrono::DateTime<Utc> {
            Utc.with_ymd_and_hms(2026, 8, 18, 12, 0, 0).unwrap()
        }

        // Serialize on the shared singleton lock, clear the store, then point it
        // at a tempdir. Order matters: the reset clears any injected path.
        fn isolated(path: &Path) -> MutexGuard<'static, ()> {
            let g = singleton_guard();
            reset_token_calibration();
            set_token_calibration_path(Some(path.to_path_buf()));
            g
        }

        fn seed(path: &Path, model: &str, estimated: i64, actual: i64) {
            write_cache(
                path,
                &BTreeMap::from([(model.to_owned(), CachedTotals { estimated, actual })]),
                Utc::now(),
            )
            .expect("seed the cache");
        }

        fn cache_path(dir: &TempDir) -> PathBuf {
            dir.path().join(CACHE_FILE_NAME)
        }

        #[test]
        fn a_persisted_ratio_calibrates_the_very_first_estimate_after_restart() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            // The measured glm-5.2 rate: 1.453 true tokens per estimated token.
            seed(&path, "z-ai/glm-5.2", 189_895, 275_924);
            let _g = isolated(&path);
            // No call recorded yet this "process" — the ratio comes off disk.
            let text = "a".repeat(400);
            assert_eq!(estimate_tokens_calibrated(&text, "z-ai/glm-5.2"), 146);
        }

        #[test]
        fn an_absent_cache_behaves_exactly_as_an_uncalibrated_store() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            let _g = isolated(&path);
            assert!(get_token_calibration().ratio("z-ai/glm-5.2").is_none());
            let text = "a".repeat(400);
            assert_eq!(estimate_tokens_calibrated(&text, "z-ai/glm-5.2"), 100);
        }

        #[test]
        fn a_corrupt_or_truncated_cache_loads_as_no_data() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            for junk in ["", "{", "garbage", r#"{"updated_at":"x","models":"#] {
                std::fs::write(&path, junk).unwrap();
                let _g = isolated(&path);
                assert!(
                    get_token_calibration().ratio("z-ai/glm-5.2").is_none(),
                    "junk {junk:?} must load as no data"
                );
            }
        }

        #[test]
        fn a_stale_cache_is_ignored() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            write_cache(
                &path,
                &BTreeMap::from([(
                    "old".to_owned(),
                    CachedTotals {
                        estimated: 100,
                        actual: 200,
                    },
                )]),
                now() - Duration::days(super::super::cache::CACHE_TTL_DAYS + 1),
            )
            .expect("write");
            let _g = isolated(&path);
            assert!(get_token_calibration().ratio("old").is_none());
        }

        #[test]
        fn the_debounce_skips_most_writes_but_eventually_persists() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            let _g = isolated(&path);
            let cal = get_token_calibration();

            // First sample of the process always lands: a one-call session must
            // not learn nothing.
            cal.record("m", 100, 150);
            let after_first = read_cache(&path).expect("first sample persisted");
            assert_eq!(
                after_first.models["m"],
                CachedTotals {
                    estimated: 100,
                    actual: 150,
                }
            );

            // The next FLUSH_EVERY_UPDATES - 1 stay in memory.
            for _ in 1..super::super::FLUSH_EVERY_UPDATES {
                cal.record("m", 100, 150);
            }
            assert_eq!(
                read_cache(&path).expect("still the first write").models,
                after_first.models,
                "debounce must not write on every update"
            );

            // Crossing the threshold persists everything accumulated.
            cal.record("m", 100, 150);
            let flushed = read_cache(&path).expect("threshold write");
            let n = i64::from(super::super::FLUSH_EVERY_UPDATES) + 1;
            assert_eq!(
                flushed.models["m"],
                CachedTotals {
                    estimated: 100 * n,
                    actual: 150 * n,
                }
            );
        }

        #[test]
        fn flush_persists_pending_updates() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            let _g = isolated(&path);
            let cal = get_token_calibration();
            cal.record("m", 100, 150); // immediate first write
            cal.record("m", 100, 150); // debounced
            cal.flush();
            assert_eq!(
                read_cache(&path).expect("flushed").models["m"],
                CachedTotals {
                    estimated: 200,
                    actual: 300,
                }
            );
        }

        #[test]
        fn a_round_trip_through_the_singleton_preserves_per_model_keys() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            {
                let _g = isolated(&path);
                let cal = get_token_calibration();
                cal.record("z-ai/glm-5.2", 189_895, 275_924);
                cal.record("z-ai/glm-5v-turbo", 53_159, 63_616);
                cal.flush();
            }
            // Fresh singleton — the "restart".
            let _g = isolated(&path);
            let cal = get_token_calibration();
            assert!((cal.ratio("z-ai/glm-5.2").expect("glm-5.2") - 1.453).abs() < 1e-3);
            assert!((cal.ratio("z-ai/glm-5v-turbo").expect("turbo") - 1.197).abs() < 1e-3);
        }

        #[test]
        fn persisted_totals_accumulate_across_restarts() {
            let dir = TempDir::new().unwrap();
            let path = cache_path(&dir);
            {
                let _g = isolated(&path);
                get_token_calibration().record("m", 100, 200);
            }
            let _g = isolated(&path);
            let cal = get_token_calibration();
            cal.record("m", 100, 100);
            // (200 + 100) / (100 + 100)
            assert!((cal.ratio("m").expect("a ratio") - 1.5).abs() < 1e-9);
        }

        #[test]
        fn a_pathless_store_never_touches_disk() {
            let dir = TempDir::new().unwrap();
            let cal = TokenCalibration::new();
            cal.record("m", 100, 150);
            cal.flush();
            assert_eq!(
                std::fs::read_dir(dir.path()).unwrap().count(),
                0,
                "TokenCalibration::new must stay in memory"
            );
        }
    }

    // --- TierBudget fields -------------------------------------------------

    #[test]
    fn overriding_one_field_leaves_others_at_default() {
        let a = TierBudget::default();
        let b = TierBudget {
            rag_tokens: 9999,
            ..TierBudget::default()
        };
        assert_eq!(b.rag_tokens, 9999);
        assert_eq!(b.recent_history_tokens, a.recent_history_tokens);
        assert_eq!(b.dossier_tokens, a.dossier_tokens);
        assert_eq!(b.max_dossier_people, a.max_dossier_people);
    }

    #[test]
    fn explicit_subcap_used_directly() {
        let b = TierBudget {
            recent_history_tokens: 500,
            ..TierBudget::default()
        };
        assert_eq!(b.recent_history_tokens, 500);
    }

    #[test]
    fn dataclass_default_is_voice_tier() {
        let b = TierBudget::default();
        assert_eq!(b.recent_history_tokens, 3000);
        assert_eq!(b.total_tokens(), 3000 + 900 + 900 + 600 + 600 + 600);
    }

    // --- TierBudget derived total ------------------------------------------

    #[test]
    fn total_is_sum_of_section_caps() {
        let b = TierBudget {
            recent_history_tokens: 1000,
            rag_tokens: 200,
            dossier_tokens: 200,
            summary_tokens: 100,
            reflection_tokens: 100,
            lorebook_tokens: 100,
            ..TierBudget::default()
        };
        assert_eq!(b.total_tokens(), 1000 + 200 + 200 + 100 + 100 + 100);
    }

    #[test]
    fn total_excludes_count_caps() {
        let base = TierBudget::default();
        let bumped = TierBudget {
            max_history_turns: base.max_history_turns + 50,
            ..TierBudget::default()
        };
        assert_eq!(bumped.total_tokens(), base.total_tokens());
    }

    #[test]
    fn total_tracks_a_section_cap_change() {
        let base = TierBudget::default();
        let bumped = TierBudget {
            rag_tokens: base.rag_tokens + 500,
            ..TierBudget::default()
        };
        assert_eq!(bumped.total_tokens(), base.total_tokens() + 500);
    }

    // --- ModelBudgetCurve --------------------------------------------------

    #[test]
    fn curve_defaults_are_all_one() {
        let c = ModelBudgetCurve::default();
        for v in [
            c.recent_history_tokens,
            c.rag_tokens,
            c.dossier_tokens,
            c.summary_tokens,
            c.reflection_tokens,
            c.lorebook_tokens,
            c.max_history_turns,
            c.max_rag_turns,
            c.max_rag_facts,
            c.max_dossier_people,
            c.max_reflections,
            c.max_lorebook_entries,
        ] {
            assert!((v - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn curve_partial_override_leaves_others_at_one() {
        let c = ModelBudgetCurve {
            recent_history_tokens: 2.0,
            rag_tokens: 1.5,
            ..ModelBudgetCurve::default()
        };
        assert!((c.recent_history_tokens - 2.0).abs() < f64::EPSILON);
        assert!((c.rag_tokens - 1.5).abs() < f64::EPSILON);
        assert!((c.dossier_tokens - 1.0).abs() < f64::EPSILON);
        assert!((c.summary_tokens - 1.0).abs() < f64::EPSILON);
    }

    // --- apply_curve -------------------------------------------------------

    #[test]
    fn identity_curve_returns_equivalent_budget() {
        let b = TierBudget {
            recent_history_tokens: 4000,
            rag_tokens: 500,
            ..TierBudget::default()
        };
        assert_eq!(b.apply_curve(&ModelBudgetCurve::default()), b);
    }

    #[test]
    fn scale_recent_history() {
        let b = TierBudget {
            recent_history_tokens: 1000,
            ..TierBudget::default()
        };
        let scaled = b.apply_curve(&ModelBudgetCurve {
            recent_history_tokens: 2.0,
            ..ModelBudgetCurve::default()
        });
        assert_eq!(scaled.recent_history_tokens, 2000);
        assert_eq!(scaled.rag_tokens, b.rag_tokens);
    }

    #[test]
    fn scale_rounds_to_nearest_int() {
        let b = TierBudget {
            rag_tokens: 1000,
            ..TierBudget::default()
        };
        let scaled = b.apply_curve(&ModelBudgetCurve {
            rag_tokens: 1.5,
            ..ModelBudgetCurve::default()
        });
        assert_eq!(scaled.rag_tokens, 1500);
    }

    #[test]
    fn scale_minimum_is_one() {
        let b = TierBudget {
            rag_tokens: 1,
            ..TierBudget::default()
        };
        let scaled = b.apply_curve(&ModelBudgetCurve {
            rag_tokens: 0.001,
            ..ModelBudgetCurve::default()
        });
        assert!(scaled.rag_tokens >= 1);
    }

    #[test]
    fn derived_total_follows_scaled_sections() {
        let b = TierBudget {
            recent_history_tokens: 2000,
            rag_tokens: 400,
            dossier_tokens: 400,
            summary_tokens: 200,
            reflection_tokens: 200,
            lorebook_tokens: 200,
            ..TierBudget::default()
        };
        let c = ModelBudgetCurve {
            recent_history_tokens: 2.0,
            rag_tokens: 1.5,
            dossier_tokens: 1.5,
            summary_tokens: 1.5,
            reflection_tokens: 1.5,
            lorebook_tokens: 1.5,
            ..ModelBudgetCurve::default()
        };
        let scaled = b.apply_curve(&c);
        assert_eq!(scaled.recent_history_tokens, 4000);
        assert_eq!(scaled.rag_tokens, 600);
        assert_eq!(scaled.dossier_tokens, 600);
        assert_eq!(scaled.total_tokens(), 4000 + 600 + 600 + 300 + 300 + 300);
    }

    #[test]
    fn scale_count_fields() {
        let b = TierBudget {
            max_rag_turns: 5,
            max_rag_facts: 3,
            max_reflections: 3,
            ..TierBudget::default()
        };
        let scaled = b.apply_curve(&ModelBudgetCurve {
            max_rag_turns: 2.0,
            max_rag_facts: 2.0,
            ..ModelBudgetCurve::default()
        });
        assert_eq!(scaled.max_rag_turns, 10);
        assert_eq!(scaled.max_rag_facts, 6);
        assert_eq!(scaled.max_reflections, b.max_reflections);
    }
}
