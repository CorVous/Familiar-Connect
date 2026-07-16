//! Token estimator + `TierBudget` / `ModelBudgetCurve` (subsystem 05; Python `budget.py`).
//!
//! Per-tier prompt-assembly budget. Each cap is a hard number — no proportional
//! derivation, no "auto-fill from total". The assembly layers consume the values
//! directly, each self-truncating to its own cap. The whole-prompt
//! [`TierBudget::total_tokens`] is a *derived* sum of the per-section token caps,
//! for reporting only — nothing trims against it.
//!
//! Token accounting uses the fast `len(text)/4` heuristic (DESIGN §4.9,
//! chars-per-token 4) — no real tokenizer on the hot path; it slightly
//! over-counts (safer for budgets). `len` counts **Unicode scalars**, not bytes.
//!
//! This is a leaf module: it names [`crate::llm::Message`] but pulls in nothing
//! from `config`/`context`, so `config` can depend on it without a cycle (DESIGN
//! D4).

use crate::llm::Message;
use crate::support::round::half_even;

/// OpenAI's well-known English heuristic; slightly over-counts.
const CHARS_PER_TOKEN: i64 = 4;

/// Per-message chat-format framing (role + delimiters).
const MESSAGE_OVERHEAD_TOKENS: i64 = 4;

/// Fast char-based token estimate: `ceil(len / 4)` over Unicode scalars, `0` for
/// the empty string. Over-counts mildly so budgets stay safe.
#[must_use]
pub fn estimate_tokens(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    // `len` in Python is the Unicode scalar count, not bytes (DESIGN §4.9).
    let n = i64::try_from(text.chars().count()).unwrap_or(i64::MAX);
    (n + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Chat-format estimate including role/name framing:
/// `estimate_tokens(content_str) + 4` (+ `estimate_tokens(name)` when `name` set).
#[must_use]
pub fn estimate_message_tokens(msg: &Message) -> i64 {
    let mut n = estimate_tokens(&msg.content_str()) + MESSAGE_OVERHEAD_TOKENS;
    if let Some(name) = &msg.name {
        // An empty name adds `estimate_tokens("") == 0`, matching Python's
        // truthy `if msg.name:` guard.
        n += estimate_tokens(name);
    }
    n
}

/// Sum [`estimate_message_tokens`] across a message list.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> i64 {
    messages.iter().map(estimate_message_tokens).sum()
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
/// with Python's banker's rounding (half-to-even, DESIGN D10).
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
        ModelBudgetCurve, TierBudget, estimate_message_tokens, estimate_messages_tokens,
        estimate_tokens,
    };
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
