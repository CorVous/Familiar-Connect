//! Cold-cache signals — research-phase instrumentation.
//!
//! Port of `familiar_connect/diagnostics/cold_cache.py`. Three pure signal
//! detectors plus [`log_signals`], which runs all three and emits one INFO line
//! per firing signal on `familiar_connect.diagnostics.cold_cache`. Informational
//! only today — nothing drives cache invalidation from these yet.

use std::collections::HashSet;
use std::sync::LazyLock;

use chrono::{DateTime, Utc};
use regex::Regex;

use crate::log_style as ls;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\w']{3,}").expect("valid"));
static PROPER_NOUN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([A-Z][a-zA-Z]{2,})\b").expect("valid"));

// Capitalized discourse markers / sentence-starters, stored lowercase, matched
// case-insensitively. Incomplete by design (spec 01 §33).
static SENTENCE_STARTER_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "actually",
        "also",
        "and",
        "but",
        "for",
        "however",
        "just",
        "like",
        "maybe",
        "now",
        "oh",
        "okay",
        "really",
        "right",
        "since",
        "something",
        "sometimes",
        "still",
        "that",
        "the",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "well",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "yeah",
        "yes",
        "you",
        "your",
    ]
    .into_iter()
    .collect()
});

/// The default sentence-starter stopword set (lowercase surface forms).
#[must_use]
pub fn default_stopwords() -> HashSet<String> {
    SENTENCE_STARTER_STOPWORDS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn tokens(text: &str) -> HashSet<String> {
    WORD_RE
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

/// Detect when `new_text` shares too few content words with prior context.
///
/// Fires iff both token sets are non-empty, `len(new_tokens) >= min_tokens`, and
/// the Jaccard overlap is `< min_overlap`. The min-token floor keeps short voice
/// fragments from firing (spec 01 §32).
#[must_use]
pub fn detect_topic_shift(
    new_text: &str,
    prior_context: &str,
    min_overlap: f64,
    min_tokens: usize,
) -> bool {
    let new_tokens = tokens(new_text);
    let old_tokens = tokens(prior_context);
    if new_tokens.is_empty() || old_tokens.is_empty() {
        return false;
    }
    if new_tokens.len() < min_tokens {
        return false;
    }
    let inter = new_tokens.intersection(&old_tokens).count();
    let union = new_tokens.union(&old_tokens).count();
    #[allow(clippy::cast_precision_loss)] // token counts are small
    let overlap = inter as f64 / union as f64;
    overlap < min_overlap
}

/// Return proper nouns in `new_text` absent from `prior_context`.
///
/// Proper noun = capitalized word of 3+ letters. Matches whose lowercase form is
/// in `stopwords` are skipped; duplicates deduped by exact surface (first-seen
/// order); a noun is reported iff its lowercase form does not occur as a
/// **substring** of `prior_context.lower()` (spec 01 §33).
#[must_use]
#[allow(clippy::implicit_hasher)] // callers use the default hasher
pub fn detect_unknown_proper_noun(
    new_text: &str,
    prior_context: &str,
    stopwords: &HashSet<String>,
) -> Vec<String> {
    let prior_lower = prior_context.to_lowercase();
    let mut unknowns: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for caps in PROPER_NOUN_RE.captures_iter(new_text) {
        let matched = caps[1].to_string();
        let lowered = matched.to_lowercase();
        if stopwords.contains(&lowered) {
            continue;
        }
        if seen.contains(&matched) {
            continue;
        }
        seen.insert(matched.clone());
        if !prior_lower.contains(&lowered) {
            unknowns.push(matched);
        }
    }
    unknowns
}

/// Return the gap in seconds if it meets or exceeds `threshold_seconds`.
///
/// `None` if there is no prior turn or the gap is below threshold; a gap exactly
/// equal to the threshold **fires** (spec 01 §34).
#[must_use]
pub fn detect_silence_gap(
    prev_turn_at: Option<DateTime<Utc>>,
    current_turn_at: DateTime<Utc>,
    threshold_seconds: f64,
) -> Option<f64> {
    let prev = prev_turn_at?;
    let delta = current_turn_at - prev;
    #[allow(clippy::cast_precision_loss)] // microsecond counts are small
    let gap = delta
        .num_microseconds()
        .map_or_else(|| delta.num_seconds() as f64, |us| us as f64 / 1_000_000.0);
    if gap < threshold_seconds {
        return None;
    }
    Some(gap)
}

/// Which signals fired in [`log_signals`]. A key is "present" (Python `in`) when
/// its accessor is truthy: `topic_shift` when `true`, `unknown_proper_nouns`
/// when non-empty, `silence_gap_s` when `Some`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fired {
    /// Whether the topic-shift signal fired.
    pub topic_shift: bool,
    /// Proper nouns reported (empty when the signal did not fire).
    pub unknown_proper_nouns: Vec<String>,
    /// Silence gap in seconds (`None` when the signal did not fire).
    pub silence_gap_s: Option<f64>,
}

/// Run all detectors; emit one INFO line per firing signal, tagged with the
/// inbound turn's channel id. Returns which signals fired.
#[must_use]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword-only signature
pub fn log_signals(
    channel_id: i64,
    turn_id: &str,
    new_text: &str,
    prior_context: &str,
    prev_turn_at: Option<DateTime<Utc>>,
    current_turn_at: DateTime<Utc>,
    topic_shift_threshold: f64,
    topic_shift_min_tokens: usize,
    silence_gap_threshold_s: f64,
) -> Fired {
    let mut fired = Fired::default();

    if detect_topic_shift(
        new_text,
        prior_context,
        topic_shift_threshold,
        topic_shift_min_tokens,
    ) {
        fired.topic_shift = true;
        tracing::info!(
            target: "familiar_connect.diagnostics.cold_cache",
            "{}",
            format!(
                "{} {} {} {}",
                ls::tag("ColdCache", ls::LY),
                ls::kv_styled("signal", "topic_shift", ls::W, ls::LY),
                ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
                ls::kv_styled("turn", turn_id, ls::W, ls::LC),
            )
        );
    }

    let unknowns = detect_unknown_proper_noun(new_text, prior_context, &default_stopwords());
    if !unknowns.is_empty() {
        let nouns = unknowns
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(",");
        tracing::info!(
            target: "familiar_connect.diagnostics.cold_cache",
            "{}",
            format!(
                "{} {} {} {} {}",
                ls::tag("ColdCache", ls::LY),
                ls::kv_styled("signal", "unknown_proper_noun", ls::W, ls::LY),
                ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
                ls::kv_styled("turn", turn_id, ls::W, ls::LC),
                ls::kv_styled("nouns", &nouns, ls::W, ls::LW),
            )
        );
        fired.unknown_proper_nouns = unknowns;
    }

    let gap = detect_silence_gap(prev_turn_at, current_turn_at, silence_gap_threshold_s);
    if let Some(gap) = gap {
        tracing::info!(
            target: "familiar_connect.diagnostics.cold_cache",
            "{}",
            format!(
                "{} {} {} {} {}",
                ls::tag("ColdCache", ls::LY),
                ls::kv_styled("signal", "silence_gap", ls::W, ls::LY),
                ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
                ls::kv_styled("turn", turn_id, ls::W, ls::LC),
                ls::kv_styled("gap_s", &format!("{gap:.1}"), ls::W, ls::LW),
            )
        );
        fired.silence_gap_s = Some(gap);
    }

    fired
}

#[cfg(test)]
mod tests {
    use super::{
        Fired, default_stopwords, detect_silence_gap, detect_topic_shift,
        detect_unknown_proper_noun, log_signals,
    };
    use crate::diagnostics::strip_ansi;
    use crate::diagnostics::testutil::{Capture, install_capture};
    use chrono::{Duration, Utc};
    use std::collections::HashSet;

    // Python default kwargs made explicit at each call site.
    const OVERLAP: f64 = 0.15;
    const MIN_TOKENS: usize = 4;
    const GAP_THRESHOLD: f64 = 300.0;

    // --- topic shift ---

    #[test]
    fn topic_shift_fires_when_vocab_disjoint() {
        assert!(detect_topic_shift(
            "Tell me about the submarine propulsion",
            "Discussed fox diets and foraging patterns",
            OVERLAP,
            MIN_TOKENS,
        ));
    }

    #[test]
    fn topic_shift_quiet_when_vocab_overlaps() {
        assert!(!detect_topic_shift(
            "More about foxes and foraging",
            "Discussed fox diets and foraging patterns",
            0.1,
            MIN_TOKENS,
        ));
    }

    #[test]
    fn topic_shift_quiet_when_prior_is_empty() {
        assert!(!detect_topic_shift("anything", "", OVERLAP, MIN_TOKENS));
    }

    #[test]
    fn topic_shift_quiet_for_short_voice_fragments() {
        assert!(!detect_topic_shift(
            "Oh, dear.",
            "earlier we discussed submarines, foxes, foraging, and reasonable fast food prices",
            OVERLAP,
            MIN_TOKENS,
        ));
    }

    #[test]
    fn topic_shift_fires_once_above_min_tokens() {
        assert!(detect_topic_shift(
            "submarine propulsion really fascinates engineers",
            "discussed fox diets and foraging patterns",
            OVERLAP,
            MIN_TOKENS,
        ));
    }

    #[test]
    fn topic_shift_min_tokens_is_configurable() {
        assert!(detect_topic_shift(
            "submarine propulsion",
            "discussed foxes",
            OVERLAP,
            2
        ));
    }

    // --- unknown proper noun ---

    fn defaults() -> HashSet<String> {
        default_stopwords()
    }

    #[test]
    fn proper_noun_finds_new_capitalized_tokens() {
        let nouns = detect_unknown_proper_noun(
            "I'm meeting Aria and Boris at the pub.",
            "We were talking about puddles.",
            &defaults(),
        );
        assert!(nouns.contains(&"Aria".to_string()));
        assert!(nouns.contains(&"Boris".to_string()));
    }

    #[test]
    fn proper_noun_skips_known_capitalized_tokens() {
        let nouns =
            detect_unknown_proper_noun("Aria said hi.", "Aria joined the channel.", &defaults());
        assert!(nouns.is_empty());
    }

    #[test]
    fn proper_noun_ignores_short_tokens() {
        let nouns = detect_unknown_proper_noun("Is OK with me", "nothing relevant", &defaults());
        assert!(nouns.is_empty());
    }

    #[test]
    fn proper_noun_skips_sentence_starter_stopwords() {
        let nouns = detect_unknown_proper_noun(
            "Which means I don't have storage. But okay. Yeah.",
            "nothing relevant",
            &defaults(),
        );
        assert!(nouns.is_empty());
    }

    #[test]
    fn proper_noun_real_alongside_starter_still_fires() {
        let nouns = detect_unknown_proper_noun(
            "But Aria mentioned something.",
            "nothing relevant",
            &defaults(),
        );
        assert!(nouns.contains(&"Aria".to_string()));
        assert!(!nouns.contains(&"But".to_string()));
    }

    #[test]
    fn proper_noun_stopwords_kwarg_overrides_default() {
        let custom: HashSet<String> = HashSet::from(["aria".to_string()]);
        let nouns = detect_unknown_proper_noun("Aria said hi.", "nothing relevant", &custom);
        assert!(nouns.is_empty());
    }

    // --- silence gap ---

    #[test]
    fn silence_gap_fires_past_threshold() {
        let t0 = Utc::now();
        let t1 = t0 + Duration::minutes(10);
        let gap = detect_silence_gap(Some(t0), t1, GAP_THRESHOLD);
        assert!(gap.is_some());
        assert!(gap.unwrap() >= 600.0);
    }

    #[test]
    fn silence_gap_quiet_below_threshold() {
        let t0 = Utc::now();
        let t1 = t0 + Duration::seconds(30);
        assert!(detect_silence_gap(Some(t0), t1, GAP_THRESHOLD).is_none());
    }

    #[test]
    fn silence_gap_quiet_when_no_prior() {
        let t1 = Utc::now();
        assert!(detect_silence_gap(None, t1, GAP_THRESHOLD).is_none());
    }

    // --- log_signals ---

    #[test]
    fn log_signals_emits_spans_for_firing_signals() {
        let cap = Capture::default();
        let _sub = install_capture(&cap);
        let t0 = Utc::now();
        let t1 = t0 + Duration::minutes(10);
        let fired = log_signals(
            42,
            "t-1",
            "Meet Aria about the submarine.",
            "We discussed foxes.",
            Some(t0),
            t1,
            OVERLAP,
            MIN_TOKENS,
            GAP_THRESHOLD,
        );
        let messages: Vec<String> = cap
            .records()
            .iter()
            .map(|r| strip_ansi(&r.message))
            .collect();
        assert!(fired.topic_shift);
        assert!(!fired.unknown_proper_nouns.is_empty());
        assert!(fired.silence_gap_s.is_some());
        assert!(messages.iter().any(|m| m.contains("signal=topic_shift")));
        assert!(
            messages
                .iter()
                .any(|m| m.contains("signal=unknown_proper_noun"))
        );
        assert!(messages.iter().any(|m| m.contains("signal=silence_gap")));
    }

    #[test]
    fn log_signals_short_fragment_emits_no_topic_shift() {
        let cap = Capture::default();
        let _sub = install_capture(&cap);
        let now = Utc::now();
        let fired = log_signals(
            1,
            "t",
            "Oh, dear.",
            "we were chatting about submarines and foraging foxes",
            Some(now - Duration::seconds(5)),
            now,
            OVERLAP,
            MIN_TOKENS,
            GAP_THRESHOLD,
        );
        assert!(!fired.topic_shift);
        assert!(
            !cap.records()
                .iter()
                .any(|r| strip_ansi(&r.message).contains("signal=topic_shift"))
        );
    }

    #[test]
    fn log_signals_topic_shift_min_tokens_plumbed_through() {
        let cap = Capture::default();
        let _sub = install_capture(&cap);
        let now = Utc::now();
        let fired = log_signals(
            1,
            "t",
            "submarine propulsion",
            "foxes foraging",
            Some(now - Duration::seconds(5)),
            now,
            OVERLAP,
            2,
            GAP_THRESHOLD,
        );
        assert!(fired.topic_shift);
    }

    #[test]
    fn log_signals_no_spans_when_nothing_fires() {
        let cap = Capture::default();
        let _sub = install_capture(&cap);
        let now = Utc::now();
        let fired = log_signals(
            1,
            "t",
            "same topic as before Aria mentioned it",
            "aria was discussing this topic before",
            Some(now - Duration::seconds(5)),
            now,
            OVERLAP,
            MIN_TOKENS,
            GAP_THRESHOLD,
        );
        assert_eq!(fired, Fired::default());
        assert!(
            !cap.records()
                .iter()
                .any(|r| strip_ansi(&r.message).contains("signal="))
        );
    }
}
