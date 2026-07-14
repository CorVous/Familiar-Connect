//! Silent-sentinel detection for LLM reply streams (subsystem 06; Python
//! `silence.py`).
//!
//! The system prompt tells the model to emit `<silent>` as its entire reply
//! when staying silent. [`SilentDetector`] watches streamed deltas and decides
//! as soon as possible, so a responder aborts before paying downstream cost
//! (Discord post, TTS synthesis).
//!
//! Prefix-only: a stray `<silent>` mid-reply is content, not a gate.

/// The model-output sentinel that gates a whole reply into silence.
pub const SILENT_TOKEN: &str = "<silent>";

/// Streaming inspector for the silent sentinel.
///
/// [`feed`](SilentDetector::feed) returns `Some(true)` once the leading
/// non-whitespace matches [`SILENT_TOKEN`], `Some(false)` on mismatch, and
/// `None` while undecided. The decision **latches** — further calls return the
/// same value and ignore their argument.
#[derive(Debug, Default)]
pub struct SilentDetector {
    buf: String,
    decided: Option<bool>,
}

impl SilentDetector {
    /// A fresh, undecided detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The latched decision: `Some(true)` silent, `Some(false)` speak, `None`
    /// pending.
    #[must_use]
    pub const fn decided(&self) -> Option<bool> {
        self.decided
    }

    /// Feed one streamed delta; returns the (possibly newly latched) decision.
    pub fn feed(&mut self, delta: &str) -> Option<bool> {
        if let Some(decided) = self.decided {
            return Some(decided);
        }
        self.buf.push_str(delta);
        let stripped = self.buf.trim_start();
        if stripped.starts_with(SILENT_TOKEN) {
            self.decided = Some(true);
            return Some(true);
        }
        // Length compare in Unicode scalars (the token is ASCII, so multibyte
        // content that reaches 8+ chars decides `false` either way).
        if stripped.chars().count() >= SILENT_TOKEN.chars().count() {
            // Enough non-whitespace seen to rule out the sentinel.
            self.decided = Some(false);
            return Some(false);
        }
        if !stripped.is_empty() && !SILENT_TOKEN.starts_with(stripped) {
            // Diverged before reaching full length.
            self.decided = Some(false);
            return Some(false);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{SILENT_TOKEN, SilentDetector};

    #[test]
    fn pending_until_first_chars() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed(""), None);
        assert_eq!(d.decided(), None);
    }

    #[test]
    fn detects_full_token_in_one_delta() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed(SILENT_TOKEN), Some(true));
        assert_eq!(d.decided(), Some(true));
    }

    #[test]
    fn detects_token_split_across_deltas() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("<sil"), None);
        assert_eq!(d.feed("ent>"), Some(true));
    }

    #[test]
    fn tolerates_leading_whitespace() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("   "), None);
        assert_eq!(d.feed("<silent>"), Some(true));
    }

    #[test]
    fn rejects_token_not_at_prefix() {
        // Mid-reply `<silent>` is content, not a gate.
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("Sure, "), Some(false));
        assert_eq!(d.feed("<silent>"), Some(false));
    }

    #[test]
    fn rejects_normal_content() {
        let mut d = SilentDetector::new();
        // 'H' diverges immediately from '<'.
        assert_eq!(d.feed("Hello world"), Some(false));
    }

    #[test]
    fn rejects_when_diverges_mid_token() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("<sil"), None);
        // 'k' diverges from expected 'e'.
        assert_eq!(d.feed("k"), Some(false));
    }

    #[test]
    fn decision_latches() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("<silent>"), Some(true));
        // Subsequent feeds return the cached decision without re-inspecting.
        assert_eq!(d.feed("anything goes here"), Some(true));
    }

    #[test]
    fn decision_latches_for_speak() {
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("Hi "), Some(false));
        assert_eq!(d.feed("<silent>"), Some(false));
    }

    #[test]
    fn long_run_of_whitespace_stays_pending() {
        // Pure whitespace can't decide; the empty-reply guard handles it.
        let mut d = SilentDetector::new();
        assert_eq!(d.feed("\n\n\n   "), None);
        assert_eq!(d.decided(), None);
    }

    #[test]
    fn parameterized() {
        let cases: &[(&[&str], bool)] = &[
            (&["<", "s", "i", "l", "e", "n", "t", ">"], true),
            (&["  <silent>"], true),
            (&["<silently I disagree>"], false),
            (&["  Hello"], false),
        ];
        for (deltas, expected) in cases {
            let mut d = SilentDetector::new();
            let mut result: Option<bool> = None;
            for delta in *deltas {
                result = d.feed(delta);
            }
            assert_eq!(result, Some(*expected), "deltas={deltas:?}");
        }
    }
}
