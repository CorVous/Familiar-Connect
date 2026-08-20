//! Shared Unicode-scalar-safe text helpers: truncation + the speakable-text
//! predicate the voice path gates TTS on.
//!
//! All truncation caps in the port count **Unicode scalars**, never bytes —
//! byte slicing lands mid-codepoint on emoji-heavy chat text. [`truncate`]
//! keeps the first `limit` scalars and appends the U+2026 ellipsis (`…`) only
//! when the input was longer than the cap. The ellipsis is *appended*, so the
//! result is at most `limit + 1` scalars.

/// The ellipsis glyph appended by [`truncate`] (U+2026, `…`).
pub const ELLIPSIS: char = '\u{2026}';

/// Truncate `text` to at most `limit` Unicode scalars, appending `…` if it was
/// longer than `limit`. Counts scalars, never bytes.
#[must_use]
pub fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() > limit {
        let mut out: String = text.chars().take(limit).collect();
        out.push(ELLIPSIS);
        out
    } else {
        text.to_owned()
    }
}

/// Whether `text` holds anything a TTS engine can voice: at least one letter or
/// digit, in any script.
///
/// Drawn at alphanumeric because that is what a synthesizer turns into sound.
/// Whitespace, punctuation, markdown, and emoji carry no phonemes on their own,
/// and Cartesia rejects such a transcript with HTTP 400 — the live failure was
/// a reply ending in a trailing emoji, which the sentence streamer hands over
/// alone as its flush tail. Symbols *beside* words are untouched: the predicate
/// asks only whether anything voiceable is present.
#[must_use]
pub fn is_speakable(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

#[cfg(test)]
mod tests {
    use super::{ELLIPSIS, is_speakable, truncate};

    #[test]
    fn shorter_than_limit_is_unchanged() {
        assert_eq!(truncate("abc", 5), "abc");
    }

    #[test]
    fn exactly_at_limit_gets_no_ellipsis() {
        // len(text) > limit is false at exactly the cap — no ellipsis.
        assert_eq!(truncate("abcde", 5), "abcde");
    }

    #[test]
    fn longer_than_limit_appends_ellipsis() {
        assert_eq!(truncate("abcdef", 5), "abcde\u{2026}");
        assert_eq!(ELLIPSIS, '\u{2026}');
    }

    #[test]
    fn counts_unicode_scalars_not_bytes() {
        // Each emoji is 4 UTF-8 bytes but a single scalar. Truncating to 3 keeps
        // three whole emoji plus the ellipsis, never slicing mid-codepoint.
        let s = "😀😀😀😀😀";
        let out = truncate(s, 3);
        assert_eq!(out, "😀😀😀\u{2026}");
        assert_eq!(out.chars().count(), 4);
    }

    #[test]
    fn multibyte_at_exact_boundary_is_not_split() {
        // "café" is 4 scalars (5 bytes). A cap of 4 keeps it whole.
        assert_eq!(truncate("café", 4), "café");
        // A cap of 3 keeps "caf" + ellipsis, not a broken 'é'.
        assert_eq!(truncate("café", 3), "caf\u{2026}");
    }

    // --- is_speakable ------------------------------------------------------

    #[test]
    fn words_and_digits_are_speakable() {
        assert!(is_speakable("Hello, world."));
        assert!(is_speakable("42"));
        assert!(is_speakable("Pi is 3.14."));
        // Non-Latin scripts count as letters.
        assert!(is_speakable("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}"));
        assert!(is_speakable("\u{43f}\u{440}\u{438}\u{432}\u{435}\u{442}"));
    }

    #[test]
    fn empty_and_whitespace_are_not_speakable() {
        assert!(!is_speakable(""));
        assert!(!is_speakable("   \n\t"));
        // Info separators U+001C-U+001F: `str::trim` leaves them, but they
        // carry no speakable content either.
        assert!(!is_speakable("\u{1c}\u{1f}"));
    }

    #[test]
    fn punctuation_only_is_not_speakable() {
        assert!(!is_speakable("..."));
        assert!(!is_speakable("?!"));
        assert!(!is_speakable(" \u{2014} "));
        assert!(!is_speakable("**"));
    }

    #[test]
    fn emoji_only_is_not_speakable() {
        // The live 400: a reply ending in a trailing emoji leaves the emoji
        // alone as the sentence streamer's flush tail.
        assert!(!is_speakable("\u{1f605}"));
        assert!(!is_speakable(" \u{1f49b}\u{1f49a} "));
        // ZWJ sequences and variation selectors are symbols too.
        assert!(!is_speakable("\u{1f469}\u{200d}\u{1f4bb}"));
        assert!(!is_speakable("\u{2764}\u{fe0f}"));
    }

    #[test]
    fn emoji_beside_words_stays_speakable() {
        assert!(is_speakable("\u{1f605} Who am I missing?"));
        assert!(is_speakable("Nice work. \u{1f49b}"));
    }
}
