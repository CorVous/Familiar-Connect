//! Shared Unicode-scalar-safe text truncation (DESIGN §4.9).
//!
//! All truncation caps in the port count **Unicode scalars**, never bytes —
//! byte slicing lands mid-codepoint on emoji-heavy chat text. [`truncate`]
//! mirrors the Python `log_style.trunc` helper: keep the first `limit` scalars
//! and append the U+2026 ellipsis (`…`) only when the input was longer than the
//! cap. The ellipsis is *appended*, so the result is at most `limit + 1`
//! scalars — matching Python's `f"{text[:limit]}{'…' if len(text) > limit …}"`.

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

#[cfg(test)]
mod tests {
    use super::{ELLIPSIS, truncate};

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
}
