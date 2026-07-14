//! Crash-safe `{key}` placeholder fill for config-sourced prompt templates
//! (subsystem 02; Python `prompt_fill.py`).
//!
//! Config-sourced (per-familiar overridable) prompt text is filled with dynamic
//! values (`{self_name}` etc.) at build time. String formatting that raised on a
//! stray brace, an unknown token, or a missing expected one would let a phrasing
//! override crash a pass. [`fill_placeholders`] fills by literal substitution:
//! only the supplied `{key}` tokens are replaced; everything else (stray braces,
//! unknown tokens) passes through verbatim. It never panics.
//!
//! Single pass / no re-expansion: each `{key}` token is filled exactly once over
//! the *original* template — an injected value containing another key's token is
//! left literal, never re-scanned. Order-independent.

use regex::{Captures, Regex};
use std::sync::LazyLock;

static TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{(\w+)\}").expect("valid token regex"));

/// Replace each `{key}` in `template` with its value from `values`.
///
/// `values` is a slice of `(key, replacement)` pairs; the first matching key
/// wins. Callers perform any `Display` conversion before calling (Python does
/// `str(value)`). Unknown `{...}` tokens and stray braces pass through unchanged.
#[must_use]
pub fn fill_placeholders(template: &str, values: &[(&str, &str)]) -> String {
    TOKEN_RE
        .replace_all(template, |caps: &Captures<'_>| {
            let key: &str = &caps[1];
            for (k, v) in values {
                if *k == key {
                    return (*v).to_owned();
                }
            }
            caps[0].to_owned()
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::fill_placeholders;

    #[test]
    fn fills_known_placeholder() {
        assert_eq!(
            fill_placeholders("hi {name}", &[("name", "Sapphire")]),
            "hi Sapphire"
        );
    }

    #[test]
    fn unknown_placeholder_passes_through() {
        assert_eq!(
            fill_placeholders("hi {name} {other}", &[("name", "Sapphire")]),
            "hi Sapphire {other}"
        );
    }

    #[test]
    fn missing_placeholder_passes_through() {
        // template lacks the supplied key — no error, unchanged.
        assert_eq!(
            fill_placeholders("plain text", &[("name", "Sapphire")]),
            "plain text"
        );
    }

    #[test]
    fn stray_braces_pass_through() {
        assert_eq!(
            fill_placeholders("a { b } c {name}", &[("name", "X")]),
            "a { b } c X"
        );
    }

    #[test]
    fn injected_value_is_not_re_expanded() {
        // Single pass: a value containing another key's token stays literal.
        // Chained per-key replacement would re-scan `a`'s injected `{b}` and
        // expand it to `X` ("X X"). One pass fills each token exactly once.
        assert_eq!(
            fill_placeholders("{a} {b}", &[("a", "{b}"), ("b", "X")]),
            "{b} X"
        );
    }

    #[test]
    fn multiple_tokens_and_repeated_keys_all_fill() {
        assert_eq!(
            fill_placeholders("{g} {n} — {g}", &[("g", "hi"), ("n", "Sapphire")]),
            "hi Sapphire — hi"
        );
    }
}
