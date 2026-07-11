//! Sentence-boundary aggregator for streamed LLM output (subsystem 09; Python
//! `sentence_streamer.py`).
//!
//! Sits between the LLM content stream and the TTS player. Buffers deltas and
//! emits whole sentences as soon as a terminator (`.` / `!` / `?`) is followed
//! by whitespace — dropping time-to-first-audio from "after the LLM finishes" to
//! "after the first sentence". [`SentenceStreamer::flush`] drains the partial
//! buffer when the stream ends without a final terminator.
//!
//! Abbreviation-aware: `Mr.` / `Dr.` / `etc.` / single-letter initials do not
//! trip a boundary. The decision is local — no language-model lookahead.

/// Titles + low-risk Latin abbreviations (lowercase, no trailing dot). Kept
/// tight: a false negative (one extra sentence) is cheaper than a false positive
/// (mid-sentence flush of `Mr.`).
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "rev", "fr", "etc", "vs", "no", "vol", "pg",
    "ft", "e.g", "i.e",
];

/// Sentence terminators.
const TERMINATORS: &[char] = &['.', '!', '?'];

fn is_terminator(ch: char) -> bool {
    TERMINATORS.contains(&ch)
}

/// Python `str.isspace()` classification for a single Unicode scalar.
///
/// Rust's [`char::is_whitespace`] tracks the Unicode `White_Space` property,
/// which — unlike CPython's `str.isspace()` — excludes the four ASCII
/// information separators U+001C–U+001F (FS/GS/RS/US). Python classifies those
/// as whitespace (their bidirectional class is `B`/`S`), so a terminator run
/// followed by one of them *is* a sentence boundary there and the separator is
/// consumed with the rest of the trailing whitespace. Spec 09 invariant 67
/// pins the boundary contract to `isspace()`, so match it exactly rather than
/// leaning on `char::is_whitespace` (which would withhold the sentence until a
/// real space or flush). The two classifications agree on every other scalar.
fn is_py_whitespace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}')
}

/// Buffers streamed text and emits on sentence boundaries.
#[derive(Clone, Debug, Default)]
pub struct SentenceStreamer {
    buf: String,
}

impl SentenceStreamer {
    /// A fresh streamer with an empty buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Append `delta`; return zero or more completed sentences.
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(delta);
        let mut out = Vec::new();
        while let Some((head, rest)) = self.try_split() {
            out.push(head);
            self.buf = rest;
        }
        out
    }

    /// Drain the remaining buffer verbatim (untrimmed); resets internal state.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    /// Find the earliest non-abbreviation terminator run followed by whitespace
    /// and split there, returning `(sentence_including_terminators, remainder)`.
    ///
    /// Operates over `char`s so multi-byte scalars (emoji, accents) never land
    /// mid-codepoint — matching Python's code-point indexing.
    fn try_split(&self) -> Option<(String, String)> {
        let chars: Vec<char> = self.buf.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            if !is_terminator(chars[i]) {
                i += 1;
                continue;
            }
            // Eat consecutive terminators ("?!" → one boundary).
            let mut end = i + 1;
            while end < n && is_terminator(chars[end]) {
                end += 1;
            }
            if end >= n {
                // Punctuation at buffer end — wait for the next delta.
                return None;
            }
            if !is_py_whitespace(chars[end]) {
                // ".5" / "1.0" / "?<tag>" — not a boundary.
                i = end;
                continue;
            }
            if chars[i..end].contains(&'.') && looks_like_abbreviation(&chars, i) {
                i = end;
                continue;
            }
            let head: String = chars[..end].iter().collect();
            // Consume all following whitespace so the next sentence starts clean.
            let mut rest_start = end;
            while rest_start < n && is_py_whitespace(chars[rest_start]) {
                rest_start += 1;
            }
            let rest: String = chars[rest_start..].iter().collect();
            return Some((head, rest));
        }
        None
    }
}

/// `dot_index` indexes the start of the terminator run; walk back the preceding
/// token and decide whether it is an abbreviation or a single-letter initial.
fn looks_like_abbreviation(chars: &[char], dot_index: usize) -> bool {
    // Collect the token immediately preceding the run, allowing inner dots so
    // "e.g" / "i.e" round-trip. `is_alphabetic` matches Python's `str.isalpha`
    // for every letter that can plausibly appear in chat text (categories
    // Lu/Ll/Lt/Lm/Lo, incl. all accented forms); the two diverge only on
    // exotic Nl / Other_Alphabetic scalars that never front an abbreviation or
    // single-letter initial, so the walk-back is faithful in practice.
    let mut start = dot_index;
    while start > 0 && (chars[start - 1].is_alphabetic() || chars[start - 1] == '.') {
        start -= 1;
    }
    let token: String = chars[start..dot_index]
        .iter()
        .collect::<String>()
        .to_lowercase();
    if token.is_empty() {
        return false;
    }
    if ABBREVIATIONS.contains(&token.as_str()) {
        return true;
    }
    // Single-letter initial: "J. K. Rowling".
    token.chars().count() == 1 && token.chars().next().is_some_and(char::is_alphabetic)
}

#[cfg(test)]
mod tests {
    use super::SentenceStreamer;

    // --- simple splitting --------------------------------------------------

    #[test]
    fn no_boundary_yet_buffers() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("Hello").is_empty());
        assert!(s.feed(", world").is_empty());
    }

    #[test]
    fn period_then_space_emits_sentence() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("Hello, world. "), vec!["Hello, world."]);
    }

    #[test]
    fn period_at_end_buffers_until_space_or_flush() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("Hello, world.").is_empty());
        assert_eq!(s.flush(), "Hello, world.");
    }

    #[test]
    fn question_and_exclamation_are_boundaries() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("Really? "), vec!["Really?"]);
        assert_eq!(s.feed("Wow! "), vec!["Wow!"]);
    }

    #[test]
    fn two_sentences_in_one_delta() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("First. Second. "), vec!["First.", "Second."]);
    }

    #[test]
    fn split_across_multiple_deltas() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("Hello").is_empty());
        assert!(s.feed(", ").is_empty());
        assert!(s.feed("world").is_empty());
        assert!(s.feed(".").is_empty());
        assert_eq!(s.feed(" How"), vec!["Hello, world."]);
        assert!(s.feed(" are you?").is_empty());
        assert_eq!(s.feed(" "), vec!["How are you?"]);
    }

    // --- abbreviations -----------------------------------------------------

    #[test]
    fn title_abbreviations_do_not_split() {
        for abbrev in [
            "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Sr.", "Jr.", "Prof.", "Rev.",
        ] {
            let mut s = SentenceStreamer::new();
            let input = format!("{abbrev} Smith arrived. ");
            let expected = format!("{abbrev} Smith arrived.");
            assert_eq!(s.feed(&input), vec![expected]);
        }
    }

    #[test]
    fn etc_does_not_split() {
        let mut s = SentenceStreamer::new();
        assert_eq!(
            s.feed("Apples, pears, etc. are fine. "),
            vec!["Apples, pears, etc. are fine."]
        );
    }

    #[test]
    fn eg_and_ie_do_not_split() {
        let mut s = SentenceStreamer::new();
        assert_eq!(
            s.feed("Some fruits, e.g. apples, work. "),
            vec!["Some fruits, e.g. apples, work."]
        );
        let mut s2 = SentenceStreamer::new();
        assert_eq!(
            s2.feed("That is, i.e. always. "),
            vec!["That is, i.e. always."]
        );
    }

    #[test]
    fn single_letter_initial_does_not_split() {
        let mut s = SentenceStreamer::new();
        assert_eq!(
            s.feed("J. K. Rowling wrote it. "),
            vec!["J. K. Rowling wrote it."]
        );
    }

    // --- flush -------------------------------------------------------------

    #[test]
    fn flush_drains_partial_buffer_verbatim() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("d0 d1 d2 ").is_empty());
        assert_eq!(s.flush(), "d0 d1 d2 ");
    }

    #[test]
    fn flush_after_emitted_sentence_returns_remainder() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("First. "), vec!["First."]);
        assert!(s.feed("partial").is_empty());
        assert_eq!(s.flush(), "partial");
    }

    #[test]
    fn flush_resets_buffer() {
        let mut s = SentenceStreamer::new();
        s.feed("hi");
        assert_eq!(s.flush(), "hi");
        assert!(s.flush().is_empty());
    }

    #[test]
    fn flush_with_trailing_period_no_space() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("Done.").is_empty());
        assert_eq!(s.flush(), "Done.");
    }

    // --- edge cases --------------------------------------------------------

    #[test]
    fn empty_delta_is_noop() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("").is_empty());
        assert!(s.flush().is_empty());
    }

    #[test]
    fn silent_sentinel_never_emits() {
        let mut s = SentenceStreamer::new();
        assert!(s.feed("<silent>").is_empty());
        assert_eq!(s.flush(), "<silent>");
    }

    #[test]
    fn consecutive_punctuation_collapses_into_one_boundary() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("Wait?! "), vec!["Wait?!"]);
    }

    #[test]
    fn newline_acts_as_terminator_whitespace() {
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("Done.\nNext"), vec!["Done."]);
    }

    // --- Python `isspace()` parity: U+001C–U+001F -------------------------
    //
    // CPython `str.isspace()` classifies the four ASCII information separators
    // FS/GS/RS/US (U+001C–U+001F) as whitespace, but Rust `char::is_whitespace`
    // (Unicode White_Space) does not. Spec 09 invariant 67 pins the boundary
    // contract to `isspace()`. These mirror the Python reference exactly:
    //   feed("Done.\x1cNext")           -> ["Done."]
    //   feed("First.\x1c\x1dSecond")    -> ["First."]   (separator run consumed)

    #[test]
    fn info_separator_acts_as_terminator_whitespace() {
        // U+001C (FILE SEPARATOR) follows the terminator: Python emits here.
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("Done.\u{1c}Next"), vec!["Done."]);
        // The separator is consumed like any trailing whitespace.
        assert_eq!(s.flush(), "Next");
    }

    #[test]
    fn all_four_info_separators_are_boundaries() {
        for sep in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            let mut s = SentenceStreamer::new();
            let input = format!("Hello world.{sep}Next");
            assert_eq!(s.feed(&input), vec!["Hello world."]);
            assert_eq!(s.flush(), "Next");
        }
    }

    #[test]
    fn info_separator_run_is_fully_consumed() {
        // A run of separators (and mixed with normal spaces) is all eaten so
        // the next sentence starts clean — matching the isspace() consume loop.
        let mut s = SentenceStreamer::new();
        assert_eq!(s.feed("First.\u{1c}\u{1d}Second"), vec!["First."]);
        assert_eq!(s.flush(), "Second");

        let mut s2 = SentenceStreamer::new();
        assert_eq!(s2.feed("First.\u{1f} \u{1c}Second"), vec!["First."]);
        assert_eq!(s2.flush(), "Second");
    }
}
