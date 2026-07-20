//! Silent-sentinel detection for LLM reply streams (subsystem 06; Python
//! `silence.py`).
//!
//! The system prompt tells the model to emit `<silent>` as its entire reply
//! when staying silent. [`SilentDetector`] watches streamed deltas and decides
//! as soon as possible, so a responder aborts before paying downstream cost
//! (Discord post, TTS synthesis).
//!
//! Prefix-only: a stray `<silent>` mid-reply is content, not a gate.
//!
//! [`StreamGate`] widens that logic for the voice path: as well as `<silent>`
//! it recognises a tool-call block the model occasionally leaks as plain text
//! (`<invoke …`, `silent(…)`, `read_channel(…)`, `<tool_call …`) so the leak is
//! caught *mid-stream* — before it reaches TTS (issue #109). The confirmed-leak
//! classification ([`classify_leading_leak`]) is shared with the return-time
//! strip guard in [`crate::tools::agentic`], the single source of truth.

use std::sync::LazyLock;

use regex::Regex;

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

// ---------------------------------------------------------------------------
// Leaked-tool-call detection (shared with the return-time strip guard in
// `tools::agentic`; behaviour-pinned, ported verbatim)
// ---------------------------------------------------------------------------

static LEADING_INVOKE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*<(?:\w+:)?invoke\b").expect("valid regex"));
static INVOKE_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<(?:\w+:)?invoke\b[^>]*\bname="([^"]+)""#).expect("valid regex")
});
static PYTHON_SILENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*silent\s*\(").expect("valid regex"));
static PYTHON_TOOL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*(read_channel|shift_focus)\s*\(").expect("valid regex"));
static TOOL_CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*<(?:\w+:)?tool_call\b").expect("valid regex"));

/// A tool-call invocation the model leaked into plain assistant text,
/// classified by shape. Single source of truth for the return-time strip guard
/// ([`crate::tools::agentic`]) and the streaming [`StreamGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeadingLeak {
    /// A leaked `<invoke>…</invoke>` block (optionally namespaced). `silent` is
    /// true when it names the `silent` tool.
    Invoke { silent: bool },
    /// A leaked Python-style `silent(…)` call — honoured as silence.
    PythonSilent,
    /// A leaked Python-style `read_channel(…)` / `shift_focus(…)` call.
    PythonTool,
}

/// Classify a leaked tool call that *leads* `content`; `None` when `content`
/// does not begin with a recognised leak token. A stray mention mid-prose is
/// content, so only a leading match counts.
pub(crate) fn classify_leading_leak(content: &str) -> Option<LeadingLeak> {
    if LEADING_INVOKE_RE.is_match(content) {
        let silent = INVOKE_NAME_RE
            .captures_iter(content)
            .any(|c| &c[1] == "silent");
        Some(LeadingLeak::Invoke { silent })
    } else if PYTHON_SILENT_RE.is_match(content) {
        Some(LeadingLeak::PythonSilent)
    } else if PYTHON_TOOL_RE.is_match(content) {
        Some(LeadingLeak::PythonTool)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Streaming leak + silent gate (voice path)
// ---------------------------------------------------------------------------

/// The latched decision of a [`StreamGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDecision {
    /// Not enough leading text yet to decide.
    Pending,
    /// Genuine prose — open the speak path.
    Speak,
    /// The reply led with `<silent>` or a leaked `silent` call — stay silent.
    Silent,
    /// The reply led with a leaked non-silent tool call — suppress to empty.
    Suppress,
}

/// Streaming gate for the voice reply path.
///
/// [`SilentDetector`] recognises only `<silent>`, latching *speak* the moment a
/// leading `<invoke …` diverges at the second char — so a leaked tool-call block
/// reaches TTS before the return-time guard runs (issue #109). `StreamGate`
/// stays *pending* while the leading text could still complete a leak token
/// (handling a token split across delta boundaries), latches *silent* /
/// *suppress* on a confirmed leak, and only latches *speak* once every leak
/// token is ruled out. `<silent>` semantics are preserved exactly (spec S1-S4).
#[derive(Debug, Default)]
pub struct StreamGate {
    buf: String,
    decided: Option<StreamDecision>,
}

impl StreamGate {
    /// A fresh, undecided gate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The latched decision, or `None` while pending.
    #[must_use]
    pub const fn decided(&self) -> Option<StreamDecision> {
        self.decided
    }

    const fn latch(&mut self, decision: StreamDecision) -> StreamDecision {
        self.decided = Some(decision);
        decision
    }

    /// Feed one streamed delta; returns the (possibly newly latched) decision.
    /// The decision **latches** — later calls return it and ignore their
    /// argument.
    pub fn feed(&mut self, delta: &str) -> StreamDecision {
        if let Some(decided) = self.decided {
            return decided;
        }
        self.buf.push_str(delta);
        let stripped = self.buf.trim_start();
        if stripped.is_empty() {
            return StreamDecision::Pending;
        }
        // `<silent>` sentinel (spec S1-S4), including the split-delta case.
        if stripped.starts_with(SILENT_TOKEN) {
            return self.latch(StreamDecision::Silent);
        }
        // Confirmed leaked tool call (shared classifier). A leaked `<invoke …`
        // latches before its `name="…"` attribute has fully streamed; for the
        // voice path suppress and silent are the same outcome (no audio, no
        // persisted turn), so an as-yet-unnamed invoke latches suppress.
        if let Some(leak) = classify_leading_leak(stripped) {
            let decision = match leak {
                LeadingLeak::Invoke { silent: true } | LeadingLeak::PythonSilent => {
                    StreamDecision::Silent
                }
                LeadingLeak::Invoke { silent: false } | LeadingLeak::PythonTool => {
                    StreamDecision::Suppress
                }
            };
            return self.latch(decision);
        }
        // Confirmed leaked `<tool_call …>` (streaming guard only; the
        // return-time strip stays byte-identical and does not touch this form).
        if TOOL_CALL_RE.is_match(stripped) {
            return self.latch(StreamDecision::Suppress);
        }
        // Undecided: stay pending while the prefix could still complete a leak
        // token, otherwise it is genuine prose.
        if leak_prefix_possible(stripped) {
            StreamDecision::Pending
        } else {
            self.latch(StreamDecision::Speak)
        }
    }
}

/// Could `stripped` (left-trimmed, non-empty) still grow into a confirmed leak
/// token? Conservative by design: it must never rule out a genuine leak prefix
/// (that would let a leak reach TTS); ruling out prose a few chars late only
/// costs a little time-to-first-audio.
fn leak_prefix_possible(stripped: &str) -> bool {
    stripped
        .strip_prefix('<')
        .map_or_else(|| call_prefix_possible(stripped), xml_prefix_possible)
}

/// XML-family prefixes: `<silent>`, `<[ns:]invoke`, `<[ns:]tool_call`.
fn xml_prefix_possible(after: &str) -> bool {
    prefix_compatible(after, "silent>")
        || local_name_possible(after, "invoke")
        || local_name_possible(after, "tool_call")
}

/// Whether `after` (the text after a leading `<`) could still complete
/// `<[ns:]local…` for the given local name.
fn local_name_possible(after: &str, local: &str) -> bool {
    if prefix_compatible(after, local) {
        return true;
    }
    if let Some((ns, rest)) = after.split_once(':') {
        return is_xml_name(ns) && prefix_compatible(rest, local);
    }
    // Namespace still being typed (no colon yet): any XML-name-so-far could be
    // a namespace that will be followed by `:local`.
    is_xml_name(after)
}

/// Python-call prefixes: `silent(`, `read_channel(`, `shift_focus(`.
fn call_prefix_possible(stripped: &str) -> bool {
    let lc = stripped.to_ascii_lowercase();
    ["silent", "read_channel", "shift_focus"]
        .iter()
        .any(|name| call_name_possible(&lc, name))
}

/// Whether the lowercased `lc` could still complete `name` (optionally followed
/// by whitespace before the `(`).
fn call_name_possible(lc: &str, name: &str) -> bool {
    if name.starts_with(lc) {
        return true;
    }
    lc.strip_prefix(name)
        .is_some_and(|rest| rest.chars().all(char::is_whitespace))
}

/// `a` is a prefix of `b`, or `b` a prefix of `a` — either can still grow into
/// the other.
fn prefix_compatible(a: &str, b: &str) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// A non-empty XML-name-so-far: starts with a letter/underscore, then
/// letters/digits/`_`/`-`. Bounds the namespace hypothesis so digit- or
/// symbol-leading content (`<3`, `<smiles>`) is ruled out promptly.
fn is_xml_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::{
        LeadingLeak, SILENT_TOKEN, SilentDetector, StreamDecision, StreamGate,
        classify_leading_leak,
    };

    /// Feed a delta sequence through a fresh gate and return the final decision.
    fn gate_run(deltas: &[&str]) -> StreamDecision {
        let mut g = StreamGate::new();
        let mut last = StreamDecision::Pending;
        for d in deltas {
            last = g.feed(d);
        }
        last
    }

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

    // -- classify_leading_leak (shared with the return-time strip guard) -----

    #[test]
    fn classify_invoke_silent_vs_nonsilent() {
        assert_eq!(
            classify_leading_leak("<invoke name=\"silent\">x</invoke>"),
            Some(LeadingLeak::Invoke { silent: true })
        );
        assert_eq!(
            classify_leading_leak("<invoke name=\"read_channel\">x</invoke>"),
            Some(LeadingLeak::Invoke { silent: false })
        );
        assert_eq!(
            classify_leading_leak("<ns:invoke name=\"silent\">x</ns:invoke>"),
            Some(LeadingLeak::Invoke { silent: true })
        );
    }

    #[test]
    fn classify_python_and_none() {
        assert_eq!(
            classify_leading_leak("Silent(reasoning=\"x\")"),
            Some(LeadingLeak::PythonSilent)
        );
        assert_eq!(
            classify_leading_leak("read_channel(limit=10)"),
            Some(LeadingLeak::PythonTool)
        );
        assert_eq!(classify_leading_leak("Let me invoke my wit."), None);
    }

    // -- StreamGate ----------------------------------------------------------

    #[test]
    fn gate_silent_sentinel() {
        assert_eq!(gate_run(&["<silent>"]), StreamDecision::Silent);
    }

    #[test]
    fn gate_silent_split_across_deltas() {
        let mut g = StreamGate::new();
        assert_eq!(g.feed("<sil"), StreamDecision::Pending);
        assert_eq!(g.feed("ent>"), StreamDecision::Silent);
    }

    #[test]
    fn gate_leaked_invoke_read_channel_split_across_deltas_suppresses() {
        // The exact issue #109 scenario: a leaked `<invoke …` streamed as
        // CONTENT, split across delta boundaries, must never open the gate. It
        // stays pending until the opening tag completes, then latches suppress.
        let mut g = StreamGate::new();
        assert_eq!(g.feed("<in"), StreamDecision::Pending);
        assert_eq!(
            g.feed("voke name=\"read_channel\">"),
            StreamDecision::Suppress
        );
        // Latches: a later delta cannot re-open it.
        assert_eq!(
            g.feed("<param>10</param></invoke>"),
            StreamDecision::Suppress
        );
    }

    #[test]
    fn gate_leaked_namespaced_invoke_suppresses() {
        assert_eq!(
            gate_run(&["<invoke name=\"read_channel\">"]),
            StreamDecision::Suppress
        );
    }

    #[test]
    fn gate_leaked_python_silent_is_silent() {
        assert_eq!(
            gate_run(&["silent(", "reasoning=x)"]),
            StreamDecision::Silent
        );
    }

    #[test]
    fn gate_leaked_python_tool_suppresses() {
        assert_eq!(
            gate_run(&["read_channel(limit=10)"]),
            StreamDecision::Suppress
        );
    }

    #[test]
    fn gate_leaked_tool_call_tag_suppresses() {
        assert_eq!(
            gate_run(&["<tool_call>", "{...}"]),
            StreamDecision::Suppress
        );
    }

    #[test]
    fn gate_normal_prose_speaks() {
        assert_eq!(gate_run(&["Hello there!"]), StreamDecision::Speak);
        assert_eq!(gate_run(&["Sure, one moment."]), StreamDecision::Speak);
    }

    #[test]
    fn gate_mid_prose_invoke_untouched_speaks() {
        // `<invoke>` only gates when it *leads*; a mention mid-reply is content.
        assert_eq!(
            gate_run(&["Let me invoke my legendary wit."]),
            StreamDecision::Speak
        );
    }

    #[test]
    fn gate_false_positive_guards_speak() {
        // Legit `<`-leading content must not be mistaken for a leak.
        assert_eq!(gate_run(&["<3 you all"]), StreamDecision::Speak);
        assert_eq!(gate_run(&["<smiles>"]), StreamDecision::Speak);
        assert_eq!(gate_run(&["*whispers* hello"]), StreamDecision::Speak);
        assert_eq!(gate_run(&["<html> tag talk"]), StreamDecision::Speak);
    }

    #[test]
    fn gate_word_prefixes_resolve_to_speak() {
        // `silence`/`read the …` share a prefix with a call token but diverge.
        assert_eq!(gate_run(&["silence is golden"]), StreamDecision::Speak);
        assert_eq!(gate_run(&["read the docs, please"]), StreamDecision::Speak);
    }

    #[test]
    fn gate_char_by_char_leaked_invoke_never_speaks() {
        // Streamed one char at a time, the gate must never pass through Speak.
        let mut g = StreamGate::new();
        let mut decided = StreamDecision::Pending;
        for c in "<invoke name=\"read_channel\">".chars() {
            let d = g.feed(&c.to_string());
            assert_ne!(d, StreamDecision::Speak, "opened on prefix");
            decided = d;
        }
        assert_eq!(decided, StreamDecision::Suppress);
    }

    #[test]
    fn gate_leading_whitespace_then_silent() {
        let mut g = StreamGate::new();
        assert_eq!(g.feed("   "), StreamDecision::Pending);
        assert_eq!(g.feed("<silent>"), StreamDecision::Silent);
    }
}
