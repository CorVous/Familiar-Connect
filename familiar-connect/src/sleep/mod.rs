//! Nightly maintenance passes: consolidation + opinion formation + registry
//! (subsystem 04; Python `sleep/`).
//!
//! Two loosely-related halves that share the "background LLM behind a seam"
//! shape, both governed by **the model proposes, code decides**: every LLM
//! proposal is validated against safety rails *in code* after the reply, so no
//! prompt override can weaken one.
//!
//! * [`consolidation`] — the consolidation pass proposes fact retire/rewrite over
//!   the whole live fact base + the day's turn window, validates every proposal
//!   ([`consolidation::validate`]), and produces a dry-run-safe
//!   [`consolidation::ConsolidationPlan`]. [`apply`] is the only mutator.
//! * [`opinion_formation`] — the two-stage opinion pass (per-day stance-moments →
//!   one synthesis) mints grounded `ego:` facts.
//! * [`maintenance`] — the explicit pass registry (DESIGN D14) sequencing the two
//!   passes and threading consolidation's retirements into the opinion pass's
//!   known-bits deny-list.

pub mod apply;
pub mod consolidation;
pub mod maintenance;
pub mod opinion_formation;

use std::collections::HashSet;

use serde_json::Value;

use crate::history::store::FactSubject;

/// Faults raised by the sleep subsystem (DESIGN §4.1). Reserved for genuine
/// programming/config errors; bad LLM output degrades to an empty plan, never an
/// `Err`.
#[derive(Debug, thiserror::Error)]
pub enum SleepError {
    /// [`maintenance::create_passes`] was handed a name not in the registry. The
    /// message names the bad name (`{name!r}`) and the sorted valid list — the
    /// Python `ValueError("unknown maintenance pass …")` contract.
    #[error("unknown maintenance pass '{name}'; valid: {valid}")]
    UnknownPass {
        /// The unrecognised pass name.
        name: String,
        /// Sorted, comma-joined registered names (or `(none)`).
        valid: String,
    },
    /// [`opinion_formation::bucket_by_day`] was handed a tz name `chrono-tz`
    /// cannot resolve. Python's `ZoneInfo(tz_name)` raises here too; the engine
    /// guard (subsystem 11) catches it — the port keeps that behavior rather than
    /// silently defaulting.
    #[error("unknown time zone: {0}")]
    InvalidTimezone(String),
    /// [`opinion_formation::validate_opinions`] accepted an opinion whose
    /// grounding ids are all absent from the window's turn→day map, so the
    /// earliest-grounding-day `min` has no argument. Python's
    /// `min(turn_day[i] for i in ids if i in turn_day)`
    /// (`opinion_formation.py:410`) raises `ValueError('min() arg is an empty
    /// sequence')` here; the port keeps that raise (the engine guard degrades
    /// wake prose to seed-only, dry-run included) rather than emitting an opinion
    /// with an empty `valid_from` date that only fails later inside
    /// `apply_opinions`. Unreachable via [`opinion_formation::plan_opinions`],
    /// where `grounding_union ⊆ turn_day` by construction.
    #[error("opinion grounding ids are all absent from the window's day map")]
    EmptyOpinionGroundingDays,
}

// ---------------------------------------------------------------------------
// Shared helpers ported from `history.store` (private there — reimplemented
// locally, byte-identical; see shared_file_requests to export them instead).
// ---------------------------------------------------------------------------

/// Deterministic normalization for near-duplicate detection: lowercase →
/// whitespace-collapse → strip ALL `'`/`"` → trim the surrounding character set
/// `.,!?;:()[]{} \t\n`. Internal non-quote punctuation is kept. Mirrors Python
/// `history.store._normalize_fact_text` (the noop rail and opinion dedup depend
/// on it).
pub(crate) fn normalize_fact_text(text: &str) -> String {
    let collapsed = text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let dequoted: String = collapsed
        .chars()
        .filter(|&c| c != '\'' && c != '"')
        .collect();
    dequoted
        .trim_matches(|c: char| ".,!?;:()[]{} \t\n".contains(c))
        .to_owned()
}

/// The set of canonical subject keys of `subjects`. Mirrors Python
/// `history.store._subject_key_set`.
pub(crate) fn subject_key_set(subjects: &[FactSubject]) -> HashSet<String> {
    subjects.iter().map(|s| s.canonical_key.clone()).collect()
}

// ---------------------------------------------------------------------------
// Python `str(x)` / repr parity helpers for JSON-decoded LLM payload fields.
// ---------------------------------------------------------------------------

/// Mirror Python `str(value)` for a JSON-decoded value: a string passes through;
/// `None`/`True`/`False` follow Python's `str`; numbers render their decimal
/// form; arrays/objects render as Python container repr via [`py_repr`]
/// (`['a', 'b']`, `{'k': 1}`), matching `str(list)`/`str(dict)`. These are
/// absurd inputs no prompt produces in the `text`/`reason`/`new_text` fields
/// this backs; the one residual vs Python is object key order, which follows
/// serde_json's sorted-key iteration rather than Python's insertion order
/// (serde_json is built without `preserve_order`).
pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => n.to_string(),
        Value::Array(_) | Value::Object(_) => py_repr(value),
    }
}

/// Mirror Python `repr(value)` for a JSON-decoded value — the element rendering
/// `str(list)`/`str(dict)` apply to their contents. Differs from [`py_str`] only
/// for strings, which are quoted-and-escaped Python-repr style; scalars match
/// [`py_str`] and containers recurse. Object keys iterate serde_json's sorted
/// order (no `preserve_order`), the one residual vs Python's insertion order.
fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => py_repr_str(s),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python `repr(str)`: wrap in `'…'`, switching to `"…"` when the string holds a
/// `'` but no `"` (Python's quote-preference rule); escape `\`, the active quote,
/// and `\n`/`\r`/`\t`; render other C0 controls and DEL as `\xNN`. Printable
/// non-ASCII passes through, as in CPython 3 (exotic non-printable code points
/// above `\x7f` are not category-escaped — an absurd-input residual).
fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\0'..='\u{1f}' | '\u{7f}' => {
                let code = c as u32;
                out.push('\\');
                out.push('x');
                out.push(char::from_digit(code >> 4, 16).unwrap_or('0'));
                out.push(char::from_digit(code & 0xf, 16).unwrap_or('0'));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Mirror Python `str(payload.get(key, ""))`: absent key → `""`; otherwise
/// [`py_str`] of the value.
pub(crate) fn py_str_field(payload: &Value, key: &str) -> String {
    payload.get(key).map_or_else(String::new, py_str)
}

/// A `usize` length as `i64`, saturating at `i64::MAX`. The lists this converts —
/// fact-id sets and accepted-opinion counts — are tiny, so saturation never
/// occurs; the conversion avoids the lossy `as` cast in the cap-budget math.
pub(crate) fn len_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Python tuple repr for an int tuple: `()`, `(1,)`, `(1, 2)`.
pub(crate) fn py_tuple_repr(ids: &[i64]) -> String {
    match ids {
        [] => "()".to_owned(),
        [x] => format!("({x},)"),
        _ => format!(
            "({})",
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python list repr for a list of strings: `['a', 'b']`.
pub(crate) fn py_str_list_repr(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Python list repr for a list of ints: `[1, 2]`.
pub(crate) fn py_int_list_repr(ids: &[i64]) -> String {
    format!(
        "[{}]",
        ids.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::{py_repr, py_str};
    use serde_json::json;

    #[test]
    fn py_str_scalars_follow_python() {
        assert_eq!(py_str(&json!("hi")), "hi");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(false)), "False");
        assert_eq!(py_str(&json!(3)), "3");
    }

    #[test]
    fn py_str_list_uses_python_container_repr() {
        // Python: str(["a", "b"]) == "['a', 'b']"
        assert_eq!(py_str(&json!(["a", "b"])), "['a', 'b']");
        // Python: str([1, True, None, "x"]) == "[1, True, None, 'x']"
        assert_eq!(py_str(&json!([1, true, null, "x"])), "[1, True, None, 'x']");
        // nesting: str([["a"], []]) == "[['a'], []]"
        assert_eq!(py_str(&json!([["a"], []])), "[['a'], []]");
    }

    #[test]
    fn py_str_object_uses_python_container_repr() {
        // single key matches Python exactly: str({"k": 1}) == "{'k': 1}"
        assert_eq!(py_str(&json!({"k": 1})), "{'k': 1}");
        assert_eq!(py_str(&json!({"a": "b"})), "{'a': 'b'}");
    }

    #[test]
    fn py_repr_string_quoting_follows_cpython() {
        // has ' and no " → double quotes, no escaping the '
        assert_eq!(py_repr(&json!("it's")), "\"it's\"");
        // has " and no ' → single quotes, no escaping the "
        assert_eq!(py_repr(&json!("say \"hi\"")), "'say \"hi\"'");
        // has both → single quotes, escape the '
        assert_eq!(py_repr(&json!("a'b\"c")), "'a\\'b\"c'");
        // control chars → named escapes
        assert_eq!(py_repr(&json!("a\nb\t")), "'a\\nb\\t'");
        // backslash doubles; C0 control → \xNN
        assert_eq!(py_repr(&json!("x\\\u{0}")), "'x\\\\\\x00'");
    }
}
