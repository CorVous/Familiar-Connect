//! Tolerant JSON coercion for possibly-fenced LLM replies (subsystem 02).
//!
//! Chat-tuned models wrap JSON in ```` ``` ```` / ```` ```json ```` fences and
//! pad it with prose. This is the PARSE half of structured output (the REQUEST
//! half lives in `structured_request`). Every function here **degrades on bad
//! input and never panics** — a model that fumbles its JSON must not crash the
//! worker reading it.
//!
//! The blob extraction is deliberately a **greedy DOTALL span** (`\{.*\}` /
//! `\[.*\]`), not a balanced-bracket matcher, so a reply containing both shapes
//! resolves the same way at every call site. Do not "fix" it into a proper
//! bracket matcher.
//!
//! ## Tolerant-parse boundary (declared; unobservable in practice)
//!
//! Two number shapes a more lenient JSON reader might accept are
//! **fundamentally unrepresentable in the Rust value types**, so this module
//! takes the closest safe option (degrade, never panic) and records the
//! boundary:
//!
//! 1. **Non-finite literals.** The non-standard tokens `NaN` / `Infinity` /
//!    `-Infinity` and an overflowing exponent like `1e999` (→ `inf`) are
//!    rejected by `serde_json` ("expected value" / "number out of range"), so
//!    [`coerce_json`] degrades to `JsonResult { value: None, parsed_ok: false }`.
//!    `serde_json::Value::Number` cannot hold a non-finite float
//!    (`Number::from_f64` returns `None` for `NaN`/`inf`), so accepting them
//!    would trade a clean failure for a wrong value. The sole downstream effect
//!    is one extra corrective re-ask in `structured_request`; LLM replies do
//!    not emit bare `NaN`/`Infinity`.
//! 2. **Integers wider than `i64`/`u64`.** Without the (crate-wide,
//!    deliberately unset) `arbitrary_precision` feature, `serde_json` coerces
//!    such a token to a lossy `f64` (`Number(1.2345678901234568e29)`).
//!    [`coerce_json`] still reports `parsed_ok: true`, but the *value* is a
//!    float, so a downstream [`coerce_positive_int_list`] drops it (`as_i64()`
//!    is `None`). `Vec<i64>` could not represent the value regardless, and
//!    fact/turn ids are small; unobservable.
//!
//! Both cases are pinned by regression tests below so the degrade-never-panic
//! contract holds across this boundary.

use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

// Greedy + DOTALL: span from the first `{`/`[` to the LAST matching bracket, so
// a multi-line object survives.
static JSON_OBJECT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\{.*\}").expect("valid object regex"));
static JSON_ARRAY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\[.*\]").expect("valid array regex"));
static FENCE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)```(?:json)?").expect("valid fence regex"));
// A signed decimal integer and nothing else (full match). ASCII-only `\d`:
// a more lenient reader would accept Unicode digits, but
// `str::parse::<i64>` will not — an untested, unobservable-in-practice
// deviation.
static INT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^-?[0-9]+$").expect("valid int regex"));

/// Which JSON shape [`coerce_json`] should extract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// First `{...}` blob only.
    Object,
    /// First `[...]` blob only.
    Array,
    /// Whichever of object/array starts earlier (tie → object).
    Any,
}

/// Outcome of a tolerant parse.
///
/// `parsed_ok` is the success signal; on failure `value` is `None`. The boolean
/// keeps "model fumbled the JSON" distinguishable from "model returned an empty
/// object/array" (`Some(Value::Object(empty))` != `None`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JsonResult {
    /// The parsed value on success, else `None`.
    pub value: Option<Value>,
    /// Whether the parse succeeded.
    pub parsed_ok: bool,
}

/// Parse a possibly-fenced LLM reply into JSON of the requested shape.
///
/// Never panics. Any failure — empty reply, no JSON of the wanted shape,
/// malformed JSON — degrades to `JsonResult { value: None, parsed_ok: false }`.
#[must_use]
pub fn coerce_json(reply: &str, expect: Expect) -> JsonResult {
    if reply.trim().is_empty() {
        return JsonResult::default();
    }
    let stripped = FENCE_RE.replace_all(reply, "");
    let cleaned = stripped.trim();
    let blob = shaped_json_blob(cleaned, expect);
    // `serde_json::from_str` is strict about numbers
    // here: bare `NaN`/`Infinity`/`-Infinity` and overflow-exponent numbers
    // fumble instead of yielding a non-finite float, and a >i64 integer becomes
    // a lossy f64. See the module-level "Tolerant-parse boundary"
    // note — both are unrepresentable in the Rust value types and unobservable
    // in real LLM replies; the degrade-never-panic contract still holds.
    serde_json::from_str::<Value>(blob).map_or_else(
        |_| JsonResult::default(),
        |value| JsonResult {
            value: Some(value),
            parsed_ok: true,
        },
    )
}

/// Return the first shape-matching blob, else the whole cleaned text.
///
/// A reply that is bare JSON (no surrounding prose) has no shape match and is
/// handed through verbatim, letting the parser accept scalars and pre-trimmed
/// payloads exactly as the per-site code does.
fn shaped_json_blob(cleaned: &str, expect: Expect) -> &str {
    let obj = if matches!(expect, Expect::Object | Expect::Any) {
        JSON_OBJECT_RE.find(cleaned)
    } else {
        None
    };
    let arr = if matches!(expect, Expect::Array | Expect::Any) {
        JSON_ARRAY_RE.find(cleaned)
    } else {
        None
    };
    match (obj, arr) {
        (Some(o), Some(a)) => {
            if o.start() <= a.start() {
                o.as_str()
            } else {
                a.as_str()
            }
        }
        (Some(m), None) | (None, Some(m)) => m.as_str(),
        (None, None) => cleaned,
    }
}

/// Distinct positive ints from a JSON value, order-preserving.
///
/// Bools are rejected outright (JSON `true` must not slip through as `1`);
/// integer numbers and integer-valued numeric strings are accepted; floats,
/// non-positive values, and duplicates are dropped. Anything not an array
/// degrades to `[]`.
#[must_use]
pub fn coerce_positive_int_list(raw: &Value) -> Vec<i64> {
    let Some(arr) = raw.as_array() else {
        return Vec::new();
    };
    let mut out: Vec<i64> = Vec::new();
    for item in arr {
        let val: i64 = match item {
            // `as_i64` is `None` for floats and out-of-range values, so JSON
            // like `1.5` / `2.0` is dropped (not an integer).
            Value::Number(n) => match n.as_i64() {
                Some(v) => v,
                None => continue,
            },
            Value::String(s) => {
                let trimmed = s.trim();
                if INT_RE.is_match(trimmed) {
                    match trimmed.parse::<i64>() {
                        Ok(v) => v,
                        Err(_) => continue,
                    }
                } else {
                    continue;
                }
            }
            // Everything else — including `Value::Bool` (JSON true/false is a
            // distinct variant, so the `true == 1` hazard cannot occur) —
            // is dropped.
            _ => continue,
        };
        if val > 0 && !out.contains(&val) {
            out.push(val);
        }
    }
    out
}

/// Distinct non-empty strings from a JSON value, order-preserving.
///
/// Blank / whitespace-only strings and non-string items are dropped; duplicates
/// collapse to the first occurrence (the *original*, un-stripped string is
/// kept). Anything not an array degrades to `[]`.
#[must_use]
pub fn coerce_str_list(raw: &Value) -> Vec<String> {
    let Some(arr) = raw.as_array() else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for item in arr {
        if let Value::String(s) = item {
            if !s.trim().is_empty() && !out.iter().any(|existing| existing == s) {
                out.push(s.clone());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Expect, coerce_json, coerce_positive_int_list, coerce_str_list};
    use serde_json::json;

    #[test]
    fn fenced_object_parses_with_ok_true() {
        let payload = json!({"retire": [{"fact_ids": [1]}], "rewrite": []});
        let reply = format!("```json\n{payload}\n```");
        let result = coerce_json(&reply, Expect::Object);
        assert!(result.parsed_ok);
        assert_eq!(result.value, Some(payload));
    }

    #[test]
    fn fenced_array_parses_with_ok_true() {
        let payload = json!([{"text": "a"}, {"text": "b"}]);
        let reply = format!("```json\n{payload}\n```");
        let result = coerce_json(&reply, Expect::Array);
        assert!(result.parsed_ok);
        assert_eq!(result.value, Some(payload));
    }

    #[test]
    fn not_json_degrades_without_raising() {
        let result = coerce_json("not json at all", Expect::Object);
        assert!(!result.parsed_ok);
        assert_eq!(result.value, None);
    }

    #[test]
    fn empty_object_is_parsed_ok_not_fumbled() {
        for expect in [Expect::Object, Expect::Any] {
            let result = coerce_json("{}", expect);
            assert!(result.parsed_ok);
            assert_eq!(result.value, Some(json!({})));
        }
    }

    #[test]
    fn garbage_is_fumbled_under_every_expect() {
        for expect in [Expect::Object, Expect::Array, Expect::Any] {
            let result = coerce_json("not json at all", expect);
            assert!(!result.parsed_ok);
            assert_eq!(result.value, None);
        }
    }

    #[test]
    fn empty_and_whitespace_replies_fumble() {
        assert_eq!(
            coerce_json("", Expect::Object),
            super::JsonResult::default()
        );
        assert_eq!(
            coerce_json("   \n\t ", Expect::Any),
            super::JsonResult::default()
        );
    }

    #[test]
    fn object_expect_skips_leading_array() {
        // A reply with BOTH shapes, array first: an object-expecting site must
        // get the OBJECT, not the array.
        let reply = r#"["b"] and then {"retire":[{"fact_ids":[1]}],"rewrite":[]}"#;
        let result = coerce_json(reply, Expect::Object);
        assert!(result.parsed_ok);
        assert_eq!(
            result.value,
            Some(json!({"retire": [{"fact_ids": [1]}], "rewrite": []}))
        );
    }

    #[test]
    fn array_expect_skips_leading_object() {
        // Both shapes, object first: an array-expecting site must get the ARRAY.
        let reply = r#"{"a":1} then [{"text":"f","source_turn_ids":[1]}]"#;
        let result = coerce_json(reply, Expect::Array);
        assert!(result.parsed_ok);
        assert_eq!(
            result.value,
            Some(json!([{"text": "f", "source_turn_ids": [1]}]))
        );
    }

    #[test]
    fn any_expect_keeps_first_blob_behavior() {
        // array starts before object here, so the array wins.
        let reply = r#"["b"] then {"k":1}"#;
        let result = coerce_json(reply, Expect::Any);
        assert!(result.parsed_ok);
        assert_eq!(result.value, Some(json!(["b"])));
    }

    #[test]
    fn positive_int_list_rejects_bools_dupes_and_non_positive_preserving_order() {
        // JSON `true` must NOT slip through as the int 1.
        assert_eq!(
            coerce_positive_int_list(&json!([3, 3, true, -1, 0, 7])),
            vec![3, 7]
        );
    }

    #[test]
    fn positive_int_list_malformed_int_strings_drop_without_raising() {
        assert_eq!(
            coerce_positive_int_list(&json!(["--5", "3-", "x", "1.5", 0, -2, 3, 3, 7])),
            vec![3, 7]
        );
    }

    #[test]
    fn positive_int_list_accepts_numeric_strings_and_drops_floats() {
        assert_eq!(
            coerce_positive_int_list(&json!(["5", " 8 ", 2, 2.0, 1.5])),
            vec![5, 8, 2]
        );
    }

    #[test]
    fn positive_int_list_non_array_degrades_to_empty() {
        assert_eq!(
            coerce_positive_int_list(&json!({"a": 1})),
            Vec::<i64>::new()
        );
        assert_eq!(coerce_positive_int_list(&json!("nope")), Vec::<i64>::new());
    }

    #[test]
    fn str_list_keeps_distinct_non_empty_strings_in_order() {
        assert_eq!(
            coerce_str_list(&json!(["a", "a", "", " ", "b"])),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn str_list_non_array_degrades_to_empty() {
        assert_eq!(coerce_str_list(&json!(42)), Vec::<String>::new());
    }

    // --- Tolerant-parse boundary (see module docs) -----------------------
    // These pin the DELIBERATE, declared boundary:
    // non-finite literals and >i64 integers are unrepresentable in the Rust
    // value types, so this module keeps the degrade-never-panic contract.
    // Documented as untested/unobservable in
    // real LLM replies; pinned here so the boundary stays intentional.

    #[test]
    fn non_finite_literals_fumble_instead_of_panicking() {
        // A lenient reader would yield nan/inf with parsed_ok=true for each of
        // these; `serde_json` rejects them, so this degrades to a fumble.
        // The contract that MUST hold: never raise/panic.
        for reply in [
            "NaN",
            "Infinity",
            "-Infinity",
            "1e999",
            "-1e999",
            r#"{"score": NaN}"#,
            "[Infinity, 1]",
        ] {
            for expect in [Expect::Object, Expect::Array, Expect::Any] {
                let result = coerce_json(reply, expect);
                assert!(
                    !result.parsed_ok,
                    "declared deviation: {reply:?} should fumble under {expect:?}"
                );
                assert_eq!(result.value, None);
            }
        }
    }

    #[test]
    fn oversized_integer_parses_lossily_and_is_dropped_downstream() {
        // An arbitrary-precision reader would keep
        // `123456789012345678901234567890` exact and `coerce_positive_int_list`
        // would return it. `serde_json` (no
        // `arbitrary_precision`) coerces the token to a lossy f64, so the parse
        // still succeeds but the value is a float — which the int-list coercion
        // then drops (`as_i64()` is None). `Vec<i64>` could not hold the value
        // regardless. Pinned as the declared, unobservable divergence.
        let big = "123456789012345678901234567890";
        let result = coerce_json(&format!("[{big}]"), Expect::Array);
        assert!(result.parsed_ok, "serde accepts the token lossily");
        let value = result.value.expect("array parsed");
        assert!(
            value[0].as_i64().is_none(),
            "the oversized int became a lossy float, not an exact i64"
        );
        assert_eq!(
            coerce_positive_int_list(&value),
            Vec::<i64>::new(),
            "downstream drops the lossy float"
        );
    }
}
