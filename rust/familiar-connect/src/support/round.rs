//! Shared half-to-even rounding matching Python `round()` (DESIGN §4.3).
//!
//! Python's built-in `round()` uses banker's rounding (round-half-to-even).
//! Rust's [`f64::round`] rounds half away from zero, so the two disagree only at
//! an exact `.5`. Everywhere the port needs parity with Python `round()` — the
//! `TierBudget::apply_curve` scaling (05/02), the voice-budget gap-ms recorder
//! (01), fact-importance paths — call [`half_even`] for bit-identical results.
//!
//! Distinct from `@span` timing, which is **truncation toward zero**
//! (`(elapsed_s * 1000.0) as i64`), not rounding — keep the two semantics apart.

/// Round to the nearest integer, ties to even (Python `round()` semantics).
#[must_use]
pub fn half_even(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    // `half_even` returns exact integer-valued f64s, so exact `==` is the right
    // assertion here; and the truncation-contrast test deliberately casts f64 to
    // i64 to demonstrate the distinct semantics.
    #![allow(clippy::float_cmp, clippy::cast_possible_truncation)]

    use super::half_even;

    #[test]
    fn half_even_rounds_ties_to_even() {
        // Exact halves resolve to the nearest *even* integer.
        assert_eq!(half_even(0.5), 0.0);
        assert_eq!(half_even(1.5), 2.0);
        assert_eq!(half_even(2.5), 2.0);
        assert_eq!(half_even(3.5), 4.0);
        assert_eq!(half_even(-0.5), 0.0);
        assert_eq!(half_even(-1.5), -2.0);
        assert_eq!(half_even(-2.5), -2.0);
    }

    #[test]
    fn half_even_matches_ordinary_rounding_away_from_ties() {
        assert_eq!(half_even(1.4), 1.0);
        assert_eq!(half_even(1.6), 2.0);
        assert_eq!(half_even(2.499), 2.0);
        assert_eq!(half_even(-1.6), -2.0);
    }

    #[test]
    fn half_even_differs_from_f64_round_at_exact_halves() {
        // This is the whole reason the helper exists: `f64::round` rounds half
        // away from zero (0.5 -> 1, 2.5 -> 3), `half_even` ties to even.
        assert_ne!(half_even(0.5), 0.5_f64.round());
        assert_ne!(half_even(2.5), 2.5_f64.round());
        assert_eq!(0.5_f64.round(), 1.0);
        assert_eq!(2.5_f64.round(), 3.0);
    }

    #[test]
    fn half_even_is_distinct_from_truncation() {
        // `@span` ms truncates toward zero via `as i64`; rounding does not.
        // Pin the contrast so the two never get conflated.
        assert_eq!(half_even(1.9), 2.0);
        assert_eq!(1.9_f64 as i64, 1); // truncation toward zero
        assert_eq!(half_even(-1.9), -2.0);
        assert_eq!(-1.9_f64 as i64, -1); // truncation toward zero
    }
}
