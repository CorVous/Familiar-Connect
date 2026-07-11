//! Shared ISO-8601 UTC helpers (DESIGN §4.2).
//!
//! [`iso_utc`] emits `%Y-%m-%dT%H:%M:%S%.6f+00:00` — **fixed-width 6-digit
//! microseconds** and a literal `+00:00` offset (never `Z`). Python omits the
//! `.ffffff` when microseconds are exactly zero; the port always writes six
//! digits, which is *more* lexicographically stable, not less. Lexicographic ==
//! chronological ordering of these strings is a correctness dependency in five
//! `history` query paths, so every timestamp written to SQLite or JSON goes
//! through [`iso_utc`].
//!
//! [`parse_iso`] is tolerant: it accepts missing microseconds, a `Z` suffix,
//! variable fractional widths, and naive (offset-less) datetimes (assumed UTC).

use chrono::{DateTime, NaiveDateTime, Utc};

/// Emit an ISO-8601 UTC string with fixed 6-digit microseconds and `+00:00`.
#[must_use]
pub fn iso_utc(dt: DateTime<Utc>) -> String {
    dt.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
}

/// Parse an ISO-8601 timestamp tolerantly into a UTC datetime.
///
/// Accepts RFC-3339 forms (`Z`, `+00:00`, any fractional-second width) as well
/// as naive `YYYY-MM-DDTHH:MM:SS[.ffffff]` (and space-separated) datetimes,
/// which are interpreted as UTC. Returns `None` on anything unparseable.
#[must_use]
pub fn parse_iso(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{iso_utc, parse_iso};
    use chrono::{Duration, TimeZone, Utc};

    #[test]
    fn iso_utc_writes_fixed_width_micros_and_plus_zero_offset() {
        let dt = Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
        // Zero microseconds still render as six digits and never as `Z`.
        assert_eq!(iso_utc(dt), "2024-01-02T03:04:05.000000+00:00");

        let with_micros = dt + Duration::microseconds(123_456);
        assert_eq!(iso_utc(with_micros), "2024-01-02T03:04:05.123456+00:00");
    }

    #[test]
    fn iso_utc_round_trips_through_parse_iso() {
        let dt =
            Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap() + Duration::microseconds(654_321);
        let s = iso_utc(dt);
        assert_eq!(parse_iso(&s), Some(dt));

        // And a zero-micros value round-trips exactly too.
        let zero = Utc.with_ymd_and_hms(1999, 12, 31, 23, 59, 59).unwrap();
        assert_eq!(parse_iso(&iso_utc(zero)), Some(zero));
    }

    #[test]
    fn parse_iso_accepts_z_suffix_and_missing_micros() {
        let expected = Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
        assert_eq!(parse_iso("2024-01-02T03:04:05Z"), Some(expected));
        assert_eq!(parse_iso("2024-01-02T03:04:05+00:00"), Some(expected));
        // Naive (no offset) is treated as UTC.
        assert_eq!(parse_iso("2024-01-02T03:04:05"), Some(expected));
    }

    #[test]
    fn parse_iso_accepts_variable_fractional_width() {
        let base = Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
        assert_eq!(
            parse_iso("2024-01-02T03:04:05.5Z"),
            Some(base + Duration::milliseconds(500))
        );
        assert_eq!(
            parse_iso("2024-01-02T03:04:05.123456+00:00"),
            Some(base + Duration::microseconds(123_456)),
        );
    }

    #[test]
    fn parse_iso_returns_none_on_garbage() {
        assert_eq!(parse_iso("not a timestamp"), None);
        assert_eq!(parse_iso(""), None);
    }

    #[test]
    fn iso_utc_strings_sort_lexicographically_as_chronologically() {
        let earlier = iso_utc(Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap());
        let later = iso_utc(Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 6).unwrap());
        assert!(earlier < later);
    }
}
