//! On-disk token-calibration totals (#183).
//!
//! [`super::TokenCalibration`] learns each model's true-vs-estimated token rate
//! from live calls, but in memory it starts blind: the first calls of every
//! process budget against the raw `len/4` heuristic, which real traffic shows
//! under-counting by 20-45% on some tokenizers. Persisting the accumulators
//! makes the *next* boot start calibrated.
//!
//! Same shape and spirit as [`crate::model_diagnostics::cache`]: platform cache
//! directory, JSON, and a missing / truncated / corrupt / stale file all read as
//! `None`. Nothing here is state — worst case the store relearns in one session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::support::time::{iso_utc, parse_iso};

/// Cache file name under the per-user cache directory.
pub const CACHE_FILE_NAME: &str = "token-calibration.json";

/// How long persisted totals stay usable.
pub const CACHE_TTL_DAYS: i64 = 30;

/// One model's persisted accumulators; the ratio is `actual / estimated`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedTotals {
    /// Σ heuristic estimate over persisted samples.
    pub estimated: i64,
    /// Σ provider-reported prompt tokens over the same samples.
    pub actual: i64,
}

/// The serialized cache: a write timestamp plus per-model totals.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedCalibration {
    /// When the totals were last written (`support::time::iso_utc`).
    pub updated_at: String,
    /// Accumulators keyed by model id. `BTreeMap` so the file is deterministic.
    pub models: BTreeMap<String, CachedTotals>,
}

/// Per-user cache path for the calibration totals.
///
/// The platform **cache** directory, beside `openrouter-models.json`: learned
/// rates are regenerable from one session of traffic, so they belong where
/// clearing caches is safe and where they can never be mistaken for state. Same
/// `ProjectDirs` qualifier the familiars root uses, so it honours
/// `XDG_CACHE_HOME`; falls back to a CWD-relative `data/cache` when no home
/// directory resolves.
#[must_use]
pub fn default_cache_path() -> PathBuf {
    ProjectDirs::from("", "", "familiar-connect").map_or_else(
        || Path::new("data").join("cache").join(CACHE_FILE_NAME),
        |dirs| dirs.cache_dir().join(CACHE_FILE_NAME),
    )
}

/// Read the cache, or `None` when it is missing, unreadable, or unparseable.
///
/// Never an error: a corrupt or truncated file is treated exactly as an absent
/// one — the process starts blind, as it always did, and the next write
/// overwrites it. Entries with a non-positive accumulator carry no ratio and are
/// dropped here rather than left for the estimator to divide by.
#[must_use]
pub fn read_cache(path: &Path) -> Option<CachedCalibration> {
    let body = std::fs::read_to_string(path).ok()?;
    let mut cached = serde_json::from_str::<CachedCalibration>(&body).ok()?;
    cached.models.retain(|_, t| t.estimated > 0 && t.actual > 0);
    Some(cached)
}

/// Write `models` to `path`, stamped `now`, creating parent directories.
///
/// Plain truncating write, like the catalog cache: a kill mid-write leaves a
/// truncated file, which [`read_cache`] reports as absent.
///
/// # Errors
/// I/O or serialization failure. Callers log and move on — a cache that cannot
/// be written costs one cold start's accuracy, nothing more.
pub fn write_cache(
    path: &Path,
    models: &BTreeMap<String, CachedTotals>,
    now: DateTime<Utc>,
) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let cached = CachedCalibration {
        updated_at: iso_utc(now),
        models: models.clone(),
    };
    let body = serde_json::to_string(&cached).map_err(std::io::Error::other)?;
    std::fs::write(path, body)
}

/// Whether a cache stamped `updated_at` is older than `ttl_days` at `now`.
///
/// A ratio of totals never decays, so without a ceiling a months-old
/// accumulator would outvote every fresh sample and pin a rate the model's
/// tokenizer may already have moved off. Unparseable and future-dated stamps
/// both count as stale — relearning costs one session, a nonsense timestamp
/// must never pin a cache forever.
#[must_use]
pub fn is_stale(updated_at: &str, now: DateTime<Utc>, ttl_days: i64) -> bool {
    parse_iso(updated_at).is_none_or(|then| {
        let age = now.signed_duration_since(then);
        age < Duration::zero() || age >= Duration::days(ttl_days)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CACHE_FILE_NAME, CACHE_TTL_DAYS, CachedCalibration, CachedTotals, default_cache_path,
        is_stale, read_cache, write_cache,
    };
    use chrono::{Duration, TimeZone as _, Utc};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn now() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 18, 12, 0, 0).unwrap()
    }

    fn totals(estimated: i64, actual: i64) -> CachedTotals {
        CachedTotals { estimated, actual }
    }

    fn two_models() -> BTreeMap<String, CachedTotals> {
        BTreeMap::from([
            ("z-ai/glm-5.2".to_owned(), totals(189_895, 275_924)),
            ("z-ai/glm-5v-turbo".to_owned(), totals(53_159, 63_616)),
        ])
    }

    #[test]
    fn write_then_read_round_trips_per_model_keys() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join(CACHE_FILE_NAME);
        write_cache(&path, &two_models(), now()).expect("write");
        let got = read_cache(&path).expect("read back");
        assert_eq!(got.models, two_models());
        assert_eq!(got.updated_at, crate::support::time::iso_utc(now()));
    }

    #[test]
    fn a_missing_cache_reads_as_absent() {
        let dir = TempDir::new().unwrap();
        assert_eq!(read_cache(&dir.path().join("nope.json")), None);
    }

    #[test]
    fn a_corrupt_or_truncated_cache_reads_as_absent_not_fatal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(CACHE_FILE_NAME);
        let full = serde_json::to_string(&CachedCalibration {
            updated_at: crate::support::time::iso_utc(now()),
            models: two_models(),
        })
        .unwrap();
        let truncated = &full[..full.len() / 2];
        for junk in ["", "{", "not json at all", r#"{"models": 3}"#, truncated] {
            std::fs::write(&path, junk).unwrap();
            assert_eq!(read_cache(&path), None, "junk {junk:?} must read as absent");
        }
    }

    #[test]
    fn a_corrupt_cache_is_overwritten_by_the_next_write() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(CACHE_FILE_NAME);
        std::fs::write(&path, "{{{").unwrap();
        write_cache(&path, &two_models(), now()).expect("write over junk");
        assert_eq!(read_cache(&path).expect("read back").models, two_models());
    }

    #[test]
    fn nonpositive_totals_are_dropped_on_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(CACHE_FILE_NAME);
        let models = BTreeMap::from([
            ("good".to_owned(), totals(100, 150)),
            ("zero".to_owned(), totals(0, 0)),
            ("negative".to_owned(), totals(-5, 10)),
        ]);
        write_cache(&path, &models, now()).expect("write");
        let got = read_cache(&path).expect("read back");
        assert_eq!(got.models.keys().collect::<Vec<_>>(), ["good"]);
    }

    #[test]
    fn freshness_is_measured_against_the_ttl() {
        let fresh = crate::support::time::iso_utc(now() - Duration::days(CACHE_TTL_DAYS - 1));
        let old = crate::support::time::iso_utc(now() - Duration::days(CACHE_TTL_DAYS + 1));
        assert!(!is_stale(&fresh, now(), CACHE_TTL_DAYS));
        assert!(is_stale(&old, now(), CACHE_TTL_DAYS));
    }

    #[test]
    fn unparseable_and_future_stamps_are_stale() {
        assert!(is_stale("not a timestamp", now(), CACHE_TTL_DAYS));
        let future = crate::support::time::iso_utc(now() + Duration::days(1));
        assert!(is_stale(&future, now(), CACHE_TTL_DAYS));
    }

    #[test]
    fn the_cache_sits_beside_the_catalog_not_in_the_familiars_root() {
        let path = default_cache_path();
        assert_eq!(path.file_name().unwrap(), CACHE_FILE_NAME);
        assert!(path.is_absolute() || path.starts_with("data"));
        assert!(
            !path.to_string_lossy().contains("familiars/"),
            "regenerable totals must not sit in the state tree: {}",
            path.display()
        );
        assert_eq!(
            path.parent(),
            crate::model_diagnostics::cache::default_cache_path().parent(),
            "both caches share one directory"
        );
    }

    #[test]
    fn the_cached_shape_is_stable_json() {
        // Pinned so an older binary can still read a newer cache's fields.
        let cached = CachedCalibration {
            updated_at: "2026-08-18T12:00:00.000000+00:00".to_owned(),
            models: BTreeMap::from([("a/b".to_owned(), totals(100, 145))]),
        };
        let body = serde_json::to_string(&cached).unwrap();
        assert_eq!(
            body,
            r#"{"updated_at":"2026-08-18T12:00:00.000000+00:00","models":{"a/b":{"estimated":100,"actual":145}}}"#
        );
        assert_eq!(
            serde_json::from_str::<CachedCalibration>(&body).unwrap(),
            cached
        );
    }
}
