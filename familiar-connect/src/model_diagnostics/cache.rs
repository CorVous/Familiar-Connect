//! On-disk last-known-good OpenRouter catalog (#204).
//!
//! Metadata-driven capability selection has to be affordable at boot, and boot
//! must never wait on the network — the invariant `model_diagnostics` and
//! `commands::run` both spell out. So the catalog is split in two:
//!
//! - **Read** — [`read_cache`] at startup, synchronous, local file only. What
//!   it returns resolves this process's tri-state `multimodal` flags.
//! - **Refresh** — background, after boot. Fetches, [`write_cache`]s, and the
//!   fresh data takes effect on the *next* start. Nothing this process does
//!   waits on it.
//!
//! A first-ever run with no cache reads `None` and every slot falls back to its
//! configured value. A truncated, corrupt, or wrong-shaped file is also `None`:
//! the cache is a cheap optimisation, never a failure mode.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::config::LLMSlotConfig;
use crate::log_style as ls;
use crate::model_diagnostics::{IMAGE_MODALITY, ModelCapabilities, lookup};
use crate::support::time::{iso_utc, parse_iso};

/// Cache file name under the per-user cache directory.
pub const CACHE_FILE_NAME: &str = "openrouter-models.json";

/// How long a cached catalog is considered fresh.
///
/// OpenRouter adds models and corrects metadata on the order of days, and the
/// only cost of a stale entry is one wrong auto-detection an explicit
/// `multimodal` override always beats. A day keeps a restart loop off the
/// `/models` endpoint while never letting the catalog drift far.
pub const CACHE_TTL_HOURS: i64 = 24;

/// The serialized cache: a fetch timestamp plus the parsed catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedCatalog {
    /// When the catalog was fetched (`support::time::iso_utc`).
    pub fetched_at: String,
    /// Capability rows, as [`parse_model_catalog`](super::parse_model_catalog)
    /// produced them.
    pub models: Vec<ModelCapabilities>,
}

/// Per-user cache path for the catalog.
///
/// The platform **cache** directory, not the data directory holding familiars:
/// this file is entirely regenerable from the network, so it must sit where
/// clearing caches is safe and where it can never be mistaken for state. Same
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
/// Never an error: a corrupt file is treated exactly as an absent one and the
/// next successful refresh overwrites it.
#[must_use]
pub fn read_cache(path: &Path) -> Option<CachedCatalog> {
    let body = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<CachedCatalog>(&body).ok()
}

/// Write `models` to `path`, stamped now, creating parent directories.
///
/// # Errors
/// I/O or serialization failure. Callers log and move on — a cache that cannot
/// be written costs a network fetch next boot, nothing more.
pub fn write_cache(
    path: &Path,
    models: &[ModelCapabilities],
    now: DateTime<Utc>,
) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let cached = CachedCatalog {
        fetched_at: iso_utc(now),
        models: models.to_vec(),
    };
    let body = serde_json::to_string(&cached).map_err(std::io::Error::other)?;
    std::fs::write(path, body)
}

/// Whether a cache stamped `fetched_at` is older than `ttl_hours` at `now`.
///
/// Unparseable and future-dated stamps both count as stale: refreshing is
/// cheap, and a nonsense timestamp must never pin a cache forever.
#[must_use]
pub fn is_stale(fetched_at: &str, now: DateTime<Utc>, ttl_hours: i64) -> bool {
    parse_iso(fetched_at).is_none_or(|then| {
        let age = now.signed_duration_since(then);
        age < Duration::zero() || age >= Duration::hours(ttl_hours)
    })
}

/// Whether the catalog says `model` accepts image input; `None` when the
/// catalog does not list it (variant suffixes fall back to the base id).
#[must_use]
pub fn detect_multimodal(catalog: &[ModelCapabilities], model: &str) -> Option<bool> {
    lookup(catalog, model).map(|m| m.input_modalities.iter().any(|s| s == IMAGE_MODALITY))
}

/// Fill every unset `multimodal` from `catalog`, leaving explicit values alone.
///
/// Returns the slots it decided, `(slot, enabled)`, for logging. Slots the
/// catalog does not list keep `None` and resolve to `false` downstream — the
/// pre-detection default.
pub fn apply_detected_multimodal(
    slots: &mut BTreeMap<String, LLMSlotConfig>,
    catalog: &[ModelCapabilities],
) -> Vec<(String, bool)> {
    let mut applied = Vec::new();
    for (name, cfg) in slots.iter_mut() {
        if cfg.multimodal.is_some() || cfg.model.is_empty() {
            continue;
        }
        if let Some(detected) = detect_multimodal(catalog, &cfg.model) {
            cfg.multimodal = Some(detected);
            applied.push((name.clone(), detected));
        }
    }
    applied
}

/// Read the cache and auto-detect into `slots`, logging nothing.
///
/// For a second config copy of the same familiar, where the decisions have
/// already been reported. Local file I/O only — safe on the boot path.
pub fn resolve_from_cache_quietly(slots: &mut BTreeMap<String, LLMSlotConfig>, path: &Path) {
    if let Some(cached) = read_cache(path) {
        apply_detected_multimodal(slots, &cached.models);
    }
}

/// Read the cache and auto-detect into `slots`; one `[Config]` line per
/// decision. Local file I/O only — safe on the boot path.
pub fn resolve_capabilities_from_cache(slots: &mut BTreeMap<String, LLMSlotConfig>, path: &Path) {
    let Some(cached) = read_cache(path) else {
        return;
    };
    for (slot, enabled) in apply_detected_multimodal(slots, &cached.models) {
        tracing::info!(
            target: "familiar_connect.llm",
            "{} {} {} {}",
            ls::tag("Config", ls::W),
            ls::kv("slot", &slot),
            ls::kv_styled("multimodal", if enabled { "auto=on" } else { "auto=off" }, ls::W, ls::LM),
            ls::kv("source", "cached OpenRouter catalog"),
        );
    }
}
