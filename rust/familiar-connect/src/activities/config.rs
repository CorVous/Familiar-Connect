//! `activities.toml` catalog + knobs loader (subsystem 11; Python `activities/config.py`).
//!
//! Sidecar per familiar: `data/familiars/<id>/activities.toml` (lorebook
//! precedent). Missing file or empty catalog ⇒ disabled config
//! ([`ActivitiesConfig::enabled`] false) — the engine is never constructed.
//! Present-but-invalid content fails loudly with [`ConfigError`] (character.toml
//! precedent), so a typo never silently disables a knob.
//!
//! The loader deep-merges the target over an optional defaults file (unlike
//! `character.toml`, the defaults file is optional — the shipped `_default`
//! skeleton is fully commented out). `catalog` lists replace wholesale on merge;
//! only table-vs-table pairs recurse (matching `config`'s deep-merge).

use std::collections::BTreeSet;
use std::path::Path;

use chrono::NaiveTime;
use toml::{Table, Value};

use crate::config::{ConfigError, parse_hhmm_range};

/// Allowed `content_source` values.
///
/// `"adapter"` is reserved for future adapter-backed types (e.g. youtube) and
/// is rejected with an explicit message until implemented.
const CONTENT_SOURCES: [&str; 1] = ["authored"];

/// Reserved catalog id — the window-scheduled sleep activity.
///
/// The wall-clock schedule (`window`/`grace_minutes`) lives in `character.toml`
/// `[sleep]`, not here; this id only marks WHICH activity the schedule drives.
/// The sleep entry's `duration_minutes` is optional (return is fixed at window
/// end, never a rolled duration).
pub const SLEEP_TYPE_ID: &str = "sleep";

/// Weekday tokens `mon..sun` → `datetime.weekday()` index (Mon=0 .. Sun=6).
const WEEKDAY_TOKENS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

/// Required per-entry keys (`duration_minutes` waived only for the sleep entry).
const ENTRY_REQUIRED: [&str; 4] = ["id", "label", "duration_minutes", "seed"];
/// All accepted per-entry keys.
const ENTRY_KEYS: [&str; 8] = [
    "id",
    "label",
    "duration_minutes",
    "seed",
    "reachable",
    "content_source",
    "active_days",
    "active_hours",
];
/// All accepted top-level keys.
const TOP_LEVEL_KEYS: [&str; 5] = [
    "archive_after_minutes",
    "idle_nudge_minutes",
    "min_gap_minutes",
    "active_hours",
    "catalog",
];

/// One `[[catalog]]` row from `activities.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivityType {
    /// Stable identifier referenced by the `start_activity` tool.
    pub id: String,
    /// Presence/status text shown while out.
    pub label: String,
    /// `(lo, hi)` roll range, `0 < lo <= hi`; `None` only on the sleep entry
    /// (window-scheduled).
    pub duration_minutes: Option<(i64, i64)>,
    /// A real @ping while out triggers a judgment turn.
    pub reachable: bool,
    /// Experience-text origin; see [`CONTENT_SOURCES`].
    pub content_source: String,
    /// Authored prompt seed for experience generation (dream prose for sleep).
    pub seed: String,
    /// Allowed weekdays as `datetime.weekday()` ints (Mon=0 .. Sun=6);
    /// `None` = any day.
    pub active_days: Option<BTreeSet<u8>>,
    /// `(start, end)` clock window; may wrap midnight; `None` = any time.
    pub active_hours: Option<(NaiveTime, NaiveTime)>,
}

impl Default for ActivityType {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            duration_minutes: None,
            reachable: true,
            content_source: "authored".to_owned(),
            seed: String::new(),
            active_days: None,
            active_hours: None,
        }
    }
}

/// Parsed `activities.toml`: catalog + engine knobs.
///
/// Disabled (falsy) when the catalog is empty — callers skip engine
/// construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivitiesConfig {
    /// Configured activity types (order preserved).
    pub catalog: Vec<ActivityType>,
    /// Absence ≥ this many minutes sets the all-channel archive watermark.
    pub archive_after_minutes: i64,
    /// Quiet threshold **and** nudge debounce, minutes.
    pub idle_nudge_minutes: i64,
    /// Post-return nudge gap (nudges only), minutes.
    pub min_gap_minutes: i64,
    /// `(start, end)` in display_tz; may wrap midnight; `None` = always.
    pub active_hours: Option<(NaiveTime, NaiveTime)>,
}

impl Default for ActivitiesConfig {
    fn default() -> Self {
        Self {
            catalog: Vec::new(),
            archive_after_minutes: 45,
            idle_nudge_minutes: 20,
            min_gap_minutes: 90,
            active_hours: None,
        }
    }
}

impl ActivitiesConfig {
    /// Falsy when the catalog is empty — callers skip engine construction.
    #[must_use]
    pub fn enabled(&self) -> bool {
        !self.catalog.is_empty()
    }
}

/// Load [`ActivitiesConfig`] from *path*, deep-merging over an optional
/// *defaults* file.
///
/// Unlike `character.toml`, the defaults file is optional (the `_default`
/// skeleton ships fully commented out). A missing *path* with no defaults
/// content ⇒ disabled config.
///
/// # Errors
/// [`ConfigError`] on invalid TOML, unknown keys, or validation failure.
pub fn load_activities_config(
    path: &Path,
    defaults_path: Option<&Path>,
) -> Result<ActivitiesConfig, ConfigError> {
    let defaults_data = match defaults_path {
        Some(p) => read_toml(p)?.unwrap_or_default(),
        None => Table::new(),
    };
    let target_data = read_toml(path)?.unwrap_or_default();
    let merged = deep_merge(&defaults_data, &target_data);
    parse_activities_config(&merged)
}

// ---------------------------------------------------------------------------
// TOML helpers (config's `_read_toml` / `_deep_merge` are private — mirror them)
// ---------------------------------------------------------------------------

fn read_toml(path: &Path) -> Result<Option<Table>, ConfigError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError(format!("failed to read config at {}: {e}", path.display())))?;
    match toml::from_str::<Table>(&text) {
        Ok(t) => Ok(Some(t)),
        Err(e) => Err(ConfigError(format!(
            "failed to parse TOML config at {}: {e}",
            path.display()
        ))),
    }
}

fn deep_merge(base: &Table, override_: &Table) -> Table {
    let mut result = Table::new();
    for (key, base_value) in base {
        if let Some(override_value) = override_.get(key) {
            if let (Value::Table(b), Value::Table(o)) = (base_value, override_value) {
                result.insert(key.clone(), Value::Table(deep_merge(b, o)));
            } else {
                result.insert(key.clone(), override_value.clone());
            }
        } else {
            result.insert(key.clone(), base_value.clone());
        }
    }
    for (key, value) in override_ {
        if !base.contains_key(key) {
            result.insert(key.clone(), value.clone());
        }
    }
    result
}

/// Render a sorted key list as a Python-`repr`-ish `['a', 'b']` blob (the
/// substring the tests match is the key name; the shape mirrors Python).
fn sorted_list_repr<I: IntoIterator<Item = String>>(items: I) -> String {
    let mut v: Vec<String> = items.into_iter().collect();
    v.sort();
    let inner: Vec<String> = v.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// Alphabetically-sorted `", "`-joined key list for the `valid keys:` portion
/// of unknown-key errors — mirrors Python's `", ".join(sorted(_KEYS))` (the
/// declared arrays are in authoring order, not sorted).
fn sorted_join(keys: &[&str]) -> String {
    let mut v: Vec<&str> = keys.to_vec();
    v.sort_unstable();
    v.join(", ")
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

fn parse_activities_config(data: &Table) -> Result<ActivitiesConfig, ConfigError> {
    let unknown: Vec<String> = data
        .keys()
        .filter(|k| !TOP_LEVEL_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        let valid = sorted_join(&TOP_LEVEL_KEYS);
        return Err(ConfigError(format!(
            "unknown activities.toml keys: {}; valid keys: {valid}",
            sorted_list_repr(unknown)
        )));
    }

    let mut cfg = ActivitiesConfig::default();
    for knob in [
        "archive_after_minutes",
        "idle_nudge_minutes",
        "min_gap_minutes",
    ] {
        let Some(value) = data.get(knob) else {
            continue;
        };
        // toml distinguishes booleans from integers, so `as_integer` already
        // rejects `= true` (the bool-is-not-int guard comes free).
        let Some(n) = value.as_integer().filter(|n| *n > 0) else {
            return Err(ConfigError(format!(
                "{knob} must be a positive integer, got {}",
                py_value_repr(value)
            )));
        };
        match knob {
            "archive_after_minutes" => cfg.archive_after_minutes = n,
            "idle_nudge_minutes" => cfg.idle_nudge_minutes = n,
            _ => cfg.min_gap_minutes = n,
        }
    }

    if let Some(v) = data.get("active_hours") {
        cfg.active_hours = Some(parse_hhmm_range(v, "active_hours")?);
    }

    let catalog_raw = match data.get("catalog") {
        None => Vec::new(),
        Some(Value::Array(arr)) => arr.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[[catalog]] must be an array of tables, got {}",
                toml_type_name(other)
            )));
        }
    };
    let mut catalog: Vec<ActivityType> = Vec::with_capacity(catalog_raw.len());
    for (idx, raw) in catalog_raw.iter().enumerate() {
        catalog.push(parse_catalog_entry(raw, idx)?);
    }

    let mut seen: BTreeSet<String> = BTreeSet::new();
    for entry in &catalog {
        if !seen.insert(entry.id.clone()) {
            return Err(ConfigError(format!("duplicate catalog id '{}'", entry.id)));
        }
    }

    cfg.catalog = catalog;
    Ok(cfg)
}

#[allow(
    clippy::too_many_lines,
    reason = "faithful 1:1 transliteration of the Python _parse_catalog_entry sequence"
)]
fn parse_catalog_entry(raw: &Value, idx: usize) -> Result<ActivityType, ConfigError> {
    let Value::Table(entry) = raw else {
        return Err(ConfigError(format!(
            "[[catalog]] entry #{idx} must be a table, got {}",
            toml_type_name(raw)
        )));
    };

    let unknown: Vec<String> = entry
        .keys()
        .filter(|k| !ENTRY_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        let valid = sorted_join(&ENTRY_KEYS);
        return Err(ConfigError(format!(
            "unknown keys in [[catalog]] entry #{idx}: {}; valid keys: {valid}",
            sorted_list_repr(unknown)
        )));
    }

    let is_sleep = entry.get("id").and_then(Value::as_str) == Some(SLEEP_TYPE_ID);
    let missing: Vec<String> = ENTRY_REQUIRED
        .iter()
        .filter(|k| !entry.contains_key(**k))
        // window-scheduled sleep: return is fixed at window end, duration unused.
        .filter(|k| !(is_sleep && **k == "duration_minutes"))
        .map(|k| (*k).to_owned())
        .collect();
    if !missing.is_empty() {
        return Err(ConfigError(format!(
            "[[catalog]] entry #{idx} missing required keys: {}",
            sorted_list_repr(missing)
        )));
    }

    let mut resolved: Vec<String> = Vec::with_capacity(3);
    for key in ["id", "label", "seed"] {
        let Some(s) = entry
            .get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        else {
            return Err(ConfigError(format!(
                "[[catalog]] entry #{idx}: {key} must be a non-empty string"
            )));
        };
        resolved.push(s.to_owned());
    }
    let entry_id = resolved[0].clone();

    let duration_minutes = match entry.get("duration_minutes") {
        Some(v) => Some(parse_duration_minutes(v, &entry_id)?),
        None => None,
    };

    let reachable = match entry.get("reachable") {
        None => true,
        Some(Value::Boolean(b)) => *b,
        Some(_) => {
            return Err(ConfigError(format!(
                "[[catalog]] '{entry_id}': reachable must be a boolean"
            )));
        }
    };

    let content_source = match entry.get("content_source") {
        None => "authored".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(ConfigError(format!(
                "[[catalog]] '{entry_id}': content_source must be a string"
            )));
        }
    };
    if content_source == "adapter" {
        return Err(ConfigError(format!(
            "[[catalog]] '{entry_id}': content_source 'adapter' is reserved for \
             future adapter-backed types; use 'authored'"
        )));
    }
    if !CONTENT_SOURCES.contains(&content_source.as_str()) {
        let valid = CONTENT_SOURCES.join(", ");
        return Err(ConfigError(format!(
            "[[catalog]] '{entry_id}': unknown content_source '{content_source}'; \
             valid values: {valid}"
        )));
    }

    let active_days = match entry.get("active_days") {
        Some(v) => Some(parse_active_days(v, &entry_id)?),
        None => None,
    };

    let active_hours = match entry.get("active_hours") {
        Some(v) => Some(
            parse_hhmm_range(v, "active_hours")
                .map_err(|e| ConfigError(format!("[[catalog]] '{entry_id}': {}", e.0)))?,
        ),
        None => None,
    };

    Ok(ActivityType {
        id: entry_id,
        label: resolved[1].clone(),
        duration_minutes,
        reachable,
        content_source,
        seed: resolved[2].clone(),
        active_days,
        active_hours,
    })
}

fn parse_active_days(value: &Value, entry_id: &str) -> Result<BTreeSet<u8>, ConfigError> {
    let valid = WEEKDAY_TOKENS.join(", ");
    let non_empty_list = match value {
        Value::Array(tokens) if !tokens.is_empty() => tokens,
        _ => {
            return Err(ConfigError(format!(
                "[[catalog]] '{entry_id}': active_days must be a non-empty list \
                 of weekday tokens ({valid})"
            )));
        }
    };
    let mut days: BTreeSet<u8> = BTreeSet::new();
    for token in non_empty_list {
        let idx = token
            .as_str()
            .and_then(|t| WEEKDAY_TOKENS.iter().position(|w| *w == t));
        let Some(idx) = idx else {
            return Err(ConfigError(format!(
                "[[catalog]] '{entry_id}': active_days has unknown weekday token \
                 {}; valid tokens: {valid}",
                py_value_repr(token)
            )));
        };
        days.insert(u8::try_from(idx).expect("weekday index fits u8"));
    }
    Ok(days)
}

fn parse_duration_minutes(value: &Value, entry_id: &str) -> Result<(i64, i64), ConfigError> {
    let err = || {
        ConfigError(format!(
            "[[catalog]] '{entry_id}': duration_minutes must be a [lo, hi] pair \
             of minutes with 0 < lo <= hi, got {}",
            py_value_repr(value)
        ))
    };
    let Value::Array(arr) = value else {
        return Err(err());
    };
    if arr.len() != 2 {
        return Err(err());
    }
    // `as_integer` rejects booleans (toml keeps them distinct) — the Python
    // `isinstance(_, bool)` guard is free here.
    let (Some(lo), Some(hi)) = (arr[0].as_integer(), arr[1].as_integer()) else {
        return Err(err());
    };
    if !(0 < lo && lo <= hi) {
        return Err(err());
    }
    Ok((lo, hi))
}

/// TOML value type name (mirrors Python's `type(x).__name__` in messages).
const fn toml_type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "str",
        Value::Integer(_) => "int",
        Value::Float(_) => "float",
        Value::Boolean(_) => "bool",
        Value::Datetime(_) => "datetime",
        Value::Array(_) => "list",
        Value::Table(_) => "dict",
    }
}

/// A short `repr`-ish rendering of a scalar TOML value for error messages.
///
/// Booleans render Python-style (`True`/`False`) to match `{value!r}`: a TOML
/// `true` reaches these paths as a `Value::Boolean`, whose `to_string()` would
/// otherwise emit the lowercase `true`/`false`.
fn py_value_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        Value::Boolean(b) => if *b { "True" } else { "False" }.to_owned(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivitiesConfig, ActivityType, SLEEP_TYPE_ID, load_activities_config};
    use chrono::NaiveTime;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    const VALID_ENTRY: &str = "\
[[catalog]]
id = \"creek_walk\"
label = \"out for a creek walk\"
duration_minutes = [20, 45]
reachable = true
content_source = \"authored\"
seed = \"Walk along the creek behind the house.\"
";

    const SLEEP_ENTRY: &str = "\
[[catalog]]
id = \"sleep\"
label = \"asleep\"
reachable = false
seed = \"The night's dream, told on waking.\"
";

    fn write_activities(dir: &Path, content: &str) -> PathBuf {
        let path = dir.join("activities.toml");
        std::fs::write(&path, content).unwrap();
        path
    }

    fn t(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    fn load(content: &str) -> Result<ActivitiesConfig, crate::config::ConfigError> {
        let dir = tempdir().unwrap();
        let path = write_activities(dir.path(), content);
        load_activities_config(&path, None)
    }

    // --- dataclass defaults -------------------------------------------------

    #[test]
    fn config_knob_defaults() {
        let cfg = ActivitiesConfig::default();
        assert!(cfg.catalog.is_empty());
        assert_eq!(cfg.archive_after_minutes, 45);
        assert_eq!(cfg.idle_nudge_minutes, 20);
        assert_eq!(cfg.min_gap_minutes, 90);
        assert!(cfg.active_hours.is_none());
    }

    #[test]
    fn empty_catalog_is_disabled() {
        assert!(!ActivitiesConfig::default().enabled());
    }

    #[test]
    fn populated_catalog_is_enabled() {
        let entry = ActivityType {
            id: "creek_walk".into(),
            label: "out for a creek walk".into(),
            duration_minutes: Some((20, 45)),
            seed: "Walk along the creek.".into(),
            ..Default::default()
        };
        let cfg = ActivitiesConfig {
            catalog: vec![entry],
            ..Default::default()
        };
        assert!(cfg.enabled());
    }

    // --- load ---------------------------------------------------------------

    #[test]
    fn missing_file_yields_disabled_config() {
        let dir = tempdir().unwrap();
        let cfg = load_activities_config(&dir.path().join("activities.toml"), None).unwrap();
        assert!(!cfg.enabled());
    }

    #[test]
    fn knobs_without_catalog_stays_disabled() {
        let cfg = load("archive_after_minutes = 60\n").unwrap();
        assert_eq!(cfg.archive_after_minutes, 60);
        assert!(!cfg.enabled());
    }

    #[test]
    fn valid_entry_parsed() {
        let cfg = load(VALID_ENTRY).unwrap();
        assert!(cfg.enabled());
        assert_eq!(cfg.catalog.len(), 1);
        let e = &cfg.catalog[0];
        assert_eq!(e.id, "creek_walk");
        assert_eq!(e.label, "out for a creek walk");
        assert_eq!(e.duration_minutes, Some((20, 45)));
        assert!(e.reachable);
        assert_eq!(e.content_source, "authored");
        assert_eq!(e.seed, "Walk along the creek behind the house.");
    }

    #[test]
    fn entry_optional_fields_default() {
        let cfg = load(
            "[[catalog]]\nid = \"creek_walk\"\nlabel = \"out for a creek walk\"\n\
             duration_minutes = [20, 45]\nseed = \"Walk along the creek.\"\n",
        )
        .unwrap();
        let e = &cfg.catalog[0];
        assert!(e.reachable);
        assert_eq!(e.content_source, "authored");
    }

    #[test]
    fn malformed_toml_raises() {
        let err = load("not [valid toml\n").unwrap_err();
        assert!(err.0.contains("failed to parse TOML"), "{}", err.0);
    }

    #[test]
    fn merges_over_defaults() {
        let dir = tempdir().unwrap();
        let defaults = dir.path().join("defaults.toml");
        std::fs::write(
            &defaults,
            "archive_after_minutes = 30\nidle_nudge_minutes = 15\n",
        )
        .unwrap();
        let path = write_activities(
            dir.path(),
            &format!("archive_after_minutes = 60\n{VALID_ENTRY}"),
        );
        let cfg = load_activities_config(&path, Some(&defaults)).unwrap();
        assert_eq!(cfg.archive_after_minutes, 60);
        assert_eq!(cfg.idle_nudge_minutes, 15);
        assert!(cfg.enabled());
    }

    #[test]
    fn missing_defaults_file_ok() {
        let dir = tempdir().unwrap();
        let path = write_activities(dir.path(), VALID_ENTRY);
        let cfg = load_activities_config(&path, Some(&dir.path().join("missing.toml"))).unwrap();
        assert!(cfg.enabled());
    }

    // --- knob validation ----------------------------------------------------

    #[test]
    fn knob_must_be_positive_int() {
        for knob in [
            "archive_after_minutes",
            "idle_nudge_minutes",
            "min_gap_minutes",
        ] {
            let err = load(&format!("{knob} = 0\n")).unwrap_err();
            assert!(err.0.contains(knob), "{}", err.0);
        }
    }

    #[test]
    fn knob_rejects_non_int() {
        let err = load("min_gap_minutes = \"soon\"\n").unwrap_err();
        assert!(err.0.contains("min_gap_minutes"), "{}", err.0);
    }

    #[test]
    fn knob_rejects_bool() {
        // Python `bool` is an `int`; the explicit rejection is replicated free
        // by toml's distinct Boolean variant.
        let err = load("min_gap_minutes = true\n").unwrap_err();
        assert!(err.0.contains("min_gap_minutes"), "{}", err.0);
        // Python renders the offender via `{value!r}` = `repr(True)` = `True`,
        // not TOML's lowercase `true`.
        assert_eq!(
            err.0,
            "min_gap_minutes must be a positive integer, got True"
        );
    }

    #[test]
    fn unknown_top_level_key_rejected() {
        let err = load("archive_minutes = 45\n").unwrap_err();
        assert!(err.0.contains("archive_minutes"), "{}", err.0);
        // The `valid keys:` list is alphabetically sorted (Python
        // `", ".join(sorted(_TOP_LEVEL_KEYS))`), not array-declaration order.
        assert_eq!(
            err.0,
            "unknown activities.toml keys: ['archive_minutes']; valid keys: \
             active_hours, archive_after_minutes, catalog, idle_nudge_minutes, \
             min_gap_minutes"
        );
    }

    // --- catalog validation -------------------------------------------------

    #[test]
    fn required_keys() {
        for missing in ["id", "label", "duration_minutes", "seed"] {
            let content: String = VALID_ENTRY
                .lines()
                .filter(|line| !line.starts_with(&format!("{missing} ")))
                .collect::<Vec<_>>()
                .join("\n");
            let err = load(&format!("{content}\n")).unwrap_err();
            assert!(err.0.contains(missing), "missing {missing}: {}", err.0);
        }
    }

    #[test]
    fn unknown_entry_key_rejected() {
        let err = load(&format!("{VALID_ENTRY}colour = \"blue\"\n")).unwrap_err();
        assert!(err.0.contains("colour"), "{}", err.0);
        // Sorted valid-keys list (Python `", ".join(sorted(_ENTRY_KEYS))`).
        assert_eq!(
            err.0,
            "unknown keys in [[catalog]] entry #0: ['colour']; valid keys: \
             active_days, active_hours, content_source, duration_minutes, id, \
             label, reachable, seed"
        );
    }

    #[test]
    fn duplicate_ids_rejected() {
        let err = load(&format!("{VALID_ENTRY}\n{VALID_ENTRY}")).unwrap_err();
        assert!(err.0.contains("duplicate"), "{}", err.0);
    }

    #[test]
    fn duration_minutes_invalid() {
        for value in [
            "[20]",
            "[20, 45, 60]",
            "[0, 45]",
            "[45, 20]",
            "[\"a\", \"b\"]",
            "20",
            "[-5, 45]",
        ] {
            let content = VALID_ENTRY.replace(
                "duration_minutes = [20, 45]",
                &format!("duration_minutes = {value}"),
            );
            let err = load(&content).unwrap_err();
            assert!(
                err.0.contains("duration_minutes"),
                "value {value}: {}",
                err.0
            );
        }
    }

    #[test]
    fn duration_minutes_equal_bounds_ok() {
        let content = VALID_ENTRY.replace("[20, 45]", "[30, 30]");
        let cfg = load(&content).unwrap();
        assert_eq!(cfg.catalog[0].duration_minutes, Some((30, 30)));
    }

    #[test]
    fn content_source_adapter_reserved() {
        let content = VALID_ENTRY.replace("\"authored\"", "\"adapter\"");
        let err = load(&content).unwrap_err();
        assert!(err.0.contains("reserved"), "{}", err.0);
    }

    #[test]
    fn content_source_unknown_rejected() {
        let content = VALID_ENTRY.replace("\"authored\"", "\"telepathy\"");
        let err = load(&content).unwrap_err();
        assert!(err.0.contains("content_source"), "{}", err.0);
    }

    #[test]
    fn empty_id_rejected() {
        let content = VALID_ENTRY.replace("id = \"creek_walk\"", "id = \"\"");
        let err = load(&content).unwrap_err();
        assert!(err.0.contains("id"), "{}", err.0);
    }

    // --- active_hours (top-level) -------------------------------------------

    #[test]
    fn active_hours_parsed_to_times() {
        let cfg = load("active_hours = \"10:00-23:00\"\n").unwrap();
        assert_eq!(cfg.active_hours, Some((t(10, 0), t(23, 0))));
    }

    #[test]
    fn active_hours_midnight_wrap_allowed() {
        let cfg = load("active_hours = \"22:00-02:30\"\n").unwrap();
        assert_eq!(cfg.active_hours, Some((t(22, 0), t(2, 30))));
    }

    #[test]
    fn active_hours_bad_format_rejected() {
        for value in ["10:00", "10-23", "25:00-23:00", "10:99-23:00", "always", ""] {
            let err = load(&format!("active_hours = \"{value}\"\n")).unwrap_err();
            assert!(err.0.contains("active_hours"), "value {value}: {}", err.0);
        }
    }

    #[test]
    fn active_hours_equal_start_end_rejected() {
        let err = load("active_hours = \"10:00-10:00\"\n").unwrap_err();
        assert!(err.0.contains("active_hours"), "{}", err.0);
    }

    // --- default skeleton ---------------------------------------------------

    #[test]
    fn default_skeleton_ships_disabled() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/familiars/_default/activities.toml"
        ));
        assert!(path.exists(), "skeleton missing at {}", path.display());
        let cfg = load_activities_config(path, None).unwrap();
        assert!(!cfg.enabled());
    }

    // --- per-entry schedule -------------------------------------------------

    #[test]
    fn per_entry_both_fields_parsed() {
        let content = format!(
            "{VALID_ENTRY}active_days = [\"mon\", \"tue\", \"wed\", \"thu\", \"fri\"]\n\
             active_hours = \"09:00-17:00\"\n"
        );
        let e = load(&content).unwrap().catalog.remove(0);
        assert_eq!(e.active_days, Some(BTreeSet::from([0, 1, 2, 3, 4])));
        assert_eq!(e.active_hours, Some((t(9, 0), t(17, 0))));
    }

    #[test]
    fn per_entry_absent_fields_default_to_none() {
        let e = load(VALID_ENTRY).unwrap().catalog.remove(0);
        assert!(e.active_days.is_none());
        assert!(e.active_hours.is_none());
    }

    #[test]
    fn weekday_mapping_pinned() {
        for (tokens, expected) in [
            ("[\"mon\"]", BTreeSet::from([0u8])),
            ("[\"sun\"]", BTreeSet::from([6])),
            ("[\"mon\", \"wed\", \"fri\"]", BTreeSet::from([0, 2, 4])),
        ] {
            let content = format!("{VALID_ENTRY}active_days = {tokens}\n");
            let e = load(&content).unwrap().catalog.remove(0);
            assert_eq!(e.active_days, Some(expected));
        }
    }

    #[test]
    fn bad_weekday_token_rejected() {
        for token in ["\"funday\"", "\"Mon\""] {
            let content = format!("{VALID_ENTRY}active_days = [{token}]\n");
            let err = load(&content).unwrap_err();
            assert!(err.0.contains("creek_walk"), "{}", err.0);
            assert!(err.0.contains("active_days"), "{}", err.0);
        }
    }

    #[test]
    fn non_string_weekday_token_rejected() {
        let content = format!("{VALID_ENTRY}active_days = [\"mon\", 5]\n");
        let err = load(&content).unwrap_err();
        assert!(
            err.0.contains("creek_walk") && err.0.contains("active_days"),
            "{}",
            err.0
        );
    }

    #[test]
    fn non_list_active_days_rejected() {
        for value in ["\"mon\"", "5"] {
            let content = format!("{VALID_ENTRY}active_days = {value}\n");
            let err = load(&content).unwrap_err();
            assert!(
                err.0.contains("creek_walk") && err.0.contains("active_days"),
                "{}",
                err.0
            );
        }
    }

    #[test]
    fn empty_active_days_rejected() {
        let content = format!("{VALID_ENTRY}active_days = []\n");
        let err = load(&content).unwrap_err();
        assert!(
            err.0.contains("creek_walk") && err.0.contains("active_days"),
            "{}",
            err.0
        );
    }

    #[test]
    fn per_entry_malformed_active_hours_rejected() {
        for value in ["9am-5pm", "17:00-17:00"] {
            let content = format!("{VALID_ENTRY}active_hours = \"{value}\"\n");
            let err = load(&content).unwrap_err();
            // "HH:MM" comes only from parse_hhmm_range — proves the parse path
            // is wired in; the entry id is prefixed for context.
            assert!(err.0.contains("HH:MM"), "value {value}: {}", err.0);
            assert!(err.0.contains("creek_walk"), "value {value}: {}", err.0);
        }
    }

    // --- sleep entry --------------------------------------------------------

    #[test]
    fn sleep_entry_parses() {
        let e = load(SLEEP_ENTRY).unwrap().catalog.remove(0);
        assert_eq!(e.id, SLEEP_TYPE_ID);
        assert!(!e.reachable);
    }

    #[test]
    fn duration_minutes_optional_for_sleep_entry() {
        let e = load(SLEEP_ENTRY).unwrap().catalog.remove(0);
        assert!(e.duration_minutes.is_none());
    }

    #[test]
    fn duration_minutes_still_parsed_if_present() {
        let e = load(&format!("{SLEEP_ENTRY}duration_minutes = [400, 500]\n"))
            .unwrap()
            .catalog
            .remove(0);
        assert_eq!(e.duration_minutes, Some((400, 500)));
    }

    #[test]
    fn non_sleep_entry_still_requires_duration() {
        let content = VALID_ENTRY.replace("duration_minutes = [20, 45]\n", "");
        let err = load(&content).unwrap_err();
        assert!(err.0.contains("duration_minutes"), "{}", err.0);
    }

    #[test]
    fn window_no_longer_a_catalog_key() {
        let err = load(&format!("{SLEEP_ENTRY}window = \"00:00-08:00\"\n")).unwrap_err();
        assert!(err.0.contains("unknown keys"), "{}", err.0);
    }

    #[test]
    fn grace_no_longer_a_catalog_key() {
        let err = load(&format!("{SLEEP_ENTRY}grace_minutes = 30\n")).unwrap_err();
        assert!(err.0.contains("unknown keys"), "{}", err.0);
    }
}
