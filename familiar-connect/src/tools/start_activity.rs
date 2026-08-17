//! `start_activity` tool (subsystem 08).
//!
//! Stages a global absence via [`StartActivityEngine::defer_start`]; actual
//! departure is applied by the engine's `end_turn` after the reply ships (the
//! `shift_focus` deferral precedent). The activity enum is built from the
//! engine's catalog at registry-build time, so each familiar's sidecar shapes
//! the schema. The description carries the entire when-to-go policy — zero
//! character-card growth by design — and is config-sourced
//! (`[prompt].start_activity_description`).
//!
//! The engine itself is Layer 3 (subsystem 11). This module defines only the
//! narrow [`StartActivityEngine`] seam and [`ActivityCatalogEntry`] the tool
//! needs; the real engine implements the trait.

use std::sync::Arc;

use chrono::NaiveTime;
use chrono_tz::Tz;
use serde_json::{Value, json};

use crate::log_style as ls;
use crate::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput};
use crate::tools::silent::SILENT_RESULT;

/// Weekday index (Mon=0 .. Sun=6) → abbreviation for availability hints.
const WEEKDAY_ABBR: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// One catalog entry the tool needs to render its enum + availability hints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivityCatalogEntry {
    /// Type id (the enum value the model picks).
    pub id: String,
    /// Human label.
    pub label: String,
    /// Weekdays the entry is choosable (Mon=0), if scheduled.
    pub active_days: Option<Vec<u8>>,
    /// Hour window the entry is choosable, if scheduled.
    pub active_hours: Option<(NaiveTime, NaiveTime)>,
}

/// The structural slice of the activity engine the tool needs.
pub trait StartActivityEngine: Send + Sync {
    /// The catalog of choosable activities (snapshot at build time).
    fn catalog(&self) -> Vec<ActivityCatalogEntry>;
    /// Zone the catalog's `active_hours` are expressed in. Named (not
    /// abbreviated): the schema is built once at boot and must not go stale
    /// across a DST transition.
    fn display_tz(&self) -> Tz {
        Tz::UTC
    }
    /// Whether the familiar is already out.
    fn is_active(&self) -> bool;
    /// Stage a departure; returns the engine's JSON result (`{"error": ...}` on
    /// rejection).
    fn defer_start(&self, type_id: &str, note: Option<&str>) -> Value;
}

fn error_output(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

/// `'id' = label` plus a `[days hours]` clause when scheduled.
fn entry_description(a: &ActivityCatalogEntry) -> String {
    let base = format!("'{}' = {}", a.id, a.label);
    let mut parts: Vec<String> = Vec::new();
    if let Some(days) = &a.active_days {
        let mut sorted = days.clone();
        sorted.sort_unstable();
        let abbrs: Vec<&str> = sorted.iter().map(|d| WEEKDAY_ABBR[*d as usize]).collect();
        parts.push(abbrs.join(" "));
    }
    if let Some((start, end)) = &a.active_hours {
        parts.push(format!("{}-{}", start.format("%H:%M"), end.format("%H:%M")));
    }
    if parts.is_empty() {
        base
    } else {
        format!("{base} [{}]", parts.join(" "))
    }
}

fn start_activity_handler(
    engine: &dyn StartActivityEngine,
    args: &Value,
    _ctx: &ToolContext,
) -> ToolOutput {
    if engine.is_active() {
        // Calling start_activity while already out signals stay-out intent —
        // the silent sentinel keeps meta narration off the channel.
        tracing::info!(
            "{} {}",
            ls::tag("\u{1f6b6} start_activity", ls::G),
            ls::kv_styled("outcome", "already-out \u{2192} silent", ls::W, ls::LW),
        );
        return ToolOutput::Text(SILENT_RESULT.to_owned());
    }

    let activity = match args.get("activity") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return error_output("missing or empty 'activity' (string)"),
    };
    let note = match args.get("note") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return error_output("'note' must be a string"),
    };

    let result = engine.defer_start(&activity, note.as_deref());
    let outcome = if result.get("error").is_some() {
        "error"
    } else {
        "staged"
    };
    tracing::info!(
        "{} {} {}",
        ls::tag("\u{1f6b6} start_activity", ls::G),
        ls::kv_styled("activity", &activity, ls::W, ls::G),
        ls::kv_styled("outcome", outcome, ls::W, ls::LW),
    );
    ToolOutput::Text(result.to_string())
}

/// Build the `start_activity` tool bound to `engine`.
///
/// `description` is the when-to-go policy from
/// `[prompt].start_activity_description` — roleplay guidance, not API contract,
/// so it lives in config (#151). The `activity` enum and its availability hints
/// stay code-built from the catalog.
#[must_use]
pub fn build_start_activity_tool(engine: Arc<dyn StartActivityEngine>, description: &str) -> Tool {
    let catalog = engine.catalog();
    let enum_ids: Vec<Value> = catalog.iter().map(|e| json!(e.id)).collect();
    // One zone statement for the whole list rather than a suffix per entry —
    // the bare `HH:MM` windows would otherwise read as UTC.
    let zone_clause = if catalog.iter().any(|e| e.active_hours.is_some()) {
        format!(" (hours in {})", engine.display_tz().name())
    } else {
        String::new()
    };
    let activity_desc = format!(
        "What to go do{zone_clause}: {}.",
        catalog
            .iter()
            .map(entry_description)
            .collect::<Vec<_>>()
            .join("; ")
    );

    Tool::new(
        "start_activity",
        description,
        json!({
            "type": "object",
            "properties": {
                "activity": {
                    "type": "string",
                    "enum": enum_ids,
                    "description": activity_desc,
                },
                "note": {
                    "type": "string",
                    "description":
                        "Optional intent — what you have in mind for it; seeds the experience.",
                },
            },
            "required": ["activity"],
        }),
        Arc::new(FnHandler(move |args: Value, ctx: ToolContext| {
            let engine = Arc::clone(&engine);
            async move { Ok(start_activity_handler(engine.as_ref(), &args, &ctx)) }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_description_reaches_the_tool_schema() {
        let engine = Arc::new(FakeEngine {
            entries: vec![unscheduled_entry()],
            tz: Tz::UTC,
        });
        let tool = build_start_activity_tool(engine, "GO_OUTSIDE_MARKER");
        assert_eq!(tool.description, "GO_OUTSIDE_MARKER");
    }

    #[test]
    fn unconfigured_description_is_empty_not_a_code_copy() {
        let engine = Arc::new(FakeEngine {
            entries: vec![unscheduled_entry()],
            tz: Tz::UTC,
        });
        assert!(build_start_activity_tool(engine, "").description.is_empty());
    }

    #[test]
    fn entry_description_scheduled_days_and_hours() {
        let e = ActivityCatalogEntry {
            id: "weekday_rounds".into(),
            label: "weekday rounds".into(),
            active_days: Some(vec![0, 1, 2, 3, 4]),
            active_hours: Some((
                NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            )),
        };
        let d = entry_description(&e);
        assert!(d.contains("Mon Tue Wed Thu Fri"));
        assert!(!d.contains("Sun"));
        assert!(d.contains("09:00-17:00"));
        assert!(d.starts_with("'weekday_rounds' = weekday rounds ["));
    }

    #[test]
    fn entry_description_days_only() {
        let e = ActivityCatalogEntry {
            id: "market_day".into(),
            label: "the saturday market".into(),
            active_days: Some(vec![5]),
            active_hours: None,
        };
        assert!(entry_description(&e).contains("[Sat]"));
    }

    #[test]
    fn entry_description_hours_only() {
        let e = ActivityCatalogEntry {
            id: "quiet_hours".into(),
            label: "winding down".into(),
            active_days: None,
            active_hours: Some((
                NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            )),
        };
        let d = entry_description(&e);
        assert!(d.contains("[09:00-17:00]"));
        for abbr in WEEKDAY_ABBR {
            assert!(!d.contains(abbr));
        }
    }

    struct FakeEngine {
        entries: Vec<ActivityCatalogEntry>,
        tz: Tz,
    }
    impl StartActivityEngine for FakeEngine {
        fn catalog(&self) -> Vec<ActivityCatalogEntry> {
            self.entries.clone()
        }
        fn display_tz(&self) -> Tz {
            self.tz
        }
        fn is_active(&self) -> bool {
            false
        }
        fn defer_start(&self, _type_id: &str, _note: Option<&str>) -> Value {
            json!({"ack": "ok"})
        }
    }

    fn activity_desc(entries: Vec<ActivityCatalogEntry>, tz: &str) -> String {
        let engine = Arc::new(FakeEngine {
            entries,
            tz: tz.parse().unwrap(),
        });
        build_start_activity_tool(engine, "").parameters["properties"]["activity"]["description"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    fn scheduled_entry() -> ActivityCatalogEntry {
        ActivityCatalogEntry {
            id: "weekday_rounds".into(),
            label: "weekday rounds".into(),
            active_days: None,
            active_hours: Some((
                NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            )),
        }
    }

    fn unscheduled_entry() -> ActivityCatalogEntry {
        ActivityCatalogEntry {
            id: "creek_walk".into(),
            label: "a creek walk".into(),
            active_days: None,
            active_hours: None,
        }
    }

    #[test]
    fn scheduled_hours_declare_the_display_tz_once() {
        // One zone statement for the whole list — not a suffix per entry.
        let desc = activity_desc(
            vec![scheduled_entry(), scheduled_entry()],
            "America/Los_Angeles",
        );
        assert!(desc.contains("America/Los_Angeles"), "{desc}");
        assert_eq!(desc.matches("America/Los_Angeles").count(), 1, "{desc}");
        assert!(desc.contains("09:00-17:00"), "{desc}");
    }

    #[test]
    fn unscheduled_catalog_carries_no_tz_clause() {
        let desc = activity_desc(vec![unscheduled_entry()], "America/Los_Angeles");
        assert!(!desc.contains("America/Los_Angeles"), "{desc}");
        assert_eq!(desc, "What to go do: 'creek_walk' = a creek walk.");
    }

    #[test]
    fn entry_description_unscheduled_is_clean() {
        let e = ActivityCatalogEntry {
            id: "creek_walk".into(),
            label: "a creek walk".into(),
            active_days: None,
            active_hours: None,
        };
        let d = entry_description(&e);
        assert_eq!(d, "'creek_walk' = a creek walk");
        assert!(!d.contains('['));
    }
}
