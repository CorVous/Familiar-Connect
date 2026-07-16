//! Registry-builder helpers (subsystem 08; Python `tools/builtins.py`).
//!
//! Extracted so tests can compose registries without `run.py`. The focus
//! manager and image-tools/describe knobs are *gates*: the tools reach the live
//! focus manager / store / scheduler through the per-call [`ToolContext`], so the
//! builder only needs to know *whether* to include each tool (the Python passes
//! the `FocusManager` object, but uses it solely as a presence gate — a `bool`
//! is the faithful Rust shape). The activity engine is passed by value because
//! its catalog shapes the `start_activity` schema at build time.

use std::sync::Arc;

use crate::tools::alarm::{build_alarm_tool, build_cancel_alarm_tool};
use crate::tools::registry::ToolRegistry;
use crate::tools::scheduler::AlarmScheduler;
use crate::tools::start_activity::StartActivityEngine;

// Re-exported so the shipped tool builders are reachable at the Python
// `tools.builtins` import path used by tests / wiring.
pub use crate::tools::read_channel::build_read_channel_tool;
pub use crate::tools::shift_focus::build_shift_focus_tool;
pub use crate::tools::silent::build_silent_tool;
pub use crate::tools::start_activity::build_start_activity_tool;

/// Voice-tier registry: `set_alarm` + `cancel_alarm` + `silent`; `shift_focus`
/// when a focus manager is present. `view_image` / `read_channel` /
/// `start_activity` are never in the voice registry.
#[must_use]
pub fn build_voice_registry(scheduler: &AlarmScheduler, with_focus_manager: bool) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(build_alarm_tool(scheduler))
        .expect("unique tool");
    registry
        .register(build_cancel_alarm_tool(scheduler))
        .expect("unique tool");
    registry.register(build_silent_tool()).expect("unique tool");
    if with_focus_manager {
        registry
            .register(build_shift_focus_tool())
            .expect("unique tool");
    }
    registry
}

/// Text-tier registry.
///
/// `set_alarm` + `cancel_alarm` + `silent`; plus `view_image` (when
/// `image_tools`), `shift_focus` + `read_channel` (when a focus manager is
/// present), and `start_activity` (when an activity engine is provided).
#[must_use]
pub fn build_text_registry(
    scheduler: &AlarmScheduler,
    image_tools: bool,
    describe_constraints: &str,
    with_focus_manager: bool,
    activity_engine: Option<Arc<dyn StartActivityEngine>>,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(build_alarm_tool(scheduler))
        .expect("unique tool");
    registry
        .register(build_cancel_alarm_tool(scheduler))
        .expect("unique tool");
    registry.register(build_silent_tool()).expect("unique tool");
    if image_tools {
        #[cfg(feature = "images")]
        registry
            .register(crate::tools::image::build_view_image_tool(
                describe_constraints,
            ))
            .expect("unique tool");
        #[cfg(not(feature = "images"))]
        let _ = describe_constraints;
    }
    if with_focus_manager {
        registry
            .register(build_shift_focus_tool())
            .expect("unique tool");
        registry
            .register(build_read_channel_tool())
            .expect("unique tool");
    }
    // text-only by design — absence while voice-connected is refused by the engine
    if let Some(engine) = activity_engine {
        registry
            .register(build_start_activity_tool(engine))
            .expect("unique tool");
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::in_process::InProcessEventBus;
    use crate::bus::protocols::EventBus;
    use crate::history::async_store::AsyncHistoryStore;
    use crate::history::store::HistoryStore;
    use crate::tools::start_activity::ActivityCatalogEntry;
    use serde_json::{Value, json};
    use std::collections::BTreeSet;

    struct FakeEngine;
    impl StartActivityEngine for FakeEngine {
        fn catalog(&self) -> Vec<ActivityCatalogEntry> {
            vec![ActivityCatalogEntry {
                id: "creek_walk".into(),
                label: "a creek walk".into(),
                active_days: None,
                active_hours: None,
            }]
        }
        fn is_active(&self) -> bool {
            false
        }
        fn defer_start(&self, _type_id: &str, _note: Option<&str>) -> Value {
            json!({"ack": "ok"})
        }
    }

    fn scheduler() -> AlarmScheduler {
        let store = Arc::new(AsyncHistoryStore::new(
            HistoryStore::open(":memory:").unwrap(),
        ));
        let bus: Arc<dyn EventBus> = Arc::new(InProcessEventBus::new());
        AlarmScheduler::new(store, bus, "fam")
    }

    fn names(reg: &ToolRegistry) -> BTreeSet<String> {
        reg.tools().map(|t| t.name.clone()).collect()
    }

    #[test]
    fn voice_registry_includes_silent() {
        assert!(names(&build_voice_registry(&scheduler(), false)).contains("silent"));
    }

    #[test]
    fn text_registry_includes_silent() {
        assert!(
            names(&build_text_registry(&scheduler(), false, "", false, None)).contains("silent")
        );
    }

    #[test]
    fn voice_registry_shift_focus_gated_on_fm() {
        assert!(!names(&build_voice_registry(&scheduler(), false)).contains("shift_focus"));
        assert!(names(&build_voice_registry(&scheduler(), true)).contains("shift_focus"));
    }

    #[test]
    fn text_registry_shift_focus_and_read_channel_gated_on_fm() {
        let with = names(&build_text_registry(&scheduler(), false, "", true, None));
        assert!(with.contains("shift_focus"));
        assert!(with.contains("read_channel"));
        let without = names(&build_text_registry(&scheduler(), false, "", false, None));
        assert!(!without.contains("shift_focus"));
    }

    #[test]
    fn voice_registry_never_has_start_activity_or_read_channel() {
        let n = names(&build_voice_registry(&scheduler(), true));
        assert!(!n.contains("start_activity"));
        assert!(!n.contains("read_channel"));
        assert!(!n.contains("view_image"));
    }

    #[test]
    fn text_registry_start_activity_gated_on_engine() {
        let with = names(&build_text_registry(
            &scheduler(),
            false,
            "",
            false,
            Some(Arc::new(FakeEngine)),
        ));
        assert!(with.contains("start_activity"));
        let without = names(&build_text_registry(&scheduler(), false, "", false, None));
        assert!(!without.contains("start_activity"));
    }

    #[cfg(feature = "images")]
    #[test]
    fn text_registry_view_image_gated_on_image_tools() {
        let with = names(&build_text_registry(
            &scheduler(),
            true,
            "be brief",
            false,
            None,
        ));
        assert!(with.contains("view_image"));
        let without = names(&build_text_registry(&scheduler(), false, "", false, None));
        assert!(!without.contains("view_image"));
    }
}
