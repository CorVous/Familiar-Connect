//! Registry-builder helpers (subsystem 08).
//!
//! Extracted so tests can compose registries without the full wiring. The focus
//! manager and image-tools/describe knobs are *gates*: the tools reach the live
//! focus manager / store / scheduler through the per-call [`ToolContext`], so the
//! builder only needs to know *whether* to include each tool, so a `bool`
//! gate is enough. The activity engine is passed by value because
//! its catalog shapes the `start_activity` schema at build time.

use std::sync::Arc;

use crate::tools::alarm::{build_alarm_tool, build_cancel_alarm_tool};
use crate::tools::image_policy::ImageUrlPolicy;
use crate::tools::registry::ToolRegistry;
use crate::tools::scheduler::AlarmScheduler;
use crate::tools::start_activity::StartActivityEngine;

// Re-exported so the shipped tool builders are reachable from this module
// path, as tests / wiring expect.
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
/// `image_tools`, gated by `image_url_policy`), `shift_focus` + `read_channel`
/// (when a focus manager is present), and `start_activity` (when an activity
/// engine is provided, with its config-sourced description).
#[must_use]
pub fn build_text_registry(
    scheduler: &AlarmScheduler,
    image_tools: bool,
    describe_constraints: &str,
    image_url_policy: &ImageUrlPolicy,
    with_focus_manager: bool,
    activity_engine: Option<Arc<dyn StartActivityEngine>>,
    start_activity_description: &str,
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
                Arc::new(crate::tools::image_policy::UrlGuard::production(
                    image_url_policy.clone(),
                )),
            ))
            .expect("unique tool");
        #[cfg(not(feature = "images"))]
        let _ = (describe_constraints, image_url_policy);
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
            .register(build_start_activity_tool(
                engine,
                start_activity_description,
            ))
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

    fn policy() -> ImageUrlPolicy {
        ImageUrlPolicy::default()
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
            names(&build_text_registry(
                &scheduler(),
                false,
                "",
                &policy(),
                false,
                None,
                ""
            ))
            .contains("silent")
        );
    }

    #[test]
    fn voice_registry_shift_focus_gated_on_fm() {
        assert!(!names(&build_voice_registry(&scheduler(), false)).contains("shift_focus"));
        assert!(names(&build_voice_registry(&scheduler(), true)).contains("shift_focus"));
    }

    #[test]
    fn text_registry_shift_focus_and_read_channel_gated_on_fm() {
        let with = names(&build_text_registry(
            &scheduler(),
            false,
            "",
            &policy(),
            true,
            None,
            "",
        ));
        assert!(with.contains("shift_focus"));
        assert!(with.contains("read_channel"));
        let without = names(&build_text_registry(
            &scheduler(),
            false,
            "",
            &policy(),
            false,
            None,
            "",
        ));
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
            &policy(),
            false,
            Some(Arc::new(FakeEngine)),
            "",
        ));
        assert!(with.contains("start_activity"));
        let without = names(&build_text_registry(
            &scheduler(),
            false,
            "",
            &policy(),
            false,
            None,
            "",
        ));
        assert!(!without.contains("start_activity"));
    }

    /// #151: the roleplay policy is config text threaded through the builder,
    /// not a constant in `start_activity.rs`.
    #[test]
    fn start_activity_description_threads_from_the_builder() {
        let reg = build_text_registry(
            &scheduler(),
            false,
            "",
            &policy(),
            false,
            Some(Arc::new(FakeEngine)),
            "POLICY_MARKER",
        );
        let tool = reg
            .tools()
            .find(|t| t.name == "start_activity")
            .expect("start_activity registered");
        assert_eq!(tool.description, "POLICY_MARKER");
    }

    #[cfg(feature = "images")]
    #[test]
    fn text_registry_view_image_gated_on_image_tools() {
        let with = names(&build_text_registry(
            &scheduler(),
            true,
            "be brief",
            &policy(),
            false,
            None,
            "",
        ));
        assert!(with.contains("view_image"));
        let without = names(&build_text_registry(
            &scheduler(),
            false,
            "",
            &policy(),
            false,
            None,
            "",
        ));
        assert!(!without.contains("view_image"));
    }
}
