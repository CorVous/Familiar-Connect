//! In-process tool registry + the per-call [`ToolContext`] (subsystem 08;
//! Python `tools/registry.py`).
//!
//! A [`ToolRegistry`] is a name-indexed, insertion-ordered bag of [`Tool`]s.
//! Each tool carries a JSON-Schema `parameters` object and an async
//! [`ToolHandler`]. Handlers reach the rest of the system through a
//! [`ToolContext`] — no globals.
//!
//! Two narrow seam traits ([`FocusControl`], [`ChannelReadStore`]) replace the
//! Python duck-typed `FocusManager` / `AsyncHistoryStore` references so tests can
//! inject scripted doubles (DESIGN §4.8); the production `FocusManager` and
//! `AsyncHistoryStore` implement them.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::bus::protocols::EventBus;
use crate::focus::FocusManager;
use crate::history::StoreError;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::HistoryTurn;
use crate::llm::{Content, LlmClient};
use crate::tools::scheduler::AlarmScheduler;

// ---------------------------------------------------------------------------
// ImageResult + ToolOutput
// ---------------------------------------------------------------------------

/// Tool result carrying a JPEG (base64) + text description.
///
/// Always carries both; the agentic loop serialises per the slot's `multimodal`
/// flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageResult {
    /// Text description (persisted in history at high quality).
    pub description: String,
    /// Base64-encoded JPEG payload.
    pub jpeg_base64: String,
    /// MIME type (default `image/jpeg`).
    pub media_type: String,
}

impl ImageResult {
    /// Construct with `media_type` defaulting to `"image/jpeg"`.
    #[must_use]
    pub fn new(description: impl Into<String>, jpeg_base64: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            jpeg_base64: jpeg_base64.into(),
            media_type: "image/jpeg".to_owned(),
        }
    }
}

/// What a [`ToolHandler`] produces: either a JSON/text string, or an
/// [`ImageResult`] the agentic loop serialises per the client's `multimodal`
/// flag. Replaces the Python `str | ImageResult` union.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolOutput {
    /// Plain string (usually a JSON object the model reads back).
    Text(String),
    /// An image result (description + JPEG); the loop serialises it.
    Image(ImageResult),
}

// ---------------------------------------------------------------------------
// Seam traits (DESIGN §4.8): FocusControl + ChannelReadStore
// ---------------------------------------------------------------------------

/// The narrow slice of `FocusManager` the attentional tools touch.
///
/// A tool holds an `Arc<dyn FocusControl>`; the production [`FocusManager`]
/// implements it, and tests inject a scripted double (mirrors the Python
/// `MagicMock` focus manager).
#[async_trait]
pub trait FocusControl: Send + Sync {
    /// Is `channel_id` a known text/voice subscription?
    fn is_subscribed(&self, channel_id: i64) -> bool;
    /// Sorted, deduped subscribed channel ids.
    fn subscribed_channels(&self) -> Vec<i64>;
    /// `#name(id)` / `#id` label for a channel.
    fn channel_label(&self, channel_id: i64) -> String;
    /// Current focus channel for `modality` (`"text"` → text pointer).
    fn get_focus(&self, modality: &str) -> Option<i64>;
    /// The catch-up window (also the `shift_focus` preview size).
    fn catch_up_limit(&self) -> usize;
    /// Apply a focus shift immediately (at tool-call time).
    async fn shift_now(&self, channel_id: i64);
}

#[async_trait]
impl FocusControl for FocusManager {
    fn is_subscribed(&self, channel_id: i64) -> bool {
        Self::is_subscribed(self, channel_id)
    }
    fn subscribed_channels(&self) -> Vec<i64> {
        Self::subscribed_channels(self)
    }
    fn channel_label(&self, channel_id: i64) -> String {
        Self::channel_label(self, Some(channel_id))
    }
    fn get_focus(&self, modality: &str) -> Option<i64> {
        Self::get_focus(self, modality)
    }
    fn catch_up_limit(&self) -> usize {
        Self::catch_up_limit(self)
    }
    async fn shift_now(&self, channel_id: i64) {
        Self::shift_now(self, channel_id).await;
    }
}

/// The narrow read slice of the history store the focus tools page over.
///
/// Both methods deliberately omit the `mode` filter (the Python tool calls pass
/// none); [`AsyncHistoryStore`] implements it, tests inject a recorder.
#[async_trait]
pub trait ChannelReadStore: Send + Sync {
    /// Recent turns in a channel, newest-window, optionally paged with
    /// `before_id`.
    async fn recent(
        &self,
        familiar_id: &str,
        channel_id: i64,
        limit: i64,
        before_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError>;

    /// `before`/`after` turns around a given `turn_id`.
    async fn turns_around(
        &self,
        familiar_id: &str,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError>;
}

#[async_trait]
impl ChannelReadStore for AsyncHistoryStore {
    async fn recent(
        &self,
        familiar_id: &str,
        channel_id: i64,
        limit: i64,
        before_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        Self::recent(
            self,
            familiar_id.to_owned(),
            channel_id,
            limit,
            None,
            before_id,
        )
        .await
    }

    async fn turns_around(
        &self,
        familiar_id: &str,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        Self::turns_around(
            self,
            familiar_id.to_owned(),
            channel_id,
            turn_id,
            before,
            after,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// ToolContext
// ---------------------------------------------------------------------------

/// Per-call context handed to tool handlers.
///
/// `history`/`bus` are `Option` so the unit suites can build a context without
/// real subsystems (mirrors the Python `cast("...", None)` doubles); no shipped
/// handler reads them. `scheduler`/`focus_manager`/`store`/`description_llm`
/// default to `None`; `images` defaults empty.
#[derive(Clone)]
pub struct ToolContext {
    /// The familiar this turn belongs to.
    pub familiar_id: String,
    /// Channel the tool was called from (the alarm wake routes back here).
    pub channel_id: i64,
    /// `"text"` | `"voice"`.
    pub channel_kind: String,
    /// The originating turn id.
    pub turn_id: String,
    /// History store handle (unused by shipped handlers; kept for parity).
    pub history: Option<Arc<AsyncHistoryStore>>,
    /// Event bus handle (unused by shipped handlers; kept for parity).
    pub bus: Option<Arc<dyn EventBus>>,
    /// Live alarm scheduler (`set_alarm` / `cancel_alarm` reach it here).
    pub scheduler: Option<Arc<AlarmScheduler>>,
    /// `img_id` → URL placeholder map injected per-turn (for `view_image`).
    pub images: HashMap<String, String>,
    /// Vision model client (`view_image` description leg).
    pub description_llm: Option<Arc<dyn LlmClient>>,
    /// Attentional focus controller (`shift_focus` / `read_channel`).
    pub focus_manager: Option<Arc<dyn FocusControl>>,
    /// Explicit store ref for the read-only focus tools.
    pub store: Option<Arc<dyn ChannelReadStore>>,
}

impl ToolContext {
    /// A context with the four required fields; everything else default.
    #[must_use]
    pub fn new(
        familiar_id: impl Into<String>,
        channel_id: i64,
        channel_kind: impl Into<String>,
        turn_id: impl Into<String>,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            channel_id,
            channel_kind: channel_kind.into(),
            turn_id: turn_id.into(),
            history: None,
            bus: None,
            scheduler: None,
            images: HashMap::new(),
            description_llm: None,
            focus_manager: None,
            store: None,
        }
    }

    /// Builder: attach the history store.
    #[must_use]
    pub fn with_history(mut self, history: Arc<AsyncHistoryStore>) -> Self {
        self.history = Some(history);
        self
    }

    /// Builder: attach the event bus.
    #[must_use]
    pub fn with_bus(mut self, bus: Arc<dyn EventBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Builder: attach the alarm scheduler.
    #[must_use]
    pub fn with_scheduler(mut self, scheduler: Arc<AlarmScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Builder: attach the per-turn `img_id` → URL map.
    #[must_use]
    pub fn with_images(mut self, images: HashMap<String, String>) -> Self {
        self.images = images;
        self
    }

    /// Builder: attach the vision description client.
    #[must_use]
    pub fn with_description_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.description_llm = Some(llm);
        self
    }

    /// Builder: attach the focus controller.
    #[must_use]
    pub fn with_focus_manager(mut self, fm: Arc<dyn FocusControl>) -> Self {
        self.focus_manager = Some(fm);
        self
    }

    /// Builder: attach the read-only store.
    #[must_use]
    pub fn with_store(mut self, store: Arc<dyn ChannelReadStore>) -> Self {
        self.store = Some(store);
        self
    }
}

// ---------------------------------------------------------------------------
// Tool handler + Tool + ToolRegistry
// ---------------------------------------------------------------------------

/// One tool's async behaviour.
///
/// Genuine faults surface as `Err` (the loop wraps them as `{"error": ...}` tool
/// results); domain errors are returned as
/// `Ok(ToolOutput::Text(json!({"error": ...})))` by the handler itself.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Run the tool with decoded `args` and the per-call `ctx`.
    async fn call(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput>;
}

/// Adapts a plain async closure `Fn(Value, ToolContext) -> Future` into a
/// [`ToolHandler`]. `ToolContext` is cloned per call (its fields are all
/// `Arc`/`Clone`), so the closure's future is `'static`.
pub struct FnHandler<F>(pub F);

#[async_trait]
impl<F, Fut> ToolHandler for FnHandler<F>
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync,
    Fut: Future<Output = anyhow::Result<ToolOutput>> + Send,
{
    async fn call(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput> {
        (self.0)(args, ctx.clone()).await
    }
}

/// One callable tool exposed to the model.
#[derive(Clone)]
pub struct Tool {
    /// Function name (the model calls it by this).
    pub name: String,
    /// Human/model-facing description.
    pub description: String,
    /// JSON-Schema `parameters` object.
    pub parameters: Value,
    /// The async handler.
    pub handler: Arc<dyn ToolHandler>,
    /// Per-call wall-clock cap (seconds); default `10.0`.
    pub timeout_s: f64,
}

impl Tool {
    /// Construct a tool with the default `10.0`s timeout.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: Arc<dyn ToolHandler>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler,
            timeout_s: 10.0,
        }
    }

    /// Builder: override the timeout (seconds).
    #[must_use]
    pub const fn with_timeout_s(mut self, timeout_s: f64) -> Self {
        self.timeout_s = timeout_s;
        self
    }
}

/// Registration error: a duplicate tool name (a programming error, like the
/// Python `ValueError`).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolError {
    /// A tool with this name is already registered.
    #[error("tool already registered: {0}")]
    Duplicate(String),
}

/// Name-indexed, insertion-ordered bag of [`Tool`].
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
    names: HashSet<String>,
}

impl ToolRegistry {
    /// A fresh, empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `tool`; a duplicate name is a [`ToolError::Duplicate`].
    pub fn register(&mut self, tool: Tool) -> Result<(), ToolError> {
        if self.names.contains(&tool.name) {
            return Err(ToolError::Duplicate(tool.name));
        }
        self.names.insert(tool.name.clone());
        self.tools.push(tool);
        Ok(())
    }

    /// The tool named `name`, or `None` (the loop treats `None` as
    /// "unknown tool").
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Iterate the tools in insertion order.
    pub fn tools(&self) -> impl Iterator<Item = &Tool> {
        self.tools.iter()
    }

    /// Whether the registry holds no tools.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Serialise to the OpenAI `tools` array shape; empty registry → `[]`.
    #[must_use]
    pub fn as_openai_tools(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })
            })
            .collect()
    }
}

/// Serialise an [`ImageResult`] for a tool message per the client's `multimodal`
/// capability.
///
/// `multimodal = false` → the description string; `multimodal = true` → a
/// `[text, image_url]` content-block list.
#[must_use]
pub fn serialize_image_result(res: &ImageResult, multimodal: bool) -> Content {
    if !multimodal {
        return Content::Text(res.description.clone());
    }
    Content::Blocks(vec![
        json!({"type": "text", "text": res.description}),
        json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{};base64,{}", res.media_type, res.jpeg_base64),
            },
        }),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_handler() -> Arc<dyn ToolHandler> {
        Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
            Ok(ToolOutput::Text("ok".to_owned()))
        }))
    }

    fn make_tool(name: &str) -> Tool {
        Tool::new(
            name,
            format!("{name} tool"),
            json!({"type": "object", "properties": {}}),
            noop_handler(),
        )
    }

    #[test]
    fn register_and_get_round_trip() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("echo")).unwrap();
        assert!(registry.get("echo").is_some());
        assert_eq!(registry.get("echo").unwrap().name, "echo");
    }

    #[test]
    fn get_unknown_name_is_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nope").is_none());
    }

    #[test]
    fn duplicate_name_register_is_error() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("echo")).unwrap();
        let err = registry.register(make_tool("echo")).unwrap_err();
        assert_eq!(err, ToolError::Duplicate("echo".to_owned()));
        assert!(err.to_string().contains("echo"));
    }

    #[test]
    fn as_openai_tools_shape_matches_function_calling_schema() {
        let mut registry = ToolRegistry::new();
        let params = json!({
            "type": "object",
            "properties": {
                "when": {"type": "string", "format": "date-time"},
                "reason": {"type": "string", "maxLength": 200},
            },
            "required": ["reason"],
        });
        registry
            .register(Tool::new(
                "set_alarm",
                "Schedule a wake.",
                params.clone(),
                noop_handler(),
            ))
            .unwrap();
        let expected = json!([
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "Schedule a wake.",
                    "parameters": params,
                },
            }
        ]);
        assert_eq!(Value::Array(registry.as_openai_tools()), expected);
    }

    #[test]
    fn as_openai_tools_empty_registry_returns_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.as_openai_tools().is_empty());
    }

    #[test]
    fn tools_iteration_returns_all_registered() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("a")).unwrap();
        registry.register(make_tool("b")).unwrap();
        let mut names: Vec<&str> = registry.tools().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["a", "b"]);
    }

    #[test]
    fn default_timeout_is_set() {
        assert!(make_tool("echo").timeout_s > 0.0);
    }

    #[test]
    fn image_result_carries_both() {
        let result = ImageResult::new("a cat", "abc123");
        assert_eq!(result.description, "a cat");
        assert_eq!(result.jpeg_base64, "abc123");
        assert_eq!(result.media_type, "image/jpeg");
    }

    #[test]
    fn tool_context_images_defaults_empty() {
        let ctx = ToolContext::new("fam-1", 42, "text", "turn-1");
        assert!(ctx.images.is_empty());
    }

    #[test]
    fn tool_context_description_llm_defaults_none() {
        let ctx = ToolContext::new("fam-1", 42, "text", "turn-1");
        assert!(ctx.description_llm.is_none());
    }

    #[test]
    fn tool_context_focus_and_store_default_none() {
        let ctx = ToolContext::new("fam", 1, "text", "t");
        assert!(ctx.focus_manager.is_none());
        assert!(ctx.store.is_none());
    }

    #[test]
    fn serialize_image_result_text_only() {
        let res = ImageResult::new("a cat", "abc123");
        assert_eq!(
            serialize_image_result(&res, false),
            Content::Text("a cat".to_owned())
        );
    }

    #[test]
    fn serialize_image_result_multimodal() {
        let res = ImageResult::new("a cat", "abc123");
        let out = serialize_image_result(&res, true);
        let Content::Blocks(blocks) = out else {
            panic!("expected blocks");
        };
        assert_eq!(blocks[0], json!({"type": "text", "text": "a cat"}));
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(
            blocks[1]["image_url"]["url"],
            "data:image/jpeg;base64,abc123"
        );
    }
}
