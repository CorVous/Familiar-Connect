//! LLM value types + `sanitize_name` (subsystem 08; Python `llm.py`).
//!
//! This module is split across two port layers (DESIGN §3). Implemented here are
//! the **Layer-0 core value types** — [`Message`], [`Content`], [`LlmDelta`], and
//! [`sanitize_name`] — which `identity` (02) and `budget` (05) depend on. The
//! Layer-2 OpenRouter transport half (the client, SSE streaming, the rate-limit
//! semaphore, `create_llm_clients`) is a separate porting task; see the stub
//! marker at the bottom of this file.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

static NAME_DISALLOWED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9_-]").expect("valid name regex"));

/// Sanitize a string for the OpenAI `name` field; `None` when empty after
/// cleanup.
///
/// Replaces every character outside `[a-zA-Z0-9_-]` with `_`, truncates to 64
/// scalars, then strips leading/trailing `_`. An all-punctuation input cleans to
/// the empty string and yields `None` (so callers fall back to a numeric id).
#[must_use]
pub fn sanitize_name(name: &str) -> Option<String> {
    let replaced = NAME_DISALLOWED.replace_all(name, "_");
    // After substitution the string is pure ASCII, so scalar and byte counts
    // agree; take 64 scalars to mirror Python's `[:64]`.
    let truncated: String = replaced.chars().take(64).collect();
    let trimmed = truncated.trim_matches('_');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// A chat message's content: plain text, or a list of content blocks for
/// multimodal / tool-result messages (e.g. vision `image_url` blocks).
///
/// Serialized untagged so `Text` renders as a JSON string and `Blocks` as a JSON
/// array — the OpenAI-compatible `content` field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain-text content.
    Text(String),
    /// Structured content blocks (`[{"type": "text", ...}, ...]`).
    Blocks(Vec<Value>),
}

impl Default for Content {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<Vec<Value>> for Content {
    fn from(blocks: Vec<Value>) -> Self {
        Self::Blocks(blocks)
    }
}

/// Chat message — role + content + optional speaker name and tool fields.
///
/// `to_dict` (via `Serialize`) omits the `None`-valued optional fields, matching
/// the Python `Message.to_dict`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// `"system"` / `"user"` / `"assistant"` / `"tool"`.
    pub role: String,
    /// Text or multimodal/tool-result content blocks.
    pub content: Content,
    /// Speaker name (user turns); omitted from `to_dict` when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// OpenAI `tool_calls` dicts on assistant turns invoking tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    /// The call this message answers, on `role="tool"` turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Construct a message with the given role and content and no optional fields.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<Content>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Builder: set the speaker `name`.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Content as plain text; joins the `"text"` blocks of multimodal lists with
    /// newlines.
    #[must_use]
    pub fn content_str(&self) -> String {
        match &self.content {
            Content::Text(s) => s.clone(),
            Content::Blocks(blocks) => {
                let parts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|block| {
                        if block.get("type").and_then(Value::as_str) == Some("text") {
                            block.get("text").and_then(Value::as_str)
                        } else {
                            None
                        }
                    })
                    .collect();
                parts.join("\n")
            }
        }
    }

    /// Serialize to an OpenAI-compatible message object, omitting `None`
    /// optionals.
    #[must_use]
    pub fn to_dict(&self) -> Value {
        serde_json::to_value(self).expect("Message serializes to JSON")
    }
}

/// One streaming chunk: content text and/or raw tool-call fragments.
///
/// `tool_calls` holds raw streaming fragments (each with an `"index"`); callers
/// accumulate `function.arguments` by index across deltas.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LlmDelta {
    /// Content delta text (`""` when this chunk carries only tool calls).
    pub content: String,
    /// Raw tool-call fragment dicts.
    pub tool_calls: Vec<Value>,
    /// Terminal finish reason (`"stop"`, `"tool_calls"`, …) when present.
    pub finish_reason: Option<String>,
}

// ============================================================================
// Layer-2 client (subsystem 08) — STUB.
//
// The OpenRouter transport half of `llm` lives here in the Python source
// (`LLMClient`, `stream_completion`/`chat`/`chat_stream`, the process-wide
// rate-limit semaphore, `_CallMetrics`, `create_llm_clients`) but is Layer 2 per
// DESIGN §3 and belongs to the subsystem-08 porting task. Only the Layer-0 value
// types above are implemented in this foundation pass. The client is filled in
// later against the `LlmClient` seam trait below.
//
// The trait itself is defined here now (Layer 1) because `structured_request`
// and the 04/07 workers type against it (DESIGN §4.8: mocks touch only these
// five methods). The concrete OpenRouter client that implements it — the SSE
// streaming, the rate-limit semaphore, `create_llm_clients` — remains the
// Layer-2 subsystem-08 task; nothing here constructs a client.
// ============================================================================

use async_trait::async_trait;
use futures::stream::BoxStream;

/// The narrow LLM client seam the rest of the system types against
/// (DESIGN §4.8).
///
/// Only these five methods are ever touched from outside the transport module,
/// so a ~5-line scripted stub satisfies it (the 04/07 worker doubles and the 06
/// responder doubles implement exactly this). The OpenRouter client (Layer 2,
/// subsystem 08) implements the trait; the streaming/chat bodies, the rate-limit
/// semaphore, and `create_llm_clients` are that later task's remit.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Blocking, 429-retrying chat completion.
    ///
    /// Transport / HTTP faults surface as `Err`; callers such as
    /// `request_structured` propagate them unchanged (only shape problems are
    /// retried).
    async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message>;

    /// SSE streaming completion. `tools` is the OpenAI `tools` payload (`None`
    /// when the registry is empty); each item is one [`LlmDelta`].
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>>;

    /// The config slot label (`"fast"` / `"prose"` / `"background"`), or `None`.
    fn slot(&self) -> Option<&str>;

    /// Whether `ImageResult`s serialize as multimodal blocks for this client.
    fn multimodal(&self) -> bool;

    /// Whether tool-calling is enabled for this slot (gates the agentic loop).
    fn tool_calling_enabled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::{Content, LlmDelta, Message, sanitize_name};
    use serde_json::{Value, json};

    // --- sanitize_name -----------------------------------------------------

    #[test]
    fn sanitize_name_replaces_spaces_and_punctuation() {
        assert_eq!(
            sanitize_name("Ada Lovelace!").as_deref(),
            Some("Ada_Lovelace")
        );
    }

    #[test]
    fn sanitize_name_all_punctuation_is_none() {
        // Fullwidth exclamation marks are outside the allow-list; the cleaned
        // string is all underscores, which strip to empty -> None.
        assert_eq!(sanitize_name("！！！"), None);
        assert_eq!(sanitize_name(""), None);
        assert_eq!(sanitize_name("***"), None);
    }

    #[test]
    fn sanitize_name_keeps_bare_id() {
        assert_eq!(sanitize_name("42").as_deref(), Some("42"));
    }

    #[test]
    fn sanitize_name_strips_leading_and_trailing_underscores() {
        assert_eq!(sanitize_name("__hi__").as_deref(), Some("hi"));
        assert_eq!(
            sanitize_name(" middle keep ").as_deref(),
            Some("middle_keep")
        );
    }

    #[test]
    fn sanitize_name_truncates_to_64() {
        let long = "a".repeat(100);
        let out = sanitize_name(&long).unwrap();
        assert_eq!(out.len(), 64);
    }

    // --- Message construction / fields ------------------------------------

    #[test]
    fn user_message_has_name() {
        let msg = Message::new("user", "hello").with_name("Alice");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "hello");
        assert_eq!(msg.name.as_deref(), Some("Alice"));
    }

    #[test]
    fn assistant_message_has_no_name() {
        let msg = Message::new("assistant", "hi back");
        assert_eq!(msg.role, "assistant");
        assert!(msg.name.is_none());
    }

    #[test]
    fn system_message() {
        let msg = Message::new("system", "You are a helpful familiar.");
        assert_eq!(msg.role, "system");
    }

    // --- Message::to_dict --------------------------------------------------

    #[test]
    fn to_dict_includes_name_when_present() {
        let msg = Message::new("user", "hello").with_name("Bob");
        assert_eq!(
            msg.to_dict(),
            json!({"role": "user", "content": "hello", "name": "Bob"})
        );
    }

    #[test]
    fn to_dict_excludes_name_when_none() {
        let msg = Message::new("assistant", "hi");
        let d = msg.to_dict();
        assert_eq!(d, json!({"role": "assistant", "content": "hi"}));
        assert!(d.get("name").is_none());
    }

    #[test]
    fn to_dict_omits_tool_fields_when_none() {
        let d = Message::new("assistant", "hi").to_dict();
        assert!(d.get("tool_calls").is_none());
        assert!(d.get("tool_call_id").is_none());
    }

    #[test]
    fn assistant_message_carries_tool_calls() {
        let tc: Vec<Value> = vec![json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "set_alarm", "arguments": "{\"reason\":\"x\"}"},
        })];
        let msg = Message {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(tc.clone()),
            ..Default::default()
        };
        let d = msg.to_dict();
        assert_eq!(d["tool_calls"], Value::Array(tc));
        assert_eq!(d["role"], "assistant");
    }

    #[test]
    fn tool_role_message_carries_tool_call_id() {
        let msg = Message {
            role: "tool".into(),
            content: r#"{"ok": true}"#.into(),
            tool_call_id: Some("call_1".into()),
            ..Default::default()
        };
        let d = msg.to_dict();
        assert_eq!(d["role"], "tool");
        assert_eq!(d["tool_call_id"], "call_1");
        assert_eq!(d["content"], r#"{"ok": true}"#);
    }

    // --- content_str over multimodal blocks -------------------------------

    #[test]
    fn content_str_joins_text_blocks_and_skips_non_text() {
        let msg = Message::new(
            "user",
            vec![
                json!({"type": "text", "text": "a"}),
                json!({"type": "image_url", "image_url": {"url": "data:..."}}),
                json!({"type": "text", "text": "b"}),
            ],
        );
        assert_eq!(msg.content_str(), "a\nb");
    }

    #[test]
    fn content_str_ignores_non_string_text_and_untyped_blocks() {
        let msg = Message::new(
            "tool",
            vec![
                json!({"type": "text", "text": 42}),
                json!({"text": "no type key"}),
                json!({"type": "text", "text": "kept"}),
            ],
        );
        assert_eq!(msg.content_str(), "kept");
    }

    #[test]
    fn list_content_serializes_as_array() {
        let msg = Message::new("tool", vec![json!({"type": "text", "text": "desc"})]);
        assert_eq!(
            msg.to_dict(),
            json!({"role": "tool", "content": [{"type": "text", "text": "desc"}]})
        );
    }

    // --- LlmDelta shape ----------------------------------------------------

    #[test]
    fn llm_delta_defaults_are_empty() {
        let d = LlmDelta::default();
        assert_eq!(d.content, "");
        assert!(d.tool_calls.is_empty());
        assert!(d.finish_reason.is_none());
    }

    #[test]
    fn llm_delta_carries_fields() {
        let d = LlmDelta {
            content: "Hel".into(),
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        assert_eq!(d.content, "Hel");
        assert_eq!(d.finish_reason.as_deref(), Some("stop"));
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn content_default_is_empty_text() {
        assert_eq!(Content::default(), Content::Text(String::new()));
    }
}
