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

// ============================================================================
// Layer-2 client (subsystem 08) — OpenRouter transport over reqwest + SSE.
//
// Feature-gated on `net` (default). The value types + `LlmClient` trait above
// stay ungated because `identity`/`budget`/`structured_request` depend on them.
// The concrete OpenRouter client, its SSE streaming, the INJECTED rate-limit
// semaphore (DESIGN D13 — no module global), and `create_llm_clients` live here.
// ============================================================================

#[cfg(feature = "net")]
mod client {
    use super::{Content, LlmClient, LlmDelta, Message};
    use crate::config::{CharacterConfig, LLM_SLOT_NAMES, LLMSlotConfig};
    use crate::diagnostics::collector::get_span_collector;
    use crate::log_style as ls;
    use crate::support;
    use anyhow::{Result, anyhow};
    use async_trait::async_trait;
    use eventsource_stream::Eventsource;
    use futures::stream::BoxStream;
    use futures::{Stream, StreamExt};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::time::{Duration, Instant};
    use tokio::sync::Semaphore;

    /// OpenRouter chat-completions base URL.
    pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

    // Retry / concurrency / transport constants (spec 08 § Config knobs).
    const MAX_RETRIES: u32 = 4;
    const BASE_DELAY_S: f64 = 1.0;
    const MAX_DELAY_S: f64 = 30.0;
    const DEFAULT_MAX_CONCURRENT: usize = 4;
    const HTTP_TIMEOUT_S: u64 = 120;
    const HTTP_ERROR_BODY_LIMIT: usize = 600;

    // -- free helpers -------------------------------------------------------

    /// Sum of the plain-string content lengths (Unicode scalars) of `messages`;
    /// list/multimodal content counts 0 (spec 08 §21).
    fn input_chars(messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| match &m.content {
                Content::Text(s) => s.chars().count(),
                Content::Blocks(_) => 0,
            })
            .sum()
    }

    /// 429 backoff delay for a zero-indexed `attempt` (spec 08 §4).
    ///
    /// `min(1.0 * 2**attempt, 30.0)`; a numeric `Retry-After` overrides with
    /// `min(value, 30.0)`; an unparseable header falls back to exponential.
    fn backoff_delay(attempt: u32, retry_after: Option<&str>) -> Duration {
        let exponential = (BASE_DELAY_S * 2.0_f64.powi(i32::try_from(attempt).unwrap_or(i32::MAX)))
            .min(MAX_DELAY_S);
        let secs = match retry_after.map(str::trim).map(str::parse::<f64>) {
            Some(Ok(v)) => v.min(MAX_DELAY_S),
            _ => exponential,
        };
        Duration::from_secs_f64(secs.max(0.0))
    }

    /// Set an Anthropic prompt-caching breakpoint on the FIRST system message
    /// (spec 08 §10). Mutates in place; a no-op when no system message exists.
    fn mark_system_cache_breakpoint(messages: &mut [Value]) {
        let Some(system) = messages
            .iter_mut()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        else {
            return;
        };
        let content = system.get("content").cloned().unwrap_or(Value::Null);
        let mut blocks: Vec<Value> = match content {
            Value::Array(a) => a,
            other => vec![json!({ "type": "text", "text": other })],
        };
        if let Some(last) = blocks.last_mut().and_then(Value::as_object_mut) {
            last.insert("cache_control".to_string(), json!({ "type": "ephemeral" }));
        }
        if let Some(obj) = system.as_object_mut() {
            obj.insert("content".to_string(), Value::Array(blocks));
        }
    }

    /// Render an SSE code field like Python `str(code)` (spec 08 §13).
    fn code_str(code: Option<&Value>) -> String {
        match code {
            None | Some(Value::Null) => "None".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        }
    }

    /// Decode one SSE `data` payload into a JSON object. `None` for `[DONE]`,
    /// blanks, non-JSON, non-object, or a top-level `error` frame (which is
    /// logged at WARNING and skipped — spec 08 §13).
    fn parse_sse_json(data: &str) -> Option<Value> {
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return None;
        }
        let obj: Value = serde_json::from_str(data).ok()?;
        if !obj.is_object() {
            return None;
        }
        if let Some(err) = obj.get("error").filter(|e| e.is_object()) {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .unwrap_or("unknown");
            let code = code_str(err.get("code"));
            let line = format!(
                "{} {} {}",
                ls::tag("LLM", ls::R),
                ls::kv_styled("sse_error", msg, ls::W, ls::R),
                ls::kv_styled("code", &code, ls::W, ls::LW),
            );
            tracing::warn!(target: "familiar_connect.llm", "{line}");
            return None;
        }
        Some(obj)
    }

    /// Assistant content deltas across every choice of a parsed chunk.
    fn content_deltas(event: &Value) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(choices) = event.get("choices").and_then(Value::as_array) {
            for choice in choices {
                if let Some(delta) = choice
                    .get("delta")
                    .and_then(|d| d.get("content"))
                    .and_then(Value::as_str)
                {
                    if !delta.is_empty() {
                        out.push(delta.to_string());
                    }
                }
            }
        }
        out
    }

    /// Raw tool-call fragment dicts across every choice of a parsed chunk.
    fn tool_call_deltas(event: &Value) -> Vec<Value> {
        let mut out = Vec::new();
        if let Some(choices) = event.get("choices").and_then(Value::as_array) {
            for choice in choices {
                if let Some(tcs) = choice
                    .get("delta")
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(Value::as_array)
                {
                    out.extend(tcs.iter().filter(|tc| tc.is_object()).cloned());
                }
            }
        }
        out
    }

    /// `finish_reason` from the first choice carrying a string.
    fn finish_reason(event: &Value) -> Option<String> {
        let choices = event.get("choices").and_then(Value::as_array)?;
        for choice in choices {
            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                return Some(fr.to_string());
            }
        }
        None
    }

    // -- per-call metrics ---------------------------------------------------

    /// Per-call timing + token signals for one OpenRouter request
    /// (Python `_CallMetrics`; spec 08 §19–21).
    struct CallMetrics {
        slot: Option<String>,
        model: String,
        input_chars: usize,
        t_start: Instant,
        t_first_byte: Option<Instant>,
        t_first_delta: Option<Instant>,
        t_end: Option<Instant>,
        status: &'static str,
        provider: Option<String>,
        in_tokens: Option<i64>,
        out_tokens: Option<i64>,
        cached_tokens: Option<i64>,
    }

    /// `max(0, round(delta * 1000))` — half-to-even (DESIGN §4.3), clamped ≥ 0.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn clamp_ms(from: Instant, to: Instant) -> i64 {
        let ms =
            support::round::half_even(to.saturating_duration_since(from).as_secs_f64() * 1000.0);
        if ms < 0.0 { 0 } else { ms as i64 }
    }

    impl CallMetrics {
        fn new(slot: Option<String>, model: String, input_chars: usize) -> Self {
            Self {
                slot,
                model,
                input_chars,
                t_start: Instant::now(),
                t_first_byte: None,
                t_first_delta: None,
                t_end: None,
                status: "ok",
                provider: None,
                in_tokens: None,
                out_tokens: None,
                cached_tokens: None,
            }
        }

        /// Pull provider + usage off any chunk carrying them (last wins).
        fn absorb(&mut self, event: &Value) {
            if let Some(p) = event.get("provider").and_then(Value::as_str) {
                self.provider = Some(p.to_string());
            }
            if let Some(usage) = event.get("usage").filter(|u| u.is_object()) {
                if let Some(pt) = usage.get("prompt_tokens").and_then(Value::as_i64) {
                    self.in_tokens = Some(pt);
                }
                if let Some(ct) = usage.get("completion_tokens").and_then(Value::as_i64) {
                    self.out_tokens = Some(ct);
                }
                if let Some(cached) = usage
                    .get("prompt_tokens_details")
                    .filter(|d| d.is_object())
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(Value::as_i64)
                {
                    self.cached_tokens = Some(cached);
                }
            }
        }

        /// Records per-phase spans into the collector and one structured
        /// `[LLM call]` INFO line (spec 01 §31, spec 08 §19–21). Both are wire
        /// contracts. Collector failures are suppressed (poison-safe).
        #[allow(clippy::similar_names)] // ttfb_ms / ttft_ms are the wire keys
        fn emit(&self) {
            let suffix = self
                .slot
                .as_deref()
                .map(|s| format!(".{s}"))
                .unwrap_or_default();
            let collector = get_span_collector();
            let mut ttfb_ms = None;
            let mut ttft_ms = None;
            let mut total_ms = None;
            if let Some(tfb) = self.t_first_byte {
                let ms = clamp_ms(self.t_start, tfb);
                ttfb_ms = Some(ms);
                collector.record(&format!("llm.ttfb{suffix}"), ms, "ok");
            }
            if let Some(tfd) = self.t_first_delta {
                let ms = clamp_ms(self.t_start, tfd);
                ttft_ms = Some(ms);
                collector.record(&format!("llm.ttft{suffix}"), ms, "ok");
            }
            if let Some(te) = self.t_end {
                let ms = clamp_ms(self.t_start, te);
                total_ms = Some(ms);
                collector.record(&format!("llm.total{suffix}"), ms, "ok");
            }

            let mut parts = vec![
                ls::tag("LLM call", ls::LM),
                ls::kv_styled("slot", self.slot.as_deref().unwrap_or("-"), ls::W, ls::LC),
                ls::kv_styled("model", &self.model, ls::W, ls::LW),
                ls::kv_styled(
                    "status",
                    self.status,
                    ls::W,
                    if self.status == "ok" { ls::LG } else { ls::R },
                ),
                ls::kv_styled("chars", &self.input_chars.to_string(), ls::W, ls::LC),
            ];
            if let Some(v) = ttfb_ms {
                parts.push(ls::kv_styled("ttfb_ms", &v.to_string(), ls::W, ls::LC));
            }
            if let Some(v) = ttft_ms {
                parts.push(ls::kv_styled("ttft_ms", &v.to_string(), ls::W, ls::LC));
            }
            if let Some(v) = total_ms {
                parts.push(ls::kv_styled("total_ms", &v.to_string(), ls::W, ls::LC));
            }
            if let Some(p) = &self.provider {
                parts.push(ls::kv_styled("provider", p, ls::W, ls::LM));
            }
            if let Some(v) = self.in_tokens {
                parts.push(ls::kv_styled("in_tokens", &v.to_string(), ls::W, ls::LW));
            }
            if let Some(v) = self.out_tokens {
                parts.push(ls::kv_styled("out_tokens", &v.to_string(), ls::W, ls::LW));
            }
            if let Some(v) = self.cached_tokens {
                parts.push(ls::kv_styled("cached", &v.to_string(), ls::W, ls::LW));
            }
            tracing::info!(target: "familiar_connect.llm", "{}", parts.join(" "));
        }
    }

    // -- streaming delta stream --------------------------------------------

    type InnerEvents = Pin<
        Box<
            dyn Stream<
                    Item = std::result::Result<
                        eventsource_stream::Event,
                        eventsource_stream::EventStreamError<reqwest::Error>,
                    >,
                > + Send,
        >,
    >;

    /// `Stream<Item = Result<LlmDelta>>` over the SSE body. The semaphore permit
    /// was already released at header-check time; this struct's job is to parse
    /// deltas and emit call metrics EXACTLY ONCE — on clean end (`ok`), on a
    /// transport error (`error`), or, via `Drop`, on consumer abandonment
    /// (`cancelled`, the barge-in path; spec 08 §16).
    struct SseDeltaStream {
        events: InnerEvents,
        metrics: CallMetrics,
        done: bool,
        emitted: bool,
    }

    impl SseDeltaStream {
        fn emit_with(&mut self, status: &'static str) {
            if self.emitted {
                return;
            }
            self.emitted = true;
            self.done = true;
            self.metrics.status = status;
            self.metrics.t_end = Some(Instant::now());
            self.metrics.emit();
        }
    }

    impl Stream for SseDeltaStream {
        type Item = Result<LlmDelta>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            if this.done {
                return Poll::Ready(None);
            }
            loop {
                match this.events.as_mut().poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => {
                        this.emit_with("ok");
                        return Poll::Ready(None);
                    }
                    Poll::Ready(Some(Err(e))) => {
                        this.emit_with("error");
                        return Poll::Ready(Some(Err(anyhow!("stream transport error: {e}"))));
                    }
                    Poll::Ready(Some(Ok(event))) => {
                        let Some(value) = parse_sse_json(&event.data) else {
                            continue;
                        };
                        this.metrics.absorb(&value);
                        let content_parts = content_deltas(&value);
                        let tool_call_parts = tool_call_deltas(&value);
                        let finish = finish_reason(&value);
                        if content_parts.is_empty()
                            && tool_call_parts.is_empty()
                            && finish.is_none()
                        {
                            continue;
                        }
                        let content = content_parts.concat();
                        if !content.is_empty() && this.metrics.t_first_delta.is_none() {
                            this.metrics.t_first_delta = Some(Instant::now());
                        }
                        return Poll::Ready(Some(Ok(LlmDelta {
                            content,
                            tool_calls: tool_call_parts,
                            finish_reason: finish,
                        })));
                    }
                }
            }
        }
    }

    impl Drop for SseDeltaStream {
        fn drop(&mut self) {
            // Abandoned before a terminal event → cancelled (barge-in). A
            // clean/errored end already emitted and set `emitted`.
            self.emit_with("cancelled");
        }
    }

    // -- the client ---------------------------------------------------------

    /// OpenRouter chat-completions client for one call-site slot.
    ///
    /// Blocking `chat` (429-retrying), SSE-streaming `stream_completion` /
    /// `chat_stream`. The process-wide rate-limit `Arc<Semaphore>` is INJECTED
    /// (DESIGN D13); `chat` holds a permit only across the POST, streaming drops
    /// it the instant response headers pass the status check (spec 08 §2–3).
    #[allow(
        clippy::struct_excessive_bools,
        reason = "mirrors the Python client's independent boolean knobs 1:1"
    )]
    pub struct OpenRouterClient {
        api_key: String,
        model: String,
        base_url: String,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<i64>,
        presence_penalty: Option<f64>,
        slot: Option<String>,
        provider_order: Option<Vec<String>>,
        provider_allow_fallbacks: bool,
        reasoning: Option<String>,
        reasoning_max_tokens: Option<i64>,
        tool_calling: bool,
        image_tools: bool,
        multimodal: bool,
        no_stream: bool,
        think_prepend: bool,
        http: reqwest::Client,
        semaphore: Arc<Semaphore>,
    }

    /// Fluent builder for [`OpenRouterClient`]. The semaphore defaults to a
    /// fresh per-client `Semaphore(4)` unless injected; `create_llm_clients`
    /// injects one shared handle across all slots.
    #[allow(
        clippy::struct_excessive_bools,
        reason = "mirrors the Python client's independent boolean knobs 1:1"
    )]
    pub struct OpenRouterClientBuilder {
        api_key: String,
        model: String,
        base_url: String,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<i64>,
        presence_penalty: Option<f64>,
        slot: Option<String>,
        provider_order: Option<Vec<String>>,
        provider_allow_fallbacks: bool,
        reasoning: Option<String>,
        reasoning_max_tokens: Option<i64>,
        tool_calling: bool,
        image_tools: bool,
        multimodal: bool,
        no_stream: bool,
        think_prepend: bool,
        semaphore: Option<Arc<Semaphore>>,
        http: Option<reqwest::Client>,
    }

    #[allow(
        clippy::return_self_not_must_use,
        clippy::missing_const_for_fn,
        clippy::must_use_candidate
    )]
    impl OpenRouterClientBuilder {
        fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: model.into(),
                base_url: OPENROUTER_BASE_URL.to_string(),
                temperature: None,
                top_p: None,
                top_k: None,
                presence_penalty: None,
                slot: None,
                provider_order: None,
                provider_allow_fallbacks: true,
                reasoning: None,
                reasoning_max_tokens: None,
                tool_calling: false,
                image_tools: false,
                multimodal: false,
                no_stream: false,
                think_prepend: false,
                semaphore: None,
                http: None,
            }
        }

        /// Override the base URL (tests / alternative endpoints).
        pub fn base_url(mut self, v: impl Into<String>) -> Self {
            self.base_url = v.into();
            self
        }

        /// Sampling temperature; `None` = provider default.
        pub fn temperature(mut self, v: Option<f64>) -> Self {
            self.temperature = v;
            self
        }

        /// Nucleus sampling `top_p`.
        pub fn top_p(mut self, v: Option<f64>) -> Self {
            self.top_p = v;
            self
        }

        /// Top-k sampling.
        pub fn top_k(mut self, v: Option<i64>) -> Self {
            self.top_k = v;
            self
        }

        /// Presence penalty.
        pub fn presence_penalty(mut self, v: Option<f64>) -> Self {
            self.presence_penalty = v;
            self
        }

        /// Call-site label (surfaces in span names + per-call log).
        pub fn slot(mut self, v: impl Into<String>) -> Self {
            self.slot = Some(v.into());
            self
        }

        /// OpenRouter provider routing override.
        pub fn provider_order(mut self, v: Option<Vec<String>>) -> Self {
            self.provider_order = v;
            self
        }

        /// Allow OpenRouter provider fallbacks.
        pub fn provider_allow_fallbacks(mut self, v: bool) -> Self {
            self.provider_allow_fallbacks = v;
            self
        }

        /// Reasoning effort (`"off"`/`"low"`/`"medium"`/`"high"`); `None` = default.
        pub fn reasoning(mut self, v: Option<String>) -> Self {
            self.reasoning = v;
            self
        }

        /// Hard thinking-token budget (wins over reasoning effort).
        pub fn reasoning_max_tokens(mut self, v: Option<i64>) -> Self {
            self.reasoning_max_tokens = v;
            self
        }

        /// Enable tool-calling (gates the agentic loop in responders).
        pub fn tool_calling(mut self, v: bool) -> Self {
            self.tool_calling = v;
            self
        }

        /// Enable `view_image` registration.
        pub fn image_tools(mut self, v: bool) -> Self {
            self.image_tools = v;
            self
        }

        /// Send image content blocks in tool-result messages.
        pub fn multimodal(mut self, v: bool) -> Self {
            self.multimodal = v;
            self
        }

        /// Skip SSE streaming (models that emit tool calls as text under stream).
        pub fn no_stream(mut self, v: bool) -> Self {
            self.no_stream = v;
            self
        }

        /// Qwen3 no-think stabiliser (append a closed think block every call).
        pub fn think_prepend(mut self, v: bool) -> Self {
            self.think_prepend = v;
            self
        }

        /// Inject the shared rate-limit semaphore (DESIGN D13).
        pub fn semaphore(mut self, sem: Arc<Semaphore>) -> Self {
            self.semaphore = Some(sem);
            self
        }

        /// Inject a pre-built HTTP client (shared pool / test transport).
        pub fn http(mut self, http: reqwest::Client) -> Self {
            self.http = Some(http);
            self
        }

        /// Finalize the client, defaulting the HTTP pool + semaphore if unset.
        #[must_use]
        pub fn build(self) -> OpenRouterClient {
            let http = self.http.unwrap_or_else(|| {
                reqwest::Client::builder()
                    .read_timeout(Duration::from_secs(HTTP_TIMEOUT_S))
                    .build()
                    .expect("reqwest client builds with rustls TLS")
            });
            let semaphore = self
                .semaphore
                .unwrap_or_else(|| Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT)));
            OpenRouterClient {
                api_key: self.api_key,
                model: self.model,
                base_url: self.base_url,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                presence_penalty: self.presence_penalty,
                slot: self.slot,
                provider_order: self.provider_order,
                provider_allow_fallbacks: self.provider_allow_fallbacks,
                reasoning: self.reasoning,
                reasoning_max_tokens: self.reasoning_max_tokens,
                tool_calling: self.tool_calling,
                image_tools: self.image_tools,
                multimodal: self.multimodal,
                no_stream: self.no_stream,
                think_prepend: self.think_prepend,
                http,
                semaphore,
            }
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    impl OpenRouterClient {
        /// Start building a client with the required `api_key` + `model`.
        #[must_use]
        pub fn builder(
            api_key: impl Into<String>,
            model: impl Into<String>,
        ) -> OpenRouterClientBuilder {
            OpenRouterClientBuilder::new(api_key, model)
        }

        /// The configured model string.
        #[must_use]
        pub fn model(&self) -> &str {
            &self.model
        }

        /// The API key this client sends.
        #[must_use]
        pub fn api_key(&self) -> &str {
            &self.api_key
        }

        /// The base URL.
        #[must_use]
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        /// Sampling temperature (`None` = provider default).
        #[must_use]
        pub fn temperature(&self) -> Option<f64> {
            self.temperature
        }

        /// `top_p`.
        #[must_use]
        pub fn top_p(&self) -> Option<f64> {
            self.top_p
        }

        /// `top_k`.
        #[must_use]
        pub fn top_k(&self) -> Option<i64> {
            self.top_k
        }

        /// Presence penalty.
        #[must_use]
        pub fn presence_penalty(&self) -> Option<f64> {
            self.presence_penalty
        }

        /// Reasoning effort string.
        #[must_use]
        pub fn reasoning(&self) -> Option<&str> {
            self.reasoning.as_deref()
        }

        /// Whether the Qwen3 think-prepend workaround is on.
        #[must_use]
        pub fn think_prepend(&self) -> bool {
            self.think_prepend
        }

        /// Whether `view_image` registration is enabled for this slot.
        #[must_use]
        pub fn image_tools_enabled(&self) -> bool {
            self.image_tools
        }

        /// Whether `ImageResult`s serialize as multimodal blocks.
        #[must_use]
        pub fn multimodal(&self) -> bool {
            self.multimodal
        }

        /// Whether tool-calling is enabled.
        #[must_use]
        pub fn tool_calling_enabled(&self) -> bool {
            self.tool_calling
        }

        /// The config slot label.
        #[must_use]
        pub fn slot(&self) -> Option<&str> {
            self.slot.as_deref()
        }

        /// The injected rate-limit semaphore (shared across slots when built by
        /// `create_llm_clients`).
        #[must_use]
        pub fn semaphore(&self) -> &Arc<Semaphore> {
            &self.semaphore
        }

        /// Request headers: `Authorization: Bearer …` + `Content-Type` (spec 08).
        #[must_use]
        pub fn build_headers(&self) -> BTreeMap<String, String> {
            let mut headers = BTreeMap::new();
            headers.insert(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            );
            headers.insert("Content-Type".to_string(), "application/json".to_string());
            headers
        }

        /// Build the OpenRouter request body from `messages` (+ optional
        /// `tools`), applying every knob per spec 08 §7–11.
        #[must_use]
        pub fn build_payload(&self, messages: &[Message], tools: Option<&[Value]>) -> Value {
            let msgs: Vec<Value> = messages.iter().map(Message::to_dict).collect();
            let mut payload = json!({ "model": self.model, "messages": msgs });
            if self.model.starts_with("anthropic/") {
                if let Some(arr) = payload["messages"].as_array_mut() {
                    mark_system_cache_breakpoint(arr);
                }
            }
            if self.think_prepend {
                if let Some(arr) = payload["messages"].as_array_mut() {
                    arr.push(json!({ "role": "assistant", "content": "<think>\n\n</think>" }));
                }
            }
            if let Some(t) = self.temperature {
                payload["temperature"] = json!(t);
            }
            if let Some(t) = self.top_p {
                payload["top_p"] = json!(t);
            }
            if let Some(t) = self.top_k {
                payload["top_k"] = json!(t);
            }
            if let Some(t) = self.presence_penalty {
                payload["presence_penalty"] = json!(t);
            }
            if let Some(order) = &self.provider_order {
                payload["provider"] =
                    json!({ "order": order, "allow_fallbacks": self.provider_allow_fallbacks });
            }
            if let Some(n) = self.reasoning_max_tokens {
                payload["reasoning"] = json!({ "max_tokens": n });
            } else if self.reasoning.as_deref() == Some("off") {
                payload["reasoning"] = json!({ "exclude": true });
            } else if let Some(r) = &self.reasoning {
                payload["reasoning"] = json!({ "effort": r });
            }
            if let Some(tools) = tools {
                if !tools.is_empty() {
                    payload["tools"] = Value::Array(tools.to_vec());
                    payload["tool_choice"] = json!("auto");
                }
            }
            payload
        }

        /// Log a ≥400 upstream error body at WARNING before erroring
        /// (spec 08 §5). Truncated to 600 scalars, slot-suffixed.
        fn log_http_error_body(&self, status: u16, body: &str) {
            let slot_suffix = self
                .slot
                .as_deref()
                .map(|s| format!(".{s}"))
                .unwrap_or_default();
            let line = format!(
                "{} {} {} {}",
                ls::tag("LLM", ls::R),
                ls::kv_styled(
                    &format!("http_error{slot_suffix}"),
                    &status.to_string(),
                    ls::W,
                    ls::R
                ),
                ls::kv_styled("model", &self.model, ls::W, ls::LW),
                ls::kv_styled(
                    "body",
                    &ls::trunc(body, HTTP_ERROR_BODY_LIMIT),
                    ls::W,
                    ls::LW
                ),
            );
            tracing::warn!(target: "familiar_connect.llm", "{line}");
        }

        /// POST with the 429 retry/backoff policy (spec 08 §4). Holds a
        /// semaphore permit ONLY across each POST — released before the backoff
        /// sleep so a retrying background call never starves live traffic.
        async fn post_with_retry(&self, url: &str, payload: &Value) -> Result<reqwest::Response> {
            let mut last: Option<reqwest::Response> = None;
            for attempt in 0..=MAX_RETRIES {
                let response = {
                    let _permit = self
                        .semaphore
                        .acquire()
                        .await
                        .map_err(|e| anyhow!("rate-limit semaphore closed: {e}"))?;
                    self.http
                        .post(url)
                        .header("Authorization", format!("Bearer {}", self.api_key))
                        .json(payload)
                        .send()
                        .await?
                };
                if response.status().as_u16() != 429 {
                    return Ok(response);
                }
                if attempt == MAX_RETRIES {
                    last = Some(response);
                    break;
                }
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                let delay = backoff_delay(attempt, retry_after.as_deref());
                tracing::warn!(
                    target: "familiar_connect.llm",
                    "429 from {url} (attempt {}/{}), retrying in {:.1}s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    delay.as_secs_f64(),
                );
                tokio::time::sleep(delay).await;
            }
            last.ok_or_else(|| anyhow!("post_with_retry produced no response"))
        }

        /// Blocking, 429-retrying chat completion (spec 08 §4–6).
        pub async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
            let url = format!("{}/chat/completions", self.base_url);
            let payload = self.build_payload(&messages, None);
            let response = self.post_with_retry(&url, &payload).await?;
            let status = response.status().as_u16();
            if status >= 400 {
                let body = response.text().await.unwrap_or_default();
                self.log_http_error_body(status, &body);
                return Err(anyhow!("OpenRouter chat failed: HTTP {status}"));
            }
            let data: Value = response.json().await?;
            let choices = data
                .get("choices")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if choices.is_empty() {
                return Err(anyhow!("No choices returned from the API"));
            }
            let reply = choices[0]
                .get("message")
                .ok_or_else(|| anyhow!("choice missing message field"))?;
            let role = reply
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant")
                .to_string();
            // Python `content = reply.get("content") or ""` (spec 08 §6): a
            // non-empty string OR a non-empty list (block form) is preserved
            // verbatim; None, an empty string, or an empty list all collapse
            // to "". Mirror that truthiness contract — assistant replies are
            // strings in practice, but the list branch keeps a multimodal /
            // block-form reply intact instead of silently dropping it to "".
            let content = match reply.get("content") {
                Some(Value::String(s)) if !s.is_empty() => Content::Text(s.clone()),
                Some(Value::Array(a)) if !a.is_empty() => Content::Blocks(a.clone()),
                _ => Content::Text(String::new()),
            };
            let tool_calls = match reply.get("tool_calls") {
                Some(Value::Array(a)) if !a.is_empty() => Some(
                    a.iter()
                        .filter(|x| x.is_object())
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            };
            Ok(Message {
                role,
                content,
                name: None,
                tool_calls,
                tool_call_id: None,
            })
        }

        /// SSE streaming completion (spec 08 §12–16). Acquires a permit, opens
        /// the request, and RELEASES the permit immediately once the response
        /// headers pass the status check — before any body iteration.
        pub async fn stream_completion(
            &self,
            messages: Vec<Message>,
            tools: Option<Vec<Value>>,
        ) -> Result<BoxStream<'static, Result<LlmDelta>>> {
            if self.no_stream {
                return self.no_stream_completion(messages).await;
            }
            let url = format!("{}/chat/completions", self.base_url);
            let mut payload = self.build_payload(&messages, tools.as_deref());
            payload["stream"] = json!(true);
            payload["usage"] = json!({ "include": true });

            let mut metrics = CallMetrics::new(
                self.slot.clone(),
                self.model.clone(),
                input_chars(&messages),
            );

            let permit = self
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| anyhow!("rate-limit semaphore closed: {e}"))?;
            let send_result = self
                .http
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&payload)
                .send()
                .await;
            let response = match send_result {
                Ok(r) => r,
                Err(e) => {
                    drop(permit);
                    metrics.status = "error";
                    metrics.emit();
                    return Err(anyhow!("OpenRouter stream request failed: {e}"));
                }
            };
            let status = response.status().as_u16();
            if status >= 400 {
                let body = response.text().await.unwrap_or_default();
                self.log_http_error_body(status, &body);
                drop(permit);
                metrics.status = "error";
                metrics.emit();
                return Err(anyhow!("OpenRouter stream failed: HTTP {status}"));
            }
            // Headers accepted — release the permit before body iteration so a
            // long stream does not occupy a rate-limit slot (spec 08 §3).
            drop(permit);
            metrics.t_first_byte = Some(Instant::now());

            let events: InnerEvents = Box::pin(response.bytes_stream().eventsource());
            Ok(Box::pin(SseDeltaStream {
                events,
                metrics,
                done: false,
                emitted: false,
            }))
        }

        /// `no_stream=True` path: delegate to `chat`, then synthesize deltas —
        /// content, one per tool call, terminal finish reason (spec 08 §18).
        async fn no_stream_completion(
            &self,
            messages: Vec<Message>,
        ) -> Result<BoxStream<'static, Result<LlmDelta>>> {
            let msg = self.chat(messages).await?;
            let mut deltas: Vec<LlmDelta> = Vec::new();
            let content = msg.content_str();
            if !content.is_empty() {
                deltas.push(LlmDelta {
                    content,
                    ..Default::default()
                });
            }
            let has_tools = msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
            if let Some(tcs) = &msg.tool_calls {
                for (i, tc) in tcs.iter().enumerate() {
                    let function = tc.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let arguments = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let id = tc
                        .get("id")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("call_{i}"), str::to_owned);
                    deltas.push(LlmDelta {
                        tool_calls: vec![json!({
                            "index": i,
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments },
                        })],
                        ..Default::default()
                    });
                }
            }
            deltas.push(LlmDelta {
                finish_reason: Some(if has_tools { "tool_calls" } else { "stop" }.to_string()),
                ..Default::default()
            });
            Ok(Box::pin(futures::stream::iter(
                deltas.into_iter().map(|d| -> Result<LlmDelta> { Ok(d) }),
            )))
        }

        /// Stream assistant content deltas as strings — a projection of
        /// `stream_completion` to non-empty `.content` (spec 08 §17). Errors
        /// propagate; the inner cleanup runs when this stream is dropped.
        pub async fn chat_stream(
            &self,
            messages: Vec<Message>,
        ) -> Result<BoxStream<'static, Result<String>>> {
            let inner = self.stream_completion(messages, None).await?;
            Ok(Box::pin(inner.filter_map(|item| async move {
                match item {
                    Ok(d) if !d.content.is_empty() => Some(Ok(d.content)),
                    Ok(_) => None,
                    Err(e) => Some(Err(e)),
                }
            })))
        }
    }

    #[async_trait]
    // `OpenRouterClient::` here is the fully-qualified INHERENT method (which
    // shadows the trait method) — using `Self::` would resolve the same way but
    // reads as recursion; the explicit type name is the clearer disambiguation.
    #[allow(clippy::use_self)]
    impl LlmClient for OpenRouterClient {
        async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
            OpenRouterClient::chat(self, messages).await
        }

        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            tools: Option<Vec<Value>>,
        ) -> Result<BoxStream<'static, Result<LlmDelta>>> {
            OpenRouterClient::stream_completion(self, messages, tools).await
        }

        fn slot(&self) -> Option<&str> {
            self.slot.as_deref()
        }

        fn multimodal(&self) -> bool {
            self.multimodal
        }

        fn tool_calling_enabled(&self) -> bool {
            self.tool_calling
        }
    }

    /// One structured `[Config]` INFO line per slot (Python parity).
    fn log_slot_config(slot_name: &str, slot: &LLMSlotConfig) {
        let temp = slot
            .temperature
            .map_or_else(|| "default".to_string(), |t| t.to_string());
        let mut parts = vec![
            ls::tag("Config", ls::W),
            ls::kv("slot", slot_name),
            ls::kv("model", &slot.model),
            ls::kv("temperature", &temp),
        ];
        if let Some(order) = &slot.provider_order {
            parts.push(ls::kv("provider_order", &order.join(",")));
            if !slot.provider_allow_fallbacks {
                parts.push(ls::kv_styled("fallbacks", "off", ls::W, ls::LY));
            }
        }
        if let Some(r) = &slot.reasoning {
            parts.push(ls::kv_styled("reasoning", r, ls::W, ls::LM));
        }
        if let Some(v) = slot.top_p {
            parts.push(ls::kv("top_p", &v.to_string()));
        }
        if let Some(v) = slot.top_k {
            parts.push(ls::kv("top_k", &v.to_string()));
        }
        if let Some(v) = slot.presence_penalty {
            parts.push(ls::kv("presence_penalty", &v.to_string()));
        }
        if slot.think_prepend {
            parts.push(ls::kv_styled("think_prepend", "on", ls::W, ls::LM));
        }
        if slot.tool_calling {
            parts.push(ls::kv_styled("tools", "on", ls::W, ls::LM));
        }
        if slot.image_tools {
            parts.push(ls::kv_styled("image_tools", "on", ls::W, ls::LM));
        }
        if slot.multimodal {
            parts.push(ls::kv_styled("multimodal", "on", ls::W, ls::LM));
        }
        tracing::info!(target: "familiar_connect.llm", "{}", parts.join(" "));
    }

    /// One [`OpenRouterClient`] per call-site slot in `LLM_SLOT_NAMES`.
    ///
    /// Plus a reserved `"__image_description__"` client when
    /// `image_description_model` is set. All clients share ONE injected
    /// rate-limit semaphore sized from `[llm].max_concurrent_requests`
    /// (spec 08 §22; DESIGN D13). A missing slot is an `Err` (Python raised
    /// `KeyError`, which run.py caught).
    pub fn create_llm_clients(
        api_key: &str,
        config: &CharacterConfig,
    ) -> Result<BTreeMap<String, OpenRouterClient>> {
        let max_concurrent = usize::try_from(config.llm_max_concurrent_requests)
            .unwrap_or(DEFAULT_MAX_CONCURRENT)
            .max(1);
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        let mut clients = BTreeMap::new();
        for slot_name in LLM_SLOT_NAMES {
            let slot = config
                .llm
                .get(slot_name)
                .ok_or_else(|| anyhow!("missing LLM slot config: {slot_name}"))?;
            let client = OpenRouterClient::builder(api_key, &slot.model)
                .temperature(slot.temperature)
                .top_p(slot.top_p)
                .top_k(slot.top_k)
                .presence_penalty(slot.presence_penalty)
                .slot(slot_name)
                .provider_order(slot.provider_order.clone())
                .provider_allow_fallbacks(slot.provider_allow_fallbacks)
                .reasoning(slot.reasoning.clone())
                .tool_calling(slot.tool_calling)
                .image_tools(slot.image_tools)
                .multimodal(slot.multimodal)
                .think_prepend(slot.think_prepend)
                .semaphore(semaphore.clone())
                .build();
            log_slot_config(slot_name, slot);
            clients.insert(slot_name.to_string(), client);
        }
        if !config.image_description_model.is_empty() {
            let client = OpenRouterClient::builder(api_key, &config.image_description_model)
                .slot("image_description")
                .semaphore(semaphore)
                .build();
            let line = format!(
                "{} {} {}",
                ls::tag("Config", ls::W),
                ls::kv("slot", "image_description"),
                ls::kv("model", &config.image_description_model),
            );
            tracing::info!(target: "familiar_connect.llm", "{line}");
            clients.insert("__image_description__".to_string(), client);
        }
        Ok(clients)
    }

    #[cfg(test)]
    mod tests {
        use super::{MAX_DELAY_S, OpenRouterClient, backoff_delay, create_llm_clients};
        use crate::config::{CharacterConfig, LLMSlotConfig};
        use crate::diagnostics::collector::{get_span_collector, reset_span_collector};
        use crate::diagnostics::testutil::{Capture, install_capture, singleton_guard, strip_ansi};
        use crate::llm::{Content, LlmDelta, Message};
        use futures::StreamExt;
        use serde_json::{Value, json};
        use std::collections::BTreeSet;
        use std::sync::Arc;
        use tokio::sync::Semaphore;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // --- helpers -------------------------------------------------------

        fn client_for(base_url: &str) -> OpenRouterClient {
            OpenRouterClient::builder("k", "m")
                .base_url(base_url)
                .build()
        }

        /// A client whose HTTP pool carries NO `read_timeout` timer — required
        /// for `start_paused` retry tests, where an auto-advancing virtual clock
        /// would otherwise jump past the 120 s read timeout while the wiremock
        /// server (a real-time background thread) is still replying.
        fn client_no_timeout(base_url: &str) -> OpenRouterClient {
            OpenRouterClient::builder("k", "m")
                .base_url(base_url)
                .http(reqwest::Client::new())
                .build()
        }

        fn sse_content(deltas: &[&str]) -> String {
            use std::fmt::Write;
            let mut body = String::new();
            for d in deltas {
                let _ = write!(
                    body,
                    "data: {{\"choices\": [{{\"delta\": {{\"role\": \"assistant\", \"content\": \"{d}\"}}}}]}}\n\n"
                );
            }
            body.push_str("data: [DONE]\n\n");
            body
        }

        fn sse_with_usage(deltas: &[&str], usage: Option<Value>, provider: Option<&str>) -> String {
            use std::fmt::Write;
            let mut body = String::new();
            for d in deltas {
                let _ = write!(
                    body,
                    "data: {{\"choices\": [{{\"delta\": {{\"content\": \"{d}\"}}}}]}}\n\n"
                );
            }
            if usage.is_some() || provider.is_some() {
                let mut chunk = json!({ "choices": [] });
                if let Some(u) = usage {
                    chunk["usage"] = u;
                }
                if let Some(p) = provider {
                    chunk["provider"] = json!(p);
                }
                let _ = write!(body, "data: {chunk}\n\n");
            }
            body.push_str("data: [DONE]\n\n");
            body
        }

        async fn mount_sse(server: &MockServer, body: String) {
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream"),
                )
                .mount(server)
                .await;
        }

        async fn mount_json(server: &MockServer, status: u16, body: Value) {
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(status).set_body_json(body))
                .mount(server)
                .await;
        }

        fn user(text: &str) -> Message {
            Message::new("user", text)
        }

        // --- build_payload -------------------------------------------------

        #[test]
        fn payload_has_model_and_messages() {
            let c = OpenRouterClient::builder("k", "openai/gpt-4o").build();
            let msgs = vec![
                Message::new("system", "You are helpful."),
                Message::new("user", "Hi").with_name("Alice"),
            ];
            let p = c.build_payload(&msgs, None);
            assert_eq!(p["model"], "openai/gpt-4o");
            let expected: Vec<Value> = msgs.iter().map(Message::to_dict).collect();
            assert_eq!(p["messages"], Value::Array(expected));
        }

        #[test]
        fn payload_includes_temperature() {
            let c = OpenRouterClient::builder("k", "m")
                .temperature(Some(0.7))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert!((p["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        }

        #[test]
        fn payload_omits_provider_when_unset() {
            let c = OpenRouterClient::builder("k", "m").build();
            let p = c.build_payload(&[user("x")], None);
            assert!(p.get("provider").is_none());
        }

        #[test]
        fn payload_pins_provider_order() {
            let c = OpenRouterClient::builder("k", "m")
                .provider_order(Some(vec!["z-ai".into(), "deepinfra".into()]))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert_eq!(
                p["provider"],
                json!({ "order": ["z-ai", "deepinfra"], "allow_fallbacks": true })
            );
        }

        #[test]
        fn payload_disables_fallbacks_when_requested() {
            let c = OpenRouterClient::builder("k", "m")
                .provider_order(Some(vec!["z-ai".into()]))
                .provider_allow_fallbacks(false)
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert_eq!(p["provider"]["allow_fallbacks"], false);
        }

        #[test]
        fn payload_omits_reasoning_when_unset() {
            let c = OpenRouterClient::builder("k", "m").build();
            assert!(
                c.build_payload(&[user("x")], None)
                    .get("reasoning")
                    .is_none()
            );
        }

        #[test]
        fn payload_reasoning_off_excludes() {
            let c = OpenRouterClient::builder("k", "m")
                .reasoning(Some("off".into()))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert_eq!(p["reasoning"], json!({ "exclude": true }));
        }

        #[test]
        fn payload_reasoning_effort_levels() {
            for level in ["low", "medium", "high"] {
                let c = OpenRouterClient::builder("k", "m")
                    .reasoning(Some(level.into()))
                    .build();
                let p = c.build_payload(&[user("x")], None);
                assert_eq!(p["reasoning"], json!({ "effort": level }));
            }
        }

        #[test]
        fn payload_omits_sampling_when_unset() {
            let c = OpenRouterClient::builder("k", "m").build();
            let p = c.build_payload(&[user("x")], None);
            assert!(p.get("top_p").is_none());
            assert!(p.get("top_k").is_none());
            assert!(p.get("presence_penalty").is_none());
        }

        #[test]
        fn payload_includes_sampling() {
            let c = OpenRouterClient::builder("k", "m")
                .top_p(Some(0.95))
                .top_k(Some(20))
                .presence_penalty(Some(1.5))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert!((p["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-9);
            assert_eq!(p["top_k"], 20);
            assert!((p["presence_penalty"].as_f64().unwrap() - 1.5).abs() < 1e-9);
        }

        #[test]
        fn payload_reasoning_max_tokens() {
            let c = OpenRouterClient::builder("k", "m")
                .reasoning_max_tokens(Some(2048))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert_eq!(p["reasoning"], json!({ "max_tokens": 2048 }));
        }

        #[test]
        fn payload_reasoning_max_tokens_wins_over_effort() {
            let c = OpenRouterClient::builder("k", "m")
                .reasoning(Some("low".into()))
                .reasoning_max_tokens(Some(2048))
                .build();
            let p = c.build_payload(&[user("x")], None);
            assert_eq!(p["reasoning"], json!({ "max_tokens": 2048 }));
        }

        #[test]
        fn payload_no_prefill_by_default() {
            let c = OpenRouterClient::builder("k", "m").build();
            let p = c.build_payload(&[user("x")], None);
            let msgs = p["messages"].as_array().unwrap();
            assert_eq!(msgs.last().unwrap()["role"], "user");
        }

        #[test]
        fn payload_think_prepend_appends_prefill() {
            let c = OpenRouterClient::builder("k", "m")
                .think_prepend(true)
                .build();
            let p = c.build_payload(&[user("x")], None);
            let msgs = p["messages"].as_array().unwrap();
            assert_eq!(
                *msgs.last().unwrap(),
                json!({ "role": "assistant", "content": "<think>\n\n</think>" })
            );
            assert_eq!(msgs[msgs.len() - 2]["role"], "user");
        }

        #[test]
        fn payload_anthropic_caches_system_prompt() {
            let c = OpenRouterClient::builder("k", "anthropic/claude-haiku-4.5").build();
            let msgs = vec![
                Message::new("system", "You are a wizard familiar."),
                Message::new("user", "Hi").with_name("Alice"),
            ];
            let p = c.build_payload(&msgs, None);
            let system = &p["messages"][0];
            assert_eq!(system["role"], "system");
            let blocks = system["content"].as_array().unwrap();
            let last = blocks.last().unwrap();
            assert_eq!(last["cache_control"], json!({ "type": "ephemeral" }));
            assert_eq!(last["text"], "You are a wizard familiar.");
        }

        #[test]
        fn payload_non_anthropic_leaves_system_string() {
            let c = OpenRouterClient::builder("k", "z-ai/glm-5.1").build();
            let msgs = vec![
                Message::new("system", "You are a wizard familiar."),
                Message::new("user", "Hi").with_name("Alice"),
            ];
            let p = c.build_payload(&msgs, None);
            assert_eq!(p["messages"][0]["content"], "You are a wizard familiar.");
            assert!(!p.to_string().contains("cache_control"));
        }

        #[test]
        fn payload_caching_leaves_user_assistant_unchanged() {
            let msgs = vec![
                Message::new("system", "System prompt."),
                Message::new("user", "Hi").with_name("Alice"),
                Message::new("assistant", "Hello!"),
            ];
            for model in ["anthropic/claude-haiku-4.5", "z-ai/glm-5.1"] {
                let c = OpenRouterClient::builder("k", model).build();
                let p = c.build_payload(&msgs, None);
                assert_eq!(p["messages"][1], msgs[1].to_dict());
                assert_eq!(p["messages"][2], msgs[2].to_dict());
            }
        }

        #[test]
        fn payload_caches_head_not_trailing_system() {
            let c = OpenRouterClient::builder("k", "anthropic/claude-haiku-4.5").build();
            let msgs = vec![
                Message::new("system", "Stable head: character + RAG."),
                Message::new("user", "Hi").with_name("Alice"),
                Message::new("system", "Volatile trailing: 12:03, 4 unread."),
            ];
            let p = c.build_payload(&msgs, None);
            let head = &p["messages"][0];
            let trailing = &p["messages"][2];
            let blocks = head["content"].as_array().unwrap();
            assert_eq!(
                blocks.last().unwrap()["cache_control"],
                json!({ "type": "ephemeral" })
            );
            assert_eq!(
                blocks.last().unwrap()["text"],
                "Stable head: character + RAG."
            );
            assert_eq!(trailing["content"], "Volatile trailing: 12:03, 4 unread.");
        }

        #[test]
        fn payload_caching_no_system_message_is_noop() {
            let c = OpenRouterClient::builder("k", "anthropic/claude-haiku-4.5").build();
            let msgs = vec![Message::new("user", "Hi").with_name("Alice")];
            let p = c.build_payload(&msgs, None);
            assert_eq!(p["messages"][0], msgs[0].to_dict());
            assert!(!p.to_string().contains("cache_control"));
        }

        #[test]
        fn payload_caching_list_system_content_no_double_wrap() {
            let c = OpenRouterClient::builder("k", "anthropic/claude-haiku-4.5").build();
            let msgs = vec![
                Message::new(
                    "system",
                    vec![
                        json!({ "type": "text", "text": "First block." }),
                        json!({ "type": "text", "text": "Second block." }),
                    ],
                ),
                Message::new("user", "Hi").with_name("Alice"),
            ];
            let p = c.build_payload(&msgs, None);
            let blocks = p["messages"][0]["content"].as_array().unwrap();
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0], json!({ "type": "text", "text": "First block." }));
            assert_eq!(
                blocks[1],
                json!({
                    "type": "text",
                    "text": "Second block.",
                    "cache_control": { "type": "ephemeral" }
                })
            );
        }

        #[test]
        fn payload_includes_tools_when_passed() {
            let c = OpenRouterClient::builder("k", "m").build();
            let tools = vec![json!({
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "Schedule a wake.",
                    "parameters": { "type": "object", "properties": {} }
                }
            })];
            let p = c.build_payload(&[user("hi")], Some(&tools));
            assert_eq!(p["tools"], Value::Array(tools));
            assert_eq!(p["tool_choice"], "auto");
        }

        #[test]
        fn payload_omits_tools_when_empty() {
            let c = OpenRouterClient::builder("k", "m").build();
            let p = c.build_payload(&[user("hi")], Some(&[]));
            assert!(p.get("tools").is_none());
            assert!(p.get("tool_choice").is_none());
        }

        #[test]
        fn payload_omits_tools_when_none() {
            let c = OpenRouterClient::builder("k", "m").build();
            assert!(c.build_payload(&[user("hi")], None).get("tools").is_none());
        }

        // --- headers / init ------------------------------------------------

        #[test]
        fn builds_request_headers() {
            let c = OpenRouterClient::builder("sk-or-test-123", "openai/gpt-4o").build();
            let h = c.build_headers();
            assert_eq!(h["Authorization"], "Bearer sk-or-test-123");
            assert!(h.contains_key("Content-Type"));
        }

        #[test]
        fn init_stores_fields() {
            let c = OpenRouterClient::builder("test-key", "anthropic/claude-sonnet-4").build();
            assert_eq!(c.api_key(), "test-key");
            assert_eq!(c.model(), "anthropic/claude-sonnet-4");
            assert!(c.base_url().contains("openrouter.ai"));
        }

        #[test]
        fn tool_calling_flag_default_off_and_ctor_on() {
            assert!(
                !OpenRouterClient::builder("k", "m")
                    .build()
                    .tool_calling_enabled()
            );
            assert!(
                OpenRouterClient::builder("k", "m")
                    .tool_calling(true)
                    .build()
                    .tool_calling_enabled()
            );
        }

        // --- backoff (pure) ------------------------------------------------

        #[test]
        fn backoff_is_exponential_and_capped() {
            assert!((backoff_delay(0, None).as_secs_f64() - 1.0).abs() < 1e-9);
            assert!((backoff_delay(1, None).as_secs_f64() - 2.0).abs() < 1e-9);
            assert!((backoff_delay(2, None).as_secs_f64() - 4.0).abs() < 1e-9);
            assert!((backoff_delay(3, None).as_secs_f64() - 8.0).abs() < 1e-9);
            // never exceeds MAX_DELAY_S.
            for attempt in 0..12 {
                assert!(backoff_delay(attempt, None).as_secs_f64() <= MAX_DELAY_S);
            }
        }

        #[test]
        fn backoff_honors_retry_after_and_caps_it() {
            assert!((backoff_delay(0, Some("2")).as_secs_f64() - 2.0).abs() < 1e-9);
            assert!((backoff_delay(0, Some("100")).as_secs_f64() - MAX_DELAY_S).abs() < 1e-9);
            // unparseable header falls back to exponential.
            assert!((backoff_delay(2, Some("garbage")).as_secs_f64() - 4.0).abs() < 1e-9);
        }

        // --- chat (wiremock) ----------------------------------------------

        #[tokio::test]
        async fn chat_returns_assistant_message() {
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "Greetings, Alice and Bob!" },
                        "finish_reason": "stop"
                    }]
                }),
            )
            .await;
            let c = client_for(&server.uri());
            let result = c
                .chat(vec![
                    Message::new("system", "You are a wizard familiar."),
                    Message::new("user", "Hello!").with_name("Alice"),
                ])
                .await
                .unwrap();
            assert_eq!(result.role, "assistant");
            assert_eq!(result.content_str(), "Greetings, Alice and Bob!");
        }

        #[tokio::test]
        async fn chat_sends_all_messages() {
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({ "choices": [{ "message": { "role": "assistant", "content": "ok" } }] }),
            )
            .await;
            let c = client_for(&server.uri());
            c.chat(vec![
                Message::new("system", "sys"),
                Message::new("user", "a").with_name("Alice"),
                Message::new("user", "b").with_name("Bob"),
            ])
            .await
            .unwrap();
            let reqs = server.received_requests().await.unwrap();
            let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
            let sent = body["messages"].as_array().unwrap();
            assert_eq!(sent.len(), 3);
            assert_eq!(sent[0]["role"], "system");
            assert_eq!(sent[1]["name"], "Alice");
            assert_eq!(sent[2]["name"], "Bob");
        }

        #[tokio::test]
        async fn chat_raises_and_logs_body_on_4xx() {
            let server = MockServer::start().await;
            let body = json!({
                "error": {
                    "message": "Unsupported value: 'temperature' does not support 0.7 with this model.",
                    "type": "invalid_request_error"
                }
            });
            mount_json(&server, 400, body).await;
            let c = client_for(&server.uri());
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let result = c.chat(vec![user("hi")]).await;
            assert!(result.is_err());
            let joined: String = cap
                .records()
                .iter()
                .map(|r| strip_ansi(&r.message))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(joined.contains("400"), "log: {joined}");
            assert!(joined.contains("temperature"), "log: {joined}");
        }

        #[tokio::test]
        async fn chat_raises_on_empty_choices() {
            let server = MockServer::start().await;
            mount_json(&server, 200, json!({ "choices": [] })).await;
            let c = client_for(&server.uri());
            let err = c.chat(vec![user("hi")]).await.unwrap_err();
            assert!(err.to_string().to_lowercase().contains("choices"));
        }

        #[tokio::test]
        async fn chat_parses_tool_calls_content_none() {
            let server = MockServer::start().await;
            let tc = json!([{
                "id": "call_abc",
                "type": "function",
                "function": { "name": "set_alarm", "arguments": "{\"reason\":\"wakeup\"}" }
            }]);
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": null, "tool_calls": tc },
                        "finish_reason": "tool_calls"
                    }]
                }),
            )
            .await;
            let c = client_for(&server.uri());
            let result = c.chat(vec![user("set an alarm")]).await.unwrap();
            assert_eq!(result.role, "assistant");
            assert_eq!(result.content_str(), "");
            assert_eq!(result.tool_calls.as_ref().unwrap().len(), 1);
            assert_eq!(result.tool_calls.unwrap()[0]["id"], "call_abc");
        }

        #[tokio::test]
        async fn chat_without_tool_calls_is_plain() {
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": "hi" },
                        "finish_reason": "stop"
                    }]
                }),
            )
            .await;
            let c = client_for(&server.uri());
            let result = c.chat(vec![user("hi")]).await.unwrap();
            assert_eq!(result.content_str(), "hi");
            assert!(result.tool_calls.is_none());
        }

        #[tokio::test]
        async fn chat_preserves_list_content_blocks() {
            // Python `reply.get("content") or ""` keeps a NON-empty list (block
            // form) verbatim; only None/empty collapses to "". The reply's
            // content must round-trip as `Content::Blocks`, not be flattened to
            // "" (which `as_str` on an array would do).
            let server = MockServer::start().await;
            let blocks = json!([
                { "type": "text", "text": "hello" },
                { "type": "image_url", "image_url": { "url": "data:..." } },
                { "type": "text", "text": "world" }
            ]);
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": blocks.clone() },
                        "finish_reason": "stop"
                    }]
                }),
            )
            .await;
            let c = client_for(&server.uri());
            let result = c.chat(vec![user("hi")]).await.unwrap();
            assert_eq!(
                result.content,
                Content::Blocks(blocks.as_array().unwrap().clone())
            );
            // `content_str` still projects the text blocks, joined by newline.
            assert_eq!(result.content_str(), "hello\nworld");
        }

        #[tokio::test]
        async fn chat_empty_list_content_collapses_to_empty() {
            // Python truthiness: `[] or "" == ""`. An empty content list is
            // falsy, so it collapses to an empty text reply (not empty Blocks).
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": { "role": "assistant", "content": [] },
                        "finish_reason": "stop"
                    }]
                }),
            )
            .await;
            let c = client_for(&server.uri());
            let result = c.chat(vec![user("hi")]).await.unwrap();
            assert_eq!(result.content, Content::Text(String::new()));
            assert_eq!(result.content_str(), "");
        }

        // --- retry loop (paused clock) ------------------------------------

        #[tokio::test(start_paused = true)]
        async fn post_retries_on_429_then_succeeds() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(429))
                .up_to_n_times(1)
                .with_priority(1)
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{ "message": { "role": "assistant", "content": "ok" } }]
                })))
                .with_priority(2)
                .mount(&server)
                .await;
            let c = client_no_timeout(&server.uri());
            let result = c.chat(vec![user("hi")]).await.unwrap();
            assert_eq!(result.content_str(), "ok");
            assert_eq!(server.received_requests().await.unwrap().len(), 2);
        }

        #[tokio::test(start_paused = true)]
        async fn post_gives_up_after_max_retries() {
            let server = MockServer::start().await;
            mount_json(&server, 429, json!({})).await;
            let c = client_no_timeout(&server.uri());
            // chat surfaces the final 429 as an error after 5 total attempts.
            assert!(c.chat(vec![user("hi")]).await.is_err());
            assert_eq!(server.received_requests().await.unwrap().len(), 5);
        }

        #[tokio::test(start_paused = true)]
        async fn post_does_not_retry_non_429() {
            let server = MockServer::start().await;
            mount_json(&server, 500, json!({})).await;
            let c = client_no_timeout(&server.uri());
            assert!(c.chat(vec![user("hi")]).await.is_err());
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
        }

        #[tokio::test]
        async fn post_respects_retry_after_header() {
            // Python parity (test_post_respects_retry_after_header): a 429
            // carrying a `Retry-After` value must drive a delay read OFF the
            // response, NOT the exponential fallback (1.0s for attempt 0). This
            // exercises the header-extraction wiring in `post_with_retry`
            // (`response.headers().get("Retry-After")` -> `backoff_delay`) that
            // the pure `backoff_delay` unit test bypasses entirely. A honored
            // `Retry-After: 0.3` makes the single real retry sleep ~0.3s, well
            // clear of the 1.0s attempt-0 exponential fallback: measuring wall
            // time cleanly separates the two (honored ≈ 0.3s ≪ fallback ≈ 1.0s)
            // without `start_paused` (whose auto-advancing clock overshoots on
            // hyper pool timers) or log capture (whose thread-local subscriber
            // races the parallel suite). Two localhost round trips add only
            // milliseconds, so the window has wide margin at both ends.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0.3"))
                .up_to_n_times(1)
                .with_priority(1)
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{ "message": { "role": "assistant", "content": "ok" } }]
                })))
                .with_priority(2)
                .mount(&server)
                .await;
            let c = client_for(&server.uri());
            let start = std::time::Instant::now();
            let result = c.chat(vec![user("hi")]).await.unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(result.content_str(), "ok");
            assert_eq!(server.received_requests().await.unwrap().len(), 2);
            // Retry-After=0.3 honored → ~0.3s wait, decisively short of the 1.0s
            // exponential fallback the header must override (>= 0.2s confirms a
            // real ~0.3s sleep happened, not near-zero).
            assert!(
                (0.2..0.75).contains(&elapsed),
                "expected ~0.3s Retry-After wait, got {elapsed:.3}s \
                 (>=0.75 ⇒ fell back to 1.0s exponential; header not honored)"
            );
        }

        // --- streaming (wiremock) -----------------------------------------

        #[tokio::test]
        async fn stream_yields_each_content_delta() {
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["Hel", "lo", ", world"])).await;
            let c = client_for(&server.uri());
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            let mut got = Vec::new();
            while let Some(item) = s.next().await {
                got.push(item.unwrap());
            }
            assert_eq!(got, vec!["Hel", "lo", ", world"]);
        }

        #[tokio::test]
        async fn stream_content_deltas_carry_no_tool_calls() {
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["Hel", "lo"])).await;
            let c = client_for(&server.uri());
            let mut s = c.stream_completion(vec![user("hi")], None).await.unwrap();
            let mut deltas: Vec<LlmDelta> = Vec::new();
            while let Some(item) = s.next().await {
                deltas.push(item.unwrap());
            }
            let contents: Vec<String> = deltas
                .iter()
                .filter(|d| !d.content.is_empty())
                .map(|d| d.content.clone())
                .collect();
            assert_eq!(contents, vec!["Hel", "lo"]);
            assert!(deltas.iter().all(|d| d.tool_calls.is_empty()));
        }

        #[tokio::test]
        async fn stream_cancel_mid_stream_stops_iteration() {
            let server = MockServer::start().await;
            let deltas: Vec<String> = (0..100).map(|i| format!("d{i}")).collect();
            let refs: Vec<&str> = deltas.iter().map(String::as_str).collect();
            mount_sse(&server, sse_content(&refs)).await;
            let c = client_for(&server.uri());
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            let mut got = Vec::new();
            while let Some(item) = s.next().await {
                got.push(item.unwrap());
                if got.len() >= 2 {
                    break;
                }
            }
            assert_eq!(got, vec!["d0", "d1"]);
        }

        #[tokio::test]
        async fn stream_sends_stream_true_and_usage_include() {
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&[])).await;
            let c = client_for(&server.uri());
            let mut s = c.chat_stream(vec![user("x")]).await.unwrap();
            while s.next().await.is_some() {}
            let reqs = server.received_requests().await.unwrap();
            let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
            assert_eq!(body["stream"], true);
            assert_eq!(body["usage"], json!({ "include": true }));
        }

        #[tokio::test]
        async fn stream_releases_permit_after_headers() {
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["d0", "d1", "d2"])).await;
            let sem = Arc::new(Semaphore::new(1));
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .semaphore(sem.clone())
                .build();
            {
                let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
                let _ = s.next().await;
                // Permit is released the instant headers pass the status check.
                assert_eq!(sem.available_permits(), 1);
            }
            assert_eq!(sem.available_permits(), 1);
        }

        #[tokio::test]
        async fn chat_holds_permit_across_post_then_releases() {
            // Barge-in contract §2: `chat` cannot POST without a permit and
            // releases it once the POST completes.
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({ "choices": [{ "message": { "role": "assistant", "content": "ok" } }] }),
            )
            .await;
            let sem = Arc::new(Semaphore::new(1));
            let held = sem.clone().acquire_owned().await.unwrap();
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .semaphore(sem.clone())
                .build();
            let handle = tokio::spawn(async move { c.chat(vec![user("hi")]).await });
            // With the only permit held externally, chat is parked on acquire.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert!(!handle.is_finished());
            assert_eq!(sem.available_permits(), 0);
            drop(held);
            let reply = handle.await.unwrap().unwrap();
            assert_eq!(reply.content_str(), "ok");
            // Permit released after the POST — never held across the caller's use.
            assert_eq!(sem.available_permits(), 1);
        }

        #[tokio::test]
        async fn stream_429_raises_without_retry() {
            let server = MockServer::start().await;
            mount_json(&server, 429, json!({})).await;
            let c = client_for(&server.uri());
            assert!(c.stream_completion(vec![user("x")], None).await.is_err());
            assert_eq!(server.received_requests().await.unwrap().len(), 1);
        }

        #[tokio::test]
        async fn stream_logs_body_on_4xx() {
            let server = MockServer::start().await;
            let body = json!({
                "error": { "message": "Unsupported value: 'temperature' does not support 0.7 with this model." }
            });
            mount_json(&server, 400, body).await;
            let c = client_for(&server.uri());
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            assert!(c.stream_completion(vec![user("x")], None).await.is_err());
            let joined: String = cap
                .records()
                .iter()
                .map(|r| strip_ansi(&r.message))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(joined.contains("400"), "log: {joined}");
            assert!(joined.contains("temperature"), "log: {joined}");
        }

        #[tokio::test]
        async fn stream_sse_error_frame_logged_and_skipped() {
            let server = MockServer::start().await;
            let body = "data: {\"error\": {\"message\": \"model not found\", \"code\": 404}}\n\ndata: [DONE]\n\n";
            mount_sse(&server, body.to_string()).await;
            let c = client_for(&server.uri());
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let mut s = c.chat_stream(vec![user("x")]).await.unwrap();
            let mut got = Vec::new();
            while let Some(item) = s.next().await {
                got.push(item.unwrap());
            }
            assert!(got.is_empty());
            let joined: String = cap
                .records()
                .iter()
                .map(|r| r.message.clone())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(joined.contains("model not found"), "log: {joined}");
        }

        #[tokio::test]
        async fn stream_accumulates_tool_call_fragments() {
            let server = MockServer::start().await;
            let body = concat!(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_xyz\",\"type\":\"function\",\"function\":{\"name\":\"set_alarm\",\"arguments\":\"{\\\"reas\"}}]}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"on\\\":\\\"wake\\\",\\\"de\"}}]}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"lay_seconds\\\":30}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n"
            );
            mount_sse(&server, body.to_string()).await;
            let c = client_for(&server.uri());
            let mut s = c
                .stream_completion(vec![user("set an alarm")], None)
                .await
                .unwrap();
            let mut frames: Vec<Vec<Value>> = Vec::new();
            while let Some(item) = s.next().await {
                let d = item.unwrap();
                if !d.tool_calls.is_empty() {
                    frames.push(d.tool_calls);
                }
            }
            assert!(!frames.is_empty(), "expected tool_call deltas");
            // Caller-side accumulation by index.
            let mut id = String::new();
            let mut name = String::new();
            let mut args = String::new();
            for frame in &frames {
                for tc in frame {
                    if let Some(v) = tc.get("id").and_then(Value::as_str) {
                        id.clear();
                        id.push_str(v);
                    }
                    if let Some(v) = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                    {
                        name.clear();
                        name.push_str(v);
                    }
                    if let Some(v) = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        args.push_str(v);
                    }
                }
            }
            assert_eq!(id, "call_xyz");
            assert_eq!(name, "set_alarm");
            let decoded: Value = serde_json::from_str(&args).unwrap();
            assert_eq!(decoded, json!({ "reason": "wake", "delay_seconds": 30 }));
        }

        #[tokio::test]
        async fn stream_sends_tools_in_payload() {
            let server = MockServer::start().await;
            mount_sse(&server, "data: [DONE]\n\n".to_string()).await;
            let c = client_for(&server.uri());
            let tools = vec![json!({
                "type": "function",
                "function": { "name": "set_alarm", "description": "", "parameters": { "type": "object" } }
            })];
            let mut s = c
                .stream_completion(vec![user("x")], Some(tools.clone()))
                .await
                .unwrap();
            while s.next().await.is_some() {}
            let reqs = server.received_requests().await.unwrap();
            let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
            assert_eq!(body["tools"], Value::Array(tools));
            assert_eq!(body["stream"], true);
        }

        #[tokio::test]
        async fn stream_yields_finish_reason() {
            let server = MockServer::start().await;
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            mount_sse(&server, body.to_string()).await;
            let c = client_for(&server.uri());
            let mut s = c.stream_completion(vec![user("x")], None).await.unwrap();
            let mut reasons: Vec<Option<String>> = Vec::new();
            while let Some(item) = s.next().await {
                reasons.push(item.unwrap().finish_reason);
            }
            assert!(reasons.iter().any(|r| r.as_deref() == Some("stop")));
        }

        // --- diagnostics (span collector singleton) -----------------------

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn stream_emits_ttfb_ttft_total_spans() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["Hel", "lo"])).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let names: BTreeSet<String> = get_span_collector()
                .all()
                .into_iter()
                .map(|r| r.name)
                .collect();
            assert!(names.contains("llm.ttfb.prose"));
            assert!(names.contains("llm.ttft.prose"));
            assert!(names.contains("llm.total.prose"));
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn stream_no_slot_falls_back_to_unsuffixed_spans() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["x"])).await;
            let c = client_for(&server.uri());
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let names: BTreeSet<String> = get_span_collector()
                .all()
                .into_iter()
                .map(|r| r.name)
                .collect();
            assert!(names.contains("llm.ttfb"));
            assert!(names.contains("llm.ttft"));
            assert!(names.contains("llm.total"));
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn stream_no_ttft_when_no_content() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&[])).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let names: BTreeSet<String> = get_span_collector()
                .all()
                .into_iter()
                .map(|r| r.name)
                .collect();
            assert!(names.contains("llm.ttfb.prose"));
            assert!(names.contains("llm.total.prose"));
            assert!(!names.contains("llm.ttft.prose"));
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn call_log_includes_chars_and_metadata() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            let usage =
                json!({ "prompt_tokens": 1234, "completion_tokens": 12, "total_tokens": 1246 });
            mount_sse(
                &server,
                sse_with_usage(&["Hi"], Some(usage), Some("openai")),
            )
            .await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let mut s = c
                .chat_stream(vec![
                    Message::new("system", "A".repeat(100)),
                    Message::new("user", "hello"),
                ])
                .await
                .unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let line = cap
                .records()
                .into_iter()
                .map(|r| strip_ansi(&r.message))
                .find(|m| m.contains("LLM call"))
                .expect("an [LLM call] line");
            assert!(line.contains("chars=105"), "line: {line}");
            assert!(line.contains("model=m"), "line: {line}");
            assert!(line.contains("slot=prose"), "line: {line}");
            assert!(line.contains("in_tokens=1234"), "line: {line}");
            assert!(line.contains("out_tokens=12"), "line: {line}");
            assert!(line.contains("provider=openai"), "line: {line}");
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn clean_completion_logs_ok() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            mount_sse(&server, sse_content(&["Hello"])).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let line = call_line(&cap);
            assert!(line.contains("status=ok"), "line: {line}");
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn consumer_break_logs_cancelled_not_error() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            let deltas: Vec<String> = (0..50).map(|i| format!("d{i}")).collect();
            let refs: Vec<&str> = deltas.iter().map(String::as_str).collect();
            mount_sse(&server, sse_content(&refs)).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let mut s = c.chat_stream(vec![user("hi")]).await.unwrap();
            let _ = s.next().await;
            drop(s); // abandon before terminal event → cancelled
            let line = call_line(&cap);
            assert!(line.contains("status=cancelled"), "line: {line}");
            assert!(!line.contains("status=error"), "line: {line}");
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn http_4xx_logs_error_status() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            mount_json(&server, 429, json!({})).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            assert!(c.stream_completion(vec![user("hi")], None).await.is_err());
            let line = call_line(&cap);
            assert!(line.contains("status=error"), "line: {line}");
        }

        #[allow(clippy::await_holding_lock)]
        #[tokio::test]
        async fn cached_tokens_surface_in_call_log() {
            let _g = singleton_guard();
            reset_span_collector();
            let server = MockServer::start().await;
            let usage = json!({
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": { "cached_tokens": 800 }
            });
            mount_sse(&server, sse_with_usage(&["x"], Some(usage), None)).await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .slot("prose")
                .build();
            let cap = Capture::default();
            let _sub = install_capture(&cap);
            let mut s = c.chat_stream(vec![user("x")]).await.unwrap();
            while s.next().await.is_some() {}
            drop(s);
            let line = call_line(&cap);
            assert!(line.contains("cached=800"), "line: {line}");
        }

        fn call_line(cap: &Capture) -> String {
            cap.records()
                .into_iter()
                .map(|r| strip_ansi(&r.message))
                .find(|m| m.contains("LLM call"))
                .expect("an [LLM call] line")
        }

        // --- create_llm_clients -------------------------------------------

        fn config_with_slots() -> CharacterConfig {
            let mut cfg = CharacterConfig::default();
            cfg.llm.insert(
                "fast".into(),
                LLMSlotConfig {
                    model: "m/fast".into(),
                    temperature: Some(0.4),
                    reasoning: Some("off".into()),
                    ..Default::default()
                },
            );
            cfg.llm.insert(
                "prose".into(),
                LLMSlotConfig {
                    model: "m/prose".into(),
                    temperature: Some(0.7),
                    reasoning: Some("medium".into()),
                    ..Default::default()
                },
            );
            cfg.llm.insert(
                "background".into(),
                LLMSlotConfig {
                    model: "m/bg".into(),
                    reasoning: Some("medium".into()),
                    ..Default::default()
                },
            );
            cfg
        }

        #[tokio::test]
        async fn create_clients_one_per_slot_shared_key() {
            let cfg = config_with_slots();
            let clients = create_llm_clients("sk-or-test-abc", &cfg).unwrap();
            let keys: BTreeSet<&str> = clients.keys().map(String::as_str).collect();
            assert_eq!(keys, BTreeSet::from(["fast", "prose", "background"]));
            for client in clients.values() {
                assert_eq!(client.api_key(), "sk-or-test-abc");
            }
        }

        #[tokio::test]
        async fn create_clients_thread_model_and_temperature() {
            let cfg = config_with_slots();
            let clients = create_llm_clients("sk", &cfg).unwrap();
            for (slot, client) in &clients {
                assert_eq!(client.model(), cfg.llm[slot].model);
                assert_eq!(client.temperature(), cfg.llm[slot].temperature);
            }
        }

        #[tokio::test]
        async fn create_clients_thread_reasoning() {
            let cfg = config_with_slots();
            let clients = create_llm_clients("sk", &cfg).unwrap();
            assert_eq!(clients["fast"].reasoning(), Some("off"));
            assert_eq!(clients["prose"].reasoning(), Some("medium"));
            assert_eq!(clients["background"].reasoning(), Some("medium"));
        }

        #[tokio::test]
        async fn create_clients_thread_sampling_and_think_prepend() {
            let mut cfg = config_with_slots();
            cfg.llm.insert(
                "fast".into(),
                LLMSlotConfig {
                    model: "qwen/qwen3.6-35b-a3b".into(),
                    top_p: Some(0.8),
                    top_k: Some(20),
                    presence_penalty: Some(1.5),
                    think_prepend: true,
                    ..Default::default()
                },
            );
            let clients = create_llm_clients("sk", &cfg).unwrap();
            let fast = &clients["fast"];
            assert_eq!(fast.top_p(), Some(0.8));
            assert_eq!(fast.top_k(), Some(20));
            assert_eq!(fast.presence_penalty(), Some(1.5));
            assert!(fast.think_prepend());
            assert_eq!(clients["prose"].top_p(), None);
            assert!(!clients["prose"].think_prepend());
        }

        #[tokio::test]
        async fn create_clients_thread_image_tools_and_multimodal() {
            let mut cfg = config_with_slots();
            cfg.llm.insert(
                "prose".into(),
                LLMSlotConfig {
                    model: "x/y".into(),
                    image_tools: true,
                    multimodal: true,
                    ..Default::default()
                },
            );
            let clients = create_llm_clients("sk", &cfg).unwrap();
            assert!(clients["prose"].image_tools_enabled());
            assert!(clients["prose"].multimodal());
        }

        #[tokio::test]
        async fn create_clients_builds_description_client_when_configured() {
            let mut cfg = config_with_slots();
            cfg.image_description_model = "openai/gpt-4o".into();
            let clients = create_llm_clients("sk", &cfg).unwrap();
            assert!(clients.contains_key("__image_description__"));
            assert_eq!(clients["__image_description__"].model(), "openai/gpt-4o");
        }

        #[tokio::test]
        async fn create_clients_no_description_client_by_default() {
            let cfg = config_with_slots();
            let clients = create_llm_clients("sk", &cfg).unwrap();
            assert!(!clients.contains_key("__image_description__"));
        }

        #[tokio::test]
        async fn create_clients_applies_shared_semaphore_cap() {
            let mut cfg = config_with_slots();
            cfg.llm_max_concurrent_requests = 6;
            let clients = create_llm_clients("sk", &cfg).unwrap();
            let fast = &clients["fast"];
            assert_eq!(fast.semaphore().available_permits(), 6);
            // One shared semaphore across every slot.
            assert!(Arc::ptr_eq(fast.semaphore(), clients["prose"].semaphore()));
            assert!(Arc::ptr_eq(
                fast.semaphore(),
                clients["background"].semaphore()
            ));
        }

        // --- no_stream synthesis ------------------------------------------

        #[tokio::test]
        async fn no_stream_synthesizes_content_and_finish() {
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{ "message": { "role": "assistant", "content": "hello there" } }]
                }),
            )
            .await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .no_stream(true)
                .build();
            let mut s = c.stream_completion(vec![user("hi")], None).await.unwrap();
            let mut deltas: Vec<LlmDelta> = Vec::new();
            while let Some(item) = s.next().await {
                deltas.push(item.unwrap());
            }
            assert_eq!(deltas[0].content, "hello there");
            assert_eq!(
                deltas.last().unwrap().finish_reason.as_deref(),
                Some("stop")
            );
        }

        #[tokio::test]
        async fn no_stream_synthesizes_tool_call_deltas() {
            let server = MockServer::start().await;
            mount_json(
                &server,
                200,
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": { "name": "set_alarm", "arguments": "{\"reason\":\"x\"}" }
                            }]
                        }
                    }]
                }),
            )
            .await;
            let c = OpenRouterClient::builder("k", "m")
                .base_url(server.uri())
                .no_stream(true)
                .build();
            let mut s = c.stream_completion(vec![user("hi")], None).await.unwrap();
            let mut deltas: Vec<LlmDelta> = Vec::new();
            while let Some(item) = s.next().await {
                deltas.push(item.unwrap());
            }
            let tc_delta = deltas.iter().find(|d| !d.tool_calls.is_empty()).unwrap();
            assert_eq!(tc_delta.tool_calls[0]["index"], 0);
            assert_eq!(tc_delta.tool_calls[0]["id"], "call_1");
            assert_eq!(tc_delta.tool_calls[0]["function"]["name"], "set_alarm");
            assert_eq!(
                deltas.last().unwrap().finish_reason.as_deref(),
                Some("tool_calls")
            );
        }

        // Silence dead-code for the `Content` import used only to assert shapes.
        #[test]
        fn content_import_is_live() {
            assert_eq!(Content::default(), Content::Text(String::new()));
        }
    }
}

#[cfg(feature = "net")]
pub use client::{
    OPENROUTER_BASE_URL, OpenRouterClient, OpenRouterClientBuilder, create_llm_clients,
};

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
