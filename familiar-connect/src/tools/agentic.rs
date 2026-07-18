//! Agentic tool-execution loop (subsystem 08; Python `tools/loop.py` — renamed
//! because `loop` is a Rust keyword, DESIGN D21).
//!
//! Drives [`LlmClient::stream_completion`] through one or more iterations: stream
//! deltas, accumulate `content` + `tool_calls`, execute tools, append results,
//! re-call. Terminates when the model returns no tool calls or `max_iterations`
//! is reached. A leak guard strips tool-calls the model occasionally emits as
//! plain text (they would otherwise ship to the user / seed a mimicry cascade);
//! a leaked `silent` call is honoured as silence.

use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};

use crate::llm::{Content, LlmClient, LlmDelta, Message};
use crate::log_style as ls;
use crate::silence::{LeadingLeak, classify_leading_leak};
use crate::tools::registry::{Tool, ToolContext, ToolOutput, ToolRegistry};
use crate::tools::silent::SILENT_RESULT;

// Re-exported here so callers can reach it at the Python `tools.loop` path.
pub use crate::tools::registry::serialize_image_result;

/// Library-default iteration cap (spec 08 §T15 / DESIGN D17).
pub const DEFAULT_MAX_ITERATIONS: usize = 5;

/// Outcome of an [`agentic_loop`] run.
#[derive(Debug, Clone)]
pub struct AgenticResult {
    /// Final assistant text (leak-guarded).
    pub final_content: String,
    /// Iterations run.
    pub iterations: usize,
    /// Total tool calls attempted.
    pub tool_calls_made: usize,
    /// The full transcript (a snapshot of `messages` at return).
    pub transcript: Vec<Message>,
    /// Set when a `silent` tool (or leaked `silent` call) ended the turn.
    pub is_silent: bool,
}

/// Streaming / iteration callbacks the responders (06) implement; defaults are
/// no-ops so tests and simple callers pass `None`.
#[async_trait]
pub trait AgenticHooks: Send + Sync {
    /// Awaited per streamed [`LlmDelta`] (responders stream to TTS/Discord).
    async fn on_delta(&self, _delta: &LlmDelta) {}
    /// Awaited after the assistant message is built but before any handler runs
    /// (voice filler-phrase hook), only when tool calls exist.
    async fn on_before_tools(&self, _assistant: &Message) {}
    /// Awaited once per non-silent iteration (responders persist intermediate
    /// turns here).
    async fn on_iteration_end(&self, _assistant: &Message, _tool_msgs: &[Message]) {}
}

// ---------------------------------------------------------------------------
// Leak-guard regexes (behavior-pinned; ported verbatim)
// ---------------------------------------------------------------------------

static INVOKE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<(?:\w+:)?invoke\b.*?</(?:\w+:)?invoke>").expect("valid regex")
});
static LEADING_THINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)^\s*(?:<think>.*?</think>|</think>)\s*").expect("valid regex")
});

/// Strip a leading think-tag artifact; reasoning never ships as text.
fn strip_think_artifacts(content: &str) -> String {
    LEADING_THINK_RE.replacen(content, 1, "").into_owned()
}

/// Strip a leaked tool invocation emitted as plain text.
///
/// Returns `(cleaned, silent_leak)`; `silent_leak` is true when the stripped
/// invocation named the `silent` tool. Only fires when content *leads* with an
/// invocation, so a stray mention mid-prose stays content. Detection is shared
/// with the streaming voice gate via [`classify_leading_leak`].
fn strip_leaked_tool_calls(content: &str) -> (String, bool) {
    match classify_leading_leak(content) {
        Some(LeadingLeak::Invoke { silent }) => {
            let cleaned = INVOKE_BLOCK_RE.replace_all(content, "").trim().to_owned();
            (cleaned, silent)
        }
        Some(LeadingLeak::PythonSilent) => (String::new(), true),
        Some(LeadingLeak::PythonTool) => (String::new(), false),
        None => (content.to_owned(), false),
    }
}

/// Belt-and-suspenders guard for a streamed reply that bypasses the return-time
/// [`AgenticResult::final_content`] path (the voice path streams `accumulated`
/// straight through). Strips a leading think artifact then a leaked tool-call
/// block so a leak never reaches TTS output or the persisted assistant turn;
/// genuine prose passes through unchanged.
#[must_use]
pub(crate) fn guard_leaked_content(content: &str) -> String {
    let stripped = strip_think_artifacts(content);
    strip_leaked_tool_calls(&stripped).0
}

/// Project a tool-message content to plain text for history persistence.
#[must_use]
pub fn tool_content_as_text(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    b.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

// ---------------------------------------------------------------------------
// Tool-call accumulation
// ---------------------------------------------------------------------------

struct PendingCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

fn accumulate_tool_calls(pending: &mut BTreeMap<i64, PendingCall>, fragments: &[Value]) {
    for frag in fragments {
        let idx = match frag.get("index") {
            None => 0,
            Some(v) => match v.as_i64() {
                Some(i) => i,
                None => continue,
            },
        };
        let bucket = pending.entry(idx).or_insert_with(|| PendingCall {
            id: String::new(),
            kind: "function".to_owned(),
            name: String::new(),
            arguments: String::new(),
        });
        if let Some(id) = frag.get("id").and_then(Value::as_str) {
            id.clone_into(&mut bucket.id);
        }
        if let Some(t) = frag.get("type").and_then(Value::as_str) {
            t.clone_into(&mut bucket.kind);
        }
        if let Some(f) = frag.get("function") {
            if f.is_object() {
                if let Some(name) = f.get("name").and_then(Value::as_str) {
                    if !name.is_empty() {
                        name.clone_into(&mut bucket.name);
                    }
                }
                if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                    bucket.arguments.push_str(a);
                }
            }
        }
    }
}

fn finalize_tool_calls(pending: &BTreeMap<i64, PendingCall>) -> Vec<Value> {
    pending
        .values()
        .filter(|c| !c.id.is_empty())
        .map(|c| {
            json!({
                "id": c.id,
                "type": c.kind,
                "function": {"name": c.name, "arguments": c.arguments},
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

/// Render an `f64` the way Python's `str(float)` does, so the timeout wire
/// string matches the reference byte-for-byte: an integer-valued float keeps a
/// trailing `.0` (`"10.0"`, not Rust's default `"10"`), while a fractional
/// value uses the shortest round-tripping form (`"0.05"`). Scoped to the small
/// positive timeouts tools carry; the extreme magnitudes where Python switches
/// to exponent notation are unreachable here.
fn py_float_str(v: f64) -> String {
    let s = format!("{v}");
    if v.is_finite() && !s.contains(['.', 'e', 'E']) {
        format!("{s}.0")
    } else {
        s
    }
}

fn tool_error_msg(call_id: &str, msg: &str) -> Message {
    Message {
        role: "tool".to_owned(),
        content: Content::Text(json!({ "error": msg }).to_string()),
        name: None,
        tool_calls: None,
        tool_call_id: Some(call_id.to_owned()),
    }
}

async fn execute_tool(tool: &Tool, args: Value, ctx: &ToolContext) -> ToolOutput {
    match tokio::time::timeout(
        Duration::from_secs_f64(tool.timeout_s),
        tool.handler.call(args, ctx),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => ToolOutput::Text(json!({ "error": format!("{e}") }).to_string()),
        Err(_) => ToolOutput::Text(
            json!({ "error": format!("timeout after {}s", py_float_str(tool.timeout_s)) })
                .to_string(),
        ),
    }
}

async fn run_tool_call(
    tc: &Value,
    registry: &ToolRegistry,
    ctx: &ToolContext,
    multimodal: bool,
) -> Message {
    let call_id = tc.get("id").and_then(Value::as_str).unwrap_or("");
    let function = tc.get("function");
    let name = function
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let raw_args = function
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let raw_args = if raw_args.is_empty() { "{}" } else { raw_args };

    let decoded: Value = if raw_args.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str::<Value>(raw_args) {
            Ok(v) => v,
            Err(e) => return tool_error_msg(call_id, &format!("invalid arguments JSON: {e}")),
        }
    };
    if !decoded.is_object() {
        return tool_error_msg(call_id, "invalid arguments JSON: not a JSON object");
    }

    let Some(tool) = registry.get(name) else {
        return tool_error_msg(call_id, &format!("unknown tool: {name}"));
    };

    let output = execute_tool(tool, decoded, ctx).await;
    let content = match output {
        ToolOutput::Text(s) => Content::Text(s),
        ToolOutput::Image(img) => serialize_image_result(&img, multimodal),
    };
    Message {
        role: "tool".to_owned(),
        content,
        name: None,
        tool_calls: None,
        tool_call_id: Some(call_id.to_owned()),
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// Run streaming + tool execution until the model stops calling tools.
///
/// `messages` is mutated in place (assistant + tool turns appended) and a
/// snapshot is returned as [`AgenticResult::transcript`]. `hooks` is an optional
/// callback bundle; `None` for the no-op case.
///
/// # Errors
/// Propagates stream-open / mid-stream errors from `stream_completion`; handler
/// faults are captured as `{"error": ...}` tool results (never `Err`).
#[allow(
    clippy::too_many_lines,
    reason = "single cohesive stream→tools→recall loop"
)]
pub async fn agentic_loop(
    llm: &dyn LlmClient,
    messages: &mut Vec<Message>,
    registry: &ToolRegistry,
    ctx: &ToolContext,
    hooks: Option<&dyn AgenticHooks>,
    max_iterations: usize,
) -> anyhow::Result<AgenticResult> {
    let tools_payload: Option<Vec<Value>> = if registry.is_empty() {
        None
    } else {
        Some(registry.as_openai_tools())
    };
    let mut last_content = String::new();
    let mut iterations = 0usize;
    let mut tool_calls_made = 0usize;

    while iterations < max_iterations {
        iterations += 1;
        let mut content_buf = String::new();
        let mut pending: BTreeMap<i64, PendingCall> = BTreeMap::new();

        let mut stream = llm
            .stream_completion(messages.clone(), tools_payload.clone())
            .await?;
        while let Some(item) = stream.next().await {
            let delta = item?;
            if !delta.content.is_empty() {
                content_buf.push_str(&delta.content);
            }
            if !delta.tool_calls.is_empty() {
                accumulate_tool_calls(&mut pending, &delta.tool_calls);
            }
            if let Some(h) = hooks {
                h.on_delta(&delta).await;
            }
        }

        let content = content_buf;
        let tool_calls = finalize_tool_calls(&pending);
        let assistant_msg = Message {
            role: "assistant".to_owned(),
            content: Content::Text(content.clone()),
            name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
            tool_call_id: None,
        };
        messages.push(assistant_msg.clone());

        if !tool_calls.is_empty() {
            if let Some(h) = hooks {
                h.on_before_tools(&assistant_msg).await;
            }
        }

        let mut tool_msgs: Vec<Message> = Vec::new();
        let multimodal = llm.multimodal();
        for tc in &tool_calls {
            tool_calls_made += 1;
            let tool_msg = run_tool_call(tc, registry, ctx, multimodal).await;
            messages.push(tool_msg.clone());
            tool_msgs.push(tool_msg);
        }

        // Detect the silent sentinel BEFORE on_iteration_end so the call + its
        // reasoning aren't persisted to history (which would re-seed the model's
        // rationale for silence next turn).
        if tool_msgs
            .iter()
            .any(|m| matches!(&m.content, Content::Text(s) if s == SILENT_RESULT))
        {
            return Ok(AgenticResult {
                final_content: String::new(),
                iterations,
                tool_calls_made,
                transcript: messages.clone(),
                is_silent: true,
            });
        }

        if let Some(h) = hooks {
            h.on_iteration_end(&assistant_msg, &tool_msgs).await;
        }

        last_content = content;
        if tool_calls.is_empty() {
            break;
        }
        if iterations >= max_iterations {
            tracing::warn!(
                "{} {}",
                ls::tag("Tools", ls::LY),
                ls::kv_styled(
                    "hit_max_iterations",
                    &max_iterations.to_string(),
                    ls::W,
                    ls::LY
                ),
            );
            break;
        }
    }

    // Guard the final content: strip a leading think artifact FIRST (so it can't
    // mask a leaked call behind it), then a leaked tool call.
    let stripped = strip_think_artifacts(&last_content);
    let (cleaned, silent_leak) = strip_leaked_tool_calls(&stripped);
    if cleaned != last_content {
        tracing::warn!(
            "{} {} {}",
            ls::tag("Tools", ls::LY),
            ls::kv_styled("leaked_tool_call_stripped", "true", ls::W, ls::LY),
            ls::kv_styled("silent", &silent_leak.to_string(), ls::W, ls::LY),
        );
    }
    let is_silent = silent_leak && cleaned.is_empty();
    Ok(AgenticResult {
        final_content: cleaned,
        iterations,
        tool_calls_made,
        transcript: messages.clone(),
        is_silent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_str_matches_python_str_float() {
        // Integer-valued floats keep the trailing `.0` (Python `str(10.0)`),
        // where Rust's default `{}` would drop it to `"10"`.
        assert_eq!(py_float_str(10.0), "10.0");
        assert_eq!(py_float_str(30.0), "30.0");
        // Fractional values use the shortest round-tripping form, matching
        // Python for the realistic timeout range.
        assert_eq!(py_float_str(0.05), "0.05");
        assert_eq!(py_float_str(0.5), "0.5");
    }

    #[test]
    fn strip_think_bare_closing_tag() {
        assert_eq!(strip_think_artifacts("</think>"), "");
    }

    #[test]
    fn strip_think_keeps_following_text() {
        assert_eq!(strip_think_artifacts("</think>\nUmu."), "Umu.");
    }

    #[test]
    fn strip_think_full_block() {
        assert_eq!(
            strip_think_artifacts("<think>\nweigh\n</think>\nUmu."),
            "Umu."
        );
    }

    #[test]
    fn strip_think_mid_prose_untouched() {
        let t = "I think </think> is a vulgar rune.";
        assert_eq!(strip_think_artifacts(t), t);
    }

    #[test]
    fn leak_silent_invoke_is_silent() {
        let leak =
            "<invoke name=\"silent\">\n<parameter name=\"reasoning\">quiet</parameter>\n</invoke>";
        let (cleaned, silent) = strip_leaked_tool_calls(leak);
        assert!(silent);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn leak_namespaced_invoke_stripped() {
        let leak = "<ns:invoke name=\"silent\">x</ns:invoke>";
        let (cleaned, silent) = strip_leaked_tool_calls(leak);
        assert!(silent);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn leak_nonsilent_invoke_stripped_not_silent() {
        let leak = "<invoke name=\"view_image\">x</invoke>";
        let (cleaned, silent) = strip_leaked_tool_calls(leak);
        assert!(!silent);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn leak_python_silent_case_insensitive() {
        let (cleaned, silent) = strip_leaked_tool_calls("Silent(reasoning=\"x\")");
        assert!(silent);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn leak_python_read_channel_not_silent() {
        let (cleaned, silent) = strip_leaked_tool_calls("read_channel(limit=10)");
        assert!(!silent);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn word_invoke_mid_prose_untouched() {
        let t = "Let me invoke my legendary wit.";
        let (cleaned, silent) = strip_leaked_tool_calls(t);
        assert!(!silent);
        assert_eq!(cleaned, t);
    }

    #[test]
    fn tool_content_as_text_passthrough_and_blocks() {
        assert_eq!(
            tool_content_as_text(&Content::Text("hello".into())),
            "hello"
        );
        let blocks = Content::Blocks(vec![
            json!({"type": "text", "text": "a cat"}),
            json!({"type": "image_url", "image_url": {"url": "data:..."}}),
        ]);
        assert_eq!(tool_content_as_text(&blocks), "a cat");
    }
}
