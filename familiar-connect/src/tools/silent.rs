//! `silent` tool + the [`SILENT_RESULT`] sentinel (subsystem 08; Python
//! `tools/silent.py`).
//!
//! Suppresses the familiar's reply for a turn. The agentic loop detects the
//! sentinel among a turn's tool results and returns
//! `AgenticResult { is_silent: true, .. }` without re-prompting the model. The
//! tool itself just logs the reasoning and returns the sentinel — even with
//! empty/missing reasoning.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::log_style as ls;
use crate::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput};

/// Sentinel returned by the `silent` tool; the loop treats a tool message whose
/// string content equals this as a silence signal.
pub const SILENT_RESULT: &str = "__SILENT__";

/// The `silent` tool handler: log the reasoning, return the sentinel.
#[must_use]
pub fn silent_handler(args: &Value, _ctx: &ToolContext) -> ToolOutput {
    let reasoning = args.get("reasoning").and_then(Value::as_str).unwrap_or("");
    tracing::info!(
        "{} {}",
        ls::tag("\u{1f4a4} silent", ls::B),
        ls::kv_styled("reason", reasoning, ls::W, ls::LB),
    );
    ToolOutput::Text(SILENT_RESULT.to_owned())
}

/// Build the `silent` tool.
#[must_use]
pub fn build_silent_tool() -> Tool {
    Tool::new(
        "silent",
        "Stay completely silent — send no reply to the channel. Use when the \
         conversation is not aimed at you and you have no stake. The reasoning \
         argument is a private internal note — never shown in chat.",
        json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Private internal note — never shown in chat.",
                },
            },
            "required": ["reasoning"],
        }),
        Arc::new(FnHandler(|args: Value, ctx: ToolContext| async move {
            Ok(silent_handler(&args, &ctx))
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext::new("fam-1", 42, "text", "turn-1")
    }

    #[test]
    fn returns_silent_result_sentinel() {
        let out = silent_handler(&json!({"reasoning": "not relevant"}), &ctx());
        assert_eq!(out, ToolOutput::Text(SILENT_RESULT.to_owned()));
    }

    #[test]
    fn returns_sentinel_even_with_empty_reasoning() {
        let out = silent_handler(&json!({"reasoning": ""}), &ctx());
        assert_eq!(out, ToolOutput::Text(SILENT_RESULT.to_owned()));
    }

    #[test]
    fn returns_sentinel_even_with_missing_reasoning() {
        let out = silent_handler(&json!({}), &ctx());
        assert_eq!(out, ToolOutput::Text(SILENT_RESULT.to_owned()));
    }

    #[test]
    fn build_silent_tool_name_and_schema() {
        let tool = build_silent_tool();
        assert_eq!(tool.name, "silent");
        let props = &tool.parameters["properties"];
        assert_eq!(props["reasoning"]["type"], "string");
        assert_eq!(tool.parameters["required"], json!(["reasoning"]));
    }
}
