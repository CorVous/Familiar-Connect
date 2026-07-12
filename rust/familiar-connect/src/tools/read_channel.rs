//! `read_channel` tool (subsystem 08; Python `tools/read_channel.py`).
//!
//! Read-only peek into the focused text channel history — returns recent turns
//! without touching `consumed_at`. Voice not supported. Deliberately unfiltered
//! by the activities archive watermark (fresh eyes may scroll archived past).

use std::sync::Arc;

use serde_json::{Value, json};

use crate::log_style as ls;
use crate::tools::channel_view::serialize_turns;
use crate::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput};

const DEFAULT_LIMIT: i64 = 20;
const MAX_LIMIT: i64 = 50;

fn error_output(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

/// The `read_channel` tool handler.
///
/// # Errors
/// Propagates a store error from the underlying paging query.
pub async fn read_channel_handler(args: &Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput> {
    let Some(fm) = ctx.focus_manager.as_ref() else {
        return Ok(error_output("focus_manager not wired into context"));
    };
    let Some(store) = ctx.store.as_ref() else {
        return Ok(error_output("store not wired into context"));
    };
    let Some(channel_id) = fm.get_focus("text") else {
        return Ok(error_output("no text focus active"));
    };

    let limit = args.get("limit").map_or(DEFAULT_LIMIT, |v| {
        v.as_i64().map_or(DEFAULT_LIMIT, |n| n.min(MAX_LIMIT))
    });

    let before_present = args.get("before_id").is_some_and(|v| !v.is_null());
    let around_present = args.get("around_id").is_some_and(|v| !v.is_null());
    if before_present && around_present {
        return Ok(error_output(
            "before_id and around_id are mutually exclusive",
        ));
    }

    let turns = if around_present {
        let around_id = args.get("around_id").and_then(Value::as_i64).unwrap_or(0);
        let half = (limit / 2).max(1);
        store
            .turns_around(&ctx.familiar_id, channel_id, around_id, half, half)
            .await?
    } else {
        let before_id = args.get("before_id").and_then(Value::as_i64);
        store
            .recent(&ctx.familiar_id, channel_id, limit, before_id)
            .await?
    };

    let views = serialize_turns(&turns);
    tracing::info!(
        "{} {} {}",
        ls::tag("\u{1f4d6} read_channel", ls::LM),
        ls::kv_styled("channel", &fm.channel_label(channel_id), ls::W, ls::LW),
        ls::kv_styled("turns", &views.len().to_string(), ls::W, ls::LW),
    );
    Ok(ToolOutput::Text(serde_json::to_string(&views)?))
}

/// Build the `read_channel` tool.
#[must_use]
pub fn build_read_channel_tool() -> Tool {
    Tool::new(
        "read_channel",
        "Read recent turns from the currently focused text channel. Page back with \
         before_id, or jump to a turn with around_id. Read-only; does not consume or \
         acknowledge messages. Voice focus not supported.",
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": DEFAULT_LIMIT,
                    "description": "Number of turns to return (max 50).",
                },
                "before_id": {
                    "type": "integer",
                    "description": "Return turns with id < before_id (for paging).",
                },
                "around_id": {
                    "type": "integer",
                    "description":
                        "Jump to this turn id; returns surrounding turns. Cannot combine with before_id.",
                },
            },
            "required": [],
        }),
        Arc::new(FnHandler(|args: Value, ctx: ToolContext| async move {
            read_channel_handler(&args, &ctx).await
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_read_channel_tool_name() {
        assert_eq!(build_read_channel_tool().name, "read_channel");
    }

    #[test]
    fn build_read_channel_tool_schema_has_limit() {
        let tool = build_read_channel_tool();
        assert_eq!(tool.parameters["properties"]["limit"]["maximum"], 50);
    }

    #[test]
    fn build_read_channel_tool_schema_has_paging_params() {
        let tool = build_read_channel_tool();
        let props = &tool.parameters["properties"];
        assert!(props.get("before_id").is_some());
        assert_eq!(props["around_id"]["type"], "integer");
    }
}
