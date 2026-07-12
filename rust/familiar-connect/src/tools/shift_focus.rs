//! `shift_focus` tool (subsystem 08; Python `tools/shift_focus.py`).
//!
//! Immediate focus shift: applies at handler time via
//! [`FocusControl::shift_now`] (modality inferred from subscriptions). No
//! deferral — focus moves the moment the tool is called, so a silent turn still
//! leaves her where she went and nothing leaks. Content-bearing: when a store is
//! wired the handler eagerly returns the target channel's recent turns (preview
//! size == the catch-up window it promotes), so the model sees the channel
//! in-turn.

use std::sync::Arc;

use serde::Serialize;
use serde_json::{Value, json};

use crate::log_style as ls;
use crate::tools::channel_view::{TurnView, serialize_turns};
use crate::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput};

fn error_output(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

/// Bare/preview ack shape for a successful shift.
#[derive(Serialize)]
struct ShiftFocusOk {
    ok: bool,
    channel_id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<TurnView>>,
}

/// The `shift_focus` tool handler.
///
/// # Errors
/// Propagates a store error from the eager recent-turns fetch.
pub async fn shift_focus_handler(args: &Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput> {
    let Some(fm) = ctx.focus_manager.as_ref() else {
        return Ok(error_output("focus_manager not wired into context"));
    };

    let Some(channel_id) = args.get("channel_id").and_then(Value::as_i64) else {
        return Ok(error_output("missing or invalid 'channel_id' (integer)"));
    };

    if !fm.is_subscribed(channel_id) {
        let available: Vec<Value> = fm
            .subscribed_channels()
            .into_iter()
            .map(|cid| json!({"channel_id": cid, "label": fm.channel_label(cid)}))
            .collect();
        tracing::info!(
            "{} rejected {} (not subscribed)",
            ls::tag("\u{1f500} shift_focus", ls::LC),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LW),
        );
        return Ok(ToolOutput::Text(
            json!({
                "error": format!(
                    "channel {channel_id} is not subscribed — cannot focus there. \
                     Pick one of available_channels."
                ),
                "available_channels": available,
            })
            .to_string(),
        ));
    }

    fm.shift_now(channel_id).await;

    let messages = if let Some(store) = ctx.store.as_ref() {
        let limit = i64::try_from(fm.catch_up_limit()).unwrap_or(i64::MAX);
        let turns = store
            .recent(&ctx.familiar_id, channel_id, limit, None)
            .await?;
        Some(serialize_turns(&turns))
    } else {
        None
    };

    tracing::info!(
        "{} {} {}",
        ls::tag("\u{1f500} shift_focus", ls::LC),
        ls::kv_styled("channel", &fm.channel_label(channel_id), ls::W, ls::LW),
        ls::kv_styled(
            "preview",
            &messages.as_ref().map_or(0, Vec::len).to_string(),
            ls::W,
            ls::LW
        ),
    );

    let resp = ShiftFocusOk {
        ok: true,
        channel_id,
        messages,
    };
    Ok(ToolOutput::Text(serde_json::to_string(&resp)?))
}

/// Build the `shift_focus` tool.
#[must_use]
pub fn build_shift_focus_tool() -> Tool {
    Tool::new(
        "shift_focus",
        "Move your attention to a different channel — for real, right now. You \
         stop following your current channel until you shift back, and any reply \
         this turn posts to the new channel. Returns the target's recent messages \
         (empty list = nothing there yet). Use it to actually go somewhere, not to \
         glance. Modality (text/voice) inferred from channel subscription.",
        json!({
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "integer",
                    "description": "Discord channel id to focus on",
                },
            },
            "required": ["channel_id"],
        }),
        Arc::new(FnHandler(|args: Value, ctx: ToolContext| async move {
            shift_focus_handler(&args, &ctx).await
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_shift_focus_tool_name() {
        assert_eq!(build_shift_focus_tool().name, "shift_focus");
    }

    #[test]
    fn build_shift_focus_tool_schema() {
        let tool = build_shift_focus_tool();
        let props = &tool.parameters["properties"];
        assert_eq!(props["channel_id"]["type"], "integer");
        assert_eq!(tool.parameters["required"], json!(["channel_id"]));
    }
}
