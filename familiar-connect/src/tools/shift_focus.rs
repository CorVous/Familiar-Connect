//! `shift_focus` tool (subsystem 08).
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

/// Resolve the `channel_id` argument to a subscribed channel id.
///
/// Discord snowflakes exceed 2^53, and some providers (Gemini) round a JSON
/// *number* argument through a float64 on the way here — `1257165677052170301`
/// arrives as `1257165677052170200`. So the schema asks for a digit **string**,
/// which is taken exactly. A number is still accepted: used as-is when it names
/// a subscribed channel, otherwise snapped to the one subscribed channel that
/// rounds to the same float64 (ambiguous or no match ⇒ left as given, and the
/// caller's not-subscribed error lists the valid ids).
///
/// `None` ⇒ missing or unparseable.
fn resolve_channel_id(arg: Option<&Value>, subscribed: &[i64]) -> Option<i64> {
    match arg? {
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Number(n) => {
            let exact = n.as_i64();
            if let Some(id) = exact {
                if subscribed.contains(&id) {
                    return Some(id);
                }
            }
            let approx = n.as_f64()?;
            #[allow(
                clippy::cast_precision_loss,
                clippy::float_cmp,
                reason = "deliberately reproducing the provider's float64 rounding"
            )]
            let mut matches = subscribed.iter().copied().filter(|c| *c as f64 == approx);
            match (matches.next(), matches.next()) {
                (Some(only), None) => Some(only),
                _ => exact,
            }
        }
        _ => None,
    }
}

/// Bare/preview ack shape for a successful shift. `channel_id` is a string so
/// the id the model reads back is the id it can pass in again.
#[derive(Serialize)]
struct ShiftFocusOk {
    ok: bool,
    channel_id: String,
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

    let subscribed = fm.subscribed_channels();
    let Some(channel_id) = resolve_channel_id(args.get("channel_id"), &subscribed) else {
        return Ok(error_output(
            "missing or invalid 'channel_id' (the channel's id digits, as a string)",
        ));
    };

    if !fm.is_subscribed(channel_id) {
        let available: Vec<Value> = subscribed
            .into_iter()
            .map(|cid| json!({"channel_id": cid.to_string(), "label": fm.channel_label(cid)}))
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
        channel_id: channel_id.to_string(),
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
                    "type": "string",
                    "description":
                        "Discord channel id to focus on: its digits, copied exactly, as a string",
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
        assert_eq!(props["channel_id"]["type"], "string");
        assert_eq!(tool.parameters["required"], json!(["channel_id"]));
    }

    const DM: i64 = 1_257_165_677_052_170_301;
    const GENERAL: i64 = 324_917_760_578_682_880;

    #[test]
    fn resolve_takes_a_digit_string_exactly() {
        let subs = [DM, GENERAL];
        let arg = json!("1257165677052170301");
        assert_eq!(resolve_channel_id(Some(&arg), &subs), Some(DM));
        let padded = json!(" 324917760578682880 ");
        assert_eq!(resolve_channel_id(Some(&padded), &subs), Some(GENERAL));
        // A string is never snapped: an unknown id stays unknown.
        let rounded = json!("1257165677052170200");
        assert_eq!(
            resolve_channel_id(Some(&rounded), &subs),
            Some(1_257_165_677_052_170_200)
        );
    }

    #[test]
    fn resolve_snaps_a_float_rounded_number_to_the_subscribed_channel() {
        let subs = [DM, GENERAL];
        // What Gemini actually sent for the two channels above.
        let dm = json!(1_257_165_677_052_170_200_i64);
        assert_eq!(resolve_channel_id(Some(&dm), &subs), Some(DM));
        let general = json!(324_917_760_578_682_900_i64);
        assert_eq!(resolve_channel_id(Some(&general), &subs), Some(GENERAL));
        // Float-typed numbers snap too.
        let float = json!(1.257_165_677_052_170_2e18);
        assert_eq!(resolve_channel_id(Some(&float), &subs), Some(DM));
    }

    #[test]
    fn resolve_keeps_exact_numbers_and_refuses_ambiguous_snaps() {
        // Exact subscribed number wins even when a float-neighbour is subscribed.
        let subs = [DM, DM - 1];
        assert_eq!(resolve_channel_id(Some(&json!(DM)), &subs), Some(DM));
        // Two subscribed ids round to the same float: no guess.
        let rounded = json!(1_257_165_677_052_170_200_i64);
        assert_eq!(
            resolve_channel_id(Some(&rounded), &subs),
            Some(1_257_165_677_052_170_200)
        );
        // Small ids and unknown numbers pass through untouched.
        assert_eq!(resolve_channel_id(Some(&json!(55)), &[55, 66]), Some(55));
        assert_eq!(resolve_channel_id(Some(&json!(77)), &[55, 66]), Some(77));
    }

    #[test]
    fn resolve_rejects_missing_and_non_id_values() {
        let subs = [DM];
        assert_eq!(resolve_channel_id(None, &subs), None);
        for bad in [
            json!(null),
            json!(true),
            json!("general"),
            json!(""),
            json!([DM]),
        ] {
            assert_eq!(resolve_channel_id(Some(&bad), &subs), None, "{bad}");
        }
    }
}
