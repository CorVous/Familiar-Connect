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

/// Longest echo of an unusable `channel_id` in the refusal message.
const GIVEN_ID_ECHO_LIMIT: usize = 80;

fn error_output(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

/// Shortest digit run that is taken for a channel id inside a longer string
/// (`<#123…>`, `#general(123…)`). Every Discord snowflake has at least 17
/// digits; a shorter run is part of a name, or a float's 16-digit mantissa.
const MIN_EMBEDDED_ID_DIGITS: usize = 17;

/// Below 2^53 a float64 holds every integer exactly, so nothing gets rounded.
const FLOAT_EXACT_LIMIT: f64 = 9_007_199_254_740_992.0;

/// Whether `id` is `approx` once both are float64s.
///
/// Past 2^53, where rounding happens, one ULP of slack is allowed: it covers
/// `serde_json` (built without `float_roundtrip`) parsing an exponent-form
/// number one ULP off. At snowflake magnitude that is a few hundred ids — still
/// inside one Discord millisecond. Below 2^53 the match must be exact.
fn same_float(id: i64, approx: f64) -> bool {
    if id <= 0 || !approx.is_finite() || approx <= 0.0 {
        return false;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "deliberately reproducing the provider's float64 rounding"
    )]
    let as_float = id as f64;
    let slack = u64::from(approx >= FLOAT_EXACT_LIMIT);
    as_float.to_bits().abs_diff(approx.to_bits()) <= slack
}

/// Read the model's `channel_id` as `(exact id, float64 value)`, generously:
/// a digit string, a number, an id wrapped in other text (`<#id>`, a label), or
/// a float spelled out in a string.
fn read_channel_arg(arg: &Value) -> (Option<i64>, Option<f64>) {
    match arg {
        Value::Number(n) => (n.as_i64(), n.as_f64()),
        Value::String(s) => {
            let text = s.trim();
            if let Ok(id) = text.parse::<i64>() {
                return (Some(id), None);
            }
            // A float spelled out ("1.25e18") before the embedded-run scan,
            // which would otherwise lift its mantissa digits as an id.
            if let Ok(float) = text.parse::<f64>() {
                return (None, Some(float));
            }
            let mut runs = text
                .split(|c: char| !c.is_ascii_digit())
                .filter(|run| run.len() >= MIN_EMBEDDED_ID_DIGITS);
            match (runs.next(), runs.next()) {
                (Some(run), None) => (run.parse::<i64>().ok(), None),
                _ => (None, None),
            }
        }
        _ => (None, None),
    }
}

/// Resolve the `channel_id` argument against the subscribed channels.
///
/// Discord snowflakes exceed 2^53, and some providers (Gemini) round a JSON
/// *number* argument through a float64 on the way here — `1257165677052170301`
/// arrives as `1257165677052170200`. So the schema asks for a digit **string**.
/// Numbers are still accepted, and rounded ids also linger as text in replayed
/// tool results, so every form goes through the same two steps:
///
/// 1. an exact id that is subscribed wins;
/// 2. otherwise the id snaps to the **one** subscribed channel with the same
///    float64 value ([`same_float`]) — two candidates is a refusal, not a guess.
///
/// Failing both, the exact id (if any) is returned as given so the caller can
/// name it in the not-subscribed error. `None` ⇒ no usable id at all.
fn resolve_channel_id(arg: &Value, subscribed: &[i64]) -> Option<i64> {
    let (exact, approx) = read_channel_arg(arg);
    if let Some(id) = exact {
        if subscribed.contains(&id) {
            return Some(id);
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "deliberately reproducing the provider's float64 rounding"
    )]
    let target = approx.or_else(|| exact.map(|id| id as f64))?;
    let mut matches = subscribed
        .iter()
        .copied()
        .filter(|id| same_float(*id, target));
    match (matches.next(), matches.next()) {
        (Some(only), None) => Some(only),
        _ => exact,
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
    let arg = args.get("channel_id").filter(|v| !v.is_null());
    let resolved = arg.and_then(|v| resolve_channel_id(v, &subscribed));
    let channel_id = match resolved {
        Some(id) if fm.is_subscribed(id) => id,
        _ => {
            // Every refusal carries the valid ids: for a quiet channel this
            // list is the main place the model can read one.
            let available: Vec<Value> = subscribed
                .into_iter()
                .map(|cid| json!({"channel_id": cid.to_string(), "label": fm.channel_label(cid)}))
                .collect();
            let given = arg.map(|v| match (resolved, v) {
                (Some(id), _) => id.to_string(),
                (None, Value::String(s)) => ls::trunc(s, GIVEN_ID_ECHO_LIMIT),
                (None, other) => ls::trunc(&other.to_string(), GIVEN_ID_ECHO_LIMIT),
            });
            tracing::info!(
                "{} rejected {} (not subscribed)",
                ls::tag("\u{1f500} shift_focus", ls::LC),
                ls::kv_styled(
                    "channel",
                    given.as_deref().unwrap_or("<missing>"),
                    ls::W,
                    ls::LW
                ),
            );
            let error = given.map_or_else(
                || {
                    "missing 'channel_id' — pass the channel_id of one of \
                     available_channels, as a string."
                        .to_owned()
                },
                |given| {
                    if resolved.is_some() {
                        format!(
                            "channel {given} is not subscribed — cannot focus there. \
                             Pick one of available_channels."
                        )
                    } else {
                        format!(
                            "could not read a channel id from '{given}' — pass the \
                             channel_id of one of available_channels, as a string."
                        )
                    }
                },
            );
            return Ok(ToolOutput::Text(
                json!({ "error": error, "available_channels": available }).to_string(),
            ));
        }
    };

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
    /// What Gemini sent for [`DM`] and [`GENERAL`].
    const DM_ROUNDED: i64 = 1_257_165_677_052_170_200;
    const GENERAL_ROUNDED: i64 = 324_917_760_578_682_900;
    /// An id whose exponent form `serde_json` parses one ULP off.
    const OFF_BY_ULP: i64 = 1_185_233_071_271_705_465;

    fn resolve(arg: &Value, subscribed: &[i64]) -> Option<i64> {
        resolve_channel_id(arg, subscribed)
    }

    /// The model's arguments as they really arrive: raw JSON text, parsed.
    fn parsed_arg(raw: &str) -> Value {
        let args: Value = serde_json::from_str(raw).unwrap();
        args["channel_id"].clone()
    }

    #[test]
    fn resolve_takes_a_digit_string_exactly() {
        let subs = [DM, GENERAL];
        assert_eq!(resolve(&json!("1257165677052170301"), &subs), Some(DM));
        assert_eq!(
            resolve(&json!(" 324917760578682880 "), &subs),
            Some(GENERAL)
        );
        assert_eq!(resolve(&json!("55"), &[55, 66]), Some(55));
    }

    #[test]
    fn resolve_finds_the_id_inside_a_mention_or_label() {
        let subs = [DM, GENERAL];
        assert_eq!(resolve(&json!("<#1257165677052170301>"), &subs), Some(DM));
        assert_eq!(
            resolve(&json!("#general2(324917760578682880)"), &subs),
            Some(GENERAL)
        );
        // Two id-length runs: no way to tell which was meant.
        let both = json!("1257165677052170301 324917760578682880");
        assert_eq!(resolve(&both, &subs), None);
    }

    #[test]
    fn resolve_snaps_a_float_rounded_number_to_the_subscribed_channel() {
        let subs = [DM, GENERAL];
        assert_eq!(resolve(&json!(DM_ROUNDED), &subs), Some(DM));
        assert_eq!(resolve(&json!(GENERAL_ROUNDED), &subs), Some(GENERAL));
        // The rounded digits copied back as a string (they sit in old tool
        // results) and a float spelled out in a string snap the same way.
        assert_eq!(resolve(&json!("1257165677052170200"), &subs), Some(DM));
        assert_eq!(resolve(&json!("1.2571656770521702e18"), &subs), Some(DM));
    }

    #[test]
    fn resolve_snaps_numbers_parsed_from_raw_json_text() {
        let subs = [DM, GENERAL];
        let int = parsed_arg(r#"{"channel_id":1257165677052170200}"#);
        assert_eq!(resolve(&int, &subs), Some(DM));
        let float = parsed_arg(r#"{"channel_id":1.2571656770521702e18}"#);
        assert_eq!(resolve(&float, &subs), Some(DM));
        // serde_json parses this exponent form one ULP below the id's float64.
        let float = parsed_arg(r#"{"channel_id":1.1852330712717053e18}"#);
        assert_eq!(resolve(&float, &[OFF_BY_ULP, GENERAL]), Some(OFF_BY_ULP));
    }

    #[test]
    fn resolve_keeps_exact_ids_and_refuses_ambiguous_snaps() {
        // An exact subscribed id wins even when a float-neighbour is subscribed.
        let subs = [DM, DM - 1];
        assert_eq!(resolve(&json!(DM), &subs), Some(DM));
        assert_eq!(resolve(&json!("1257165677052170300"), &subs), Some(DM - 1));
        // Two subscribed ids share the rounded float: no guess, id as given.
        assert_eq!(resolve(&json!(DM_ROUNDED), &subs), Some(DM_ROUNDED));
        assert_eq!(
            resolve(&json!("1.2571656770521702e18"), &subs),
            None,
            "an ambiguous float has no exact id to fall back on"
        );
        // Small and unknown ids pass through untouched.
        assert_eq!(resolve(&json!(55), &[55, 66]), Some(55));
        assert_eq!(resolve(&json!(77), &[55, 66]), Some(77));
        assert_eq!(resolve(&json!(-5), &[55, 66]), Some(-5));
        // Below 2^53 nothing is rounded, so a near-miss float is not a match.
        assert_eq!(resolve(&json!(54.999_999_999_999_99), &[55, 66]), None);
        assert_eq!(resolve(&json!(55.0), &[55, 66]), Some(55));
    }

    #[test]
    fn resolve_rejects_values_that_hold_no_id() {
        let subs = [DM];
        for bad in [
            json!(null),
            json!(true),
            json!("general"),
            json!("#general2"),
            // A float inside other text: its mantissa is not an id.
            json!("about 1.2571656770521702e18"),
            json!(""),
            json!([DM]),
            json!(u64::MAX),
        ] {
            assert_eq!(resolve(&bad, &subs), None, "{bad}");
        }
    }
}
