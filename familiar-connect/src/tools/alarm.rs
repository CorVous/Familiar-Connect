//! `set_alarm` / `cancel_alarm` tools (subsystem 08; Python `tools/alarm.py`).
//!
//! Both route the wake to the channel the user spoke in
//! (`ctx.channel_id` / `ctx.channel_kind`, `originating_turn_id = ctx.turn_id`).
//! The scheduler is reached at call time via `ctx.scheduler` (the `build_*`
//! functions take a scheduler only for API symmetry; the live one is on the
//! context).

use std::sync::Arc;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde_json::{Value, json};

use crate::support::time::iso_utc;
use crate::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput};
use crate::tools::scheduler::AlarmScheduler;

const MAX_REASON_LEN: usize = 200;
const MIN_DELAY_S: i64 = 1;
const MAX_DELAY_S: i64 = 60 * 60 * 24 * 365; // one year cap — defensive
/// Allow tiny past-skew (5 s) so an immediate `now` doesn't bounce.
const PAST_SKEW_MS: i64 = 5_000;

fn error_output(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

/// Does this string parse as an offset-less ISO-8601 datetime?
fn is_naive_datetime(s: &str) -> bool {
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if chrono::NaiveDateTime::parse_from_str(s, fmt).is_ok() {
            return true;
        }
    }
    false
}

/// Resolve the target time from `when` (ISO with tz) or `delay_seconds`.
///
/// Returns `Ok(datetime)` or `Err(message)` where the message is a user-facing
/// tool-error string.
fn resolve_when(args: &Value) -> Result<DateTime<Utc>, String> {
    let when = args.get("when").and_then(Value::as_str);
    if let Some(w) = when {
        if !w.is_empty() {
            return match DateTime::parse_from_rfc3339(w) {
                Ok(dt) => {
                    let target = dt.with_timezone(&Utc);
                    let skew_ms = (Utc::now() - target).num_milliseconds();
                    if skew_ms > PAST_SKEW_MS {
                        return Err(format!("'when' is {}s in the past", skew_ms / 1000));
                    }
                    Ok(target)
                }
                Err(e) => {
                    if is_naive_datetime(w) {
                        Err("invalid 'when' (must include timezone, e.g. '...+00:00')".to_owned())
                    } else {
                        Err(format!("invalid 'when' (must be ISO-8601): {e}"))
                    }
                }
            };
        }
    }

    // `delay_seconds` must be an int (bool explicitly rejected).
    if let Some(dv) = args.get("delay_seconds") {
        if !dv.is_boolean() {
            if let Some(d) = dv.as_i64() {
                if d < MIN_DELAY_S {
                    return Err(format!("'delay_seconds' must be \u{2265} {MIN_DELAY_S}"));
                }
                if d > MAX_DELAY_S {
                    return Err(format!("'delay_seconds' must be \u{2264} {MAX_DELAY_S}"));
                }
                return Ok(Utc::now() + ChronoDuration::seconds(d));
            }
        }
    }

    Err("missing 'when' (ISO-8601 timestamp) or 'delay_seconds' (int)".to_owned())
}

/// The `set_alarm` tool handler. Validation failures are returned as
/// `{"error": ...}` tool output; only a store fault propagates as `Err`.
async fn set_alarm_handler(args: &Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput> {
    let Some(scheduler) = ctx.scheduler.as_ref() else {
        return Ok(error_output("alarm scheduler not wired into context"));
    };

    let reason = match args.get("reason").and_then(Value::as_str) {
        Some(r) if !r.trim().is_empty() => r,
        _ => return Ok(error_output("missing or empty 'reason' (string)")),
    };
    if reason.chars().count() > MAX_REASON_LEN {
        return Ok(error_output(&format!(
            "'reason' too long (>{MAX_REASON_LEN} chars)"
        )));
    }

    let resolved = match resolve_when(args) {
        Ok(dt) => dt,
        Err(msg) => return Ok(error_output(&msg)),
    };

    let alarm_id = scheduler
        .add(
            ctx.channel_id,
            &ctx.channel_kind,
            resolved,
            reason,
            Some(&ctx.turn_id),
        )
        .await?;

    Ok(ToolOutput::Text(
        json!({
            "alarm_id": alarm_id,
            "scheduled_at": iso_utc(resolved),
            "ack": "ok",
        })
        .to_string(),
    ))
}

/// The `cancel_alarm` tool handler. Domain outcomes return `Ok(ToolOutput)`; a
/// genuine store write fault propagates as `Err` (the loop wraps it as an
/// `{"error": ...}` result), mirroring Python letting the DB error raise.
async fn cancel_alarm_handler(args: &Value, ctx: &ToolContext) -> anyhow::Result<ToolOutput> {
    let Some(scheduler) = ctx.scheduler.as_ref() else {
        return Ok(error_output("alarm scheduler not wired into context"));
    };

    let alarm_id = match args.get("alarm_id").and_then(Value::as_str) {
        Some(a) if !a.is_empty() => a,
        _ => return Ok(error_output("missing or empty 'alarm_id'")),
    };

    if scheduler.cancel(alarm_id).await? {
        Ok(ToolOutput::Text(
            json!({ "alarm_id": alarm_id, "ack": "ok" }).to_string(),
        ))
    } else {
        Ok(error_output(&format!(
            "no pending alarm with id {alarm_id}"
        )))
    }
}

/// Build the `set_alarm` tool. `scheduler` is accepted for symmetry; the live
/// scheduler is reached via `ctx.scheduler`.
#[must_use]
pub fn build_alarm_tool(_scheduler: &AlarmScheduler) -> Tool {
    Tool::new(
        "set_alarm",
        "Schedule a future wake. The familiar will be re-prompted in the channel \
         where this tool was called when the time arrives. Provide one of 'when' \
         (ISO-8601 UTC) or 'delay_seconds' (positive integer).",
        json!({
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description":
                        "Absolute target time as ISO-8601 with timezone (e.g. '2030-01-01T12:00:00+00:00').",
                },
                "delay_seconds": {
                    "type": "integer",
                    "minimum": MIN_DELAY_S,
                    "maximum": MAX_DELAY_S,
                    "description": "Wait this many seconds before waking.",
                },
                "reason": {
                    "type": "string",
                    "maxLength": MAX_REASON_LEN,
                    "description": "Short note shown back to the familiar on wake.",
                },
            },
            "required": ["reason"],
        }),
        Arc::new(FnHandler(|args: Value, ctx: ToolContext| async move {
            set_alarm_handler(&args, &ctx).await
        })),
    )
}

/// Build the `cancel_alarm` tool.
#[must_use]
pub fn build_cancel_alarm_tool(_scheduler: &AlarmScheduler) -> Tool {
    Tool::new(
        "cancel_alarm",
        "Cancel a previously scheduled alarm. The id is the 'alarm_id' returned \
         from set_alarm.",
        json!({
            "type": "object",
            "properties": {
                "alarm_id": {
                    "type": "string",
                    "description": "Id of the alarm to cancel.",
                },
            },
            "required": ["alarm_id"],
        }),
        Arc::new(FnHandler(|args: Value, ctx: ToolContext| async move {
            cancel_alarm_handler(&args, &ctx).await
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_when_missing_both_errors() {
        let err = resolve_when(&json!({})).unwrap_err();
        assert!(err.contains("missing"));
    }

    #[test]
    fn resolve_when_rejects_naive_timestamp() {
        let err = resolve_when(&json!({"when": "2030-01-01T12:00:00"})).unwrap_err();
        assert!(err.contains("timezone"));
    }

    #[test]
    fn resolve_when_rejects_unparseable() {
        let err = resolve_when(&json!({"when": "not a date"})).unwrap_err();
        assert!(err.contains("ISO-8601"));
    }

    #[test]
    fn resolve_when_rejects_past_timestamp() {
        let past = iso_utc(Utc::now() - ChronoDuration::hours(1));
        let err = resolve_when(&json!({ "when": past })).unwrap_err();
        assert!(err.to_lowercase().contains("past"));
    }

    #[test]
    fn resolve_when_accepts_future_timestamp() {
        let future = Utc::now() + ChronoDuration::minutes(5);
        let got = resolve_when(&json!({ "when": iso_utc(future) })).unwrap();
        assert_eq!(iso_utc(got), iso_utc(future));
    }

    #[test]
    fn resolve_when_rejects_bool_delay() {
        let err = resolve_when(&json!({"delay_seconds": true})).unwrap_err();
        assert!(err.contains("missing"));
    }

    #[test]
    fn resolve_when_rejects_below_min_delay() {
        let err = resolve_when(&json!({"delay_seconds": 0})).unwrap_err();
        assert!(err.contains("delay_seconds"));
    }

    #[test]
    fn resolve_when_accepts_valid_delay() {
        assert!(resolve_when(&json!({"delay_seconds": 30})).is_ok());
    }

    #[test]
    fn max_delay_is_one_year() {
        assert_eq!(MAX_DELAY_S, 31_536_000);
    }
}
