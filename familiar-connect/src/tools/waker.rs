//! Alarm waker processor (subsystem 08; Python `tools/waker.py`).
//!
//! Listens on [`TOPIC_ALARM_FIRED`] and republishes a synthetic
//! `discord.text`-shaped event so the existing `TextResponder` picks it up and
//! produces a follow-up reply. Voice-origin alarms fall back to text in the same
//! channel id (MVP). The synthetic payload carries the *waker's* configured
//! `familiar_id` (alarm payloads carry none); per-familiar filtering happens in
//! the responder. The `alarm: true` marker pierces activity absence gating.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::bus::envelope::{Event, payload};
use crate::bus::protocols::{EventBus, Processor};
use crate::bus::topics::{TOPIC_ALARM_FIRED, TOPIC_DISCORD_TEXT};
use crate::log_style as ls;

/// Translate `alarm.fired` into a synthetic text-channel turn.
pub struct AlarmWaker {
    familiar_id: String,
    topics: [&'static str; 1],
}

impl AlarmWaker {
    /// New waker bound to a familiar id.
    #[must_use]
    pub fn new(familiar_id: impl Into<String>) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            topics: [TOPIC_ALARM_FIRED],
        }
    }
}

#[async_trait]
impl Processor for AlarmWaker {
    #[allow(
        clippy::unnecessary_literal_bound,
        reason = "return type is fixed by the Processor trait signature"
    )]
    fn name(&self) -> &str {
        "alarm-waker"
    }

    fn topics(&self) -> &[&str] {
        &self.topics
    }

    async fn handle(&self, event: std::sync::Arc<Event>, bus: &dyn EventBus) -> anyhow::Result<()> {
        if event.topic != TOPIC_ALARM_FIRED {
            return Ok(());
        }
        let Some(payload_val) = event.payload.downcast_ref::<Value>() else {
            return Ok(());
        };
        if !payload_val.is_object() {
            return Ok(());
        }

        let Some(channel_id) = payload_val.get("channel_id").and_then(Value::as_i64) else {
            return Ok(());
        };
        let reason = payload_val
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("");

        let kind = payload_val
            .get("channel_kind")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("text");
        if kind != "text" && kind != "voice" {
            tracing::warn!(
                "{} {}",
                ls::tag("AlarmWaker", ls::LY),
                ls::kv_styled("unknown_channel_kind", kind, ls::W, ls::LY),
            );
            return Ok(());
        }
        if kind == "voice" {
            tracing::info!(
                "{} {} {}",
                ls::tag("AlarmWaker", ls::LM),
                ls::kv_styled(
                    "voice_fallback_to_text",
                    &channel_id.to_string(),
                    ls::W,
                    ls::LM
                ),
                ls::kv_styled("reason", &ls::trunc(reason, 80), ls::W, ls::LW),
            );
        }

        let synth_event_id = uuid::Uuid::new_v4().simple().to_string();
        let alarm_id = payload_val
            .get("alarm_id")
            .and_then(Value::as_str)
            .unwrap_or(&synth_event_id);
        let synth_payload: Value = json!({
            "familiar_id": self.familiar_id,
            "channel_id": channel_id,
            "content": format!("[alarm fired: {reason}]"),
            "author": null,
            "guild_id": null,
            "message_id": null,
            "reply_to_message_id": null,
            "mentions": [],
            "alarm": true,
        });

        bus.publish(Event {
            event_id: synth_event_id.clone(),
            turn_id: format!("wake-{alarm_id}"),
            session_id: channel_id.to_string(),
            parent_event_ids: vec![event.event_id.clone()],
            topic: TOPIC_DISCORD_TEXT.to_owned(),
            timestamp: chrono::Utc::now(),
            sequence_number: 0,
            payload: payload(synth_payload),
        })
        .await;
        Ok(())
    }
}
