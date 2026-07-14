//! Alarm scheduler (subsystem 08; Python `tools/scheduler.py`).
//!
//! One `tokio` task per pending alarm. Tasks sleep until the target time, then
//! do a **conditional** `mark_alarm_fired` (`fired_at IS NULL AND cancelled_at
//! IS NULL`) and, only if that stamped a row, publish [`TOPIC_ALARM_FIRED`]. On
//! [`start`](AlarmScheduler::start) the scheduler reloads rows left pending from
//! a previous process and reschedules them — past-due rows fire immediately.
//!
//! The DB insert happens **before** the timer task is spawned (the row is
//! durable before any timer exists), and the conditional update is the entire
//! cancel/fire race story: a cancelled or already-fired alarm can never
//! double-publish.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde_json::{Value, json};
use tokio::task::JoinHandle;

use crate::bus::envelope::{Event, payload};
use crate::bus::protocols::EventBus;
use crate::bus::topics::TOPIC_ALARM_FIRED;
use crate::history::async_store::AsyncHistoryStore;
use crate::log_style as ls;
use crate::support::time::{iso_utc, parse_iso};

/// The data one timer task needs to fire an alarm.
#[derive(Clone)]
struct PendingAlarm {
    id: String,
    channel_id: i64,
    channel_kind: String,
    scheduled_at: DateTime<Utc>,
    reason: String,
    originating_turn_id: Option<String>,
}

/// Per-familiar wake scheduler — DB-backed, `tokio`-driven.
pub struct AlarmScheduler {
    history: Arc<AsyncHistoryStore>,
    bus: Arc<dyn EventBus>,
    familiar_id: String,
    tasks: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
    started: Mutex<bool>,
}

impl AlarmScheduler {
    /// New scheduler bound to a store, bus, and familiar id.
    #[must_use]
    pub fn new(
        history: Arc<AsyncHistoryStore>,
        bus: Arc<dyn EventBus>,
        familiar_id: impl Into<String>,
    ) -> Self {
        Self {
            history,
            bus,
            familiar_id: familiar_id.into(),
            tasks: Arc::new(Mutex::new(HashMap::new())),
            started: Mutex::new(false),
        }
    }

    /// Load pending alarms and schedule each. Idempotent.
    ///
    /// # Errors
    /// Propagates a store error while listing pending alarms.
    pub async fn start(&self) -> Result<(), crate::history::StoreError> {
        {
            let mut started = self.started.lock().expect("started mutex");
            if *started {
                return Ok(());
            }
            *started = true;
        }
        let pending = self
            .history
            .list_pending_alarms(self.familiar_id.clone())
            .await?;
        let count = pending.len();
        for row in pending {
            let Some(scheduled_at) = parse_iso(&row.scheduled_at) else {
                tracing::warn!(
                    "{} {}",
                    ls::tag("Alarm", ls::LY),
                    ls::kv_styled("bad_scheduled_at", &row.id, ls::W, ls::LY),
                );
                continue;
            };
            self.spawn(PendingAlarm {
                id: row.id,
                channel_id: row.channel_id,
                channel_kind: row.channel_kind,
                scheduled_at,
                reason: row.reason,
                originating_turn_id: row.originating_turn_id,
            });
        }
        tracing::info!(
            "{} {} {}",
            ls::tag("Alarm", ls::LC),
            ls::kv_styled("loaded_pending", &count.to_string(), ls::W, ls::LC),
            ls::kv_styled("familiar_id", &self.familiar_id, ls::W, ls::LW),
        );
        Ok(())
    }

    /// Cancel all in-flight timer tasks. Does **not** touch the DB — pending rows
    /// reload on next boot.
    pub async fn shutdown(&self) {
        let handles: Vec<JoinHandle<()>> = {
            let mut tasks = self.tasks.lock().expect("tasks mutex");
            tasks.drain().map(|(_, h)| h).collect()
        };
        for h in &handles {
            h.abort();
        }
        for h in handles {
            let _ = h.await;
        }
    }

    /// Insert + schedule a new alarm; returns the alarm id.
    ///
    /// # Errors
    /// Propagates a store error from the row insert.
    pub async fn add(
        &self,
        channel_id: i64,
        channel_kind: &str,
        scheduled_at: DateTime<Utc>,
        reason: &str,
        originating_turn_id: Option<&str>,
    ) -> Result<String, crate::history::StoreError> {
        let alarm_id = self
            .history
            .insert_alarm(
                self.familiar_id.clone(),
                channel_id,
                channel_kind.to_owned(),
                iso_utc(scheduled_at),
                reason.to_owned(),
                originating_turn_id.map(ToOwned::to_owned),
            )
            .await?;
        self.spawn(PendingAlarm {
            id: alarm_id.clone(),
            channel_id,
            channel_kind: channel_kind.to_owned(),
            scheduled_at,
            reason: reason.to_owned(),
            originating_turn_id: originating_turn_id.map(ToOwned::to_owned),
        });
        Ok(alarm_id)
    }

    /// Stop the in-flight timer (if any) and stamp the row cancelled.
    ///
    /// Returns the DB result: cancelling an alarm loaded by a *different* process
    /// instance still returns `Ok(true)` and prevents its fire via the
    /// conditional `mark_alarm_fired`; an alarm that was never pending returns
    /// `Ok(false)`.
    ///
    /// # Errors
    /// Propagates a store write fault (Python lets `cancel_alarm` raise to the
    /// caller); the tool layer surfaces it as an `{"error": ...}` result rather
    /// than a benign "no pending alarm".
    pub async fn cancel(&self, alarm_id: &str) -> Result<bool, crate::history::StoreError> {
        let handle = self.tasks.lock().expect("tasks mutex").remove(alarm_id);
        if let Some(h) = handle {
            h.abort();
        }
        self.history
            .cancel_alarm(alarm_id.to_owned(), iso_utc(Utc::now()))
            .await
    }

    fn spawn(&self, alarm: PendingAlarm) {
        let history = Arc::clone(&self.history);
        let bus = Arc::clone(&self.bus);
        let tasks = Arc::clone(&self.tasks);
        let id = alarm.id.clone();
        // Gate the task body until `add`/`start` has recorded our handle, so a
        // past-due (immediate) fire cannot self-remove before the insert.
        let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            // If the gate sender is dropped (task aborted before release) this
            // resolves Err and we bail without firing.
            if gate_rx.await.is_err() {
                return;
            }
            let self_id = alarm.id.clone();
            sleep_then_fire(&history, bus.as_ref(), &alarm).await;
            tasks.lock().expect("tasks mutex").remove(&self_id);
        });
        self.tasks.lock().expect("tasks mutex").insert(id, handle);
        let _ = gate_tx.send(());
    }
}

/// Sleep until the target time, conditionally stamp fired, and publish.
async fn sleep_then_fire(history: &AsyncHistoryStore, bus: &dyn EventBus, alarm: &PendingAlarm) {
    let delay = alarm.scheduled_at - Utc::now();
    if delay > ChronoDuration::zero() {
        if let Ok(std_delay) = delay.to_std() {
            tokio::time::sleep(std_delay).await;
        }
    }
    let fired_at = Utc::now();
    let updated = match history
        .mark_alarm_fired(alarm.id.clone(), iso_utc(fired_at))
        .await
    {
        Ok(v) => v,
        Err(e) => {
            // A store write fault is a genuine fault, not a benign "already
            // fired". Python lets the fire task raise (its runtime logs the
            // fault); surface it here instead of silently coercing to `false`.
            // Publish is skipped either way, but the error is now visible.
            tracing::error!(
                "{} {} {}",
                ls::tag("Alarm", ls::R),
                ls::kv_styled("mark_fired_error", &alarm.id, ls::W, ls::R),
                ls::kv_styled("err", &format!("{e}"), ls::W, ls::LW),
            );
            return;
        }
    };
    if !updated {
        // Already fired or cancelled by another path — skip publish.
        return;
    }
    let payload_value: Value = json!({
        "alarm_id": alarm.id,
        "channel_id": alarm.channel_id,
        "channel_kind": alarm.channel_kind,
        "reason": alarm.reason,
        "scheduled_at": iso_utc(alarm.scheduled_at),
        "fired_at": iso_utc(fired_at),
        "originating_turn_id": alarm.originating_turn_id,
    });
    bus.publish(Event {
        event_id: uuid::Uuid::new_v4().simple().to_string(),
        turn_id: format!("alarm-{}", alarm.id),
        session_id: format!("alarm:{}", alarm.channel_id),
        parent_event_ids: Vec::new(),
        topic: TOPIC_ALARM_FIRED.to_owned(),
        timestamp: fired_at,
        sequence_number: 0,
        payload: payload(payload_value),
    })
    .await;
    tracing::info!(
        "{} {} {}",
        ls::tag("Alarm", ls::LM),
        ls::kv_styled("fired", &alarm.id, ls::W, ls::LM),
        ls::kv_styled("reason", &ls::trunc(&alarm.reason, 80), ls::W, ls::LW),
    );
}
