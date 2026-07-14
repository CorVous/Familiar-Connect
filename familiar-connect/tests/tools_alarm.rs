//! Ported from Python `tests/test_alarm_scheduler.py`, `test_alarm_tool.py`,
//! and `test_alarm_waker.py` — the DB-backed [`AlarmScheduler`], the
//! `set_alarm` / `cancel_alarm` tools, and the [`AlarmWaker`] bus processor.
//!
//! Timing uses real (short) sleeps, mirroring the Python suite; each test
//! subscribes *before* `start`/`add` so a past-due immediate fire cannot race
//! ahead of the subscriber (a multi-threaded-runtime concern the single-threaded
//! Python event loop did not have).

use std::sync::Arc;
use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};
use serde_json::{Value, json};

use familiar_connect::bus::envelope::{Event, payload};
use familiar_connect::bus::in_process::{InProcessEventBus, Subscription};
use familiar_connect::bus::protocols::{BackpressurePolicy, EventBus, Processor};
use familiar_connect::bus::topics::{TOPIC_ALARM_FIRED, TOPIC_DISCORD_TEXT};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::HistoryStore;
use familiar_connect::support::time::iso_utc;
use familiar_connect::tools::alarm::{build_alarm_tool, build_cancel_alarm_tool};
use familiar_connect::tools::registry::{ToolContext, ToolOutput};
use familiar_connect::tools::scheduler::AlarmScheduler;
use familiar_connect::tools::waker::AlarmWaker;

const FAMILIAR: &str = "aria";

fn make_store() -> Arc<AsyncHistoryStore> {
    Arc::new(AsyncHistoryStore::new(
        HistoryStore::open(":memory:").unwrap(),
    ))
}

fn make_bus() -> Arc<dyn EventBus> {
    Arc::new(InProcessEventBus::new())
}

async fn drain_one(sub: &mut Subscription, secs: u64) -> Option<Arc<Event>> {
    tokio::time::timeout(Duration::from_secs(secs), sub.recv())
        .await
        .ok()
        .flatten()
}

fn payload_of(event: &Event) -> Value {
    event
        .payload
        .downcast_ref::<Value>()
        .expect("payload is a JSON Value")
        .clone()
}

fn text(out: ToolOutput) -> Value {
    match out {
        ToolOutput::Text(s) => serde_json::from_str(&s).unwrap(),
        ToolOutput::Image(_) => panic!("expected text output"),
    }
}

// ---------------------------------------------------------------------------
// AlarmScheduler
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn alarm_fires_at_scheduled_time() {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let mut sub = bus.subscribe(&[TOPIC_ALARM_FIRED], BackpressurePolicy::Unbounded, 0);
    let scheduler = AlarmScheduler::new(store.clone(), bus.clone(), FAMILIAR);
    scheduler.start().await.unwrap();

    let alarm_id = scheduler
        .add(
            42,
            "text",
            Utc::now() + ChronoDuration::milliseconds(80),
            "ping",
            None,
        )
        .await
        .unwrap();
    let event = drain_one(&mut sub, 2).await.expect("alarm fired");
    assert_eq!(event.topic, TOPIC_ALARM_FIRED);
    let p = payload_of(&event);
    assert_eq!(p["alarm_id"], alarm_id);
    assert_eq!(p["channel_id"], 42);
    assert_eq!(p["channel_kind"], "text");
    assert_eq!(p["reason"], "ping");

    scheduler.shutdown().await;
    bus.shutdown().await;
    assert!(
        store
            .sync()
            .list_pending_alarms(FAMILIAR)
            .unwrap()
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn past_due_alarm_fires_immediately_on_start() {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let mut sub = bus.subscribe(&[TOPIC_ALARM_FIRED], BackpressurePolicy::Unbounded, 0);

    let past = iso_utc(Utc::now() - ChronoDuration::seconds(10));
    store
        .sync()
        .insert_alarm(FAMILIAR, 7, "text", &past, "overdue", None)
        .unwrap();

    let scheduler = AlarmScheduler::new(store.clone(), bus.clone(), FAMILIAR);
    scheduler.start().await.unwrap();

    let event = drain_one(&mut sub, 2).await.expect("past-due alarm fired");
    assert_eq!(payload_of(&event)["reason"], "overdue");

    scheduler.shutdown().await;
    bus.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_prevents_fire() {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let mut sub = bus.subscribe(&[TOPIC_ALARM_FIRED], BackpressurePolicy::Unbounded, 0);
    let scheduler = AlarmScheduler::new(store.clone(), bus.clone(), FAMILIAR);
    scheduler.start().await.unwrap();

    let alarm_id = scheduler
        .add(
            1,
            "text",
            Utc::now() + ChronoDuration::seconds(30),
            "should-cancel",
            None,
        )
        .await
        .unwrap();
    assert!(scheduler.cancel(&alarm_id).await.unwrap());
    // No event within 200ms.
    assert!(
        tokio::time::timeout(Duration::from_millis(200), sub.recv())
            .await
            .is_err()
    );

    scheduler.shutdown().await;
    bus.shutdown().await;
    assert!(
        store
            .sync()
            .list_pending_alarms(FAMILIAR)
            .unwrap()
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_unknown_returns_false() {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let scheduler = AlarmScheduler::new(store, bus.clone(), FAMILIAR);
    scheduler.start().await.unwrap();
    assert!(!scheduler.cancel("no-such-id").await.unwrap());
    scheduler.shutdown().await;
    bus.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn payload_carries_originating_turn_id() {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let mut sub = bus.subscribe(&[TOPIC_ALARM_FIRED], BackpressurePolicy::Unbounded, 0);
    let scheduler = AlarmScheduler::new(store, bus.clone(), FAMILIAR);
    scheduler.start().await.unwrap();

    scheduler
        .add(
            11,
            "voice",
            Utc::now() + ChronoDuration::milliseconds(50),
            "audit",
            Some("turn-abc"),
        )
        .await
        .unwrap();
    let event = drain_one(&mut sub, 2).await.expect("alarm fired");
    let p = payload_of(&event);
    assert_eq!(p["originating_turn_id"], "turn-abc");
    assert_eq!(p["channel_kind"], "voice");

    scheduler.shutdown().await;
    bus.shutdown().await;
}

// ---------------------------------------------------------------------------
// set_alarm / cancel_alarm tools
// ---------------------------------------------------------------------------

struct AlarmFixture {
    store: Arc<AsyncHistoryStore>,
    bus: Arc<dyn EventBus>,
    scheduler: Arc<AlarmScheduler>,
}

async fn fixture(channel_id: i64, channel_kind: &str) -> (AlarmFixture, ToolContext) {
    let store = make_store();
    let bus = make_bus();
    bus.start().await;
    let scheduler = Arc::new(AlarmScheduler::new(store.clone(), bus.clone(), FAMILIAR));
    scheduler.start().await.unwrap();
    let ctx = ToolContext::new(FAMILIAR, channel_id, channel_kind, "turn-test")
        .with_scheduler(scheduler.clone());
    (
        AlarmFixture {
            store,
            bus,
            scheduler,
        },
        ctx,
    )
}

impl AlarmFixture {
    async fn teardown(self) {
        self.scheduler.shutdown().await;
        self.bus.shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_alarm_with_delay_seconds_inserts_row() {
    let (fix, ctx) = fixture(42, "text").await;
    let tool = build_alarm_tool(&fix.scheduler);
    let body = text(
        tool.handler
            .call(json!({"reason": "ping", "delay_seconds": 30}), &ctx)
            .await
            .unwrap(),
    );
    assert!(body.get("alarm_id").is_some());
    assert!(body.get("scheduled_at").is_some());
    assert_eq!(body["ack"], "ok");

    let pending = fix.store.sync().list_pending_alarms(FAMILIAR).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].reason, "ping");
    fix.scheduler
        .cancel(body["alarm_id"].as_str().unwrap())
        .await
        .unwrap();
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_alarm_with_iso_when_inserts_row() {
    let (fix, ctx) = fixture(42, "text").await;
    let future = iso_utc(Utc::now() + ChronoDuration::minutes(1));
    let tool = build_alarm_tool(&fix.scheduler);
    let body = text(
        tool.handler
            .call(json!({"reason": "later", "when": future}), &ctx)
            .await
            .unwrap(),
    );
    assert_eq!(body["scheduled_at"], future);
    fix.scheduler
        .cancel(body["alarm_id"].as_str().unwrap())
        .await
        .unwrap();
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_alarm_rejects_past_timestamp() {
    let (fix, ctx) = fixture(42, "text").await;
    let past = iso_utc(Utc::now() - ChronoDuration::hours(1));
    let tool = build_alarm_tool(&fix.scheduler);
    let body = text(
        tool.handler
            .call(json!({"reason": "rip", "when": past}), &ctx)
            .await
            .unwrap(),
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("past")
    );
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_alarm_rejects_missing_reason() {
    let (fix, ctx) = fixture(42, "text").await;
    let tool = build_alarm_tool(&fix.scheduler);
    let body = text(
        tool.handler
            .call(json!({"delay_seconds": 10}), &ctx)
            .await
            .unwrap(),
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("reason")
    );
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_alarm_uses_caller_channel_from_ctx() {
    let (fix, ctx) = fixture(777, "voice").await;
    let tool = build_alarm_tool(&fix.scheduler);
    let body = text(
        tool.handler
            .call(json!({"reason": "echo", "delay_seconds": 60}), &ctx)
            .await
            .unwrap(),
    );
    let pending = fix.store.sync().list_pending_alarms(FAMILIAR).unwrap();
    assert_eq!(pending[0].channel_id, 777);
    assert_eq!(pending[0].channel_kind, "voice");
    fix.scheduler
        .cancel(body["alarm_id"].as_str().unwrap())
        .await
        .unwrap();
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_alarm_tool_cancels_pending() {
    let (fix, ctx) = fixture(42, "text").await;
    let set_body = text(
        build_alarm_tool(&fix.scheduler)
            .handler
            .call(json!({"reason": "x", "delay_seconds": 60}), &ctx)
            .await
            .unwrap(),
    );
    let alarm_id = set_body["alarm_id"].as_str().unwrap().to_owned();
    let cancel_body = text(
        build_cancel_alarm_tool(&fix.scheduler)
            .handler
            .call(json!({"alarm_id": alarm_id}), &ctx)
            .await
            .unwrap(),
    );
    assert_eq!(cancel_body["ack"], "ok");
    assert!(
        fix.store
            .sync()
            .list_pending_alarms(FAMILIAR)
            .unwrap()
            .is_empty()
    );
    fix.teardown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_unknown_returns_error() {
    let (fix, ctx) = fixture(42, "text").await;
    let body = text(
        build_cancel_alarm_tool(&fix.scheduler)
            .handler
            .call(json!({"alarm_id": "no-such"}), &ctx)
            .await
            .unwrap(),
    );
    assert!(body.get("error").is_some());
    fix.teardown().await;
}

// ---------------------------------------------------------------------------
// AlarmWaker
// ---------------------------------------------------------------------------

fn alarm_fired_event(channel_id: i64, channel_kind: &str, reason: &str) -> Event {
    let p = json!({
        "alarm_id": "alarm-1",
        "channel_id": channel_id,
        "channel_kind": channel_kind,
        "reason": reason,
        "scheduled_at": iso_utc(Utc::now()),
        "fired_at": iso_utc(Utc::now()),
        "originating_turn_id": null,
    });
    Event {
        event_id: uuid::Uuid::new_v4().simple().to_string(),
        turn_id: "alarm-alarm-1".to_owned(),
        session_id: format!("alarm:{channel_id}"),
        parent_event_ids: vec![],
        topic: TOPIC_ALARM_FIRED.to_owned(),
        timestamp: Utc::now(),
        sequence_number: 0,
        payload: payload(p),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waker_republishes_as_discord_text() {
    let bus = make_bus();
    bus.start().await;
    let mut text_sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
    let waker = AlarmWaker::new(FAMILIAR);
    waker
        .handle(
            Arc::new(alarm_fired_event(42, "text", "ping")),
            bus.as_ref(),
        )
        .await
        .unwrap();
    let event = drain_one(&mut text_sub, 1)
        .await
        .expect("synthetic text event");
    assert_eq!(event.topic, TOPIC_DISCORD_TEXT);
    let p = payload_of(&event);
    assert_eq!(p["familiar_id"], FAMILIAR);
    assert_eq!(p["channel_id"], 42);
    assert!(
        p["content"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("alarm")
    );
    assert!(p["content"].as_str().unwrap().contains("ping"));
    bus.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waker_payload_carries_alarm_marker() {
    let bus = make_bus();
    bus.start().await;
    let mut text_sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
    let waker = AlarmWaker::new(FAMILIAR);
    waker
        .handle(
            Arc::new(alarm_fired_event(42, "text", "ping")),
            bus.as_ref(),
        )
        .await
        .unwrap();
    let event = drain_one(&mut text_sub, 1)
        .await
        .expect("synthetic text event");
    assert_eq!(payload_of(&event)["alarm"], true);
    bus.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waker_stamps_own_familiar_id() {
    let bus = make_bus();
    bus.start().await;
    let mut text_sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
    let waker = AlarmWaker::new("other-fam");
    waker
        .handle(
            Arc::new(alarm_fired_event(42, "text", "ping")),
            bus.as_ref(),
        )
        .await
        .unwrap();
    let event = drain_one(&mut text_sub, 1)
        .await
        .expect("synthetic text event");
    assert_eq!(payload_of(&event)["familiar_id"], "other-fam");
    bus.shutdown().await;
}
