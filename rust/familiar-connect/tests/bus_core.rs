//! Ported from `tests/test_bus.py` — `InProcessEventBus` lifecycle, fan-out,
//! backpressure, and shutdown-drain semantics.

use std::time::Duration;

use chrono::Utc;
use familiar_connect::bus::envelope::{Event, payload};
use familiar_connect::bus::{BackpressurePolicy, EventBus, InProcessEventBus, Lifecycle};
use tokio::time::Instant;

fn mk_event(topic: &str, seq: u64) -> Event {
    Event {
        event_id: format!("e-{seq}"),
        turn_id: "turn-0".to_owned(),
        session_id: "sess-0".to_owned(),
        parent_event_ids: Vec::new(),
        topic: topic.to_owned(),
        timestamp: Utc::now(),
        sequence_number: seq,
        payload: payload(()),
    }
}

// --- Lifecycle -------------------------------------------------------------

#[test]
fn starts_in_starting() {
    let bus = InProcessEventBus::new();
    assert_eq!(bus.lifecycle(), Lifecycle::Starting);
}

#[tokio::test]
async fn running_after_start() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    assert_eq!(bus.lifecycle(), Lifecycle::Running);
    bus.shutdown().await;
}

#[tokio::test]
async fn draining_then_stopped_on_shutdown() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    bus.shutdown().await;
    assert_eq!(bus.lifecycle(), Lifecycle::Stopped);
}

// --- Fan-out ---------------------------------------------------------------

#[tokio::test]
async fn single_subscriber_receives_published_event() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["t"], BackpressurePolicy::Block, 0);
    let task = tokio::spawn(async move { sub.recv().await });
    bus.publish(mk_event("t", 0)).await;
    let received = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("consumer resolves within the timeout")
        .expect("consumer task joins cleanly");
    let ev = received.expect("an event was delivered");
    assert_eq!(ev.sequence_number, 0);
    bus.shutdown().await;
}

#[tokio::test]
async fn two_subscribers_each_get_every_event() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut a = bus.subscribe(&["t"], BackpressurePolicy::Block, 0);
    let mut b = bus.subscribe(&["t"], BackpressurePolicy::Block, 0);
    let task_a = tokio::spawn(async move {
        let mut got = Vec::new();
        while got.len() < 3 {
            match a.recv().await {
                Some(ev) => got.push(ev.sequence_number),
                None => break,
            }
        }
        got
    });
    let task_b = tokio::spawn(async move {
        let mut got = Vec::new();
        while got.len() < 3 {
            match b.recv().await {
                Some(ev) => got.push(ev.sequence_number),
                None => break,
            }
        }
        got
    });
    for i in 0_u64..3 {
        bus.publish(mk_event("t", i)).await;
    }
    let (got_a, got_b) = tokio::join!(task_a, task_b);
    assert_eq!(got_a.expect("task a joins"), vec![0_u64, 1, 2]);
    assert_eq!(got_b.expect("task b joins"), vec![0_u64, 1, 2]);
    bus.shutdown().await;
}

#[tokio::test]
async fn topic_isolation() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["wanted"], BackpressurePolicy::Block, 0);
    let task = tokio::spawn(async move { sub.recv().await.map(|e| e.sequence_number) });
    bus.publish(mk_event("ignored", 99)).await;
    bus.publish(mk_event("wanted", 1)).await;
    let got = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("consumer resolves within the timeout")
        .expect("consumer task joins cleanly");
    assert_eq!(got, Some(1));
    bus.shutdown().await;
}

// --- Backpressure ----------------------------------------------------------

#[tokio::test]
async fn drop_oldest_keeps_newest() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    // Never consume during publish; fill the queue; the newest must survive.
    let mut sub = bus.subscribe(&["audio"], BackpressurePolicy::DropOldest, 2);
    for i in 0_u64..5 {
        bus.publish(mk_event("audio", i)).await;
    }
    let mut got = Vec::new();
    while got.len() < 2 {
        got.push(sub.recv().await.expect("event").sequence_number);
    }
    assert_eq!(got, vec![3_u64, 4]);
    bus.shutdown().await;
}

#[tokio::test]
async fn drop_newest_keeps_oldest() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["t"], BackpressurePolicy::DropNewest, 2);
    for i in 0_u64..5 {
        bus.publish(mk_event("t", i)).await;
    }
    let mut got = Vec::new();
    while got.len() < 2 {
        got.push(sub.recv().await.expect("event").sequence_number);
    }
    assert_eq!(got, vec![0_u64, 1]);
    bus.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn block_policy_waits_on_slow_consumer() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["t"], BackpressurePolicy::Block, 1);
    let consumer = tokio::spawn(async move {
        let mut got = Vec::new();
        while got.len() < 3 {
            match sub.recv().await {
                Some(ev) => {
                    got.push(ev.sequence_number);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                None => break,
            }
        }
        got
    });
    tokio::task::yield_now().await; // let the consumer reach its first `recv`
    let start = Instant::now();
    for i in 0_u64..3 {
        bus.publish(mk_event("t", i)).await;
    }
    let elapsed = start.elapsed();
    let got = tokio::time::timeout(Duration::from_secs(5), consumer)
        .await
        .expect("consumer resolves within the timeout")
        .expect("consumer task joins cleanly");
    assert_eq!(got, vec![0_u64, 1, 2]);
    // The publisher is measurably back-pressured: with a cap-1 queue and a 20 ms
    // consumer, the final publish cannot complete until the consumer drains an
    // earlier event, so at least one full consumer sleep elapses. (Python's
    // wall-clock `>= 30 ms` was a loose margin around this; the deterministic
    // virtual-time value under `tokio::time::pause` is one 20 ms consumer sleep —
    // see the deviation note.)
    assert!(
        elapsed >= Duration::from_millis(20),
        "elapsed = {elapsed:?}"
    );
    bus.shutdown().await;
}

#[tokio::test]
async fn unbounded_never_drops() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["t"], BackpressurePolicy::Unbounded, 0);
    for i in 0_u64..100 {
        bus.publish(mk_event("t", i)).await;
    }
    let mut got = Vec::new();
    while got.len() < 100 {
        got.push(sub.recv().await.expect("event").sequence_number);
    }
    assert_eq!(got, (0_u64..100).collect::<Vec<u64>>());
    bus.shutdown().await;
}

// --- Shutdown --------------------------------------------------------------

#[tokio::test]
async fn subscribers_exit_cleanly_on_shutdown() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let mut sub = bus.subscribe(&["t"], BackpressurePolicy::Block, 0);
    let consumer = tokio::spawn(async move {
        let mut seen = 0_u32;
        while sub.recv().await.is_some() {
            seen += 1;
        }
        seen
    });
    bus.publish(mk_event("t", 0)).await;
    tokio::task::yield_now().await;
    bus.shutdown().await;
    let seen = tokio::time::timeout(Duration::from_secs(1), consumer)
        .await
        .expect("consumer resolves within the timeout")
        .expect("consumer task joins cleanly");
    // The consumer received the pre-shutdown event, then exited cleanly on close.
    assert_eq!(seen, 1);
}
