//! Integration tests for `DebugLoggerProcessor` (subsystem 06; Python
//! `tests/test_debug_logger_processor.py`).

#[path = "responders_support/mod.rs"]
mod support;

use familiar_connect::bus::envelope::{Event, payload as wrap_payload};
use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::protocols::{BackpressurePolicy, EventBus};
use familiar_connect::bus::topics::TOPIC_DISCORD_TEXT;
use familiar_connect::processors::debug_logger::DebugLoggerProcessor;

use support::LogCapture;

fn event(topic: &str, seq: u64) -> Event {
    Event {
        event_id: format!("e-{seq}"),
        turn_id: format!("t-{seq}"),
        session_id: "s".to_owned(),
        parent_event_ids: Vec::new(),
        topic: topic.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: seq,
        payload: wrap_payload(()),
    }
}

#[tokio::test]
async fn logs_every_event_on_subscribed_topics() {
    let bus = InProcessEventBus::new();
    bus.start().await;
    let proc = DebugLoggerProcessor::new([TOPIC_DISCORD_TEXT]);

    let topics = proc.topics();
    let mut sub = bus.subscribe(&topics, BackpressurePolicy::Block, 0);

    let capture = LogCapture::install();
    bus.publish(event(TOPIC_DISCORD_TEXT, 1)).await;
    let ev = sub.recv().await.expect("event delivered");
    proc.handle(&ev, &bus).await.unwrap();
    let out = capture.contents();
    drop(capture);
    bus.shutdown().await;

    assert!(out.contains("discord.text"), "{out}");
    assert!(out.contains("e-1"), "{out}");
}

#[test]
fn topics_and_name_surface() {
    let proc = DebugLoggerProcessor::new([TOPIC_DISCORD_TEXT]);
    assert_eq!(proc.topics(), vec![TOPIC_DISCORD_TEXT]);
    assert_eq!(proc.name(), "debug-logger");
}
