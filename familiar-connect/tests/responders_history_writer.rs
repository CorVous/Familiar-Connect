//! Integration tests for `HistoryWriter` (subsystem 06; Python
//! `tests/test_history_writer.py`).

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::Arc;

use familiar_connect::bus::envelope::{Event, payload as wrap_payload};
use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::topics::TOPIC_DISCORD_TEXT;
use familiar_connect::identity::Author;
use familiar_connect::processors::history_writer::HistoryWriter;

use support::{discord_text_event, store, text_payload};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

#[tokio::test]
async fn persists_discord_text_event() {
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let ev = discord_text_event(text_payload(42, "hello"), "e-1");
    writer.handle(&ev, &bus()).await.unwrap();

    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].content, "hello");
    assert_eq!(turns[0].role, "user");
    let author = turns[0].author.as_ref().unwrap();
    assert_eq!(author.display_name.as_deref(), Some("Alice"));
}

#[tokio::test]
async fn dedups_on_event_id() {
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let ev = discord_text_event(text_payload(42, "once"), "dup-1");
    writer.handle(&ev, &bus()).await.unwrap();
    writer.handle(&ev, &bus()).await.unwrap();

    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert_eq!(turns.len(), 1);
}

#[tokio::test]
async fn ignores_events_for_other_familiars() {
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let mut payload = text_payload(42, "not mine");
    payload.familiar_id = "other".to_owned();
    let ev = Event {
        payload: wrap_payload(payload),
        ..discord_text_event(text_payload(42, "not mine"), "other-1")
    };
    writer.handle(&ev, &bus()).await.unwrap();
    assert_eq!(s.sync().count("fam", None).unwrap(), 0);
}

#[tokio::test]
async fn ignores_empty_content() {
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let ev = discord_text_event(text_payload(42, ""), "e-1");
    writer.handle(&ev, &bus()).await.unwrap();
    assert_eq!(s.sync().count("fam", None).unwrap(), 0);
}

#[tokio::test]
async fn author_taken_without_isinstance_check() {
    // H1: the author is threaded through onto the persisted user turn.
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let mut payload = text_payload(7, "hi");
    payload.author = Some(Author::new(
        "discord",
        "5",
        Some("bob".to_owned()),
        Some("Bob".to_owned()),
    ));
    let ev = discord_text_event(payload, "e-9");
    writer.handle(&ev, &bus()).await.unwrap();
    let turns = s.sync().recent("fam", 7, 10, None, None).unwrap();
    assert_eq!(turns[0].author.as_ref().unwrap().user_id, "5");
}

#[tokio::test]
async fn persists_pings_bot_on_user_turn() {
    // Regression (#182): a pinging message must record pings_bot=true, matching
    // TextResponder's user-turn write (responders_text.rs::persists_pings_bot_on_user_turn).
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let mut payload = text_payload(42, "you there @bot?");
    payload.pings_bot = true;
    let ev = discord_text_event(payload, "e-1");
    writer.handle(&ev, &bus()).await.unwrap();

    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert_eq!(turns.len(), 1);
    assert!(turns[0].pings_bot);
}

#[tokio::test]
async fn user_turn_pings_bot_false_without_ping() {
    // The default (non-pinging) path stays pings_bot=false.
    let s = store();
    let writer = HistoryWriter::new(Arc::clone(&s), "fam");
    let ev = discord_text_event(text_payload(42, "just chatting"), "e-1");
    writer.handle(&ev, &bus()).await.unwrap();

    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert_eq!(turns.len(), 1);
    assert!(!turns[0].pings_bot);
}

#[test]
fn processor_surface() {
    let s = store();
    let writer = HistoryWriter::new(s, "fam");
    assert_eq!(writer.name(), "history-writer");
    assert!(writer.topics().contains(&TOPIC_DISCORD_TEXT));
}
