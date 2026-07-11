//! Integration tests for `history::async_store::AsyncHistoryStore` — the async
//! facade that runs every synchronous store call on a blocking thread. Ports the
//! `TestAsyncStoreWrappers` (test_attentional_store.py) and
//! `TestAsyncStoreActivityWrappers` (test_history_store.py) suites.

use chrono::{TimeZone, Utc};
use familiar_connect::history::{AppendTurn, AsyncHistoryStore, HistoryStore};

const FAM: &str = "aria";
const CHANNEL: i64 = 200;

fn store() -> AsyncHistoryStore {
    AsyncHistoryStore::new(HistoryStore::open(":memory:").unwrap())
}

// --- attentional-stream wrappers ------------------------------------------

#[tokio::test]
async fn stage_turn_async() {
    let store = store();
    let turn = store
        .stage_turn(AppendTurn::new(FAM, 1, "user", "staged"))
        .await
        .unwrap();
    assert!(turn.consumed_at.is_none());
    store.close();
}

#[tokio::test]
async fn promote_staged_turns_async() {
    let store = store();
    store
        .stage_turn(AppendTurn::new(FAM, 1, "user", "staged"))
        .await
        .unwrap();
    let promo = store
        .promote_staged_turns(FAM.to_owned(), 1, None)
        .await
        .unwrap();
    assert_eq!(promo.consumed, 1);
    store.close();
}

#[tokio::test]
async fn get_set_focus_pointers_async() {
    let store = store();
    store
        .set_focus_pointers(FAM.to_owned(), Some(5), Some(6))
        .await
        .unwrap();
    let fp = store.get_focus_pointers(FAM.to_owned()).await.unwrap();
    let fp = fp.expect("focus pointers set");
    assert_eq!(fp.text_channel_id, Some(5));
    store.close();
}

#[tokio::test]
async fn get_set_digest_watermark_async() {
    let ts = Utc.with_ymd_and_hms(2025, 4, 1, 0, 0, 0).unwrap();
    let store = store();
    store
        .set_digest_watermark(FAM.to_owned(), ts)
        .await
        .unwrap();
    let got = store.get_digest_watermark(FAM.to_owned()).await.unwrap();
    assert_eq!(got, Some(ts));
    store.close();
}

#[tokio::test]
async fn recent_cross_channel_async() {
    let store = store();
    store
        .append_turn(AppendTurn::new(FAM, 1, "user", "msg"))
        .await
        .unwrap();
    let got = store
        .recent_cross_channel(FAM.to_owned(), 10, false)
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    store.close();
}

// --- activity + archive-window wrappers -----------------------------------

#[tokio::test]
async fn activity_round_trip_async() {
    let t0 = Utc.with_ymd_and_hms(2025, 1, 1, 8, 0, 0).unwrap();
    let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 9, 0, 0).unwrap();
    let store = store();
    let activity_id = store
        .create_activity(
            FAM.to_owned(),
            "walk".to_owned(),
            "on a walk".to_owned(),
            t0,
            t1,
            None,
        )
        .await
        .unwrap();
    let rec = store.active_activity(FAM.to_owned()).await.unwrap();
    assert_eq!(rec.map(|r| r.id), Some(activity_id));
    store
        .finish_activity(activity_id, "completed".to_owned(), t1, None)
        .await
        .unwrap();
    assert!(
        store
            .active_activity(FAM.to_owned())
            .await
            .unwrap()
            .is_none()
    );
    store.close();
}

#[tokio::test]
async fn archive_watermark_and_windows_async() {
    let store = store();
    let mut turns = Vec::new();
    for i in 0..5 {
        turns.push(
            store
                .append_turn(AppendTurn::new(FAM, CHANNEL, "user", format!("turn {i}")))
                .await
                .unwrap(),
        );
    }
    store
        .set_archive_watermark(FAM.to_owned(), CHANNEL, turns[1].id)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_archive_watermark(FAM.to_owned(), CHANNEL)
            .await
            .unwrap(),
        Some(turns[1].id)
    );
    let got = store
        .recent_cross_channel(FAM.to_owned(), 10, true)
        .await
        .unwrap();
    let contents: Vec<&str> = got.iter().map(|t| t.content.as_str()).collect();
    assert_eq!(contents, ["turn 2", "turn 3", "turn 4"]);
    let window = store
        .turns_around(FAM.to_owned(), CHANNEL, turns[2].id, 1, 1)
        .await
        .unwrap();
    let window_contents: Vec<&str> = window.iter().map(|t| t.content.as_str()).collect();
    assert_eq!(window_contents, ["turn 1", "turn 2", "turn 3"]);
    store.close();
}

#[tokio::test]
async fn promote_staged_turns_since_async() {
    let store = store();
    let departure = store
        .append_turn(AppendTurn::new(FAM, CHANNEL, "assistant", "departure"))
        .await
        .unwrap();
    store
        .stage_turn(AppendTurn::new(FAM, CHANNEL + 1, "user", "during"))
        .await
        .unwrap();
    let promo = store
        .promote_staged_turns_since(FAM.to_owned(), departure.id, None)
        .await
        .unwrap();
    assert_eq!(promo.consumed, 1);
    store.close();
}
