//! Ported from Python `tests/test_summary_worker.py`.
//!
//! Watermark-driven regeneration of the focus-stream rolling summary (the
//! consumed cross-channel stream the familiar attended to).

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::sync::Arc;

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendTurn, FOCUS_STREAM_CHANNEL_ID, SummaryEntry};
use familiar_connect::llm::LlmClient;
use familiar_connect::processors::summary_worker::SummaryWorker;

use helpers::{ScriptedLlm, joined, store};

const SUMMARY_DEFAULT: &str = "(nothing to summarise)";

fn seed_turns(store: &AsyncHistoryStore, count: i64, channel_id: i64, consumed: bool) {
    for i in 0..count {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        store
            .sync()
            .append_turn(
                AppendTurn::new("fam", channel_id, role, format!("message {i}")).consumed(consumed),
            )
            .unwrap();
    }
}

fn focus_summary(store: &AsyncHistoryStore) -> Option<SummaryEntry> {
    store
        .sync()
        .get_summary("fam", FOCUS_STREAM_CHANNEL_ID)
        .unwrap()
}

fn worker(
    store: &Arc<AsyncHistoryStore>,
    llm: &Arc<ScriptedLlm>,
    backfill: Option<i64>,
) -> SummaryWorker {
    let mut w = SummaryWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .turns_threshold(10);
    if let Some(cap) = backfill {
        w = w.backfill_cap(cap);
    }
    w
}

#[tokio::test]
async fn summarises_consumed_cross_channel_stream() {
    let store = store();
    seed_turns(&store, 6, 1, true);
    seed_turns(&store, 6, 2, true); // 12 consumed across two channels
    let llm = ScriptedLlm::new(["Cross-channel: hi everywhere."], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();

    let summary = focus_summary(&store).expect("focus summary written");
    assert_eq!(summary.last_summarised_id, 12);
    assert!(summary.last_consumed_at.is_some());
    assert!(!summary.summary_text.is_empty());
}

#[tokio::test]
async fn noop_below_threshold() {
    let store = store();
    seed_turns(&store, 3, 1, true);
    let llm = ScriptedLlm::new(["should not be used"], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();

    assert!(focus_summary(&store).is_none());
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn ignores_staged_turns() {
    let store = store();
    seed_turns(&store, 3, 1, true); // consumed
    seed_turns(&store, 20, 2, false); // staged, unfocused
    let llm = ScriptedLlm::new(["should not fire"], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();

    assert!(focus_summary(&store).is_none()); // 3 consumed < threshold
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn compounds_prior_summary_into_prompt() {
    let store = store();
    seed_turns(&store, 12, 1, true);
    let llm = ScriptedLlm::new(
        [
            "Round 1 summary: early chat.",
            "Round 2 summary: extended with 10 more.",
        ],
        SUMMARY_DEFAULT,
    );
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();
    seed_turns(&store, 10, 1, true);
    w.tick().await.unwrap();

    let summary = focus_summary(&store).expect("summary");
    assert!(summary.summary_text.contains("Round 2"));
    let calls = llm.calls();
    assert!(joined(&calls[1]).contains("Round 1 summary"));
}

#[tokio::test]
async fn backfill_cap_bounds_first_run() {
    let store = store();
    seed_turns(&store, 500, 1, true);
    let llm = ScriptedLlm::new(["batch 1", "batch 2"], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, Some(200));

    w.tick().await.unwrap();
    assert_eq!(focus_summary(&store).unwrap().last_summarised_id, 200);

    w.tick().await.unwrap();
    assert_eq!(focus_summary(&store).unwrap().last_summarised_id, 400);
}

#[tokio::test]
async fn late_promoted_turn_picked_up_on_next_tick() {
    let store = store();
    seed_turns(&store, 12, 1, true); // consumed; ids 1..12
    let llm = ScriptedLlm::new(["round 1", "round 2 with dormant content"], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();
    assert_eq!(focus_summary(&store).unwrap().last_summarised_id, 12);

    // 10 staged turns in a dormant channel; consumed_at NULL.
    for i in 0..10 {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 2, "user", format!("dormant {i}")).consumed(false))
            .unwrap();
    }
    // Nothing consumed yet -> noop.
    w.tick().await.unwrap();
    assert_eq!(focus_summary(&store).unwrap().last_summarised_id, 12);

    // Focus shifts -> promote; consumed_at = NOW (> watermark).
    store.sync().promote_staged_turns("fam", 2, None).unwrap();
    w.tick().await.unwrap();
    assert_eq!(focus_summary(&store).unwrap().last_summarised_id, 22);
    let calls = llm.calls();
    assert!(joined(&calls[1]).contains("dormant"));
}

#[tokio::test]
async fn per_channel_summary_no_longer_written() {
    let store = store();
    seed_turns(&store, 12, 1, true);
    let llm = ScriptedLlm::new(["focus summary"], SUMMARY_DEFAULT);
    let w = worker(&store, &llm, None);
    w.tick().await.unwrap();

    assert!(store.sync().get_summary("fam", 1).unwrap().is_none());
    assert!(focus_summary(&store).is_some());
}
