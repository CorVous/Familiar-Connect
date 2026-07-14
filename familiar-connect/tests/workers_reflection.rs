//! Ported from Python `tests/test_reflection_worker.py`.
//!
//! Watermark-driven reflection writes — fires when enough new turns have
//! accumulated since the previous reflection's watermark; persists each answer
//! with `cited_turn_ids` / `cited_fact_ids` provenance.

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::sync::Arc;

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, AppendTurn, Reflection};
use familiar_connect::llm::LlmClient;
use familiar_connect::processors::reflection_worker::ReflectionWorker;

use helpers::{ScriptedLlm, store, user_text};

fn seed_turns(store: &AsyncHistoryStore, count: i64) -> Vec<i64> {
    let mut out = Vec::new();
    for i in 0..count {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        let t = store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, role, format!("message {i}")))
            .unwrap();
        out.push(t.id);
    }
    out
}

fn seed_facts(store: &AsyncHistoryStore, count: i64) -> Vec<i64> {
    let mut out = Vec::new();
    for i in 0..count {
        let f = store
            .sync()
            .append_fact(AppendFact::new(
                "fam",
                Some(1),
                format!("fact {i}"),
                vec![i + 1],
            ))
            .unwrap();
        out.push(f.id);
    }
    out
}

fn reflections(store: &AsyncHistoryStore) -> Vec<Reflection> {
    store.sync().recent_reflections("fam", None, 10).unwrap()
}

fn worker(store: &Arc<AsyncHistoryStore>, llm: &Arc<ScriptedLlm>) -> ReflectionWorker {
    ReflectionWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .turns_threshold(20)
}

#[tokio::test]
async fn noop_below_threshold() {
    let store = store();
    seed_turns(&store, 5); // below threshold
    let llm = ScriptedLlm::new(["should not be called"], "[]");
    worker(&store, &llm).tick().await.unwrap();
    assert_eq!(llm.call_count(), 0);
    assert!(reflections(&store).is_empty());
}

#[tokio::test]
async fn writes_reflection_when_threshold_crossed() {
    let store = store();
    let turn_ids = seed_turns(&store, 25);
    let fact_ids = seed_facts(&store, 4);
    let reply = format!(
        "[{{\"text\": \"Crew morale dipped after Friday.\", \
         \"cited_turn_ids\": [{}, {}], \"cited_fact_ids\": [{}]}}]",
        turn_ids[3], turn_ids[10], fact_ids[1]
    );
    let llm = ScriptedLlm::new([reply], "[]");
    worker(&store, &llm).tick().await.unwrap();

    let rows = reflections(&store);
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert!(r.text.contains("morale"));
    assert!(r.cited_turn_ids.contains(&turn_ids[3]));
    assert!(r.cited_fact_ids.contains(&fact_ids[1]));
    assert_eq!(r.last_turn_id, *turn_ids.last().unwrap());
    assert_eq!(r.last_fact_id, *fact_ids.last().unwrap());
}

#[tokio::test]
async fn does_not_refire_until_more_turns_arrive() {
    let store = store();
    let turn_ids = seed_turns(&store, 25);
    let first_reply = format!(
        "[{{\"text\": \"first reflection\", \"cited_turn_ids\": [{}], \"cited_fact_ids\": []}}]",
        turn_ids[0]
    );
    let llm = ScriptedLlm::new([first_reply, "[]".to_string()], "[]");
    let w = worker(&store, &llm);
    w.tick().await.unwrap();
    assert_eq!(llm.call_count(), 1);
    // No new turns — should be a noop.
    w.tick().await.unwrap();
    assert_eq!(llm.call_count(), 1);
    // Now add 25 more turns to cross the threshold.
    seed_turns(&store, 25);
    w.tick().await.unwrap();
    assert_eq!(llm.call_count(), 2);
}

#[tokio::test]
async fn writes_multiple_rows_when_llm_returns_multiple() {
    let store = store();
    let turn_ids = seed_turns(&store, 25);
    let reply = format!(
        "[{{\"text\": \"first\", \"cited_turn_ids\": [{}], \"cited_fact_ids\": []}},\
          {{\"text\": \"second\", \"cited_turn_ids\": [{}], \"cited_fact_ids\": []}}]",
        turn_ids[0], turn_ids[1]
    );
    let llm = ScriptedLlm::new([reply], "[]");
    worker(&store, &llm).tick().await.unwrap();

    let texts: std::collections::BTreeSet<String> =
        reflections(&store).into_iter().map(|r| r.text).collect();
    assert_eq!(
        texts,
        ["first".to_string(), "second".to_string()]
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn drops_rows_with_unknown_citations() {
    let store = store();
    let turn_ids = seed_turns(&store, 25);
    // 999 isn't a real turn id — dropped from the row, but the row lands if any
    // cited id is valid.
    let reply = format!(
        "[{{\"text\": \"valid\", \"cited_turn_ids\": [{}, 999], \"cited_fact_ids\": []}}]",
        turn_ids[0]
    );
    let llm = ScriptedLlm::new([reply], "[]");
    worker(&store, &llm).tick().await.unwrap();
    let rows = reflections(&store);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].cited_turn_ids, vec![turn_ids[0]]);
}

#[tokio::test]
async fn skips_row_with_empty_text() {
    let store = store();
    let turn_ids = seed_turns(&store, 25);
    let reply = format!(
        "[{{\"text\": \"\", \"cited_turn_ids\": [{}], \"cited_fact_ids\": []}}]",
        turn_ids[0]
    );
    let llm = ScriptedLlm::new([reply], "[]");
    worker(&store, &llm).tick().await.unwrap();
    assert!(reflections(&store).is_empty());
}

#[tokio::test]
async fn handles_malformed_llm_output() {
    let store = store();
    seed_turns(&store, 25);
    let llm = ScriptedLlm::new(["not json at all"], "[]");
    // Should not raise.
    worker(&store, &llm).tick().await.unwrap();
    assert!(reflections(&store).is_empty());
}

#[tokio::test]
async fn watermark_advances_when_llm_returns_empty() {
    let store = store();
    seed_turns(&store, 25);
    let llm = ScriptedLlm::new(["[]", "[]"], "[]");
    let w = worker(&store, &llm);
    w.tick().await.unwrap();
    // Add 5 more — still below threshold relative to the new mark.
    seed_turns(&store, 5);
    w.tick().await.unwrap();
    // Only one call: the second tick noops because the watermark advanced.
    assert_eq!(llm.call_count(), 1);
}

#[tokio::test]
async fn watermark_advances_when_malformed_reply() {
    let store = store();
    seed_turns(&store, 25);
    // Enough bad replies to outlast the structured-output retries.
    let llm = ScriptedLlm::new(["not json", "not json", "not json", "not json"], "[]");
    let w = worker(&store, &llm);
    w.tick().await.unwrap();
    let after_first = llm.call_count();
    seed_turns(&store, 5);
    w.tick().await.unwrap();
    // Second tick is a noop — the watermark advanced despite the garbage reply.
    assert_eq!(llm.call_count(), after_first);
}

#[tokio::test]
async fn caps_turn_window_per_tick() {
    let store = store();
    let turn_ids = seed_turns(&store, 500);
    let llm = ScriptedLlm::new(["[]"], "[]");
    let w = ReflectionWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .turns_threshold(20)
        .max_turns_per_tick(50);
    w.tick().await.unwrap();

    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    let prompt = user_text(&calls[0]);
    // Count "- id=" markers — one per turn included.
    assert_eq!(prompt.matches("- id=").count(), 50);
    // Most recent turns present; oldest absent.
    assert!(prompt.contains(&format!("id={} ", turn_ids[turn_ids.len() - 1])));
    assert!(!prompt.contains(&format!("id={} ", turn_ids[0])));
    assert!(prompt.contains(&format!("id={} ", turn_ids[turn_ids.len() - 50])));
    assert!(!prompt.contains(&format!("id={} ", turn_ids[turn_ids.len() - 51])));
}
