//! Ported from Python `tests/test_fact_supersede_worker.py`.

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::sync::Arc;

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, FactSubject};
use familiar_connect::llm::LlmClient;
use familiar_connect::processors::fact_supersede_worker::FactSupersedeWorker;

use helpers::{ScriptedLlm, store};

const SUPERSEDE_DEFAULT: &str = "{\"superseded_ids\": []}";

fn ids_json(ids: &[i64]) -> String {
    serde_json::json!({ "superseded_ids": ids }).to_string()
}

fn subjects() -> Vec<FactSubject> {
    vec![FactSubject {
        canonical_key: "discord:111".to_string(),
        display_at_write: "Aria".to_string(),
    }]
}

/// Pre-existing fact, optional watermark prime, then a new fact arrives.
/// Returns `(old_id, new_id)`.
fn seed_subject_facts(
    store: &AsyncHistoryStore,
    worker: Option<&FactSupersedeWorker>,
) -> (i64, i64) {
    let old = store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria loves hiking.", vec![1]).subjects(subjects()),
        )
        .unwrap();
    if let Some(w) = worker {
        w.prime_watermark().unwrap();
    }
    let new = store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria hates hiking now.", vec![2]).subjects(subjects()),
        )
        .unwrap();
    (old.id, new.id)
}

fn current_ids(store: &AsyncHistoryStore) -> Vec<i64> {
    let mut ids: Vec<i64> = store
        .sync()
        .recent_facts("fam", 10, false, None)
        .unwrap()
        .iter()
        .map(|f| f.id)
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn no_new_facts_is_noop() {
    let store = store();
    seed_subject_facts(&store, None);
    let llm = ScriptedLlm::new([ids_json(&[])], SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    // Pre-advance the watermark past everything.
    worker.prime_watermark().unwrap();
    worker.tick().await.unwrap();
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn supersedes_prior_when_llm_flags_it() {
    let store = store();
    let llm = ScriptedLlm::new(Vec::<String>::new(), SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    let (old_id, new_id) = seed_subject_facts(&store, Some(&worker));
    llm.push_reply(ids_json(&[old_id]));
    worker.tick().await.unwrap();

    // Old retired -> only the new one is current.
    assert_eq!(current_ids(&store), vec![new_id]);
    // And the old row carries the supersede metadata.
    let all = store.sync().recent_facts("fam", 10, true, None).unwrap();
    let old_row = all.iter().find(|f| f.id == old_id).unwrap();
    assert_eq!(old_row.superseded_by, Some(new_id));
    assert!(old_row.superseded_at.is_some());
}

#[tokio::test]
async fn empty_reply_leaves_facts_untouched() {
    let store = store();
    let llm = ScriptedLlm::new([ids_json(&[])], SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    let (old_id, new_id) = seed_subject_facts(&store, Some(&worker));
    worker.tick().await.unwrap();
    let current = current_ids(&store);
    assert_eq!(current.len(), 2);
    assert!(current.contains(&old_id) && current.contains(&new_id));
}

#[tokio::test]
async fn hallucinated_id_outside_candidate_set_ignored() {
    let store = store();
    // 9999 isn't a real fact id; supersede must skip it, not crash.
    let llm = ScriptedLlm::new([ids_json(&[9999])], SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    let (old_id, new_id) = seed_subject_facts(&store, Some(&worker));
    worker.tick().await.unwrap();
    let current = current_ids(&store);
    assert_eq!(current.len(), 2);
    assert!(current.contains(&old_id) && current.contains(&new_id));
}

#[tokio::test]
async fn bad_json_is_swallowed() {
    let store = store();
    let llm = ScriptedLlm::new(["not json at all"], SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    let (old_id, new_id) = seed_subject_facts(&store, Some(&worker));
    worker.tick().await.unwrap();
    let current = current_ids(&store);
    assert_eq!(current.len(), 2);
    assert!(current.contains(&old_id) && current.contains(&new_id));
}

#[tokio::test]
async fn does_not_propose_self_supersede() {
    // The worker must never instruct the LLM to retire the new fact itself.
    let store = store();
    let llm = ScriptedLlm::new(Vec::<String>::new(), SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    let (old_id, new_id) = seed_subject_facts(&store, Some(&worker));
    // If the LLM erroneously names the new fact, the worker filters it.
    llm.push_reply(ids_json(&[new_id]));
    worker.tick().await.unwrap();
    let current = current_ids(&store);
    assert!(current.contains(&new_id));
    assert!(current.contains(&old_id));
}

#[tokio::test]
async fn advances_watermark_so_next_tick_is_noop() {
    let store = store();
    let llm = ScriptedLlm::new([ids_json(&[]), ids_json(&[])], SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    seed_subject_facts(&store, Some(&worker));
    worker.tick().await.unwrap();
    let first = llm.call_count();
    worker.tick().await.unwrap();
    assert_eq!(
        llm.call_count(),
        first,
        "second tick should not re-evaluate a fact already seen"
    );
}

#[tokio::test]
async fn skips_facts_with_no_subjects() {
    // Facts without canonical-key subjects can't be paired with priors.
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "The weather is nice today.",
            vec![1],
        ))
        .unwrap();
    let llm = ScriptedLlm::new(Vec::<String>::new(), SUPERSEDE_DEFAULT);
    let worker = FactSupersedeWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");
    worker.tick().await.unwrap();
    assert_eq!(llm.call_count(), 0);
}
