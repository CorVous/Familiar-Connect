//! Ported from Python `tests/test_people_dossier_worker.py`.
//!
//! Watermark-driven per-`canonical_key` dossier refresh: fires when a subject's
//! max `facts.id` exceeds its dossier's `last_fact_id`; compounds prior text
//! with only the new facts. The self-dossier (`ego:<id>`) applies an importance
//! filter/ordering and an opinion-preserving prompt.

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, FactSubject, NewFact};
use familiar_connect::identity::Author;
use familiar_connect::llm::{LlmClient, LlmDelta, Message};
use familiar_connect::processors::people_dossier_worker::PeopleDossierWorker;
use futures::stream::BoxStream;
use serde_json::Value;

use helpers::{ScriptedLlm, joined, store, system_text, user_text};

const DOSSIER_DEFAULT: &str = "(no more scripted replies)";

fn seed_subject_fact(store: &AsyncHistoryStore, text: &str, key: &str, display: &str) -> i64 {
    store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), text, vec![1]).subjects(vec![FactSubject {
                canonical_key: key.to_string(),
                display_at_write: display.to_string(),
            }]),
        )
        .unwrap()
        .id
}

fn seed_self_fact(store: &AsyncHistoryStore, text: &str, importance: Option<i64>) {
    let mut af = AppendFact::new("fam", Some(1), text, vec![1]).subjects(vec![FactSubject {
        canonical_key: "ego:fam".to_string(),
        display_at_write: "Sapphire".to_string(),
    }]);
    if let Some(i) = importance {
        af = af.importance(i);
    }
    store.sync().append_fact(af).unwrap();
}

fn worker(store: &Arc<AsyncHistoryStore>, llm: &Arc<ScriptedLlm>) -> PeopleDossierWorker {
    PeopleDossierWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
}

fn self_worker(store: &Arc<AsyncHistoryStore>, llm: &Arc<ScriptedLlm>) -> PeopleDossierWorker {
    worker(store, llm).familiar_display_name("Sapphire")
}

#[tokio::test]
async fn creates_dossier_for_new_subject() {
    let store = store();
    store
        .sync()
        .upsert_account(&Author::new(
            "discord",
            "1",
            Some("cass_login".into()),
            Some("Cass".into()),
        ))
        .unwrap();
    seed_subject_fact(&store, "Cass likes pho.", "discord:1", "Cass");
    let llm = ScriptedLlm::new(["Cass: enjoys pho."], DOSSIER_DEFAULT);
    worker(&store, &llm).tick().await.unwrap();

    let entry = store
        .sync()
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .expect("dossier written");
    assert!(entry.dossier_text.contains("pho"));
    assert_eq!(entry.last_fact_id, 1);
}

#[tokio::test]
async fn skips_subject_with_unchanged_watermark() {
    let store = store();
    seed_subject_fact(&store, "A.", "discord:1", "C");
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 1, "prior")
        .unwrap();
    let llm = ScriptedLlm::new(["should not be used"], DOSSIER_DEFAULT);
    worker(&store, &llm).tick().await.unwrap();

    assert_eq!(llm.call_count(), 0);
    let entry = store
        .sync()
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "prior");
}

#[tokio::test]
async fn compounds_prior_dossier_with_new_facts() {
    let store = store();
    seed_subject_fact(&store, "Cass likes pho.", "discord:1", "Cass");
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 1, "Cass likes pho.")
        .unwrap();
    seed_subject_fact(&store, "Cass moved to Toronto.", "discord:1", "Cass");
    let llm = ScriptedLlm::new(
        ["Cass enjoys pho and recently moved to Toronto."],
        DOSSIER_DEFAULT,
    );
    worker(&store, &llm).tick().await.unwrap();

    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    let body = joined(&calls[0]);
    assert!(body.contains("Cass likes pho.")); // the prior dossier
    assert!(body.contains("Toronto")); // the new fact

    let entry = store
        .sync()
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    assert!(entry.dossier_text.contains("Toronto"));
    assert_eq!(entry.last_fact_id, 2);
}

#[tokio::test]
async fn handles_multiple_subjects_in_one_tick() {
    let store = store();
    seed_subject_fact(&store, "Cass fact.", "discord:1", "Cass");
    seed_subject_fact(&store, "Aria fact.", "discord:2", "Aria");
    let llm = ScriptedLlm::new(["Cass dossier.", "Aria dossier."], DOSSIER_DEFAULT);
    worker(&store, &llm).tick().await.unwrap();

    let c = store
        .sync()
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    let a = store
        .sync()
        .get_people_dossier("fam", "discord:2")
        .unwrap()
        .unwrap();
    assert_ne!(c.dossier_text, a.dossier_text);
}

#[tokio::test]
async fn empty_llm_reply_does_not_overwrite() {
    let store = store();
    seed_subject_fact(&store, "Cass likes pho.", "discord:1", "Cass");
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 0, "keep me")
        .unwrap();
    let llm = ScriptedLlm::new(["   "], DOSSIER_DEFAULT);
    worker(&store, &llm).tick().await.unwrap();

    let entry = store
        .sync()
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "keep me");
}

#[tokio::test]
async fn builds_self_dossier_with_familiar_label() {
    let store = store();
    seed_subject_fact(
        &store,
        "Sapphire ran a gaslighting bit and felt proud.",
        "ego:fam",
        "Sapphire",
    );
    let llm = ScriptedLlm::new(
        ["Sapphire: enjoys running provocative bits."],
        DOSSIER_DEFAULT,
    );
    self_worker(&store, &llm).tick().await.unwrap();

    let entry = store
        .sync()
        .get_people_dossier("fam", "ego:fam")
        .unwrap()
        .unwrap();
    assert_eq!(entry.last_fact_id, 1);
    let body = joined(&llm.calls()[0]);
    assert!(body.contains("Sapphire"));
    assert!(!body.contains("ego:fam"));
}

#[tokio::test]
async fn self_dossier_strips_echoed_importance_tag() {
    let store = store();
    seed_subject_fact(
        &store,
        "Sapphire guards her autonomy fiercely.",
        "ego:fam",
        "Sapphire",
    );
    let llm = ScriptedLlm::new(
        ["(importance 8) Sapphire guards her autonomy and keeps her own records."],
        DOSSIER_DEFAULT,
    );
    self_worker(&store, &llm).tick().await.unwrap();

    let entry = store
        .sync()
        .get_people_dossier("fam", "ego:fam")
        .unwrap()
        .unwrap();
    assert!(!entry.dossier_text.contains("(importance"));
    assert!(entry.dossier_text.contains("Sapphire guards her autonomy"));
}

#[tokio::test]
async fn self_dossier_prompt_preserves_opinions() {
    let store = store();
    seed_subject_fact(
        &store,
        "Sapphire warmed to SpaceFish and stays wary of KaillaDame.",
        "ego:fam",
        "Sapphire",
    );
    let llm = ScriptedLlm::new(
        ["Sapphire: warm to SpaceFish, wary of KaillaDame."],
        DOSSIER_DEFAULT,
    );
    self_worker(&store, &llm).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("opinion") || system.contains("stance"));
    assert!(!system.contains("drop transient feelings"));
}

#[tokio::test]
async fn self_dossier_excludes_low_importance_texture() {
    let store = store();
    seed_self_fact(
        &store,
        "Sapphire is fiercely protective of her autonomy.",
        Some(9),
    );
    seed_self_fact(
        &store,
        "Sapphire was briefly curious about dream BLTs.",
        Some(2),
    );
    seed_self_fact(&store, "Sapphire keeps records out of habit.", None); // legacy NULL
    let llm = ScriptedLlm::new(["Sapphire: autonomous, keeps records."], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();

    let body = joined(&llm.calls()[0]);
    assert!(body.contains("autonomy")); // durable kept
    assert!(body.contains("keeps records")); // legacy NULL kept
    assert!(!body.contains("dream BLTs")); // texture excluded
}

#[tokio::test]
async fn self_dossier_orders_facts_by_importance_desc() {
    let store = store();
    seed_self_fact(&store, "fact-five.", Some(5));
    seed_self_fact(&store, "fact-nine.", Some(9));
    seed_self_fact(&store, "fact-seven.", Some(7));
    seed_self_fact(&store, "fact-null.", None);
    let llm = ScriptedLlm::new(["ok"], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();

    let body = user_text(&llm.calls()[0]);
    let idx = |needle: &str| body.find(needle).unwrap();
    // 9, 7, then the 5-band (5 before NULL, stable).
    assert!(idx("fact-nine.") < idx("fact-seven."));
    assert!(idx("fact-seven.") < idx("fact-five."));
    assert!(idx("fact-five.") < idx("fact-null."));
}

#[tokio::test]
async fn self_dossier_annotates_and_biases_by_importance() {
    let store = store();
    seed_self_fact(&store, "Sapphire guards her autonomy.", Some(9));
    let llm = ScriptedLlm::new(["ok"], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();

    let body = user_text(&llm.calls()[0]);
    assert!(body.contains("- (importance 9) Sapphire guards her autonomy."));
    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("importance"));
    assert!(system.contains("weight higher-importance"));
}

#[tokio::test]
async fn self_dossier_null_importance_renders_untagged() {
    let store = store();
    seed_self_fact(&store, "Sapphire keeps records out of habit.", None);
    let llm = ScriptedLlm::new(["ok"], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();

    let body = user_text(&llm.calls()[0]);
    assert!(body.contains("- Sapphire keeps records out of habit."));
    assert!(!body.contains("(importance"));
}

#[tokio::test]
async fn non_self_dossier_no_importance_tags_or_bias() {
    let store = store();
    store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes pho.", vec![1])
                .subjects(vec![FactSubject {
                    canonical_key: "discord:9".to_string(),
                    display_at_write: "Aria".to_string(),
                }])
                .importance(8),
        )
        .unwrap();
    let llm = ScriptedLlm::new(["Aria: likes pho."], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();

    let body = user_text(&llm.calls()[0]);
    assert!(body.contains("- Aria likes pho.")); // plain render, no tag
    assert!(!body.contains("(importance"));
    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(!system.contains("weight higher-importance"));
}

#[tokio::test]
async fn non_self_dossier_keeps_all_importances() {
    let store = store();
    store
        .sync()
        .append_fact(
            AppendFact::new(
                "fam",
                Some(1),
                "Aria mentioned a minor preference.",
                vec![1],
            )
            .subjects(vec![FactSubject {
                canonical_key: "discord:9".to_string(),
                display_at_write: "Aria".to_string(),
            }])
            .importance(2),
        )
        .unwrap();
    let llm = ScriptedLlm::new(["Aria: has a minor preference."], DOSSIER_DEFAULT);
    self_worker(&store, &llm).tick().await.unwrap();
    let body = joined(&llm.calls()[0]);
    assert!(body.contains("minor preference")); // low-importance non-self fact kept
}

#[tokio::test]
async fn no_subjects_means_no_llm_call() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "A subject-less fact.",
            vec![1],
        ))
        .unwrap();
    let llm = ScriptedLlm::new(["unused"], DOSSIER_DEFAULT);
    worker(&store, &llm).tick().await.unwrap();

    assert_eq!(llm.call_count(), 0);
}

/// An LLM double that supersedes `old_id` → `new_id` (deleting the subject's
/// dossier row as cache-invalidation) on its FIRST `chat` call, modelling a
/// concurrent `FactSupersedeWorker` firing DURING the dossier rebuild's LLM
/// call — the interleave that #130 item 1 hardens against.
struct SupersedingLlm {
    store: Arc<AsyncHistoryStore>,
    old_id: i64,
    new_id: i64,
    reply: String,
    calls: AtomicUsize,
}

impl SupersedingLlm {
    fn new(store: Arc<AsyncHistoryStore>, old_id: i64, new_id: i64, reply: &str) -> Arc<Self> {
        Arc::new(Self {
            store,
            old_id,
            new_id,
            reply: reply.to_owned(),
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl LlmClient for SupersedingLlm {
    async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
        // Fire the racing supersede exactly once, on the first rebuild call.
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.store
                .sync()
                .supersede("fam", &[self.old_id], NewFact::Repoint(self.new_id))
                .unwrap();
        }
        Ok(Message::new("assistant", self.reply.clone()))
    }
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        Ok(Box::pin(futures::stream::empty()))
    }
    fn slot(&self) -> Option<&str> {
        Some("background")
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

#[tokio::test]
async fn rebuild_racing_supersede_delete_does_not_resurrect_dossier() {
    // #130 item 1: seed a dossier at watermark = old_id plus a newer fact that
    // triggers a rebuild. A supersede deletes the dossier row mid-LLM-call; the
    // stale CAS write must be dropped (no orphan), and the next tick rebuilds
    // the row cleanly from prior=None.
    let store = store();
    let old_id = seed_subject_fact(&store, "Aria loves hiking.", "discord:A", "Aria");
    store
        .sync()
        .put_people_dossier("fam", "discord:A", old_id, "Aria loves hiking.")
        .unwrap();
    let new_id = seed_subject_fact(&store, "Aria hates hiking now.", "discord:A", "Aria");

    let llm = SupersedingLlm::new(
        Arc::clone(&store),
        old_id,
        new_id,
        "Aria used to love hiking.",
    );
    let worker = PeopleDossierWorker::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam");

    // Tick 1: rebuild reads prior (wm=old_id), supersede deletes the row during
    // the LLM call, CAS at old_id finds no row → dropped, not resurrected.
    worker.tick().await.unwrap();
    assert!(
        store
            .sync()
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none(),
        "raced write must not resurrect the invalidated dossier",
    );

    // Tick 2: prior=None, so the worker rebuilds cleanly against the surviving
    // (repointed) fact — the subject self-heals.
    worker.tick().await.unwrap();
    let entry = store
        .sync()
        .get_people_dossier("fam", "discord:A")
        .unwrap()
        .expect("dossier rebuilt on the next tick");
    assert_eq!(entry.last_fact_id, new_id);
    assert!(entry.dossier_text.contains("Aria"));
}
