//! Ported from Python `tests/test_sleep_pass_orchestrators.py` — the
//! `execute_consolidation` / `execute_opinion_formation` plan→apply
//! orchestrators: dry-run-never-mutates (facts + watermark), apply mutates +
//! advances, and the rail-rejection WARNING signal (the surviving audit trail).

#[path = "sleep_helpers/mod.rs"]
mod helpers;

use std::sync::{Arc, Mutex};

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, AppendTurn, FactSubject};
use familiar_connect::identity::is_ego_key;
use familiar_connect::sleep::consolidation::{
    DEFAULT_FACTS_MAX, DEFAULT_RETIRE_CAP, DEFAULT_TURNS_MAX,
};
use familiar_connect::sleep::maintenance::{execute_consolidation, execute_opinion_formation};
use familiar_connect::sleep::opinion_formation::DEFAULT_OPINION_CAP;
use serde_json::json;

use helpers::{ScriptedLlm, store};

fn consolidation_store() -> Arc<AsyncHistoryStore> {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    s.sync()
        .append_fact(AppendFact::new("fam", Some(1), "noise", vec![1]))
        .unwrap();
    s.sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes tea.", vec![1]).subjects(vec![
                FactSubject {
                    canonical_key: "discord:A".to_owned(),
                    display_at_write: "Aria".to_owned(),
                },
            ]),
        )
        .unwrap();
    s
}

fn retire_reply() -> String {
    json!({"retire": [{"fact_ids": [1], "reason": "noise"}], "rewrite": []}).to_string()
}

fn current_texts(s: &AsyncHistoryStore) -> std::collections::HashSet<String> {
    s.sync()
        .recent_facts("fam", 10, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect()
}

// ---------------------------------------------------------------------------
// execute_consolidation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn consolidation_dry_run_does_not_mutate() {
    let s = consolidation_store();
    let reply = retire_reply();
    let llm = ScriptedLlm::new(&[&reply]);
    let plan = execute_consolidation(
        &s,
        &llm,
        "fam",
        Some("Sapphire"),
        false,
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "",
    )
    .await
    .unwrap();
    // live facts untouched.
    assert_eq!(
        current_texts(&s),
        ["noise", "Aria likes tea."]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    );
    // no watermark written on dry-run.
    assert!(s.sync().get_sleep_watermark("fam").unwrap().is_none());
    assert_eq!(plan.retire.len(), 1);
}

#[tokio::test]
async fn consolidation_apply_mutates_and_advances_watermark() {
    let s = consolidation_store();
    let reply = retire_reply();
    let llm = ScriptedLlm::new(&[&reply]);
    execute_consolidation(
        &s,
        &llm,
        "fam",
        Some("Sapphire"),
        true,
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "",
    )
    .await
    .unwrap();
    assert!(!current_texts(&s).contains("noise"));
    assert!(s.sync().get_sleep_watermark("fam").unwrap().is_some());
}

#[tokio::test]
async fn rail_rejection_is_logged_with_rail_name() {
    let s = consolidation_store();
    // fact id 999 does not exist → the unknown_id rail blocks the retire.
    let reply =
        json!({"retire": [{"fact_ids": [999], "reason": "phantom"}], "rewrite": []}).to_string();
    let llm = ScriptedLlm::new(&[&reply]);

    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(VecWriter(buf.clone()))
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .finish();
    {
        let _guard = tracing::subscriber::set_default(subscriber);
        execute_consolidation(
            &s,
            &llm,
            "fam",
            Some("Sapphire"),
            false,
            DEFAULT_FACTS_MAX,
            DEFAULT_TURNS_MAX,
            DEFAULT_RETIRE_CAP,
            "",
        )
        .await
        .unwrap();
    }
    let captured = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    assert!(captured.contains("unknown_id"), "{captured}");
}

// ---------------------------------------------------------------------------
// execute_opinion_formation
// ---------------------------------------------------------------------------

fn opinion_store() -> Arc<AsyncHistoryStore> {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music, fight me",
        ))
        .unwrap();
    s
}

fn opinion_replies() -> (String, String) {
    (
        json!({"candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]}).to_string(),
        json!({
            "opinions": [{
                "text": "Sapphire is fiercely protective of lo-fi as real music.",
                "source_turn_ids": [1],
                "reason": "defended it",
            }]
        })
        .to_string(),
    )
}

#[tokio::test]
async fn opinion_dry_run_does_not_mint() {
    let s = opinion_store();
    let (cand, synth) = opinion_replies();
    let llm = ScriptedLlm::new(&[&cand, &synth]);
    let plan = execute_opinion_formation(
        &s,
        &llm,
        "fam",
        Some("Sapphire"),
        "UTC",
        false,
        &[],
        DEFAULT_OPINION_CAP,
        "",
        "",
    )
    .await
    .unwrap();
    assert!(
        s.sync()
            .recent_facts("fam", 10, false, None)
            .unwrap()
            .is_empty()
    );
    assert!(s.sync().get_sleep_watermark("fam").unwrap().is_none());
    assert_eq!(plan.opinions.len(), 1);
}

#[tokio::test]
async fn opinion_apply_mints_self_facts() {
    let s = opinion_store();
    let (cand, synth) = opinion_replies();
    let llm = ScriptedLlm::new(&[&cand, &synth]);
    execute_opinion_formation(
        &s,
        &llm,
        "fam",
        Some("Sapphire"),
        "UTC",
        true,
        &[],
        DEFAULT_OPINION_CAP,
        "",
        "",
    )
    .await
    .unwrap();
    let facts = s.sync().recent_facts("fam", 10, false, None).unwrap();
    assert_eq!(facts.len(), 1);
    assert!(is_ego_key(&facts[0].subjects[0].canonical_key));
    let wm = s.sync().get_sleep_watermark("fam").unwrap().unwrap();
    assert_eq!(wm.last_turn_id, 1);
}

// --- tracing capture writer -------------------------------------------------

#[derive(Clone)]
struct VecWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for VecWriter {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}
