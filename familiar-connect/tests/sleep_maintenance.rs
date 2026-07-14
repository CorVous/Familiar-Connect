//! Ported from Python `tests/test_maintenance_passes.py` — registry
//! order/unknown-raises/known-set/DEFAULT_PASSES, the denylist data-flow
//! (consolidation's retired text reaches the opinion stance prompt), and the
//! default run applying both passes.
//!
//! The Python denylist test monkeypatches `execute_opinion_formation` to spy on
//! the `denylist` kwarg. Rust has no monkeypatch: we assert the equivalent
//! observable — the retired fact's text arrives in the stance prompt's KNOWN
//! BITS block (the ONLY place the denylist is consumed).

#[path = "sleep_helpers/mod.rs"]
mod helpers;

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, AppendTurn, FactSubject};
use familiar_connect::identity::is_ego_key;
use familiar_connect::llm::LlmClient;
use familiar_connect::sleep::maintenance::{
    CONSOLIDATION_PASS, DEFAULT_PASSES, MaintenanceContext, OPINION_PASS, create_passes,
    known_passes, run_passes,
};
use serde_json::json;

use helpers::{ScriptedLlm, store};

fn maintenance_store() -> Arc<AsyncHistoryStore> {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music, fight me",
        ))
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

fn ctx(store: Arc<AsyncHistoryStore>, llm: Arc<dyn LlmClient>, apply: bool) -> MaintenanceContext {
    MaintenanceContext::new(store, llm, "fam", Some("Sapphire".to_owned()), "UTC", apply)
}

fn consolidation_reply() -> String {
    json!({"retire": [{"fact_ids": [1], "reason": "noise"}], "rewrite": []}).to_string()
}

fn opinion_replies() -> (String, String) {
    (
        json!({"candidates": [{"text": "defends lo-fi", "turn_ids": [2]}]}).to_string(),
        json!({
            "opinions": [{
                "text": "Sapphire is fiercely protective of lo-fi as real music.",
                "source_turn_ids": [2],
                "reason": "defended it",
            }]
        })
        .to_string(),
    )
}

// ---------------------------------------------------------------------------
// registry shape
// ---------------------------------------------------------------------------

#[test]
fn create_passes_returns_in_order() {
    let llm: Arc<dyn LlmClient> = ScriptedLlm::shared(&[]);
    let c = ctx(store(), llm, false);
    let passes = create_passes(&[CONSOLIDATION_PASS, OPINION_PASS], &c).unwrap();
    assert_eq!(
        passes.iter().map(|p| p.name()).collect::<Vec<_>>(),
        vec![CONSOLIDATION_PASS, OPINION_PASS]
    );
}

#[test]
fn create_passes_unknown_name_raises() {
    let llm: Arc<dyn LlmClient> = ScriptedLlm::shared(&[]);
    let c = ctx(store(), llm, false);
    let err = create_passes(&["nope"], &c)
        .err()
        .expect("unknown pass errors");
    assert!(err.to_string().contains("nope"));
}

#[test]
fn known_passes_lists_registered() {
    assert_eq!(
        known_passes(),
        [CONSOLIDATION_PASS, OPINION_PASS]
            .iter()
            .map(|s| (*s).to_owned())
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn default_passes_is_consolidation_then_opinion() {
    assert_eq!(DEFAULT_PASSES, [CONSOLIDATION_PASS, OPINION_PASS]);
}

// ---------------------------------------------------------------------------
// denylist data-flow
// ---------------------------------------------------------------------------

#[tokio::test]
async fn consolidation_retirement_reaches_opinion_denylist() {
    let s = maintenance_store();
    let (cand, synth) = opinion_replies();
    let c_reply = consolidation_reply();
    let llm = ScriptedLlm::shared(&[&c_reply, &cand, &synth]);
    let c = ctx(s, llm.clone(), true);
    let passes = create_passes(&DEFAULT_PASSES, &c).unwrap();
    run_passes(&passes, None).await.unwrap();

    // fact id 1 ("noise") retired this run reaches the opinion deny-list — it
    // shows up in the stance prompt's KNOWN BITS block (call 1 = the stance
    // call: call 0 is consolidation, call 2 is synthesis).
    let calls = llm.calls();
    let stance_system = calls[1][0].content_str();
    assert!(stance_system.contains("KNOWN BITS"));
    assert!(stance_system.contains("- noise"));
}

// ---------------------------------------------------------------------------
// default run applies both passes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_run_executes_consolidation_then_opinion() {
    let s = maintenance_store();
    let (cand, synth) = opinion_replies();
    let c_reply = consolidation_reply();
    let llm = ScriptedLlm::shared(&[&c_reply, &cand, &synth]);
    let c = ctx(s.clone(), llm, true);
    let passes = create_passes(&DEFAULT_PASSES, &c).unwrap();
    run_passes(&passes, None).await.unwrap();

    let facts = s.sync().recent_facts("fam", 10, false, None).unwrap();
    let texts: HashSet<String> = facts.iter().map(|f| f.text.clone()).collect();
    // consolidation retired "noise".
    assert!(!texts.contains("noise"));
    // opinion pass recorded exactly one ego opinion.
    let opinions = facts
        .iter()
        .filter(|f| {
            f.subjects
                .first()
                .is_some_and(|s| is_ego_key(&s.canonical_key))
        })
        .count();
    assert_eq!(opinions, 1);
}
