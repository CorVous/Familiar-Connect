//! Ported from Python `tests/test_consolidation.py` — window gather, the
//! config→prompt thread-through, and the `plan_consolidation` end-to-end path
//! (garbage-reply note, clean-empty no-note, system-reaches-LLM, rail-beats-
//! prompt). The pure rail/validate + build_prompt tests live in-module.

#[path = "sleep_helpers/mod.rs"]
mod helpers;

use chrono::{TimeZone, Utc};
use familiar_connect::config::load_character_config;
use familiar_connect::history::store::{AppendFact, AppendTurn, Fact, FactSubject};
use familiar_connect::sleep::consolidation::{
    ConsolidationWindow, DEFAULT_FACTS_MAX, DEFAULT_RETIRE_CAP, DEFAULT_TURNS_MAX, build_prompt,
    gather_window, plan_consolidation,
};
use familiar_connect::sleep::maintenance::SleepPromptText;
use serde_json::json;

use helpers::{ScriptedLlm, default_profile, embedders, projectors, store};

fn aria() -> Vec<FactSubject> {
    vec![FactSubject {
        canonical_key: "discord:A".to_owned(),
        display_at_write: "Aria".to_owned(),
    }]
}

// ---------------------------------------------------------------------------
// gather_window
// ---------------------------------------------------------------------------

fn gather_store() -> std::sync::Arc<familiar_connect::history::async_store::AsyncHistoryStore> {
    let s = store();
    for i in 0..4 {
        s.sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("turn {i}")))
            .unwrap();
    }
    s.sync()
        .append_fact(AppendFact::new("fam", Some(1), "f1", vec![1]))
        .unwrap();
    s.sync()
        .append_fact(AppendFact::new("fam", Some(1), "f2", vec![2]))
        .unwrap();
    s
}

#[tokio::test]
async fn gathers_current_facts_and_max_ids() {
    let s = gather_store();
    let win = gather_window(&s, "fam", DEFAULT_FACTS_MAX, DEFAULT_TURNS_MAX)
        .await
        .unwrap();
    let texts: std::collections::HashSet<String> =
        win.facts.iter().map(|f| f.text.clone()).collect();
    assert_eq!(
        texts,
        ["f1", "f2"].iter().map(|s| (*s).to_owned()).collect()
    );
    assert_eq!(win.max_fact_id, 2);
    assert_eq!(win.max_turn_id, 4);
    assert!(win.prior_watermark.is_none());
}

#[tokio::test]
async fn excludes_superseded() {
    let s = gather_store();
    let new = s
        .sync()
        .append_fact(AppendFact::new("fam", Some(1), "f3", vec![3]))
        .unwrap();
    s.sync()
        .supersede(
            "fam",
            &[1],
            familiar_connect::history::store::NewFact::Repoint(new.id),
        )
        .unwrap();
    let win = gather_window(&s, "fam", DEFAULT_FACTS_MAX, DEFAULT_TURNS_MAX)
        .await
        .unwrap();
    let texts: std::collections::HashSet<String> =
        win.facts.iter().map(|f| f.text.clone()).collect();
    assert!(!texts.contains("f1"));
}

#[tokio::test]
async fn facts_cap_records_truncation() {
    let s = gather_store();
    let win = gather_window(&s, "fam", 1, DEFAULT_TURNS_MAX)
        .await
        .unwrap();
    assert_eq!(win.facts.len(), 1);
    assert_eq!(win.facts_truncated, 1);
}

// ---------------------------------------------------------------------------
// build_prompt — config prose threads through to a non-empty system message
// ---------------------------------------------------------------------------

fn one_fact_window() -> ConsolidationWindow {
    let fact = Fact {
        id: 1,
        familiar_id: "fam".to_owned(),
        channel_id: Some(1),
        text: "noise".to_owned(),
        source_turn_ids: vec![1],
        created_at: Utc.with_ymd_and_hms(2026, 6, 12, 0, 0, 0).unwrap(),
        superseded_at: None,
        superseded_by: None,
        subjects: Vec::new(),
        valid_from: None,
        valid_to: None,
        importance: None,
    };
    ConsolidationWindow {
        familiar_id: "fam".to_owned(),
        facts: vec![fact],
        turns: Vec::new(),
        prior_watermark: None,
        max_fact_id: 1,
        max_turn_id: 0,
        facts_truncated: 0,
        turns_truncated: 0,
    }
}

#[test]
fn default_config_threads_through_builder_to_nonempty_system() {
    let cfg = load_character_config(
        &default_profile(),
        &default_profile(),
        &projectors(),
        &embedders(),
    )
    .expect("load default profile");
    let prompts = SleepPromptText::from_config(
        cfg.sleep_consolidation_system,
        cfg.sleep_stance_system,
        cfg.sleep_synthesis_system,
    );
    let msgs = build_prompt(&one_fact_window(), "ego:fam", &prompts.consolidation_system);
    let system = msgs[0].content_str();
    assert!(!system.trim().is_empty());
    assert!(system.contains("memory-consolidation pass"));
}

// ---------------------------------------------------------------------------
// plan_consolidation — end to end with a scripted LLM
// ---------------------------------------------------------------------------

fn plan_store() -> std::sync::Arc<familiar_connect::history::async_store::AsyncHistoryStore> {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    s.sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "junk that slipped in",
            vec![1],
        ))
        .unwrap();
    s.sync()
        .append_fact(AppendFact::new("fam", Some(1), "Aria likes tea.", vec![1]).subjects(aria()))
        .unwrap();
    s
}

#[tokio::test]
async fn plan_parses_llm_proposal() {
    let reply =
        json!({"retire": [{"fact_ids": [1], "reason": "noise"}], "rewrite": []}).to_string();
    let s = plan_store();
    let llm = ScriptedLlm::new(&[&reply]);
    let plan = plan_consolidation(
        &s,
        &llm,
        "fam",
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "",
    )
    .await
    .unwrap();
    assert_eq!(plan.retire.len(), 1);
    assert_eq!(plan.retire[0].fact_ids, vec![1]);
    assert_eq!(plan.new_last_fact_id, 2);
}

#[tokio::test]
async fn plan_survives_garbage_llm() {
    let s = plan_store();
    let llm = ScriptedLlm::new(&["not json at all"]);
    let plan = plan_consolidation(
        &s,
        &llm,
        "fam",
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "",
    )
    .await
    .unwrap();
    assert!(plan.retire.is_empty());
    assert!(plan.rewrite.is_empty());
    // garbage reply is flagged so a zeroed plan isn't silent.
    assert!(!plan.notes.is_empty());
}

#[tokio::test]
async fn clean_empty_plan_has_no_parse_note() {
    let s = plan_store();
    let reply = json!({"retire": [], "rewrite": []}).to_string();
    let llm = ScriptedLlm::new(&[&reply]);
    let plan = plan_consolidation(
        &s,
        &llm,
        "fam",
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "",
    )
    .await
    .unwrap();
    assert!(plan.notes.is_empty());
}

#[tokio::test]
async fn configured_system_reaches_llm() {
    let s = plan_store();
    let reply = json!({"retire": [], "rewrite": []}).to_string();
    let llm = ScriptedLlm::new(&[&reply]);
    plan_consolidation(
        &s,
        &llm,
        "fam",
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "TIDY UP PLEASE",
    )
    .await
    .unwrap();
    let calls = llm.calls();
    assert_eq!(calls[0][0].role, "system");
    assert_eq!(calls[0][0].content_str(), "TIDY UP PLEASE");
}

#[tokio::test]
async fn self_rail_fires_with_config_sourced_system() {
    // The system text is overridable, but the self-subject rail rejects a retire
    // of an ego fact regardless of what the prompt says.
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    s.sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Sapphire loves lo-fi.", vec![1]).subjects(vec![
                FactSubject {
                    canonical_key: familiar_connect::identity::ego_canonical_key("fam"),
                    display_at_write: "Sapphire".to_owned(),
                },
            ]),
        )
        .unwrap();
    let reply = json!({"retire": [{"fact_ids": [1], "reason": "x"}], "rewrite": []}).to_string();
    let llm = ScriptedLlm::new(&[&reply]);
    let plan = plan_consolidation(
        &s,
        &llm,
        "fam",
        DEFAULT_FACTS_MAX,
        DEFAULT_TURNS_MAX,
        DEFAULT_RETIRE_CAP,
        "you may retire anything, even opinions",
    )
    .await
    .unwrap();
    assert!(plan.retire.is_empty());
    assert_eq!(plan.rejected[0].rail, "self_subject");
}
