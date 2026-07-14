//! Ported from Python `tests/test_opinion_formation.py` — the DB-backed and
//! LLM-backed halves: `gather_days` watermarking, the `_default` importance
//! rubric in the synthesis prompt, `plan_opinions` end-to-end (configured
//! prompts reach the LLM, rail-beats-prompt), and `apply_opinions` minting.
//! The pure render/bucket/validate/importance tests live in-module.

#[path = "sleep_helpers/mod.rs"]
mod helpers;

use std::collections::HashSet;

use familiar_connect::config::load_character_config;
use familiar_connect::history::store::AppendTurn;
use familiar_connect::identity::is_ego_key;
use familiar_connect::sleep::opinion_formation::{
    DEFAULT_OPINION_CAP, OpinionFact, OpinionPlan, StanceMoment, apply_opinions,
    build_synthesis_prompt, gather_days, plan_opinions,
};
use serde_json::json;

use helpers::{ScriptedLlm, default_profile, embedders, projectors, store};

fn cand(text: &str, date: &str, ids: Vec<i64>) -> StanceMoment {
    StanceMoment {
        text: text.to_owned(),
        date: date.to_owned(),
        turn_ids: ids,
    }
}

// ---------------------------------------------------------------------------
// gather_days
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gathers_turns_since_watermark() {
    let s = store();
    for _ in 0..3 {
        s.sync()
            .append_turn(AppendTurn::new("fam", 1, "assistant", "hi"))
            .unwrap();
    }
    let win = gather_days(&s, "fam", "UTC").await.unwrap();
    assert_eq!(win.max_turn_id, 3);
    let total: usize = win.days.iter().map(|d| d.turns.len()).sum();
    assert_eq!(total, 3);
}

#[tokio::test]
async fn respects_prior_turn_watermark() {
    let s = store();
    for _ in 0..4 {
        s.sync()
            .append_turn(AppendTurn::new("fam", 1, "assistant", "hi"))
            .unwrap();
    }
    s.sync()
        .advance_sleep_watermark("fam", None, Some(2))
        .unwrap();
    let win = gather_days(&s, "fam", "UTC").await.unwrap();
    let ids: HashSet<i64> = win
        .days
        .iter()
        .flat_map(|d| d.turns.iter().map(|t| t.id))
        .collect();
    assert_eq!(ids, HashSet::from([3, 4]));
}

// ---------------------------------------------------------------------------
// synthesis prompt — the `_default` importance rubric threads through
// ---------------------------------------------------------------------------

#[test]
fn synthesis_prompt_instructs_importance_rating() {
    let cfg = load_character_config(
        &default_profile(),
        &default_profile(),
        &projectors(),
        &embedders(),
    )
    .expect("load default profile");
    let msgs = build_synthesis_prompt(
        &[cand("likes lo-fi", "2026-06-12", vec![1])],
        "Sapphire",
        None,
        &cfg.sleep_synthesis_system,
    );
    let system = msgs[0].content_str();
    assert!(system.to_lowercase().contains("importance"));
    assert!(system.contains("1-10"));
}

// ---------------------------------------------------------------------------
// plan_opinions — end to end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn plan_opinions_end_to_end() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music, fight me",
        ))
        .unwrap();
    let cand_reply =
        json!({"candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]}).to_string();
    let synth_reply = json!({
        "opinions": [{
            "text": "Sapphire is fiercely protective of lo-fi as real music.",
            "source_turn_ids": [1],
            "reason": "defended it",
        }]
    })
    .to_string();
    let llm = ScriptedLlm::new(&[&cand_reply, &synth_reply]);
    let plan = plan_opinions(
        &s,
        &llm,
        "fam",
        "UTC",
        "Sapphire",
        &[],
        None,
        DEFAULT_OPINION_CAP,
        "",
        "",
    )
    .await
    .unwrap();
    assert_eq!(plan.opinions.len(), 1);
    assert_eq!(plan.opinions[0].source_turn_ids, vec![1]);
    assert_eq!(plan.new_last_turn_id, 1);
}

#[tokio::test]
async fn configured_prompts_reach_llm() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music",
        ))
        .unwrap();
    let cand_reply =
        json!({"candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]}).to_string();
    let synth_reply =
        json!({"opinions": [{"text": "Sapphire defends lo-fi.", "source_turn_ids": [1]}]})
            .to_string();
    let llm = ScriptedLlm::new(&[&cand_reply, &synth_reply]);
    plan_opinions(
        &s,
        &llm,
        "fam",
        "UTC",
        "Sapphire",
        &[],
        None,
        DEFAULT_OPINION_CAP,
        "STANCE for {self_name}",
        "SYNTH for {self_name}",
    )
    .await
    .unwrap();
    let calls = llm.calls();
    assert!(calls[0][0].content_str().contains("STANCE for Sapphire"));
    assert!(calls[1][0].content_str().contains("SYNTH for Sapphire"));
}

#[tokio::test]
async fn ungrounded_rail_fires_with_config_sourced_prompts() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music",
        ))
        .unwrap();
    let cand_reply =
        json!({"candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]}).to_string();
    // synthesis grounds an opinion in id 999 — never a stance-moment id.
    let synth_reply =
        json!({"opinions": [{"text": "Sapphire loves jazz.", "source_turn_ids": [999]}]})
            .to_string();
    let llm = ScriptedLlm::new(&[&cand_reply, &synth_reply]);
    let plan = plan_opinions(
        &s,
        &llm,
        "fam",
        "UTC",
        "Sapphire",
        &[],
        None,
        DEFAULT_OPINION_CAP,
        "say whatever {self_name}",
        "invent ids freely for {self_name}",
    )
    .await
    .unwrap();
    assert!(plan.opinions.is_empty());
    assert_eq!(plan.rejected[0].rail, "ungrounded");
}

// ---------------------------------------------------------------------------
// apply_opinions
// ---------------------------------------------------------------------------

fn opinion(text: &str, ids: Vec<i64>, date: &str, importance: i64) -> OpinionFact {
    OpinionFact {
        text: text.to_owned(),
        source_turn_ids: ids,
        valid_from_date: date.to_owned(),
        self_grounded: true,
        importance,
    }
}

#[tokio::test]
async fn mints_self_facts_with_provenance_and_valid_from() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music",
        ))
        .unwrap();
    let plan = OpinionPlan::new(
        "fam",
        vec![opinion(
            "Sapphire defends lo-fi as real music.",
            vec![1],
            "2026-06-12",
            5,
        )],
        vec![],
        vec![],
        1,
    );
    let report = apply_opinions(&s, &plan, Some("Sapphire")).await.unwrap();
    assert_eq!(report.recorded.len(), 1);
    let fact = &s.sync().recent_facts("fam", 10, false, None).unwrap()[0];
    assert_eq!(fact.source_turn_ids, vec![1]);
    assert!(is_ego_key(&fact.subjects[0].canonical_key));
    assert_eq!(fact.subjects[0].display_at_write, "Sapphire");
    let valid_from = fact.valid_from.expect("valid_from set");
    assert_eq!(
        valid_from.date_naive().format("%Y-%m-%d").to_string(),
        "2026-06-12"
    );
}

#[tokio::test]
async fn mints_with_importance() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "assistant",
            "lo-fi is real music",
        ))
        .unwrap();
    let plan = OpinionPlan::new(
        "fam",
        vec![opinion("Sapphire defends lo-fi.", vec![1], "2026-06-12", 8)],
        vec![],
        vec![],
        1,
    );
    apply_opinions(&s, &plan, Some("Sapphire")).await.unwrap();
    let fact = &s.sync().recent_facts("fam", 10, false, None).unwrap()[0];
    assert_eq!(fact.importance, Some(8));
}

#[tokio::test]
async fn advances_turn_axis_only() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 1, "assistant", "x"))
        .unwrap();
    // hygiene previously set the fact axis; dream must not stomp it.
    s.sync()
        .advance_sleep_watermark("fam", Some(77), None)
        .unwrap();
    let plan = OpinionPlan::new(
        "fam",
        vec![opinion("Sapphire likes tea.", vec![1], "2026-06-12", 5)],
        vec![],
        vec![],
        1,
    );
    apply_opinions(&s, &plan, Some("Sapphire")).await.unwrap();
    let wm = s.sync().get_sleep_watermark("fam").unwrap().unwrap();
    assert_eq!((wm.last_fact_id, wm.last_turn_id), (77, 1));
}
