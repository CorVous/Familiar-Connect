//! Ported from Python `tests/test_sleep_apply.py` — the consolidation apply
//! path: retire → superseded, rewrite → mint with union provenance +
//! backlink, ego-subject display, concurrent-supersede skip-not-raise,
//! fact-axis-only watermark, and the full plan→apply e2e.

#[path = "sleep_helpers/mod.rs"]
mod helpers;

use std::collections::HashSet;
use std::sync::Arc;

use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, AppendTurn, FactSubject, NewFact};
use familiar_connect::identity::ego_canonical_key;
use familiar_connect::sleep::apply::apply_consolidation;
use familiar_connect::sleep::consolidation::{
    ConsolidationPlan, DEFAULT_FACTS_MAX, DEFAULT_RETIRE_CAP, DEFAULT_TURNS_MAX, RetireAction,
    RewriteAction, plan_consolidation,
};
use serde_json::json;

use helpers::{ScriptedLlm, store};

fn aria() -> Vec<FactSubject> {
    vec![FactSubject {
        canonical_key: "discord:A".to_owned(),
        display_at_write: "Aria".to_owned(),
    }]
}

/// Store with three turns + `(junk, d1, d2)` facts; returns their ids.
fn store_with_facts() -> (Arc<AsyncHistoryStore>, i64, i64, i64) {
    let s = store();
    for _ in 0..3 {
        s.sync()
            .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
            .unwrap();
    }
    let junk = s
        .sync()
        .append_fact(AppendFact::new("fam", Some(1), "noise", vec![1]))
        .unwrap();
    let d1 = s
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes berries.", vec![2]).subjects(aria()),
        )
        .unwrap();
    let d2 = s
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria really likes berries.", vec![3]).subjects(aria()),
        )
        .unwrap();
    (s, junk.id, d1.id, d2.id)
}

fn current_texts(s: &AsyncHistoryStore) -> HashSet<String> {
    s.sync()
        .recent_facts("fam", 10, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect()
}

#[tokio::test]
async fn retire_marks_superseded() {
    let (s, junk_id, _, _) = store_with_facts();
    let plan = ConsolidationPlan::new(
        "fam",
        vec![RetireAction {
            fact_ids: vec![junk_id],
            reason: "noise".to_owned(),
        }],
        vec![],
        vec![],
        3,
        3,
    );
    let report = apply_consolidation(&s, &plan, None).await.unwrap();
    assert_eq!(report.retired_fact_ids, vec![junk_id]);
    assert!(!current_texts(&s).contains("noise"));
}

#[tokio::test]
async fn rewrite_merges_with_union_provenance() {
    let (s, _, d1, d2) = store_with_facts();
    let plan = ConsolidationPlan::new(
        "fam",
        vec![],
        vec![RewriteAction {
            old_fact_ids: vec![d1, d2],
            new_text: "Aria likes berries.".to_owned(),
            subject_keys: vec!["discord:A".to_owned()],
            reason: "merge".to_owned(),
        }],
        vec![],
        3,
        3,
    );
    let report = apply_consolidation(&s, &plan, None).await.unwrap();
    assert_eq!(report.rewritten.len(), 1);
    let (old_ids, new_id) = &report.rewritten[0];
    assert_eq!(*old_ids, vec![d1, d2]);
    let current = s.sync().recent_facts("fam", 10, false, None).unwrap();
    let new_fact = current.iter().find(|f| f.id == *new_id).unwrap();
    assert_eq!(new_fact.text, "Aria likes berries.");
    assert_eq!(
        new_fact
            .source_turn_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        HashSet::from([2, 3])
    );
    assert_eq!(new_fact.subjects[0].canonical_key, "discord:A");
    let all_facts = s.sync().recent_facts("fam", 10, true, None).unwrap();
    let old1 = all_facts.iter().find(|f| f.id == d1).unwrap();
    assert_eq!(old1.superseded_by, Some(*new_id));
}

#[tokio::test]
async fn rewrite_to_self_subject() {
    let (s, _, d1, _) = store_with_facts();
    let self_key = ego_canonical_key("fam");
    let plan = ConsolidationPlan::new(
        "fam",
        vec![],
        vec![RewriteAction {
            old_fact_ids: vec![d1],
            new_text: "Sapphire teases Aria about berries.".to_owned(),
            subject_keys: vec![self_key.clone()],
            reason: "bit".to_owned(),
        }],
        vec![],
        3,
        3,
    );
    let report = apply_consolidation(&s, &plan, Some("Sapphire"))
        .await
        .unwrap();
    let (_, new_id) = &report.rewritten[0];
    let current = s.sync().recent_facts("fam", 10, false, None).unwrap();
    let new_fact = current.iter().find(|f| f.id == *new_id).unwrap();
    assert_eq!(new_fact.subjects[0].canonical_key, self_key);
    assert_eq!(new_fact.subjects[0].display_at_write, "Sapphire");
}

#[tokio::test]
async fn skips_concurrently_superseded_fact_without_raising() {
    let (s, junk_id, d1, d2) = store_with_facts();
    // simulate the live bot retiring the junk fact during the plan→apply gap.
    s.sync()
        .supersede("fam", &[junk_id], NewFact::Retire)
        .unwrap();
    let plan = ConsolidationPlan::new(
        "fam",
        vec![RetireAction {
            fact_ids: vec![junk_id],
            reason: "noise".to_owned(),
        }],
        vec![RewriteAction {
            old_fact_ids: vec![d1, d2],
            new_text: "Aria likes berries.".to_owned(),
            subject_keys: vec!["discord:A".to_owned()],
            reason: "merge".to_owned(),
        }],
        vec![],
        3,
        3,
    );
    // must not raise; the still-valid rewrite still applies.
    let report = apply_consolidation(&s, &plan, None).await.unwrap();
    assert!(!report.retired_fact_ids.contains(&junk_id));
    assert!(report.skipped.iter().any(|(_, fid, _)| *fid == junk_id));
    assert_eq!(report.rewritten.len(), 1);
    // watermark still advanced despite the skip.
    assert!(s.sync().get_sleep_watermark("fam").unwrap().is_some());
}

#[tokio::test]
async fn advances_sleep_watermark_fact_axis_only() {
    let (s, junk_id, _, _) = store_with_facts();
    let plan = ConsolidationPlan::new(
        "fam",
        vec![RetireAction {
            fact_ids: vec![junk_id],
            reason: "x".to_owned(),
        }],
        vec![],
        vec![],
        3,
        3,
    );
    apply_consolidation(&s, &plan, None).await.unwrap();
    let wm = s.sync().get_sleep_watermark("fam").unwrap().unwrap();
    // consolidation owns the fact axis only — turn axis (opinion's) untouched.
    assert_eq!((wm.last_fact_id, wm.last_turn_id), (3, 0));
}

#[tokio::test]
async fn full_plan_apply_via_plan_consolidation() {
    let (s, junk_id, d1, d2) = store_with_facts();
    let reply = json!({
        "retire": [{"fact_ids": [junk_id], "reason": "noise"}],
        "rewrite": [{
            "old_fact_ids": [d1, d2],
            "new_text": "Aria likes berries.",
            "subject_keys": ["discord:A"],
            "reason": "merge dups",
        }],
    })
    .to_string();
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
    apply_consolidation(&s, &plan, None).await.unwrap();
    assert_eq!(
        current_texts(&s),
        HashSet::from(["Aria likes berries.".to_owned()])
    );
}

#[tokio::test]
async fn rewrite_missing_source_fact_raises() {
    // A rewrite whose source id was never snapshotted (never existed) has no
    // entry in the up-front facts_by_ids map. Python indexes `by_id[fid]`,
    // raising KeyError; the port mirrors the raise rather than silently
    // degrading (skipping the subject / yielding channel_id=None). Unreachable
    // in the pipeline: supersede is the only removal path, so a fact current at
    // plan time is still fetchable by exact id at apply time.
    let (s, _, _, _) = store_with_facts();
    let missing = 999_999;
    let plan = ConsolidationPlan::new(
        "fam",
        vec![],
        vec![RewriteAction {
            old_fact_ids: vec![missing],
            new_text: "orphan".to_owned(),
            subject_keys: vec!["discord:A".to_owned()],
            reason: "merge".to_owned(),
        }],
        vec![],
        3,
        3,
    );
    let err = apply_consolidation(&s, &plan, None).await.unwrap_err();
    assert!(err.to_string().contains("missing from snapshot"));
}
