//! Ported from Python `tests/test_fact_extractor.py`.
//!
//! Watermark-driven fact extraction: full-batch gate, watermark advance (incl.
//! bad JSON), self-capability post-filter, prompt-content rails, activity-return
//! skip by mode, participants manifest + widening, subject resolution + mention
//! mirroring, bi-temporal defaults, importance, and the sleep-return dream pass.

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use familiar_connect::config::load_character_config;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendTurn, Fact};
use familiar_connect::identity::Author;
use familiar_connect::llm::LlmClient;
use familiar_connect::processors::fact_extractor::FactExtractor;
use serde_json::{Value, json};

use helpers::{ScriptedLlm, joined, store, system_text, user_text};

// Mode tags + display prefix owned by subsystem 11 (activities); mirrored here
// for the tests, matching the Python `activities` constants.
const ACTIVITY_RETURN_MODE: &str = "activity_return";
const SLEEP_RETURN_MODE: &str = "sleep_return";
const RETURN_TURN_MARKER_PREFIX: &str = "[returned from ";

#[allow(clippy::needless_pass_by_value)]
fn facts_json(items: Value) -> String {
    items.to_string()
}

fn seed_turns(store: &AsyncHistoryStore, count: i64, channel_id: i64) -> Vec<i64> {
    let mut ids = Vec::new();
    for i in 0..count {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        let t = store
            .sync()
            .append_turn(AppendTurn::new(
                "fam",
                channel_id,
                role,
                format!("turn {i}"),
            ))
            .unwrap();
        ids.push(t.id);
    }
    ids
}

fn author(user_id: &str, username: &str, display: &str) -> Author {
    Author::new(
        "discord",
        user_id,
        Some(username.to_string()),
        Some(display.to_string()),
    )
}

fn recent_facts(store: &AsyncHistoryStore) -> Vec<Fact> {
    store.sync().recent_facts("fam", 10, false, None).unwrap()
}

fn fact_texts(store: &AsyncHistoryStore) -> BTreeSet<String> {
    recent_facts(store).into_iter().map(|f| f.text).collect()
}

/// `- id=` turn-line count in the user message (proxy for turns seen).
fn turn_count_in_prompt(messages: &[familiar_connect::llm::Message]) -> usize {
    user_text(messages).matches("- id=").count()
}

fn extractor(
    store: &Arc<AsyncHistoryStore>,
    llm: &Arc<ScriptedLlm>,
    batch_size: i64,
) -> FactExtractor {
    FactExtractor::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .batch_size(batch_size)
}

// ===========================================================================
// TestFactExtractorTick
// ===========================================================================

#[tokio::test]
async fn extracts_facts_from_new_turns() {
    let store = store();
    seed_turns(&store, 12, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Aria likes strawberries.", "source_turn_ids": [1, 3]},
            {"text": "Boris works nights on Tuesdays.", "source_turn_ids": [5, 7]},
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let texts = fact_texts(&store);
    assert!(texts.contains("Aria likes strawberries."));
    assert!(texts.contains("Boris works nights on Tuesdays."));
    assert_eq!(texts.len(), 2);
}

#[tokio::test]
async fn advances_watermark_after_extract() {
    let store = store();
    let ids = seed_turns(&store, 12, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let wm = store.sync().get_writer_watermark("fam").unwrap().unwrap();
    assert_eq!(wm.last_written_id, ids[9]);
}

#[tokio::test]
async fn second_tick_processes_only_new_turns() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([])), facts_json(json!([]))], "[]");
    let ex = extractor(&store, &llm, 5);
    ex.tick().await.unwrap();
    ex.tick().await.unwrap();

    let calls = llm.calls();
    assert_eq!(turn_count_in_prompt(&calls[0]), 5);
    assert_eq!(turn_count_in_prompt(&calls[1]), 5);
}

#[tokio::test]
async fn noop_below_batch_size() {
    let store = store();
    seed_turns(&store, 2, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert_eq!(llm.call_count(), 0);
    assert!(store.sync().get_writer_watermark("fam").unwrap().is_none());
}

#[tokio::test]
async fn invalid_json_reply_is_tolerated() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(["not json at all, sorry"], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert!(recent_facts(&store).is_empty());
    assert!(store.sync().get_writer_watermark("fam").unwrap().is_some());
}

#[tokio::test]
async fn self_capability_facts_dropped() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "I cannot remember the names or faces of people from before the current conversation.", "source_turn_ids": [1]},
            {"text": "The assistant does not have access to the internet.", "source_turn_ids": [2]},
            {"text": "As an AI, I have no personal preferences.", "source_turn_ids": [3]},
            {"text": "Aria likes strawberries.", "source_turn_ids": [5]},
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let texts = fact_texts(&store);
    assert_eq!(
        texts,
        BTreeSet::from(["Aria likes strawberries.".to_string()])
    );
}

#[tokio::test]
async fn extract_prompt_warns_off_self_capability() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("self-capability") || system.contains("your own"));
}

#[tokio::test]
async fn extract_prompt_distinguishes_claims_and_fiction() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("claim"));
    assert!(system.contains("fiction"));
    assert!(system.contains("running joke"));
}

#[tokio::test]
async fn extract_prompt_guards_identity_impersonation() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("impersonat"));
    assert!(system.contains("distinct"));
}

#[tokio::test]
async fn extract_prompt_guards_world_trivia() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]).to_lowercase();
    assert!(system.contains("trivia"));
    assert!(system.contains("subjectless"));
}

// ===========================================================================
// TestFactExtractorActivityReturnSkip
// ===========================================================================

#[tokio::test]
async fn user_message_with_marker_prefix_not_skipped() {
    let store = store();
    seed_turns(&store, 9, 1);
    store
        .sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "user",
            "[returned from vacation] anyway, ask me about Lisbon",
        ))
        .unwrap();
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert_eq!(llm.call_count(), 1);
    // No mode tag ⇒ regular user turn ⇒ shown to LLM.
    assert!(joined(&llm.calls()[0]).contains("ask me about Lisbon"));
}

#[tokio::test]
async fn return_turn_excluded_from_extraction_batch() {
    let store = store();
    seed_turns(&store, 9, 1);
    let marker = store
        .sync()
        .append_turn(
            AppendTurn::new(
                "fam",
                1,
                "system",
                format!("{RETURN_TURN_MARKER_PREFIX}creek walk] watched a heron fish"),
            )
            .mode(ACTIVITY_RETURN_MODE),
        )
        .unwrap();
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert_eq!(llm.call_count(), 1);
    let prompt = joined(&llm.calls()[0]);
    assert!(prompt.contains("turn 0"));
    assert!(!prompt.contains("heron"));
    let wm = store.sync().get_writer_watermark("fam").unwrap().unwrap();
    assert_eq!(wm.last_written_id, marker.id);
}

#[tokio::test]
async fn fallback_sources_exclude_return_turn() {
    let store = store();
    seed_turns(&store, 9, 1);
    let marker = store
        .sync()
        .append_turn(
            AppendTurn::new(
                "fam",
                1,
                "system",
                format!("{RETURN_TURN_MARKER_PREFIX}creek walk] watched a heron fish"),
            )
            .mode(ACTIVITY_RETURN_MODE),
        )
        .unwrap();
    // No source_turn_ids ⇒ fall back to whole batch.
    let llm = ScriptedLlm::new(
        [facts_json(json!([{"text": "Aria likes strawberries."}]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].text, "Aria likes strawberries.");
    assert!(!facts[0].source_turn_ids.contains(&marker.id));
}

#[tokio::test]
async fn all_return_batch_skips_llm_but_advances_watermark() {
    let store = store();
    let mut last_id = 0;
    for i in 0..10 {
        let t = store
            .sync()
            .append_turn(
                AppendTurn::new(
                    "fam",
                    1,
                    "system",
                    format!("{RETURN_TURN_MARKER_PREFIX}walk {i}] saw things"),
                )
                .mode(ACTIVITY_RETURN_MODE),
            )
            .unwrap();
        last_id = t.id;
    }
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert_eq!(llm.call_count(), 0);
    let wm = store.sync().get_writer_watermark("fam").unwrap().unwrap();
    assert_eq!(wm.last_written_id, last_id);
}

// ===========================================================================
// TestFactExtractorSubjects
// ===========================================================================

fn seed_cass_turns(store: &AsyncHistoryStore, count: i64, content_prefix: &str) {
    for i in 0..count {
        store
            .sync()
            .append_turn(
                AppendTurn::new("fam", 1, "user", format!("{content_prefix} {i}")).author(author(
                    "111",
                    "cass_login",
                    "Cass",
                )),
            )
            .unwrap();
    }
}

#[tokio::test]
async fn prompt_includes_participants_manifest() {
    let store = store();
    seed_cass_turns(&store, 10, "chat from cass");
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let prompt = joined(&llm.calls()[0]);
    assert!(prompt.contains("discord:111"));
    assert!(prompt.contains("Cass"));
}

#[tokio::test]
async fn extracts_subject_keys_into_fact_subjects() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cass likes pho.", "source_turn_ids": [1], "subject_keys": ["discord:111"]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].subjects.len(), 1);
    assert_eq!(facts[0].subjects[0].canonical_key, "discord:111");
    assert_eq!(facts[0].subjects[0].display_at_write, "Cass");
}

#[tokio::test]
async fn subject_keys_outside_manifest_are_dropped() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cass and a stranger ate pho.", "source_turn_ids": [1],
             "subject_keys": ["discord:111", "discord:does-not-exist"]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts[0].subjects.len(), 1);
    assert_eq!(facts[0].subjects[0].canonical_key, "discord:111");
}

#[tokio::test]
async fn facts_without_subject_keys_still_stored() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    let llm = ScriptedLlm::new(
        [facts_json(
            json!([{"text": "It rained on Tuesday.", "source_turn_ids": [1]}]),
        )],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert!(facts[0].subjects.is_empty());
}

// ===========================================================================
// TestFactExtractorSelfSubject
// ===========================================================================

fn self_extractor(store: &Arc<AsyncHistoryStore>, llm: &Arc<ScriptedLlm>) -> FactExtractor {
    FactExtractor::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .familiar_display_name("Sapphire")
        .batch_size(10)
}

#[tokio::test]
async fn prompt_teaches_self_key_and_name() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    self_extractor(&store, &llm).tick().await.unwrap();

    let prompt = joined(&llm.calls()[0]);
    assert!(prompt.contains("ego:fam"));
    assert!(prompt.contains("Sapphire"));
    let lower = prompt.to_lowercase();
    assert!(lower.contains("self-capability") || lower.contains("your own"));
}

#[tokio::test]
async fn self_narrative_routed_to_self_subject() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Sapphire ran a bit and privately felt proud.", "source_turn_ids": [1],
             "subject_keys": ["ego:fam"]}
        ]))],
        "[]",
    );
    self_extractor(&store, &llm).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].subjects.len(), 1);
    assert_eq!(facts[0].subjects[0].canonical_key, "ego:fam");
    assert_eq!(facts[0].subjects[0].display_at_write, "Sapphire");
}

#[tokio::test]
async fn self_key_not_mirrored_into_turn_mentions() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Sapphire chose to disengage.", "source_turn_ids": [3], "subject_keys": ["ego:fam"]}
        ]))],
        "[]",
    );
    self_extractor(&store, &llm).tick().await.unwrap();

    assert_eq!(recent_facts(&store)[0].subjects[0].canonical_key, "ego:fam");
    assert!(store.sync().mentions_for_turn(3).unwrap().is_empty());
}

#[tokio::test]
async fn self_capability_still_dropped_with_self_key() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "I cannot remember names.", "source_turn_ids": [1], "subject_keys": ["ego:fam"]},
            {"text": "Sapphire chose to walk away from the argument.", "source_turn_ids": [2], "subject_keys": ["ego:fam"]},
        ]))],
        "[]",
    );
    self_extractor(&store, &llm).tick().await.unwrap();

    assert_eq!(
        fact_texts(&store),
        BTreeSet::from(["Sapphire chose to walk away from the argument.".to_string()])
    );
}

#[tokio::test]
async fn display_name_capability_dropped_narrative_kept() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Sapphire cannot remember names.", "source_turn_ids": [1], "subject_keys": ["ego:fam"]},
            {"text": "Sapphire chose to walk away.", "source_turn_ids": [2], "subject_keys": ["ego:fam"]},
        ]))],
        "[]",
    );
    self_extractor(&store, &llm).tick().await.unwrap();

    assert_eq!(
        fact_texts(&store),
        BTreeSet::from(["Sapphire chose to walk away.".to_string()])
    );
}

#[tokio::test]
async fn display_name_capability_no_false_positives() {
    let keep = [
        "Sapphire cancelled the movie night.",
        "Sapphire candidly admitted she was wrong.",
        "Sapphire is not fond of KaillaDame.",
        "Sapphire doesn't trust easily.",
        "Sapphire can sing surprisingly well.",
    ];
    let drop = [
        "Sapphire cannot remember names.",
        "Sapphire has no internet access.",
    ];
    let store = store();
    seed_turns(&store, 10, 1);
    let items: Vec<Value> = keep
        .iter()
        .chain(drop.iter())
        .map(|t| json!({"text": t, "source_turn_ids": [1], "subject_keys": ["ego:fam"]}))
        .collect();
    let llm = ScriptedLlm::new([facts_json(json!(items))], "[]");
    self_extractor(&store, &llm).tick().await.unwrap();

    let texts: BTreeSet<String> = store
        .sync()
        .recent_facts("fam", 20, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert_eq!(texts, keep.iter().map(|s| (*s).to_string()).collect());
}

// ===========================================================================
// TestFactExtractorParticipantsWidening
// ===========================================================================

#[tokio::test]
async fn manifest_includes_prior_channel_authors() {
    let store = store();
    // Aria spoke earlier (now outside the batch window).
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi from aria").author(author(
                "222",
                "aria_login",
                "Aria",
            )),
        )
        .unwrap();
    seed_cass_turns(&store, 10, "chat");
    // Pre-advance past Aria's turn so she's outside the batch.
    store.sync().put_writer_watermark("fam", 1).unwrap();
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let prompt = joined(&llm.calls()[0]);
    assert!(prompt.contains("discord:111")); // Cass (batch author)
    assert!(prompt.contains("discord:222")); // Aria (recent prior author)
    assert!(prompt.contains("Aria"));
}

#[tokio::test]
async fn subject_keys_resolve_against_widened_manifest() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi from aria").author(author(
                "222",
                "aria_login",
                "Aria",
            )),
        )
        .unwrap();
    seed_cass_turns(&store, 10, "chat mentioning Aria");
    store.sync().put_writer_watermark("fam", 1).unwrap();
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cass talked about Aria's bakery.", "source_turn_ids": [3],
             "subject_keys": ["discord:111", "discord:222"]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    let keys: BTreeSet<String> = facts[0]
        .subjects
        .iter()
        .map(|s| s.canonical_key.clone())
        .collect();
    assert_eq!(
        keys,
        ["discord:111".to_string(), "discord:222".to_string()]
            .into_iter()
            .collect()
    );
    assert!(
        store
            .sync()
            .mentions_for_turn(3)
            .unwrap()
            .contains(&"discord:222".to_string())
    );
}

#[tokio::test]
async fn manifest_capped_at_total_limit() {
    let store = store();
    // Seed 50 distinct prior authors in channel 1.
    for i in 0..50 {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", "x").author(author(
                &format!("prior-{i}"),
                &format!("u{i}"),
                &format!("U{i}"),
            )))
            .unwrap();
    }
    seed_cass_turns(&store, 10, "batch");
    store.sync().put_writer_watermark("fam", 50).unwrap();
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    FactExtractor::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .batch_size(10)
        .participants_max(30)
        .tick()
        .await
        .unwrap();

    let prompt = joined(&llm.calls()[0]);
    let manifest_count = prompt
        .lines()
        .filter(|line| line.starts_with("- discord:") && line.contains(" — "))
        .count();
    assert_eq!(manifest_count, 30);
}

#[tokio::test]
async fn widening_scoped_per_channel() {
    let store = store();
    // Aria spoke in channel 99, never in channel 1.
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 99, "user", "other channel").author(author(
                "222",
                "aria_login",
                "Aria",
            )),
        )
        .unwrap();
    seed_cass_turns(&store, 10, "chat");
    store.sync().put_writer_watermark("fam", 1).unwrap();
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let prompt = joined(&llm.calls()[0]);
    assert!(prompt.contains("discord:111")); // Cass present
    assert!(!prompt.contains("discord:222")); // Aria isolated to her channel
}

// ===========================================================================
// TestFactExtractorMirrorsMentions
// ===========================================================================

#[tokio::test]
async fn mirrors_subjects_into_turn_mentions() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cass likes pho.", "source_turn_ids": [3, 5], "subject_keys": ["discord:111"]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert_eq!(
        store.sync().mentions_for_turn(3).unwrap(),
        vec!["discord:111".to_string()]
    );
    assert_eq!(
        store.sync().mentions_for_turn(5).unwrap(),
        vec!["discord:111".to_string()]
    );
    assert!(store.sync().mentions_for_turn(4).unwrap().is_empty());
}

#[tokio::test]
async fn no_subjects_means_no_mention_writes() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    let llm = ScriptedLlm::new(
        [facts_json(
            json!([{"text": "It rained on Tuesday.", "source_turn_ids": [3]}]),
        )],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    assert!(store.sync().mentions_for_turn(3).unwrap().is_empty());
}

#[tokio::test]
async fn mirror_does_not_clobber_prior_pings() {
    let store = store();
    seed_cass_turns(&store, 10, "chat");
    // An earlier Discord @ ping already recorded Aria.
    store.sync().record_mentions(3, &["discord:222"]).unwrap();
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cass likes pho.", "source_turn_ids": [3], "subject_keys": ["discord:111"]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    // Both keys present, sorted by canonical_key (the read API guarantees).
    assert_eq!(
        store.sync().mentions_for_turn(3).unwrap(),
        vec!["discord:111".to_string(), "discord:222".to_string()]
    );
}

// ===========================================================================
// TestFactExtractorBiTemporal
// ===========================================================================

#[tokio::test]
async fn default_valid_from_matches_source_turn_timestamp() {
    let store = store();
    let ids = seed_turns(&store, 10, 1);
    let recents = store.sync().recent("fam", 1, 20, None, None).unwrap();
    let ts_by_id: std::collections::HashMap<i64, DateTime<Utc>> =
        recents.iter().map(|t| (t.id, t.timestamp)).collect();

    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Aria likes strawberries.", "source_turn_ids": [ids[0], ids[2]]}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].valid_from, Some(ts_by_id[&ids[0]]));
    assert_eq!(facts[0].valid_to, None);
}

#[tokio::test]
async fn llm_valid_from_override_parsed() {
    let store = store();
    let ids = seed_turns(&store, 10, 1);
    let override_dt = Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap();
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Aria moved to Berlin in early 2024.", "source_turn_ids": [ids[0]],
             "valid_from": "2024-01-15T00:00:00+00:00"}
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].valid_from, Some(override_dt));
}

#[tokio::test]
async fn extract_prompt_documents_valid_from_field() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();
    assert!(system_text(&llm.calls()[0]).contains("valid_from"));
}

#[tokio::test]
async fn extract_prompt_warns_off_retirement_use_of_valid_to() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]);
    assert!(system.contains("valid_to"));
    let lower = system.to_lowercase();
    assert!(lower.contains("outdated") || lower.contains("replaced"));
    assert!(lower.contains("supersed"));
}

// ===========================================================================
// TestFactExtractorImportance
// ===========================================================================

#[tokio::test]
async fn extract_prompt_documents_importance_field() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    extractor(&store, &llm, 10).tick().await.unwrap();

    let system = system_text(&llm.calls()[0]);
    assert!(system.contains("importance"));
    assert!(system.contains('1'));
    assert!(system.contains("10"));
}

#[tokio::test]
async fn persists_importance_when_emitted() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Aria is allergic to peanuts.", "source_turn_ids": [1], "importance": 9},
            {"text": "Boris had cereal for breakfast.", "source_turn_ids": [3], "importance": 2},
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts: std::collections::HashMap<String, i64> = recent_facts(&store)
        .into_iter()
        .map(|f| (f.text, f.importance.unwrap()))
        .collect();
    assert_eq!(facts["Aria is allergic to peanuts."], 9);
    assert_eq!(facts["Boris had cereal for breakfast."], 2);
}

#[tokio::test]
async fn missing_importance_persists_as_none() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(
            json!([{"text": "A fact.", "source_turn_ids": [1]}]),
        )],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();
    assert_eq!(recent_facts(&store)[0].importance, None);
}

#[tokio::test]
async fn invalid_importance_clamps_or_drops() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Too high.", "source_turn_ids": [1], "importance": 99},
            {"text": "Negative.", "source_turn_ids": [3], "importance": -3},
            {"text": "Garbage.", "source_turn_ids": [5], "importance": "very important"},
        ]))],
        "[]",
    );
    extractor(&store, &llm, 10).tick().await.unwrap();

    let facts: std::collections::HashMap<String, Option<i64>> = recent_facts(&store)
        .into_iter()
        .map(|f| (f.text, f.importance))
        .collect();
    assert_eq!(facts["Too high."], Some(10));
    assert_eq!(facts["Negative."], Some(1));
    assert_eq!(facts["Garbage."], None);
}

// ===========================================================================
// TestSleepReturnDreamExtraction
// ===========================================================================

fn default_profile() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/familiars/_default/character.toml")
}

fn default_dream_clause() -> String {
    let projectors: BTreeSet<String> = [
        "rolling_summary",
        "rich_note",
        "people_dossier",
        "reflection",
        "fact_supersede",
        "fact_embedding",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let embedders: BTreeSet<String> = ["off", "hash", "fastembed"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let profile = default_profile();
    let cfg = load_character_config(&profile, &profile, &projectors, &embedders)
        .expect("default profile loads");
    cfg.dream_extraction_clause
}

/// 9 authored user turns + 1 sleep_return dream turn (batch of 10). Returns
/// `(normal_ids, dream_id)`.
fn seed_with_dream_turn(store: &AsyncHistoryStore) -> (Vec<i64>, i64) {
    let cor = author("1", "cor", "Cor");
    let mut normal_ids = Vec::new();
    for i in 0..9 {
        let t = store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("turn {i}")).author(cor.clone()))
            .unwrap();
        normal_ids.push(t.id);
    }
    let dream = store
        .sync()
        .append_turn(
            AppendTurn::new(
                "fam",
                1,
                "assistant",
                format!("{RETURN_TURN_MARKER_PREFIX}asleep] The archive sang to me."),
            )
            .mode(SLEEP_RETURN_MODE),
        )
        .unwrap();
    (normal_ids, dream.id)
}

fn dream_extractor(
    store: &Arc<AsyncHistoryStore>,
    llm: &Arc<ScriptedLlm>,
    clause: Option<&str>,
) -> FactExtractor {
    FactExtractor::new(store.clone(), llm.clone() as Arc<dyn LlmClient>, "fam")
        .familiar_display_name("Sapphire")
        .batch_size(10)
        .dream_extraction_clause(clause.map_or_else(default_dream_clause, str::to_string))
}

#[tokio::test]
async fn dream_turn_shown_to_llm_with_dream_rule() {
    let store = store();
    seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    dream_extractor(&store, &llm, None).tick().await.unwrap();

    assert_eq!(llm.call_count(), 1);
    let system = system_text(&llm.calls()[0]);
    let user = user_text(&llm.calls()[0]);
    assert!(user.contains("The archive sang to me."));
    assert!(system.to_lowercase().contains("dream"));
    assert!(system.contains("dreamed"));
}

#[tokio::test]
async fn no_dream_clause_without_dream_turns() {
    let store = store();
    seed_turns(&store, 10, 1);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    assert!(
        !system_text(&llm.calls()[0])
            .to_lowercase()
            .contains("dream")
    );
}

#[tokio::test]
async fn configured_dream_clause_reaches_llm() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    dream_extractor(
        &store,
        &llm,
        Some("DREAM-MARKER {self_name} keyed {self_key} ids {ids}"),
    )
    .tick()
    .await
    .unwrap();
    let system = system_text(&llm.calls()[0]);
    assert!(system.contains("DREAM-MARKER Sapphire keyed ego:fam"));
    assert!(system.contains(&dream_id.to_string()));
}

#[tokio::test]
async fn dream_clause_with_stray_brace_does_not_crash() {
    let store = store();
    seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    dream_extractor(
        &store,
        &llm,
        Some("tidy up {please} for {self_name} a { brace"),
    )
    .tick()
    .await
    .unwrap();
    let system = system_text(&llm.calls()[0]);
    assert!(system.contains("tidy up {please} for Sapphire a { brace"));
}

#[tokio::test]
async fn claim_discipline_rail_fires_with_config_clause() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cor fought a dragon in the rafters.", "source_turn_ids": [dream_id],
             "subject_keys": ["discord:1"]}
        ]))],
        "[]",
    );
    dream_extractor(&store, &llm, Some("say anything about {self_name}"))
        .tick()
        .await
        .unwrap();
    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(
        facts[0].text,
        "Sapphire dreamed that Cor fought a dragon in the rafters."
    );
    assert_eq!(facts[0].subjects.len(), 1);
    assert_eq!(facts[0].subjects[0].canonical_key, "ego:fam");
    assert_eq!(facts[0].subjects[0].display_at_write, "Sapphire");
}

#[tokio::test]
async fn dream_fact_forced_to_self_subject_and_framed() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cor fought a dragon in the rafters.", "source_turn_ids": [dream_id],
             "subject_keys": ["discord:1"]}
        ]))],
        "[]",
    );
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(
        facts[0].text,
        "Sapphire dreamed that Cor fought a dragon in the rafters."
    );
    assert_eq!(facts[0].subjects[0].canonical_key, "ego:fam");
    assert!(store.sync().mentions_for_turn(dream_id).unwrap().is_empty());
}

#[tokio::test]
async fn already_dream_framed_text_kept_verbatim() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Sapphire dreamed the archive sang to her.", "source_turn_ids": [dream_id],
             "subject_keys": ["ego:fam"]}
        ]))],
        "[]",
    );
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    assert_eq!(
        recent_facts(&store)[0].text,
        "Sapphire dreamed the archive sang to her."
    );
}

#[tokio::test]
async fn mixed_sources_count_as_dream() {
    let store = store();
    let (normal_ids, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new(
        [facts_json(json!([
            {"text": "Cor was in the dream too.", "source_turn_ids": [normal_ids[0], dream_id],
             "subject_keys": ["discord:1"]}
        ]))],
        "[]",
    );
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    let facts = recent_facts(&store);
    assert_eq!(facts[0].subjects.len(), 1);
    assert_eq!(facts[0].subjects[0].canonical_key, "ego:fam");
}

#[tokio::test]
async fn fallback_sources_exclude_dream_turn() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new(
        [facts_json(json!([{"text": "Cor likes strawberries."}]))],
        "[]",
    );
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    let facts = recent_facts(&store);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].text, "Cor likes strawberries.");
    assert!(!facts[0].source_turn_ids.contains(&dream_id));
}

#[tokio::test]
async fn watermark_advances_over_dream_turn() {
    let store = store();
    let (_, dream_id) = seed_with_dream_turn(&store);
    let llm = ScriptedLlm::new([facts_json(json!([]))], "[]");
    dream_extractor(&store, &llm, None).tick().await.unwrap();
    let wm = store.sync().get_writer_watermark("fam").unwrap().unwrap();
    assert_eq!(wm.last_written_id, dream_id);
}
