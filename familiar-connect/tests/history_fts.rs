//! Integration tests for the tantivy FTS side-indexes reached through
//! `HistoryStore` — search scoping, ranking determinism, rebuild, query
//! sanitization, and `append_turn` resilience when the FTS write fails.
//! Ports `test_history_fts.py` (`TestFtsSearch`) plus the FTS-backed
//! `search_facts` / `search_facts_scored` cases of `test_facts_store.py`
//! (behavior 20 — both indexes are real `TantivyFts::in_memory()` under
//! `HistoryStore::open(":memory:")`).
//!
//! The pure commit-retry tests (Python monkeypatched `FtsIndex._commit_writer`)
//! live as in-module unit tests in `src/history/fts.rs`, exercising the
//! `TantivyFts::set_commit_fault` seam directly.

use chrono::{TimeZone, Utc};
use familiar_connect::history::{
    AppendFact, AppendTurn, Author, FactDraft, FactSubject, FtsIndex, HistoryStore, NewFact,
    StoreError,
};

const FAM: &str = "fam";

fn alice() -> Author {
    Author::new(
        "discord",
        "1",
        Some("alice".to_owned()),
        Some("Alice".to_owned()),
    )
}

/// Five turns; the fox mentions land on channel 100 (turns 0, 2, 4), the
/// non-fox rows on channel 101 (turns 1, 3) — mirrors the Python fixture.
fn store_with_turns() -> HistoryStore {
    let store = HistoryStore::open(":memory:").unwrap();
    let texts = [
        "The fox jumped over the moon.",
        "I like strawberry jam.",
        "The quick brown fox.",
        "Rainy day in Seattle.",
        "Anything with foxes is cool.",
    ];
    for (i, text) in texts.iter().enumerate() {
        let channel = 100 + i64::try_from(i % 2).unwrap();
        store
            .append_turn(AppendTurn::new(FAM, channel, "user", *text).author(alice()))
            .unwrap();
    }
    store
}

fn search(
    store: &HistoryStore,
    query: &str,
    limit: i64,
) -> Vec<familiar_connect::history::HistoryTurn> {
    store.search_turns(FAM, query, limit, None, None).unwrap()
}

#[test]
fn finds_matching_turns() {
    let store = store_with_turns();
    let results = search(&store, "fox", 10);
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.content.to_lowercase().contains("fox"))
    );
    // three "fox" turns above
    assert_eq!(results.len(), 3);
}

#[test]
fn scopes_to_channel_when_requested() {
    let store = store_with_turns();
    let results = store.search_turns(FAM, "fox", 10, Some(100), None).unwrap();
    // channel 100 gets turns 0, 2, 4 — all fox mentions
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.channel_id == 100));

    let results_101 = store.search_turns(FAM, "fox", 10, Some(101), None).unwrap();
    // channel 101 gets turns 1 (jam) and 3 (rainy) — no fox
    assert!(results_101.is_empty());
}

#[test]
fn scopes_to_familiar() {
    let store = store_with_turns();
    store
        .append_turn(AppendTurn::new(
            "other",
            100,
            "user",
            "A unicorn appeared on the other familiar.",
        ))
        .unwrap();
    // "unicorn" is unique to the other-familiar row, which is scoped out.
    let results = search(&store, "unicorn", 10);
    assert!(results.is_empty());
}

#[test]
fn respects_limit() {
    let store = store_with_turns();
    assert_eq!(search(&store, "fox", 1).len(), 1);
}

#[test]
fn deterministic_order() {
    let store = store_with_turns();
    let r1: Vec<i64> = search(&store, "fox", 10).iter().map(|t| t.id).collect();
    let r2: Vec<i64> = search(&store, "fox", 10).iter().map(|t| t.id).collect();
    assert_eq!(r1, r2);
}

#[test]
fn rebuild_from_scratch() {
    let store = store_with_turns();
    assert!(!search(&store, "fox", 10).is_empty());
    store.rebuild_fts().unwrap();
    assert_eq!(search(&store, "fox", 10).len(), 3);
}

#[test]
fn empty_query_returns_nothing() {
    let store = store_with_turns();
    assert!(search(&store, "", 10).is_empty());
}

#[test]
fn unknown_term_returns_nothing() {
    let store = store_with_turns();
    assert!(search(&store, "zyzzyx", 10).is_empty());
}

#[test]
fn punctuation_in_query_is_tolerated() {
    let store = store_with_turns();
    // Must not panic; may return empty.
    let _ = search(&store, "fox?", 10);
}

#[test]
fn chat_punctuation_does_not_poison_recall() {
    let store = store_with_turns();
    for cue in [
        "isn't that fox cool",   // apostrophe contraction
        "the fox's tail",        // possessive
        "fox: real or not",      // colon (tantivy field syntax)
        "he said \"fox\" again", // quotes (tantivy phrase syntax)
        "fox (the animal)",      // parens (tantivy grouping)
    ] {
        let results = search(&store, cue, 10);
        assert!(!results.is_empty(), "cue {cue:?} returned no hits");
        assert!(
            results
                .iter()
                .all(|r| r.content.to_lowercase().contains("fox")),
            "cue {cue:?} surfaced a non-fox turn"
        );
    }
}

#[test]
fn uppercase_boolean_words_treated_as_text() {
    // AND/OR/NOT in chat are words, not operators — recall stays disjunctive.
    let store = store_with_turns();
    let results = search(&store, "fox AND strawberry", 10);
    let contents: Vec<String> = results.iter().map(|r| r.content.to_lowercase()).collect();
    assert!(contents.iter().any(|c| c.contains("fox")));
    assert!(contents.iter().any(|c| c.contains("strawberry")));
}

#[test]
fn latest_indexed_id_tracks_writes() {
    let store = store_with_turns();
    assert_eq!(
        store.latest_fts_id(FAM).unwrap(),
        store.latest_id(FAM, None).unwrap().unwrap()
    );
}

#[test]
fn chat_cue_with_stopwords_still_recalls_substantive_match() {
    let store = store_with_turns();
    let results = search(&store, "Hey, do you know anything about the fox?", 10);
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.content.to_lowercase().contains("fox"))
    );
}

#[test]
fn query_of_only_stopwords_returns_nothing() {
    let store = store_with_turns();
    assert!(search(&store, "hi the a do you", 10).is_empty());
}

#[test]
fn search_turns_respects_max_id() {
    let store = store_with_turns();
    let full = search(&store, "fox", 10);
    let max_id = full.iter().map(|t| t.id).min().unwrap();
    let bounded = store
        .search_turns(FAM, "fox", 10, None, Some(max_id))
        .unwrap();
    assert!(!bounded.is_empty());
    assert!(bounded.iter().all(|r| r.id <= max_id));
    // bounded is a strict subset of full
    assert!(bounded.len() < full.len());
}

// --- append_turn survives an FTS write failure ----------------------------

/// A store-side FTS double whose every write fails — stands in for Python's
/// monkeypatched `store._fts_turns.add`.
struct FailingFts;

impl FtsIndex for FailingFts {
    fn add(&self, _row_id: i64, _content: &str) -> Result<(), StoreError> {
        Err(StoreError::Fts(
            "Failed to open file for write: PermissionDenied".to_owned(),
        ))
    }
    fn add_many(&self, _rows: &[(i64, String)]) -> Result<(), StoreError> {
        Err(StoreError::Fts("boom".to_owned()))
    }
    fn delete(&self, _row_id: i64) -> Result<(), StoreError> {
        Err(StoreError::Fts("boom".to_owned()))
    }
    fn clear(&self) -> Result<(), StoreError> {
        Err(StoreError::Fts("boom".to_owned()))
    }
    fn search(&self, _query: &str, _limit: usize) -> Vec<(i64, f32)> {
        Vec::new()
    }
}

#[test]
fn append_turn_survives_fts_commit_failure() {
    // SQL row persists even when the FTS index can't commit.
    let store = HistoryStore::open_with_fts(":memory:", Box::new(FailingFts), Box::new(FailingFts))
        .unwrap();
    let turn = store
        .append_turn(AppendTurn::new(FAM, 100, "user", "this should still persist").author(alice()))
        .unwrap();
    assert!(turn.id > 0);
    let recent = store.recent(FAM, 100, 10, None, None).unwrap();
    assert!(
        recent
            .iter()
            .any(|r| r.content == "this should still persist")
    );
}

// --- search_facts / search_facts_scored (behavior 20) ---------------------
//
// The facts index is the real in-memory tantivy index (same analyzer as turns),
// so these exercise: BM25 positivity / higher-is-better, the 4x overfetch, the
// facts-validity join over FTS candidates (superseded exclusion + as_of slice),
// and the (-score, -id) re-rank. Ports the FTS-backed cases of
// `test_facts_store.py`.

fn subj(key: &str, display: &str) -> FactSubject {
    FactSubject {
        canonical_key: key.to_owned(),
        display_at_write: display.to_owned(),
    }
}

/// Two facts (`strawberries`, `night shifts`) under `FAM`, mirroring Python's
/// `_store_with_turns_and_facts` fixture.
fn store_with_facts() -> HistoryStore {
    let store = HistoryStore::open(":memory:").unwrap();
    for i in 0..5 {
        store
            .append_turn(AppendTurn::new(FAM, 1, "user", format!("turn text {i}")))
            .unwrap();
    }
    store
        .append_fact(AppendFact::new(
            FAM,
            Some(1),
            "Aria likes strawberries.",
            vec![1, 2],
        ))
        .unwrap();
    store
        .append_fact(AppendFact::new(
            FAM,
            Some(1),
            "Boris works night shifts on Tuesdays.",
            vec![3, 4],
        ))
        .unwrap();
    store
}

#[test]
fn search_facts_finds_by_content() {
    let store = store_with_facts();
    // English stemmer maps "strawberry"/"strawberries" to a shared stem.
    let found = store
        .search_facts(FAM, "strawberry", 5, false, None)
        .unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].text.contains("strawberries"));
}

#[test]
fn search_facts_empty_query_returns_nothing() {
    let store = store_with_facts();
    assert!(
        store
            .search_facts(FAM, "", 10, false, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn search_facts_respects_familiar() {
    let store = store_with_facts();
    store
        .append_fact(AppendFact::new(
            "other",
            Some(1),
            "Other familiar knows strawberries too.",
            vec![1],
        ))
        .unwrap();
    let found = store
        .search_facts(FAM, "strawberry", 10, false, None)
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].familiar_id, FAM);
}

#[test]
fn search_facts_returns_subjects() {
    let store = HistoryStore::open(":memory:").unwrap();
    let subjects = vec![subj("discord:123", "Cass")];
    store
        .append_fact(
            AppendFact::new(FAM, Some(1), "Cass likes pho.", vec![1]).subjects(subjects.clone()),
        )
        .unwrap();
    let found = store.search_facts(FAM, "pho", 5, false, None).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].subjects, subjects);
}

#[test]
fn search_facts_excludes_superseded() {
    let store = store_with_facts();
    let new = store
        .append_fact(AppendFact::new(
            FAM,
            Some(1),
            "Aria is allergic to strawberries now.",
            vec![5],
        ))
        .unwrap();
    store
        .supersede(FAM, &[1], NewFact::Repoint(new.id))
        .unwrap();
    let texts: std::collections::HashSet<String> = store
        .search_facts(FAM, "strawberry", 10, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert!(!texts.contains("Aria likes strawberries."));
    assert!(texts.contains("Aria is allergic to strawberries now."));
}

#[test]
fn search_facts_supports_as_of() {
    let store = HistoryStore::open(":memory:").unwrap();
    let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
    store
        .append_fact(
            AppendFact::new(FAM, Some(1), "Aria worked at Acme.", vec![1])
                .valid_from(t1)
                .valid_to(t2),
        )
        .unwrap();
    store
        .append_fact(AppendFact::new(FAM, Some(1), "Aria works at Globex.", vec![2]).valid_from(t2))
        .unwrap();
    let march = Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap();
    let texts: std::collections::HashSet<String> = store
        .search_facts(FAM, "Aria", 10, false, Some(march))
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert_eq!(
        texts,
        std::iter::once("Aria worked at Acme.".to_owned()).collect()
    );
}

#[test]
fn search_facts_scored_returns_positive_bm25_best_first() {
    let store = HistoryStore::open(":memory:").unwrap();
    store
        .append_fact(
            AppendFact::new(FAM, Some(1), "Aria likes strawberries.", vec![1]).importance(8),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new(FAM, Some(1), "Boris likes strawberries too.", vec![2]).importance(2),
        )
        .unwrap();
    let scored = store
        .search_facts_scored(FAM, "strawberry", 5, false, None)
        .unwrap();
    assert_eq!(scored.len(), 2);
    // tantivy BM25 is positive, higher = better — exposed verbatim for fusion.
    for (_fact, score) in &scored {
        assert!(*score > 0.0, "expected positive BM25, got {score}");
    }
    // Result is sorted BM25 desc (best first). The two docs tokenize to the same
    // multiset ("too" is a stopword), so scores tie and `<=` holds — matching
    // the Python assertion `scored[0][1] <= scored[1][1]`.
    assert!(scored[0].1 <= scored[1].1);
}

#[test]
fn search_facts_finds_merge_minted_fact() {
    // The atomic merge mints its replacement via the same append path, so the
    // minted row must be FTS-searchable (behavior 20 + 31).
    let store = HistoryStore::open(":memory:").unwrap();
    for i in 0..3 {
        store
            .append_turn(AppendTurn::new(FAM, 1, "user", format!("turn {i}")))
            .unwrap();
    }
    let a = store
        .append_fact(AppendFact::new(FAM, Some(1), "Aria likes A.", vec![1]))
        .unwrap();
    let b = store
        .append_fact(AppendFact::new(FAM, Some(1), "Aria likes B.", vec![2]))
        .unwrap();
    let c = store
        .append_fact(AppendFact::new(FAM, Some(1), "Aria likes C.", vec![3]))
        .unwrap();
    let result = store
        .supersede(
            FAM,
            &[a.id, b.id, c.id],
            NewFact::Merge(FactDraft {
                channel_id: Some(1),
                text: "Aria likes A, B, and C.".to_owned(),
                subjects: vec![],
            }),
        )
        .unwrap();
    let minted = result.minted.unwrap();
    let found = store.search_facts(FAM, "Aria", 10, false, None).unwrap();
    assert!(found.iter().any(|f| f.id == minted.id));
}
