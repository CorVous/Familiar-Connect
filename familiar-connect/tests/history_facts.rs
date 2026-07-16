//! Integration tests for facts, supersession, dedup, bi-temporal validity,
//! importance, fact embeddings, people dossiers, and reflections. Ports the
//! FTS-independent parts of `test_facts_store.py`, `test_fact_embeddings_store.py`,
//! `test_people_dossiers_store.py`, `test_reflections_store.py`.
//!
//! Test-local numeric casts (loop index → id/vector) and single-item set
//! builders are harmless here; silence the corresponding lints for test code.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::iter_on_single_items
)]

#[path = "log_capture/mod.rs"]
mod log_capture;

use chrono::{TimeZone, Utc};
use familiar_connect::history::{
    AppendFact, AppendTurn, FactDraft, FactSubject, HistoryStore, NewFact,
};

use log_capture::LogCapture;

fn mem() -> HistoryStore {
    HistoryStore::open(":memory:").unwrap()
}

fn subj(key: &str, display: &str) -> FactSubject {
    FactSubject {
        canonical_key: key.to_owned(),
        display_at_write: display.to_owned(),
    }
}

fn seed_turns_and_facts() -> HistoryStore {
    let store = mem();
    for i in 0..5 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", format!("turn text {i}")))
            .unwrap();
    }
    store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria likes strawberries.",
            vec![1, 2],
        ))
        .unwrap();
    store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Boris works night shifts on Tuesdays.",
            vec![3, 4],
        ))
        .unwrap();
    store
}

// --- fact store -----------------------------------------------------------

#[test]
fn append_returns_fact_with_provenance() {
    let store = mem();
    let fact = store
        .append_fact(AppendFact::new("fam", Some(1), "A fact.", vec![7, 9, 11]))
        .unwrap();
    assert_eq!(fact.source_turn_ids, [7, 9, 11]);
    assert_eq!(fact.text, "A fact.");
    assert_eq!(fact.channel_id, Some(1));
    assert!(fact.id > 0);
}

#[test]
fn recent_facts_newest_first_and_provenance() {
    let store = seed_turns_and_facts();
    let recents = store.recent_facts("fam", 10, false, None).unwrap();
    assert_eq!(recents.len(), 2);
    assert!(recents[0].text.starts_with("Boris"));
    assert!(recents[1].text.starts_with("Aria"));
    assert_eq!(recents[0].source_turn_ids, [3, 4]);
    assert_eq!(recents[1].source_turn_ids, [1, 2]);
}

#[test]
fn latest_fact_id_and_all_ids() {
    let store = seed_turns_and_facts();
    assert_eq!(store.latest_fact_id("fam").unwrap(), 2);
    assert_eq!(store.latest_fact_id("nobody").unwrap(), 0);
    assert_eq!(store.all_fact_ids("fam").unwrap().len(), 2);
}

#[test]
fn subjects_round_trip() {
    let store = mem();
    let subjects = vec![subj("discord:123", "Cass"), subj("discord:456", "Aria")];
    let fact = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass and Aria are roommates.", vec![1])
                .subjects(subjects.clone()),
        )
        .unwrap();
    assert_eq!(fact.subjects, subjects);
    let recents = store.recent_facts("fam", 10, false, None).unwrap();
    assert_eq!(recents[0].subjects, subjects);
    // default empty
    let bare = store
        .append_fact(AppendFact::new("fam", Some(1), "no subjects.", vec![1]))
        .unwrap();
    assert!(bare.subjects.is_empty());
}

// --- supersession ---------------------------------------------------------

#[test]
fn supersede_marks_old_and_links_new() {
    let store = seed_turns_and_facts();
    let new = store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria is allergic to strawberries now.",
            vec![5],
        ))
        .unwrap();
    store
        .supersede("fam", &[1], NewFact::Repoint(new.id))
        .unwrap();
    let all = store.recent_facts("fam", 10, true, None).unwrap();
    let old = all.iter().find(|f| f.id == 1).unwrap();
    assert!(old.superseded_at.is_some());
    assert_eq!(old.superseded_by, Some(new.id));
    // default excludes superseded
    let current: std::collections::HashSet<String> = store
        .recent_facts("fam", 10, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert!(!current.contains("Aria likes strawberries."));
    assert!(current.contains("Aria is allergic to strawberries now."));
}

#[test]
fn retire_form_excludes_and_leaves_by() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    let f = store
        .append_fact(AppendFact::new("fam", Some(1), "junk.", vec![1]))
        .unwrap();
    let result = store.supersede("fam", &[f.id], NewFact::Retire).unwrap();
    assert!(result.minted.is_none());
    assert!(
        store
            .recent_facts("fam", 10, false, None)
            .unwrap()
            .is_empty()
    );
    let all = store.recent_facts("fam", 10, true, None).unwrap();
    let row = all.iter().find(|r| r.id == f.id).unwrap();
    assert!(row.superseded_at.is_some());
    assert!(row.superseded_by.is_none());
}

#[test]
fn merge_points_all_at_minted_and_ancestry_and_provenance() {
    let store = mem();
    for i in 0..6 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", format!("turn {i}")))
            .unwrap();
    }
    let a = store
        .append_fact(AppendFact::new("fam", Some(1), "Aria likes A.", vec![1, 2]))
        .unwrap();
    let b = store
        .append_fact(AppendFact::new("fam", Some(1), "Aria likes B.", vec![2, 3]))
        .unwrap();
    let c = store
        .append_fact(AppendFact::new("fam", Some(1), "Aria likes C.", vec![4]))
        .unwrap();
    let result = store
        .supersede(
            "fam",
            &[a.id, b.id, c.id],
            NewFact::Merge(FactDraft {
                channel_id: Some(1),
                text: "Aria likes A, B, and C.".to_owned(),
                subjects: vec![],
            }),
        )
        .unwrap();
    let minted = result.minted.unwrap();
    let all = store.recent_facts("fam", 10, true, None).unwrap();
    let by_id: std::collections::HashMap<i64, _> = all.iter().map(|f| (f.id, f)).collect();
    assert_eq!(by_id[&a.id].superseded_by, Some(minted.id));
    assert_eq!(by_id[&b.id].superseded_by, Some(minted.id));
    assert_eq!(by_id[&c.id].superseded_by, Some(minted.id));
    // minted is current
    assert!(
        store
            .recent_facts("fam", 10, false, None)
            .unwrap()
            .iter()
            .any(|r| r.id == minted.id)
    );
    // ancestry (store-owned) + provenance union
    let ancestors: std::collections::HashSet<i64> = store
        .ancestors_of("fam", minted.id)
        .unwrap()
        .into_iter()
        .map(|f| f.id)
        .collect();
    assert_eq!(ancestors, [a.id, b.id, c.id].into_iter().collect());
    assert_eq!(
        minted
            .source_turn_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        [1, 2, 3, 4].into_iter().collect()
    );
}

#[test]
fn merge_declined_atomically_when_any_stale() {
    let store = mem();
    for i in 0..6 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", format!("t{i}")))
            .unwrap();
    }
    let a = store
        .append_fact(AppendFact::new("fam", Some(1), "fact A.", vec![1]))
        .unwrap();
    let b = store
        .append_fact(AppendFact::new("fam", Some(1), "fact B.", vec![2]))
        .unwrap();
    let before = store.latest_fact_id("fam").unwrap();
    store.supersede("fam", &[a.id], NewFact::Retire).unwrap();
    let result = store
        .supersede(
            "fam",
            &[a.id, b.id],
            NewFact::Merge(FactDraft {
                channel_id: Some(1),
                text: "merged.".to_owned(),
                subjects: vec![],
            }),
        )
        .unwrap();
    assert!(result.minted.is_none());
    assert!(result.superseded.is_empty());
    assert_eq!(store.latest_fact_id("fam").unwrap(), before);
    let all = store.recent_facts("fam", 10, true, None).unwrap();
    let by_id: std::collections::HashMap<i64, _> = all.iter().map(|f| (f.id, f)).collect();
    assert!(by_id[&b.id].superseded_at.is_none());
    assert!(result.skipped.iter().any(|(id, _)| *id == a.id));
}

#[test]
fn merge_empty_obsolete_is_noop() {
    let store = mem();
    let before = store.latest_fact_id("fam").unwrap();
    let result = store
        .supersede(
            "fam",
            &[],
            NewFact::Merge(FactDraft {
                channel_id: Some(1),
                text: "orphan.".to_owned(),
                subjects: vec![],
            }),
        )
        .unwrap();
    assert!(result.minted.is_none());
    assert!(result.superseded.is_empty());
    assert_eq!(store.latest_fact_id("fam").unwrap(), before);
}

#[test]
fn retire_and_repoint_skip_stale_process_rest() {
    let store = mem();
    for i in 0..6 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", format!("t{i}")))
            .unwrap();
    }
    let a = store
        .append_fact(AppendFact::new("fam", Some(1), "fact A.", vec![1]))
        .unwrap();
    let b = store
        .append_fact(AppendFact::new("fam", Some(1), "fact B.", vec![2]))
        .unwrap();
    store.supersede("fam", &[a.id], NewFact::Retire).unwrap();
    let result = store
        .supersede("fam", &[a.id, b.id], NewFact::Retire)
        .unwrap();
    assert!(result.superseded.contains(&b.id));
    assert!(!result.superseded.contains(&a.id));
    assert!(
        result
            .skipped
            .iter()
            .any(|(id, msg)| *id == a.id && msg == &format!("fact id={} already superseded", a.id))
    );

    // repoint form
    let store2 = mem();
    for i in 0..6 {
        store2
            .append_turn(AppendTurn::new("fam", 1, "user", format!("t{i}")))
            .unwrap();
    }
    let a = store2
        .append_fact(AppendFact::new("fam", Some(1), "fact A.", vec![1]))
        .unwrap();
    let b = store2
        .append_fact(AppendFact::new("fam", Some(1), "fact B.", vec![2]))
        .unwrap();
    let existing = store2
        .append_fact(AppendFact::new("fam", Some(1), "replacement.", vec![3]))
        .unwrap();
    store2.supersede("fam", &[a.id], NewFact::Retire).unwrap();
    let before = store2.latest_fact_id("fam").unwrap();
    let result = store2
        .supersede("fam", &[a.id, b.id], NewFact::Repoint(existing.id))
        .unwrap();
    assert_eq!(store2.latest_fact_id("fam").unwrap(), before);
    assert!(result.minted.is_none());
    assert!(result.superseded.contains(&b.id));
    let all = store2.recent_facts("fam", 10, true, None).unwrap();
    let by_id: std::collections::HashMap<i64, _> = all.iter().map(|f| (f.id, f)).collect();
    assert_eq!(by_id[&b.id].superseded_by, Some(existing.id));
}

#[test]
fn unknown_id_skip_reason() {
    let store = mem();
    let result = store.supersede("fam", &[999], NewFact::Retire).unwrap();
    assert!(
        result
            .skipped
            .iter()
            .any(|(id, msg)| *id == 999 && msg == "unknown fact id=999")
    );
}

// --- dossier invalidation on supersede ------------------------------------

fn store_with_subject_fact() -> (HistoryStore, i64, i64) {
    let store = mem();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    let subjects = vec![subj("discord:A", "Aria")];
    let old = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria loves hiking.", vec![1])
                .subjects(subjects.clone()),
        )
        .unwrap();
    let new = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria hates hiking now.", vec![1]).subjects(subjects),
        )
        .unwrap();
    (store, old.id, new.id)
}

#[test]
fn supersede_deletes_subject_dossier() {
    let (store, old_id, new_id) = store_with_subject_fact();
    store
        .put_people_dossier("fam", "discord:A", old_id, "Aria loves hiking.")
        .unwrap();
    store
        .supersede("fam", &[old_id], NewFact::Repoint(new_id))
        .unwrap();
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none()
    );
}

#[test]
fn put_dossier_if_current_updates_when_watermark_matches() {
    // The common (non-raced) path: the row is still at the read-time watermark,
    // so the CAS lands and advances it.
    let store = mem();
    store
        .put_people_dossier("fam", "discord:A", 5, "v1")
        .unwrap();
    let landed = store
        .put_people_dossier_if_current("fam", "discord:A", Some(5), 9, "v2")
        .unwrap();
    assert!(landed);
    let entry = store
        .get_people_dossier("fam", "discord:A")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "v2");
    assert_eq!(entry.last_fact_id, 9);
}

#[test]
fn put_dossier_if_current_drops_write_after_supersede_delete() {
    // #130: a concurrent supersede deletes the row between the worker's read (at
    // watermark = old_id) and its write; the stale CAS must NOT resurrect it.
    let (store, old_id, new_id) = store_with_subject_fact();
    store
        .put_people_dossier("fam", "discord:A", old_id, "Aria loves hiking.")
        .unwrap();
    store
        .supersede("fam", &[old_id], NewFact::Repoint(new_id))
        .unwrap();
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none()
    );

    let landed = store
        .put_people_dossier_if_current("fam", "discord:A", Some(old_id), new_id, "stale prose")
        .unwrap();
    assert!(!landed, "stale CAS must not land");
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none(),
        "dropped write must not resurrect the invalidated row",
    );
}

#[test]
fn put_dossier_if_current_stale_watermark_does_not_clobber() {
    // Another writer already advanced the watermark; a CAS at the old watermark
    // must not overwrite the newer row.
    let store = mem();
    store
        .put_people_dossier("fam", "discord:A", 3, "current")
        .unwrap();
    let landed = store
        .put_people_dossier_if_current("fam", "discord:A", Some(1), 9, "stale")
        .unwrap();
    assert!(!landed);
    let entry = store
        .get_people_dossier("fam", "discord:A")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "current");
    assert_eq!(entry.last_fact_id, 3);
}

#[test]
fn put_dossier_if_current_inserts_when_absent() {
    // None-prior (fresh subject): a clean insert lands.
    let store = mem();
    let landed = store
        .put_people_dossier_if_current("fam", "discord:A", None, 4, "fresh")
        .unwrap();
    assert!(landed);
    let entry = store
        .get_people_dossier("fam", "discord:A")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "fresh");
    assert_eq!(entry.last_fact_id, 4);
}

#[test]
fn put_dossier_if_current_none_prior_does_not_clobber_racing_insert() {
    // None-prior but a racing writer already created the row: DO NOTHING and
    // report the write did not land.
    let store = mem();
    store
        .put_people_dossier("fam", "discord:A", 7, "winner")
        .unwrap();
    let landed = store
        .put_people_dossier_if_current("fam", "discord:A", None, 4, "loser")
        .unwrap();
    assert!(!landed);
    let entry = store
        .get_people_dossier("fam", "discord:A")
        .unwrap()
        .unwrap();
    assert_eq!(entry.dossier_text, "winner");
    assert_eq!(entry.last_fact_id, 7);
}

#[test]
fn supersede_null_subject_leaves_dossiers() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    let old = store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "The weather is nice.",
            vec![1],
        ))
        .unwrap();
    let new = store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "The weather turned grim.",
            vec![1],
        ))
        .unwrap();
    store
        .put_people_dossier("fam", "discord:A", 1, "Aria loves hiking.")
        .unwrap();
    store
        .supersede("fam", &[old.id], NewFact::Repoint(new.id))
        .unwrap();
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_some()
    );
}

#[test]
fn supersede_spares_other_subjects_and_familiars() {
    let (store, old_id, new_id) = store_with_subject_fact();
    store
        .put_people_dossier("fam", "discord:A", old_id, "Aria loves hiking.")
        .unwrap();
    store
        .put_people_dossier("fam", "discord:B", old_id, "Boris works nights.")
        .unwrap();
    store
        .put_people_dossier("other", "discord:A", old_id, "Aria, per other familiar.")
        .unwrap();
    store
        .supersede("fam", &[old_id], NewFact::Repoint(new_id))
        .unwrap();
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get_people_dossier("fam", "discord:B")
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .get_people_dossier("other", "discord:A")
            .unwrap()
            .is_some()
    );
}

#[test]
fn merge_invalidates_dossier_per_obsolete_subject() {
    let store = mem();
    for i in 0..6 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", format!("t{i}")))
            .unwrap();
    }
    let a = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria thing.", vec![1])
                .subjects(vec![subj("discord:A", "Aria")]),
        )
        .unwrap();
    let b = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Boris thing.", vec![2])
                .subjects(vec![subj("discord:B", "Boris")]),
        )
        .unwrap();
    for key in ["discord:A", "discord:B"] {
        store
            .put_people_dossier("fam", key, a.id, "baked.")
            .unwrap();
    }
    store
        .supersede(
            "fam",
            &[a.id, b.id],
            NewFact::Merge(FactDraft {
                channel_id: Some(1),
                text: "merged.".to_owned(),
                subjects: vec![],
            }),
        )
        .unwrap();
    assert!(
        store
            .get_people_dossier("fam", "discord:A")
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get_people_dossier("fam", "discord:B")
            .unwrap()
            .is_none()
    );
}

// --- facts_by_ids ---------------------------------------------------------

#[test]
fn facts_by_ids_includes_superseded_and_scoped() {
    let store = seed_turns_and_facts();
    let new = store
        .append_fact(AppendFact::new("fam", Some(1), "replacement", vec![5]))
        .unwrap();
    store
        .supersede("fam", &[1], NewFact::Repoint(new.id))
        .unwrap();
    let got = store.facts_by_ids("fam", &[1, 2]).unwrap();
    assert_eq!(
        got.iter()
            .map(|f| f.id)
            .collect::<std::collections::HashSet<_>>(),
        [1, 2].into_iter().collect()
    );
    assert!(
        got.iter()
            .find(|f| f.id == 1)
            .unwrap()
            .superseded_at
            .is_some()
    );
    assert!(store.facts_by_ids("fam", &[]).unwrap().is_empty());
    assert!(store.facts_by_ids("fam", &[999]).unwrap().is_empty());
}

// --- sleep watermark ------------------------------------------------------

#[test]
fn sleep_watermark_partial_axis() {
    let store = mem();
    assert!(store.get_sleep_watermark("fam").unwrap().is_none());
    store
        .advance_sleep_watermark("fam", Some(42), None)
        .unwrap();
    let wm = store.get_sleep_watermark("fam").unwrap().unwrap();
    assert_eq!((wm.last_fact_id, wm.last_turn_id), (42, 0));
    store
        .advance_sleep_watermark("fam", None, Some(99))
        .unwrap();
    let wm = store.get_sleep_watermark("fam").unwrap().unwrap();
    assert_eq!((wm.last_fact_id, wm.last_turn_id), (42, 99));
    // noop
    let store2 = mem();
    store2.advance_sleep_watermark("fam", None, None).unwrap();
    assert!(store2.get_sleep_watermark("fam").unwrap().is_none());
}

// --- bi-temporal ----------------------------------------------------------

#[test]
fn bitemporal_defaults_and_explicit() {
    let store = mem();
    let before = Utc::now();
    let fact = store
        .append_fact(AppendFact::new("fam", Some(1), "A fact.", vec![]))
        .unwrap();
    let after = Utc::now();
    let vf = fact.valid_from.unwrap();
    assert!(before <= vf && vf <= after);
    assert!(fact.valid_to.is_none());

    let when = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
    let f2 = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria moved to Berlin.", vec![1]).valid_from(when),
        )
        .unwrap();
    assert_eq!(f2.valid_from, Some(when));
}

#[test]
fn bitemporal_excludes_expired_and_as_of_slices() {
    let store = mem();
    let past = Utc::now() - chrono::Duration::days(30);
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria worked at Acme.", vec![1])
                .valid_from(past - chrono::Duration::days(365))
                .valid_to(past),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria works at Globex.", vec![2]).valid_from(past),
        )
        .unwrap();
    let current: std::collections::HashSet<String> = store
        .recent_facts("fam", 10, false, None)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert_eq!(
        current,
        ["Aria works at Globex.".to_owned()].into_iter().collect()
    );

    // as_of world-time slice
    let store2 = mem();
    let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
    store2
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria worked at Acme.", vec![1])
                .valid_from(t1)
                .valid_to(t2),
        )
        .unwrap();
    store2
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria works at Globex.", vec![2]).valid_from(t2),
        )
        .unwrap();
    let march = Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap();
    let slice: std::collections::HashSet<String> = store2
        .recent_facts("fam", 10, false, Some(march))
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert_eq!(
        slice,
        ["Aria worked at Acme.".to_owned()].into_iter().collect()
    );
}

#[test]
fn as_of_includes_superseded_for_audit() {
    let store = mem();
    let old_time = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
    let old = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes strawberries.", vec![1])
                .valid_from(old_time),
        )
        .unwrap();
    let new = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria is allergic to strawberries.", vec![2])
                .valid_from(Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()),
        )
        .unwrap();
    store
        .supersede("fam", &[old.id], NewFact::Repoint(new.id))
        .unwrap();
    let feb = Utc.with_ymd_and_hms(2025, 2, 1, 0, 0, 0).unwrap();
    let slice: std::collections::HashSet<String> = store
        .recent_facts("fam", 10, false, Some(feb))
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect();
    assert!(slice.contains("Aria likes strawberries."));
}

// --- importance -----------------------------------------------------------

#[test]
fn importance_round_trip_and_clamp() {
    let store = mem();
    let fact = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria is allergic to peanuts.", vec![1]).importance(9),
        )
        .unwrap();
    assert_eq!(fact.importance, Some(9));
    let none = store
        .append_fact(AppendFact::new("fam", Some(1), "Casual aside.", vec![1]))
        .unwrap();
    assert!(none.importance.is_none());
    let low = store
        .append_fact(AppendFact::new("fam", Some(1), "too low.", vec![1]).importance(0))
        .unwrap();
    let high = store
        .append_fact(AppendFact::new("fam", Some(1), "too high.", vec![2]).importance(42))
        .unwrap();
    assert_eq!(low.importance, Some(1));
    assert_eq!(high.importance, Some(10));
}

// --- dedup ----------------------------------------------------------------

#[test]
fn dedup_normalized_duplicate_skips_insert() {
    let store = mem();
    let sj = vec![subj("discord:1", "Cor")];
    let first = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Postbirb Prime is called Cor.", vec![1])
                .subjects(sj.clone()),
        )
        .unwrap();
    let again = store
        .append_fact(
            AppendFact::new(
                "fam",
                Some(2),
                "  \"postbirb   prime is called cor\"  ",
                vec![2],
            )
            .subjects(sj),
        )
        .unwrap();
    assert_eq!(
        store.all_fact_ids("fam").unwrap(),
        [first.id].into_iter().collect()
    );
    assert_eq!(again.id, first.id);
}

// Guard audit-log convention (issue #132): the near-duplicate skip on the
// DB-insert path emits exactly one structured debug line naming the guard, and
// stays silent when the insert actually mints. Behaviour is unchanged — the
// second append still collapses onto the first fact.
#[test]
fn dedup_skip_emits_guard_audit_line() {
    let capture = LogCapture::install();
    let store = mem();
    let sj = vec![subj("discord:1", "Cor")];
    let first = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes tea.", vec![1]).subjects(sj.clone()),
        )
        .unwrap();
    // A distinct fact mints (no audit line); the identical fact is skipped.
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes coffee.", vec![2]).subjects(sj.clone()),
        )
        .unwrap();
    let again = store
        .append_fact(AppendFact::new("fam", Some(2), "cor likes tea.", vec![3]).subjects(sj))
        .unwrap();
    assert_eq!(again.id, first.id, "dedup collapses onto the existing fact");

    let out = capture.contents();
    drop(capture);
    assert_eq!(
        out.lines()
            .filter(|l| l.contains("append_fact_dedup"))
            .count(),
        1,
        "exactly one dedup skip line expected: {out}"
    );
    assert!(out.contains("near_duplicate"), "{out}");
    assert!(out.contains("db_insert"), "{out}");
}

#[test]
fn dedup_scoping_matrix() {
    // differing text inserts
    let store = mem();
    let sj = vec![subj("discord:1", "Cor")];
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes tea.", vec![1]).subjects(sj.clone()),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes coffee.", vec![2]).subjects(sj.clone()),
        )
        .unwrap();
    assert_eq!(store.all_fact_ids("fam").unwrap().len(), 2);

    // same text, different subject inserts
    let store = mem();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "likes tea.", vec![1])
                .subjects(vec![subj("discord:1", "Cor")]),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "likes tea.", vec![2])
                .subjects(vec![subj("discord:2", "Aria")]),
        )
        .unwrap();
    assert_eq!(store.all_fact_ids("fam").unwrap().len(), 2);

    // null subject not dup of keyed
    let store = mem();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "likes tea.", vec![1])
                .subjects(vec![subj("discord:1", "Cor")]),
        )
        .unwrap();
    store
        .append_fact(AppendFact::new("fam", Some(1), "likes tea.", vec![2]))
        .unwrap();
    assert_eq!(store.all_fact_ids("fam").unwrap().len(), 2);

    // per-familiar
    let store = mem();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes tea.", vec![1]).subjects(sj.clone()),
        )
        .unwrap();
    store
        .append_fact(AppendFact::new("other", Some(1), "Cor likes tea.", vec![2]).subjects(sj))
        .unwrap();
    assert_eq!(store.all_fact_ids("fam").unwrap().len(), 1);
    assert_eq!(store.all_fact_ids("other").unwrap().len(), 1);
}

#[test]
fn dedup_superseded_does_not_block_and_valid_to_bypasses() {
    let store = mem();
    let sj = vec![subj("discord:1", "Cor")];
    let old = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes tea.", vec![1]).subjects(sj.clone()),
        )
        .unwrap();
    let replacement = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor prefers coffee.", vec![2]).subjects(sj.clone()),
        )
        .unwrap();
    store
        .supersede("fam", &[old.id], NewFact::Repoint(replacement.id))
        .unwrap();
    let again = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor likes tea.", vec![3]).subjects(sj.clone()),
        )
        .unwrap();
    assert!(again.id != old.id && again.id != replacement.id);

    // valid_to bypasses dedup
    let store2 = mem();
    let first = store2
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor lives in Berlin.", vec![1]).subjects(sj.clone()),
        )
        .unwrap();
    let bounding = store2
        .append_fact(
            AppendFact::new("fam", Some(1), "Cor lives in Berlin.", vec![2])
                .subjects(sj)
                .valid_to(Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()),
        )
        .unwrap();
    assert_ne!(bounding.id, first.id);
    assert_eq!(store2.all_fact_ids("fam").unwrap().len(), 2);
}

// --- fact embeddings ------------------------------------------------------

fn store_with_facts(n: usize) -> HistoryStore {
    let store = mem();
    for i in 0..n {
        store
            .append_fact(AppendFact::new(
                "fam",
                Some(1),
                format!("fact number {i}"),
                vec![i as i64],
            ))
            .unwrap();
    }
    store
}

#[test]
fn embedding_round_trip_and_empty_and_upsert() {
    let store = store_with_facts(1);
    let vec = [0.1_f32, -0.2, 0.3, 0.4];
    store.set_fact_embedding(1, "hash-v1", &vec).unwrap();
    let got = store.get_fact_embeddings(&[1], "hash-v1").unwrap();
    assert_eq!(
        got.keys()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        [1].into_iter().collect()
    );
    for (a, b) in got[&1].iter().zip(vec.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    // empty vector errors
    assert!(store.set_fact_embedding(1, "hash-v1", &[]).is_err());
    // upsert overwrites with new dim
    store
        .set_fact_embedding(1, "hash-v1", &[3.0, 4.0, 5.0])
        .unwrap();
    let got = store.get_fact_embeddings(&[1], "hash-v1").unwrap();
    assert_eq!(got[&1].len(), 3);
    assert!((got[&1][2] - 5.0).abs() < 1e-6);
}

#[test]
fn embedding_model_isolation_and_filters() {
    let store = store_with_facts(3);
    store.set_fact_embedding(1, "hash-v1", &[0.1, 0.2]).unwrap();
    store
        .set_fact_embedding(1, "bge-small", &[0.5, 0.6, 0.7])
        .unwrap();
    assert_eq!(
        store.get_fact_embeddings(&[1], "hash-v1").unwrap()[&1].len(),
        2
    );
    assert_eq!(
        store.get_fact_embeddings(&[1], "bge-small").unwrap()[&1].len(),
        3
    );
    assert!(
        store
            .get_fact_embeddings(&[1], "model-b")
            .unwrap()
            .is_empty()
    );
    for i in 1..=3 {
        store.set_fact_embedding(i, "m", &[i as f32]).unwrap();
    }
    let got = store.get_fact_embeddings(&[1, 3], "m").unwrap();
    assert_eq!(
        got.keys()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        [1, 3].into_iter().collect()
    );
    assert!(store.get_fact_embeddings(&[], "m").unwrap().is_empty());
}

#[test]
fn unembedded_facts_and_latest() {
    let store = store_with_facts(3);
    let pending: Vec<i64> = store
        .unembedded_facts("fam", "m", 10)
        .unwrap()
        .iter()
        .map(|f| f.id)
        .collect();
    assert_eq!(pending, [1, 2, 3]);
    store.set_fact_embedding(2, "m", &[0.1]).unwrap();
    let pending: Vec<i64> = store
        .unembedded_facts("fam", "m", 10)
        .unwrap()
        .iter()
        .map(|f| f.id)
        .collect();
    assert_eq!(pending, [1, 3]);
    // superseded excluded
    let store = store_with_facts(2);
    store.supersede("fam", &[1], NewFact::Repoint(2)).unwrap();
    let pending: Vec<i64> = store
        .unembedded_facts("fam", "m", 10)
        .unwrap()
        .iter()
        .map(|f| f.id)
        .collect();
    assert_eq!(pending, [2]);
    assert!(store.unembedded_facts("fam", "m", 0).unwrap().is_empty());
    // latest embedded
    let store = store_with_facts(3);
    assert_eq!(store.latest_embedded_fact_id("fam", "m").unwrap(), 0);
    store.set_fact_embedding(1, "m", &[0.1]).unwrap();
    store.set_fact_embedding(3, "m", &[0.3]).unwrap();
    assert_eq!(store.latest_embedded_fact_id("fam", "m").unwrap(), 3);
}

// --- people dossiers ------------------------------------------------------

#[test]
fn dossier_crud_and_isolation() {
    let store = mem();
    assert!(
        store
            .get_people_dossier("fam", "discord:1")
            .unwrap()
            .is_none()
    );
    store
        .put_people_dossier("fam", "discord:1", 7, "Cass likes pho. Lives in Toronto.")
        .unwrap();
    let entry = store
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    assert_eq!(entry.last_fact_id, 7);
    assert!(entry.dossier_text.contains("pho"));
    store
        .put_people_dossier("fam", "discord:1", 10, "v2")
        .unwrap();
    let entry = store
        .get_people_dossier("fam", "discord:1")
        .unwrap()
        .unwrap();
    assert_eq!(entry.last_fact_id, 10);
    assert_eq!(entry.dossier_text, "v2");
    store
        .put_people_dossier("famB", "discord:1", 1, "B's view")
        .unwrap();
    assert_eq!(
        store
            .get_people_dossier("famB", "discord:1")
            .unwrap()
            .unwrap()
            .dossier_text,
        "B's view"
    );
}

#[test]
fn subjects_with_facts_max_id_and_skips_subjectless() {
    let store = mem();
    assert!(store.subjects_with_facts("fam").unwrap().is_empty());
    store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "A subject-less fact.",
            vec![1],
        ))
        .unwrap();
    assert!(store.subjects_with_facts("fam").unwrap().is_empty());

    let store = mem();
    let cass = subj("discord:1", "Cass");
    let aria = subj("discord:2", "Aria");
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass likes pho.", vec![1])
                .subjects(vec![cass.clone()]),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes ramen.", vec![2])
                .subjects(vec![aria.clone()]),
        )
        .unwrap();
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass and Aria are roommates.", vec![3])
                .subjects(vec![cass, aria]),
        )
        .unwrap();
    let latest = store.subjects_with_facts("fam").unwrap();
    assert_eq!(latest["discord:1"], 3);
    assert_eq!(latest["discord:2"], 3);
}

#[test]
fn facts_for_subject_membership_min_id_superseded() {
    let store = mem();
    let cass = subj("discord:1", "Cass");
    let aria = subj("discord:2", "Aria");
    store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass fact.", vec![1]).subjects(vec![cass.clone()]),
        )
        .unwrap();
    store
        .append_fact(AppendFact::new("fam", Some(1), "Aria fact.", vec![2]).subjects(vec![aria]))
        .unwrap();
    let out = store
        .facts_for_subject("fam", "discord:1", 0, false, None)
        .unwrap();
    assert_eq!(
        out.iter().map(|f| f.text.clone()).collect::<Vec<_>>(),
        ["Cass fact."]
    );

    let store = mem();
    for i in 0..3 {
        store
            .append_fact(
                AppendFact::new("fam", Some(1), format!("Cass fact {i}."), vec![i as i64])
                    .subjects(vec![cass.clone()]),
            )
            .unwrap();
    }
    let out = store
        .facts_for_subject("fam", "discord:1", 1, false, None)
        .unwrap();
    assert_eq!(out.iter().map(|f| f.id).collect::<Vec<_>>(), [2, 3]);

    // superseded excluded by default
    let store = mem();
    let old = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass was a baker.", vec![1])
                .subjects(vec![cass.clone()]),
        )
        .unwrap();
    let new = store
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass is a chef.", vec![2]).subjects(vec![cass]),
        )
        .unwrap();
    store
        .supersede("fam", &[old.id], NewFact::Repoint(new.id))
        .unwrap();
    let out = store
        .facts_for_subject("fam", "discord:1", 0, false, None)
        .unwrap();
    assert_eq!(
        out.iter().map(|f| f.text.clone()).collect::<Vec<_>>(),
        ["Cass is a chef."]
    );
}

// --- reflections ----------------------------------------------------------

#[test]
fn reflection_append_and_null_channel() {
    let store = mem();
    let r = store
        .append_reflection(
            "fam",
            Some(1),
            "circling homesickness.",
            &[1, 2, 3],
            &[5, 7],
            10,
            8,
        )
        .unwrap();
    assert!(r.id > 0);
    assert_eq!(r.cited_turn_ids, [1, 2, 3]);
    assert_eq!(r.cited_fact_ids, [5, 7]);
    assert_eq!(r.last_turn_id, 10);
    assert_eq!(r.last_fact_id, 8);
    assert_eq!(r.channel_id, Some(1));
    let r2 = store
        .append_reflection("fam", None, "cross-channel.", &[1], &[], 1, 0)
        .unwrap();
    assert!(r2.channel_id.is_none());
}

#[test]
fn reflections_newest_first_and_channel_filter() {
    let store = mem();
    for i in 0..3 {
        store
            .append_reflection("fam", Some(1), &format!("reflection {i}"), &[i], &[], i, 0)
            .unwrap();
    }
    let recents = store.recent_reflections("fam", Some(1), 10).unwrap();
    assert_eq!(
        recents.iter().map(|r| r.text.clone()).collect::<Vec<_>>(),
        ["reflection 2", "reflection 1", "reflection 0"]
    );

    let store = mem();
    store
        .append_reflection("fam", Some(1), "channel 1 reflection", &[1], &[], 1, 0)
        .unwrap();
    store
        .append_reflection("fam", None, "global reflection", &[1], &[], 1, 0)
        .unwrap();
    store
        .append_reflection("fam", Some(2), "channel 2 reflection", &[1], &[], 1, 0)
        .unwrap();
    let ch1: std::collections::HashSet<String> = store
        .recent_reflections("fam", Some(1), 10)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    assert!(ch1.contains("channel 1 reflection"));
    assert!(ch1.contains("global reflection"));
    assert!(!ch1.contains("channel 2 reflection"));
    assert_eq!(store.recent_reflections("fam", None, 10).unwrap().len(), 3);
    assert!(
        store
            .recent_reflections("fam", Some(1), 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn reflection_watermarks_fallback_chain() {
    let store = mem();
    assert_eq!(store.latest_reflection_watermarks("fam").unwrap(), (0, 0));
    store
        .append_reflection("fam", Some(1), "older", &[1], &[], 10, 3)
        .unwrap();
    store
        .append_reflection("fam", Some(1), "newer", &[1], &[], 42, 11)
        .unwrap();
    assert_eq!(store.latest_reflection_watermarks("fam").unwrap(), (42, 11));
    store.set_reflection_watermark("fam", 100, 50).unwrap();
    assert_eq!(
        store.latest_reflection_watermarks("fam").unwrap(),
        (100, 50)
    );
}

#[test]
fn superseded_fact_ids_subset() {
    let store = mem();
    assert!(store.superseded_fact_ids("fam", &[]).unwrap().is_empty());
    let ids: Vec<i64> = (0..3)
        .map(|i| {
            store
                .append_fact(AppendFact::new(
                    "fam",
                    Some(1),
                    format!("fact {i}"),
                    vec![i + 1],
                ))
                .unwrap()
                .id
        })
        .collect();
    let replacement = store
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "replacement fact",
            vec![ids[1]],
        ))
        .unwrap();
    store
        .supersede("fam", &[ids[1]], NewFact::Repoint(replacement.id))
        .unwrap();
    let result = store.superseded_fact_ids("fam", &ids).unwrap();
    assert_eq!(result, [ids[1]].into_iter().collect());
    // other familiars ignored
    let store = mem();
    let f1 = store
        .append_fact(AppendFact::new("other", Some(1), "other", vec![1]))
        .unwrap();
    let f2 = store
        .append_fact(AppendFact::new("other", Some(1), "replacement", vec![1]))
        .unwrap();
    store
        .supersede("other", &[f1.id], NewFact::Repoint(f2.id))
        .unwrap();
    assert!(
        store
            .superseded_fact_ids("fam", &[f1.id, f2.id])
            .unwrap()
            .is_empty()
    );
}
