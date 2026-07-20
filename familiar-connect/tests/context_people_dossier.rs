//! Ported from Python `tests/test_people_dossier_layer.py`.

#[path = "context_helpers/mod.rs"]
mod helpers;

use familiar_connect::context::{Layer, PeopleDossierLayer};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::AppendTurn;
use familiar_connect::identity::Author;

use helpers::{author, store, tctx};

fn seed_person(
    store: &AsyncHistoryStore,
    uid: &str,
    display: &str,
    dossier: &str,
    last_fact_id: i64,
) {
    let a = author(uid, display);
    store.sync().upsert_account(&a).unwrap();
    store
        .sync()
        .put_people_dossier("fam", &format!("discord:{uid}"), last_fact_id, dossier)
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(a))
        .unwrap();
}

#[tokio::test]
async fn empty_when_no_turns() {
    let store = store();
    let layer = PeopleDossierLayer::builder(store).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn empty_when_dossier_missing() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn renders_dossier_for_recent_speaker() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store.sync().upsert_account(&author("1", "Cass")).unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 5, "Cass enjoys pho.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Cass"));
    assert!(out.contains("pho"));
}

#[tokio::test]
async fn renders_profile_metadata_when_available() {
    let store = store();
    let mut a = Author::new(
        "discord",
        "1",
        Some("cass_login".into()),
        Some("Cass".into()),
    );
    a.global_name = Some("Cass".into());
    a.pronouns = Some("she/her".into());
    a.bio = Some("Lover of pho.".into());
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(a.clone()))
        .unwrap();
    store.sync().upsert_account(&a).unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 5, "Cass enjoys pho.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("### Cass"));
    assert!(out.contains("@cass_login"));
    assert!(out.contains("she/her"));
    assert!(out.contains("Lover of pho."));
    assert!(out.contains("Cass enjoys pho."));
}

#[tokio::test]
async fn renders_without_optional_profile_fields() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store.sync().upsert_account(&author("1", "Cass")).unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 5, "Cass enjoys pho.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Cass enjoys pho."));
    assert!(!out.contains("Pronouns:"));
    assert!(!out.contains("Bio:"));
}

#[tokio::test]
async fn renders_dossier_for_mentioned_user() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hey, what about <@2>?").author(author("1", "Cass")),
        )
        .unwrap();
    store.sync().record_mentions(1, &["discord:2"]).unwrap();
    store.sync().upsert_account(&author("2", "Aria")).unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:2", 3, "Aria runs a bakery.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Aria"));
    assert!(out.contains("bakery"));
}

#[tokio::test]
async fn prioritises_most_recent_when_capped() {
    let store = store();
    for (uid, display) in [("1", "Cass"), ("2", "Aria"), ("3", "Bo")] {
        seed_person(&store, uid, display, &format!("{display} dossier."), 1);
    }
    let layer = PeopleDossierLayer::builder(store).max_people(2).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Bo dossier"));
    assert!(out.contains("Aria dossier"));
    assert!(!out.contains("Cass dossier"));
}

#[tokio::test]
async fn dedups_repeated_subjects_keeping_most_recent() {
    let store = store();
    for (uid, display) in [("1", "Cass"), ("2", "Aria"), ("1", "Cass")] {
        seed_person(&store, uid, display, &format!("{display} dossier."), 1);
    }
    let layer = PeopleDossierLayer::builder(store).max_people(2).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Cass dossier"));
    assert!(out.contains("Aria dossier"));
    assert!(out.find("Cass dossier").unwrap() < out.find("Aria dossier").unwrap());
}

#[tokio::test]
async fn scoped_to_channel() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 99, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 1, "Cass dossier.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn self_dossier_always_injected() {
    let store = store();
    store
        .sync()
        .put_people_dossier(
            "fam",
            "ego:fam",
            7,
            "Sapphire favours sharp, provocative bits.",
        )
        .unwrap();
    let layer = PeopleDossierLayer::builder(store)
        .familiar_display_name("Sapphire")
        .build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Sapphire favours sharp, provocative bits."));
    assert!(!out.contains("ego:fam"));
}

#[tokio::test]
async fn self_dossier_injected_alongside_people() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store.sync().upsert_account(&author("1", "Cass")).unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 5, "Cass enjoys pho.")
        .unwrap();
    store
        .sync()
        .put_people_dossier("fam", "ego:fam", 7, "Sapphire favours sharp bits.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store)
        .familiar_display_name("Sapphire")
        .build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("Cass enjoys pho."));
    assert!(out.contains("Sapphire favours sharp bits."));
}

#[tokio::test]
async fn invalidation_key_changes_when_new_turn() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 1, "Cass dossier.")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store.clone()).build();
    let k1 = layer.invalidation_key(&tctx(1)).await;
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "more").author(author("1", "Cass")))
        .unwrap();
    let k2 = layer.invalidation_key(&tctx(1)).await;
    assert_ne!(k1, k2);
}

#[tokio::test]
async fn invalidation_key_changes_when_dossier_watermark_moves() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author("1", "Cass")))
        .unwrap();
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 1, "v1")
        .unwrap();
    let layer = PeopleDossierLayer::builder(store.clone()).build();
    let k1 = layer.invalidation_key(&tctx(1)).await;
    store
        .sync()
        .put_people_dossier("fam", "discord:1", 10, "v2")
        .unwrap();
    let k2 = layer.invalidation_key(&tctx(1)).await;
    assert_ne!(k1, k2);
}
