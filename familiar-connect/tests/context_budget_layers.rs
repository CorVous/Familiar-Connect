//! Ported from Python `tests/test_context_budget_layers.py` — per-layer
//! `max_tokens` self-truncation.

#[path = "context_helpers/mod.rs"]
mod helpers;

use familiar_connect::budget::estimate_tokens;
use familiar_connect::context::{
    ConversationSummaryLayer, Layer, PeopleDossierLayer, RagContextLayer, RecentHistoryLayer,
};
use familiar_connect::history::FOCUS_STREAM_CHANNEL_ID;
use familiar_connect::history::store::AppendTurn;

use helpers::{author, store, vctx};

#[tokio::test]
async fn recent_history_drops_oldest_to_fit_cap() {
    let store = store();
    for i in 0..10 {
        store
            .sync()
            .append_turn(AppendTurn::new(
                "fam",
                1,
                "user",
                format!("message number {i:02} blah"),
            ))
            .unwrap();
    }
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .max_tokens(Some(40))
        .build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(msgs.last().unwrap().content_str().ends_with("09 blah"));
    let total: i64 = msgs.iter().map(|m| estimate_tokens(&m.content_str())).sum();
    assert!(total <= 60);
}

#[tokio::test]
async fn recent_history_no_cap_means_full_window() {
    let store = store();
    for i in 0..5 {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("m{i}")))
            .unwrap();
    }
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    assert_eq!(layer.recent_messages(&vctx(1)).await.len(), 5);
}

#[tokio::test]
async fn rag_caps_total_section_tokens() {
    let store = store();
    for i in 0..20 {
        store
            .sync()
            .append_turn(AppendTurn::new(
                "fam",
                1,
                "user",
                format!("discussion of widgets and gadgets number {i}"),
            ))
            .unwrap();
    }
    let layer = RagContextLayer::builder(store)
        .max_results(10)
        .max_facts(0)
        .max_tokens(Some(20))
        .build();
    layer.set_current_cue("widgets");
    let out = layer.build(&vctx(1)).await;
    assert!(estimate_tokens(&out) <= 40);
}

#[tokio::test]
async fn dossier_drops_trailing_past_cap() {
    let store = store();
    for i in 0..5 {
        let uid = format!("u{i}");
        let a = author(&uid, &format!("User{i}"));
        store.sync().upsert_account(&a).unwrap();
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("hello from user{i}")).author(a))
            .unwrap();
        store
            .sync()
            .put_people_dossier("fam", &format!("discord:{uid}"), i + 1, &"X".repeat(400))
            .unwrap();
    }
    let layer = PeopleDossierLayer::builder(store)
        .window_size(20)
        .max_people(10)
        .max_tokens(Some(200))
        .build();
    let out = layer.build(&vctx(1)).await;
    assert!(estimate_tokens(&out) <= 400);
}

#[tokio::test]
async fn summary_truncates_long_summary() {
    let store = store();
    let long_text = "summary content blah ".repeat(200);
    store
        .sync()
        .put_summary("fam", 1, &long_text, FOCUS_STREAM_CHANNEL_ID, None)
        .unwrap();
    let layer = ConversationSummaryLayer::new(store).with_max_tokens(50);
    let out = layer.build(&vctx(1)).await;
    assert!(estimate_tokens(&out) <= 80);
}
