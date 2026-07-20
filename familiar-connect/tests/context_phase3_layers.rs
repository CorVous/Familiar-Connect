//! Ported from Python `tests/test_phase3_layers.py` — ConversationSummaryLayer,
//! RagContextLayer, and tool-turn narration.

#[path = "context_helpers/mod.rs"]
mod helpers;

use chrono::{TimeZone, Utc};
use familiar_connect::context::{
    ConversationSummaryLayer, Layer, RagContextLayer, RecentHistoryLayer,
};
use familiar_connect::history::FOCUS_STREAM_CHANNEL_ID;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::{AppendFact, AppendTurn, FactSubject};
use familiar_connect::identity::Author;
use familiar_connect::llm::Message;

use helpers::{author, set_ts, store, vctx};

fn put_focus_summary(
    store: &AsyncHistoryStore,
    summary_text: &str,
    last_summarised_id: i64,
    last_consumed_at: &str,
) {
    store
        .sync()
        .put_summary(
            "fam",
            last_summarised_id,
            summary_text,
            FOCUS_STREAM_CHANNEL_ID,
            Some(last_consumed_at),
        )
        .unwrap();
}

// ---------------------------------------------------------------------------
// ConversationSummaryLayer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn summary_returns_text() {
    let store = store();
    put_focus_summary(
        &store,
        "Earlier they talked about foxes.",
        5,
        "2026-06-13T10:00:00+00:00",
    );
    let layer = ConversationSummaryLayer::new(store);
    assert!(layer.build(&vctx(1)).await.contains("foxes"));
}

#[tokio::test]
async fn summary_empty_when_none() {
    let store = store();
    let layer = ConversationSummaryLayer::new(store);
    assert!(layer.build(&vctx(1)).await.is_empty());
}

#[tokio::test]
async fn summary_ignores_ctx_channel_id() {
    let store = store();
    put_focus_summary(
        &store,
        "the one true thread",
        5,
        "2026-06-13T10:00:00+00:00",
    );
    let layer = ConversationSummaryLayer::new(store);
    assert!(layer.build(&vctx(1)).await.contains("thread"));
    assert!(layer.build(&vctx(999)).await.contains("thread"));
}

#[tokio::test]
async fn summary_same_in_text_and_voice() {
    let store = store();
    put_focus_summary(
        &store,
        "modality-agnostic thread",
        5,
        "2026-06-13T10:00:00+00:00",
    );
    let layer = ConversationSummaryLayer::new(store);
    let text_out = layer
        .build(
            &familiar_connect::context::AssemblyContext::new("fam", Some(1))
                .with_viewer_mode("text"),
        )
        .await;
    let voice_out = layer.build(&vctx(1)).await;
    assert_eq!(text_out, voice_out);
    assert!(text_out.contains("modality-agnostic"));
}

#[tokio::test]
async fn summary_key_tracks_composite_watermark() {
    let store = store();
    let layer = ConversationSummaryLayer::new(store.clone());
    assert_eq!(layer.invalidation_key(&vctx(1)).await, "none");
    put_focus_summary(&store, "v1", 5, "2026-06-13T10:00:00+00:00");
    let k1 = layer.invalidation_key(&vctx(1)).await;
    put_focus_summary(&store, "v2", 10, "2026-06-13T11:00:00+00:00");
    let k2 = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(k1, k2);
    assert_ne!(k1, "none");
}

// ---------------------------------------------------------------------------
// RagContextLayer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rag_empty_when_no_cues() {
    let store = store();
    let layer = RagContextLayer::builder(store).max_results(5).build();
    assert!(layer.build(&vctx(1)).await.is_empty());
}

#[tokio::test]
async fn rag_returns_matches_for_cue() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "user",
            "Let's discuss the fox plan tomorrow at noon.",
        ))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "assistant", "Sure thing."))
        .unwrap();
    let layer = RagContextLayer::builder(store).max_results(5).build();
    layer.set_current_cue("fox");
    assert!(layer.build(&vctx(1)).await.contains("fox plan"));
}

#[tokio::test]
async fn rag_scoped_to_familiar() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new(
            "other",
            1,
            "user",
            "Fox on a different familiar.",
        ))
        .unwrap();
    let layer = RagContextLayer::builder(store).max_results(5).build();
    layer.set_current_cue("fox");
    assert!(layer.build(&vctx(1)).await.is_empty());
}

#[tokio::test]
async fn rag_key_reflects_cue_and_watermark() {
    let store = store();
    let layer = RagContextLayer::builder(store.clone())
        .max_results(5)
        .build();
    layer.set_current_cue("fox");
    let k1 = layer.invalidation_key(&vctx(1)).await;
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "new turn"))
        .unwrap();
    let k2 = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(k1, k2);
    layer.set_current_cue("otter");
    let k3 = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(k3, k1);
    assert_ne!(k3, k2);
}

#[tokio::test]
async fn rag_excludes_turns_within_recent_window() {
    let store = store();
    for i in 0..30 {
        store
            .sync()
            .append_turn(AppendTurn::new(
                "fam",
                1,
                "user",
                format!("strawberry observation number {i}"),
            ))
            .unwrap();
    }
    let layer = RagContextLayer::builder(store)
        .max_results(10)
        .recent_window_size(20)
        .build();
    layer.set_current_cue("strawberry");
    let out = layer.build(&vctx(1)).await;
    for older in 0..10 {
        assert!(out.contains(&format!("number {older}")), "{older}: {out}");
    }
    for recent in 10..30 {
        assert!(
            !out.contains(&format!("number {recent}")),
            "{recent}: {out}"
        );
    }
}

#[tokio::test]
async fn rag_fact_with_renamed_subject_gets_annotation() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hello from cass").author(Author::new(
                "discord",
                "111",
                Some("cass_login".into()),
                Some("Cass".into()),
            )),
        )
        .unwrap();
    store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass likes pho.", vec![1]).subjects(vec![
                FactSubject {
                    canonical_key: "discord:111".into(),
                    display_at_write: "Cass".into(),
                },
            ]),
        )
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "still here under a new name").author(Author::new(
                "discord",
                "111",
                Some("cass_login".into()),
                Some("peeks".into()),
            )),
        )
        .unwrap();
    let layer = RagContextLayer::builder(store).max_facts(3).build();
    layer.set_current_cue("pho");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("Cass likes pho."));
    assert!(out.contains("peeks"));
    assert!(out.contains("Cass"));
}

#[tokio::test]
async fn rag_fact_without_rename_no_annotation() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hello from cass").author(Author::new(
                "discord",
                "111",
                Some("cass_login".into()),
                Some("Cass".into()),
            )),
        )
        .unwrap();
    store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Cass likes pho.", vec![1]).subjects(vec![
                FactSubject {
                    canonical_key: "discord:111".into(),
                    display_at_write: "Cass".into(),
                },
            ]),
        )
        .unwrap();
    let layer = RagContextLayer::builder(store).max_facts(3).build();
    layer.set_current_cue("pho");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("Cass likes pho."));
    assert!(!out.to_lowercase().contains("now known as"));
    assert!(!out.to_lowercase().contains("formerly"));
}

#[tokio::test]
async fn rag_legacy_fact_without_subjects_unchanged() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "An old fact about pho.",
            vec![],
        ))
        .unwrap();
    let layer = RagContextLayer::builder(store).max_facts(3).build();
    layer.set_current_cue("pho");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("An old fact about pho."));
    assert!(!out.to_lowercase().contains("now known as"));
}

#[tokio::test]
async fn rag_renders_date_header_and_12h_clock() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "my brain's dying").author(Author::new(
                "discord",
                "111",
                Some("peebo".into()),
                Some("Peebo".into()),
            )),
        )
        .unwrap();
    set_ts(
        &store,
        1,
        Utc.with_ymd_and_hms(2026, 5, 3, 14, 29, 0).unwrap(),
    );
    let layer = RagContextLayer::builder(store)
        .max_results(5)
        .context_window(0)
        .build();
    layer.set_current_cue("brain");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("## Possibly relevant earlier turns"));
    assert!(out.contains("2026-05-03:"));
    assert!(out.contains("> [2:29PM Peebo]: my brain's dying"));
}

#[tokio::test]
async fn rag_renders_in_display_tz() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "my brain's dying").author(Author::new(
                "discord",
                "111",
                Some("peebo".into()),
                Some("Peebo".into()),
            )),
        )
        .unwrap();
    set_ts(
        &store,
        1,
        Utc.with_ymd_and_hms(2026, 5, 3, 2, 29, 0).unwrap(),
    );
    let layer = RagContextLayer::builder(store)
        .max_results(5)
        .context_window(0)
        .display_tz("America/Los_Angeles")
        .build();
    layer.set_current_cue("brain");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("2026-05-02:"));
    assert!(out.contains("> [7:29PM Peebo]: my brain's dying"));
}

#[tokio::test]
async fn rag_multiline_prefixes_every_line() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new(
                "fam",
                1,
                "assistant",
                "*small surprised blink*\n\nzero?\n\n*looks at cassidy*",
            )
            .author(Author::new(
                "discord",
                "222",
                Some("assistant".into()),
                Some("assistant".into()),
            )),
        )
        .unwrap();
    set_ts(
        &store,
        1,
        Utc.with_ymd_and_hms(2026, 5, 2, 3, 2, 0).unwrap(),
    );
    let layer = RagContextLayer::builder(store)
        .max_results(5)
        .context_window(0)
        .build();
    layer.set_current_cue("blink");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("> [3:02AM assistant]: *small surprised blink*"));
    assert!(out.contains("> zero?"));
    assert!(out.contains("> *looks at cassidy*"));
    assert!(!out.contains("\nzero?"));
    assert!(!out.contains("\n*looks at cassidy*"));
}

#[tokio::test]
async fn rag_includes_neighbour_context() {
    let store = store();
    for content in [
        "warmup chatter A",
        "marker turn that mentions strawberry",
        "follow up reply B",
        "unrelated middle",
        "later mention of strawberry too",
    ] {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", content))
            .unwrap();
    }
    let layer = RagContextLayer::builder(store)
        .max_results(5)
        .context_window(1)
        .build();
    layer.set_current_cue("strawberry");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("warmup chatter A"));
    assert!(out.contains("marker turn that mentions strawberry"));
    assert!(out.contains("follow up reply B"));
    assert!(out.contains("later mention of strawberry too"));
}

#[tokio::test]
async fn rag_zero_recent_window_keeps_old_behavior() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "strawberry observation"))
        .unwrap();
    let layer = RagContextLayer::builder(store).max_results(5).build();
    layer.set_current_cue("strawberry");
    assert!(
        layer
            .build(&vctx(1))
            .await
            .contains("strawberry observation")
    );
}

// ---------------------------------------------------------------------------
// Tool turns
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_turn_rendered_as_text_not_tool_role() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "look at this").author(author("111", "Cor")))
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "assistant", "").tool_calls_json(
                r#"[{"id": "call_1", "type": "function", "function": {"name": "view_image", "arguments": "{\"image_id\":\"img_0\"}"}}]"#,
            ),
        )
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "tool", "# Image Description\n\nA red fox spirit.")
                .tool_call_id("call_1"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(msgs.iter().all(|m| m.role != "tool"));
    assert!(
        msgs.iter()
            .any(|m| m.content_str().contains("A red fox spirit."))
    );
    assert!(msgs.iter().all(|m| m.tool_call_id.is_none()));
}

#[tokio::test]
async fn tool_result_replays_as_user_not_assistant() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "tool", "# Image Description\n\nA red fox spirit.")
                .tool_call_id("call_1"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let leaked: Vec<&Message> = msgs
        .iter()
        .filter(|m| m.content_str().contains("A red fox spirit."))
        .collect();
    assert!(!leaked.is_empty());
    assert!(leaked.iter().all(|m| m.role == "user"));
}
