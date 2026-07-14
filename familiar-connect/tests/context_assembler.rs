//! Ported from Python `tests/test_context_assembler.py` — assembler compose +
//! cache, and RecentHistoryLayer rendering (prefix, coalesce, silence fold,
//! guild-nick labels, reply markers, mention rewriting).

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use chrono::{Duration, TimeZone, Utc};
use familiar_connect::context::{
    Assembler, AssemblyContext, CharacterCardLayer, Layer, OperatingModeLayer, RecentHistoryLayer,
};
use familiar_connect::history::store::AppendTurn;
use familiar_connect::identity::Author;
use familiar_connect::llm::Message;
use regex::Regex;

use helpers::{author, set_ts, store, vctx, vctx_guild};

fn modes(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

fn contents(msgs: &[Message]) -> Vec<String> {
    msgs.iter().map(Message::content_str).collect()
}

fn user_count(msgs: &[Message]) -> usize {
    msgs.iter().filter(|m| m.role == "user").count()
}

// ---------------------------------------------------------------------------
// CharacterCardLayer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn card_reads_from_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let card = dir.path().join("character.md");
    std::fs::write(&card, "## Persona\n\nA playful familiar named Aria.\n").unwrap();
    let layer = CharacterCardLayer::new(card);
    assert!(layer.build(&vctx(1)).await.contains("Aria"));
}

#[tokio::test]
async fn card_empty_when_no_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let layer = CharacterCardLayer::new(dir.path().join("no-card.md"));
    assert!(layer.build(&vctx(1)).await.is_empty());
}

#[tokio::test]
async fn card_invalidation_key_changes_on_edit() {
    let dir = tempfile::tempdir().unwrap();
    let card = dir.path().join("character.md");
    std::fs::write(&card, "v1").unwrap();
    let layer = CharacterCardLayer::new(&card);
    let key_1 = layer.invalidation_key(&vctx(1)).await;
    std::fs::write(&card, "v2").unwrap();
    let key_2 = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(key_1, key_2);
}

// ---------------------------------------------------------------------------
// OperatingModeLayer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn operating_mode_voice() {
    let layer = OperatingModeLayer::new(modes(&[
        ("voice", "Speak in short sentences."),
        ("text", "You may use markdown."),
    ]));
    let out = layer
        .build(&AssemblyContext::new("fam", Some(1)).with_viewer_mode("voice"))
        .await;
    assert!(out.contains("short sentences"));
}

#[tokio::test]
async fn operating_mode_text() {
    let layer = OperatingModeLayer::new(modes(&[
        ("voice", "Speak in short sentences."),
        ("text", "You may use markdown."),
    ]));
    let out = layer
        .build(&AssemblyContext::new("fam", Some(1)).with_viewer_mode("text"))
        .await;
    assert!(out.contains("markdown"));
}

#[tokio::test]
async fn operating_mode_unknown_returns_empty() {
    let layer = OperatingModeLayer::new(modes(&[("voice", "x")]));
    let out = layer
        .build(&AssemblyContext::new("fam", Some(1)).with_viewer_mode("weird"))
        .await;
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// RecentHistoryLayer basics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recent_pulls_turns_from_store() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "user", "hello").author(alice))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "assistant", "hi back"))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let all = contents(&msgs);
    assert!(all.iter().any(|c| c.contains("hello")));
    assert!(all.iter().any(|c| c.contains("hi back")));
}

#[tokio::test]
async fn recent_user_gets_author_name_and_prefix() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "user", "hey").author(author("42", "Alice")))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    assert!(user.name.is_some());
    assert!(user.content_str().contains("Alice #"));
    assert!(user.content_str().contains("hey"));
}

#[tokio::test]
async fn recent_user_includes_utc_timestamp_prefix() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "user", "hey").author(author("42", "Alice")))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    let re = Regex::new(r"^\[\d{2}:\d{2} Alice #10\] hey$").unwrap();
    assert!(re.is_match(&user.content_str()), "{}", user.content_str());
}

#[tokio::test]
async fn recent_timestamp_prefix_uses_display_tz() {
    let store = store();
    let row = store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "user", "hey").author(author("42", "Alice")))
        .unwrap();
    set_ts(
        &store,
        row.id,
        Utc.with_ymd_and_hms(2026, 5, 4, 21, 30, 0).unwrap(),
    );
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .display_tz("America/Los_Angeles")
        .build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    assert!(user.content_str().starts_with("[14:30 Alice #10] hey"));
}

#[tokio::test]
async fn recent_respects_window_size() {
    let store = store();
    for i in 0..50 {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("m{i}")))
            .unwrap();
    }
    let layer = RecentHistoryLayer::builder(store).window_size(5).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(msgs.len(), 5);
}

#[tokio::test]
async fn recent_build_returns_empty_system_contribution() {
    let store = store();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    assert!(layer.build(&vctx(1)).await.is_empty());
}

// ---------------------------------------------------------------------------
// Coalescing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn coalesce_same_speaker_voice_fragments_merge() {
    let store = store();
    let cass = author("111", "Cassidy");
    let base = Utc.with_ymd_and_hms(2026, 5, 5, 1, 35, 0).unwrap();
    let fragments = [
        "I okay.",
        "So, Tam, here's the etiquette.",
        "Big Discord calls like this. Right?",
        "You can jump in",
    ];
    for (i, text) in fragments.iter().enumerate() {
        let row = store
            .sync()
            .append_turn(AppendTurn::new("fam", 10, "user", *text).author(cass.clone()))
            .unwrap();
        set_ts(
            &store,
            row.id,
            base + Duration::seconds(2 * i64::try_from(i).unwrap()),
        );
    }
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let user: Vec<&Message> = msgs.iter().filter(|m| m.role == "user").collect();
    assert_eq!(user.len(), 1);
    assert_eq!(
        user[0].content_str(),
        "[01:35 Cassidy #10] I okay. So, Tam, here's the etiquette. \
         Big Discord calls like this. Right? You can jump in"
    );
}

#[tokio::test]
async fn coalesce_different_speakers_do_not_merge() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hey").author(author("1", "Cass")))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "yo").author(author("2", "Tam")))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(user_count(&msgs), 2);
}

#[tokio::test]
async fn coalesce_assistant_turn_breaks_run() {
    let store = store();
    let cass = author("1", "Cass");
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "part one").author(cass.clone()))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "assistant", "ack"))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "part two").author(cass))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let roles: Vec<&str> = msgs.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "user"]);
}

#[tokio::test]
async fn coalesce_text_turns_with_message_id_do_not_merge() {
    let store = store();
    let cass = author("1", "Cass");
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "first text msg")
                .author(cass.clone())
                .platform_message_id("100"),
        )
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "second text msg")
                .author(cass)
                .platform_message_id("101"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(user_count(&msgs), 2);
}

#[tokio::test]
async fn coalesce_gap_exceeding_max_breaks_run() {
    let store = store();
    let cass = author("1", "Cass");
    let base = Utc.with_ymd_and_hms(2026, 5, 5, 1, 0, 0).unwrap();
    let r1 = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "long ago").author(cass.clone()))
        .unwrap();
    set_ts(&store, r1.id, base);
    let r2 = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "much later").author(cass))
        .unwrap();
    set_ts(&store, r2.id, base + Duration::seconds(60));
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(user_count(&msgs), 2);
}

#[tokio::test]
async fn coalesce_max_gap_knob_overrides_default() {
    let store = store();
    let cass = author("1", "Cass");
    let base = Utc.with_ymd_and_hms(2026, 5, 5, 1, 0, 0).unwrap();
    let r1 = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "a").author(cass.clone()))
        .unwrap();
    set_ts(&store, r1.id, base);
    let r2 = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "b").author(cass))
        .unwrap();
    set_ts(&store, r2.id, base + Duration::seconds(10));

    let strict = RecentHistoryLayer::builder(store.clone())
        .window_size(20)
        .coalesce_max_gap_seconds(5.0)
        .build();
    assert_eq!(user_count(&strict.recent_messages(&vctx(1)).await), 2);

    let disabled = RecentHistoryLayer::builder(store)
        .window_size(20)
        .coalesce_max_gap_seconds(0.0)
        .build();
    assert_eq!(user_count(&disabled.recent_messages(&vctx(1)).await), 2);
}

// ---------------------------------------------------------------------------
// Silence gap fold
// ---------------------------------------------------------------------------

fn store_with_gaps(gaps: &[i64]) -> Arc<familiar_connect::history::async_store::AsyncHistoryStore> {
    let store = store();
    let mut t = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
    let roles = ["user", "assistant"];
    let row = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, roles[0], "turn 0"))
        .unwrap();
    set_ts(&store, row.id, t);
    for (i, gap) in gaps.iter().enumerate() {
        t += Duration::seconds(*gap);
        let row = store
            .sync()
            .append_turn(AppendTurn::new(
                "fam",
                1,
                roles[(i + 1) % 2],
                format!("turn {}", i + 1),
            ))
            .unwrap();
        set_ts(&store, row.id, t);
    }
    store
}

#[tokio::test]
async fn silence_no_gap_keeps_all() {
    let store = store_with_gaps(&[60, 60, 60]);
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .silence_gap_fold_seconds(300.0)
        .build();
    assert_eq!(layer.recent_messages(&vctx(1)).await.len(), 4);
}

#[tokio::test]
async fn silence_gap_folds_turns_before_it() {
    let store = store_with_gaps(&[60, 600, 60]);
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .silence_gap_fold_seconds(300.0)
        .build();
    let all = contents(&layer.recent_messages(&vctx(1)).await);
    assert!(all.iter().any(|c| c.contains("turn 2")));
    assert!(all.iter().any(|c| c.contains("turn 3")));
    assert!(!all.iter().any(|c| c.contains("turn 0")));
    assert!(!all.iter().any(|c| c.contains("turn 1")));
}

#[tokio::test]
async fn silence_uses_last_qualifying_gap() {
    let store = store_with_gaps(&[600, 600, 60]);
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .silence_gap_fold_seconds(300.0)
        .build();
    let all = contents(&layer.recent_messages(&vctx(1)).await);
    assert!(all.iter().any(|c| c.contains("turn 2")));
    assert!(all.iter().any(|c| c.contains("turn 3")));
    assert!(!all.iter().any(|c| c.contains("turn 0")));
    assert!(!all.iter().any(|c| c.contains("turn 1")));
}

#[tokio::test]
async fn silence_zero_threshold_keeps_all() {
    let store = store_with_gaps(&[3600, 3600]);
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .silence_gap_fold_seconds(0.0)
        .build();
    assert_eq!(layer.recent_messages(&vctx(1)).await.len(), 3);
}

#[tokio::test]
async fn silence_gap_at_last_position_keeps_only_newest() {
    let store = store_with_gaps(&[60, 60, 3600]);
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .silence_gap_fold_seconds(300.0)
        .build();
    let all = contents(&layer.recent_messages(&vctx(1)).await);
    assert!(all.iter().any(|c| c.contains("turn 3")));
    assert!(!all.iter().any(|c| c.contains("turn 0")));
    assert!(!all.iter().any(|c| c.contains("turn 1")));
    assert!(!all.iter().any(|c| c.contains("turn 2")));
}

// ---------------------------------------------------------------------------
// Guild-aware names
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recent_uses_guild_nick_from_accounts() {
    let store = store();
    let mut cass = Author::new(
        "discord",
        "111",
        Some("cass_login".into()),
        Some("Cass".into()),
    );
    cass.global_name = Some("Cassidy".into());
    store.sync().upsert_account(&cass).unwrap();
    store
        .sync()
        .upsert_guild_nick("discord:111", 42, Some("Aria"))
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi")
                .author(cass)
                .guild_id(42),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx_guild(1, 42)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    assert!(user.content_str().contains("Aria"));
    assert!(!user.content_str().contains("Cass"));
}

// ---------------------------------------------------------------------------
// Reply markers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reply_marker_only_when_parent_in_window() {
    let store = store();
    let mut bob = Author::new("discord", "2", Some("bob".into()), Some("Bob".into()));
    bob.global_name = Some("Bob".into());
    store.sync().upsert_account(&bob).unwrap();
    let mut alice = Author::new("discord", "1", Some("alice".into()), Some("Alice".into()));
    alice.global_name = Some("Alice".into());
    store.sync().upsert_account(&alice).unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "parent message")
                .author(bob)
                .platform_message_id("msg-1"),
        )
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "child message")
                .author(alice)
                .platform_message_id("msg-2")
                .reply_to_message_id("msg-1"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let child = msgs
        .iter()
        .find(|m| m.content_str().contains("child message"))
        .unwrap();
    assert!(child.content_str().contains('\u{21a9}'));
    assert!(child.content_str().contains("Bob"));
    assert!(child.content_str().contains("parent message"));
}

#[tokio::test]
async fn reply_inlines_full_parent_when_outside_window() {
    let store = store();
    let mut bob = Author::new("discord", "2", Some("bob".into()), Some("Bob".into()));
    bob.global_name = Some("Bob".into());
    store.sync().upsert_account(&bob).unwrap();
    let mut alice = Author::new("discord", "1", Some("alice".into()), Some("Alice".into()));
    alice.global_name = Some("Alice".into());
    store.sync().upsert_account(&alice).unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "parent text far away")
                .author(bob)
                .platform_message_id("msg-1"),
        )
        .unwrap();
    for i in 0..5 {
        store
            .sync()
            .append_turn(
                AppendTurn::new("fam", 1, "user", format!("filler {i}")).author(alice.clone()),
            )
            .unwrap();
    }
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "child reply")
                .author(alice)
                .platform_message_id("msg-child")
                .reply_to_message_id("msg-1"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(2).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let child = msgs
        .iter()
        .find(|m| m.content_str().contains("child reply"))
        .unwrap();
    assert!(child.content_str().contains('\u{21a9}'));
    assert!(child.content_str().contains("parent text far away"));
}

#[tokio::test]
async fn reply_drops_marker_when_parent_unknown() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "orphan reply")
                .author(author("1", "Alice"))
                .platform_message_id("msg-x")
                .reply_to_message_id("msg-ghost"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let m = msgs
        .iter()
        .find(|m| m.content_str().contains("orphan reply"))
        .unwrap();
    assert!(!m.content_str().contains('\u{21a9}'));
}

// ---------------------------------------------------------------------------
// Mention rewriting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mention_rewrites_to_display_name() {
    let store = store();
    let mut bob = Author::new("discord", "222", Some("bob".into()), Some("Bob".into()));
    bob.global_name = Some("Bob".into());
    store.sync().upsert_account(&bob).unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hey <@222>, look at this")
                .author(author("111", "Alice")),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let user = msgs
        .iter()
        .find(|m| m.content_str().contains("look at this"))
        .unwrap();
    assert!(!user.content_str().contains("<@222>"));
    assert!(user.content_str().contains("[@Bob]"));
}

#[tokio::test]
async fn mention_unknown_id_falls_back_to_id() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hey <@999>").author(author("111", "Alice")))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let user = msgs
        .iter()
        .find(|m| m.content_str().contains("hey"))
        .unwrap();
    assert!(user.content_str().contains("[@999]"));
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

#[tokio::test]
async fn assembler_composes_non_empty_layers_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let card = dir.path().join("card.md");
    std::fs::write(&card, "CARD\n").unwrap();

    let asm = Assembler::builder()
        .layer(Arc::new(CharacterCardLayer::new(&card)))
        .layer(Arc::new(OperatingModeLayer::new(modes(&[(
            "voice", "BE_TERSE",
        )]))))
        .build();
    let prompt = asm
        .assemble(&AssemblyContext::new("fam", Some(1)).with_viewer_mode("voice"))
        .await;
    assert!(prompt.system_prompt.contains("CARD"));
    assert!(prompt.system_prompt.contains("BE_TERSE"));

    let asm2 = Assembler::builder()
        .layer(Arc::new(CharacterCardLayer::new(
            dir.path().join("missing.md"),
        )))
        .layer(Arc::new(CharacterCardLayer::new(&card)))
        .build();
    let prompt2 = asm2.assemble(&vctx(1)).await;
    assert!(prompt2.system_prompt.contains("CARD"));
    assert!(prompt2.system_prompt.trim().starts_with("CARD"));
}

#[tokio::test]
async fn assembler_recent_history_contributes_messages() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "hey"))
        .unwrap();
    let asm = Assembler::builder()
        .recent_history(RecentHistoryLayer::builder(store).window_size(20).build())
        .build();
    let prompt = asm.assemble(&vctx(1)).await;
    assert!(prompt.system_prompt.is_empty());
    assert_eq!(prompt.recent_history.len(), 1);
    assert!(prompt.recent_history[0].content_str().ends_with("hey"));
}

struct CountingLayer {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Layer for CountingLayer {
    fn name(&self) -> &'static str {
        "counter"
    }
    async fn build(&self, _ctx: &AssemblyContext) -> String {
        self.calls.fetch_add(1, Ordering::SeqCst);
        "X".to_owned()
    }
    async fn invalidation_key(&self, _ctx: &AssemblyContext) -> String {
        "stable".to_owned()
    }
}

#[tokio::test]
async fn assembler_cache_reuses_on_same_key() {
    let calls = Arc::new(AtomicUsize::new(0));
    let asm = Assembler::builder()
        .layer(Arc::new(CountingLayer {
            calls: calls.clone(),
        }))
        .build();
    asm.assemble(&vctx(1)).await;
    asm.assemble(&vctx(1)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn assembler_recent_history_returned_as_layer_provides() {
    let store = store();
    for i in 0..5 {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", 1, "user", format!("m{i}")))
            .unwrap();
    }
    let asm = Assembler::builder()
        .recent_history(RecentHistoryLayer::builder(store).window_size(20).build())
        .build();
    let prompt = asm.assemble(&vctx(1)).await;
    assert_eq!(prompt.recent_history.len(), 5);
}

struct DynamicLayer {
    calls: Arc<AtomicUsize>,
    key: Arc<Mutex<String>>,
}

#[async_trait]
impl Layer for DynamicLayer {
    fn name(&self) -> &'static str {
        "dyn"
    }
    async fn build(&self, _ctx: &AssemblyContext) -> String {
        self.calls.fetch_add(1, Ordering::SeqCst);
        format!("v={}", self.key.lock().unwrap())
    }
    async fn invalidation_key(&self, _ctx: &AssemblyContext) -> String {
        self.key.lock().unwrap().clone()
    }
}

#[tokio::test]
async fn assembler_cache_invalidates_on_key_change() {
    let calls = Arc::new(AtomicUsize::new(0));
    let key = Arc::new(Mutex::new("a".to_owned()));
    let asm = Assembler::builder()
        .layer(Arc::new(DynamicLayer {
            calls: calls.clone(),
            key: key.clone(),
        }))
        .build();
    let out1 = asm.assemble(&vctx(1)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    "b".clone_into(&mut key.lock().unwrap());
    let out2 = asm.assemble(&vctx(1)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_ne!(out1.system_prompt, out2.system_prompt);
}
