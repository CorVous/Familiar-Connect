//! Ported from Python `tests/test_message_reactions.py` — the RecentHistoryLayer
//! reaction-render halves (`TestRecentHistoryReactions`) and the single batched
//! query spy (`TestRecentHistoryReactionsBatch`) via the DB trace hook.

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::sync::{Arc, Mutex};

use familiar_connect::context::RecentHistoryLayer;
use familiar_connect::history::db::TraceCallback;
use familiar_connect::history::store::AppendTurn;

use helpers::{author, store, vctx};

#[tokio::test]
async fn user_message_renders_reactions() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi")
                .author(author("1", "Alice"))
                .platform_message_id("m1"),
        )
        .unwrap();
    store
        .sync()
        .set_reaction("fam", "m1", "\u{1f44d}", 2)
        .unwrap();
    store
        .sync()
        .set_reaction("fam", "m1", "\u{2764}\u{fe0f}", 1)
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(msgs[0].role, "user");
    assert!(msgs[0].content_str().contains("\u{1f44d} x2"));
    assert!(msgs[0].content_str().contains("\u{2764}\u{fe0f} x1"));
}

#[tokio::test]
async fn assistant_message_renders_reactions() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "assistant", "hi back").platform_message_id("bot1"))
        .unwrap();
    store
        .sync()
        .set_reaction("fam", "bot1", "\u{1f389}", 1)
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(msgs[0].role, "assistant");
    assert!(msgs[0].content_str().contains("\u{1f389} x1"));
}

#[tokio::test]
async fn no_reactions_no_suffix() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi")
                .author(author("1", "Alice"))
                .platform_message_id("m1"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(!msgs[0].content_str().to_lowercase().contains("reactions"));
}

#[tokio::test]
async fn window_resolves_reactions_in_one_query() {
    let store = store();
    for i in 0..4 {
        store
            .sync()
            .append_turn(
                AppendTurn::new("fam", 1, "user", format!("msg {i}"))
                    .author(author("1", "Alice"))
                    .platform_message_id(format!("m{i}")),
            )
            .unwrap();
        store
            .sync()
            .set_reaction("fam", &format!("m{i}"), "\u{1f44d}", 1)
            .unwrap();
    }
    let layer = RecentHistoryLayer::builder(store.clone())
        .window_size(20)
        .build();

    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let cb: TraceCallback = Arc::new(move |sql: &str| sink.lock().unwrap().push(sql.to_owned()));
    store.sync().conn().set_trace_callback(Some(cb));
    layer.recent_messages(&vctx(1)).await;
    store.sync().conn().set_trace_callback(None);

    let reaction_selects = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|s| {
            s.contains("message_reactions") && s.trim_start().to_uppercase().starts_with("SELECT")
        })
        .count();
    assert_eq!(reaction_selects, 1);
}
