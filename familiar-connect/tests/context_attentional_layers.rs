//! Ported from Python `tests/test_attentional_layers.py` — the RecentHistoryLayer
//! halves (cross-channel window, archive watermark, `#channel_id` tags, channel
//! markers, token-trim realignment). The final-reminder focus/unread halves live
//! in the `context::final_reminder` in-module tests.

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::collections::HashMap;
use std::sync::Arc;

use familiar_connect::budget::estimate_message_tokens;
use familiar_connect::context::{ChannelResolver, RecentHistoryLayer};
use familiar_connect::history::store::AppendTurn;
use familiar_connect::llm::Message;
use regex::Regex;

use helpers::{author, store, vctx};

fn resolver(pairs: &[(i64, &str)]) -> ChannelResolver {
    let map: HashMap<i64, String> = pairs.iter().map(|(k, v)| (*k, (*v).to_owned())).collect();
    Arc::new(move |cid| map.get(&cid).cloned())
}

fn none_resolver() -> ChannelResolver {
    Arc::new(|_cid| None)
}

fn is_marker(msg: &Message) -> bool {
    !msg.content_str().starts_with('[')
}

fn markers(msgs: &[Message]) -> Vec<String> {
    msgs.iter()
        .filter(|m| is_marker(m))
        .map(Message::content_str)
        .collect()
}

fn marker_indices(msgs: &[Message]) -> Vec<usize> {
    msgs.iter()
        .enumerate()
        .filter(|(_, m)| is_marker(m))
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// Cross-channel window
// ---------------------------------------------------------------------------

#[tokio::test]
async fn uses_recent_cross_channel() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "message in channel 1").author(alice.clone()),
        )
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 2, "user", "message in channel 2").author(alice))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let all: String = msgs
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("channel 1"));
    assert!(all.contains("channel 2"));
}

#[tokio::test]
async fn only_consumed_turns_appear() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "consumed message").author(alice.clone()))
        .unwrap();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 1, "user", "staged message")
                .author(alice)
                .consumed(false),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let all: String = msgs
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("consumed message"));
    assert!(!all.contains("staged message"));
}

// ---------------------------------------------------------------------------
// Archive watermark
// ---------------------------------------------------------------------------

#[tokio::test]
async fn archive_no_watermark_includes_all() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "before departure").author(alice.clone()))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "after return").author(alice))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let all: String = layer
        .recent_messages(&vctx(1))
        .await
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("before departure"));
    assert!(all.contains("after return"));
}

#[tokio::test]
async fn archive_watermark_hides_pre_archive() {
    let store = store();
    let alice = author("1", "Alice");
    let departure = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "before departure").author(alice.clone()))
        .unwrap();
    store
        .sync()
        .set_archive_watermark("fam", 1, departure.id)
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "after return").author(alice))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let all: String = layer
        .recent_messages(&vctx(1))
        .await
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!all.contains("before departure"));
    assert!(all.contains("after return"));
}

#[tokio::test]
async fn archive_watermark_does_not_leak_across_channels() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 2, "user", "other channel chatter").author(alice.clone()),
        )
        .unwrap();
    let departure = store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "before departure").author(alice))
        .unwrap();
    store
        .sync()
        .set_archive_watermark("fam", 1, departure.id)
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let all: String = layer
        .recent_messages(&vctx(1))
        .await
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all.contains("other channel chatter"));
    assert!(!all.contains("before departure"));
}

// ---------------------------------------------------------------------------
// Channel-id tag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn user_turn_includes_channel_id_in_prefix() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 10, "user", "hello").author(author("42", "Alice")))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(10)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    let re = Regex::new(r"^\[\d{2}:\d{2} Alice #10\] hello$").unwrap();
    assert!(re.is_match(&user.content_str()), "{}", user.content_str());
}

#[tokio::test]
async fn channel_id_tag_in_all_user_turns() {
    let store = store();
    let alice = author("1", "Alice");
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 1, "user", "ch1 msg").author(alice.clone()))
        .unwrap();
    store
        .sync()
        .append_turn(AppendTurn::new("fam", 2, "user", "ch2 msg").author(alice))
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let re = Regex::new(r"^\[\d{2}:\d{2} \w+ #\d+\]").unwrap();
    for msg in msgs.iter().filter(|m| m.role == "user") {
        assert!(re.is_match(&msg.content_str()), "{}", msg.content_str());
    }
}

#[tokio::test]
async fn message_id_tag_alongside_channel_tag() {
    let store = store();
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", 5, "user", "hi")
                .author(author("1", "Alice"))
                .platform_message_id("msg-99"),
        )
        .unwrap();
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(5)).await;
    let user = msgs.iter().find(|m| m.role == "user").unwrap();
    assert!(user.content_str().contains("#5"));
    assert!(user.content_str().contains("#msg-99"));
}

// ---------------------------------------------------------------------------
// Channel markers
// ---------------------------------------------------------------------------

fn seed_marker_turn(
    store: &familiar_connect::history::async_store::AsyncHistoryStore,
    channel: i64,
    content: &str,
    msg_id: &str,
) {
    store
        .sync()
        .append_turn(
            AppendTurn::new("fam", channel, "user", content)
                .author(author("1", "Alice"))
                .platform_message_id(msg_id),
        )
        .unwrap();
}

#[tokio::test]
async fn single_channel_emits_no_markers() {
    let store = store();
    seed_marker_turn(&store, 1, "one", "m1");
    seed_marker_turn(&store, 1, "two", "m2");
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(markers(&msgs).is_empty());
    assert_eq!(msgs.len(), 2);
}

#[tokio::test]
async fn multi_channel_leading_and_change_markers() {
    let store = store();
    seed_marker_turn(&store, 1, "a one", "m1");
    seed_marker_turn(&store, 1, "a two", "m2");
    seed_marker_turn(&store, 2, "b one", "m3");
    seed_marker_turn(&store, 1, "a three", "m4");
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(marker_indices(&msgs), vec![0, 3, 5]);
    assert_eq!(msgs.len(), 7);
    assert!(msgs[1].content_str().ends_with("a one"));
    assert!(msgs[2].content_str().ends_with("a two"));
    assert!(msgs[4].content_str().ends_with("b one"));
    assert!(msgs[6].content_str().ends_with("a three"));
}

#[tokio::test]
async fn marker_resolves_channel_and_server_names() {
    let store = store();
    seed_marker_turn(&store, 1, "hi", "m1");
    seed_marker_turn(&store, 2, "yo", "m2");
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .channel_name_resolver(resolver(&[(1, "general"), (2, "random")]))
        .guild_name_resolver(resolver(&[(1, "My Server"), (2, "My Server")]))
        .build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(markers(&msgs).contains(&"My Server/#general".to_owned()));
}

#[tokio::test]
async fn marker_falls_back_to_channel_id_without_resolvers() {
    let store = store();
    seed_marker_turn(&store, 1, "hi", "m1");
    seed_marker_turn(&store, 2, "yo", "m2");
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let m = markers(&msgs);
    assert!(m.contains(&"#1".to_owned()));
    assert!(m.contains(&"#2".to_owned()));
}

#[tokio::test]
async fn marker_omits_server_when_guild_unknown() {
    let store = store();
    seed_marker_turn(&store, 1, "hi", "m1");
    seed_marker_turn(&store, 2, "yo", "m2");
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .channel_name_resolver(resolver(&[(1, "general"), (2, "random")]))
        .guild_name_resolver(none_resolver())
        .build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let m = markers(&msgs);
    assert!(m.contains(&"#general".to_owned()));
    assert!(m.iter().all(|s| !s.contains('/')));
}

#[tokio::test]
async fn marker_is_distinct_user_message_without_name() {
    let store = store();
    seed_marker_turn(&store, 1, "hi", "m1");
    seed_marker_turn(&store, 2, "yo", "m2");
    let layer = RecentHistoryLayer::builder(store).window_size(20).build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    let marker = msgs.iter().find(|m| is_marker(m)).unwrap();
    assert_eq!(marker.role, "user");
    assert!(marker.name.is_none());
}

async fn per_turn_cost(
    store: Arc<familiar_connect::history::async_store::AsyncHistoryStore>,
) -> i64 {
    let probe = RecentHistoryLayer::builder(store)
        .window_size(20)
        .build()
        .recent_messages(&vctx(1))
        .await;
    probe
        .iter()
        .find(|m| !is_marker(m))
        .map(estimate_message_tokens)
        .unwrap()
}

#[tokio::test]
async fn token_trim_realigns_markers_to_surviving_window() {
    let store = store();
    seed_marker_turn(&store, 9, "same body", "m1");
    seed_marker_turn(&store, 9, "same body", "m2");
    seed_marker_turn(&store, 1, "same body", "m3");
    seed_marker_turn(&store, 2, "same body", "m4");
    let cost = per_turn_cost(store.clone()).await;
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .max_tokens(Some(2 * cost))
        .build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert_eq!(msgs.len(), 4);
    assert_eq!(marker_indices(&msgs), vec![0, 2]);
    assert_eq!(markers(&msgs), vec!["#1".to_owned(), "#2".to_owned()]);
    assert!(msgs[1].content_str().contains("#1"));
    assert!(msgs[3].content_str().contains("#2"));
}

#[tokio::test]
async fn multi_before_trim_single_after_trim_emits_no_markers() {
    let store = store();
    seed_marker_turn(&store, 1, "same body", "m1");
    seed_marker_turn(&store, 1, "same body", "m2");
    seed_marker_turn(&store, 2, "same body", "m3");
    seed_marker_turn(&store, 2, "same body", "m4");
    let cost = per_turn_cost(store.clone()).await;
    let layer = RecentHistoryLayer::builder(store)
        .window_size(20)
        .max_tokens(Some(2 * cost))
        .build();
    let msgs = layer.recent_messages(&vctx(1)).await;
    assert!(markers(&msgs).is_empty());
    assert_eq!(msgs.len(), 2);
}
