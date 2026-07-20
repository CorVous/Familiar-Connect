//! Ported from Python `tests/test_lorebook_layer.py`.

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::path::{Path, PathBuf};

use familiar_connect::context::{Layer, LorebookLayer};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::AppendTurn;

use helpers::{store, tctx};

fn seed(store: &AsyncHistoryStore, texts: &[&str], channel_id: i64) {
    for text in texts {
        store
            .sync()
            .append_turn(AppendTurn::new("fam", channel_id, "user", *text))
            .unwrap();
    }
}

fn write_lorebook(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("lorebook.toml");
    std::fs::write(&path, body).unwrap();
    path
}

#[tokio::test]
async fn empty_when_file_missing() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["anything"], 1);
    let layer = LorebookLayer::builder(store, dir.path().join("missing.toml")).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn empty_when_no_entries_match() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["hello there"], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"dragon\"]\ncontent = \"Dragons breathe fire.\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn single_entry_triggers_on_key_match() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["Tell me about Paris."], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"Paris is the capital of France.\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("## Lorebook"));
    assert!(out.contains("Paris is the capital of France."));
}

#[tokio::test]
async fn match_is_case_insensitive() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["PARIS is lovely"], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"City info.\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.contains("City info."));
}

#[tokio::test]
async fn multiple_entries_sorted_by_priority_desc() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["paris and dragons"], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"low-pri Paris\"\npriority = 1\n\n\
         [[entries]]\nkeys = [\"dragon\"]\ncontent = \"high-pri dragon\"\npriority = 100\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.find("high-pri dragon").unwrap() < out.find("low-pri Paris").unwrap());
}

#[tokio::test]
async fn selective_requires_all_keys() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(
        &store,
        &["We met in Paris. No mention of the second key."],
        1,
    );
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\", \"berlin\"]\nselective = true\ncontent = \"Both cities matter.\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn selective_fires_when_all_keys_present() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["We met in Paris and later moved to Berlin."], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\", \"berlin\"]\nselective = true\ncontent = \"Both cities matter.\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.contains("Both cities matter."));
}

#[tokio::test]
async fn max_entries_caps_output() {
    use std::fmt::Write as _;
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["alpha beta gamma delta"], 1);
    let mut body = String::new();
    for (k, p) in [("alpha", 4), ("beta", 3), ("gamma", 2), ("delta", 1)] {
        let _ = write!(
            body,
            "[[entries]]\nkeys = [\"{k}\"]\ncontent = \"{k}-content\"\npriority = {p}\n\n"
        );
    }
    let book = write_lorebook(dir.path(), &body);
    let layer = LorebookLayer::builder(store, book).max_entries(2).build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("alpha-content"));
    assert!(out.contains("beta-content"));
    assert!(!out.contains("gamma-content"));
    assert!(!out.contains("delta-content"));
}

#[tokio::test]
async fn max_tokens_truncates_block() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["alpha beta"], 1);
    let long = "x".repeat(4000);
    let body = format!(
        "[[entries]]\nkeys = [\"alpha\"]\ncontent = \"{long}\"\npriority = 2\n\n\
         [[entries]]\nkeys = [\"beta\"]\ncontent = \"second entry\"\npriority = 1\n"
    );
    let book = write_lorebook(dir.path(), &body);
    let layer = LorebookLayer::builder(store, book)
        .max_tokens(Some(20))
        .build();
    let out = layer.build(&tctx(1)).await;
    assert!(out.chars().count() < 200);
}

#[tokio::test]
async fn only_recent_window_scanned() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["I love Paris"], 1);
    for i in 0..50 {
        seed(&store, &[&format!("filler turn {i}")], 1);
    }
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"Paris info.\"\n",
    );
    let layer = LorebookLayer::builder(store, book)
        .recent_window(10)
        .build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn empty_keys_list_never_fires() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["anything goes"], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = []\ncontent = \"should not appear\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn invalidation_key_changes_when_match_set_changes() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"city\"\n",
    );
    let layer = LorebookLayer::builder(store.clone(), book).build();
    let k0 = layer.invalidation_key(&tctx(1)).await;
    seed(&store, &["tell me about paris"], 1);
    let k1 = layer.invalidation_key(&tctx(1)).await;
    assert_ne!(k0, k1);
}

#[tokio::test]
async fn invalidation_key_changes_when_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["paris"], 1);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"v1\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    let k0 = layer.invalidation_key(&tctx(1)).await;
    write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"v2\"\n",
    );
    let k1 = layer.invalidation_key(&tctx(1)).await;
    assert_ne!(k0, k1);
}

#[tokio::test]
async fn malformed_toml_yields_empty_block() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["anything"], 1);
    let book = write_lorebook(dir.path(), "this is = not valid [[ toml");
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn channel_scope_uses_active_channel() {
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    seed(&store, &["paris"], 1);
    seed(&store, &["no key here"], 2);
    let book = write_lorebook(
        dir.path(),
        "[[entries]]\nkeys = [\"paris\"]\ncontent = \"city\"\n",
    );
    let layer = LorebookLayer::builder(store, book).build();
    assert!(layer.build(&tctx(2)).await.is_empty());
    assert!(layer.build(&tctx(1)).await.contains("city"));
}
