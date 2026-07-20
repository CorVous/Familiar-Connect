//! Ported from Python `tests/test_reflection_layer.py`.

#[path = "context_helpers/mod.rs"]
mod helpers;

use familiar_connect::context::{Layer, ReflectionLayer};
use familiar_connect::history::store::{AppendFact, NewFact};

use helpers::{store, tctx};

const fn reflection_layer(
    store: std::sync::Arc<familiar_connect::history::async_store::AsyncHistoryStore>,
    max_reflections: i64,
) -> ReflectionLayer {
    ReflectionLayer::new(store).with_max_reflections(max_reflections)
}

#[tokio::test]
async fn empty_when_no_reflections() {
    let store = store();
    let layer = reflection_layer(store, 3);
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn renders_text_with_citation_breadcrumbs() {
    let store = store();
    store
        .sync()
        .append_reflection(
            "fam",
            Some(1),
            "Crew morale dipped after Friday.",
            &[42],
            &[7],
            42,
            7,
        )
        .unwrap();
    let layer = reflection_layer(store, 3);
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("## Recent reflections"));
    assert!(out.contains("Crew morale dipped after Friday."));
    assert!(out.contains("[T#42, F#7]"));
    assert!(!out.contains("(stale)"));
}

#[tokio::test]
async fn flags_stale_when_any_cited_fact_superseded() {
    let store = store();
    let f1 = store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria lives in Paris.",
            vec![1],
        ))
        .unwrap();
    let f2 = store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria lives in Berlin.",
            vec![2],
        ))
        .unwrap();
    store
        .sync()
        .supersede("fam", &[f1.id], NewFact::Repoint(f2.id))
        .unwrap();
    store
        .sync()
        .append_reflection(
            "fam",
            Some(1),
            "Aria's location keeps shifting.",
            &[1],
            &[f1.id, f2.id],
            2,
            f2.id,
        )
        .unwrap();
    let layer = reflection_layer(store, 3);
    assert!(layer.build(&tctx(1)).await.contains("(stale)"));
}

#[tokio::test]
async fn no_stale_when_no_facts_superseded() {
    let store = store();
    let f = store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria lives in Berlin.",
            vec![1],
        ))
        .unwrap();
    store
        .sync()
        .append_reflection(
            "fam",
            Some(1),
            "Aria likes the cold.",
            &[1],
            &[f.id],
            1,
            f.id,
        )
        .unwrap();
    let layer = reflection_layer(store, 3);
    assert!(!layer.build(&tctx(1)).await.contains("(stale)"));
}

#[tokio::test]
async fn max_reflections_caps_output() {
    let store = store();
    for i in 0..5 {
        store
            .sync()
            .append_reflection(
                "fam",
                Some(1),
                &format!("reflection {i}"),
                &[i + 1],
                &[],
                i + 1,
                0,
            )
            .unwrap();
    }
    let layer = reflection_layer(store, 2);
    let out = layer.build(&tctx(1)).await;
    assert_eq!(out.matches("- reflection").count(), 2);
    assert!(out.contains("reflection 4"));
    assert!(out.contains("reflection 3"));
    assert!(!out.contains("reflection 2"));
}

#[tokio::test]
async fn max_reflections_zero_opts_out() {
    let store = store();
    store
        .sync()
        .append_reflection("fam", Some(1), "something", &[1], &[], 1, 0)
        .unwrap();
    let layer = reflection_layer(store, 0);
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn invalidation_key_changes_on_new_reflection() {
    let store = store();
    let layer = reflection_layer(store.clone(), 3);
    let k0 = layer.invalidation_key(&tctx(1)).await;
    store
        .sync()
        .append_reflection("fam", Some(1), "r1", &[1], &[], 1, 0)
        .unwrap();
    let k1 = layer.invalidation_key(&tctx(1)).await;
    store
        .sync()
        .append_reflection("fam", Some(1), "r2", &[1], &[], 2, 0)
        .unwrap();
    let k2 = layer.invalidation_key(&tctx(1)).await;
    assert_ne!(k0, k1);
    assert_ne!(k1, k2);
}

#[tokio::test]
async fn channel_scope_excludes_other_channels() {
    let store = store();
    store
        .sync()
        .append_reflection("fam", Some(1), "ch1 reflection", &[1], &[], 1, 0)
        .unwrap();
    store
        .sync()
        .append_reflection("fam", Some(2), "ch2 reflection", &[1], &[], 1, 0)
        .unwrap();
    let layer = reflection_layer(store, 5);
    let out = layer.build(&tctx(1)).await;
    assert!(out.contains("ch1 reflection"));
    assert!(!out.contains("ch2 reflection"));
}

#[tokio::test]
async fn max_tokens_truncates_block() {
    let store = store();
    let long = "x".repeat(4000);
    store
        .sync()
        .append_reflection("fam", Some(1), &long, &[1], &[], 1, 0)
        .unwrap();
    let layer = ReflectionLayer::new(store)
        .with_max_reflections(3)
        .with_max_tokens(20);
    let out = layer.build(&tctx(1)).await;
    assert!(out.chars().count() < 200);
}
