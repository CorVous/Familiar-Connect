//! Ported from Python `tests/test_rag_merge_facts.py` — facts+turns dual
//! sections and importance-weighted reranking.

#[path = "context_helpers/mod.rs"]
mod helpers;

use familiar_connect::context::{Layer, RagContextLayer};
use familiar_connect::history::store::{AppendFact, AppendTurn};

use helpers::{store, vctx};

#[tokio::test]
async fn renders_fact_and_turn_sections() {
    let store = store();
    store
        .sync()
        .append_turn(AppendTurn::new(
            "fam",
            1,
            "user",
            "Mentioned strawberries in passing.",
        ))
        .unwrap();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria likes strawberries.",
            vec![1],
        ))
        .unwrap();
    let layer = RagContextLayer::builder(store)
        .max_results(5)
        .max_facts(3)
        .build();
    layer.set_current_cue("strawberry");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("relevant facts"));
    assert!(out.contains("Aria likes strawberries"));
    assert!(out.contains("relevant earlier turns"));
    assert!(out.contains("Mentioned strawberries in passing"));
}

#[tokio::test]
async fn fact_only_when_no_matching_turns() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria likes strawberries.",
            vec![99],
        ))
        .unwrap();
    let layer = RagContextLayer::builder(store).max_facts(3).build();
    layer.set_current_cue("strawberry");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("Aria likes strawberries"));
    assert!(!out.contains("earlier turns"));
}

#[tokio::test]
async fn key_reflects_fact_watermark() {
    let store = store();
    let layer = RagContextLayer::builder(store.clone()).build();
    layer.set_current_cue("foo");
    let k1 = layer.invalidation_key(&vctx(1)).await;
    store
        .sync()
        .append_fact(AppendFact::new("fam", Some(1), "New fact.", vec![1]))
        .unwrap();
    let k2 = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(k1, k2);
}

#[tokio::test]
async fn high_importance_outranks_low_with_equal_match() {
    let store = store();
    store
        .sync()
        .append_fact(
            AppendFact::new(
                "fam",
                Some(1),
                "Aria casually mentioned strawberries once.",
                vec![1],
            )
            .importance(1),
        )
        .unwrap();
    store
        .sync()
        .append_fact(
            AppendFact::new(
                "fam",
                Some(1),
                "Aria is severely allergic to strawberries.",
                vec![2],
            )
            .importance(10),
        )
        .unwrap();
    let layer = RagContextLayer::builder(store)
        .max_facts(1)
        .bm25_weight(1.0)
        .importance_weight(5.0)
        .recency_weight(0.0)
        .build();
    layer.set_current_cue("strawberry");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("severely allergic"));
    assert!(!out.contains("casually mentioned"));
}

#[tokio::test]
async fn default_weights_preserve_bm25_order() {
    let store = store();
    store
        .sync()
        .append_fact(
            AppendFact::new("fam", Some(1), "Aria likes strawberries.", vec![1]).importance(1),
        )
        .unwrap();
    let layer = RagContextLayer::builder(store).max_facts(5).build();
    layer.set_current_cue("strawberry");
    assert!(
        layer
            .build(&vctx(1))
            .await
            .contains("Aria likes strawberries")
    );
}

#[tokio::test]
async fn legacy_facts_treated_as_neutral() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Legacy fact about strawberries.",
            vec![1],
        ))
        .unwrap();
    let layer = RagContextLayer::builder(store)
        .max_facts(1)
        .bm25_weight(1.0)
        .importance_weight(2.0)
        .build();
    layer.set_current_cue("strawberry");
    assert!(
        layer
            .build(&vctx(1))
            .await
            .contains("Legacy fact about strawberries")
    );
}
