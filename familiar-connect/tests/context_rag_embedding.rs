//! Ported from Python `tests/test_rag_embedding_rerank.py` — the M6 embedding
//! signal in RagContextLayer. The warn-once assertion uses a `tracing` capture
//! layer (DESIGN §4.8) instead of Python's `caplog`.
// The `Embedder::name` impls return string literals; the trait fixes the return
// type to `&str`, so clippy's `&'static str` suggestion cannot apply here.
#![allow(clippy::unnecessary_literal_bound)]

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use familiar_connect::context::{Layer, RagContextLayer};
use familiar_connect::embedding::protocol::Embedder;
use familiar_connect::history::store::AppendFact;

use helpers::{store, vctx};

// --- embedder doubles -------------------------------------------------------

struct FixedEmbedder {
    table: HashMap<String, Vec<f32>>,
}

impl FixedEmbedder {
    fn new(pairs: &[(&str, Vec<f32>)]) -> Self {
        Self {
            table: pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        }
    }
}

#[async_trait]
impl Embedder for FixedEmbedder {
    fn name(&self) -> &str {
        "fixed-v1"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| self.table.get(t).cloned().expect("cue in table"))
            .collect())
    }
}

struct BoomEmbedder;

#[async_trait]
impl Embedder for BoomEmbedder {
    fn name(&self) -> &str {
        "boom-v1"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        panic!("embedder must not be called when weight is 0");
    }
}

struct ErringEmbedder;

#[async_trait]
impl Embedder for ErringEmbedder {
    fn name(&self) -> &str {
        "erring-v1"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!("simulated backend fault"))
    }
}

struct CountingEmbedder {
    calls: AtomicUsize,
}

#[async_trait]
impl Embedder for CountingEmbedder {
    fn name(&self) -> &str {
        "count-v1"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
    }
}

// --- tracing capture --------------------------------------------------------

#[derive(Clone)]
struct VecWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for VecWriter {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

// --- tests ------------------------------------------------------------------

#[tokio::test]
async fn high_cosine_outranks_low_with_equal_bm25() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "apple ORTHOGONAL_FACT",
            vec![1],
        ))
        .unwrap();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "apple ALIGNED_FACT",
            vec![2],
        ))
        .unwrap();
    store
        .sync()
        .set_fact_embedding(1, "fixed-v1", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    store
        .sync()
        .set_fact_embedding(2, "fixed-v1", &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let embedder = Arc::new(FixedEmbedder::new(&[("apple", vec![1.0, 0.0, 0.0, 0.0])]));
    let layer = RagContextLayer::builder(store)
        .max_facts(1)
        .bm25_weight(1.0)
        .embedding_weight(5.0)
        .embedder(embedder)
        .build();
    layer.set_current_cue("apple");
    let out = layer.build(&vctx(1)).await;
    assert!(out.contains("ALIGNED_FACT"));
    assert!(!out.contains("ORTHOGONAL_FACT"));
}

#[tokio::test]
async fn zero_weight_disables_signal() {
    let store = store();
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
        .max_facts(3)
        .bm25_weight(1.0)
        .embedding_weight(0.0)
        .embedder(Arc::new(BoomEmbedder))
        .build();
    layer.set_current_cue("strawberry");
    assert!(
        layer
            .build(&vctx(1))
            .await
            .contains("Aria likes strawberries")
    );
}

#[tokio::test]
async fn missing_embedder_warns_once_and_falls_back() {
    let store = store();
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
        .max_facts(3)
        .bm25_weight(1.0)
        .embedding_weight(1.0)
        .build();
    layer.set_current_cue("strawberry");

    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(VecWriter(buf.clone()))
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .finish();
    let (out1, out2) = {
        let _guard = tracing::subscriber::set_default(subscriber);
        let out1 = layer.build(&vctx(1)).await;
        let out2 = layer.build(&vctx(1)).await;
        (out1, out2)
    };

    let captured = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    assert_eq!(
        captured.matches("embedding_weight").count(),
        1,
        "{captured}"
    );
    assert!(out1.contains("Aria likes strawberries"));
    assert!(out2.contains("Aria likes strawberries"));
}

#[tokio::test]
async fn unembedded_facts_get_neutral_score_not_zero() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "strawberries are red",
            vec![1],
        ))
        .unwrap();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "strawberries grow in summer",
            vec![2],
        ))
        .unwrap();
    store
        .sync()
        .set_fact_embedding(1, "fixed-v1", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    let embedder = Arc::new(FixedEmbedder::new(&[(
        "strawberries",
        vec![1.0, 0.0, 0.0, 0.0],
    )]));
    let layer = RagContextLayer::builder(store)
        .max_facts(1)
        .bm25_weight(0.0)
        .embedding_weight(1.0)
        .embedder(embedder)
        .build();
    layer.set_current_cue("strawberries");
    assert!(layer.build(&vctx(1)).await.contains("strawberries"));
}

#[tokio::test]
async fn erring_embedder_warns_once_and_falls_back() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria likes strawberries.",
            vec![1],
        ))
        .unwrap();
    // A stored vector under the embedder's name is required so the layer
    // actually calls embed() (the cue-embedding step) and hits the fault.
    store
        .sync()
        .set_fact_embedding(1, "erring-v1", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    let layer = RagContextLayer::builder(store)
        .max_facts(3)
        .bm25_weight(1.0)
        .embedding_weight(1.0)
        .embedder(Arc::new(ErringEmbedder))
        .build();
    layer.set_current_cue("strawberry");

    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(VecWriter(buf.clone()))
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .finish();
    let (out1, out2) = {
        let _guard = tracing::subscriber::set_default(subscriber);
        let out1 = layer.build(&vctx(1)).await;
        let out2 = layer.build(&vctx(1)).await;
        (out1, out2)
    };

    let captured = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    assert_eq!(captured.matches("embed() failed").count(), 1, "{captured}");
    // A faulting embedder must not panic; recall degrades to BM25-only, so the
    // fact still surfaces on both builds.
    assert!(out1.contains("Aria likes strawberries"));
    assert!(out2.contains("Aria likes strawberries"));
}

#[tokio::test]
async fn no_stored_vectors_skips_embedder_call() {
    let store = store();
    store
        .sync()
        .append_fact(AppendFact::new(
            "fam",
            Some(1),
            "Aria likes strawberries.",
            vec![1],
        ))
        .unwrap();
    let embedder = Arc::new(CountingEmbedder {
        calls: AtomicUsize::new(0),
    });
    let layer = RagContextLayer::builder(store)
        .max_facts(3)
        .bm25_weight(1.0)
        .embedding_weight(1.0)
        .embedder(embedder.clone())
        .build();
    layer.set_current_cue("strawberry");
    layer.build(&vctx(1)).await;
    assert_eq!(embedder.calls.load(Ordering::SeqCst), 0);
}
