//! Ported from Python `tests/test_fact_embedding_worker.py`.
//!
//! Validates the watermark-driven embedding loop: pulls `unembedded_facts`,
//! calls the embedder once per batch, persists each vector, and is idempotent.

#[path = "workers_helpers/mod.rs"]
mod helpers;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use familiar_connect::embedding::{Embedder, HashEmbedder};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::AppendFact;
use familiar_connect::processors::fact_embedding_worker::FactEmbeddingWorker;

use helpers::store;

/// Embedder stub recording call counts for batch-size assertions.
struct CountingEmbedder {
    name: String,
    calls: Mutex<Vec<Vec<String>>>,
}

impl CountingEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            name: "stub-v1".to_string(),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn last_batch_len(&self) -> usize {
        self.calls.lock().unwrap().last().map_or(0, Vec::len)
    }
}

#[async_trait]
impl Embedder for CountingEmbedder {
    fn name(&self) -> &str {
        &self.name
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.calls.lock().unwrap().push(texts.to_vec());
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                vec![
                    f32::from(u16::try_from(i + 1).unwrap_or(u16::MAX)),
                    0.0,
                    0.0,
                    0.0,
                ]
            })
            .collect())
    }
}

/// Backend bug: returns fewer vectors than requested.
struct BadEmbedder {
    name: String,
}

impl BadEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            name: "bad-v1".to_string(),
        })
    }
}

#[async_trait]
impl Embedder for BadEmbedder {
    fn name(&self) -> &str {
        &self.name
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0, 0.0, 0.0, 0.0]])
    }
}

fn store_with_facts(n: i64) -> Arc<AsyncHistoryStore> {
    let s = store();
    for i in 0..n {
        s.sync()
            .append_fact(AppendFact::new(
                "fam",
                Some(1),
                format!("fact about thing {i}"),
                vec![i],
            ))
            .unwrap();
    }
    s
}

#[tokio::test]
async fn first_tick_embeds_all_pending_within_batch() {
    let store = store_with_facts(3);
    let embedder = CountingEmbedder::new();
    let worker = FactEmbeddingWorker::new(store.clone(), embedder.clone(), "fam").batch_size(10);
    let written = worker.tick().await.unwrap();
    assert_eq!(written, 3);
    assert_eq!(embedder.call_count(), 1);
    assert_eq!(embedder.last_batch_len(), 3);
    assert_eq!(
        store
            .sync()
            .latest_embedded_fact_id("fam", "stub-v1")
            .unwrap(),
        3
    );
}

#[tokio::test]
async fn batch_size_caps_per_tick_writes() {
    let store = store_with_facts(5);
    let embedder = CountingEmbedder::new();
    let worker = FactEmbeddingWorker::new(store.clone(), embedder, "fam").batch_size(2);
    let written = worker.tick().await.unwrap();
    assert_eq!(written, 2);
    assert_eq!(
        store
            .sync()
            .latest_embedded_fact_id("fam", "stub-v1")
            .unwrap(),
        2
    );
}

#[tokio::test]
async fn subsequent_tick_skips_embedded() {
    let store = store_with_facts(3);
    let embedder = CountingEmbedder::new();
    let worker = FactEmbeddingWorker::new(store.clone(), embedder.clone(), "fam").batch_size(10);
    worker.tick().await.unwrap();
    let before = embedder.call_count();
    // Another tick with no new facts: no work, no embedder call.
    let written = worker.tick().await.unwrap();
    assert_eq!(written, 0);
    assert_eq!(embedder.call_count(), before);
}

#[tokio::test]
async fn resume_after_partial_progress() {
    let store = store_with_facts(5);
    let embedder = CountingEmbedder::new();
    let worker = FactEmbeddingWorker::new(store.clone(), embedder, "fam").batch_size(2);
    let n1 = worker.tick().await.unwrap();
    let n2 = worker.tick().await.unwrap();
    let n3 = worker.tick().await.unwrap();
    let n4 = worker.tick().await.unwrap();
    assert_eq!((n1, n2, n3, n4), (2, 2, 1, 0));
    assert_eq!(
        store
            .sync()
            .latest_embedded_fact_id("fam", "stub-v1")
            .unwrap(),
        5
    );
}

#[tokio::test]
async fn real_hash_embedder_persists_vectors_for_recall() {
    let store = store_with_facts(2);
    let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(64).unwrap());
    let model = embedder.name().to_string();
    let worker = FactEmbeddingWorker::new(store.clone(), embedder, "fam");
    worker.tick().await.unwrap();
    let got = store.sync().get_fact_embeddings(&[1, 2], &model).unwrap();
    let mut keys: Vec<i64> = got.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, vec![1, 2]);
    assert!(got.values().all(|v| v.len() == 64));
}

#[tokio::test]
async fn empty_store_tick_is_noop() {
    let store = store_with_facts(0);
    let embedder = CountingEmbedder::new();
    let worker = FactEmbeddingWorker::new(store, embedder.clone(), "fam");
    assert_eq!(worker.tick().await.unwrap(), 0);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test]
async fn mismatched_embedder_batch_skips_without_writing() {
    let store = store_with_facts(3);
    let worker = FactEmbeddingWorker::new(store.clone(), BadEmbedder::new(), "fam");
    assert_eq!(worker.tick().await.unwrap(), 0);
    assert_eq!(
        store
            .sync()
            .latest_embedded_fact_id("fam", "bad-v1")
            .unwrap(),
        0
    );
}
