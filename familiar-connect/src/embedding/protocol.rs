//! Embedder seam trait (subsystem 04; Python `embedding/protocol.py`).
//!
//! The swappable text → fixed-dim vector seam. Consumers (`FactEmbeddingWorker`
//! in 07, `RagContextLayer` in 05) hold an `Option<Arc<dyn Embedder>>` and treat
//! "no embedder" and "seam off" identically, so every call site stays
//! non-conditional (DESIGN §4.8: the trait accepts non-registered impls — a bare
//! struct with `name`/`dim`/`embed` satisfies it, no registry needed).

use async_trait::async_trait;

/// Stable text → vector seam.
///
/// Contract (mirrors the Python `Embedder` Protocol):
///
/// * [`name`](Embedder::name) — backend label persisted with each vector so a
///   model swap accumulates new `(fact_id, model)` rows beside old ones instead
///   of corrupting the similarity space.
/// * [`dim`](Embedder::dim) — fixed dimensionality; storage assumes one
///   fixed-size vector per row. May legitimately be `0` (an unknown FastEmbed
///   model, pre-first-embed) — downstream code tolerates it.
/// * [`embed`](Embedder::embed) — batch interface; output order matches input,
///   `out.len() == texts.len()` always, empty input → empty vec, blank strings
///   map to a stable zero vector, never an error for content reasons.
///
/// `Err` is reserved for genuine backend faults (e.g. a FastEmbed model load
/// failure), never for blank/empty content. Implementations must be safe to call
/// from a background async task; CPU-bound work belongs off the reactor
/// (`spawn_blocking`).
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Backend label persisted alongside each vector (the storage `model` key).
    fn name(&self) -> &str;

    /// Fixed vector dimensionality (may be `0` before a probing backend's first
    /// embed).
    fn dim(&self) -> usize;

    /// One float vector per input text, order-preserving.
    ///
    /// Empty input yields an empty vec; blank strings project to a zero vector.
    /// Returns `Err` only on a genuine backend fault.
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}
