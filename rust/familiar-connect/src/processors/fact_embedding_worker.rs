//! FactEmbeddingWorker backfill (subsystem 07; Python `processors/fact_embedding_worker.py`).
//!
//! For each current fact (`superseded_at IS NULL`) lacking a `fact_embeddings`
//! row for the active embedder's [`Embedder::name`], run the embedder and
//! persist the vector. Idempotent: a model swap re-runs against the new `model`
//! key, accumulating rows beside the old ones and preserving audit history.
//!
//! The implicit watermark is the `unembedded_facts` query itself (there is no
//! watermark table). Per-row persistence means an interrupted batch resumes
//! where it stopped. A backend that returns the wrong number of vectors is a bug
//! — it is logged and the batch skipped without writing, so the same window
//! retries next tick.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::embedding::protocol::Embedder;
use crate::history::async_store::AsyncHistoryStore;
use crate::log_style as ls;

/// Log/task label + registry name for this projector.
const NAME: &str = "fact-embedding-worker";

/// Embeds new facts in batches off `fact_embeddings`.
///
/// The only projector with no LLM client — it consumes the [`Embedder`] seam
/// (04) instead. Embedders may vectorise across the batch; per-row upsert means
/// a partial batch still advances the implicit watermark.
pub struct FactEmbeddingWorker {
    store: Arc<AsyncHistoryStore>,
    embedder: Arc<dyn Embedder>,
    familiar_id: String,
    batch_size: i64,
    tick_interval: Duration,
}

impl FactEmbeddingWorker {
    /// Construct with the required handles; knobs default per spec
    /// (`batch_size = 32`, `tick_interval_s = 15.0`) and are set via the
    /// consuming builders below.
    #[must_use]
    pub fn new(
        store: Arc<AsyncHistoryStore>,
        embedder: Arc<dyn Embedder>,
        familiar_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            embedder,
            familiar_id: familiar_id.into(),
            batch_size: 32,
            tick_interval: Duration::from_secs_f64(15.0),
        }
    }

    /// Max facts per tick (clamped to `>= 1`, mirroring `max(1, x)`).
    #[must_use]
    pub const fn batch_size(mut self, batch_size: i64) -> Self {
        self.batch_size = if batch_size < 1 { 1 } else { batch_size };
        self
    }

    /// Idle-loop interval in seconds.
    #[must_use]
    pub fn tick_interval_s(mut self, secs: f64) -> Self {
        self.tick_interval = Duration::from_secs_f64(secs);
        self
    }

    /// The projector's log/task label.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        NAME
    }

    /// Forever loop; tick on interval. Cancel the token to stop.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Err(exc) = self.tick().await {
                tracing::warn!(
                    target: "familiar_connect.processors.fact_embedding_worker",
                    "{} {}",
                    ls::tag("FactEmbed", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// Embed up to `batch_size` unembedded facts; return the count written.
    pub async fn tick(&self) -> anyhow::Result<i64> {
        timed_async("fact_embedding.tick", async move {
            let pending = self
                .store
                .unembedded_facts(
                    self.familiar_id.clone(),
                    self.embedder.name().to_string(),
                    self.batch_size,
                )
                .await?;
            if pending.is_empty() {
                return Ok(0);
            }
            let texts: Vec<String> = pending.iter().map(|f| f.text.clone()).collect();
            let vectors = self.embedder.embed(&texts).await?;
            if vectors.len() != pending.len() {
                // Backend bug — surface loudly, skip batch; the same window
                // retries next tick (nothing persisted, no state advanced).
                tracing::warn!(
                    target: "familiar_connect.processors.fact_embedding_worker",
                    "{} {}",
                    ls::tag("FactEmbed", ls::R),
                    ls::kv_styled(
                        "mismatch",
                        &format!("{}!={}", vectors.len(), pending.len()),
                        ls::W,
                        ls::R,
                    ),
                );
                return Ok(0);
            }
            for (fact, vec) in pending.iter().zip(vectors.into_iter()) {
                self.store
                    .set_fact_embedding(fact.id, self.embedder.name().to_string(), vec)
                    .await?;
            }
            let written = i64::try_from(pending.len()).unwrap_or(i64::MAX);
            tracing::info!(
                target: "familiar_connect.processors.fact_embedding_worker",
                "{} {} {} {} {}",
                ls::tag("FactEmbed", ls::LC),
                ls::kv_styled("model", self.embedder.name(), ls::W, ls::LW),
                ls::kv_styled("dim", &self.embedder.dim().to_string(), ls::W, ls::LW),
                ls::kv_styled("written", &written.to_string(), ls::W, ls::LC),
                ls::kv_styled(
                    "latest_id",
                    &pending[pending.len() - 1].id.to_string(),
                    ls::W,
                    ls::LW,
                ),
            );
            Ok(written)
        })
        .await
    }
}
