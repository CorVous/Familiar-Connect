//! FastEmbed ONNX embedder (subsystem 04; Python `embedding/fastembed.py`;
//! feature `local-embed`).
//!
//! An ONNX-compiled sentence-transformer wrapper — the intended production
//! backend for paraphrase-tolerant fact recall, replacing the `hash` baseline's
//! token-overlap proxy with real semantic similarity. Default model is
//! BGE-small (384-dim), per the ecosystem report.
//!
//! **Lazy load** (spec 04 §3): the underlying ONNX model is constructed on the
//! first non-empty [`embed`](FastEmbedEmbedder::embed) call, double-checked under
//! a lock so exactly one model is built even under concurrent first calls. Both
//! the construction and the (CPU-bound) inference run on a blocking worker via
//! [`tokio::task::spawn_blocking`] so the reactor keeps draining. `embed([])`
//! returns `[]` without loading.
//!
//! **`dim` is mutable state** (spec 04 §4, Rust port note): known models seed
//! `dim` at construction from a static table; an unknown model reports `0` until
//! the first real vector probes it (a nonzero pre-known dim is never overwritten
//! by the probe). The interior [`AtomicUsize`] backs the `&self` [`Embedder::dim`]
//! accessor.
//!
//! **The model loader is a seam** ([`ModelLoader`]): the production
//! [`FastembedLoader`] wraps the `fastembed` crate, but the lazy-load / dim-probe
//! / single-load / cache-dir behaviors are pinned by tests through stub loaders
//! that never touch ONNX (Python stubbed `sys.modules["fastembed"]`; here we
//! inject the loader). Tests needing a real ~130 MB model download are
//! `#[ignore]`.
#![cfg(feature = "local-embed")]

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::embedding::protocol::Embedder;

/// Default model name (BGE-small; 384-dim). Matches the Python
/// `DEFAULT_MODEL_NAME` and the `[providers.embedding].fastembed_model` default.
pub const DEFAULT_MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";

/// Known model dimensionalities, so `dim` can be advertised before the first
/// embed lands. A model not listed here reports `dim == 0` until the first
/// embed probes a real vector (Python `_KNOWN_DIMS`).
#[must_use]
fn known_dim(model_name: &str) -> usize {
    match model_name {
        "BAAI/bge-small-en-v1.5"
        | "sentence-transformers/all-MiniLM-L6-v2"
        | "intfloat/e5-small-v2"
        | "intfloat/multilingual-e5-small" => 384,
        "BAAI/bge-base-en-v1.5" => 768,
        "BAAI/bge-large-en-v1.5" => 1024,
        _ => 0,
    }
}

/// Seam: a loaded text-embedding model, callable synchronously (CPU-bound).
///
/// `embed` takes `&self` so a shared [`Arc`] can be handed to
/// [`spawn_blocking`](tokio::task::spawn_blocking); the production impl wraps the
/// `fastembed` model (whose own `embed` needs `&mut`) in a [`std::sync::Mutex`].
trait TextModel: Send + Sync {
    /// One float vector per input text, order-preserving.
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Seam: constructs (loads) a [`TextModel`]. Blocking and fallible — a load
/// failure (missing runtime, unmappable model, blocked download) surfaces as an
/// `Err` on every embed attempt, leaving the model unloaded and retryable.
trait ModelLoader: Send + Sync {
    /// Load the model named `model_name`, honoring `cache_dir` when set.
    fn load(&self, model_name: &str, cache_dir: Option<&str>)
    -> anyhow::Result<Box<dyn TextModel>>;
}

/// ONNX sentence embedder. Lazy model load on first non-empty use.
pub struct FastEmbedEmbedder {
    model_name: String,
    cache_dir: Option<String>,
    /// `fastembed:{model_name}` — the storage `model` key, so a model swap
    /// accumulates new `(fact_id, model)` rows beside old ones.
    name: String,
    /// Fixed dimensionality; `0` until a probe fills it for unknown models.
    dim: AtomicUsize,
    loader: Arc<dyn ModelLoader>,
    /// The loaded model, `None` until first embed; guarded so exactly one load
    /// happens under concurrency.
    model: Mutex<Option<Arc<dyn TextModel>>>,
}

impl FastEmbedEmbedder {
    /// Construct with the given model name and optional cache dir, using the
    /// production `fastembed`-backed loader.
    #[must_use]
    pub fn new(model_name: impl Into<String>, cache_dir: Option<String>) -> Self {
        Self::with_loader(model_name, cache_dir, Arc::new(FastembedLoader))
    }

    /// Construct with an injected model loader (test seam + wiring extension).
    fn with_loader(
        model_name: impl Into<String>,
        cache_dir: Option<String>,
        loader: Arc<dyn ModelLoader>,
    ) -> Self {
        let model_name = model_name.into();
        let dim = known_dim(&model_name);
        let name = format!("fastembed:{model_name}");
        Self {
            model_name,
            cache_dir,
            name,
            dim: AtomicUsize::new(dim),
            loader,
            model: Mutex::new(None),
        }
    }

    /// The configured model name (the HuggingFace-style identifier).
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Load the model once, returning the shared handle. Double-checked under
    /// the lock: concurrent first callers serialize and exactly one load runs.
    async fn ensure_loaded(&self) -> anyhow::Result<Arc<dyn TextModel>> {
        let mut guard = self.model.lock().await;
        if let Some(model) = guard.as_ref() {
            return Ok(model.clone());
        }
        let loader = self.loader.clone();
        let model_name = self.model_name.clone();
        let cache_dir = self.cache_dir.clone();
        let loaded =
            tokio::task::spawn_blocking(move || loader.load(&model_name, cache_dir.as_deref()))
                .await??;
        let model: Arc<dyn TextModel> = Arc::from(loaded);
        *guard = Some(model.clone());
        Ok(model)
    }
}

impl Default for FastEmbedEmbedder {
    /// The default model is BGE-small, no cache-dir override.
    fn default() -> Self {
        Self::new(DEFAULT_MODEL_NAME, None)
    }
}

#[async_trait]
impl Embedder for FastEmbedEmbedder {
    fn name(&self) -> &str {
        &self.name
    }

    fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.ensure_loaded().await?;
        let owned: Vec<String> = texts.to_vec();
        let vectors = tokio::task::spawn_blocking(move || model.embed(&owned)).await??;
        // Probe `dim` opportunistically the first time we see a real vector —
        // covers models absent from `known_dim`. A nonzero pre-known dim is
        // never overwritten (guard is "current dim == 0"), matching Python's
        // `if vectors and not self.dim`.
        if self.dim.load(Ordering::Relaxed) == 0 {
            if let Some(first) = vectors.first() {
                self.dim.store(first.len(), Ordering::Relaxed);
            }
        }
        Ok(vectors)
    }
}

// ---------------------------------------------------------------------------
// Production loader — the real `fastembed` crate behind the seam.
// ---------------------------------------------------------------------------

/// The production [`ModelLoader`], backed by the `fastembed` crate's ONNX
/// runtime. Loading downloads (and caches) the model on first use.
struct FastembedLoader;

/// Map the config model name (HuggingFace-style, the storage key) to the
/// `fastembed` crate's internal `EmbeddingModel` enum. The crate keys its own
/// registry on different `model_code` strings (e.g. `Xenova/...`), so this is an
/// explicit table over the names the port advertises in `known_dim`.
fn resolve_model(model_name: &str) -> anyhow::Result<fastembed::EmbeddingModel> {
    use fastembed::EmbeddingModel;
    let model = match model_name {
        "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
        "intfloat/multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
        other => {
            anyhow::bail!(
                "fastembed model {other:?} is not mapped to a fastembed EmbeddingModel variant"
            )
        }
    };
    Ok(model)
}

impl ModelLoader for FastembedLoader {
    fn load(
        &self,
        model_name: &str,
        cache_dir: Option<&str>,
    ) -> anyhow::Result<Box<dyn TextModel>> {
        use fastembed::{InitOptions, TextEmbedding};

        let model_enum = resolve_model(model_name)?;
        let mut options = InitOptions::new(model_enum);
        if let Some(dir) = cache_dir {
            options = options.with_cache_dir(std::path::PathBuf::from(dir));
        }
        let model = TextEmbedding::try_new(options)
            .map_err(|err| anyhow::anyhow!("fastembed model load failed: {err}"))?;
        Ok(Box::new(FastembedModel {
            model: StdMutex::new(model),
        }))
    }
}

/// Wraps a `fastembed::TextEmbedding` (whose `embed` needs `&mut self`) so the
/// `&self` [`TextModel::embed`] seam can drive it from a blocking worker.
struct FastembedModel {
    model: StdMutex<fastembed::TextEmbedding>,
}

impl TextModel for FastembedModel {
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("fastembed model mutex poisoned"))?;
        let vectors = model.embed(texts, None)?;
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::{FastEmbedEmbedder, ModelLoader, TextModel};
    use crate::embedding::protocol::Embedder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- stub models + loaders (no ONNX) ----------------------------------

    /// A stub model yielding a fixed vector per input text.
    struct StubModel {
        vector: Vec<f32>,
    }

    impl TextModel for StubModel {
        fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| self.vector.clone()).collect())
        }
    }

    /// Records how many times a model was loaded (pins single-load semantics).
    struct RecordingLoader {
        loads: Arc<AtomicUsize>,
        vector: Vec<f32>,
    }

    impl ModelLoader for RecordingLoader {
        fn load(
            &self,
            _model_name: &str,
            _cache_dir: Option<&str>,
        ) -> anyhow::Result<Box<dyn TextModel>> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(StubModel {
                vector: self.vector.clone(),
            }))
        }
    }

    /// Fails to load, mimicking a missing `local-embed` dependency; the error
    /// message carries the install hint and the model stays unloaded.
    struct FailingLoader;

    impl ModelLoader for FailingLoader {
        fn load(
            &self,
            _model_name: &str,
            _cache_dir: Option<&str>,
        ) -> anyhow::Result<Box<dyn TextModel>> {
            anyhow::bail!(
                "FastEmbedEmbedder requires the 'local-embed' extra. \
                 Install with `uv sync --extra local-embed`."
            )
        }
    }

    /// Records the loader kwargs (pins cache-dir threading).
    struct CapturingLoader {
        seen_model: Arc<std::sync::Mutex<Option<String>>>,
        seen_cache: Arc<std::sync::Mutex<Option<Option<String>>>>,
    }

    impl ModelLoader for CapturingLoader {
        fn load(
            &self,
            model_name: &str,
            cache_dir: Option<&str>,
        ) -> anyhow::Result<Box<dyn TextModel>> {
            *self.seen_model.lock().unwrap() = Some(model_name.to_owned());
            *self.seen_cache.lock().unwrap() = Some(cache_dir.map(str::to_owned));
            Ok(Box::new(StubModel { vector: vec![0.1] }))
        }
    }

    fn with_loader(
        model_name: &str,
        cache_dir: Option<String>,
        loader: Arc<dyn ModelLoader>,
    ) -> FastEmbedEmbedder {
        FastEmbedEmbedder::with_loader(model_name, cache_dir, loader)
    }

    // --- metadata ----------------------------------------------------------

    #[test]
    fn name_carries_model() {
        let e = FastEmbedEmbedder::new("BAAI/bge-small-en-v1.5", None);
        assert_eq!(e.name(), "fastembed:BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn dim_known_for_bge_small() {
        assert_eq!(
            FastEmbedEmbedder::new("BAAI/bge-small-en-v1.5", None).dim(),
            384
        );
    }

    #[test]
    fn dim_known_for_bge_base() {
        assert_eq!(
            FastEmbedEmbedder::new("BAAI/bge-base-en-v1.5", None).dim(),
            768
        );
    }

    #[test]
    fn dim_zero_for_unknown_model_until_first_embed() {
        assert_eq!(FastEmbedEmbedder::new("custom/model", None).dim(), 0);
    }

    #[test]
    fn distinct_models_get_distinct_names() {
        let a = FastEmbedEmbedder::new("BAAI/bge-small-en-v1.5", None);
        let b = FastEmbedEmbedder::new("BAAI/bge-base-en-v1.5", None);
        assert_ne!(a.name(), b.name());
    }

    #[test]
    fn default_is_bge_small() {
        let e = FastEmbedEmbedder::default();
        assert_eq!(e.name(), "fastembed:BAAI/bge-small-en-v1.5");
        assert_eq!(e.dim(), 384);
    }

    // --- lazy load ---------------------------------------------------------

    #[tokio::test]
    async fn empty_input_returns_empty_without_loading() {
        let loads = Arc::new(AtomicUsize::new(0));
        let e = with_loader(
            "BAAI/bge-small-en-v1.5",
            None,
            Arc::new(RecordingLoader {
                loads: loads.clone(),
                vector: vec![0.1, 0.2, 0.3, 0.4],
            }),
        );
        let out = e.embed(&[]).await.unwrap();
        assert!(out.is_empty());
        // No texts → no model load.
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn missing_extra_surfaces_pointed_error() {
        let e = with_loader("BAAI/bge-small-en-v1.5", None, Arc::new(FailingLoader));
        let err = e
            .embed(&["any text".to_owned()])
            .await
            .expect_err("missing dependency surfaces an error");
        assert!(err.to_string().contains("local-embed"));
    }

    #[tokio::test]
    async fn load_is_idempotent_and_dim_probed() {
        let loads = Arc::new(AtomicUsize::new(0));
        let e = with_loader(
            "custom/test",
            None,
            Arc::new(RecordingLoader {
                loads: loads.clone(),
                vector: vec![0.1, 0.2, 0.3, 0.4],
            }),
        );
        let v1 = e.embed(&["alpha".to_owned()]).await.unwrap();
        let v2 = e
            .embed(&["beta".to_owned(), "gamma".to_owned()])
            .await
            .unwrap();
        // Two embed calls reuse the loaded model — exactly one load.
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(v1.len(), 1);
        assert_eq!(v2.len(), 2);
        // Dim discovered from the first probed vector.
        assert_eq!(e.dim(), 4);
    }

    #[tokio::test]
    async fn known_dim_preserved_after_embed() {
        let e = with_loader(
            "BAAI/bge-small-en-v1.5",
            None,
            Arc::new(RecordingLoader {
                loads: Arc::new(AtomicUsize::new(0)),
                // An oddly-shaped probe vector must NOT clobber the known dim.
                vector: vec![0.0; 999],
            }),
        );
        assert_eq!(e.dim(), 384); // seeded from the lookup table
        e.embed(&["x".to_owned()]).await.unwrap();
        assert_eq!(e.dim(), 384); // unchanged
    }

    #[tokio::test]
    async fn cache_dir_threaded_to_loader() {
        let seen_model = Arc::new(std::sync::Mutex::new(None));
        let seen_cache = Arc::new(std::sync::Mutex::new(None));
        let e = with_loader(
            "m",
            Some("/tmp/fastembed-test".to_owned()),
            Arc::new(CapturingLoader {
                seen_model: seen_model.clone(),
                seen_cache: seen_cache.clone(),
            }),
        );
        e.embed(&["x".to_owned()]).await.unwrap();
        assert_eq!(seen_model.lock().unwrap().as_deref(), Some("m"));
        assert_eq!(
            seen_cache.lock().unwrap().clone(),
            Some(Some("/tmp/fastembed-test".to_owned()))
        );
    }

    #[tokio::test]
    async fn cache_dir_omitted_when_unset() {
        let seen_cache = Arc::new(std::sync::Mutex::new(None));
        let e = with_loader(
            "m",
            None,
            Arc::new(CapturingLoader {
                seen_model: Arc::new(std::sync::Mutex::new(None)),
                seen_cache: seen_cache.clone(),
            }),
        );
        e.embed(&["x".to_owned()]).await.unwrap();
        // No cache dir pinned → loader sees None (fastembed's own default
        // `~/.cache/fastembed`).
        assert_eq!(seen_cache.lock().unwrap().clone(), Some(None));
    }

    // --- real backend (needs a ~130 MB model download) --------------------

    #[tokio::test]
    #[ignore = "needs native runtime + model download (onnxruntime, BGE-small)"]
    async fn real_bge_small_embeds() {
        let e = FastEmbedEmbedder::new("BAAI/bge-small-en-v1.5", None);
        let out = e
            .embed(&["hello world".to_owned(), "a second sentence".to_owned()])
            .await
            .expect("real fastembed embed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 384);
        assert_eq!(e.dim(), 384);
    }
}
