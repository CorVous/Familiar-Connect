//! Embedder factory registry + `known_embedders` (subsystem 04; Python
//! `embedding/factory.py`).
//!
//! DESIGN D14 / §4.8: explicit [`EmbedderRegistry::with_builtins`] builder plus
//! [`EmbedderRegistry::register`], **not** an import-time global dict. Names are
//! kept sorted (a `BTreeMap`) and the error strings are byte-exact so config
//! validation (02, which consults [`known_embedders`]) and the tests match on
//! them (`"unknown embedding backend"`, `"local-embed"`).
//!
//! Built-ins:
//! * `off` → `None` (disables the seam end to end).
//! * `hash` → [`HashEmbedder`](crate::embedding::hash::HashEmbedder).
//! * `fastembed` → fails with the `local-embed` install hint. The real ONNX
//!   backend is Layer 2 (feature `local-embed`, `embedding::fastembed`); the
//!   wiring injects it via `register("fastembed", …)` when the extra is present,
//!   exactly the Python "re-registration overwrites" seam. Absent the extra a
//!   deploy that selects `fastembed` refuses to start rather than crashing
//!   mid-turn.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::config::EmbeddingConfig;
use crate::embedding::EmbeddingError;
use crate::embedding::hash::HashEmbedder;
use crate::embedding::protocol::Embedder;

/// A backend factory: turns config into an optional shared embedder, or an
/// error (unavailable extra / bad dimensionality).
///
/// `Ok(None)` is the `off` outcome — the seam is disabled. The `Arc` lets one
/// instance be shared across assemblers and the projector context (the Python
/// wiring shares a single embedder from startup).
pub type EmbedderFactory = Arc<
    dyn Fn(&EmbeddingConfig) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> + Send + Sync,
>;

/// Name-indexed embedder factory registry.
///
/// Keys stay sorted (`BTreeMap`), so [`known_embedders`](EmbedderRegistry::known_embedders)
/// and the valid-name list in the unknown-backend error are deterministic.
#[derive(Clone)]
pub struct EmbedderRegistry {
    factories: BTreeMap<String, EmbedderFactory>,
}

impl EmbedderRegistry {
    /// An empty registry (no built-ins).
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: BTreeMap::new(),
        }
    }

    /// A registry with the three built-in backends registered.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register("off", Arc::new(|_config| Ok(None)));
        registry.register("hash", Arc::new(hash_factory));
        registry.register("fastembed", Arc::new(fastembed_factory));
        // The real ONNX backend (feature `local-embed`) overrides the fail-fast
        // stub above via silent re-registration — the "wiring injects it" step the
        // module docs describe. Without this, `backend = "fastembed"` errors even
        // when `local-embed` is compiled in.
        #[cfg(feature = "local-embed")]
        registry.register("fastembed", Arc::new(real_fastembed_factory));
        registry
    }

    /// Register `factory` under `name`; re-registration overwrites silently
    /// (matching the Python module-global registry).
    pub fn register(&mut self, name: impl Into<String>, factory: EmbedderFactory) {
        self.factories.insert(name.into(), factory);
    }

    /// Registered backend names (built-ins + third-party additions), sorted.
    ///
    /// Consumed by config parsing (02) at load to validate
    /// `[providers.embedding].backend`.
    #[must_use]
    pub fn known_embedders(&self) -> BTreeSet<String> {
        self.factories.keys().cloned().collect()
    }

    /// Instantiate the embedder selected by `config.backend`.
    ///
    /// # Errors
    /// [`EmbeddingError::UnknownBackend`] when `config.backend` is not
    /// registered (message names the sorted valid list), plus whatever the
    /// selected factory returns (`DimTooSmall`, `FastembedMissing`).
    pub fn create(
        &self,
        config: &EmbeddingConfig,
    ) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
        self.factories.get(&config.backend).map_or_else(
            || {
                Err(EmbeddingError::UnknownBackend {
                    backend: config.backend.clone(),
                    valid: self.valid_names(),
                })
            },
            |factory| factory(config),
        )
    }

    /// The sorted, comma-joined valid-name list (or `"(none)"` when empty).
    fn valid_names(&self) -> String {
        if self.factories.is_empty() {
            "(none)".to_string()
        } else {
            self.factories
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

impl Default for EmbedderRegistry {
    /// The default registry carries the built-ins.
    fn default() -> Self {
        Self::with_builtins()
    }
}

/// `hash` built-in: a [`HashEmbedder`] sized by `config.dim`.
fn hash_factory(config: &EmbeddingConfig) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
    let embedder = HashEmbedder::new(config.dim)?;
    Ok(Some(Arc::new(embedder)))
}

/// `fastembed` built-in: fail-fast with the `local-embed` install hint.
///
/// Mirrors the Python import-probe: without the extra, refuse to start. The real
/// backend (Layer 2, feature `local-embed`) is injected by the wiring via
/// [`EmbedderRegistry::register`].
fn fastembed_factory(
    _config: &EmbeddingConfig,
) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
    Err(EmbeddingError::FastembedMissing)
}

/// `fastembed` backend when the `local-embed` extra is compiled: constructs the
/// real ONNX [`FastEmbedEmbedder`], overriding [`fastembed_factory`]'s stub via
/// re-registration in [`EmbedderRegistry::with_builtins`]. An empty configured
/// model falls back to the default (BGE-small).
#[cfg(feature = "local-embed")]
fn real_fastembed_factory(
    config: &EmbeddingConfig,
) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
    let model = if config.fastembed_model.is_empty() {
        crate::embedding::fastembed::DEFAULT_MODEL_NAME.to_owned()
    } else {
        config.fastembed_model.clone()
    };
    Ok(Some(Arc::new(
        crate::embedding::fastembed::FastEmbedEmbedder::new(
            model,
            config.fastembed_cache_dir.clone(),
        ),
    )))
}

/// Built-in registered names, sorted (convenience over a fresh
/// [`EmbedderRegistry::with_builtins`]).
///
/// Mirrors the Python module-level `known_embedders()`; config parsing (02)
/// injects this set.
#[must_use]
pub fn known_embedders() -> BTreeSet<String> {
    EmbedderRegistry::with_builtins().known_embedders()
}

/// Instantiate the embedder selected by `config.backend` from the built-ins.
///
/// Mirrors the Python module-level `create_embedder()`; the wiring uses a shared
/// [`EmbedderRegistry`] when third-party backends need registering.
///
/// # Errors
/// See [`EmbedderRegistry::create`].
pub fn create_embedder(
    config: &EmbeddingConfig,
) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
    EmbedderRegistry::with_builtins().create(config)
}

#[cfg(test)]
mod tests {
    use super::{EmbedderRegistry, create_embedder, known_embedders};
    use crate::config::EmbeddingConfig;
    use crate::embedding::EmbeddingError;

    fn config(backend: &str, dim: i64) -> EmbeddingConfig {
        EmbeddingConfig {
            backend: backend.to_string(),
            dim,
            ..EmbeddingConfig::default()
        }
    }

    #[test]
    fn known_embedders_includes_builtins() {
        let known = known_embedders();
        assert!(known.contains("off"));
        assert!(known.contains("hash"));
        assert!(known.contains("fastembed"));
    }

    #[test]
    fn off_returns_none() {
        let out = create_embedder(&config("off", 256)).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn hash_returns_hash_embedder() {
        let out = create_embedder(&config("hash", 128)).unwrap().unwrap();
        assert_eq!(out.name(), "hash-v1");
        assert_eq!(out.dim(), 128);
    }

    #[test]
    fn unknown_backend_raises() {
        let err = create_embedder(&config("nonexistent", 256))
            .err()
            .expect("unknown backend errors");
        let msg = err.to_string();
        assert!(msg.contains("unknown embedding backend"));
        // The valid list is sorted + comma-joined.
        assert!(msg.contains("fastembed, hash, off"));
    }

    #[cfg(feature = "local-embed")]
    #[test]
    fn fastembed_with_extra_resolves_to_the_real_backend() {
        // With the extra compiled, `with_builtins` overrides the stub with the
        // real ONNX backend, so `fastembed` resolves to a live embedder (lazy
        // model load — construction here does not download).
        let out = create_embedder(&config("fastembed", 384))
            .expect("fastembed backend resolves when local-embed is compiled");
        let embedder = out.expect("fastembed is not the `off` backend");
        assert_eq!(embedder.name(), "fastembed:BAAI/bge-small-en-v1.5");
    }

    #[cfg(not(feature = "local-embed"))]
    #[test]
    fn fastembed_without_extra_fails_at_load() {
        let err = create_embedder(&config("fastembed", 256))
            .err()
            .expect("fastembed without the extra errors");
        assert!(matches!(err, EmbeddingError::FastembedMissing));
        assert!(err.to_string().contains("local-embed"));
    }

    #[test]
    fn empty_registry_reports_none_valid() {
        let err = EmbedderRegistry::new()
            .create(&config("hash", 256))
            .err()
            .expect("empty registry has no backends");
        assert!(err.to_string().contains("(none)"));
    }

    #[test]
    fn register_overrides_and_extends() {
        let mut registry = EmbedderRegistry::with_builtins();
        // A third-party backend registered after construction is visible and
        // usable (the dynamic extension seam).
        registry.register("custom", std::sync::Arc::new(|_cfg| Ok(None)));
        assert!(registry.known_embedders().contains("custom"));
        assert!(registry.create(&config("custom", 256)).unwrap().is_none());
    }
}
