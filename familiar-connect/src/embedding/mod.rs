//! Embedder seam + factory registry with built-in backends
//! (subsystem 04).
//!
//! Public surface: the [`Embedder`]
//! trait, the [`HashEmbedder`] built-in, the [`EmbedderRegistry`] builder, and
//! the [`known_embedders`] / [`create_embedder`] convenience functions. The
//! optional `fastembed` ONNX backend is Layer 2 (feature `local-embed`).

pub mod factory;
pub mod fastembed;
pub mod hash;
pub mod protocol;

pub use factory::{EmbedderFactory, EmbedderRegistry, create_embedder, known_embedders};
pub use hash::HashEmbedder;
pub use protocol::Embedder;

/// FastEmbed model name → native output dim. Sorted by name.
///
/// Static metadata — no ONNX, no download — so it compiles without
/// `local-embed` and config validation (02) can cross-check
/// `[providers.embedding].dim` against the selected model. The single source of
/// truth: `fastembed::known_dim` reads it too. A model absent here has an
/// unknowable dim until the first real vector probes it.
pub const FASTEMBED_NATIVE_DIMS: &[(&str, usize)] = &[
    ("BAAI/bge-base-en-v1.5", 768),
    ("BAAI/bge-large-en-v1.5", 1024),
    ("BAAI/bge-small-en-v1.5", 384),
    ("intfloat/e5-small-v2", 384),
    ("intfloat/multilingual-e5-small", 384),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
];

/// Native dim of `model_name`, or `None` when unmapped.
#[must_use]
pub fn fastembed_native_dim(model_name: &str) -> Option<usize> {
    FASTEMBED_NATIVE_DIMS
        .iter()
        .find_map(|(name, dim)| (*name == model_name).then_some(*dim))
}

/// Errors from the embedding subsystem (one `thiserror` enum per
/// subsystem; byte-stable messages are test contracts).
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// `HashEmbedder` requires `dim >= 8`. The message contains `>= 8`.
    #[error("HashEmbedder dim must be >= 8, got {0}")]
    DimTooSmall(i64),

    /// The configured backend is not registered. The message names the sorted,
    /// comma-joined valid list (or `(none)`).
    #[error("unknown embedding backend '{backend}'; valid: {valid}")]
    UnknownBackend {
        /// The unrecognised backend name from config.
        backend: String,
        /// Sorted, comma-joined registered names (or `(none)`).
        valid: String,
    },

    /// The `fastembed` backend was selected without the `local-embed` feature.
    #[error(
        "embedding backend 'fastembed' requires the 'local-embed' feature. \
         Rebuild with `cargo build --release --features local-embed`."
    )]
    FastembedMissing,
}
