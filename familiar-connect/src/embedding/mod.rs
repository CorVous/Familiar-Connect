//! Embedder seam + factory registry with built-in backends
//! (subsystem 04; Python `embedding/`).
//!
//! Public surface (mirrors Python `embedding/__init__.py`): the [`Embedder`]
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

/// Errors from the embedding subsystem (DESIGN §4.1 — one `thiserror` enum per
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

    /// The `fastembed` backend was selected without the `local-embed` extra.
    #[error(
        "embedding backend 'fastembed' requires the 'local-embed' extra. \
         Install with `uv sync --extra local-embed`."
    )]
    FastembedMissing,
}
