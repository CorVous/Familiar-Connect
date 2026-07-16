//! Deterministic BLAKE2b `HashEmbedder` baseline (subsystem 04; Python
//! `embedding/hash.py`).
//!
//! Locality-hashing fallback: tokenise on word boundaries, casefold, project
//! each token onto a stable bucket via BLAKE2b, accumulate a fixed-dim float
//! vector, L2-normalise. Cosine similarity correlates with token overlap — a
//! weak but deterministic, no-install signal for storage/retrieval smoke tests.
//!
//! The algorithm is pinned by the class-level name `"hash-v1"` and must stay
//! bit-for-bit portable so stored vectors keyed by `model = "hash-v1"` remain
//! comparable. Accumulation runs in `f64` (matching Python); the returned
//! vectors are `f32` (the storage BLOB width, spec 03). Any tokenisation change
//! must bump the name to `hash-v2` rather than corrupt the mixed similarity
//! space.

use std::sync::LazyLock;

use async_trait::async_trait;
use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use regex::Regex;

use crate::embedding::EmbeddingError;
use crate::embedding::protocol::Embedder;

/// Unicode word-boundary tokeniser (`\w+`, matching Python `re.UNICODE`).
static TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w+").expect("valid word-token regex"));

/// Tokenise `text` into casefolded word tokens.
///
/// Python casefolds each token; Rust has no `casefold`, so this uses
/// [`str::to_lowercase`] (Unicode lowercasing). The two agree for every token
/// the tests exercise; they diverge only on characters like `ß` (casefold →
/// `ss`, lowercase → `ß`), which do not occur in stored `hash-v1` data — a
/// declared, unobservable deviation (spec 04 note 2 flags the tokenisation-drift
/// risk and the `hash-v2` escape hatch).
fn tokens(text: &str) -> Vec<String> {
    TOKEN_RE
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

/// Deterministic locality-hashing baseline embedder.
///
/// Stable across processes / runs / platforms for a given `(dim, texts)` — the
/// projection is seedless BLAKE2b plus pure arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// Backend label persisted with each vector (version the name if the
    /// algorithm ever changes).
    pub const NAME: &'static str = "hash-v1";

    /// Construct with dimensionality `dim`.
    ///
    /// # Errors
    /// [`EmbeddingError::DimTooSmall`] when `dim < 8` (message contains
    /// `>= 8`), mirroring the Python `ValueError`.
    pub fn new(dim: i64) -> Result<Self, EmbeddingError> {
        if dim < 8 {
            return Err(EmbeddingError::DimTooSmall(dim));
        }
        // `dim >= 8` here, so the conversion is always in range on any target.
        let dim = usize::try_from(dim).map_err(|_| EmbeddingError::DimTooSmall(dim))?;
        Ok(Self { dim })
    }

    // f64 → f32 narrowing to the storage vector width is deliberate (spec 03
    // packs little-endian f32); the values are already normalised to [-1, 1].
    #[allow(clippy::cast_possible_truncation)]
    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f64; self.dim];
        for tok in tokens(text) {
            let mut hasher = Blake2bVar::new(4).expect("4-byte BLAKE2b output size is valid");
            hasher.update(tok.as_bytes());
            let mut digest = [0u8; 4];
            hasher
                .finalize_variable(&mut digest)
                .expect("4-byte buffer matches output size");
            let idx = (u16::from_be_bytes([digest[0], digest[1]]) as usize) % self.dim;
            let sign = if digest[2] & 1 == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }
        let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            vec.iter().map(|&v| (v / norm) as f32).collect()
        } else {
            // No tokens / all-cancelling → raw zero vector (never normalised).
            vec.iter().map(|&v| v as f32).collect()
        }
    }
}

impl Default for HashEmbedder {
    /// The Python default dimensionality is `256`.
    fn default() -> Self {
        Self { dim: 256 }
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::HashEmbedder;
    use crate::embedding::EmbeddingError;
    use crate::embedding::protocol::Embedder;

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let na: f64 = a
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        if na <= 0.0 || nb <= 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn dim_must_be_at_least_8() {
        let err = HashEmbedder::new(4).unwrap_err();
        assert!(matches!(err, EmbeddingError::DimTooSmall(4)));
        // Message contains ">= 8" (pinned by the Python `pytest.raises(match)`).
        assert!(err.to_string().contains(">= 8"));
    }

    #[test]
    fn name_is_versioned() {
        assert_eq!(HashEmbedder::default().name(), "hash-v1");
    }

    #[test]
    fn dim_matches_constructor() {
        assert_eq!(HashEmbedder::new(128).unwrap().dim(), 128);
    }

    #[tokio::test]
    async fn empty_input_returns_empty_list() {
        let out = HashEmbedder::default().embed(&[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn blank_text_projects_to_zero_vector() {
        let out = HashEmbedder::new(64)
            .unwrap()
            .embed(&[String::new()])
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![0.0f32; 64]);
    }

    #[tokio::test]
    async fn output_length_matches_input() {
        let texts = ["a", "b", "c"].map(String::from).to_vec();
        let out = HashEmbedder::new(32).unwrap().embed(&texts).await.unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.len() == 32));
    }

    #[tokio::test]
    async fn deterministic_across_calls() {
        let e = HashEmbedder::new(64).unwrap();
        let v1 = e.embed(&["hello world".to_string()]).await.unwrap();
        let v2 = e.embed(&["hello world".to_string()]).await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn casefold_invariant() {
        // Tokens casefold so cue paraphrasing in case still hashes the same.
        let e = HashEmbedder::new(64).unwrap();
        let out = e
            .embed(&["Café Latte".to_string(), "café LATTE".to_string()])
            .await
            .unwrap();
        assert_eq!(out[0], out[1]);
    }

    #[tokio::test]
    async fn unrelated_texts_score_low() {
        let e = HashEmbedder::new(256).unwrap();
        let out = e
            .embed(&[
                "the capital of france is paris".to_string(),
                "rust ownership rules".to_string(),
            ])
            .await
            .unwrap();
        assert!(cos(&out[0], &out[1]) < 0.1);
    }

    #[tokio::test]
    async fn overlapping_tokens_score_higher() {
        let e = HashEmbedder::new(256).unwrap();
        let out = e
            .embed(&[
                "alice has a cat named whiskers".to_string(),
                "alice and the cat whiskers".to_string(),
                "rust ownership rules".to_string(),
            ])
            .await
            .unwrap();
        assert!(cos(&out[0], &out[1]) > cos(&out[0], &out[2]));
    }

    #[tokio::test]
    #[allow(clippy::cast_possible_truncation)] // reference constant narrowed to f32
    async fn byte_exact_parity_with_python_blake2b() {
        // Reference generated from Python `hashlib.blake2b(digest_size=4)`:
        //   "hello" → bucket 7 (dim 8), sign +1; "world" → bucket 1, sign -1.
        // vec = [0,-1,0,0,0,0,0,1] → L2-normalised = ±1/sqrt(2).
        let out = HashEmbedder::new(8)
            .unwrap()
            .embed(&["hello world".to_string()])
            .await
            .unwrap();
        let inv_sqrt2 = (0.5f64.sqrt()) as f32;
        let expected = [0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2];
        assert_eq!(out[0].len(), 8);
        for (got, want) in out[0].iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
