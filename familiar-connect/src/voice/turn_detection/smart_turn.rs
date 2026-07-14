//! Smart Turn v3 wrapper — semantic utterance-completion classifier (subsystem
//! 09; Python `voice/turn_detection/smart_turn.py`).
//!
//! Pipecat's [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) is a
//! small wav2vec2-derived model classifying whether the current audio buffer
//! ends on a turn-complete boundary. Where TEN-VAD answers "is anyone speaking
//! right now?", Smart Turn answers "did the speaker actually finish?". It is
//! trained on filler-word audio that STT routinely drops — why it beats
//! transcription-based endpointing.
//!
//! Stateless: feed buffered utterance audio after the VAD reports silence; the
//! classifier returns a completion probability. Output handles both common
//! export shapes (spec 09 §24):
//!
//! - 2-class logits `[incomplete, complete]` → numerically-stable softmax, take
//!   class 1;
//! - single sigmoid logit `[complete_score]` → `1/(1+exp(-x))`.
//!
//! The ONNX runtime ([`ort`]) is gated behind the `local-turn` feature; the pure
//! preprocessing (int16 → f32/32768, 16 s tail-truncation) and postprocessing
//! (softmax/sigmoid) are always compiled and injected-model testable via the
//! [`SmartTurnModel`] seam (DESIGN §4.8).

use std::path::Path;

/// Errors from the Smart Turn classifier.
#[derive(Debug, thiserror::Error)]
pub enum SmartTurnError {
    /// The model emitted a logits head whose last dim is neither 1 nor 2.
    #[error("unsupported logits shape [1, {0}]; expected last dim 1 or 2")]
    UnsupportedLogitsShape(usize),
    /// The ONNX runtime backend is not compiled in (`local-turn` off).
    #[error("SmartTurnDetector requires the 'local-turn' extra (ONNX runtime)")]
    MissingBackend,
    /// The ONNX model failed to load or run.
    #[error("Smart Turn model error: {0}")]
    Model(String),
}

/// Completion-classifier seam the endpointer consumes.
///
/// Shared across users (stateless beyond the loaded model). The endpointer runs
/// it off the reactor via `spawn_blocking`, so implementations may be slow
/// (wav2vec2 over ≤16 s of audio).
pub trait SmartTurn: Send + Sync {
    /// True when the buffered utterance ends on a complete turn boundary.
    fn is_turn_complete(&self, pcm_audio: &[u8]) -> bool;
}

/// The ONNX inference seam: run the classifier over `[1, n]` float32 samples and
/// return the logits row (length 1 = sigmoid head, 2 = softmax head).
///
/// Injected so the pure pre/post-processing is testable without an ONNX runtime;
/// the production impl is the `local-turn`-gated [`ort`]-backed session.
pub trait SmartTurnModel: Send + Sync {
    /// Run inference on `samples` (the `[1, n]` waveform row); return the logits.
    fn run(&self, samples: &[f32]) -> Vec<f32>;
}

/// Semantic turn-completion classifier (Pipecat Smart Turn v3 ONNX).
pub struct SmartTurnDetector {
    threshold: f32,
    max_samples: usize,
    model: Box<dyn SmartTurnModel>,
}

impl SmartTurnDetector {
    /// Default input sample rate.
    pub const DEFAULT_SAMPLE_RATE: usize = 16000;
    /// Default max classified window (seconds) — the tail is kept.
    pub const DEFAULT_MAX_DURATION_S: f64 = 16.0;

    /// Build a detector over an injected inference model.
    ///
    /// `max_duration_s * sample_rate` caps the classified window; longer buffers
    /// keep only the most-recent samples (turn-end semantics live in the tail).
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn with_model(
        model: Box<dyn SmartTurnModel>,
        threshold: f32,
        sample_rate: usize,
        max_duration_s: f64,
    ) -> Self {
        let max_samples = (max_duration_s * sample_rate as f64) as usize;
        Self {
            threshold,
            max_samples,
            model,
        }
    }

    /// Build a production detector from an ONNX model file.
    ///
    /// # Errors
    /// [`SmartTurnError::MissingBackend`] when the `local-turn` feature is off;
    /// [`SmartTurnError::Model`] when the ONNX session fails to load.
    pub fn from_path(model_path: &Path, threshold: f32) -> Result<Self, SmartTurnError> {
        Self::from_path_with(
            model_path,
            threshold,
            Self::DEFAULT_SAMPLE_RATE,
            Self::DEFAULT_MAX_DURATION_S,
        )
    }

    /// [`from_path`](Self::from_path) with explicit rate / max-duration knobs.
    ///
    /// # Errors
    /// See [`from_path`](Self::from_path).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::missing_const_for_fn,
        unused_variables
    )]
    pub fn from_path_with(
        model_path: &Path,
        threshold: f32,
        sample_rate: usize,
        max_duration_s: f64,
    ) -> Result<Self, SmartTurnError> {
        #[cfg(feature = "local-turn")]
        {
            let model = ort_model::OrtSmartTurnModel::load(model_path)?;
            Ok(Self::with_model(
                Box::new(model),
                threshold,
                sample_rate,
                max_duration_s,
            ))
        }
        #[cfg(not(feature = "local-turn"))]
        {
            Err(SmartTurnError::MissingBackend)
        }
    }

    /// Configured completion threshold.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Run the classifier; return the `[0, 1]` is-complete probability.
    ///
    /// # Errors
    /// [`SmartTurnError::UnsupportedLogitsShape`] when the model's logits head is
    /// neither a 2-class softmax nor a single sigmoid logit.
    pub fn completion_probability(&self, pcm_audio: &[u8]) -> Result<f32, SmartTurnError> {
        let mut audio: Vec<f32> = pcm_audio
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect();
        if audio.len() > self.max_samples {
            // Keep the most-recent window — turn-end semantics live there.
            audio = audio.split_off(audio.len() - self.max_samples);
        }
        let logits = self.model.run(&audio);
        match logits.len() {
            2 => {
                // Numerically-stable softmax over [incomplete, complete]; class 1.
                let m = logits[0].max(logits[1]);
                let e0 = (logits[0] - m).exp();
                let e1 = (logits[1] - m).exp();
                Ok(e1 / (e0 + e1))
            }
            1 => Ok(1.0 / (1.0 + (-logits[0]).exp())),
            n => Err(SmartTurnError::UnsupportedLogitsShape(n)),
        }
    }

    /// Whether the buffer classifies complete (`probability >= threshold`).
    ///
    /// # Errors
    /// See [`completion_probability`](Self::completion_probability).
    pub fn is_complete(&self, pcm_audio: &[u8]) -> Result<bool, SmartTurnError> {
        Ok(self.completion_probability(pcm_audio)? >= self.threshold)
    }
}

impl SmartTurn for SmartTurnDetector {
    fn is_turn_complete(&self, pcm_audio: &[u8]) -> bool {
        // Parity with Python (smart_turn.py:94-98): `is_complete` ->
        // `completion_probability` RAISES `ValueError` on an unexpected logits
        // shape, and via `asyncio.to_thread` that exception propagates out of the
        // endpointer's `_on_vad_frame`/`feed_audio` (endpointer.py:176-178). A
        // malformed logits head is a model-export mismatch (a deployment fault),
        // so surface it loudly rather than silently classifying every turn as
        // incomplete and stranding it in POST_INCOMPLETE.
        self.is_complete(pcm_audio)
            .expect("Smart Turn model emitted an unsupported logits shape")
    }
}

#[cfg(feature = "local-turn")]
mod ort_model {
    //! `ort`-backed [`SmartTurnModel`] (feature `local-turn`).
    //!
    //! Feeds the waveform as an `[1, n]` float32 tensor under the graph's first
    //! input name (Pipecat exports use `input_values`, the Wav2Vec2 convention)
    //! and returns the logits row.

    use super::{SmartTurnError, SmartTurnModel};
    use std::path::Path;
    use std::sync::Mutex;

    use ort::session::Session;
    use ort::value::Tensor;

    /// ONNX Smart Turn session + cached first-input name.
    pub struct OrtSmartTurnModel {
        session: Mutex<Session>,
        input_name: String,
    }

    impl OrtSmartTurnModel {
        /// Load the ONNX model at `model_path`.
        pub fn load(model_path: &Path) -> Result<Self, SmartTurnError> {
            let session = Session::builder()
                .and_then(|mut b| b.commit_from_file(model_path))
                .map_err(|e| SmartTurnError::Model(e.to_string()))?;
            let input_name = session
                .inputs()
                .first()
                .map_or_else(|| "input_values".to_owned(), |i| i.name().to_owned());
            Ok(Self {
                session: Mutex::new(session),
                input_name,
            })
        }
    }

    impl SmartTurnModel for OrtSmartTurnModel {
        // The `Mutex` guard is held across `run()` and the output extraction:
        // `SessionOutputs` and the tensor views taken from it borrow the session,
        // so the guard cannot be tightened past the extraction.
        #[allow(
            clippy::significant_drop_tightening,
            reason = "outputs borrow the guarded session; it must live through extraction"
        )]
        fn run(&self, samples: &[f32]) -> Vec<f32> {
            let Ok(len) = i64::try_from(samples.len()) else {
                return Vec::new();
            };
            let Ok(tensor) = Tensor::from_array(([1_i64, len], samples.to_vec())) else {
                return Vec::new();
            };
            let mut session = self.session.lock().expect("smart turn session poisoned");
            let Ok(outputs) = session.run(ort::inputs![self.input_name.as_str() => tensor]) else {
                return Vec::new();
            };
            let Some(first) = outputs.values().next() else {
                return Vec::new();
            };
            match first.try_extract_tensor::<f32>() {
                // `(shape, data)` — the logits row is the trailing dim.
                Ok((shape, data)) => {
                    let last = shape
                        .last()
                        .map_or(data.len(), |d| usize::try_from(*d).unwrap_or(data.len()));
                    data.iter().rev().take(last).rev().copied().collect()
                }
                Err(_) => Vec::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SmartTurn, SmartTurnDetector, SmartTurnError, SmartTurnModel};
    use std::sync::{Arc, Mutex};

    /// Model that returns fixed logits and records the samples it received.
    struct ScriptedModel {
        logits: Vec<f32>,
        last: Arc<Mutex<Vec<f32>>>,
    }

    impl SmartTurnModel for ScriptedModel {
        fn run(&self, samples: &[f32]) -> Vec<f32> {
            *self.last.lock().unwrap() = samples.to_vec();
            self.logits.clone()
        }
    }

    fn detector(logits: Vec<f32>) -> (SmartTurnDetector, Arc<Mutex<Vec<f32>>>) {
        let last = Arc::new(Mutex::new(Vec::new()));
        let model = ScriptedModel {
            logits,
            last: Arc::clone(&last),
        };
        let det = SmartTurnDetector::with_model(Box::new(model), 0.5, 16000, 16.0);
        (det, last)
    }

    fn pcm(samples: usize, amplitude: i16) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples * 2);
        for _ in 0..samples {
            out.extend_from_slice(&amplitude.to_le_bytes());
        }
        out
    }

    // --- output-shape handling (test_smart_turn.py TestCompletionProbability)

    #[test]
    fn softmax_2class_logits() {
        // logits [0, 2] → softmax class-1 ≈ 0.881.
        let (det, _) = detector(vec![0.0, 2.0]);
        let prob = det.completion_probability(&pcm(8000, 1000)).unwrap();
        assert!((prob - 0.881).abs() < 1e-3, "prob={prob}");
    }

    #[test]
    fn sigmoid_1class_logit() {
        // sigmoid(1) ≈ 0.731.
        let (det, _) = detector(vec![1.0]);
        let prob = det.completion_probability(&pcm(8000, 1000)).unwrap();
        assert!((prob - 0.731).abs() < 1e-3, "prob={prob}");
    }

    #[test]
    fn input_is_float32_normalised() {
        let (det, last) = detector(vec![2.0, 0.0]);
        det.completion_probability(&pcm(1000, 16384)).unwrap();
        let samples = last.lock().unwrap().clone();
        assert_eq!(samples.len(), 1000);
        assert!((samples[0] - 16384.0 / 32768.0).abs() < 1e-4);
    }

    #[test]
    fn truncates_to_max_duration() {
        // max 1.0 s at 16 kHz = 16000 samples; 2 s in → most-recent 16000 out.
        let last = Arc::new(Mutex::new(Vec::new()));
        let model = ScriptedModel {
            logits: vec![2.0, 0.0],
            last: Arc::clone(&last),
        };
        let det = SmartTurnDetector::with_model(Box::new(model), 0.5, 16000, 1.0);
        det.completion_probability(&pcm(32000, 1000)).unwrap();
        assert_eq!(last.lock().unwrap().len(), 16000);
    }

    // --- threshold (TestThreshold) -----------------------------------------

    #[test]
    fn is_complete_above_threshold() {
        let (det, _) = detector(vec![0.0, 5.0]);
        assert!(det.is_complete(&pcm(1000, 1000)).unwrap());
    }

    #[test]
    fn is_not_complete_below_threshold() {
        let (det, _) = detector(vec![5.0, 0.0]);
        assert!(!det.is_complete(&pcm(1000, 1000)).unwrap());
    }

    // --- rejection (TestRejection) -----------------------------------------

    #[test]
    fn rejects_unexpected_output_shape() {
        let (det, _) = detector(vec![0.1, 0.2, 0.3]);
        let err = det.completion_probability(&pcm(1000, 1000)).unwrap_err();
        assert!(matches!(err, SmartTurnError::UnsupportedLogitsShape(3)));
        assert!(err.to_string().contains("logits shape"));
    }

    // The endpointer seam (`SmartTurn::is_turn_complete`) must NOT swallow the
    // shape error as a `false` verdict — Python's `is_complete` raises through
    // `asyncio.to_thread` and crashes the pump; we mirror that with a panic.
    #[test]
    #[should_panic(expected = "unsupported logits shape")]
    fn is_turn_complete_panics_on_unexpected_shape() {
        let (det, _) = detector(vec![0.1, 0.2, 0.3]);
        let _ = det.is_turn_complete(&pcm(1000, 1000));
    }
}
