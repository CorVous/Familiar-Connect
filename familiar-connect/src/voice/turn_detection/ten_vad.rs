//! TEN-VAD wrapper — voice activity detection on 16 kHz mono PCM (subsystem 09;
//! Python `voice/turn_detection/ten_vad.py`).
//!
//! Wraps Agora's TEN-VAD. The native shared library + bundled ONNX arrive as a
//! `ten-vad-sys` FFI crate (DESIGN D2) so `unsafe` stays out of this crate; until
//! then the native handle is an **injected seam** ([`NativeTenVad`]) and
//! [`TenVad::new`] degrades to [`VadError::MissingBackend`].
//!
//! Feed 16 ms (256-sample) or 10 ms (160-sample) frames of 16 kHz mono int16
//! PCM; get back a probability. The native binary verdict flag is ignored — the
//! wrapper re-thresholds the returned probability so the same handle can be
//! re-tuned without a rebuild (spec 09 §23). Stateful: the C handle accumulates
//! across calls; [`reset`](TenVad::reset) rebuilds it so per-utterance state
//! doesn't leak.
//!
//! [`Vad`] is the narrow seam the
//! [`UtteranceEndpointer`](super::endpointer::UtteranceEndpointer) consumes — a
//! scripted double satisfies it in ~5 lines (DESIGN §4.8).

use std::sync::Arc;

/// TEN-VAD native hop sizes at 16 kHz: 160 (10 ms) or 256 (16 ms).
pub const VALID_HOP_SIZES_16K: [usize; 2] = [160, 256];

/// Errors from the VAD wrapper.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VadError {
    /// TEN-VAD only runs at 16 kHz.
    #[error("sample_rate must be 16000 Hz; got {0}")]
    SampleRate(i64),
    /// Hop size not in {160, 256}.
    #[error("hop_size {hop} not supported at 16 kHz; valid: {valid}")]
    HopSize {
        /// The rejected hop size.
        hop: usize,
        /// Comma-joined valid hop sizes.
        valid: String,
    },
    /// A chunk did not carry exactly `hop_size` int16 samples.
    #[error("expected {expected} samples, got {got}")]
    WrongChunkLength {
        /// Samples the wrapper requires (`hop_size`).
        expected: usize,
        /// Samples actually supplied.
        got: usize,
    },
    /// The native TEN-VAD backend is not compiled in (no `ten-vad-sys` yet).
    #[error("TenVad requires the 'local-turn' extra (native ten-vad backend)")]
    MissingBackend,
}

/// Voice-activity seam the endpointer state machine consumes.
///
/// Deliberately narrow: the endpointer only ever asks "is this 16 ms frame
/// speech?" and "clear accumulated state". A scripted test double implements it
/// trivially; [`TenVad`] is the production impl.
pub trait Vad: Send {
    /// True when `frame` (exactly `hop_size` int16 samples, s16le) is speech at
    /// the configured threshold. The endpointer only ever feeds well-formed
    /// 256-sample frames; the production [`TenVad`] impl panics on a wrong-length
    /// frame (parity with Python raising `ValueError`), never a silent `false`.
    fn is_speech(&mut self, frame: &[u8]) -> bool;

    /// Drop accumulated state — call at an utterance boundary.
    fn reset(&mut self);
}

/// The native TEN-VAD handle seam (Agora ten-vad; arrives as `ten-vad-sys`, D2).
///
/// Mirrors the native `process(frame) -> (probability, flag)`; the wrapper reads
/// only the probability. A `Box<dyn NativeTenVad>` is rebuilt on
/// [`TenVad::reset`] via the factory the wrapper was constructed with.
pub trait NativeTenVad: Send {
    /// Probability that `samples` (exactly `hop_size` int16 samples) is speech.
    fn process(&mut self, samples: &[i16]) -> f32;
}

/// Factory that rebuilds a native handle from `(hop_size, threshold)`.
type NativeFactory = Arc<dyn Fn(usize, f32) -> Box<dyn NativeTenVad> + Send + Sync>;

/// In-process voice activity detector (Agora TEN-VAD).
///
/// Construct with [`TenVad::new`] (production — native backend, currently
/// [`VadError::MissingBackend`] until `ten-vad-sys` lands) or
/// [`TenVad::with_native`] (dependency-injected native handle, used by tests and
/// once the FFI arrives).
pub struct TenVad {
    sample_rate: i64,
    hop_size: usize,
    threshold: f32,
    native: Box<dyn NativeTenVad>,
    factory: NativeFactory,
}

impl std::fmt::Debug for TenVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenVad")
            .field("sample_rate", &self.sample_rate)
            .field("hop_size", &self.hop_size)
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl TenVad {
    /// Default hop size (16 ms at 16 kHz).
    pub const DEFAULT_HOP_SIZE: usize = 256;
    /// Default activation threshold.
    pub const DEFAULT_THRESHOLD: f32 = 0.5;

    /// Validate constructor arguments (rate must be 16 kHz; hop ∈ {160, 256}).
    fn validate(sample_rate: i64, hop_size: usize) -> Result<(), VadError> {
        if sample_rate != 16000 {
            return Err(VadError::SampleRate(sample_rate));
        }
        if !VALID_HOP_SIZES_16K.contains(&hop_size) {
            let valid = VALID_HOP_SIZES_16K
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(VadError::HopSize {
                hop: hop_size,
                valid,
            });
        }
        Ok(())
    }

    /// Build a VAD from an injected native-handle factory.
    ///
    /// The factory is called once now (matching Python's native ctor call) and
    /// again on every [`reset`](TenVad::reset), each time with
    /// `(hop_size, threshold)`.
    ///
    /// # Errors
    /// [`VadError::SampleRate`] / [`VadError::HopSize`] on invalid arguments.
    pub fn with_native<F>(
        sample_rate: i64,
        hop_size: usize,
        threshold: f32,
        factory: F,
    ) -> Result<Self, VadError>
    where
        F: Fn(usize, f32) -> Box<dyn NativeTenVad> + Send + Sync + 'static,
    {
        Self::validate(sample_rate, hop_size)?;
        let factory: NativeFactory = Arc::new(factory);
        let native = factory(hop_size, threshold);
        Ok(Self {
            sample_rate,
            hop_size,
            threshold,
            native,
            factory,
        })
    }

    /// Build a production VAD backed by the native TEN-VAD library.
    ///
    /// Until the `ten-vad-sys` FFI crate lands (DESIGN D2) there is no native
    /// backend in any feature configuration, so this always returns
    /// [`VadError::MissingBackend`] — mirroring Python raising `RuntimeError`
    /// mentioning the `local-turn` extra when the native package is absent. The
    /// backend-missing check precedes argument validation, matching Python.
    ///
    /// # Errors
    /// [`VadError::MissingBackend`] — the native backend is not compiled in.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(_sample_rate: i64, _hop_size: usize, _threshold: f32) -> Result<Self, VadError> {
        Err(VadError::MissingBackend)
    }

    /// Configured sample rate (always 16000).
    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    /// Configured hop size (samples per frame).
    #[must_use]
    pub const fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Configured activation threshold.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Run inference; return the `[0, 1]` speech probability.
    ///
    /// # Errors
    /// [`VadError::WrongChunkLength`] when `pcm_chunk` does not carry exactly
    /// `hop_size` int16 samples.
    pub fn speech_probability(&mut self, pcm_chunk: &[u8]) -> Result<f32, VadError> {
        let got = pcm_chunk.len() / 2;
        if pcm_chunk.len() % 2 != 0 || got != self.hop_size {
            return Err(VadError::WrongChunkLength {
                expected: self.hop_size,
                got,
            });
        }
        let samples: Vec<i16> = pcm_chunk
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(self.native.process(&samples))
    }

    /// Whether `pcm_chunk` is speech (`probability >= threshold`).
    ///
    /// # Errors
    /// [`VadError::WrongChunkLength`] — see [`speech_probability`](TenVad::speech_probability).
    pub fn is_speech(&mut self, pcm_chunk: &[u8]) -> Result<bool, VadError> {
        Ok(self.speech_probability(pcm_chunk)? >= self.threshold)
    }
}

impl Vad for TenVad {
    fn is_speech(&mut self, frame: &[u8]) -> bool {
        // Parity with Python (ten_vad.py:83-86): `speech_probability` RAISES
        // `ValueError` when the frame isn't exactly `hop_size` samples, and that
        // propagates through `_on_vad_frame`/`feed_audio` to the awaiting pump. A
        // wrong-length frame is a framing/config bug, so surface it loudly rather
        // than silently classifying every frame as silence (which would strand
        // the endpointer, never detecting a turn).
        self.speech_probability(frame)
            .expect("TEN-VAD frame was not hop_size int16 samples")
            >= self.threshold
    }

    fn reset(&mut self) {
        // No public C reset — rebuild the native handle to drop accumulated
        // state (spec 09 §23), reusing the same hop/threshold.
        self.native = (self.factory)(self.hop_size, self.threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeTenVad, TenVad, Vad, VadError};
    use std::sync::{Arc, Mutex};

    /// Records every native construction + its args, and the probability the
    /// handles return from `process`.
    #[derive(Clone, Default)]
    struct FakeNativeFactory {
        /// `(hop_size, threshold)` per construction.
        calls: Arc<Mutex<Vec<(usize, f32)>>>,
        /// Probability the built handles return.
        prob: Arc<Mutex<f32>>,
    }

    struct FakeNative {
        prob: Arc<Mutex<f32>>,
        last_samples: Arc<Mutex<Vec<i16>>>,
    }

    impl NativeTenVad for FakeNative {
        fn process(&mut self, samples: &[i16]) -> f32 {
            *self.last_samples.lock().unwrap() = samples.to_vec();
            *self.prob.lock().unwrap()
        }
    }

    impl FakeNativeFactory {
        fn last_samples() -> Arc<Mutex<Vec<i16>>> {
            Arc::new(Mutex::new(Vec::new()))
        }

        fn build(
            &self,
            last_samples: &Arc<Mutex<Vec<i16>>>,
        ) -> impl Fn(usize, f32) -> Box<dyn NativeTenVad> + Send + Sync + 'static {
            let calls = Arc::clone(&self.calls);
            let prob = Arc::clone(&self.prob);
            let last = Arc::clone(last_samples);
            move |hop, threshold| {
                calls.lock().unwrap().push((hop, threshold));
                Box::new(FakeNative {
                    prob: Arc::clone(&prob),
                    last_samples: Arc::clone(&last),
                })
            }
        }
    }

    fn pcm_zeros(samples: usize) -> Vec<u8> {
        vec![0u8; samples * 2]
    }

    fn pcm_tone(samples: usize, amplitude: i16) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples * 2);
        for _ in 0..samples {
            out.extend_from_slice(&amplitude.to_le_bytes());
        }
        out
    }

    // --- construction / validation (test_ten_vad.py TestTenVADInit) --------

    #[test]
    fn constructs_native_with_defaults() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        assert_eq!(vad.sample_rate(), 16000);
        assert_eq!(vad.hop_size(), 256);
        // Native constructed exactly once with (hop_size=256, threshold=0.5).
        let calls = f.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 256);
        assert!((calls[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_unsupported_hop_size() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let err = TenVad::with_native(16000, 512, 0.5, f.build(&ls)).unwrap_err();
        assert!(matches!(err, VadError::HopSize { hop: 512, .. }));
        assert!(err.to_string().contains("hop_size"));
    }

    #[test]
    fn rejects_non_16k_sample_rate() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let err = TenVad::with_native(48000, 256, 0.5, f.build(&ls)).unwrap_err();
        assert!(matches!(err, VadError::SampleRate(48000)));
        assert!(err.to_string().contains("sample_rate"));
    }

    // --- speech_probability (TestSpeechProbability) ------------------------

    #[test]
    fn returns_probability_from_process() {
        let f = FakeNativeFactory::default();
        *f.prob.lock().unwrap() = 0.87;
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        let prob = vad.speech_probability(&pcm_zeros(256)).unwrap();
        assert!((prob - 0.87).abs() < 1e-5);
    }

    #[test]
    fn passes_int16_samples_at_hop_size() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        vad.speech_probability(&pcm_tone(256, 5000)).unwrap();
        let samples = ls.lock().unwrap().clone();
        assert_eq!(samples.len(), 256);
        assert_eq!(samples[0], 5000);
    }

    #[test]
    fn supports_10ms_hop_size() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 160, 0.5, f.build(&ls)).unwrap();
        vad.speech_probability(&pcm_zeros(160)).unwrap();
        assert_eq!(ls.lock().unwrap().len(), 160);
    }

    #[test]
    fn rejects_wrong_chunk_length() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        let err = vad.speech_probability(&pcm_zeros(128)).unwrap_err();
        assert!(matches!(
            err,
            VadError::WrongChunkLength {
                expected: 256,
                got: 128
            }
        ));
        assert!(err.to_string().contains("expected"));
    }

    // --- threshold (TestThreshold) -----------------------------------------

    #[test]
    fn is_speech_above_threshold() {
        let f = FakeNativeFactory::default();
        *f.prob.lock().unwrap() = 0.6;
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        assert!(vad.is_speech(&pcm_zeros(256)).unwrap());
    }

    #[test]
    fn is_not_speech_below_threshold() {
        let f = FakeNativeFactory::default();
        *f.prob.lock().unwrap() = 0.4;
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        assert!(!vad.is_speech(&pcm_zeros(256)).unwrap());
    }

    // The endpointer seam (`Vad::is_speech`) must NOT swallow a wrong-length
    // frame as `false`: Python's `speech_probability` raises and crashes the
    // pump. A hop/frame mismatch (e.g. hop_size=160 fed 256-sample frames) must
    // surface loudly, not silently classify every frame as silence.
    #[test]
    #[should_panic(expected = "hop_size")]
    fn vad_is_speech_panics_on_wrong_chunk_length() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.5, f.build(&ls)).unwrap();
        let _ = Vad::is_speech(&mut vad, &pcm_zeros(128));
    }

    // --- reset (TestReset) -------------------------------------------------

    #[test]
    fn reset_recreates_native_handle() {
        let f = FakeNativeFactory::default();
        let ls = FakeNativeFactory::last_samples();
        let mut vad = TenVad::with_native(16000, 256, 0.4, f.build(&ls)).unwrap();
        assert_eq!(f.calls.lock().unwrap().len(), 1);
        Vad::reset(&mut vad);
        let calls = f.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        // Second construction reuses the same hop/threshold.
        assert_eq!(calls[1].0, 256);
        assert!((calls[1].1 - 0.4).abs() < 1e-6);
    }

    // --- production backend absent -----------------------------------------

    #[test]
    fn new_reports_missing_backend() {
        let err = TenVad::new(16000, 256, 0.5).unwrap_err();
        assert!(matches!(err, VadError::MissingBackend));
        assert!(err.to_string().contains("local-turn"));
    }
}
