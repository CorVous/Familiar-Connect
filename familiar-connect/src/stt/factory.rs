//! STT backend selector — dispatches on `[providers.stt].backend`
//! (subsystem 09; Python `stt/factory.py`).
//!
//! Each per-backend builder takes its typed `[providers.stt.<backend>]`
//! sub-table; only secrets (`DEEPGRAM_API_KEY`) come from the environment.

use crate::config::STTConfig;
use crate::stt::deepgram::build_deepgram_transcriber;
use crate::stt::protocol::{SttError, Transcriber};

/// Known backends, sorted (the factory error renders this list).
pub const KNOWN_BACKENDS: [&str; 3] = ["deepgram", "faster_whisper", "parakeet"];

/// Build a [`Transcriber`] from `config`, reading `DEEPGRAM_API_KEY` from the
/// environment for the Deepgram backend.
///
/// # Errors
/// - [`SttError::UnknownBackend`] when `backend` is not one of [`KNOWN_BACKENDS`].
/// - [`SttError::MissingApiKey`] when the Deepgram backend is selected but
///   `DEEPGRAM_API_KEY` is unset.
/// - [`SttError::LocalSttUnavailable`] when a local backend (`parakeet` /
///   `faster_whisper`) is selected — no local engine is built into this binary
///   yet (DESIGN §6). The caller logs + degrades to text-only, mirroring the
///   Python lazy-import degradation.
pub fn create_transcriber(config: &STTConfig) -> Result<Box<dyn Transcriber>, SttError> {
    let deepgram_key = std::env::var("DEEPGRAM_API_KEY").ok();
    create_transcriber_with_secrets(config, deepgram_key)
}

/// [`create_transcriber`] with the Deepgram secret injected (hermetic seam: the
/// crate forbids `unsafe`, so tests cannot mutate the process environment).
pub(crate) fn create_transcriber_with_secrets(
    config: &STTConfig,
    deepgram_key: Option<String>,
) -> Result<Box<dyn Transcriber>, SttError> {
    let backend = config.backend.as_str();
    if !KNOWN_BACKENDS.contains(&backend) {
        return Err(SttError::UnknownBackend {
            backend: backend.to_string(),
            known: KNOWN_BACKENDS.join(", "),
        });
    }

    match backend {
        "deepgram" => {
            let t = build_deepgram_transcriber(&config.deepgram, deepgram_key)?;
            Ok(Box::new(t))
        }
        // Parakeet / faster-whisper are buffer-and-finalize local backends. No
        // Rust local-STT engine is wired yet (the `local-stt` feature has no
        // engine chosen — DESIGN §6), so selecting them degrades with a clear
        // error, exactly as the Python lazy-import path re-raises its
        // RuntimeError as ValueError for the run command to catch + warn.
        "parakeet" => Err(SttError::LocalSttUnavailable("parakeet".to_string())),
        "faster_whisper" => Err(SttError::LocalSttUnavailable("faster_whisper".to_string())),
        // Unreachable while every KNOWN_BACKENDS entry has a dispatch arm.
        other => Err(SttError::UnknownBackend {
            backend: other.to_string(),
            known: KNOWN_BACKENDS.join(", "),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::STTConfig;

    #[test]
    fn deepgram_backend_dispatches() {
        let cfg = STTConfig {
            backend: "deepgram".to_string(),
            ..STTConfig::default()
        };
        let t = create_transcriber_with_secrets(&cfg, Some("test-key".to_string())).unwrap();
        assert_eq!(t.backend_name(), "deepgram");
        assert!((t.idle_close_s() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn deepgram_missing_key_is_value_error() {
        let cfg = STTConfig {
            backend: "deepgram".to_string(),
            ..STTConfig::default()
        };
        let err = create_transcriber_with_secrets(&cfg, None).err().unwrap();
        assert!(format!("{err}").contains("DEEPGRAM_API_KEY"));
    }

    #[test]
    fn parakeet_backend_degrades() {
        let cfg = STTConfig {
            backend: "parakeet".to_string(),
            ..STTConfig::default()
        };
        let err = create_transcriber_with_secrets(&cfg, None).err().unwrap();
        assert!(matches!(err, SttError::LocalSttUnavailable(ref b) if b == "parakeet"));
        assert!(format!("{err}").contains("parakeet"));
    }

    #[test]
    fn faster_whisper_backend_degrades() {
        let cfg = STTConfig {
            backend: "faster_whisper".to_string(),
            ..STTConfig::default()
        };
        let err = create_transcriber_with_secrets(&cfg, None).err().unwrap();
        assert!(matches!(err, SttError::LocalSttUnavailable(ref b) if b == "faster_whisper"));
        assert!(format!("{err}").contains("faster_whisper"));
    }

    #[test]
    fn unknown_backend_rejected() {
        let cfg = STTConfig {
            backend: "vosk".to_string(),
            ..STTConfig::default()
        };
        let err = create_transcriber_with_secrets(&cfg, Some("test-key".to_string()))
            .err()
            .unwrap();
        assert!(format!("{err}").contains("vosk"));
        assert!(matches!(err, SttError::UnknownBackend { .. }));
    }

    #[test]
    fn known_backends_sorted() {
        let mut sorted = KNOWN_BACKENDS;
        sorted.sort_unstable();
        assert_eq!(KNOWN_BACKENDS, sorted);
    }
}
