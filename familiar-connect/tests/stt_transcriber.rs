//! Integration coverage for the public `stt` surface (subsystem 09).
//!
//! Exercises the [`Transcriber`] trait object, the public factory dispatch +
//! error contracts, `TranscriptionResult::to_message`, and the Deepgram
//! builder — everything reachable without a live network or an injected secret.
//! The transport-level lifecycle (receive loop, reconnect, replay, keepalive)
//! is covered by the in-module unit tests with the mock transport.

use std::collections::HashMap;

use familiar_connect::config::{DeepgramSTTConfig, STTConfig};
use familiar_connect::stt::deepgram::{
    DEEPGRAM_WS_URL, DEFAULT_IDLE_FINALIZE_S, DeepgramTranscriber, build_deepgram_transcriber,
};
use familiar_connect::stt::{
    KNOWN_BACKENDS, SttError, Transcriber, TranscriptionEvent, TranscriptionResult,
    create_transcriber,
};

#[test]
fn transcription_result_to_message() {
    let mut r = TranscriptionResult::new("hey there", true, 0.0, 1.0);
    let plain = r.to_message(None);
    assert_eq!(plain.role, "user");
    assert_eq!(plain.content_str(), "[Voice] hey there");
    assert_eq!(plain.name.as_deref(), Some("Voice"));

    r.speaker = Some(2);
    let names = HashMap::from([(2_i64, "Alice".to_string())]);
    assert_eq!(r.to_message(Some(&names)).name.as_deref(), Some("Alice"));
}

#[test]
fn transcription_event_is_result_alias() {
    let e: TranscriptionEvent = TranscriptionResult::new("x", false, 0.0, 0.0);
    assert_eq!(e.text, "x");
}

#[test]
fn deepgram_transcriber_is_a_boxed_transcriber() {
    let mut t: Box<dyn Transcriber> = Box::new(DeepgramTranscriber::new("k"));
    assert_eq!(t.backend_name(), "deepgram");
    assert!((t.idle_close_s() - 30.0).abs() < f64::EPSILON);
    // Deepgram honours the endpointing poke; the clone is an independent box.
    assert!(t.set_endpointing_ms(10));
    let cloned = t.clone_transcriber();
    assert_eq!(cloned.backend_name(), "deepgram");
}

#[test]
fn factory_rejects_unknown_backend() {
    // Env-independent: the unknown-backend guard fires before any secret matters.
    let cfg = STTConfig {
        backend: "vosk".to_string(),
        ..STTConfig::default()
    };
    let err = create_transcriber(&cfg).err().unwrap();
    assert!(matches!(err, SttError::UnknownBackend { .. }));
    assert!(format!("{err}").contains("vosk"));
}

#[test]
fn factory_degrades_local_backends() {
    for backend in ["parakeet", "faster_whisper"] {
        let cfg = STTConfig {
            backend: backend.to_string(),
            ..STTConfig::default()
        };
        let err = create_transcriber(&cfg).err().unwrap();
        assert!(matches!(err, SttError::LocalSttUnavailable(_)));
        assert!(format!("{err}").contains(backend));
    }
}

#[test]
fn known_backends_are_the_three_documented() {
    assert_eq!(KNOWN_BACKENDS, ["deepgram", "faster_whisper", "parakeet"]);
}

#[test]
fn deepgram_builder_flows_config_and_requires_key() {
    let cfg = DeepgramSTTConfig {
        model: "nova-2".to_string(),
        endpointing_ms: 275,
        ..DeepgramSTTConfig::default()
    };
    let t = build_deepgram_transcriber(&cfg, Some("sk-test".to_string())).unwrap();
    assert_eq!(t.model, "nova-2");
    assert_eq!(t.endpointing_ms, 275);
    assert!(t.build_ws_url().starts_with(DEEPGRAM_WS_URL));

    let err = build_deepgram_transcriber(&cfg, None).err().unwrap();
    assert!(matches!(err, SttError::MissingApiKey));
}

#[test]
fn default_idle_finalize_constant() {
    // Imported by the wiring layer for plain-Deepgram idle flush.
    assert!((DEFAULT_IDLE_FINALIZE_S - 0.5).abs() < f64::EPSILON);
}
