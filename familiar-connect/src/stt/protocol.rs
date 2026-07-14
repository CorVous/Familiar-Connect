//! Transcriber seam + `TranscriptionResult` (subsystem 09; Python `stt/protocol.py`).
//!
//! Lifts the implicit shape of the Deepgram backend into a [`Transcriber`] trait
//! so local-model backends (Parakeet / faster-whisper) drop in behind
//! `[providers.stt].backend` without code-path changes downstream. The trait is
//! object-safe (via `async_trait` + an object-safe [`Transcriber::clone_transcriber`]
//! in place of Python's `clone()` returning `Self`).

use std::collections::HashMap;

use async_trait::async_trait;
use thiserror::Error;

use crate::llm::Message;

/// Channel the wiring layer (subsystem 10) drains transcription results from.
///
/// Python passes an unbounded `asyncio.Queue[TranscriptionEvent]` into
/// [`Transcriber::start`]; the Rust equivalent is an unbounded mpsc sender.
/// Results leave the backend with `user_id = None`; the per-user fan-in task
/// stamps the Discord user id.
pub type TranscriptSender = tokio::sync::mpsc::UnboundedSender<TranscriptionResult>;

/// Single streaming transcription result.
///
/// `user_id` is the Discord user id when known (stamped by the per-user fan-in
/// in the wiring layer — per-SSRC audio gives attribution for free, no provider
/// diarization needed). `speaker` is the provider-side diarization label (when
/// enabled); it is unused by the Discord pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionResult {
    /// Recognized text.
    pub text: String,
    /// Whether this is a final (vs interim) result.
    pub is_final: bool,
    /// Segment start offset, seconds.
    pub start: f64,
    /// Segment end offset, seconds (`start + duration`).
    pub end: f64,
    /// Recognition confidence (`0.0` when unknown).
    pub confidence: f64,
    /// Provider diarization label, when diarization is enabled.
    pub speaker: Option<i64>,
    /// Discord user id, stamped downstream by the fan-in task.
    pub user_id: Option<i64>,
}

impl TranscriptionResult {
    /// Construct a result with default `confidence = 0.0`, `speaker = None`,
    /// `user_id = None` — matching the Python dataclass field defaults.
    #[must_use]
    pub fn new(text: impl Into<String>, is_final: bool, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            is_final,
            start,
            end,
            confidence: 0.0,
            speaker: None,
            user_id: None,
        }
    }

    /// Convert to an LLM [`Message`]; prefix content with `[Voice]`.
    ///
    /// `speaker_names` may key by `user_id` (Discord) or by provider `speaker`
    /// label — `user_id` wins over `speaker`; unmapped ids fall back to `Voice`.
    #[must_use]
    #[allow(
        clippy::implicit_hasher,
        reason = "the map is always the default hasher HashMap<i64, String> the wiring builds"
    )]
    pub fn to_message(&self, speaker_names: Option<&HashMap<i64, String>>) -> Message {
        // user_id wins over speaker; unmapped ids fall through to "Voice".
        let name = speaker_names
            .and_then(|names| {
                self.user_id
                    .filter(|u| names.contains_key(u))
                    .or_else(|| self.speaker.filter(|s| names.contains_key(s)))
                    .map(|key| names[&key].clone())
            })
            .unwrap_or_else(|| "Voice".to_string());
        Message::new("user", format!("[Voice] {}", self.text)).with_name(name)
    }
}

/// `TranscriptionEvent` is an alias of [`TranscriptionResult`] (mirrors the
/// Python `TranscriptionEvent = TranscriptionResult`).
pub type TranscriptionEvent = TranscriptionResult;

/// Errors surfaced by the STT factory + backends.
///
/// Byte-stable messages are test contracts: `"DEEPGRAM_API_KEY"`, the
/// `"not connected"` substring, and the backend name in the unknown-backend and
/// local-stt-unavailable variants.
#[derive(Debug, Error)]
pub enum SttError {
    /// `[providers.stt].backend` names an unknown backend.
    #[error(
        "Unknown STT backend '{backend}'. Known: {known}. Set [providers.stt].backend in character.toml."
    )]
    UnknownBackend {
        /// The rejected backend name.
        backend: String,
        /// Comma-joined sorted list of known backends.
        known: String,
    },
    /// `DEEPGRAM_API_KEY` is not set (or empty).
    #[error("DEEPGRAM_API_KEY environment variable is required")]
    MissingApiKey,
    /// `send_audio`/`finalize` reached a transcriber that was never `start`ed
    /// (or has been `stop`ped).
    #[error("Transcriber is not connected — call start() first")]
    NotConnected,
    /// A local-STT backend was selected but no local engine is built into this
    /// binary (the `local-stt` feature has no engine chosen yet — DESIGN §6).
    ///
    /// Mirrors the Python lazy-import degradation (`RuntimeError` re-raised as
    /// `ValueError`): the run command logs a warning and continues text-only.
    #[error(
        "STT backend '{0}' requires a local-stt engine, which is not built into this binary (no engine chosen for the local-stt feature yet — see DESIGN §6)"
    )]
    LocalSttUnavailable(String),
    /// The Deepgram WebSocket connection could not be opened at `start`.
    #[error("failed to connect to the Deepgram WebSocket")]
    ConnectFailed,
}

/// Streaming PCM → [`TranscriptionResult`] surface.
///
/// Lifecycle: [`clone_transcriber`](Transcriber::clone_transcriber) per
/// user/channel → `start(sender)` once → `send_audio(pcm)` per chunk →
/// `finalize()` to flush a pending segment → `stop()` to tear down (idempotent).
/// Feed rate is 48 kHz mono s16le from the sink.
///
/// The clone-as-template method is object-safe (returns `Box<dyn Transcriber>`)
/// in place of Python's `clone() -> Self`. The `set_endpointing_ms` /
/// `idle_close_s` / `backend_name` seams carry default no-ops so a minimal
/// scripted stub only needs the five lifecycle methods (DESIGN §4.8).
#[async_trait]
pub trait Transcriber: Send {
    /// Fresh independent instance with the same config (Python `clone()`).
    fn clone_transcriber(&self) -> Box<dyn Transcriber>;

    /// Begin transcription, pushing results onto `output` in wire order.
    async fn start(&mut self, output: TranscriptSender) -> Result<(), SttError>;

    /// Feed one PCM chunk (linear16, sample rate set by the impl — 48 kHz here).
    ///
    /// Errors with [`SttError::NotConnected`] when called before `start` (or
    /// after `stop`).
    async fn send_audio(&mut self, data: &[u8]) -> Result<(), SttError>;

    /// Force the impl to flush its pending segment as final. No-op when there is
    /// nothing to flush, or before `start` (idle-watchdog safe).
    async fn finalize(&mut self);

    /// Tear down. Idempotent.
    async fn stop(&mut self);

    /// Set the Deepgram hosted-endpointer silence window (ms), returning `true`
    /// when the backend honours the poke.
    ///
    /// This is the typed replacement for Python's duck-typed
    /// `hasattr(clone, "endpointing_ms")` + `clone.endpointing_ms = 10` seam
    /// (spec 09 J.70): the wiring calls this before `start`; backends that do
    /// not endpoint server-side (Parakeet / faster-whisper) inherit the default
    /// `false` so the wiring keys off support exactly as the Python `hasattr`
    /// guard did.
    fn set_endpointing_ms(&mut self, _ms: i64) -> bool {
        false
    }

    /// Per-user idle-close window (seconds); the wiring arms its idle watchdog
    /// from this (spec 09 J.73). `0` disables. Default `30.0`.
    fn idle_close_s(&self) -> f64 {
        30.0
    }

    /// Short backend identifier (`"deepgram"`, …); diagnostics + dispatch checks.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> HashMap<i64, String> {
        HashMap::from([(2_i64, "Alice".to_string()), (5_i64, "Bob".to_string())])
    }

    #[test]
    fn dataclass_fields() {
        let r = TranscriptionResult {
            text: "hello world".to_string(),
            is_final: true,
            start: 0.0,
            end: 1.5,
            confidence: 0.98,
            speaker: Some(0),
            user_id: None,
        };
        assert_eq!(r.text, "hello world");
        assert!(r.is_final);
        assert!((r.start - 0.0).abs() < f64::EPSILON);
        assert!((r.end - 1.5).abs() < f64::EPSILON);
        assert!((r.confidence - 0.98).abs() < 1e-9);
        assert_eq!(r.speaker, Some(0));
    }

    #[test]
    fn default_values() {
        let r = TranscriptionResult::new("test", false, 0.0, 0.5);
        assert_eq!(r.speaker, None);
        assert_eq!(r.user_id, None);
        assert!((r.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn to_message_plain() {
        let r = TranscriptionResult::new("hello there", true, 0.0, 1.0);
        let msg = r.to_message(None);
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "[Voice] hello there");
        assert_eq!(msg.name.as_deref(), Some("Voice"));
    }

    #[test]
    fn to_message_with_speaker_names() {
        let mut r = TranscriptionResult::new("how are you", true, 0.0, 1.0);
        r.speaker = Some(2);
        let msg = r.to_message(Some(&names()));
        assert_eq!(msg.name.as_deref(), Some("Alice"));
        assert_eq!(msg.content_str(), "[Voice] how are you");
    }

    #[test]
    fn to_message_with_unknown_speaker() {
        let mut r = TranscriptionResult::new("hi", true, 0.0, 0.5);
        r.speaker = Some(99);
        let names = HashMap::from([(2_i64, "Alice".to_string())]);
        let msg = r.to_message(Some(&names));
        assert_eq!(msg.name.as_deref(), Some("Voice"));
    }

    #[test]
    fn to_message_user_id_wins_over_speaker() {
        let mut r = TranscriptionResult::new("hey", true, 0.0, 0.5);
        r.user_id = Some(5);
        r.speaker = Some(2);
        let msg = r.to_message(Some(&names()));
        // user_id 5 -> "Bob" wins over speaker 2 -> "Alice".
        assert_eq!(msg.name.as_deref(), Some("Bob"));
    }

    #[test]
    fn to_message_falls_through_to_speaker_when_user_id_unmapped() {
        let mut r = TranscriptionResult::new("hey", true, 0.0, 0.5);
        r.user_id = Some(999); // not in map
        r.speaker = Some(2); // in map -> "Alice"
        let msg = r.to_message(Some(&names()));
        assert_eq!(msg.name.as_deref(), Some("Alice"));
    }

    #[test]
    fn transcription_event_is_alias() {
        // Compile-time proof the alias resolves to the same type.
        let e: TranscriptionEvent = TranscriptionResult::new("x", true, 0.0, 0.0);
        let r: TranscriptionResult = e;
        assert_eq!(r.text, "x");
    }
}
