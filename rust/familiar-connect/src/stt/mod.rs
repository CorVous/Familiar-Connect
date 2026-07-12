//! Speech-to-text: [`Transcriber`] seam + factory + Deepgram backend
//! (subsystem 09; Python `stt/`).
//!
//! Lifts the implicit shape of the Deepgram transcriber behind a
//! [`protocol::Transcriber`] trait so local-model backends (Parakeet /
//! faster-whisper) drop in behind `[providers.stt].backend` without touching the
//! wiring layer. [`factory::create_transcriber`] dispatches on the backend name.
//!
//! Backends: `deepgram` (streaming WebSocket, implemented here); `parakeet` /
//! `faster_whisper` are buffer-and-finalize local backends whose Rust engine is
//! not yet chosen (DESIGN §6) — see [`parakeet`] / [`faster_whisper`] for the
//! contract they will implement.

pub mod deepgram;
pub mod factory;
pub mod faster_whisper;
pub mod parakeet;
pub mod protocol;

pub use factory::{KNOWN_BACKENDS, create_transcriber};
pub use protocol::{
    SttError, Transcriber, TranscriptSender, TranscriptionEvent, TranscriptionResult,
};
