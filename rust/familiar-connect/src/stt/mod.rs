//! Speech-to-text: Transcriber seam + factory + Deepgram/Parakeet/faster-whisper
//! backends (subsystem 09; Python `stt/`).

pub mod deepgram;
pub mod factory;
pub mod faster_whisper;
pub mod parakeet;
pub mod protocol;
