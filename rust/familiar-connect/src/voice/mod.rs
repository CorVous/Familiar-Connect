//! Discord voice in: DAVE E2EE client, per-SSRC recording, PCM conversion,
//! per-user turn detection (subsystem 09; Python `voice/`).

pub mod audio;
pub mod dave_client;
pub mod dave_ws;
pub mod recording_sink;
pub mod turn_detection;
