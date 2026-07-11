//! Per-user utterance endpointing: TEN-VAD + Smart Turn v3 + state machine +
//! factory (subsystem 09; Python `voice/turn_detection/`).

pub mod endpointer;
pub mod factory;
pub mod smart_turn;
pub mod ten_vad;
