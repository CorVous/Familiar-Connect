//! Local turn detection — TEN-VAD + Smart Turn v3 + endpointer state machine +
//! factory (subsystem 09; Python `voice/turn_detection/`).
//!
//! The audio pump forks per-user PCM into both the transcriber and the
//! [`UtteranceEndpointer`] when `[providers.turn_detection].strategy =
//! "ten+smart_turn"`; on a turn-complete verdict the endpointer's callback
//! drives `transcriber.finalize()` (wired in the Layer-4 voice source).
//!
//! Seams (per DESIGN §4.8): [`Vad`] and [`SmartTurn`] are the two detector traits
//! the endpointer consumes; [`NativeTenVad`] and [`SmartTurnModel`] are the
//! injected native/ONNX handles; [`ModelDownloader`] resolves Smart Turn weights.

pub mod endpointer;
pub mod factory;
pub mod smart_turn;
pub mod ten_vad;

pub use endpointer::{OnTurnComplete, UtteranceEndpointer};
pub use factory::{
    LocalTurnDetector, ModelDownloader, TurnDetectionError, create_local_turn_detector,
    create_local_turn_detector_with,
};
pub use smart_turn::{SmartTurn, SmartTurnDetector, SmartTurnError, SmartTurnModel};
pub use ten_vad::{NativeTenVad, TenVad, Vad, VadError};
