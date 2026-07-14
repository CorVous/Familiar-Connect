//! `TtsPlayer` seam (subsystem 09; Python `tts_player/protocol.py`).
//!
//! The speak surface used by the voice responder (06). Production wraps
//! Cartesia/Azure/Gemini + Discord voice playback via
//! [`DiscordVoicePlayer`](super::discord_player::DiscordVoicePlayer); tests use
//! [`MockTTSPlayer`](super::mock::MockTTSPlayer).

use async_trait::async_trait;

use crate::bus::envelope::TurnScope;

/// Synthesize text and play it. Cancellable mid-speech.
#[async_trait]
pub trait TtsPlayer: Send + Sync {
    /// Speak `text` until complete or `scope` is cancelled.
    ///
    /// Implementations check `scope.is_cancelled()` at the finest granularity
    /// their API permits (per audio-chunk / poll tick).
    async fn speak(&self, text: &str, scope: &TurnScope);

    /// Flush in-flight audio immediately (barge-in fast path).
    async fn stop(&self);
}
