//! Logging-only TTS player (subsystem 09; Python `tts_player/logging_player.py`).
//!
//! Phase-2 production default: logs what it *would* speak but produces no audio.
//! Honours cancellation so the barge-in loop behaves the same as with a real
//! player. Pacing sleeps `ms_per_word` per whitespace word, polling
//! cancellation / stop every `poll_ms`.

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::bus::envelope::TurnScope;
use crate::log_style as ls;
use crate::support::text::truncate;
use crate::tts_player::protocol::TtsPlayer;

/// No-audio TTS stand-in that paces + honours cancellation.
pub struct LoggingTTSPlayer {
    ms_per_word: u64,
    poll_ms: u64,
    /// Re-created per `speak`; `stop` cancels the current utterance only.
    stop_token: Mutex<CancellationToken>,
}

impl LoggingTTSPlayer {
    /// Player with the given pacing (`ms_per_word`) and cancel granularity
    /// (`poll_ms`).
    #[must_use]
    pub fn new(ms_per_word: u64, poll_ms: u64) -> Self {
        Self {
            ms_per_word,
            poll_ms,
            stop_token: Mutex::new(CancellationToken::new()),
        }
    }
}

impl Default for LoggingTTSPlayer {
    fn default() -> Self {
        Self::new(200, 20)
    }
}

#[async_trait]
impl TtsPlayer for LoggingTTSPlayer {
    async fn speak(&self, text: &str, scope: &TurnScope) {
        let words = text.split_whitespace().count() as u64;
        let budget_ms = words * self.ms_per_word;
        let mut played_ms: u64 = 0;

        tracing::info!(
            target: "familiar_connect.tts_player.logging",
            "{} {} {} {}",
            ls::tag("🔊 Say", ls::G),
            ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
            ls::kv_styled("words", &words.to_string(), ls::W, ls::LY),
            ls::kv_styled("text", &truncate(text, 200), ls::W, ls::LW),
        );

        // Reset the per-call stop gate on re-entry.
        let token = CancellationToken::new();
        *self.stop_token.lock().expect("stop_token poisoned") = token.clone();

        while played_ms < budget_ms {
            if scope.is_cancelled() || token.is_cancelled() {
                tracing::info!(
                    target: "familiar_connect.tts_player.logging",
                    "{} {} {}",
                    ls::tag("🔊 Cut", ls::Y),
                    ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
                    ls::kv_styled("played_ms", &played_ms.to_string(), ls::W, ls::LY),
                );
                return;
            }
            let step = self.poll_ms.min(budget_ms - played_ms);
            tokio::select! {
                () = token.cancelled() => return,
                () = tokio::time::sleep(Duration::from_millis(step)) => played_ms += step,
            }
        }
    }

    async fn stop(&self) {
        self.stop_token
            .lock()
            .expect("stop_token poisoned")
            .cancel();
    }
}
