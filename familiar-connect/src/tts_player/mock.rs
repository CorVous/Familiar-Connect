//! In-process mock TTS player for tests (subsystem 09; Python `tts_player/mock.py`).
//!
//! Simulates playback by sleeping a fraction of the word duration per tick,
//! checking [`TurnScope::is_cancelled`] between ticks so barge-in tests can
//! measure how promptly cancellation cuts speech short. Records
//! `(text, was_cut)` per call and accumulates `total_played_ms`. `speak_started`
//! fires when playback begins (barge-in tests await it to interrupt mid-speech).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::bus::envelope::TurnScope;
use crate::tts_player::protocol::TtsPlayer;

/// Record-what-you-played TTS stand-in.
pub struct MockTTSPlayer {
    ms_per_word: u64,
    poll_ms: u64,
    calls: Mutex<Vec<(String, bool)>>,
    total_played_ms: AtomicU64,
    /// Re-created per `speak`; `stop` cancels the current utterance only.
    stop_token: Mutex<CancellationToken>,
    /// Fired (once) when playback begins.
    speak_started: CancellationToken,
}

impl MockTTSPlayer {
    /// Player with the given pacing (`ms_per_word`) and cancel granularity
    /// (`poll_ms`).
    #[must_use]
    pub fn new(ms_per_word: u64, poll_ms: u64) -> Self {
        Self {
            ms_per_word,
            poll_ms,
            calls: Mutex::new(Vec::new()),
            total_played_ms: AtomicU64::new(0),
            stop_token: Mutex::new(CancellationToken::new()),
            speak_started: CancellationToken::new(),
        }
    }

    /// Recorded `(text, was_cut)` per `speak` call.
    #[must_use]
    pub fn calls(&self) -> Vec<(String, bool)> {
        self.calls.lock().expect("calls poisoned").clone()
    }

    /// Cumulative played milliseconds across all `speak` calls.
    #[must_use]
    pub fn total_played_ms(&self) -> u64 {
        self.total_played_ms.load(Ordering::SeqCst)
    }

    /// Whether playback has begun at least once.
    #[must_use]
    pub fn speak_started(&self) -> bool {
        self.speak_started.is_cancelled()
    }

    /// Await the first `speak` entry (barge-in tests interrupt mid-speech).
    pub async fn await_speak_started(&self) {
        self.speak_started.cancelled().await;
    }
}

impl Default for MockTTSPlayer {
    fn default() -> Self {
        Self::new(200, 5)
    }
}

#[async_trait]
impl TtsPlayer for MockTTSPlayer {
    async fn speak(&self, text: &str, scope: &TurnScope) {
        self.speak_started.cancel();
        let words = text.split_whitespace().count() as u64;
        let budget_ms = words * self.ms_per_word;
        let mut played_ms: u64 = 0;
        let mut cancelled_or_stopped = false;

        // Reset the per-call stop gate on re-entry.
        let token = CancellationToken::new();
        *self.stop_token.lock().expect("stop_token poisoned") = token.clone();

        while played_ms < budget_ms {
            if scope.is_cancelled() || token.is_cancelled() {
                cancelled_or_stopped = true;
                break;
            }
            let step = self.poll_ms.min(budget_ms - played_ms);
            tokio::select! {
                () = token.cancelled() => { cancelled_or_stopped = true; break; }
                () = tokio::time::sleep(Duration::from_millis(step)) => played_ms += step,
            }
        }

        self.total_played_ms.fetch_add(played_ms, Ordering::SeqCst);
        self.calls
            .lock()
            .expect("calls poisoned")
            .push((text.to_owned(), cancelled_or_stopped));
    }

    async fn stop(&self) {
        self.stop_token
            .lock()
            .expect("stop_token poisoned")
            .cancel();
    }
}
