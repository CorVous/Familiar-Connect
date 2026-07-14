//! Utterance endpointer — TEN-VAD + Smart Turn over a 48 kHz PCM stream
//! (subsystem 09; Python `voice/turn_detection/endpointer.py`).
//!
//! Drives per-user local turn detection: 48 kHz mono int16 PCM in (any chunk
//! length), a `on_turn_complete` callback out. Three building blocks compose:
//! [`Resampler48to16`](crate::voice::audio::Resampler48to16) (3:1 decimation to
//! TEN-VAD's native rate), a [`Vad`] (per-16 ms is-speech), and a [`SmartTurn`]
//! (semantic completion classifier).
//!
//! State machine (per user):
//!
//! - `IDLE` → speech burst detected → `SPEAKING`
//! - `SPEAKING` → silence streak ≥ `silence_ms` → run Smart Turn
//!   - `complete` → fire callback, reset to `IDLE`
//!   - `incomplete` → `POST_INCOMPLETE`
//! - `POST_INCOMPLETE` → fresh speech → `SPEAKING` again
//! - `POST_INCOMPLETE` → continued silence → no reclassification
//!
//! The classifier is invoked on the silence-after-speech edge only; extra
//! silence after an `incomplete` verdict does not refire it. A subsequent
//! speech-then-silence cycle does. Smart Turn runs off the reactor via
//! `spawn_blocking` — wav2vec2 over up to 16 s of audio would otherwise stall
//! Deepgram keepalives and Discord's 10 s voice heartbeat (spec 09 §27).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::voice::audio::Resampler48to16;
use crate::voice::turn_detection::smart_turn::SmartTurn;
use crate::voice::turn_detection::ten_vad::Vad;

/// TEN-VAD native frame at 16 kHz: 256 int16 samples = 512 bytes = 16 ms.
const VAD_CHUNK_SAMPLES: usize = 256;
const VAD_CHUNK_BYTES: usize = VAD_CHUNK_SAMPLES * 2;

/// The turn-complete callback: awaited with the buffered utterance whenever a
/// turn ends. Mirrors Python's `Callable[[bytes], Awaitable[None]]`.
pub type OnTurnComplete =
    Box<dyn Fn(Vec<u8>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Chunk threshold: `int(ms / 16.0)` floored to ≥ 1 (spec 09 §25).
///
/// `_VAD_CHUNK_MS` is exactly `16.0` (`256/16000*1000`), a power-of-two divisor,
/// so integer `ms / 16` equals Python's `int(ms / 16.0)` truncation for every
/// non-negative `ms` — no float cast needed. Defaults land on 200 ms → 12 frames
/// (192 ms effective) and 100 ms → 6 frames (96 ms).
const fn chunk_threshold(ms: i64) -> i64 {
    let raw = ms / 16;
    if raw < 1 { 1 } else { raw }
}

/// Per-user local turn-detection state machine.
///
/// Caller feeds 48 kHz mono int16 PCM via [`feed_audio`](Self::feed_audio) (any
/// chunk length); the endpointer resamples, frames into 16 ms VAD windows, and
/// on the silence-after-speech edge runs Smart Turn over the buffered 16 kHz
/// utterance audio. `on_turn_complete` is awaited with the buffered audio
/// whenever Smart Turn classifies `complete`.
pub struct UtteranceEndpointer {
    vad: Box<dyn Vad>,
    smart_turn: Arc<dyn SmartTurn>,
    on_complete: OnTurnComplete,
    silence_chunks_threshold: i64,
    speech_chunks_threshold: i64,

    resampler: Resampler48to16,
    /// Remainder bytes from a partial VAD frame — accumulate next feed.
    frame_carry: Vec<u8>,
    utterance: Vec<u8>,

    speaking: bool,
    post_incomplete: bool,
    speech_streak: i64,
    silence_streak: i64,
}

impl UtteranceEndpointer {
    /// Build an endpointer over the given VAD + Smart Turn + callback.
    ///
    /// `silence_ms` (default 200) is the trailing silence before classification;
    /// `speech_start_ms` (default 100) is the speech run before "speaking"
    /// latches. Both convert to chunk counts via [`chunk_threshold`].
    #[must_use]
    pub fn new(
        vad: Box<dyn Vad>,
        smart_turn: Arc<dyn SmartTurn>,
        on_turn_complete: OnTurnComplete,
        silence_ms: i64,
        speech_start_ms: i64,
    ) -> Self {
        Self {
            vad,
            smart_turn,
            on_complete: on_turn_complete,
            silence_chunks_threshold: chunk_threshold(silence_ms),
            speech_chunks_threshold: chunk_threshold(speech_start_ms),
            resampler: Resampler48to16::new(),
            frame_carry: Vec::new(),
            utterance: Vec::new(),
            speaking: false,
            post_incomplete: false,
            speech_streak: 0,
            silence_streak: 0,
        }
    }

    /// Drop buffered audio + VAD/streak state; resets the resampler too.
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.frame_carry.clear();
        self.utterance.clear();
        self.speaking = false;
        self.post_incomplete = false;
        self.speech_streak = 0;
        self.silence_streak = 0;
        self.vad.reset();
    }

    /// Emit buffered speech as a complete turn — the external idle fallback.
    ///
    /// The state machine only advances on [`feed_audio`](Self::feed_audio)
    /// frames, but Discord's client VAD halts RTP during silence, so trailing
    /// silence delivers no frames to re-trigger classification. Two stranding
    /// cases follow: a Smart Turn `incomplete` misfire (sitting in
    /// `POST_INCOMPLETE`), or a burst that stopped before the silence streak
    /// reached `silence_ms` (still `SPEAKING`). The audio pump calls this after
    /// an idle gap to drain the buffered audio.
    ///
    /// Fires on the *state* (`SPEAKING` or `POST_INCOMPLETE`), not on buffered
    /// bytes — after an `incomplete` verdict the memory-bounding idle-clear can
    /// drain the buffer while the turn is still stranded, and consumers key off
    /// the turn ending (STT finalize), not the payload.
    ///
    /// Returns `true` when a turn was emitted, `false` when nothing was pending
    /// (pure idle, or already drained by normal classification).
    pub async fn force_complete_if_pending(&mut self) -> bool {
        if !(self.speaking || self.post_incomplete) {
            return false;
        }
        let audio = std::mem::take(&mut self.utterance);
        self.speaking = false;
        self.post_incomplete = false;
        self.speech_streak = 0;
        self.silence_streak = 0;
        self.vad.reset();
        (self.on_complete)(audio).await;
        true
    }

    /// Resample, frame into 16 ms VAD windows, advance the state machine.
    ///
    /// # Panics
    /// Panics when `pcm_48k` has an odd (non-`i16`-aligned) byte length. The pump
    /// always feeds even mono chunks, so this never fires on the happy path; a
    /// malformed chunk is a caller bug surfaced loudly, matching Python where
    /// `Resampler48to16.feed` raises `ValueError` and the exception propagates out
    /// to the awaiting pump (audio.py:49-51, endpointer.py:133).
    pub async fn feed_audio(&mut self, pcm_48k: &[u8]) {
        if pcm_48k.is_empty() {
            return;
        }
        // Odd-length input can't be int16-framed. Python raises `ValueError` here
        // and it propagates to the pump; mirror that loud failure rather than
        // silently dropping the chunk.
        let resampled = self
            .resampler
            .feed(pcm_48k)
            .expect("endpointer fed a non-int16-aligned 48 kHz chunk");
        if resampled.is_empty() {
            return;
        }
        self.frame_carry.extend_from_slice(&resampled);
        // Consume as many full VAD frames as the carry holds.
        while self.frame_carry.len() >= VAD_CHUNK_BYTES {
            let frame: Vec<u8> = self.frame_carry.drain(..VAD_CHUNK_BYTES).collect();
            self.on_vad_frame(&frame).await;
        }
    }

    /// Single 16 ms decision step.
    async fn on_vad_frame(&mut self, frame: &[u8]) {
        // ALWAYS buffer the frame before the verdict so pre-latch ramp audio is
        // included.
        self.utterance.extend_from_slice(frame);
        let is_speech = self.vad.is_speech(frame);

        if is_speech {
            self.silence_streak = 0;
            if !self.speaking {
                self.speech_streak += 1;
                if self.speech_streak >= self.speech_chunks_threshold {
                    self.speaking = true;
                    self.post_incomplete = false;
                }
            }
            return;
        }

        // Silence frame.
        self.speech_streak = 0;
        if !self.speaking {
            // Idle silence — drop the buffer to bound memory; nothing to
            // classify. This ALSO runs in POST_INCOMPLETE, draining the held
            // audio while the state stays stranded.
            self.utterance.clear();
            return;
        }

        self.silence_streak += 1;
        if self.silence_streak < self.silence_chunks_threshold {
            return;
        }

        // Silence threshold hit after speech → classify, unless we already did
        // and got `incomplete` with no new speech since.
        if self.post_incomplete {
            return;
        }

        // Smart Turn ONNX runs wav2vec2 over up to 16 s of audio; dispatch off
        // the reactor so a slow call can't stall Deepgram keepalives or the
        // Discord voice heartbeat (10 s watchdog).
        let smart_turn = Arc::clone(&self.smart_turn);
        let buffer = self.utterance.clone();
        let verdict = tokio::task::spawn_blocking(move || smart_turn.is_turn_complete(&buffer))
            .await
            .expect("smart turn classify task panicked");

        if verdict {
            let audio = std::mem::take(&mut self.utterance);
            self.speaking = false;
            self.post_incomplete = false;
            self.silence_streak = 0;
            self.vad.reset();
            (self.on_complete)(audio).await;
        } else {
            // Keep the buffer; await fresh speech, then a fresh silence streak.
            self.post_incomplete = true;
            self.speaking = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::chunk_threshold;

    #[test]
    fn chunk_threshold_truncates_and_floors() {
        // int(200 / 16.0) = 12; int(100 / 16.0) = 6; int(32 / 16.0) = 2.
        assert_eq!(chunk_threshold(200), 12);
        assert_eq!(chunk_threshold(100), 6);
        assert_eq!(chunk_threshold(64), 4);
        assert_eq!(chunk_threshold(32), 2);
        // floored to >= 1 even for sub-frame windows.
        assert_eq!(chunk_threshold(10), 1);
        assert_eq!(chunk_threshold(0), 1);
    }
}
