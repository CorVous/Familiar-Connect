//! Endpointer state-machine tests (ported from Python
//! `tests/test_utterance_endpointer.py`).
//!
//! The native VAD + ONNX Smart Turn are replaced by scripted trait doubles
//! (DESIGN §4.8): a pattern-walking [`Vad`] and a verdict-popping [`SmartTurn`].
//! Timing is deterministic — each 768-sample (16 ms-equivalent) input chunk
//! resamples to exactly one 256-sample VAD frame.

// Scripted doubles guard shared state with a `Mutex` and build via factory
// `new`s that return a `(handle, state)` tuple.
#![allow(clippy::significant_drop_tightening, clippy::new_ret_no_self)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use familiar_connect::voice::turn_detection::{
    OnTurnComplete, SmartTurn, UtteranceEndpointer, Vad,
};

// 16 ms at 16 kHz mono int16 — TEN-VAD default hop.
const VAD_CHUNK_SAMPLES: usize = 256;
// 48 kHz feed = 3× 16 kHz; one VAD chunk = 3× input samples.
const INPUT_CHUNK_SAMPLES: usize = VAD_CHUNK_SAMPLES * 3;
const INPUT_CHUNK_BYTES: usize = INPUT_CHUNK_SAMPLES * 2;

fn input_chunk(amplitude: i16) -> Vec<u8> {
    let mut out = Vec::with_capacity(INPUT_CHUNK_BYTES);
    for _ in 0..INPUT_CHUNK_SAMPLES {
        out.extend_from_slice(&amplitude.to_le_bytes());
    }
    out
}

// --- scripted VAD ----------------------------------------------------------

#[derive(Default)]
struct VadState {
    idx: usize,
    is_speech_calls: usize,
    reset_calls: usize,
}

/// Mock TEN-VAD whose `is_speech` walks a pattern (repeating the last value);
/// `reset` rewinds the pattern iterator — matching Python's `_make_vad`.
#[derive(Clone)]
struct PatternVad {
    pattern: Arc<Vec<bool>>,
    state: Arc<Mutex<VadState>>,
}

impl PatternVad {
    fn new(pattern: Vec<bool>) -> (Box<dyn Vad>, Arc<Mutex<VadState>>) {
        let state = Arc::new(Mutex::new(VadState::default()));
        let vad = Self {
            pattern: Arc::new(pattern),
            state: Arc::clone(&state),
        };
        (Box::new(vad), state)
    }
}

impl Vad for PatternVad {
    fn is_speech(&mut self, _frame: &[u8]) -> bool {
        let mut s = self.state.lock().unwrap();
        let idx = s.idx.min(self.pattern.len() - 1);
        s.idx += 1;
        s.is_speech_calls += 1;
        self.pattern[idx]
    }

    fn reset(&mut self) {
        let mut s = self.state.lock().unwrap();
        s.idx = 0;
        s.reset_calls += 1;
    }
}

// --- scripted Smart Turn ---------------------------------------------------

#[derive(Default)]
struct StState {
    idx: usize,
    buffers: Vec<Vec<u8>>,
    threads: Vec<ThreadId>,
}

/// Mock SmartTurn — each `is_turn_complete` pops one verdict and records the
/// buffer + calling thread.
struct ScriptedSmartTurn {
    verdicts: Vec<bool>,
    state: Arc<Mutex<StState>>,
}

impl ScriptedSmartTurn {
    fn new(verdicts: Vec<bool>) -> (Arc<dyn SmartTurn>, Arc<Mutex<StState>>) {
        let state = Arc::new(Mutex::new(StState::default()));
        let st = Self {
            verdicts,
            state: Arc::clone(&state),
        };
        (Arc::new(st), state)
    }
}

impl SmartTurn for ScriptedSmartTurn {
    fn is_turn_complete(&self, pcm_audio: &[u8]) -> bool {
        let mut s = self.state.lock().unwrap();
        s.buffers.push(pcm_audio.to_vec());
        s.threads.push(std::thread::current().id());
        let verdict = self.verdicts[s.idx];
        s.idx += 1;
        verdict
    }
}

// --- callback --------------------------------------------------------------

fn capture_callback(bucket: Arc<Mutex<Vec<Vec<u8>>>>) -> OnTurnComplete {
    Box::new(move |audio: Vec<u8>| {
        let bucket = Arc::clone(&bucket);
        let fut: Pin<Box<dyn Future<Output = ()> + Send>> = Box::pin(async move {
            bucket.lock().unwrap().push(audio);
        });
        fut
    })
}

// ---------------------------------------------------------------------------
// TestSilenceOnly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_classification_when_no_speech() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (vad, vad_state) = PatternVad::new(vec![false]);
    let (st, st_state) = ScriptedSmartTurn::new(vec![]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 100);

    for _ in 0..10 {
        ep.feed_audio(&input_chunk(0)).await;
    }

    assert!(vad_state.lock().unwrap().is_speech_calls >= 1);
    assert_eq!(st_state.lock().unwrap().buffers.len(), 0);
    assert!(calls.lock().unwrap().is_empty());
}

// An odd-length (non-int16-aligned) chunk can't be framed; Python's resampler
// raises `ValueError` and it propagates to the pump. The Rust endpointer mirrors
// that loud failure with a panic rather than silently dropping the chunk.
#[tokio::test]
#[should_panic(expected = "non-int16-aligned")]
async fn feed_audio_panics_on_odd_length_chunk() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (vad, _) = PatternVad::new(vec![false]);
    let (st, _) = ScriptedSmartTurn::new(vec![]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(calls), 200, 100);

    ep.feed_audio(&[0u8; 3]).await;
}

// ---------------------------------------------------------------------------
// TestCompleteUtterance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn speech_then_silence_classifies_and_fires_callback() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 5];
    pattern.extend(vec![false; 16]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, st_state) = ScriptedSmartTurn::new(vec![true]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1000)).await;
    }

    assert_eq!(st_state.lock().unwrap().buffers.len(), 1);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].is_empty());
    assert_eq!(calls[0].len() % 2, 0);
}

// ---------------------------------------------------------------------------
// TestIncompleteThenComplete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn incomplete_holds_callback_until_next_pause() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 16]);
    pattern.extend(vec![true; 3]);
    pattern.extend(vec![false; 16]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, st_state) = ScriptedSmartTurn::new(vec![false, true]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1500)).await;
    }

    assert_eq!(st_state.lock().unwrap().buffers.len(), 2);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// TestExtendedSilenceAfterIncomplete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_reclassification_during_continued_silence() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 24]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, st_state) = ScriptedSmartTurn::new(vec![false]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1500)).await;
    }

    assert_eq!(st_state.lock().unwrap().buffers.len(), 1);
    assert!(calls.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// TestForceCompleteIfPending
// ---------------------------------------------------------------------------

#[tokio::test]
async fn force_complete_emits_buffer_after_incomplete_verdict() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 12]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, _) = ScriptedSmartTurn::new(vec![false]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1500)).await;
    }
    assert!(calls.lock().unwrap().is_empty());

    let fired = ep.force_complete_if_pending().await;

    assert!(fired);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].is_empty());
}

#[tokio::test]
async fn force_complete_fires_on_post_incomplete_even_if_buffer_drained() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 20]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, _) = ScriptedSmartTurn::new(vec![false]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1500)).await;
    }

    let fired = ep.force_complete_if_pending().await;

    assert!(fired);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn force_complete_emits_buffer_when_speech_never_classified() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (vad, _) = PatternVad::new(vec![true; 4]);
    let (st, st_state) = ScriptedSmartTurn::new(vec![]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..4 {
        ep.feed_audio(&input_chunk(1500)).await;
    }
    assert_eq!(st_state.lock().unwrap().buffers.len(), 0);

    let fired = ep.force_complete_if_pending().await;

    assert!(fired);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn force_complete_noop_when_nothing_pending() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (vad, _) = PatternVad::new(vec![false]);
    let (st, _) = ScriptedSmartTurn::new(vec![]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 100);

    for _ in 0..8 {
        ep.feed_audio(&input_chunk(0)).await;
    }

    let fired = ep.force_complete_if_pending().await;

    assert!(!fired);
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn force_complete_does_not_refire_after_normal_completion() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 5];
    pattern.extend(vec![false; 12]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, _) = ScriptedSmartTurn::new(vec![true]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1000)).await;
    }
    assert_eq!(calls.lock().unwrap().len(), 1);

    let fired = ep.force_complete_if_pending().await;

    assert!(!fired);
    assert_eq!(calls.lock().unwrap().len(), 1); // no duplicate emission
}

// ---------------------------------------------------------------------------
// TestReset
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reset_drops_buffer_and_state() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 16]);
    let (vad, _) = PatternVad::new(pattern);
    let (st, st_state) = ScriptedSmartTurn::new(vec![true]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    for _ in 0..2 {
        ep.feed_audio(&input_chunk(1500)).await;
    }
    ep.reset();
    for _ in 0..8 {
        ep.feed_audio(&input_chunk(0)).await;
    }

    assert_eq!(st_state.lock().unwrap().buffers.len(), 0);
    assert!(calls.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// TestSmartTurnOffloaded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_complete_runs_off_event_loop_thread() {
    // Smart Turn ONNX inference must dispatch off the reactor thread (spec §27):
    // sync wav2vec2 over up to 16 s of audio would trip Discord's heartbeat and
    // Deepgram's keepalive. Pin the offload via thread ids.
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut pattern = vec![true; 3];
    pattern.extend(vec![false; 16]);
    let (vad, _) = PatternVad::new(pattern.clone());
    let (st, st_state) = ScriptedSmartTurn::new(vec![true]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 32);

    let loop_thread = std::thread::current().id();
    for _ in 0..pattern.len() {
        ep.feed_audio(&input_chunk(1500)).await;
    }

    let seen = st_state.lock().unwrap().threads.clone();
    assert!(!seen.is_empty(), "is_turn_complete should have been called");
    assert!(
        !seen.contains(&loop_thread),
        "is_turn_complete blocked the reactor thread"
    );
}

// ---------------------------------------------------------------------------
// TestSubchunkFraming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn partial_chunks_buffer_until_full_vad_window() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (vad, vad_state) = PatternVad::new(vec![false]);
    let (st, _) = ScriptedSmartTurn::new(vec![]);
    let mut ep = UtteranceEndpointer::new(vad, st, capture_callback(Arc::clone(&calls)), 200, 100);

    let half = INPUT_CHUNK_BYTES / 2;
    ep.feed_audio(&vec![0u8; half]).await;
    assert_eq!(vad_state.lock().unwrap().is_speech_calls, 0);
    ep.feed_audio(&vec![0u8; INPUT_CHUNK_BYTES - half]).await;
    assert_eq!(vad_state.lock().unwrap().is_speech_calls, 1);
}
