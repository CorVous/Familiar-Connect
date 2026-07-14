//! Audio-fixture integration tests for the endpointer (ported from Python
//! `tests/test_endpointer_audio_fixtures.py`).
//!
//! Synthesised 48 kHz mono int16 PCM flows through the **real**
//! [`Resampler48to16`] + framer + state machine. The VAD is stubbed but
//! thresholds on actual post-resample frame energy, so the audio fixture drives
//! the `IDLE → SPEAKING → silence-after-speech → classify` transitions the way
//! live audio would; Smart Turn pops a scripted verdict per call.

// Fixture math converts sample-count floats to indices; the durations are small
// non-negative constants, so the sign/truncation casts are intentional.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::type_complexity,
    clippy::significant_drop_tightening,
    clippy::new_ret_no_self
)]

use std::f64::consts::PI;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use familiar_connect::voice::audio::Resampler48to16;
use familiar_connect::voice::turn_detection::{
    OnTurnComplete, SmartTurn, UtteranceEndpointer, Vad,
};

const INPUT_SR: f64 = 48_000.0; // Discord-side sample rate fed to the endpointer
const VAD_SR: f64 = 16_000.0; // rate after the 3:1 decimator

// 16 kHz post-resample → 16 ms frame = 256 samples = 512 bytes.
const VAD_FRAME_BYTES_16K: usize = 256 * 2;

// Energy threshold (RMS) the VAD stub uses to flag a 16 ms frame as speech.
const VAD_ENERGY_THRESHOLD: f64 = 200.0;

// Speech tone — a 220 Hz sine at modest amplitude; the mid-band frequency
// survives 3:1 boxcar decimation with plenty of energy left.
const SPEECH_FREQ_HZ: f64 = 220.0;
const SPEECH_AMPLITUDE: f64 = 6000.0;

// --- fixture builders ------------------------------------------------------

fn silence_pcm(duration_ms: f64) -> Vec<u8> {
    let n_samples = (duration_ms / 1000.0 * INPUT_SR) as usize;
    vec![0u8; n_samples * 2]
}

fn speech_pcm(duration_ms: f64, phase_offset: f64) -> Vec<u8> {
    let n_samples = (duration_ms / 1000.0 * INPUT_SR) as usize;
    let omega = 2.0 * PI * SPEECH_FREQ_HZ / INPUT_SR;
    let mut out = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        #[allow(clippy::cast_possible_truncation)]
        let sample = (SPEECH_AMPLITUDE * (omega * i as f64 + phase_offset).sin()) as i16;
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

enum Seg {
    Speech(f64),
    Silence(f64),
}

fn build_fixture(segments: &[Seg]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut phase = 0.0_f64;
    for seg in segments {
        match *seg {
            Seg::Silence(ms) => out.extend_from_slice(&silence_pcm(ms)),
            Seg::Speech(ms) => {
                out.extend_from_slice(&speech_pcm(ms, phase));
                phase += 2.0 * PI * SPEECH_FREQ_HZ * (ms / 1000.0);
                phase %= 2.0 * PI;
            }
        }
    }
    out
}

fn frame_rms(frame: &[u8]) -> f64 {
    let n = frame.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let sq: f64 = frame
        .chunks_exact(2)
        .map(|c| {
            let s = f64::from(i16::from_le_bytes([c[0], c[1]]));
            s * s
        })
        .sum();
    (sq / n as f64).sqrt()
}

// --- stubs -----------------------------------------------------------------

/// VAD that thresholds on actual post-resample frame energy.
struct EnergyVad {
    threshold: f64,
}

impl Vad for EnergyVad {
    fn is_speech(&mut self, frame: &[u8]) -> bool {
        frame_rms(frame) >= self.threshold
    }
    fn reset(&mut self) {}
}

#[derive(Default)]
struct StState {
    idx: usize,
    buffers: Vec<Vec<u8>>,
}

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
        let verdict = self.verdicts[s.idx];
        s.idx += 1;
        verdict
    }
}

fn capture_callback(bucket: Arc<Mutex<Vec<Vec<u8>>>>) -> OnTurnComplete {
    Box::new(move |audio: Vec<u8>| {
        let bucket = Arc::clone(&bucket);
        let fut: Pin<Box<dyn Future<Output = ()> + Send>> = Box::pin(async move {
            bucket.lock().unwrap().push(audio);
        });
        fut
    })
}

fn energy_endpointer(
    verdicts: Vec<bool>,
    silence_ms: i64,
    speech_start_ms: i64,
) -> (
    UtteranceEndpointer,
    Arc<Mutex<Vec<Vec<u8>>>>,
    Arc<Mutex<StState>>,
) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let vad: Box<dyn Vad> = Box::new(EnergyVad {
        threshold: VAD_ENERGY_THRESHOLD,
    });
    let (st, st_state) = ScriptedSmartTurn::new(verdicts);
    let ep = UtteranceEndpointer::new(
        vad,
        st,
        capture_callback(Arc::clone(&calls)),
        silence_ms,
        speech_start_ms,
    );
    (ep, calls, st_state)
}

async fn feed_in_chunks(ep: &mut UtteranceEndpointer, pcm: &[u8]) {
    // 20 ms matches Discord's voice frame size; sub-frame boundaries exercise
    // the resampler/framer carry path.
    let chunk_bytes = (20.0 / 1000.0 * INPUT_SR) as usize * 2;
    let mut off = 0;
    while off < pcm.len() {
        let end = (off + chunk_bytes).min(pcm.len());
        ep.feed_audio(&pcm[off..end]).await;
        off = end;
    }
}

// ---------------------------------------------------------------------------
// TestCompleteSentenceFixture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn complete_sentence_fires_callback_once() {
    let (mut ep, calls, st_state) = energy_endpointer(vec![true], 200, 64);
    let pcm = build_fixture(&[
        Seg::Silence(80.0), // leading idle (dropped from buffer)
        Seg::Speech(600.0), // ~600 ms utterance
        Seg::Silence(400.0),
    ]);

    feed_in_chunks(&mut ep, &pcm).await;

    let buffers = st_state.lock().unwrap().buffers.clone();
    assert_eq!(buffers.len(), 1);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    // Buffer fed to SmartTurn equals the buffer surfaced to the callback.
    assert_eq!(buffers[0], calls[0]);
    // int16-aligned, and at least 60 % of the speech segment's worth.
    assert_eq!(calls[0].len() % 2, 0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let min_bytes = (0.6 * 600.0 / 1000.0 * VAD_SR * 2.0) as usize;
    assert!(calls[0].len() >= min_bytes);
}

// ---------------------------------------------------------------------------
// TestMidThoughtFixture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn short_mid_utterance_pause_does_not_trigger_classification() {
    let (mut ep, calls, st_state) = energy_endpointer(vec![true], 200, 64);
    let pcm = build_fixture(&[
        Seg::Speech(400.0),
        Seg::Silence(96.0), // ~6 VAD frames — below the 200 ms threshold
        Seg::Speech(400.0),
        Seg::Silence(400.0), // long trailing silence — fires classify
    ]);

    feed_in_chunks(&mut ep, &pcm).await;

    assert_eq!(st_state.lock().unwrap().buffers.len(), 1);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    // Buffer spans both speech bursts — ≥ 60 % of two 400 ms bursts.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let min_bytes = (0.6 * 800.0 / 1000.0 * VAD_SR * 2.0) as usize;
    assert!(calls[0].len() >= min_bytes);
}

#[tokio::test]
async fn pause_just_below_threshold_keeps_speaking_state() {
    let (mut ep, calls, st_state) = energy_endpointer(vec![true], 200, 64);
    // 144 ms gap = 9 VAD frames < 12-frame (200 ms) threshold.
    let pcm = build_fixture(&[
        Seg::Speech(300.0),
        Seg::Silence(144.0),
        Seg::Speech(300.0),
        Seg::Silence(400.0),
    ]);

    feed_in_chunks(&mut ep, &pcm).await;

    assert_eq!(st_state.lock().unwrap().buffers.len(), 1);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// TestFillerFixture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filler_holds_callback_until_resumed_speech_classifies_complete() {
    // incomplete (filler) → complete (true end of turn).
    let (mut ep, calls, st_state) = energy_endpointer(vec![false, true], 200, 64);
    let pcm = build_fixture(&[
        Seg::Speech(300.0),  // "uh"
        Seg::Silence(320.0), // past silence_ms — first classify
        Seg::Speech(500.0),  // resumed thought
        Seg::Silence(400.0), // past silence_ms — second classify
    ]);

    feed_in_chunks(&mut ep, &pcm).await;

    let buffers = st_state.lock().unwrap().buffers.clone();
    // Two classify calls: filler (incomplete) and resumed (complete).
    assert_eq!(buffers.len(), 2);
    // Only one callback — the second, complete verdict.
    assert_eq!(calls.lock().unwrap().len(), 1);
    // The complete-verdict buffer must exceed the filler buffer.
    assert!(buffers[1].len() > buffers[0].len());
}

#[tokio::test]
async fn filler_then_extended_silence_does_not_refire() {
    let (mut ep, calls, st_state) = energy_endpointer(vec![false], 200, 64);
    let pcm = build_fixture(&[
        Seg::Speech(300.0),
        Seg::Silence(800.0), // long tail past the threshold + lots more
    ]);

    feed_in_chunks(&mut ep, &pcm).await;

    // SmartTurn fires exactly once — silence after an `incomplete` verdict does
    // not retrigger classification without fresh speech.
    assert_eq!(st_state.lock().unwrap().buffers.len(), 1);
    assert!(calls.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// TestFixtureBuilders — pin the fixture-helper invariants
// ---------------------------------------------------------------------------

#[test]
fn silence_fixture_has_zero_rms() {
    let pcm = silence_pcm(20.0);
    // 20 ms at 48 kHz mono int16 → 1920 bytes.
    assert_eq!(pcm.len(), (0.020 * INPUT_SR) as usize * 2);
    assert!(frame_rms(&pcm) < 1e-9);
}

#[test]
fn speech_fixture_passes_energy_threshold() {
    // Post-resample energy of a 220 Hz tone must clear the VAD threshold.
    let mut resampler = Resampler48to16::new();
    let decimated = resampler.feed(&speech_pcm(64.0, 0.0)).unwrap();
    let frame = &decimated[..VAD_FRAME_BYTES_16K];
    assert_eq!(frame.len(), VAD_FRAME_BYTES_16K);
    assert!(frame_rms(frame) >= VAD_ENERGY_THRESHOLD);
}
