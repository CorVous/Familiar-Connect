//! PCM conversion + resampler + streaming source (subsystem 09; Python
//! `voice/audio.py`).
//!
//! Pure audio primitives for the Discord voice path. The averaging here uses
//! floor division (`div_euclid`) to match Python's `//` bit-for-bit — Rust's
//! integer `/` truncates toward zero, which differs at negatives (DESIGN §4.3;
//! byte-pinned by `tests/test_audio_resample.py`). All samples are little-endian
//! `i16` (`<i2`).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// Discord requires 48 kHz s16le stereo PCM in 20 ms frames:
/// `48000 * 2ch * 2B * 0.020s = 3840` bytes/frame.
pub const DISCORD_FRAME_SIZE: usize = 3840;

/// 3:1 decimation — 48000 / 16000.
const RESAMPLE_RATIO: usize = 3;

/// Errors from the PCM conversion primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AudioError {
    /// Input length is not `i16`-aligned (odd byte count).
    #[error("PCM data length must be even, got {0}")]
    OddLength(usize),
    /// Stereo input length is not a whole number of L/R frames.
    #[error("Stereo PCM data length must be divisible by 4, got {0}")]
    NotDivisibleByFour(usize),
}

/// Streaming 48 kHz → 16 kHz `i16` PCM resampler.
///
/// Stateful: holds up to two `i16` samples between calls so callers can feed
/// arbitrary chunk lengths. Each output sample is the integer floor-mean of
/// three consecutive inputs (boxcar pre-filter + 3:1 decimation). TEN-VAD
/// tolerates the residual aliasing above the 8 kHz Nyquist.
#[derive(Clone, Debug, Default)]
pub struct Resampler48to16 {
    /// Held samples carried into the next `feed` (0, 1, or 2).
    carry: Vec<i16>,
}

impl Resampler48to16 {
    /// A resampler with no held state.
    #[must_use]
    pub const fn new() -> Self {
        Self { carry: Vec::new() }
    }

    /// Drop the held remainder; the next `feed` starts a fresh triplet.
    pub fn reset(&mut self) {
        self.carry.clear();
    }

    /// Resample arbitrary-length 48 kHz `i16` PCM to 16 kHz `i16` PCM.
    ///
    /// # Errors
    /// [`AudioError::OddLength`] when `pcm_48k` has an odd byte length.
    // The boxcar mean of three `i16` samples always lies within `i16` range, so
    // the `as i16` narrowing cannot truncate.
    #[allow(clippy::cast_possible_truncation)]
    pub fn feed(&mut self, pcm_48k: &[u8]) -> Result<Vec<u8>, AudioError> {
        if pcm_48k.len() % 2 != 0 {
            return Err(AudioError::OddLength(pcm_48k.len()));
        }
        let n_in = pcm_48k.len() / 2;
        if n_in == 0 && self.carry.is_empty() {
            return Ok(Vec::new());
        }
        let mut samples: Vec<i16> = std::mem::take(&mut self.carry);
        samples.reserve(n_in);
        for chunk in pcm_48k.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        let n_full = samples.len() / RESAMPLE_RATIO;
        let mut out = Vec::with_capacity(n_full * 2);
        for i in 0..n_full {
            let base = i * RESAMPLE_RATIO;
            let sum = i32::from(samples[base])
                + i32::from(samples[base + 1])
                + i32::from(samples[base + 2]);
            let avg = sum.div_euclid(3) as i16;
            out.extend_from_slice(&avg.to_le_bytes());
        }
        self.carry = samples.split_off(n_full * RESAMPLE_RATIO);
        Ok(out)
    }

    /// Flush the held remainder, zero-padding to a full triplet (at most one
    /// output sample; empty carry → empty).
    #[allow(clippy::cast_possible_truncation)]
    pub fn close(&mut self) -> Vec<u8> {
        if self.carry.is_empty() {
            return Vec::new();
        }
        let mut samples = std::mem::take(&mut self.carry);
        while samples.len() < RESAMPLE_RATIO {
            samples.push(0);
        }
        let sum = i32::from(samples[0]) + i32::from(samples[1]) + i32::from(samples[2]);
        let avg = sum.div_euclid(3) as i16;
        avg.to_le_bytes().to_vec()
    }
}

/// Duplicate each `i16` sample into L+R; produces 2x output.
///
/// # Errors
/// [`AudioError::OddLength`] when `data` has an odd byte length.
pub fn mono_to_stereo(data: &[u8]) -> Result<Vec<u8>, AudioError> {
    if data.len() % 2 != 0 {
        return Err(AudioError::OddLength(data.len()));
    }
    let mut out = Vec::with_capacity(data.len() * 2);
    for chunk in data.chunks_exact(2) {
        out.extend_from_slice(chunk);
        out.extend_from_slice(chunk);
    }
    Ok(out)
}

/// Average L+R `i16` samples into mono; produces 0.5x output.
///
/// # Errors
/// [`AudioError::NotDivisibleByFour`] when `data`'s length is not a multiple of
/// four.
// `(left + right) / 2` for two `i16` values stays within `i16` range.
#[allow(clippy::cast_possible_truncation)]
pub fn stereo_to_mono(data: &[u8]) -> Result<Vec<u8>, AudioError> {
    if data.len() % 4 != 0 {
        return Err(AudioError::NotDivisibleByFour(data.len()));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for frame in data.chunks_exact(4) {
        let left = i16::from_le_bytes([frame[0], frame[1]]);
        let right = i16::from_le_bytes([frame[2], frame[3]]);
        let avg = (i32::from(left) + i32::from(right)).div_euclid(2) as i16;
        out.extend_from_slice(&avg.to_le_bytes());
    }
    Ok(out)
}

/// Buffer + priming/EOS state guarded by the source's condvar.
struct SourceState {
    buf: VecDeque<u8>,
    closed: bool,
    primed: bool,
}

/// Thread-safe streaming PCM source for the voice player.
///
/// The producer (an async task) calls [`feed`](StreamingPcmSource::feed) with
/// stereo s16le bytes; the audio thread drains 20 ms frames via
/// [`read`](StreamingPcmSource::read). [`close_input`](StreamingPcmSource::close_input)
/// flips an EOS flag — once the buffer drains, `read` returns empty bytes so the
/// player stops. Without the flag, `read` blocks on the condvar until fed.
///
/// Two opt-in jitter-buffer knobs smooth bursty producers:
/// * `prebuffer_bytes` — the first `read` blocks until at least this many bytes
///   are buffered (or EOS), building a cushion before playback. `0` (default)
///   starts immediately.
/// * `pad_underrun` — in steady state, an empty-but-open buffer returns one
///   frame of silence instead of blocking. `false` (default) blocks. EOS always
///   overrides padding so playback ends instead of emitting silence forever.
pub struct StreamingPcmSource {
    state: Mutex<SourceState>,
    cond: Condvar,
    prebuffer_bytes: usize,
    pad_underrun: bool,
}

impl StreamingPcmSource {
    /// Construct a source with the given jitter-buffer knobs.
    #[must_use]
    pub const fn new(prebuffer_bytes: usize, pad_underrun: bool) -> Self {
        Self {
            state: Mutex::new(SourceState {
                buf: VecDeque::new(),
                closed: false,
                // Pre-roll gates the first read only when a threshold is set.
                primed: prebuffer_bytes == 0,
            }),
            cond: Condvar::new(),
            prebuffer_bytes,
            pad_underrun,
        }
    }

    /// Append `data` (stereo s16le); notify a waiting reader. Empty feed is a
    /// no-op.
    pub fn feed(&self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.state
            .lock()
            .expect("StreamingPcmSource mutex poisoned")
            .buf
            .extend(data);
        self.cond.notify_one();
    }

    /// Signal EOS; the reader drains then returns empty bytes.
    pub fn close_input(&self) {
        self.state
            .lock()
            .expect("StreamingPcmSource mutex poisoned")
            .closed = true;
        self.cond.notify_all();
    }

    /// Drain one 3840-byte frame (blocking per the jitter knobs), or empty bytes
    /// at EOS. Called from the audio thread.
    ///
    /// The lock scope deliberately spans the condvar waits and the buffer
    /// inspection — that is the reader half of the producer/consumer protocol,
    /// so [`significant_drop_tightening`](clippy::significant_drop_tightening) is
    /// suppressed here.
    #[must_use]
    #[allow(clippy::significant_drop_tightening)]
    pub fn read(&self) -> Vec<u8> {
        let mut state = self
            .state
            .lock()
            .expect("StreamingPcmSource mutex poisoned");
        if !state.primed {
            // Pre-roll: build a cushion before the first frame plays; EOS
            // overrides so a short reply still plays out.
            while state.buf.len() < self.prebuffer_bytes && !state.closed {
                state = self.cond.wait(state).expect("condvar wait");
            }
            state.primed = true;
        }

        if state.buf.len() < DISCORD_FRAME_SIZE && !state.closed {
            // Underrun while still producing: pad with silence (opt-in) or block.
            if self.pad_underrun {
                return vec![0u8; DISCORD_FRAME_SIZE];
            }
            while state.buf.len() < DISCORD_FRAME_SIZE && !state.closed {
                state = self.cond.wait(state).expect("condvar wait");
            }
        }

        if state.buf.len() >= DISCORD_FRAME_SIZE {
            return state.buf.drain(..DISCORD_FRAME_SIZE).collect();
        }
        if !state.buf.is_empty() {
            // Zero-pad the final partial frame so it plays before stop.
            let mut out: Vec<u8> = state.buf.drain(..).collect();
            out.resize(DISCORD_FRAME_SIZE, 0);
            return out;
        }
        Vec::new()
    }

    /// Always `false` — the source yields raw PCM, never Opus.
    #[must_use]
    pub const fn is_opus(&self) -> bool {
        false
    }

    /// Called by the player on stop; releases any blocked reader (== EOS).
    pub fn cleanup(&self) {
        self.close_input();
    }
}

impl Default for StreamingPcmSource {
    /// No pre-roll, block on underrun (the Cartesia steady-cadence default).
    fn default() -> Self {
        Self::new(0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AudioError, DISCORD_FRAME_SIZE, Resampler48to16, StreamingPcmSource, mono_to_stereo,
        stereo_to_mono,
    };
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    fn pcm(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    fn unpack1(bytes: &[u8]) -> i16 {
        i16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn unpack(bytes: &[u8]) -> Vec<i16> {
        bytes.chunks_exact(2).map(unpack1).collect()
    }

    // --- Resampler48to16 (test_audio_resample.py) --------------------------

    #[test]
    fn resample_empty_input_returns_empty() {
        let mut r = Resampler48to16::new();
        assert!(r.feed(b"").unwrap().is_empty());
    }

    #[test]
    fn three_input_samples_yield_one_output_sample() {
        let mut r = Resampler48to16::new();
        let out = r.feed(&pcm(&[100, 200, 300])).unwrap();
        assert_eq!(unpack1(&out), 200);
    }

    #[test]
    fn output_length_is_one_third_input() {
        let mut r = Resampler48to16::new();
        let out = r.feed(&pcm(&[0, 0, 0, 100, 100, 100])).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn carries_remainder_across_feed_calls() {
        let mut r = Resampler48to16::new();
        let out1 = r.feed(&pcm(&[100, 200])).unwrap();
        let out2 = r.feed(&pcm(&[300, 400])).unwrap();
        assert!(out1.is_empty());
        assert_eq!(unpack1(&out2), 200);
    }

    #[test]
    fn close_drains_carry_with_zero_padding() {
        let mut r = Resampler48to16::new();
        let _ = r.feed(&pcm(&[300, 300])).unwrap();
        let out = r.close();
        assert_eq!(unpack1(&out), 200);
    }

    #[test]
    fn close_when_aligned_emits_nothing() {
        let mut r = Resampler48to16::new();
        let _ = r.feed(&pcm(&[0, 0, 0])).unwrap();
        assert!(r.close().is_empty());
    }

    #[test]
    fn reset_drops_held_remainder() {
        let mut r = Resampler48to16::new();
        let _ = r.feed(&pcm(&[100, 200])).unwrap();
        r.reset();
        let out = r.feed(&pcm(&[90, 90, 90])).unwrap();
        assert_eq!(unpack1(&out), 90);
    }

    #[test]
    fn rejects_odd_byte_length() {
        let mut r = Resampler48to16::new();
        assert_eq!(r.feed(b"\x00"), Err(AudioError::OddLength(1)));
    }

    #[test]
    fn clipped_to_int16_range() {
        let mut r = Resampler48to16::new();
        let out = r.feed(&pcm(&[32767, 32767, 32767])).unwrap();
        assert_eq!(unpack1(&out), 32767);
    }

    #[test]
    fn negative_values_average_correctly() {
        // Floor division: (-100 + -200 + -300) // 3 == -200 (matches Python `//`).
        let mut r = Resampler48to16::new();
        let out = r.feed(&pcm(&[-100, -200, -300])).unwrap();
        assert_eq!(unpack1(&out), -200);
    }

    #[test]
    fn floor_division_at_negative_boundary() {
        // Pin the div_euclid vs truncation contrast: (-3 + 0 + 0) // 3 == -1 in
        // Python; -3/3 == -1 in Rust too, but (-1 + 0 + 0) // 3 == -1 (floor)
        // whereas truncation gives 0. Use a triplet summing to -1.
        let mut r = Resampler48to16::new();
        let out = r.feed(&pcm(&[-1, 0, 0])).unwrap();
        assert_eq!(unpack1(&out), -1);
    }

    // --- mono_to_stereo (test_voice_audio.py) ------------------------------

    #[test]
    fn mono_to_stereo_doubles_length() {
        assert_eq!(mono_to_stereo(&[0u8; 8]).unwrap().len(), 16);
    }

    #[test]
    fn mono_to_stereo_duplicates_each_sample() {
        let mono = pcm(&[0x0102, 0x0304]);
        let stereo = unpack(&mono_to_stereo(&mono).unwrap());
        assert_eq!(stereo, vec![0x0102, 0x0102, 0x0304, 0x0304]);
    }

    #[test]
    fn mono_to_stereo_empty_returns_empty() {
        assert!(mono_to_stereo(b"").unwrap().is_empty());
    }

    #[test]
    fn mono_to_stereo_raises_on_odd_length() {
        assert_eq!(
            mono_to_stereo(b"\x00\x01\x02"),
            Err(AudioError::OddLength(3))
        );
    }

    #[test]
    fn mono_to_stereo_single_sample() {
        let stereo = unpack(&mono_to_stereo(&pcm(&[-1000])).unwrap());
        assert_eq!(stereo, vec![-1000, -1000]);
    }

    #[test]
    fn mono_to_stereo_preserves_sample_values() {
        let samples = [100, -200, 32767, -32768, 0];
        let stereo = unpack(&mono_to_stereo(&pcm(&samples)).unwrap());
        for (i, &original) in samples.iter().enumerate() {
            assert_eq!(stereo[i * 2], original);
            assert_eq!(stereo[i * 2 + 1], original);
        }
    }

    #[test]
    fn mono_to_stereo_byte_identical_to_reference_duplication() {
        for samples in [
            vec![],
            vec![0],
            vec![-32768],
            vec![32767],
            vec![0x0102, 0x0304],
            vec![100, -200, 32767, -32768, 0],
            vec![-1, 1, -32768, 32767, 12345, -12345, 0],
        ] {
            let mono = pcm(&samples);
            // Independent reference: each 2-byte LE sample appears twice in order.
            let mut expected = Vec::new();
            for chunk in mono.chunks_exact(2) {
                expected.extend_from_slice(chunk);
                expected.extend_from_slice(chunk);
            }
            assert_eq!(mono_to_stereo(&mono).unwrap(), expected);
        }
    }

    // --- stereo_to_mono ----------------------------------------------------

    #[test]
    fn stereo_to_mono_halves_length() {
        assert_eq!(stereo_to_mono(&[0u8; 16]).unwrap().len(), 8);
    }

    #[test]
    fn stereo_to_mono_averages_left_and_right() {
        let out = stereo_to_mono(&pcm(&[100, 200])).unwrap();
        assert_eq!(unpack1(&out), 150);
    }

    #[test]
    fn stereo_to_mono_empty_returns_empty() {
        assert!(stereo_to_mono(b"").unwrap().is_empty());
    }

    #[test]
    fn stereo_to_mono_raises_on_invalid_length() {
        assert_eq!(
            stereo_to_mono(b"\x00\x01\x02"),
            Err(AudioError::NotDivisibleByFour(3))
        );
    }

    #[test]
    fn stereo_to_mono_single_frame() {
        let out = stereo_to_mono(&pcm(&[-1000, -1000])).unwrap();
        assert_eq!(unpack1(&out), -1000);
    }

    #[test]
    fn stereo_to_mono_roundtrip() {
        let original = pcm(&[100, -200, 32767, -32768, 0]);
        let round = stereo_to_mono(&mono_to_stereo(&original).unwrap()).unwrap();
        assert_eq!(round, original);
    }

    // --- DISCORD_FRAME_SIZE ------------------------------------------------

    #[test]
    fn frame_size_is_3840() {
        assert_eq!(DISCORD_FRAME_SIZE, 3840);
    }

    // --- StreamingPcmSource ------------------------------------------------

    #[test]
    fn read_returns_full_frame_when_buffered() {
        let src = StreamingPcmSource::default();
        src.feed(&vec![0xab; DISCORD_FRAME_SIZE * 2]);
        let first = src.read();
        let second = src.read();
        assert_eq!(first.len(), DISCORD_FRAME_SIZE);
        assert_eq!(second.len(), DISCORD_FRAME_SIZE);
        assert_eq!(first, vec![0xab; DISCORD_FRAME_SIZE]);
    }

    #[test]
    fn read_returns_empty_when_closed_and_drained() {
        let src = StreamingPcmSource::default();
        src.close_input();
        assert!(src.read().is_empty());
    }

    #[test]
    fn partial_frame_zero_padded_on_close() {
        let src = StreamingPcmSource::default();
        let partial = [0x01, 0x02, 0x03, 0x04];
        src.feed(&partial);
        src.close_input();
        let out = src.read();
        assert_eq!(out.len(), DISCORD_FRAME_SIZE);
        assert_eq!(&out[..partial.len()], &partial);
        assert!(out[partial.len()..].iter().all(|&b| b == 0));
        assert!(src.read().is_empty());
    }

    #[test]
    fn read_blocks_until_data_arrives() {
        let src = Arc::new(StreamingPcmSource::default());
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
        // Reader is blocked: no result within the grace window.
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        src.feed(&vec![0x42; DISCORD_FRAME_SIZE]);
        let got = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(got, vec![0x42; DISCORD_FRAME_SIZE]);
        handle.join().unwrap();
    }

    #[test]
    fn close_unblocks_reader() {
        let src = Arc::new(StreamingPcmSource::default());
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        src.close_input();
        let got = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(got.is_empty());
        handle.join().unwrap();
    }

    #[test]
    fn is_opus_false() {
        assert!(!StreamingPcmSource::default().is_opus());
    }

    #[test]
    fn cleanup_closes_input() {
        let src = StreamingPcmSource::default();
        src.cleanup();
        assert!(src.read().is_empty());
    }

    #[test]
    fn feed_empty_is_noop() {
        let src = StreamingPcmSource::default();
        src.feed(b"");
        src.close_input();
        assert!(src.read().is_empty());
    }

    // --- StreamingPcmSource jitter buffer ----------------------------------

    #[test]
    fn preroll_blocks_until_threshold_met() {
        let src = Arc::new(StreamingPcmSource::new(DISCORD_FRAME_SIZE * 2, false));
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
        // Sub-threshold feed: one frame, below the 2-frame gate.
        src.feed(&vec![0x11; DISCORD_FRAME_SIZE]);
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        // Cross the threshold; reader unblocks and returns the first frame.
        src.feed(&vec![0x11; DISCORD_FRAME_SIZE]);
        let got = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(got, vec![0x11; DISCORD_FRAME_SIZE]);
        handle.join().unwrap();
    }

    #[test]
    fn preroll_only_gates_first_read() {
        let src = StreamingPcmSource::new(DISCORD_FRAME_SIZE, false);
        src.feed(&vec![0x22; DISCORD_FRAME_SIZE * 2]);
        let first = src.read(); // primes
        let second = src.read(); // must not block on the gate again
        assert_eq!(first, vec![0x22; DISCORD_FRAME_SIZE]);
        assert_eq!(second, vec![0x22; DISCORD_FRAME_SIZE]);
    }

    #[test]
    fn eos_overrides_preroll() {
        let src = StreamingPcmSource::new(DISCORD_FRAME_SIZE * 4, false);
        let partial = [0x07, 0x08, 0x09, 0x0a];
        src.feed(&partial);
        src.close_input();
        let out = src.read();
        assert_eq!(out.len(), DISCORD_FRAME_SIZE);
        assert_eq!(&out[..partial.len()], &partial);
        assert!(src.read().is_empty());
    }

    #[test]
    fn underrun_pads_silence_without_blocking() {
        let src = Arc::new(StreamingPcmSource::new(0, true));
        src.feed(&vec![0x33; DISCORD_FRAME_SIZE]);
        assert_eq!(src.read(), vec![0x33; DISCORD_FRAME_SIZE]);
        // Buffer empty and NOT closed: must return silence, not block.
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
        let got = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(got, vec![0u8; DISCORD_FRAME_SIZE]);
        handle.join().unwrap();
    }

    #[test]
    fn eos_overrides_underrun_padding() {
        let src = StreamingPcmSource::new(0, true);
        src.feed(&vec![0x44; DISCORD_FRAME_SIZE]);
        assert_eq!(src.read(), vec![0x44; DISCORD_FRAME_SIZE]);
        src.close_input();
        assert!(src.read().is_empty());
    }

    #[test]
    fn default_underrun_still_blocks() {
        let src = Arc::new(StreamingPcmSource::default());
        src.feed(&vec![0x55; DISCORD_FRAME_SIZE]);
        assert_eq!(src.read(), vec![0x55; DISCORD_FRAME_SIZE]);
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        src.close_input(); // release the blocked reader for cleanup
        let got = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(got.is_empty());
        handle.join().unwrap();
    }
}
