//! Per-SSRC recording sink → mono PCM fan-in (subsystem 09; Python
//! `voice/recording_sink.py`).
//!
//! Bridges decoded Discord voice audio to the transcription pump. In Python this
//! is a py-cord `Sink` whose `write(data, user)` is called from a background
//! recording thread; it converts 48 kHz stereo s16le → mono `((L+R)//2)` and
//! hands `(user_id, mono)` to an asyncio queue via `call_soon_threadsafe`.
//!
//! Under songbird the recording thread is gone — the driver delivers per-SSRC
//! **decoded** PCM in 20 ms ticks. This port keeps the package serenity-free by
//! modelling that against a [`VoiceTick`]-shaped seam plus an [`SsrcResolver`]
//! (songbird's SSRC→user map): [`RecordingSink::on_tick`] demuxes each speaking
//! SSRC, resolves its user id, converts stereo→mono, and fans the tagged
//! `(user_id, mono)` tuple into an unbounded channel. The tokio `mpsc` sender is
//! `Send + Sync`, so the Python `call_soon_threadsafe` thread-hop collapses to a
//! plain `send` (the queue stays unbounded — backpressure is deliberately
//! absent here, spec 09 §15).
//!
//! The per-user *transcriber clone* and idle watchdog live in the Layer-4 voice
//! source that drains this channel — keeping this Layer-2 package free of the
//! `stt` seam (see the port notes / deviations).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc::UnboundedSender;

use crate::log_style as ls;
use crate::voice::audio::{AudioError, stereo_to_mono};

/// Log target mirroring the Python logger name.
const LOG_TARGET: &str = "familiar_connect.voice.recording_sink";

/// A `(user_id, mono s16le 48 kHz PCM)` tuple pushed to the transcription pump.
pub type AudioChunk = (u64, Vec<u8>);

/// One driver tick's decoded per-SSRC audio (songbird `VoiceTick` shape).
///
/// `speaking` maps an RTP SSRC to its decoded 48 kHz **stereo** s16le PCM for
/// this 20 ms tick; `silent` lists SSRCs that produced no audio. Kept as owned
/// bytes so this package never names a songbird type.
#[derive(Clone, Debug, Default)]
pub struct VoiceTick {
    /// SSRC → decoded 48 kHz stereo s16le PCM for this tick.
    pub speaking: BTreeMap<u32, Vec<u8>>,
    /// SSRCs marked silent this tick.
    pub silent: BTreeSet<u32>,
}

/// Resolves an RTP SSRC to a Discord user id (songbird's speaking-state map).
pub trait SsrcResolver: Send + Sync {
    /// The Discord user id owning `ssrc`, if known this session.
    fn user_id(&self, ssrc: u32) -> Option<u64>;
}

/// Stereo→mono + tagged fan-in bridge for transcription.
pub struct RecordingSink {
    audio_out: UnboundedSender<AudioChunk>,
    finished: AtomicBool,
    /// Latched once the first unmapped SSRC is routed under a provisional id, so
    /// the warn (below) fires exactly once instead of per-20 ms-tick.
    warned_unmapped: AtomicBool,
}

impl RecordingSink {
    /// Build a sink that fans `(user_id, mono)` tuples into `audio_out`.
    #[must_use]
    pub const fn new(audio_out: UnboundedSender<AudioChunk>) -> Self {
        Self {
            audio_out,
            finished: AtomicBool::new(false),
            warned_unmapped: AtomicBool::new(false),
        }
    }

    /// Convert one 48 kHz stereo s16le buffer to mono and push `(user, mono)`.
    ///
    /// Mirrors the Python `write(data, user)`: must not block. A closed receiver
    /// (pump gone) drops the chunk — the sink never errors on send.
    ///
    /// # Errors
    /// [`AudioError::NotDivisibleByFour`] when `data` is not whole stereo frames.
    pub fn write(&self, data: &[u8], user: u64) -> Result<(), AudioError> {
        let mono = stereo_to_mono(data)?;
        // Plain debug string, byte-for-byte with Python (recording_sink.py:49-54):
        // this call site deliberately does NOT use the `log_style` tag/kv helpers.
        tracing::debug!(
            target: LOG_TARGET,
            "[Sink] user={} stereo={} bytes \u{2192} mono={} bytes",
            user,
            data.len(),
            mono.len(),
        );
        // Best-effort: the pump's receiver may be gone (channel teardown).
        let _ = self.audio_out.send((user, mono));
        Ok(())
    }

    /// Demux one driver tick: per speaking SSRC, resolve the user and fan in.
    ///
    /// When songbird's speaking-state map has no user for an SSRC yet, the frame
    /// is routed under a **provisional** user id equal to the SSRC rather than
    /// dropped ("first-audio-chunk lazy creation"; the turn is anonymous until the
    /// real mapping lands). This matters because the op-5 Speaking event that
    /// binds SSRC→user fires inconsistently (songbird PR #291) and DAVE decrypt is
    /// MLS-sender-identified, so decoded audio routinely arrives *before* any
    /// `SpeakingStateUpdate` — dropping it here silently eats the whole turn.
    /// SSRCs are `u32` (< 2^32) and Discord snowflakes are > 2^52, so a
    /// provisional id can never collide with a real user id. Malformed audio
    /// (not whole stereo frames) is still dropped.
    pub fn on_tick(&self, tick: &VoiceTick, resolver: &dyn SsrcResolver) {
        for (&ssrc, pcm) in &tick.speaking {
            let user = resolver.user_id(ssrc).unwrap_or_else(|| {
                if !self.warned_unmapped.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "[Sink] WARNING unmapped_ssrc={ssrc} action=provisional-id \
                         hint=op-5 Speaking not observed yet; transcript is anonymous \
                         until it lands"
                    );
                }
                u64::from(ssrc)
            });
            if let Err(err) = self.write(pcm, user) {
                tracing::debug!(
                    target: LOG_TARGET,
                    "{} {} {}",
                    ls::tag("Sink", ls::LC),
                    ls::kv_styled("dropped_ssrc", &ssrc.to_string(), ls::W, ls::LW),
                    ls::kv_styled("reason", &err.to_string(), ls::W, ls::LW),
                );
            }
        }
    }

    /// Signal recording finished (py-cord `cleanup`).
    pub fn cleanup(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    /// Whether [`cleanup`](Self::cleanup) has been called.
    #[must_use]
    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::{RecordingSink, SsrcResolver, VoiceTick};
    use std::collections::HashMap;

    fn stereo(left: i16, right: i16) -> Vec<u8> {
        let mut out = Vec::with_capacity(4);
        out.extend_from_slice(&left.to_le_bytes());
        out.extend_from_slice(&right.to_le_bytes());
        out
    }

    fn mono(sample: i16) -> Vec<u8> {
        sample.to_le_bytes().to_vec()
    }

    #[test]
    fn write_puts_user_id_and_mono_tuple_on_channel() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = RecordingSink::new(tx);
        // Stereo frame L=100, R=100 → mono=100.
        sink.write(&stereo(100, 100), 12345).unwrap();
        let (user, pcm) = rx.try_recv().unwrap();
        assert_eq!(user, 12345);
        assert_eq!(pcm, mono(100));
    }

    #[test]
    fn write_preserves_different_user_ids() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = RecordingSink::new(tx);
        sink.write(&stereo(50, 50), 111).unwrap();
        sink.write(&stereo(50, 50), 222).unwrap();
        assert_eq!(rx.try_recv().unwrap().0, 111);
        assert_eq!(rx.try_recv().unwrap().0, 222);
    }

    #[test]
    fn cleanup_sets_finished() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = RecordingSink::new(tx);
        assert!(!sink.finished());
        sink.cleanup();
        assert!(sink.finished());
    }

    struct MapResolver(HashMap<u32, u64>);
    impl SsrcResolver for MapResolver {
        fn user_id(&self, ssrc: u32) -> Option<u64> {
            self.0.get(&ssrc).copied()
        }
    }

    #[test]
    fn on_tick_demuxes_per_ssrc_and_tags_user() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = RecordingSink::new(tx);
        let mut tick = VoiceTick::default();
        tick.speaking.insert(10, stereo(200, 200)); // ssrc 10 → user 111, mono 200
        tick.speaking.insert(20, stereo(40, 60)); // ssrc 20 → user 222, mono (40+60)/2=50
        tick.speaking.insert(30, stereo(9, 9)); // ssrc 30 → unmapped → provisional id 30
        let resolver = MapResolver(HashMap::from([(10, 111), (20, 222)]));

        sink.on_tick(&tick, &resolver);

        // BTreeMap iterates by ascending SSRC → deterministic order 10, 20, 30.
        assert_eq!(rx.try_recv().unwrap(), (111, mono(200)));
        assert_eq!(rx.try_recv().unwrap(), (222, mono(50)));
        // Unmapped SSRC is no longer dropped: it fans in under a provisional user
        // id equal to the SSRC (anonymous turn until op-5 Speaking lands).
        assert_eq!(rx.try_recv().unwrap(), (30, mono(9)));
        assert!(rx.try_recv().is_err());
    }
}
