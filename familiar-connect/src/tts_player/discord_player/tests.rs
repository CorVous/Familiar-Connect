//! Port of `tests/test_discord_voice_player.py` — buffered + streaming playback,
//! stereo conversion, skip paths, barge-in `vc.stop`, play-lock serialization,
//! cancel-then-speak drain, and the `playback_start` budget stamp.

// Test ergonomics: holding the `std::Mutex` singleton guard across `.await`
// deliberately serializes budget/collector singleton access for the whole test,
// and holding short-lived `plays`/state guards across assertions is contention-free.
#![allow(clippy::significant_drop_tightening, clippy::await_holding_lock)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt as _;

use super::{AudioSource, DiscordVoicePlayer, PlayError, VoiceClientLike};
use crate::bus::envelope::TurnScope;
use crate::diagnostics::collector::{get_span_collector, reset_span_collector};
use crate::diagnostics::testutil::singleton_guard;
use crate::diagnostics::voice_budget::{
    PHASE_TTS_FIRST_AUDIO, get_voice_budget_recorder, reset_voice_budget_recorder,
};
use crate::tts::{JitterHints, StreamingTtsClient, TTSResult, TtsClient, TtsError, TtsStream};
use crate::tts_player::protocol::TtsPlayer;
use crate::voice::audio::{DISCORD_FRAME_SIZE, mono_to_stereo};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn mono_pcm(count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * 2);
    for i in 0..count {
        out.extend_from_slice(&i16::try_from(i).unwrap().to_le_bytes());
    }
    out
}

fn scope(turn: &str) -> TurnScope {
    TurnScope::new(turn, "voice:1")
}

// --- stub TTS clients ------------------------------------------------------

struct StubTts {
    audio: Vec<u8>,
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl TtsClient for StubTts {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        self.calls.lock().unwrap().push(text.to_owned());
        Ok(TTSResult::audio_only(self.audio.clone()))
    }
}

fn stub_tts(audio: Vec<u8>) -> (Arc<StubTts>, Arc<Mutex<Vec<String>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(StubTts {
            audio,
            calls: Arc::clone(&calls),
        }),
        calls,
    )
}

struct StreamingStubTts {
    chunks: Vec<Vec<u8>>,
    delay: Duration,
    calls: Arc<Mutex<Vec<String>>>,
    consumed: Arc<AtomicBool>,
    prebuffer: usize,
    pad: bool,
}

#[async_trait]
impl TtsClient for StreamingStubTts {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        self.calls.lock().unwrap().push(text.to_owned());
        Ok(TTSResult::audio_only(self.chunks.concat()))
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        Some(self)
    }
}

impl StreamingTtsClient for StreamingStubTts {
    fn synthesize_stream(&self, text: &str) -> TtsStream {
        self.calls.lock().unwrap().push(text.to_owned());
        let chunks = self.chunks.clone();
        let delay = self.delay;
        let consumed = Arc::clone(&self.consumed);
        futures::stream::unfold(
            (0usize, chunks, delay, consumed),
            |(i, chunks, delay, consumed)| async move {
                if i >= chunks.len() {
                    consumed.store(true, Ordering::SeqCst);
                    return None;
                }
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let chunk = chunks[i].clone();
                Some((Ok(chunk), (i + 1, chunks, delay, consumed)))
            },
        )
        .boxed()
    }

    fn jitter_hints(&self) -> JitterHints {
        JitterHints {
            prebuffer_bytes: self.prebuffer,
            pad_underrun: self.pad,
        }
    }
}

fn streaming_stub(
    chunks: Vec<Vec<u8>>,
    delay: Duration,
    prebuffer: usize,
    pad: bool,
) -> Arc<StreamingStubTts> {
    Arc::new(StreamingStubTts {
        chunks,
        delay,
        calls: Arc::new(Mutex::new(Vec::new())),
        consumed: Arc::new(AtomicBool::new(false)),
        prebuffer,
        pad,
    })
}

struct BoomTts {
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl TtsClient for BoomTts {
    async fn synthesize(&self, _text: &str) -> Result<TTSResult, TtsError> {
        Ok(TTSResult::default())
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        Some(self)
    }
}

impl StreamingTtsClient for BoomTts {
    fn synthesize_stream(&self, text: &str) -> TtsStream {
        self.calls.lock().unwrap().push(text.to_owned());
        futures::stream::once(async { Err(TtsError::Runtime("ws auth failed".to_owned())) }).boxed()
    }
}

// --- mock voice client -----------------------------------------------------

type BoolFn = Box<dyn FnMut() -> bool + Send>;
type PlayFn = Box<dyn FnMut() -> Result<(), PlayError> + Send>;
type UnitFn = Box<dyn FnMut() + Send>;

struct MockVc {
    connected: bool,
    plays: Arc<Mutex<Vec<AudioSource>>>,
    stop_count: Arc<AtomicUsize>,
    is_playing: Mutex<BoolFn>,
    play: Mutex<PlayFn>,
    on_stop: Mutex<UnitFn>,
}

impl VoiceClientLike for MockVc {
    fn is_connected(&self) -> bool {
        self.connected
    }

    fn is_playing(&self) -> bool {
        (self.is_playing.lock().unwrap())()
    }

    fn play(&self, source: AudioSource) -> Result<(), PlayError> {
        let r = (self.play.lock().unwrap())();
        if r.is_ok() {
            self.plays.lock().unwrap().push(source);
        }
        r
    }

    fn stop(&self) {
        self.stop_count.fetch_add(1, Ordering::SeqCst);
        (self.on_stop.lock().unwrap())();
    }
}

fn mk_vc(connected: bool, is_playing: BoolFn, play: PlayFn, on_stop: UnitFn) -> Arc<MockVc> {
    Arc::new(MockVc {
        connected,
        plays: Arc::new(Mutex::new(Vec::new())),
        stop_count: Arc::new(AtomicUsize::new(0)),
        is_playing: Mutex::new(is_playing),
        play: Mutex::new(play),
        on_stop: Mutex::new(on_stop),
    })
}

/// `is_playing()` returns true for `n` calls, then false.
fn vc_play_durations(connected: bool, n: usize) -> Arc<MockVc> {
    let counter = Arc::new(AtomicUsize::new(0));
    let is_playing: BoolFn = Box::new(move || counter.fetch_add(1, Ordering::SeqCst) < n);
    mk_vc(connected, is_playing, Box::new(|| Ok(())), Box::new(|| {}))
}

/// `is_playing()` stays true until `stop()` is called.
fn vc_sticky(connected: bool) -> Arc<MockVc> {
    let playing = Arc::new(AtomicBool::new(true));
    let ip = {
        let playing = Arc::clone(&playing);
        Box::new(move || playing.load(Ordering::SeqCst)) as BoolFn
    };
    let stop = {
        let playing = Arc::clone(&playing);
        Box::new(move || playing.store(false, Ordering::SeqCst)) as UnitFn
    };
    mk_vc(connected, ip, Box::new(|| Ok(())), stop)
}

fn player_with(tts: Arc<dyn TtsClient>, vc: Arc<MockVc>) -> DiscordVoicePlayer {
    DiscordVoicePlayer::new(tts, move || {
        Some(Arc::clone(&vc) as Arc<dyn VoiceClientLike>)
    })
}

fn player_no_vc(tts: Arc<dyn TtsClient>) -> DiscordVoicePlayer {
    DiscordVoicePlayer::new(tts, || None)
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synthesizes_and_plays() {
    let (tts, calls) = stub_tts(mono_pcm(8));
    let vc = vc_play_durations(true, 2);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("hello world", &scope("t")).await;
    assert_eq!(*calls.lock().unwrap(), vec!["hello world".to_owned()]);
    assert_eq!(vc.plays.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn pcm_converted_to_stereo() {
    let mono = mono_pcm(4);
    let (tts, _calls) = stub_tts(mono.clone());
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("x", &scope("t")).await;
    let plays = vc.plays.lock().unwrap();
    match &plays[0] {
        AudioSource::Buffered(bytes) => {
            assert_eq!(bytes.len(), mono.len() * 2);
            assert_eq!(*bytes, mono_to_stereo(&mono).unwrap());
        }
        AudioSource::Streaming(_) => panic!("expected buffered"),
    }
}

// ---------------------------------------------------------------------------
// Skip paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_voice_client_skips() {
    let (tts, calls) = stub_tts(mono_pcm(4));
    let player = player_no_vc(tts);
    player.speak("hello", &scope("t")).await;
    // Buffered path synthesizes before checking the voice client.
    assert_eq!(*calls.lock().unwrap(), vec!["hello".to_owned()]);
}

#[tokio::test]
async fn disconnected_voice_client_skips() {
    let (tts, _calls) = stub_tts(mono_pcm(4));
    let vc = vc_play_durations(false, 1);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("hello", &scope("t")).await;
    assert_eq!(vc.plays.lock().unwrap().len(), 0);
}

#[tokio::test]
async fn empty_text_skips_synthesize() {
    let (tts, calls) = stub_tts(mono_pcm(4));
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("   \n\t", &scope("t")).await;
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(vc.plays.lock().unwrap().len(), 0);
}

#[tokio::test]
async fn already_cancelled_scope_short_circuits() {
    let (tts, _calls) = stub_tts(mono_pcm(4));
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, Arc::clone(&vc));
    let s = scope("t");
    s.cancel();
    player.speak("hi", &s).await;
    assert_eq!(vc.plays.lock().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// Barge-in
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_during_playback_calls_vc_stop() {
    let (tts, _calls) = stub_tts(mono_pcm(8));
    let vc = vc_sticky(true);
    let player = player_with(tts, Arc::clone(&vc));
    let s = Arc::new(scope("t"));
    let s2 = Arc::clone(&s);
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        s2.cancel();
    });
    let t0 = std::time::Instant::now();
    player.speak("a long utterance", s.as_ref()).await;
    let elapsed = t0.elapsed();
    canceller.await.unwrap();
    assert!(vc.stop_count.load(Ordering::SeqCst) >= 1);
    assert!(
        elapsed < Duration::from_millis(200),
        "barge-in took {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Concurrent speak — play-lock serialization
// ---------------------------------------------------------------------------

struct ConcState {
    playing: bool,
    polls_remaining: i32,
}

#[tokio::test]
async fn concurrent_speaks_do_not_collide() {
    let (tts, calls) = stub_tts(mono_pcm(8));
    let state = Arc::new(Mutex::new(ConcState {
        playing: false,
        polls_remaining: 0,
    }));
    let is_playing: BoolFn = {
        let state = Arc::clone(&state);
        Box::new(move || {
            let mut st = state.lock().unwrap();
            if st.polls_remaining > 0 {
                st.polls_remaining -= 1;
                true
            } else {
                st.playing = false;
                false
            }
        })
    };
    let play: PlayFn = {
        let state = Arc::clone(&state);
        Box::new(move || {
            let mut st = state.lock().unwrap();
            if st.playing {
                return Err(PlayError::AlreadyPlaying);
            }
            st.playing = true;
            st.polls_remaining = 3;
            Ok(())
        })
    };
    let vc = mk_vc(true, is_playing, play, Box::new(|| {}));
    let player = player_with(tts, Arc::clone(&vc));
    let s1 = TurnScope::new("t1", "voice:1:user:101");
    let s2 = TurnScope::new("t2", "voice:1:user:202");
    tokio::join!(
        player.speak("alice reply", &s1),
        player.speak("bob reply", &s2),
    );
    assert_eq!(vc.plays.lock().unwrap().len(), 2);
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["alice reply".to_owned(), "bob reply".to_owned()]
    );
}

struct StopLagState {
    playing: bool,
    stop_lag_polls: i32,
    stopping: bool,
    natural_polls: i32,
}

#[tokio::test]
async fn cancel_then_immediate_speak_does_not_collide() {
    let (tts, _calls) = stub_tts(mono_pcm(8));
    let state = Arc::new(Mutex::new(StopLagState {
        playing: false,
        stop_lag_polls: 0,
        stopping: false,
        natural_polls: 0,
    }));
    let is_playing: BoolFn = {
        let state = Arc::clone(&state);
        Box::new(move || {
            let mut st = state.lock().unwrap();
            if st.stopping {
                st.stop_lag_polls -= 1;
                if st.stop_lag_polls <= 0 {
                    st.playing = false;
                    st.stopping = false;
                }
            } else if st.playing {
                st.natural_polls -= 1;
                if st.natural_polls <= 0 {
                    st.playing = false;
                }
            }
            st.playing
        })
    };
    let play: PlayFn = {
        let state = Arc::clone(&state);
        Box::new(move || {
            let mut st = state.lock().unwrap();
            if st.playing {
                return Err(PlayError::AlreadyPlaying);
            }
            st.playing = true;
            st.stopping = false;
            st.stop_lag_polls = 0;
            st.natural_polls = 8;
            Ok(())
        })
    };
    let on_stop: UnitFn = {
        let state = Arc::clone(&state);
        Box::new(move || {
            let mut st = state.lock().unwrap();
            if st.playing && !st.stopping {
                st.stopping = true;
                st.stop_lag_polls = 4;
            }
        })
    };
    let vc = mk_vc(true, is_playing, play, on_stop);
    let player = player_with(tts, Arc::clone(&vc));
    let s_a = Arc::new(TurnScope::new("ta", "voice:1:user:101"));
    let s_b = TurnScope::new("tb", "voice:1:user:202");
    let s_a2 = Arc::clone(&s_a);
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        s_a2.cancel();
    });
    tokio::join!(
        player.speak("alice", s_a.as_ref()),
        player.speak("bob", &s_b)
    );
    canceller.await.unwrap();
    assert_eq!(vc.plays.lock().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// Budget stamp
// ---------------------------------------------------------------------------

fn span_names() -> Vec<String> {
    get_span_collector()
        .all()
        .iter()
        .map(|r| r.name.clone())
        .collect()
}

#[tokio::test]
async fn play_records_playback_start() {
    let _g = singleton_guard();
    reset_voice_budget_recorder();
    reset_span_collector();
    // Pre-stamp the preceding phase so the player's playback_start emits a gap.
    get_voice_budget_recorder().record("t-budget", PHASE_TTS_FIRST_AUDIO, Some(0.0));

    let (tts, _calls) = stub_tts(mono_pcm(8));
    let vc = vc_play_durations(true, 2);
    let player = player_with(tts, vc);
    player
        .speak("hello", &TurnScope::new("t-budget", "voice:1"))
        .await;

    assert!(span_names().contains(&"voice.tts_to_playback".to_owned()));
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn uses_streaming_when_client_supports_it() {
    let chunks = vec![
        vec![0x10; DISCORD_FRAME_SIZE],
        vec![0x20; DISCORD_FRAME_SIZE],
    ];
    let tts = streaming_stub(chunks, Duration::ZERO, 0, false);
    let vc = vc_play_durations(true, 2);
    let player = player_with(tts.clone(), Arc::clone(&vc));
    player.speak("hello world", &scope("t")).await;
    assert_eq!(*tts.calls.lock().unwrap(), vec!["hello world".to_owned()]);
    let plays = vc.plays.lock().unwrap();
    assert_eq!(plays.len(), 1);
    assert!(matches!(plays[0], AudioSource::Streaming(_)));
}

#[test]
fn source_uses_default_jitter_params_without_attrs() {
    // A Cartesia-like client exposes no jitter hints, so the player must build the
    // source with prebuffer=0 (first read starts at once) and pad_underrun=false
    // (an open-but-empty buffer blocks rather than padding silence). Python asserts
    // `source._prebuffer_bytes == 0` / `source._pad_underrun is False`; those fields
    // are private in the landed voice::audio, so guard the *player's* forwarding by
    // the built source's observable read behavior — not the stub's own hints.
    let tts = streaming_stub(
        vec![vec![0x10; DISCORD_FRAME_SIZE]],
        Duration::ZERO,
        0,
        false,
    );
    let source = super::build_streaming_source(&*tts);

    // prebuffer=0 → the first read is not pre-roll gated: one buffered frame comes
    // back immediately (a nonzero threshold would block here).
    source.feed(&vec![0x10; DISCORD_FRAME_SIZE]);
    assert_eq!(source.read(), vec![0x10; DISCORD_FRAME_SIZE]);

    // pad_underrun=false → an open, empty buffer blocks instead of returning silence.
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = Arc::clone(&source);
    let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
    assert!(
        rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "pad_underrun=false must block, not pad silence"
    );
    source.close_input(); // release the blocked reader
    assert!(rx.recv_timeout(Duration::from_secs(1)).unwrap().is_empty());
    handle.join().unwrap();
}

#[test]
fn source_uses_client_jitter_params() {
    // An Azure-like client sets pre-roll (3 frames) + underrun padding; the player
    // must forward BOTH into StreamingPcmSource::new(prebuffer_bytes, pad_underrun).
    // Python asserts `source._prebuffer_bytes == DISCORD_FRAME_SIZE * 3` and
    // `source._pad_underrun is True`; guard the same via observable read behavior.
    let tts = streaming_stub(
        vec![vec![0x10; DISCORD_FRAME_SIZE]],
        Duration::ZERO,
        DISCORD_FRAME_SIZE * 3,
        true,
    );
    let source = super::build_streaming_source(&*tts);

    // prebuffer=3*FRAME → the first read is gated until the threshold is buffered.
    source.feed(&vec![0x10; DISCORD_FRAME_SIZE]); // one frame: below the gate
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = Arc::clone(&source);
    let handle = std::thread::spawn(move || tx.send(reader.read()).unwrap());
    assert!(
        rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "prebuffer must gate the first read"
    );
    // Cross the threshold: the reader unblocks with the first frame.
    source.feed(&vec![0x10; DISCORD_FRAME_SIZE * 2]);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        vec![0x10; DISCORD_FRAME_SIZE]
    );
    handle.join().unwrap();

    // Now primed; drain the two remaining buffered frames.
    assert_eq!(source.read(), vec![0x10; DISCORD_FRAME_SIZE]);
    assert_eq!(source.read(), vec![0x10; DISCORD_FRAME_SIZE]);
    // pad_underrun=true → an open, empty buffer returns a silence frame, not block.
    assert_eq!(source.read(), vec![0u8; DISCORD_FRAME_SIZE]);
}

#[tokio::test]
async fn chunks_fed_to_source_as_stereo() {
    let chunks = vec![vec![0x01, 0x02], vec![0x03, 0x04]];
    let tts = streaming_stub(chunks, Duration::ZERO, 0, false);
    let vc = vc_play_durations(true, 4);
    let player = player_with(tts.clone(), Arc::clone(&vc));
    player.speak("x", &scope("t")).await;

    assert!(tts.consumed.load(Ordering::SeqCst));
    let source = {
        let plays = vc.plays.lock().unwrap();
        match &plays[0] {
            AudioSource::Streaming(s) => Arc::clone(s),
            AudioSource::Buffered(_) => panic!("expected streaming"),
        }
    };
    let mut all = Vec::new();
    loop {
        let frame = source.read();
        if frame.is_empty() {
            break;
        }
        all.extend_from_slice(&frame);
    }
    assert_eq!(&all[..8], &[0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x03, 0x04]);
}

#[tokio::test]
async fn play_called_before_stream_drained() {
    let chunks = vec![vec![0xaa; 8], vec![0xbb; 8], vec![0xcc; 8]];
    let tts = streaming_stub(chunks, Duration::from_millis(10), 0, false);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let is_playing = {
        let counter = Arc::new(AtomicUsize::new(0));
        Box::new(move || counter.fetch_add(1, Ordering::SeqCst) < 10) as BoolFn
    };
    let play: PlayFn = {
        let consumed = Arc::clone(&tts.consumed);
        let seen = Arc::clone(&seen);
        Box::new(move || {
            seen.lock().unwrap().push(consumed.load(Ordering::SeqCst));
            Ok(())
        })
    };
    let vc = mk_vc(true, is_playing, play, Box::new(|| {}));
    let player = player_with(tts.clone(), vc);
    player.speak("x", &scope("t")).await;
    assert_eq!(*seen.lock().unwrap(), vec![false]);
}

#[tokio::test]
async fn records_playback_start_in_budget_streaming() {
    let _g = singleton_guard();
    reset_voice_budget_recorder();
    reset_span_collector();
    get_voice_budget_recorder().record("t-stream", PHASE_TTS_FIRST_AUDIO, Some(0.0));

    let tts = streaming_stub(vec![vec![0x10; 4]], Duration::ZERO, 0, false);
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, vc);
    player
        .speak("hello", &TurnScope::new("t-stream", "voice:1"))
        .await;

    assert!(span_names().contains(&"voice.tts_to_playback".to_owned()));
}

#[tokio::test]
async fn cancellation_during_stream_calls_vc_stop() {
    let chunks: Vec<Vec<u8>> = (0..20).map(|_| vec![0x33; 4]).collect();
    let tts = streaming_stub(chunks, Duration::from_millis(10), 0, false);
    let vc = vc_sticky(true);
    let player = player_with(tts, Arc::clone(&vc));
    let s = Arc::new(scope("t"));
    let s2 = Arc::clone(&s);
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        s2.cancel();
    });
    let t0 = std::time::Instant::now();
    player.speak("long", s.as_ref()).await;
    let elapsed = t0.elapsed();
    canceller.await.unwrap();
    assert!(vc.stop_count.load(Ordering::SeqCst) >= 1);
    assert!(
        elapsed < Duration::from_millis(200),
        "barge-in took {elapsed:?}"
    );
}

#[tokio::test]
async fn empty_stream_logs_skip_no_play() {
    let tts = streaming_stub(vec![], Duration::ZERO, 0, false);
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("nothing", &scope("t")).await;
    assert_eq!(vc.plays.lock().unwrap().len(), 0);
}

#[tokio::test]
async fn stream_error_on_first_chunk_logs_no_play() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let tts = Arc::new(BoomTts {
        calls: Arc::clone(&calls),
    });
    let vc = vc_play_durations(true, 1);
    let player = player_with(tts, Arc::clone(&vc));
    player.speak("hi", &scope("t")).await;
    assert_eq!(vc.plays.lock().unwrap().len(), 0);
    assert_eq!(*calls.lock().unwrap(), vec!["hi".to_owned()]);
}

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_calls_vc_stop_when_playing() {
    let (tts, _calls) = stub_tts(mono_pcm(4));
    let vc = mk_vc(
        true,
        Box::new(|| true),
        Box::new(|| Ok(())),
        Box::new(|| {}),
    );
    let player = player_with(tts, Arc::clone(&vc));
    player.stop().await;
    assert_eq!(vc.stop_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stop_noop_when_not_playing() {
    let (tts, _calls) = stub_tts(mono_pcm(4));
    let vc = mk_vc(
        true,
        Box::new(|| false),
        Box::new(|| Ok(())),
        Box::new(|| {}),
    );
    let player = player_with(tts, Arc::clone(&vc));
    player.stop().await;
    assert_eq!(vc.stop_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn stop_no_voice_client() {
    let (tts, _calls) = stub_tts(mono_pcm(4));
    let player = player_no_vc(tts);
    player.stop().await;
}
