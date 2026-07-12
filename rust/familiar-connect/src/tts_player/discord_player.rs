//! Discord-voice `TtsPlayer` (subsystem 09; Python `tts_player/discord_player.py`).
//!
//! Wraps a [`TtsClient`] and feeds Discord-format stereo s16le @ 48 kHz PCM
//! through a live voice client. Two synthesis paths:
//!
//! * **Streaming** — when the client exposes [`TtsClient::as_streaming`], chunks
//!   feed into a [`StreamingPcmSource`] as they arrive so playback starts within
//!   ~one TTFB.
//! * **Buffered** — fallback for buffered-only clients (Gemini): synthesize the
//!   whole utterance, then play.
//!
//! Both paths poll `is_playing()` every [`POLL`] so [`TurnScope::is_cancelled`]
//! cuts playback within ~20 ms when a new turn arrives. A single [`tokio::sync::Mutex`]
//! serializes all playback across speakers (the voice client is single-track).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt as _;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::bus::envelope::TurnScope;
use crate::diagnostics::voice_budget::{PHASE_PLAYBACK_START, get_voice_budget_recorder};
use crate::log_style as ls;
use crate::tts::{StreamingTtsClient, TTSResult, TtsClient, TtsStream};
use crate::tts_player::protocol::TtsPlayer;
use crate::voice::audio::{StreamingPcmSource, mono_to_stereo};

/// Cancel/`is_playing` poll interval.
const POLL: Duration = Duration::from_millis(20);
/// Max wait after `stop()` for the audio thread to flip `is_playing()` false
/// before releasing the play lock (drains the barge-in → next-speak race).
const STOP_DRAIN: Duration = Duration::from_millis(200);

const TARGET: &str = "familiar_connect.tts_player.discord";

/// A source handed to a voice client's `play`.
///
/// The buffered path wraps a whole stereo PCM buffer (Python `discord.PCMAudio`
/// over a `BytesIO`); the streaming path hands a shared [`StreamingPcmSource`]
/// fed incrementally by the drain task.
pub enum AudioSource {
    /// Whole stereo s16le buffer.
    Buffered(Vec<u8>),
    /// Incrementally-fed source.
    Streaming(Arc<StreamingPcmSource>),
}

/// Playback rejection (Python `discord.ClientException`).
#[derive(Debug, thiserror::Error)]
pub enum PlayError {
    /// A second concurrent `play` while audio is already playing.
    #[error("Already playing audio.")]
    AlreadyPlaying,
}

/// The four-method structural voice-client surface (DESIGN §4.8).
///
/// Kept narrow so tests inject mocks without the full Discord voice-client API.
pub trait VoiceClientLike: Send + Sync {
    /// Whether the gateway/voice connection is live.
    fn is_connected(&self) -> bool;
    /// Whether audio is currently playing.
    fn is_playing(&self) -> bool;
    /// Start playing `source`.
    ///
    /// # Errors
    /// [`PlayError::AlreadyPlaying`] when audio is already playing.
    fn play(&self, source: AudioSource) -> Result<(), PlayError>;
    /// Stop playback (flips `is_playing` after the audio thread's next tick).
    fn stop(&self);
}

/// Factory returning the current voice client (or `None` when disconnected).
type GetVoiceClient = Box<dyn Fn() -> Option<Arc<dyn VoiceClientLike>> + Send + Sync>;

/// Synthesize and play through a live Discord voice client.
pub struct DiscordVoicePlayer {
    tts: Arc<dyn TtsClient>,
    get_voice_client: GetVoiceClient,
    /// Serializes playback: turn scopes are per-user, but the voice client is
    /// single-track, so a second `play` while playing raises `AlreadyPlaying`.
    play_lock: tokio::sync::Mutex<()>,
}

impl DiscordVoicePlayer {
    /// Build a player over `tts` and a voice-client factory.
    pub fn new<F>(tts: Arc<dyn TtsClient>, get_voice_client: F) -> Self
    where
        F: Fn() -> Option<Arc<dyn VoiceClientLike>> + Send + Sync + 'static,
    {
        Self {
            tts,
            get_voice_client: Box::new(get_voice_client),
            play_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Streaming path — feed chunks into a [`StreamingPcmSource`] as they arrive.
    async fn speak_streaming(&self, text: &str, scope: &TurnScope) {
        let Some(streaming) = self.tts.as_streaming() else {
            return;
        };
        let source = build_streaming_source(streaming);

        let _guard = self.play_lock.lock().await;
        if scope.is_cancelled() {
            return;
        }
        let Some(vc) = self.connected_vc() else {
            warn_skip("no_voice_client", None);
            return;
        };

        let mut stream = streaming.synthesize_stream(text);
        let first_chunk = match stream.next().await {
            None => {
                warn_skip("empty_stream", Some(&scope.turn_id));
                return;
            }
            Some(Err(exc)) => {
                warn_player_error("synthesize_error", &exc);
                return;
            }
            Some(Ok(chunk)) => chunk,
        };
        if scope.is_cancelled() {
            drop(stream);
            return;
        }
        let stereo = match mono_to_stereo(&first_chunk) {
            Ok(s) => s,
            Err(exc) => {
                warn_player_error("stream_error", &exc);
                return;
            }
        };
        source.feed(&stereo);

        let mut drain = Some(tokio::spawn(drain_stream(
            stream,
            Arc::clone(&source),
            scope.cancellation_token(),
        )));

        info_say(&scope.turn_id, "mode", "stream", ls::LM);

        if let Err(exc) = vc.play(AudioSource::Streaming(Arc::clone(&source))) {
            warn_player_error("play_error", &exc);
            source.close_input();
            if let Some(handle) = drain.take() {
                handle.abort();
                let _ = handle.await;
            }
            return;
        }
        get_voice_budget_recorder().record(&scope.turn_id, PHASE_PLAYBACK_START, None);

        // Poll loop; the cleanup after it always runs (Python `finally`).
        loop {
            if !vc.is_playing() {
                break;
            }
            if scope.is_cancelled() {
                info_cut(&scope.turn_id);
                vc.stop();
                await_stop_drain(vc.as_ref()).await;
                break;
            }
            tokio::time::sleep(POLL).await;
        }
        source.close_input();
        if let Some(handle) = drain.take() {
            let _ = handle.await;
        }
    }

    /// Buffered path — synthesize the whole utterance, then play.
    async fn speak_buffered(&self, text: &str, scope: &TurnScope) {
        let result: TTSResult = match self.tts.synthesize(text).await {
            Ok(r) => r,
            Err(exc) => {
                warn_player_error("synthesize_error", &exc);
                return;
            }
        };
        if scope.is_cancelled() {
            return;
        }
        let Some(vc) = self.connected_vc() else {
            warn_skip("no_voice_client", None);
            return;
        };
        let stereo = match mono_to_stereo(&result.audio) {
            Ok(s) => s,
            Err(exc) => {
                warn_player_error("convert_error", &exc);
                return;
            }
        };
        let byte_len = stereo.len();
        let source = AudioSource::Buffered(stereo);

        let _guard = self.play_lock.lock().await;
        if scope.is_cancelled() {
            return;
        }
        info_say(&scope.turn_id, "bytes", &byte_len.to_string(), ls::LW);
        if let Err(exc) = vc.play(source) {
            warn_player_error("play_error", &exc);
            return;
        }
        get_voice_budget_recorder().record(&scope.turn_id, PHASE_PLAYBACK_START, None);
        loop {
            if !vc.is_playing() {
                break;
            }
            if scope.is_cancelled() {
                info_cut(&scope.turn_id);
                vc.stop();
                await_stop_drain(vc.as_ref()).await;
                return;
            }
            tokio::time::sleep(POLL).await;
        }
    }

    /// The current voice client iff present and connected.
    fn connected_vc(&self) -> Option<Arc<dyn VoiceClientLike>> {
        (self.get_voice_client)().filter(|vc| vc.is_connected())
    }
}

#[async_trait]
impl TtsPlayer for DiscordVoicePlayer {
    async fn speak(&self, text: &str, scope: &TurnScope) {
        if scope.is_cancelled() {
            return;
        }
        // Defense-in-depth: Cartesia 400s on empty/whitespace transcript.
        if text.trim().is_empty() {
            warn_skip("empty_text", Some(&scope.turn_id));
            return;
        }
        if self.tts.as_streaming().is_some() {
            self.speak_streaming(text, scope).await;
        } else {
            self.speak_buffered(text, scope).await;
        }
    }

    async fn stop(&self) {
        // Barge-in fast path: does NOT take the play lock.
        let Some(vc) = (self.get_voice_client)() else {
            return;
        };
        if vc.is_playing() {
            vc.stop();
        }
    }
}

/// Warn `[Player] skip=<reason> [turn=<id>]` (Python `ls.tag('Player', Y)`).
fn warn_skip(reason: &str, turn: Option<&str>) {
    if let Some(turn) = turn {
        tracing::warn!(
            target: TARGET,
            "{} {} {}",
            ls::tag("Player", ls::Y),
            ls::kv_styled("skip", reason, ls::W, ls::LY),
            ls::kv_styled("turn", turn, ls::W, ls::LC),
        );
    } else {
        tracing::warn!(
            target: TARGET,
            "{} {}",
            ls::tag("Player", ls::Y),
            ls::kv_styled("skip", reason, ls::W, ls::LY),
        );
    }
}

/// Warn `[Player] <key>=<repr>` in red (synthesize/play/stream/convert errors).
fn warn_player_error(key: &str, exc: &dyn std::fmt::Debug) {
    tracing::warn!(
        target: TARGET,
        "{} {}",
        ls::tag("Player", ls::R),
        ls::kv_styled(key, &format!("{exc:?}"), ls::W, ls::R),
    );
}

/// Info `[🔊 Say] turn=<id> <key>=<val>` (Python `ls.tag('🔊 Say', G)`).
fn info_say(turn: &str, key: &str, val: &str, vc: &str) {
    tracing::info!(
        target: TARGET,
        "{} {} {}",
        ls::tag("🔊 Say", ls::G),
        ls::kv_styled("turn", turn, ls::W, ls::LC),
        ls::kv_styled(key, val, ls::W, vc),
    );
}

/// Info `[🔊 Cut] turn=<id>` (barge-in, Python `ls.tag('🔊 Cut', Y)`).
fn info_cut(turn: &str) {
    tracing::info!(
        target: TARGET,
        "{} {}",
        ls::tag("🔊 Cut", ls::Y),
        ls::kv_styled("turn", turn, ls::W, ls::LC),
    );
}

/// Build the streaming source, forwarding the client's duck-typed jitter hints
/// (pre-roll + underrun padding) into the [`StreamingPcmSource`] knobs. Bursty
/// providers (Azure) opt into a cushion; steady-cadence ones (Cartesia) keep the
/// defaults (spec 09 §60; DESIGN §4.8). Extracted so the forwarding is guarded by
/// observing the built source's read behavior (its jitter fields are private).
fn build_streaming_source(streaming: &dyn StreamingTtsClient) -> Arc<StreamingPcmSource> {
    let hints = streaming.jitter_hints();
    Arc::new(StreamingPcmSource::new(
        hints.prebuffer_bytes,
        hints.pad_underrun,
    ))
}

/// Feed remaining stream chunks into `source` until exhausted or cancelled;
/// always closes the source (lets the reader hit EOF).
async fn drain_stream(
    mut stream: TtsStream,
    source: Arc<StreamingPcmSource>,
    cancel: CancellationToken,
) {
    loop {
        match stream.next().await {
            None => break,
            Some(Err(exc)) => {
                warn_player_error("stream_error", &exc);
                break;
            }
            Some(Ok(chunk)) => {
                if cancel.is_cancelled() {
                    break;
                }
                match mono_to_stereo(&chunk) {
                    Ok(stereo) => source.feed(&stereo),
                    Err(exc) => {
                        warn_player_error("stream_error", &exc);
                        break;
                    }
                }
            }
        }
    }
    source.close_input();
}

/// Poll `is_playing()` until the audio thread is idle, bounded by [`STOP_DRAIN`]
/// so a wedged player can't pin the play lock. The caller must have called
/// `vc.stop()` first.
async fn await_stop_drain(vc: &dyn VoiceClientLike) {
    let deadline = Instant::now() + STOP_DRAIN;
    while Instant::now() < deadline {
        if !vc.is_playing() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
mod tests;
