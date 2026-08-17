//! Cartesia TTS client + greeting cache (subsystem 09).
//!
//! Every provider returns the uniform [`TTSResult`] (raw PCM audio + per-word
//! timestamps). Two seams the rest of the system types against, kept
//! provider-agnostic so further backends can slot in:
//!
//! * [`TtsClient`] — the buffered `synthesize` surface every client offers, plus
//!   [`TtsClient::as_streaming`], a typed downcast in place of an untyped
//!   capability check.
//! * [`StreamingTtsClient`] — the incremental `synthesize_stream` surface, with
//!   the [`JitterHints`] the player duck-typed as `stream_prebuffer_bytes` /
//!   `stream_pad_underrun`.
//!
//! Timestamps are milliseconds. Cartesia is the only implemented backend; its
//! WebSocket rides `tokio-tungstenite` behind the default `net` feature.

#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use base64::prelude::{BASE64_STANDARD, Engine as _};
use futures::stream::{BoxStream, StreamExt as _};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

use crate::config::TTSConfig;
use crate::log_style as ls;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Log target for per-synthesis TTS telemetry.
const TTS_TARGET: &str = "familiar_connect.tts";

/// Cartesia REST base (used for header construction / non-WS call sites).
pub const CARTESIA_BASE_URL: &str = "https://api.cartesia.ai";
/// Cartesia TTS WebSocket endpoint.
pub const CARTESIA_WS_URL: &str = "wss://api.cartesia.ai/tts/websocket";
/// Cartesia API version pinned in the auth query / REST headers.
pub const CARTESIA_API_VERSION: &str = "2024-06-10";
/// Discord-native output rate; the default for every client.
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;

/// File-based greeting audio cache directory (raw PCM, keyed by sha256).
pub const GREETING_CACHE_DIR: &str = "data/cache/greetings";

// ---------------------------------------------------------------------------
// Errors + result value types
// ---------------------------------------------------------------------------

/// Failure surface for TTS synthesis, factory construction, and playback prep.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// Factory misconfiguration (unknown provider / missing secret / empty
    /// field).
    #[error("{0}")]
    Config(String),
    /// Provider-side or protocol failure surfaced at synthesis time (Cartesia
    /// `error` event, unexpected WS close).
    #[error("{0}")]
    Runtime(String),
    /// WebSocket / HTTP transport failure.
    #[error("{0}")]
    Transport(String),
    /// PCM conversion rejected malformed input.
    #[error("audio conversion: {0}")]
    Audio(#[from] crate::voice::audio::AudioError),
}

/// Per-word playback window (ms from audio start).
#[derive(Clone, Debug, PartialEq)]
pub struct WordTimestamp {
    /// The spoken token.
    pub word: String,
    /// Start offset in milliseconds.
    pub start_ms: f64,
    /// End offset in milliseconds.
    pub end_ms: f64,
}

impl WordTimestamp {
    /// Construct a word window.
    #[must_use]
    pub fn new(word: impl Into<String>, start_ms: f64, end_ms: f64) -> Self {
        Self {
            word: word.into(),
            start_ms,
            end_ms,
        }
    }
}

/// Synthesized audio + per-word timestamps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TTSResult {
    /// Raw mono `pcm_s16le` at the client's sample rate.
    pub audio: Vec<u8>,
    /// Per-word timing (empty when the provider yields none / cache path).
    pub timestamps: Vec<WordTimestamp>,
}

impl TTSResult {
    /// Audio with no timestamps.
    #[must_use]
    pub const fn audio_only(audio: Vec<u8>) -> Self {
        Self {
            audio,
            timestamps: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Player-facing seam traits
// ---------------------------------------------------------------------------

/// Jitter-buffer hints for the player's streaming source.
///
/// A bursty provider opts into pre-roll + underrun padding; steady-cadence ones
/// (Cartesia) leave the defaults. The player reads these dynamically to
/// configure [`StreamingPcmSource`](crate::voice::audio::StreamingPcmSource).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JitterHints {
    /// First-read pre-roll threshold in bytes (`0` = start immediately).
    pub prebuffer_bytes: usize,
    /// Pad an open-but-empty buffer with silence instead of blocking.
    pub pad_underrun: bool,
}

/// Owned async stream of raw mono `pcm_s16le` chunks.
///
/// Dropping the stream tears down the underlying transport (WS close), since
/// Rust streams have no consumer-drop hook. `Some(Err(_))` is a mid-stream
/// failure; `None` is a clean end.
pub type TtsStream = BoxStream<'static, Result<Vec<u8>, TtsError>>;

/// The buffered synth surface every TTS client exposes.
///
/// [`as_streaming`](TtsClient::as_streaming) is a typed capability check:
/// `Some` selects the player's
/// streaming path, `None` the buffered path.
#[async_trait]
pub trait TtsClient: Send + Sync {
    /// Synthesize the whole utterance, returning audio + word timestamps.
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError>;

    /// The streaming view of this client, or `None` for buffered-only clients.
    /// Default: buffered-only.
    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        None
    }
}

/// The incremental synth surface (Cartesia).
pub trait StreamingTtsClient: Send + Sync {
    /// Open a fresh stream of PCM chunks. The transport connects lazily on first
    /// poll.
    fn synthesize_stream(&self, text: &str) -> TtsStream;

    /// Jitter-buffer hints for the player's source. Default: none.
    fn jitter_hints(&self) -> JitterHints {
        JitterHints::default()
    }
}

// ---------------------------------------------------------------------------
// Per-synthesis telemetry
// ---------------------------------------------------------------------------

/// Emit the buffered-synthesis INFO line: `🔉 TTS <Provider> words=..
/// audio=..b timing=..ms→..ms`.
fn log_buffered_synth(provider: &str, audio: &[u8], timestamps: &[WordTimestamp]) {
    let start_ms = timestamps.first().map_or(0.0, |t| t.start_ms);
    let end_ms = timestamps.last().map_or(0.0, |t| t.end_ms);
    tracing::info!(
        target: TTS_TARGET,
        "{} {} {} {} {}",
        ls::tag("🔉 TTS", ls::C),
        ls::word(provider, ls::C),
        ls::kv_styled("words", &timestamps.len().to_string(), ls::W, ls::LW),
        ls::kv_styled("audio", &format!("{}b", audio.len()), ls::W, ls::LW),
        ls::kv_styled(
            "timing",
            &format!("{start_ms:.0}ms→{end_ms:.0}ms"),
            ls::W,
            ls::LW,
        ),
    );
}

/// Per-stream Cartesia telemetry: total bytes + first/last chunk arrival for
/// the `span_ms` INFO line emitted on clean stream end. Timing uses wall-clock
/// deltas between chunk arrivals.
#[derive(Default)]
struct StreamTel {
    total_bytes: usize,
    first_at: Option<std::time::Instant>,
    last_at: Option<std::time::Instant>,
}

impl StreamTel {
    /// Account for one yielded chunk of `n` bytes.
    fn record(&mut self, n: usize) {
        let now = std::time::Instant::now();
        self.total_bytes += n;
        self.first_at.get_or_insert(now);
        self.last_at = Some(now);
    }

    /// Emit `🔉 TTS Cartesia/stream audio=..b span_ms=..` (clean-end only).
    fn log(&self) {
        let span_ms = match (self.first_at, self.last_at) {
            (Some(first), Some(last)) => (last - first).as_secs_f64() * 1000.0,
            _ => 0.0,
        };
        tracing::info!(
            target: TTS_TARGET,
            "{} {} {} {}",
            ls::tag("🔉 TTS", ls::C),
            ls::word("Cartesia/stream", ls::C),
            ls::kv_styled("audio", &format!("{}b", self.total_bytes), ls::W, ls::LW),
            ls::kv_styled("span_ms", &format!("{span_ms:.0}"), ls::W, ls::LW),
        );
    }
}

// ---------------------------------------------------------------------------
// Cartesia
// ---------------------------------------------------------------------------

/// Cartesia TTS WebSocket client; one connection per `synthesize`.
#[derive(Clone, Debug)]
pub struct CartesiaTTSClient {
    /// API key (auth query string).
    pub api_key: String,
    /// Voice id.
    pub voice_id: String,
    /// Model id.
    pub model: String,
    /// REST base URL.
    pub base_url: String,
    /// WebSocket URL.
    pub ws_url: String,
    /// Output sample rate.
    pub sample_rate: u32,
}

impl CartesiaTTSClient {
    /// Client with the default base/ws URLs and 48 kHz output.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        voice_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: model.into(),
            base_url: CARTESIA_BASE_URL.to_owned(),
            ws_url: CARTESIA_WS_URL.to_owned(),
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }

    /// WebSocket URL with auth in the query string.
    #[must_use]
    pub fn build_ws_url(&self) -> String {
        format!(
            "{}?api_key={}&cartesia_version={}",
            self.ws_url, self.api_key, CARTESIA_API_VERSION
        )
    }

    /// REST headers (non-WS call sites / tests).
    #[must_use]
    pub fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::with_capacity(3);
        headers.insert("X-API-Key".to_owned(), self.api_key.clone());
        headers.insert(
            "Cartesia-Version".to_owned(),
            CARTESIA_API_VERSION.to_owned(),
        );
        headers.insert("Content-Type".to_owned(), "application/json".to_owned());
        headers
    }

    /// JSON payload for one-shot synthesis.
    #[must_use]
    pub fn build_payload(&self, text: &str, context_id: &str) -> Value {
        json!({
            "context_id": context_id,
            "model_id": self.model,
            "transcript": text,
            "voice": { "mode": "id", "id": self.voice_id },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "language": "en",
            "add_timestamps": true,
            "continue": false,
        })
    }

    /// Drive one buffered synthesis over `conn`; always closes `conn` (finally).
    async fn synthesize_conn(
        &self,
        text: &str,
        conn: &mut dyn CartesiaConn,
    ) -> Result<TTSResult, TtsError> {
        let context_id = uuid::Uuid::new_v4().simple().to_string();
        let payload = self.build_payload(text, &context_id);
        let mut audio_parts: Vec<Vec<u8>> = Vec::new();
        let mut timestamps: Vec<WordTimestamp> = Vec::new();
        let outcome =
            drive_cartesia_synthesize(conn, &payload, &mut audio_parts, &mut timestamps).await;
        if !conn.is_closed() {
            conn.close().await;
        }
        outcome?;
        let audio = audio_parts.concat();
        log_buffered_synth("Cartesia", &audio, &timestamps);
        Ok(TTSResult { audio, timestamps })
    }
}

/// A decoded frame from the Cartesia WebSocket.
#[derive(Debug)]
enum CartesiaFrame {
    /// A JSON text event.
    Text(String),
    /// Server closed the socket.
    Closed,
    /// Transport error frame.
    Errored,
    /// A non-text frame (binary/ping/pong) — ignored, loop continues.
    Other,
}

/// The Cartesia WS connection seam (real transport or a scripted test fake).
#[async_trait]
trait CartesiaConn: Send {
    async fn send_json(&mut self, value: &Value) -> Result<(), TtsError>;
    async fn recv(&mut self) -> Option<CartesiaFrame>;
    async fn close(&mut self);
    fn is_closed(&self) -> bool;
}

/// Send the payload, then consume events into `audio_parts` / `timestamps` until
/// `done` or the socket ends. Does not close `conn` (the caller's finally does).
async fn drive_cartesia_synthesize(
    conn: &mut dyn CartesiaConn,
    payload: &Value,
    audio_parts: &mut Vec<Vec<u8>>,
    timestamps: &mut Vec<WordTimestamp>,
) -> Result<(), TtsError> {
    conn.send_json(payload).await?;
    loop {
        match conn.recv().await {
            None => break,
            Some(CartesiaFrame::Text(data)) => {
                let event: Value = serde_json::from_str(&data)
                    .map_err(|e| TtsError::Runtime(format!("Cartesia bad JSON: {e}")))?;
                if handle_cartesia_event(&event, audio_parts, timestamps)? {
                    break;
                }
            }
            Some(CartesiaFrame::Closed | CartesiaFrame::Errored) => {
                return Err(TtsError::Runtime(
                    "Cartesia WebSocket closed unexpectedly".to_owned(),
                ));
            }
            Some(CartesiaFrame::Other) => {}
        }
    }
    Ok(())
}

/// Dispatch one parsed Cartesia event; `Ok(true)` when the `done` event arrives.
fn handle_cartesia_event(
    event: &Value,
    audio_parts: &mut Vec<Vec<u8>>,
    timestamps: &mut Vec<WordTimestamp>,
) -> Result<bool, TtsError> {
    match event.get("type").and_then(Value::as_str) {
        Some("chunk") => {
            if let Some(data) = event.get("data").and_then(Value::as_str) {
                let bytes = BASE64_STANDARD
                    .decode(data)
                    .map_err(|e| TtsError::Runtime(format!("Cartesia base64: {e}")))?;
                audio_parts.push(bytes);
            }
            Ok(false)
        }
        Some("timestamps") => {
            let raw = event.get("word_timestamps").cloned().unwrap_or(Value::Null);
            timestamps.extend(parse_word_timestamps(&raw));
            Ok(false)
        }
        Some("done") => Ok(true),
        Some("error") => Err(cartesia_error(event)),
        _ => Ok(false),
    }
}

/// Build the `RuntimeError` for a Cartesia `error` event.
fn cartesia_error(event: &Value) -> TtsError {
    let err = event
        .get("error")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown error");
    let status = match event.get("status_code") {
        None | Some(Value::Null) => "None".to_owned(),
        Some(v) => v.to_string(),
    };
    TtsError::Runtime(format!("Cartesia TTS error (status={status}): {err}"))
}

/// Convert Cartesia's parallel-array word timestamps (seconds) to `WordTimestamp`
/// (ms). Zips to the shortest of the three arrays.
fn parse_word_timestamps(raw: &Value) -> Vec<WordTimestamp> {
    let empty: Vec<Value> = Vec::new();
    let words = raw.get("words").and_then(Value::as_array).unwrap_or(&empty);
    let starts = raw.get("start").and_then(Value::as_array).unwrap_or(&empty);
    let ends = raw.get("end").and_then(Value::as_array).unwrap_or(&empty);
    let count = words.len().min(starts.len()).min(ends.len());
    (0..count)
        .map(|i| {
            let word = words[i]
                .as_str()
                .map_or_else(|| words[i].to_string(), ToString::to_string);
            let start_ms = starts[i].as_f64().unwrap_or(0.0) * 1000.0;
            let end_ms = ends[i].as_f64().unwrap_or(0.0) * 1000.0;
            WordTimestamp {
                word,
                start_ms,
                end_ms,
            }
        })
        .collect()
}

// --- Cartesia streaming -----------------------------------------------------

/// One step of the streaming state machine.
enum CartesiaStep {
    /// A non-empty decoded chunk to yield.
    Chunk(Vec<u8>),
    /// Clean end (`done` / socket exhausted).
    End,
    /// Terminal failure to surface to the consumer.
    Fail(TtsError),
}

/// Consume frames until a non-empty chunk or a terminal (`done`/error/close).
async fn cartesia_stream_step(conn: &mut dyn CartesiaConn) -> CartesiaStep {
    loop {
        match conn.recv().await {
            None => return CartesiaStep::End,
            Some(CartesiaFrame::Text(data)) => {
                let event: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(e) => {
                        return CartesiaStep::Fail(TtsError::Runtime(format!(
                            "Cartesia bad JSON: {e}"
                        )));
                    }
                };
                match event.get("type").and_then(Value::as_str) {
                    Some("chunk") => {
                        if let Some(data) = event.get("data").and_then(Value::as_str) {
                            match BASE64_STANDARD.decode(data) {
                                Ok(bytes) if bytes.is_empty() => {}
                                Ok(bytes) => return CartesiaStep::Chunk(bytes),
                                Err(e) => {
                                    return CartesiaStep::Fail(TtsError::Runtime(format!(
                                        "Cartesia base64: {e}"
                                    )));
                                }
                            }
                        }
                    }
                    Some("done") => return CartesiaStep::End,
                    Some("error") => return CartesiaStep::Fail(cartesia_error(&event)),
                    _ => {}
                }
            }
            Some(CartesiaFrame::Closed | CartesiaFrame::Errored) => {
                return CartesiaStep::Fail(TtsError::Runtime(
                    "Cartesia WebSocket closed unexpectedly".to_owned(),
                ));
            }
            Some(CartesiaFrame::Other) => {}
        }
    }
}

/// Streaming state carried across `unfold` polls.
enum CartesiaStreamState {
    /// Lazily connect on first poll, then send + first step.
    Connecting { url: String, payload: Value },
    /// Connected (test entry): send payload, then first step.
    #[allow(
        dead_code,
        reason = "constructed only by the in-module streaming tests"
    )]
    Sending {
        conn: Box<dyn CartesiaConn>,
        payload: Value,
    },
    /// Streaming chunks.
    Receiving {
        conn: Box<dyn CartesiaConn>,
        tel: StreamTel,
    },
    /// Terminal.
    Done,
}

type CartesiaYield = Option<(Result<Vec<u8>, TtsError>, CartesiaStreamState)>;

/// Send the payload then produce the first chunk-or-terminal.
async fn cartesia_drive_sending(mut conn: Box<dyn CartesiaConn>, payload: Value) -> CartesiaYield {
    if let Err(e) = conn.send_json(&payload).await {
        conn.close().await;
        return Some((Err(e), CartesiaStreamState::Done));
    }
    match cartesia_stream_step(&mut *conn).await {
        CartesiaStep::Chunk(c) => {
            let mut tel = StreamTel::default();
            tel.record(c.len());
            Some((Ok(c), CartesiaStreamState::Receiving { conn, tel }))
        }
        CartesiaStep::End => {
            conn.close().await;
            StreamTel::default().log();
            None
        }
        CartesiaStep::Fail(e) => {
            conn.close().await;
            Some((Err(e), CartesiaStreamState::Done))
        }
    }
}

/// Build the streaming stream from an initial state.
fn cartesia_stream_from_state(state: CartesiaStreamState) -> TtsStream {
    futures::stream::unfold(state, |state| async move {
        match state {
            CartesiaStreamState::Done => None,
            CartesiaStreamState::Connecting { url, payload } => {
                let conn = match cartesia_connect(&url).await {
                    Ok(c) => c,
                    Err(e) => return Some((Err(e), CartesiaStreamState::Done)),
                };
                cartesia_drive_sending(conn, payload).await
            }
            CartesiaStreamState::Sending { conn, payload } => {
                cartesia_drive_sending(conn, payload).await
            }
            CartesiaStreamState::Receiving { mut conn, mut tel } => {
                match cartesia_stream_step(&mut *conn).await {
                    CartesiaStep::Chunk(c) => {
                        tel.record(c.len());
                        Some((Ok(c), CartesiaStreamState::Receiving { conn, tel }))
                    }
                    CartesiaStep::End => {
                        conn.close().await;
                        tel.log();
                        None
                    }
                    CartesiaStep::Fail(e) => {
                        conn.close().await;
                        Some((Err(e), CartesiaStreamState::Done))
                    }
                }
            }
        }
    })
    .boxed()
}

#[async_trait]
impl TtsClient for CartesiaTTSClient {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        #[cfg(feature = "net")]
        {
            let mut conn = cartesia_connect(&self.build_ws_url()).await?;
            self.synthesize_conn(text, conn.as_mut()).await
        }
        #[cfg(not(feature = "net"))]
        {
            let _ = text;
            Err(TtsError::Transport(
                "Cartesia requires the `net` feature".to_owned(),
            ))
        }
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        Some(self)
    }
}

impl StreamingTtsClient for CartesiaTTSClient {
    fn synthesize_stream(&self, text: &str) -> TtsStream {
        let context_id = uuid::Uuid::new_v4().simple().to_string();
        let payload = self.build_payload(text, &context_id);
        cartesia_stream_from_state(CartesiaStreamState::Connecting {
            url: self.build_ws_url(),
            payload,
        })
    }
}

// --- Cartesia real transport (net) ------------------------------------------

#[cfg(feature = "net")]
type CartesiaWs =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[cfg(feature = "net")]
struct RealCartesiaConn {
    /// `None` once `close()` (or `Drop`) has taken the socket to shut it down.
    ws: Option<CartesiaWs>,
    closed: bool,
}

#[cfg(feature = "net")]
async fn cartesia_connect(url: &str) -> Result<Box<dyn CartesiaConn>, TtsError> {
    let (ws, _resp) = tokio_tungstenite::connect_async(url)
        .await
        .map_err(|e| TtsError::Transport(format!("Cartesia WS connect failed: {e}")))?;
    Ok(Box::new(RealCartesiaConn {
        ws: Some(ws),
        closed: false,
    }))
}

#[cfg(not(feature = "net"))]
async fn cartesia_connect(_url: &str) -> Result<Box<dyn CartesiaConn>, TtsError> {
    Err(TtsError::Transport(
        "Cartesia requires the `net` feature".to_owned(),
    ))
}

#[cfg(feature = "net")]
#[async_trait]
impl CartesiaConn for RealCartesiaConn {
    async fn send_json(&mut self, value: &Value) -> Result<(), TtsError> {
        use futures::SinkExt as _;
        let txt = serde_json::to_string(value)
            .map_err(|e| TtsError::Runtime(format!("Cartesia serialize: {e}")))?;
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| TtsError::Transport("Cartesia WS already closed".to_owned()))?;
        ws.send(tokio_tungstenite::tungstenite::Message::Text(txt))
            .await
            .map_err(|e| TtsError::Transport(format!("Cartesia send: {e}")))
    }

    async fn recv(&mut self) -> Option<CartesiaFrame> {
        use tokio_tungstenite::tungstenite::Message;
        let ws = self.ws.as_mut()?;
        match ws.next().await {
            Some(Ok(Message::Text(t))) => Some(CartesiaFrame::Text(t)),
            Some(Ok(Message::Close(_))) => Some(CartesiaFrame::Closed),
            Some(Ok(_)) => Some(CartesiaFrame::Other),
            Some(Err(_)) => Some(CartesiaFrame::Errored),
            None => None,
        }
    }

    async fn close(&mut self) {
        self.closed = true;
        if let Some(mut ws) = self.ws.take() {
            let _ = ws.close(None).await;
        }
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(feature = "net")]
impl Drop for RealCartesiaConn {
    fn drop(&mut self) {
        // Consumer dropped the stream mid-flight (barge-in) without the End/Fail
        // path reaching `close()`: close gracefully by
        // handing the still-open socket to a detached task that sends a graceful
        // Close (1000). Without this the server only ever observes an abnormal
        // 1006. If `close()` already
        // ran the socket is gone and this is a no-op.
        let Some(mut ws) = self.ws.take() else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = ws.close(None).await;
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Greeting cache (shared across all providers)
// ---------------------------------------------------------------------------

/// Filesystem path for cached greeting audio:
/// `<dir>/<hex sha256 of "provider:voice_id:greeting">.bin`.
#[must_use]
fn greeting_cache_path(dir: &Path, provider: &str, voice_id: &str, greeting: &str) -> PathBuf {
    let key = format!("{provider}:{voice_id}:{greeting}");
    let digest = Sha256::digest(key.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    dir.join(format!("{hex}.bin"))
}

/// TTS audio for `greeting`, cached under the default cache dir.
///
/// On hit, reads bytes from disk (empty timestamps). On miss, synthesizes via
/// `client`, writes the bytes, and returns them (also empty timestamps —
/// timestamps are never cached).
pub async fn get_cached_greeting_audio(
    provider: &str,
    voice_id: &str,
    greeting: &str,
    client: &dyn TtsClient,
) -> Result<TTSResult, TtsError> {
    get_cached_greeting_audio_in(
        Path::new(GREETING_CACHE_DIR),
        provider,
        voice_id,
        greeting,
        client,
    )
    .await
}

/// [`get_cached_greeting_audio`] with an explicit cache directory (testable).
async fn get_cached_greeting_audio_in(
    dir: &Path,
    provider: &str,
    voice_id: &str,
    greeting: &str,
    client: &dyn TtsClient,
) -> Result<TTSResult, TtsError> {
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|e| TtsError::Runtime(format!("greeting cache mkdir: {e}")))?;
    let path = greeting_cache_path(dir, provider, voice_id, greeting);
    let is_file = tokio::fs::metadata(&path).await.is_ok_and(|m| m.is_file());
    if is_file {
        let audio = tokio::fs::read(&path)
            .await
            .map_err(|e| TtsError::Runtime(format!("greeting cache read: {e}")))?;
        return Ok(TTSResult::audio_only(audio));
    }
    let result = client.synthesize(greeting).await?;
    tokio::fs::write(&path, &result.audio)
        .await
        .map_err(|e| TtsError::Runtime(format!("greeting cache write: {e}")))?;
    Ok(TTSResult::audio_only(result.audio))
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// The concrete TTS client for the active provider. One variant today; the
/// enum is the widening point for further backends behind [`TtsClient`].
#[derive(Debug)]
pub enum TtsClientKind {
    /// Cartesia WS client.
    Cartesia(CartesiaTTSClient),
}

impl TtsClientKind {
    /// Erase to a shared `dyn TtsClient` for the player / greeting cache.
    #[must_use]
    pub fn into_dyn(self) -> Arc<dyn TtsClient> {
        match self {
            Self::Cartesia(c) => Arc::new(c),
        }
    }
}

/// Instantiate the TTS client for `[tts].provider`, reading secrets from the
/// process environment.
///
/// # Errors
/// [`TtsError::Config`] for an unknown provider, a missing env secret, or an
/// empty required field.
pub fn create_tts_client(cfg: &TTSConfig) -> Result<TtsClientKind, TtsError> {
    build_tts_client(cfg, |key| std::env::var(key).ok())
}

/// [`create_tts_client`] with an injectable env lookup (testable, race-free).
fn build_tts_client(
    cfg: &TTSConfig,
    env: impl Fn(&str) -> Option<String>,
) -> Result<TtsClientKind, TtsError> {
    match cfg.provider.as_str() {
        "cartesia" => {
            let api_key = env("CARTESIA_API_KEY")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    TtsError::Config("CARTESIA_API_KEY environment variable is required".to_owned())
                })?;
            let voice_id = cfg
                .cartesia_voice_id
                .clone()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    TtsError::Config(
                        "TTS cartesia_voice_id is required \
                         (set [tts].cartesia_voice_id in character.toml)"
                            .to_owned(),
                    )
                })?;
            let model = cfg
                .cartesia_model
                .clone()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    TtsError::Config(
                        "TTS cartesia_model is required \
                         (set [tts].cartesia_model in character.toml)"
                            .to_owned(),
                    )
                })?;
            Ok(TtsClientKind::Cartesia(CartesiaTTSClient::new(
                api_key, voice_id, model,
            )))
        }
        // The message quotes the provider repr-style:
        // single-quotes the string; `{:?}` (Debug) would double-quote it.
        other => Err(TtsError::Config(format!(
            "Unknown TTS provider '{other}'; expected 'cartesia'"
        ))),
    }
}

#[cfg(test)]
mod tests;
