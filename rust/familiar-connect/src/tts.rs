//! Cartesia/Azure/Gemini TTS clients + greeting cache (subsystem 09; Python `tts.py`).
//!
//! All three providers return the uniform [`TTSResult`] (raw PCM audio + per-word
//! timestamps). Two seams the rest of the system types against:
//!
//! * [`TtsClient`] — the buffered `synthesize` surface every client offers, plus
//!   [`TtsClient::as_streaming`] which reifies Python's duck-typed
//!   `hasattr(client, "synthesize_stream")` check into a typed downcast.
//! * [`StreamingTtsClient`] — the incremental `synthesize_stream` surface
//!   (Cartesia, Azure) with the [`JitterHints`] the player duck-typed as
//!   `stream_prebuffer_bytes` / `stream_pad_underrun`.
//!
//! Integer math (Gemini 2× upsample midpoint) uses floor division (`div_euclid`)
//! to match Python `//` bit-for-bit (DESIGN §4.3). Timestamps are milliseconds.
//!
//! Transport parity note: the Cartesia WebSocket is wired over
//! `tokio-tungstenite` behind the default `net` feature. The Azure Speech SDK and
//! Google Gemini SDK have no drop-in Rust equivalent that matches the Python mock
//! surface, so their live backends are deferred to the wiring layer (10): the
//! buffered/streaming *logic* is fully ported and unit-tested against mockable
//! backend seams ([`AzureBufferedBackend`], [`AzureStreamReader`],
//! [`GeminiBackend`]), while the concrete network binding returns a "not wired"
//! [`TtsError::Runtime`]. See the port summary deviations.

#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use base64::prelude::{BASE64_STANDARD, Engine as _};
use futures::Stream;
use futures::stream::{BoxStream, StreamExt as _};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

use crate::config::TTSConfig;
use crate::log_style as ls;

// ---------------------------------------------------------------------------
// Constants (Python module-level)
// ---------------------------------------------------------------------------

/// Log target for per-synthesis TTS telemetry (Python `familiar_connect.tts`).
const TTS_TARGET: &str = "familiar_connect.tts";

/// Cartesia REST base (used for header construction / non-WS call sites).
pub const CARTESIA_BASE_URL: &str = "https://api.cartesia.ai";
/// Cartesia TTS WebSocket endpoint.
pub const CARTESIA_WS_URL: &str = "wss://api.cartesia.ai/tts/websocket";
/// Cartesia API version pinned in the auth query / REST headers.
pub const CARTESIA_API_VERSION: &str = "2024-06-10";
/// Discord-native output rate; the default for every client.
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;

/// Default Azure Neural voice; mirrors `config::DEFAULT_AZURE_TTS_VOICE`.
pub const DEFAULT_AZURE_VOICE: &str = "en-US-AmberNeural";

/// 100-nanosecond ticks per millisecond — Azure SDK word-offset unit.
pub const AZURE_TICKS_PER_MS: f64 = 10_000.0;

/// Read-buffer size for incremental `AudioDataStream` reads.
pub const AZURE_STREAM_BUFFER_BYTES: usize = 32 * 1024;

/// Bound on joining the stream worker on early-close (never wedge `aclose`).
pub const AZURE_STREAM_JOIN_TIMEOUT_S: f64 = 2.0;

/// Gemini TTS native rate (24 kHz); 2× upsampled to 48 kHz before use.
pub const GEMINI_SAMPLE_RATE: u32 = 24_000;

/// File-based greeting audio cache directory (raw PCM, keyed by sha256).
pub const GREETING_CACHE_DIR: &str = "data/cache/greetings";

// ---------------------------------------------------------------------------
// Errors + result value types
// ---------------------------------------------------------------------------

/// Failure surface for TTS synthesis, factory construction, and playback prep.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// Factory misconfiguration (unknown provider / missing secret / empty field).
    /// Mirrors Python's `ValueError` from `create_tts_client`.
    #[error("{0}")]
    Config(String),
    /// Provider-side or protocol failure surfaced at synthesis time
    /// (Cartesia `error` event, Azure cancellation, Gemini missing audio part,
    /// unexpected WS close). Mirrors Python's `RuntimeError`.
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
/// A bursty provider (Azure) opts into pre-roll + underrun padding; steady-cadence
/// ones (Cartesia) leave the defaults. The player reads these dynamically to
/// configure [`StreamingPcmSource`](crate::voice::audio::StreamingPcmSource).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JitterHints {
    /// First-read pre-roll threshold in bytes (`0` = start immediately).
    pub prebuffer_bytes: usize,
    /// Pad an open-but-empty buffer with silence instead of blocking.
    pub pad_underrun: bool,
}

/// Owned async stream of raw mono `pcm_s16le` chunks (Cartesia/Azure streaming).
///
/// Dropping the stream tears down the underlying transport (WS close / Azure
/// worker abort) — the port's replacement for Python's `aclose` finally, since
/// Rust streams have no consumer-drop hook. `Some(Err(_))` is a mid-stream
/// failure; `None` is a clean end (Python `StopAsyncIteration`).
pub type TtsStream = BoxStream<'static, Result<Vec<u8>, TtsError>>;

/// The buffered synth surface every TTS client exposes.
///
/// [`as_streaming`](TtsClient::as_streaming) reifies the Python duck-typed
/// `hasattr(client, "synthesize_stream")` check: `Some` selects the player's
/// streaming path, `None` the buffered path.
#[async_trait]
pub trait TtsClient: Send + Sync {
    /// Synthesize the whole utterance, returning audio + word timestamps.
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError>;

    /// The streaming view of this client, or `None` for buffered-only clients
    /// (Gemini). Default: buffered-only.
    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        None
    }
}

/// The incremental synth surface (Cartesia, Azure).
pub trait StreamingTtsClient: Send + Sync {
    /// Open a fresh stream of PCM chunks. The transport connects lazily on first
    /// poll (matching Python's connect-on-first-`anext`).
    fn synthesize_stream(&self, text: &str) -> TtsStream;

    /// Jitter-buffer hints for the player's source. Default: none.
    fn jitter_hints(&self) -> JitterHints {
        JitterHints::default()
    }
}

// ---------------------------------------------------------------------------
// Per-synthesis telemetry (Python `_logger.info` lines)
// ---------------------------------------------------------------------------

/// Emit the buffered-synthesis INFO line: `🔉 TTS <Provider> words=.. audio=..b
/// timing=..ms→..ms` (Python `synthesize` / `_synthesize_sync` tail log).
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

/// Emit the Azure streaming INFO line: `🔉 TTS Azure/stream audio=..b`
/// (Python `_read_stream_chunks` tail log).
fn log_azure_stream(total_bytes: usize) {
    tracing::info!(
        target: TTS_TARGET,
        "{} {} {}",
        ls::tag("🔉 TTS", ls::C),
        ls::word("Azure/stream", ls::C),
        ls::kv_styled("audio", &format!("{total_bytes}b"), ls::W, ls::LW),
    );
}

/// Emit the Gemini INFO line: `🔉 TTS Gemini words=.. audio=..b duration=..ms`
/// (Python `_synthesize_sync` tail log).
fn log_gemini_synth(audio: &[u8], timestamps: &[WordTimestamp], total_ms: f64) {
    tracing::info!(
        target: TTS_TARGET,
        "{} {} {} {} {}",
        ls::tag("🔉 TTS", ls::C),
        ls::word("Gemini", ls::C),
        ls::kv_styled("words", &timestamps.len().to_string(), ls::W, ls::LW),
        ls::kv_styled("audio", &format!("{}b", audio.len()), ls::W, ls::LW),
        ls::kv_styled("duration", &format!("{total_ms:.0}ms"), ls::W, ls::LW),
    );
}

/// Per-stream Cartesia telemetry: total bytes + first/last chunk arrival for the
/// `span_ms` INFO line emitted on clean stream end (Python `synthesize_stream`
/// tail log). Timing uses wall-clock deltas between chunk arrivals.
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
        // path reaching `close()`: mirror Python's `finally: await ws.close()` by
        // handing the still-open socket to a detached task that sends a graceful
        // Close (1000). Without this the server only ever observes an abnormal
        // 1006 (DESIGN TtsStream drop contract; spec 09 §48). If `close()` already
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
    let is_file = tokio::fs::metadata(&path)
        .await
        .is_ok_and(|m| m.is_file());
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
// Azure
// ---------------------------------------------------------------------------

/// Azure Cognitive Services TTS client.
///
/// The jitter hints are off: Azure delivery was measured faster-than-realtime,
/// so streaming runs the plain path (like Cartesia). Kept at off as a documented
/// reversal lever the player reads dynamically.
#[derive(Clone, Debug)]
pub struct AzureTTSClient {
    /// Subscription key.
    pub subscription_key: String,
    /// Region.
    pub region: String,
    /// Neural voice name.
    pub voice_name: String,
    /// Output sample rate.
    pub sample_rate: u32,
    /// First-read pre-roll (bytes); `0` — see the type doc.
    pub stream_prebuffer_bytes: usize,
    /// Underrun padding; `false` — see the type doc.
    pub stream_pad_underrun: bool,
}

impl AzureTTSClient {
    /// Client with the default (off) jitter hints and 48 kHz output.
    #[must_use]
    pub fn new(
        subscription_key: impl Into<String>,
        region: impl Into<String>,
        voice_name: impl Into<String>,
        sample_rate: u32,
    ) -> Self {
        Self {
            subscription_key: subscription_key.into(),
            region: region.into(),
            voice_name: voice_name.into(),
            sample_rate,
            stream_prebuffer_bytes: 0,
            stream_pad_underrun: false,
        }
    }
}

/// Azure SDK word-boundary event type (only `Word` produces a timestamp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzureBoundaryType {
    /// A word boundary — counted.
    Word,
    /// Any other boundary (punctuation, sentence) — skipped.
    Other,
}

/// One Azure word-boundary event.
#[derive(Clone, Debug)]
pub struct AzureWordBoundary {
    /// Spoken token.
    pub text: String,
    /// Boundary kind.
    pub boundary_type: AzureBoundaryType,
    /// Audio offset in 100 ns ticks.
    pub audio_offset_ticks: i64,
    /// Duration in milliseconds (`duration.total_seconds() * 1000`).
    pub duration_ms: f64,
}

/// The outcome of a blocking Azure buffered synthesis.
pub enum AzureSynthOutcome {
    /// Completed with audio + raw boundary events.
    Completed {
        /// Raw PCM.
        audio: Vec<u8>,
        /// All boundary events (filtered to `Word` when building timestamps).
        words: Vec<AzureWordBoundary>,
    },
    /// Cancelled/failed with cancellation details.
    Canceled {
        /// Cancellation reason.
        reason: String,
        /// Detail text.
        error_details: String,
    },
}

/// Convert an Azure buffered outcome into a [`TTSResult`] (ticks → ms; `Word`
/// events only), or an error from the cancellation details.
#[allow(
    clippy::cast_precision_loss,
    reason = "audio offsets are small tick counts; f64 mantissa holds them exactly"
)]
fn build_azure_result(outcome: AzureSynthOutcome) -> Result<TTSResult, TtsError> {
    match outcome {
        AzureSynthOutcome::Canceled {
            reason,
            error_details,
        } => Err(azure_error(&reason, &error_details)),
        AzureSynthOutcome::Completed { audio, words } => {
            let timestamps: Vec<WordTimestamp> = words
                .into_iter()
                .filter(|w| w.boundary_type == AzureBoundaryType::Word)
                .map(|w| {
                    let start_ms = w.audio_offset_ticks as f64 / AZURE_TICKS_PER_MS;
                    WordTimestamp {
                        word: w.text,
                        start_ms,
                        end_ms: start_ms + w.duration_ms,
                    }
                })
                .collect();
            log_buffered_synth("Azure", &audio, &timestamps);
            Ok(TTSResult { audio, timestamps })
        }
    }
}

/// The `RuntimeError` message Azure raises from cancellation details.
fn azure_error(reason: &str, error_details: &str) -> TtsError {
    TtsError::Runtime(format!(
        "Azure TTS synthesis failed: {reason} — {error_details}"
    ))
}

/// The blocking buffered Azure backend seam (real SDK or a test fake).
pub trait AzureBufferedBackend: Send + Sync {
    /// Run blocking synthesis to completion.
    fn speak(&self, text: &str) -> AzureSynthOutcome;
}

/// Run a buffered Azure backend off the event loop (`spawn_blocking`) and build
/// the result — the async wrapper over the blocking SDK call.
async fn azure_synthesize_with(
    backend: Arc<dyn AzureBufferedBackend>,
    text: String,
) -> Result<TTSResult, TtsError> {
    let outcome = tokio::task::spawn_blocking(move || backend.speak(&text))
        .await
        .map_err(|e| TtsError::Runtime(format!("Azure worker join: {e}")))?;
    build_azure_result(outcome)
}

/// Terminal state of the Azure audio stream after a zero-length read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzureStreamStatus {
    /// Clean end.
    AllData,
    /// Mid-stream failure (token expiry / network drop) reported via status.
    Canceled,
}

/// The blocking incremental Azure reader (worker-thread side).
pub trait AzureStreamReader: Send {
    /// `start_speaking_text_async().get()`; `Err` when cancelled at start.
    fn start(&mut self, text: &str) -> Result<(), TtsError>;
    /// One blocking `read_data`; empty vec signals a zero-length read.
    fn read(&mut self) -> Vec<u8>;
    /// Terminal status after a zero-length read.
    fn status(&self) -> AzureStreamStatus;
    /// The error from `cancellation_details`.
    fn cancel_error(&self) -> TtsError;
}

/// The abort handle for an in-flight blocking read (`stop_speaking_async`).
pub trait AzureStreamAborter: Send + Sync {
    /// Unblock the in-flight `read_data` so the worker can observe stop + exit.
    fn stop_speaking(&self);
}

/// Bridges a blocking [`AzureStreamReader`] to a [`TtsStream`]; `Drop` sets the
/// stop flag and calls the aborter so `read_data` unblocks (barge-in).
struct AzureStream {
    rx: tokio::sync::mpsc::UnboundedReceiver<Result<Vec<u8>, TtsError>>,
    stop: Arc<AtomicBool>,
    aborter: Arc<dyn AzureStreamAborter>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl Stream for AzureStream {
    type Item = Result<Vec<u8>, TtsError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for AzureStream {
    fn drop(&mut self) {
        // Barge-in / consumer drop: unblock the in-flight read and let the daemon
        // worker exit on its own (matching Python's stop-event + daemon thread).
        self.stop.store(true, Ordering::SeqCst);
        self.aborter.stop_speaking();
        if let Some(handle) = self.worker.take() {
            handle.abort();
        }
    }
}

/// Spawn the blocking reader loop and return the bridged stream.
fn azure_stream(
    mut reader: Box<dyn AzureStreamReader>,
    aborter: Arc<dyn AzureStreamAborter>,
    text: String,
) -> TtsStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = Arc::clone(&stop);
    let worker = tokio::task::spawn_blocking(move || {
        if let Err(e) = reader.start(&text) {
            let _ = tx.send(Err(e));
            return;
        }
        let mut total_bytes = 0usize;
        while !stop_worker.load(Ordering::SeqCst) {
            let chunk = reader.read();
            if chunk.is_empty() {
                break;
            }
            total_bytes += chunk.len();
            if tx.send(Ok(chunk)).is_err() {
                break;
            }
        }
        // A `read_data == 0` is ambiguous: only a *natural* end (not a
        // consumer-driven stop) inspects the terminal status for a failure.
        // The mid-stream cancel raises (matching Python) and skips the tail log.
        if !stop_worker.load(Ordering::SeqCst) && reader.status() == AzureStreamStatus::Canceled {
            let _ = tx.send(Err(reader.cancel_error()));
            return;
        }
        log_azure_stream(total_bytes);
    });
    AzureStream {
        rx,
        stop,
        aborter,
        worker: Some(worker),
    }
    .boxed()
}

#[async_trait]
impl TtsClient for AzureTTSClient {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        let backend = self.buffered_backend()?;
        azure_synthesize_with(backend, text.to_owned()).await
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        Some(self)
    }
}

impl StreamingTtsClient for AzureTTSClient {
    fn synthesize_stream(&self, text: &str) -> TtsStream {
        match self.stream_backend() {
            Ok((reader, aborter)) => azure_stream(reader, aborter, text.to_owned()),
            Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
        }
    }

    fn jitter_hints(&self) -> JitterHints {
        JitterHints {
            prebuffer_bytes: self.stream_prebuffer_bytes,
            pad_underrun: self.stream_pad_underrun,
        }
    }
}

/// The live incremental Azure reader + its abort handle.
type AzureStreamParts = (Box<dyn AzureStreamReader>, Arc<dyn AzureStreamAborter>);

impl AzureTTSClient {
    /// Construct the live buffered SDK backend. Deferred to the wiring layer;
    /// the Azure Speech SDK has no drop-in Rust equivalent matching the mock
    /// surface, so this reports "not wired" until subsystem 10 binds it.
    #[allow(clippy::unused_self, reason = "keyed on client config once wired")]
    fn buffered_backend(&self) -> Result<Arc<dyn AzureBufferedBackend>, TtsError> {
        Err(TtsError::Runtime(
            "Azure TTS backend not wired (deferred to wiring layer 10)".to_owned(),
        ))
    }

    /// Construct the live incremental SDK reader + aborter. Deferred (see
    /// [`buffered_backend`](Self::buffered_backend)).
    #[allow(clippy::unused_self, reason = "keyed on client config once wired")]
    fn stream_backend(&self) -> Result<AzureStreamParts, TtsError> {
        Err(TtsError::Runtime(
            "Azure TTS backend not wired (deferred to wiring layer 10)".to_owned(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------------

/// Upsample 16-bit signed LE mono PCM by 2× via linear interpolation: each pair
/// `(a, b)` emits `(a, (a + b) // 2)`; the last sample is doubled. Floor
/// division matches Python `//`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "floor-mean of two i16 samples is always within i16 range"
)]
pub fn upsample_s16le_2x(audio: &[u8]) -> Vec<u8> {
    if audio.is_empty() {
        return Vec::new();
    }
    let samples: Vec<i16> = audio
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n = samples.len();
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let s = samples[i];
        out.extend_from_slice(&s.to_le_bytes());
        let nxt = if i + 1 < n { samples[i + 1] } else { s };
        let mid = (i32::from(s) + i32::from(nxt)).div_euclid(2) as i16;
        out.extend_from_slice(&mid.to_le_bytes());
    }
    out
}

/// Distribute `total_ms` uniformly across whitespace-split words of `text`.
/// Empty when `text` has no words or `total_ms <= 0`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "word counts are tiny; f64 mantissa easily holds them"
)]
pub fn estimate_word_timestamps(text: &str, total_ms: f64) -> Vec<WordTimestamp> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || total_ms <= 0.0 {
        return Vec::new();
    }
    let per = total_ms / words.len() as f64;
    words
        .iter()
        .enumerate()
        .map(|(i, w)| WordTimestamp {
            word: (*w).to_owned(),
            start_ms: i as f64 * per,
            end_ms: (i as f64 + 1.0) * per,
        })
        .collect()
}

/// `Some(&str)` when the option holds a non-empty string (Python truthiness).
fn nonempty(opt: Option<&str>) -> Option<&str> {
    opt.filter(|s| !s.is_empty())
}

/// Compose the Gemini style prompt from structured `[tts]` fields.
///
/// Sections emit in order — `Audio Profile`, `Scene`, `Director's Notes` —
/// newline-joined and suffixed `\nSay:`. `None` when all six style fields unset.
#[must_use]
pub fn compose_gemini_style_prompt(cfg: &TTSConfig) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(profile) = nonempty(cfg.gemini_audio_profile.as_deref()) {
        parts.push(format!("Audio Profile: {profile}"));
    }
    let scene_bits: Vec<&str> = [
        nonempty(cfg.gemini_scene.as_deref()),
        nonempty(cfg.gemini_context.as_deref()),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !scene_bits.is_empty() {
        parts.push(format!("Scene: {}", scene_bits.join(" ")));
    }
    let mut notes: Vec<String> = Vec::new();
    if let Some(style) = nonempty(cfg.gemini_style.as_deref()) {
        notes.push(format!("Style: {style}."));
    }
    if let Some(pace) = nonempty(cfg.gemini_pace.as_deref()) {
        notes.push(format!("Pace: {pace}."));
    }
    if let Some(accent) = nonempty(cfg.gemini_accent.as_deref()) {
        notes.push(format!("Accent: {accent}."));
    }
    if !notes.is_empty() {
        parts.push(format!("Director's Notes: {}", notes.join(" ")));
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!("{}\nSay:", parts.join("\n")))
}

/// Google Gemini Flash TTS client (24 kHz native → 2× upsampled to 48 kHz).
#[derive(Clone, Debug)]
pub struct GeminiTTSClient {
    /// API key.
    pub api_key: String,
    /// Voice name.
    pub voice_name: String,
    /// Model id.
    pub model: String,
    /// Composed style prompt (prepended to the text when set).
    pub style_prompt: Option<String>,
    /// Output sample rate.
    pub sample_rate: u32,
}

impl GeminiTTSClient {
    /// Client with 48 kHz output.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        voice_name: impl Into<String>,
        model: impl Into<String>,
        style_prompt: Option<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            voice_name: voice_name.into(),
            model: model.into(),
            style_prompt,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }
}

/// The blocking Gemini `generate_content` seam (real SDK or a test fake).
pub trait GeminiBackend: Send + Sync {
    /// Generate audio for `contents`; `None` when no audio part is present.
    fn generate(&self, contents: &str, voice_name: &str, model: &str) -> Option<Vec<u8>>;
}

/// Contents string: style prompt + text, or bare text.
fn gemini_build_contents(style_prompt: Option<&str>, text: &str) -> String {
    style_prompt.map_or_else(|| text.to_owned(), |prompt| format!("{prompt}\n\n{text}"))
}

/// Upsample the native 24 kHz PCM and estimate word timestamps from the original
/// `text`. `None` PCM → "no audio part" error.
#[allow(
    clippy::cast_precision_loss,
    reason = "audio byte counts fit f64 mantissa for any realistic utterance"
)]
fn gemini_result_from_pcm(
    pcm_24k: Option<Vec<u8>>,
    text: &str,
    sample_rate: u32,
) -> Result<TTSResult, TtsError> {
    let pcm =
        pcm_24k.ok_or_else(|| TtsError::Runtime("Gemini TTS returned no audio part".to_owned()))?;
    let audio = upsample_s16le_2x(&pcm);
    let total_ms = audio.len() as f64 / 2.0 / f64::from(sample_rate) * 1000.0;
    let timestamps = estimate_word_timestamps(text, total_ms);
    log_gemini_synth(&audio, &timestamps, total_ms);
    Ok(TTSResult { audio, timestamps })
}

/// Run a Gemini backend off the event loop and build the result.
async fn gemini_synthesize_with(
    backend: Arc<dyn GeminiBackend>,
    style_prompt: Option<String>,
    voice_name: String,
    model: String,
    sample_rate: u32,
    text: String,
) -> Result<TTSResult, TtsError> {
    let contents = gemini_build_contents(style_prompt.as_deref(), &text);
    let pcm = tokio::task::spawn_blocking(move || backend.generate(&contents, &voice_name, &model))
        .await
        .map_err(|e| TtsError::Runtime(format!("Gemini worker join: {e}")))?;
    gemini_result_from_pcm(pcm, &text, sample_rate)
}

#[async_trait]
impl TtsClient for GeminiTTSClient {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        let backend = self.backend()?;
        gemini_synthesize_with(
            backend,
            self.style_prompt.clone(),
            self.voice_name.clone(),
            self.model.clone(),
            self.sample_rate,
            text.to_owned(),
        )
        .await
    }
}

impl GeminiTTSClient {
    /// Construct the live SDK backend. Deferred to the wiring layer (`google-genai`
    /// has no Rust SDK matching the mock surface); reports "not wired" until then.
    #[allow(clippy::unused_self, reason = "keyed on client config once wired")]
    fn backend(&self) -> Result<Arc<dyn GeminiBackend>, TtsError> {
        Err(TtsError::Runtime(
            "Gemini TTS backend not wired (deferred to wiring layer 10)".to_owned(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// The concrete TTS client for the active provider.
#[derive(Debug)]
pub enum TtsClientKind {
    /// Cartesia WS client.
    Cartesia(CartesiaTTSClient),
    /// Azure SDK client.
    Azure(AzureTTSClient),
    /// Gemini SDK client.
    Gemini(GeminiTTSClient),
}

impl TtsClientKind {
    /// Erase to a shared `dyn TtsClient` for the player / greeting cache.
    #[must_use]
    pub fn into_dyn(self) -> Arc<dyn TtsClient> {
        match self {
            Self::Cartesia(c) => Arc::new(c),
            Self::Azure(c) => Arc::new(c),
            Self::Gemini(c) => Arc::new(c),
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
        "azure" => {
            let key = env("AZURE_SPEECH_KEY")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    TtsError::Config(
                        "AZURE_SPEECH_KEY environment variable is required for Azure TTS"
                            .to_owned(),
                    )
                })?;
            let region = env("AZURE_SPEECH_REGION")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    TtsError::Config(
                        "AZURE_SPEECH_REGION environment variable is required for Azure TTS"
                            .to_owned(),
                    )
                })?;
            Ok(TtsClientKind::Azure(AzureTTSClient::new(
                key,
                region,
                cfg.azure_voice.clone(),
                DEFAULT_SAMPLE_RATE,
            )))
        }
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
        "gemini" => {
            let api_key = env("GOOGLE_API_KEY")
                .filter(|s| !s.is_empty())
                .or_else(|| env("GEMINI_API_KEY").filter(|s| !s.is_empty()))
                .ok_or_else(|| {
                    TtsError::Config(
                        "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable \
                         is required for Gemini TTS"
                            .to_owned(),
                    )
                })?;
            Ok(TtsClientKind::Gemini(GeminiTTSClient::new(
                api_key,
                cfg.gemini_voice.clone(),
                cfg.gemini_model.clone(),
                compose_gemini_style_prompt(cfg),
            )))
        }
        // Python: `f"Unknown TTS provider {provider!r}; ..."` — `{!r}` (repr)
        // single-quotes the string; `{:?}` (Debug) would double-quote it.
        other => Err(TtsError::Config(format!(
            "Unknown TTS provider '{other}'; expected 'azure', 'cartesia', or 'gemini'"
        ))),
    }
}

#[cfg(test)]
mod tests;
