//! Deepgram streaming transcription backend (subsystem 09; Python `stt/deepgram.py`).
//!
//! Concrete [`Transcriber`] over Deepgram's `/v1/listen` WebSocket. Holds the WS
//! lifecycle, the replay-on-reconnect buffer, the reconnect classification /
//! exponential-backoff policy, `finalize()`-driven flush, and the env-override
//! factory.
//!
//! ## Transport seam (parity + testability)
//!
//! The Python code drives a raw `aiohttp` WebSocket and the tests monkeypatch
//! `_ws_connect` to inject fake sockets. The Rust port keeps that seam explicit:
//! [`WsConnector`] (the `_ws_connect` replacement) yields a [`WsTransport`], and
//! all the lifecycle logic (receive loop, keepalive, reconnect, replay, backoff)
//! is pure Rust over those traits — fully exercised by the mock transport in the
//! test module, with no live network. The real transport ([`net_ws`]) is a
//! hand-rolled `tokio-tungstenite` WebSocket gated behind the default `net`
//! feature. We deliberately do **not** use the `deepgram` crate (the
//! `stt-deepgram` feature): the tested behaviors — exact URL params, the
//! whole-chunk replay buffer, the 1008/4xxx close-code classification, and the
//! `Finalize`-driven flush — live at the raw-frame level the Python pins, which
//! the SDK's higher-level reconnect logic would hide. The `deepgram` crate stays
//! available for a future higher-level path.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::Mutex as TokioMutex;
use tokio::task::JoinHandle;

use crate::config::DeepgramSTTConfig;
use crate::stt::protocol::{SttError, Transcriber, TranscriptSender, TranscriptionResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Deepgram streaming WebSocket endpoint.
pub const DEEPGRAM_WS_URL: &str = "wss://api.deepgram.com/v1/listen";
/// Default Deepgram model.
pub const DEFAULT_MODEL: &str = "nova-3";
/// Default recognition language.
pub const DEFAULT_LANGUAGE: &str = "en";

/// Silence gap before forcing a Deepgram `Finalize`.
///
/// Discord client-side VAD halts RTP during silence, so Deepgram's endpointer
/// never sees in-stream silence and holds the final until the next speech burst.
/// After this many idle seconds the pump sends `{"type":"Finalize"}` to flush.
/// Also reused as the post-replay cushion in [`reconnect`].
pub const DEFAULT_IDLE_FINALIZE_S: f64 = 0.5;

const DEFAULT_KEEPALIVE_INTERVAL: f64 = 3.0;
const DEFAULT_MAX_RECONNECTS: i64 = 5;
const DEFAULT_RECONNECT_DELAY: f64 = 1.0; // base delay; first attempt is immediate
const DEFAULT_RECONNECT_BACKOFF_CAP: f64 = 16.0;
const DEFAULT_IDLE_CLOSE_S: f64 = 30.0;

/// Practical cap on the merged (config + voice-member) keyterm set (#198).
///
/// Keyterm prompting biases nova-3 toward a handful of proper nouns; a busy
/// voice channel could otherwise push dozens of member names into the connect
/// URL, past the point the prompting stays useful (and toward Deepgram's
/// per-request keyterm ceiling). Config keyterms are merged first, so jargon
/// wins the cap over member names.
const MAX_KEYTERMS: usize = 100;

const FINALIZE_JSON: &str = r#"{"type":"Finalize"}"#;
const KEEPALIVE_JSON: &str = r#"{"type":"KeepAlive"}"#;
const CLOSESTREAM_JSON: &str = r#"{"type":"CloseStream"}"#;

// ---------------------------------------------------------------------------
// WebSocket transport seam
// ---------------------------------------------------------------------------

/// A received WebSocket event (mirrors the aiohttp `WSMessage` types the Python
/// receive loop dispatches on).
#[derive(Debug)]
pub enum WsEvent {
    /// A text frame (Deepgram JSON).
    Text(String),
    /// A binary frame (unused by Deepgram RX; passed through and ignored).
    Binary(Vec<u8>),
    /// The server sent a CLOSE frame (carries the optional close reason).
    Close {
        /// Close code from the frame, when present.
        code: Option<i64>,
        /// Close reason text, when present.
        reason: Option<String>,
    },
    /// The transport is closing.
    Closing,
    /// The transport is closed.
    Closed,
    /// A transport-level error occurred.
    Error,
}

/// Transport-level WebSocket error. Almost every call site suppresses these
/// (matching Python's `contextlib.suppress`); the variant is informational.
#[derive(Debug, Error)]
pub enum WsError {
    /// Generic transport failure (connect / send / protocol).
    #[error("websocket transport error")]
    Transport,
    /// No WebSocket transport is available in this build (the `net` feature is
    /// off).
    #[error("websocket transport unavailable (build without the `net` feature)")]
    Unavailable,
}

/// One WebSocket connection.
///
/// All methods take `&self` (interior mutability) so a receive loop can block on
/// [`receive`](WsTransport::receive) while the audio pump and keepalive
/// concurrently [`send_bytes`](WsTransport::send_bytes) /
/// [`send_text`](WsTransport::send_text) — matching aiohttp's concurrent
/// send-during-receive.
#[async_trait]
pub trait WsTransport: Send + Sync {
    /// Send a binary (audio) frame.
    async fn send_bytes(&self, data: &[u8]) -> Result<(), WsError>;
    /// Send a text (JSON control) frame.
    async fn send_text(&self, text: &str) -> Result<(), WsError>;
    /// Await the next event. Returns [`WsEvent::Closed`]/[`WsEvent::Error`] on
    /// transport end rather than erroring (mirrors aiohttp `receive()`).
    async fn receive(&self) -> WsEvent;
    /// Close the connection.
    async fn close(&self);
    /// Whether the transport is closed.
    fn is_closed(&self) -> bool;
    /// The close code observed on this connection, when known.
    fn close_code(&self) -> Option<i64>;
}

/// Opens [`WsTransport`]s. The injectable replacement for Python's `_ws_connect`.
#[async_trait]
pub trait WsConnector: Send + Sync {
    /// Open a WebSocket to `url` with the given headers.
    async fn connect(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> Result<Box<dyn WsTransport>, WsError>;
}

fn default_connector() -> Arc<dyn WsConnector> {
    #[cfg(feature = "net")]
    {
        Arc::new(net_ws::TungsteniteConnector)
    }
    #[cfg(not(feature = "net"))]
    {
        Arc::new(UnavailableConnector)
    }
}

#[cfg(not(feature = "net"))]
struct UnavailableConnector;

#[cfg(not(feature = "net"))]
#[async_trait]
impl WsConnector for UnavailableConnector {
    async fn connect(
        &self,
        _url: &str,
        _headers: &[(String, String)],
    ) -> Result<Box<dyn WsTransport>, WsError> {
        Err(WsError::Unavailable)
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (fully unit-tested without any transport)
// ---------------------------------------------------------------------------

const fn bool_str(b: bool) -> &'static str {
    if b { "true" } else { "false" }
}

/// Encode a query component to match Python's `urlencode` (default
/// `quote_via=quote_plus`, deepgram.py:136): the unreserved set
/// (`A-Za-z0-9-._~`, identical to CPython's `quote` `ALWAYS_SAFE`) is kept
/// verbatim, a space becomes `+`, and every other byte becomes `%XX`. Byte-for-
/// byte identical to the Python-built URL; still round-trips through
/// `parse_qs`-style decoders (which map both `+` and `%20` back to a space).
fn encode_query(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(char::from(b));
            }
            // Python quote_plus renders a space as `+`, not `%20`.
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(char::from(HEX[usize::from(b >> 4)]));
                out.push(char::from(HEX[usize::from(b & 0x0F)]));
            }
        }
    }
    out
}

/// Merge + clean a keyterm set (#198): trim each term, drop empty /
/// whitespace-only and letter-free tokens (emoji-only handles, bare
/// punctuation/digits), case-insensitively dedupe (first spelling wins), and
/// cap the total at [`MAX_KEYTERMS`]. Order is preserved so config keyterms —
/// passed first — survive the cap ahead of runtime member names.
fn normalize_keyterms(terms: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for term in terms {
        let trimmed = term.trim();
        // Drop empty and letter-free tokens (emoji-only handles, `!!!`, `2024`):
        // they add no biasing value and can be noise in the request.
        if trimmed.is_empty() || !trimmed.chars().any(char::is_alphabetic) {
            continue;
        }
        if seen.insert(trimmed.to_lowercase()) {
            out.push(trimmed.to_owned());
            if out.len() >= MAX_KEYTERMS {
                break;
            }
        }
    }
    out
}

/// Classify a close code. `false` → stop; `true` → reconnect.
///
/// Non-int / no-frame (transport drop) → retry; `1008` (policy) → give up;
/// `4000..5000` (Deepgram app errors: auth/billing/quota) → give up; else retry.
fn should_reconnect(close_code: Option<i64>) -> bool {
    match close_code {
        Some(1008) => false,
        Some(c) if (4000..5000).contains(&c) => false,
        None | Some(_) => true,
    }
}

/// Exponential backoff: attempt 1 immediate (`0.0`); attempt `n ≥ 2` waits
/// `min(delay * 2^(n-2), cap)`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the exponent is a small reconnect count; saturating to i32::MAX yields the cap"
)]
fn compute_backoff(consecutive: i64, delay: f64, cap: f64) -> f64 {
    if consecutive > 1 {
        let exponent = i32::try_from(consecutive - 2).unwrap_or(i32::MAX);
        (delay * 2.0_f64.powi(exponent)).min(cap)
    } else {
        0.0
    }
}

/// Real-time duration (seconds) represented by `replay_bytes` of linear16 audio.
#[allow(
    clippy::cast_precision_loss,
    reason = "byte/rate counts are small enough that f64 is exact here"
)]
fn replay_seconds(replay_bytes: usize, sample_rate: i64, channels: i64) -> f64 {
    replay_bytes as f64 / (sample_rate as f64 * channels as f64 * 2.0)
}

/// Parse a Deepgram response frame. `None` for non-`Results` frames or empty
/// transcripts.
fn parse_response(data: &serde_json::Value) -> Option<TranscriptionResult> {
    if data.get("type").and_then(serde_json::Value::as_str) != Some("Results") {
        return None;
    }
    let best = data
        .get("channel")
        .and_then(|c| c.get("alternatives"))
        .and_then(serde_json::Value::as_array)
        .and_then(|alts| alts.first())?;

    let transcript = best
        .get("transcript")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if transcript.is_empty() {
        return None;
    }

    let speaker = best
        .get("words")
        .and_then(serde_json::Value::as_array)
        .and_then(|w| w.first())
        .and_then(|first| first.get("speaker"))
        .and_then(json_to_i64);

    let start = data
        .get("start")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let duration = data
        .get("duration")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let confidence = best
        .get("confidence")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);

    Some(TranscriptionResult {
        text: transcript.to_string(),
        is_final: data
            .get("is_final")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        start,
        end: start + duration,
        confidence,
        speaker,
        user_id: None,
    })
}

/// `int(value)` for a JSON speaker label, mirroring Python
/// `int(words[0]["speaker"])`: a JSON integer passes through, a JSON float
/// truncates toward zero, and a JSON string is parsed as a base-10 integer
/// (Python `int("1") == 1`; leading/trailing whitespace and a sign are
/// tolerated, an unparseable/float-shaped string yields `None`). The Deepgram
/// wire contract always sends an integer, so the string branch is defensive
/// parity for the `int()` coercion rather than an observed frame.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Python int() truncates the diarization label toward zero; matches"
)]
fn json_to_i64(v: &serde_json::Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_f64().map(|f| f as i64))
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
}

// ---------------------------------------------------------------------------
// DeepgramTranscriber
// ---------------------------------------------------------------------------

/// Streaming Deepgram transcriber. Config fields are public (they mirror the
/// Python instance attributes the bot / tests read); the runtime handles are
/// private.
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors Deepgram's independent boolean request knobs 1:1"
)]
pub struct DeepgramTranscriber {
    /// Deepgram API key.
    pub api_key: String,
    /// Model name (default `nova-3`).
    pub model: String,
    /// Language code (default `en`).
    pub language: String,
    /// Input sample rate (Hz; Discord native 48000).
    pub sample_rate: i64,
    /// Input channel count.
    pub channels: i64,
    /// Enable provider diarization.
    pub diarize: bool,
    /// Request interim (non-final) results.
    pub interim_results: bool,
    /// Speech-end grace window (ms).
    pub utterance_end_ms: i64,
    /// Emit Deepgram VAD events.
    pub vad_events: bool,
    /// Hosted-endpointer silence window (ms).
    pub endpointing_ms: i64,
    /// Enable smart formatting.
    pub smart_format: bool,
    /// Enable punctuation.
    pub punctuate: bool,
    /// Keyterm prompting biases.
    pub keyterms: Vec<String>,
    /// Replay-buffer window (seconds).
    pub replay_buffer_s: f64,
    /// KeepAlive interval (seconds).
    pub keepalive_interval_s: f64,
    /// Max consecutive reconnect attempts before giving up.
    pub max_reconnects: i64,
    /// Base reconnect delay (seconds); first attempt is immediate.
    pub reconnect_delay: f64,
    /// Reconnect backoff cap (seconds).
    pub reconnect_backoff_cap: f64,
    /// Per-user idle-close window (seconds) the wiring arms its watchdog from.
    pub idle_close_s: f64,

    connector: Arc<dyn WsConnector>,
    shared: Option<Arc<Shared>>,
    receive_task: Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
}

#[derive(Default)]
struct ReplayState {
    chunks: VecDeque<Vec<u8>>,
    bytes: usize,
}

/// Runtime state shared with the spawned receive + keepalive tasks. Built at
/// [`DeepgramTranscriber::start`] (after any `endpointing_ms` poke), so the
/// connection URL / tunables reflect the final config.
struct Shared {
    connector: Arc<dyn WsConnector>,
    ws: StdMutex<Option<Arc<dyn WsTransport>>>,
    shutting_down: AtomicBool,
    closing: AtomicBool,
    send_lock: TokioMutex<ReplayState>,
    url: String,
    headers: Vec<(String, String)>,
    keepalive_interval_s: f64,
    max_reconnects: i64,
    reconnect_delay: f64,
    reconnect_backoff_cap: f64,
    sample_rate: i64,
    channels: i64,
}

impl DeepgramTranscriber {
    /// Build a transcriber with the default (real, `net`-backed) connector.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_connector(api_key, default_connector())
    }

    /// Build a transcriber with an injected connector (test / wiring seam).
    #[must_use]
    pub fn with_connector(api_key: impl Into<String>, connector: Arc<dyn WsConnector>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            language: DEFAULT_LANGUAGE.to_string(),
            sample_rate: 48000,
            channels: 1,
            diarize: false,
            interim_results: true,
            utterance_end_ms: 1500,
            vad_events: false,
            endpointing_ms: 500,
            smart_format: true,
            punctuate: true,
            keyterms: Vec::new(),
            replay_buffer_s: 5.0,
            keepalive_interval_s: DEFAULT_KEEPALIVE_INTERVAL,
            max_reconnects: DEFAULT_MAX_RECONNECTS,
            reconnect_delay: DEFAULT_RECONNECT_DELAY,
            reconnect_backoff_cap: DEFAULT_RECONNECT_BACKOFF_CAP,
            idle_close_s: DEFAULT_IDLE_CLOSE_S,
            connector,
            shared: None,
            receive_task: None,
            keepalive_task: None,
        }
    }

    /// Deepgram WebSocket URL with query params.
    ///
    /// `interim_results` / `utterance_end_ms` are emitted only when interims are
    /// enabled (Deepgram requires `interim_results=true` for `utterance_end_ms`).
    /// `endpointing` is always emitted. `keyterm` is repeated per term.
    #[must_use]
    pub fn build_ws_url(&self) -> String {
        let mut params: Vec<(&str, String)> = vec![
            ("model", self.model.clone()),
            ("language", self.language.clone()),
            ("sample_rate", self.sample_rate.to_string()),
            ("channels", self.channels.to_string()),
            ("encoding", "linear16".to_string()),
            ("vad_events", bool_str(self.vad_events).to_string()),
            ("endpointing", self.endpointing_ms.to_string()),
            ("smart_format", bool_str(self.smart_format).to_string()),
            ("punctuate", bool_str(self.punctuate).to_string()),
        ];
        if self.interim_results {
            params.push(("interim_results", "true".to_string()));
            params.push(("utterance_end_ms", self.utterance_end_ms.to_string()));
        }
        if self.diarize {
            params.push(("diarize", "true".to_string()));
        }
        for term in &self.keyterms {
            params.push(("keyterm", term.clone()));
        }
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", encode_query(k), encode_query(v)))
            .collect::<Vec<_>>()
            .join("&");
        format!("{DEEPGRAM_WS_URL}?{query}")
    }

    /// Auth headers for the Deepgram WebSocket.
    #[must_use]
    pub fn build_headers(&self) -> Vec<(String, String)> {
        vec![(
            "Authorization".to_string(),
            format!("Token {}", self.api_key),
        )]
    }

    /// Transcriber with the same config + carried tunables, an independent WS,
    /// and no runtime state (Python `clone()`).
    #[must_use]
    fn clone_config(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            language: self.language.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            diarize: self.diarize,
            interim_results: self.interim_results,
            utterance_end_ms: self.utterance_end_ms,
            vad_events: self.vad_events,
            endpointing_ms: self.endpointing_ms,
            smart_format: self.smart_format,
            punctuate: self.punctuate,
            keyterms: self.keyterms.clone(),
            replay_buffer_s: self.replay_buffer_s,
            // Python clone() carries these four env-tuned attrs; reconnect_delay
            // is NOT copied (it resets to the class default 1.0).
            keepalive_interval_s: self.keepalive_interval_s,
            max_reconnects: self.max_reconnects,
            reconnect_delay: DEFAULT_RECONNECT_DELAY,
            reconnect_backoff_cap: self.reconnect_backoff_cap,
            idle_close_s: self.idle_close_s,
            connector: self.connector.clone(),
            shared: None,
            receive_task: None,
            keepalive_task: None,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "Python int(replay_buffer_s * rate * ch * 2) truncates toward zero; matches"
    )]
    fn replay_max_bytes(&self) -> usize {
        let raw = self.replay_buffer_s * self.sample_rate as f64 * self.channels as f64 * 2.0;
        raw as u64 as usize
    }

    async fn start_impl(&mut self, output: TranscriptSender) -> Result<(), SttError> {
        let url = self.build_ws_url();
        let headers = self.build_headers();
        tracing::info!(
            target: "familiar_connect.stt.deepgram",
            url = %url,
            "Deepgram WebSocket connecting"
        );
        let ws = self
            .connector
            .connect(&url, &headers)
            .await
            .map_err(|_| SttError::ConnectFailed)?;
        let ws: Arc<dyn WsTransport> = Arc::from(ws);

        let shared = Arc::new(Shared {
            connector: self.connector.clone(),
            ws: StdMutex::new(Some(ws)),
            shutting_down: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            send_lock: TokioMutex::new(ReplayState::default()),
            url,
            headers,
            keepalive_interval_s: self.keepalive_interval_s,
            max_reconnects: self.max_reconnects,
            reconnect_delay: self.reconnect_delay,
            reconnect_backoff_cap: self.reconnect_backoff_cap,
            sample_rate: self.sample_rate,
            channels: self.channels,
        });
        self.shared = Some(shared.clone());

        let recv_shared = shared.clone();
        self.receive_task = Some(tokio::spawn(async move {
            receive_loop(recv_shared, output).await;
        }));
        let ka_shared = shared;
        self.keepalive_task = Some(tokio::spawn(async move {
            keepalive_loop(ka_shared).await;
        }));
        Ok(())
    }

    #[allow(
        clippy::significant_drop_tightening,
        reason = "the send_lock guard intentionally spans the send so replay-drain and live sends stay ordered (Python holds _send_lock across the send)"
    )]
    async fn send_audio_impl(&self, data: &[u8]) -> Result<(), SttError> {
        let Some(shared) = self.shared.clone() else {
            return Err(SttError::NotConnected);
        };
        if shared.ws.lock().unwrap().is_none() {
            return Err(SttError::NotConnected);
        }
        let max_bytes = self.replay_max_bytes();

        let mut state = shared.send_lock.lock().await;
        // Buffer first — a mid-send drop is still buffered for replay.
        state.chunks.push_back(data.to_vec());
        state.bytes += data.len();
        while state.bytes > max_bytes {
            match state.chunks.pop_front() {
                Some(evicted) => state.bytes -= evicted.len(),
                None => break,
            }
        }

        let ws = shared.ws.lock().unwrap().clone();
        let Some(ws) = ws else { return Ok(()) };
        // `closing` covers the window between the server CLOSE frame and
        // transport-closed; writing there corrupts a clean close.
        if ws.is_closed() || shared.closing.load(Ordering::Relaxed) {
            return Ok(());
        }
        let _ = ws.send_bytes(data).await;
        Ok(())
    }

    async fn finalize_impl(&self) {
        let Some(shared) = self.shared.clone() else {
            return;
        };
        let ws = shared.ws.lock().unwrap().clone();
        let Some(ws) = ws else { return };
        if ws.is_closed() {
            return;
        }
        let _ = ws.send_text(FINALIZE_JSON).await;
    }

    async fn stop_impl(&mut self) {
        // Flip before CloseStream so a late close frame in the receive loop
        // doesn't race into reconnect.
        if let Some(shared) = self.shared.clone() {
            shared.shutting_down.store(true, Ordering::Relaxed);
        }
        if let Some(task) = self.keepalive_task.take() {
            task.abort();
            let _ = task.await;
        }
        if let Some(shared) = self.shared.clone() {
            let ws = shared.ws.lock().unwrap().clone();
            if let Some(ws) = ws {
                if !ws.is_closed() {
                    let _ = ws.send_text(CLOSESTREAM_JSON).await;
                    ws.close().await;
                }
            }
            *shared.ws.lock().unwrap() = None;
        }
        if let Some(task) = self.receive_task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

#[async_trait]
impl Transcriber for DeepgramTranscriber {
    fn clone_transcriber(&self) -> Box<dyn Transcriber> {
        Box::new(self.clone_config())
    }

    async fn start(&mut self, output: TranscriptSender) -> Result<(), SttError> {
        self.start_impl(output).await
    }

    async fn send_audio(&mut self, data: &[u8]) -> Result<(), SttError> {
        self.send_audio_impl(data).await
    }

    async fn finalize(&mut self) {
        self.finalize_impl().await;
    }

    async fn stop(&mut self) {
        self.stop_impl().await;
    }

    fn set_endpointing_ms(&mut self, ms: i64) -> bool {
        self.endpointing_ms = ms;
        true
    }

    fn set_keyterms(&mut self, terms: Vec<String>) -> bool {
        // Merge config keyterms (already on `self.keyterms`) with the provided
        // member names, config first so jargon wins the cap. The result flows
        // into `build_ws_url`, which `start()` reads — so setting before start
        // takes effect for this per-user clone (#198).
        let merged = std::mem::take(&mut self.keyterms)
            .into_iter()
            .chain(terms)
            .collect::<Vec<_>>();
        self.keyterms = normalize_keyterms(merged);
        true
    }

    fn idle_close_s(&self) -> f64 {
        self.idle_close_s
    }

    fn backend_name(&self) -> &'static str {
        "deepgram"
    }
}

fn snapshot_ws(shared: &Shared) -> Option<Arc<dyn WsTransport>> {
    shared.ws.lock().unwrap().clone()
}

async fn receive_loop(shared: Arc<Shared>, output: TranscriptSender) {
    let mut consecutive: i64 = 0;

    while consecutive <= shared.max_reconnects {
        let Some(ws) = snapshot_ws(&shared) else {
            return;
        };
        // Inner loop: read until CLOSE/CLOSING/CLOSED/ERROR. Explicit `receive`
        // (not a `for`-style drain) so we observe the CLOSE frame and flip
        // `closing` immediately.
        loop {
            match ws.receive().await {
                WsEvent::Text(text) => {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                        let msg_type = value
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("");
                        if msg_type == "Results" {
                            if let Some(result) = parse_response(&value) {
                                let _ = output.send(result);
                                consecutive = 0; // real data resets the counter
                            }
                        } else {
                            tracing::info!(
                                target: "familiar_connect.stt.deepgram",
                                event_type = msg_type,
                                data = %text,
                                "Deepgram non-Results event"
                            );
                        }
                    }
                }
                WsEvent::Close { .. } => {
                    shared.closing.store(true, Ordering::Relaxed);
                    break;
                }
                WsEvent::Closing | WsEvent::Closed | WsEvent::Error => break,
                WsEvent::Binary(_) => {}
            }
        }

        let close_code = ws.close_code();
        if shared.shutting_down.load(Ordering::Relaxed) {
            tracing::info!(
                target: "familiar_connect.stt.deepgram",
                close_code = ?close_code,
                "Deepgram shutdown"
            );
            return;
        }
        if !should_reconnect(close_code) {
            tracing::error!(
                target: "familiar_connect.stt.deepgram",
                close_code = ?close_code,
                "Deepgram non-recoverable close"
            );
            return;
        }

        consecutive += 1;
        let backoff = compute_backoff(
            consecutive,
            shared.reconnect_delay,
            shared.reconnect_backoff_cap,
        );
        tracing::warn!(
            target: "familiar_connect.stt.deepgram",
            close_code = ?close_code,
            attempt = consecutive,
            max = shared.max_reconnects,
            backoff_s = backoff,
            "Deepgram reconnecting"
        );
        if backoff > 0.0 {
            tokio::time::sleep(std::time::Duration::from_secs_f64(backoff)).await;
        }
        if reconnect(&shared).await.is_err() {
            tracing::error!(target: "familiar_connect.stt.deepgram", "Deepgram reconnection failed");
            return;
        }
    }

    tracing::error!(
        target: "familiar_connect.stt.deepgram",
        attempts = shared.max_reconnects,
        "Deepgram max reconnects exhausted"
    );
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "the send_lock guard intentionally spans the replay drain so live send_audio queues behind it (Python holds _send_lock across the drain)"
)]
async fn reconnect(shared: &Arc<Shared>) -> Result<(), WsError> {
    // Tear down old.
    if let Some(ws) = snapshot_ws(shared) {
        if !ws.is_closed() {
            ws.close().await;
        }
    }
    // Open new.
    let new_ws = shared
        .connector
        .connect(&shared.url, &shared.headers)
        .await?;
    let new_ws: Arc<dyn WsTransport> = Arc::from(new_ws);
    *shared.ws.lock().unwrap() = Some(new_ws.clone());
    shared.closing.store(false, Ordering::Relaxed);
    tracing::info!(target: "familiar_connect.stt.deepgram", "Deepgram reconnected");

    // Replay buffered audio under send_lock so live senders queue behind.
    let (chunks_replayed, replay_bytes) = {
        let mut state = shared.send_lock.lock().await;
        let n = state.chunks.len();
        let bytes = state.bytes;
        for chunk in &state.chunks {
            let _ = new_ws.send_bytes(chunk).await;
        }
        state.chunks.clear();
        state.bytes = 0;
        (n, bytes)
    };

    if chunks_replayed > 0 {
        // Replay arrives faster than real-time; wait ~replay duration + cushion
        // before Finalize so Deepgram consumes the burst first.
        let replay_s = replay_seconds(replay_bytes, shared.sample_rate, shared.channels);
        tokio::time::sleep(std::time::Duration::from_secs_f64(
            replay_s + DEFAULT_IDLE_FINALIZE_S,
        ))
        .await;
        {
            let _guard = shared.send_lock.lock().await;
            if let Some(ws) = snapshot_ws(shared) {
                if !ws.is_closed() {
                    let _ = ws.send_text(FINALIZE_JSON).await;
                }
            }
        }
        tracing::info!(
            target: "familiar_connect.stt.deepgram",
            chunks = chunks_replayed,
            "Deepgram replay complete"
        );
    }
    Ok(())
}

async fn keepalive_loop(shared: Arc<Shared>) {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs_f64(
            shared.keepalive_interval_s,
        ))
        .await;
        let Some(ws) = snapshot_ws(&shared) else {
            continue;
        };
        if ws.is_closed() || shared.closing.load(Ordering::Relaxed) {
            continue;
        }
        let _ = ws.send_text(KEEPALIVE_JSON).await;
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a [`DeepgramTranscriber`] from `[providers.stt.deepgram]` config.
///
/// `DEEPGRAM_API_KEY` is the only env input.
///
/// # Errors
/// [`SttError::MissingApiKey`] when `DEEPGRAM_API_KEY` is unset or empty.
pub fn create_deepgram_transcriber(
    config: &DeepgramSTTConfig,
) -> Result<DeepgramTranscriber, SttError> {
    let api_key = std::env::var("DEEPGRAM_API_KEY").ok();
    build_deepgram_transcriber(config, api_key)
}

/// Build a [`DeepgramTranscriber`] from config + an injected API key.
///
/// The key-injection seam keeps the factory hermetic (the crate forbids
/// `unsafe`, so tests cannot mutate the process environment): the public
/// [`create_deepgram_transcriber`] reads the env and calls through here.
///
/// # Errors
/// [`SttError::MissingApiKey`] when `api_key` is `None`/empty.
pub fn build_deepgram_transcriber(
    config: &DeepgramSTTConfig,
    api_key: Option<String>,
) -> Result<DeepgramTranscriber, SttError> {
    let api_key = api_key
        .filter(|k| !k.is_empty())
        .ok_or(SttError::MissingApiKey)?;

    let mut t = DeepgramTranscriber::new(api_key);
    t.model.clone_from(&config.model);
    t.language.clone_from(&config.language);
    t.endpointing_ms = config.endpointing_ms;
    t.utterance_end_ms = config.utterance_end_ms;
    t.smart_format = config.smart_format;
    t.punctuate = config.punctuate;
    t.keyterms.clone_from(&config.keyterms);
    t.replay_buffer_s = config.replay_buffer_s;
    t.keepalive_interval_s = config.keepalive_interval_s;
    t.max_reconnects = config.reconnect_max_attempts;
    t.reconnect_backoff_cap = config.reconnect_backoff_cap_s;
    t.idle_close_s = config.idle_close_s;
    Ok(t)
}

// ---------------------------------------------------------------------------
// Real network transport (feature: net) — hand-rolled tokio-tungstenite WS
// ---------------------------------------------------------------------------

#[cfg(feature = "net")]
mod net_ws {
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;
    use futures::{SinkExt, StreamExt};
    use tokio::net::TcpStream;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

    use super::{WsConnector, WsError, WsEvent, WsTransport};

    type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
    type WsSink = futures::stream::SplitSink<WsStream, Message>;
    type WsSource = futures::stream::SplitStream<WsStream>;

    /// Real Deepgram transport over `tokio-tungstenite` (rustls).
    pub struct TungsteniteConnector;

    #[async_trait]
    impl WsConnector for TungsteniteConnector {
        async fn connect(
            &self,
            url: &str,
            headers: &[(String, String)],
        ) -> Result<Box<dyn WsTransport>, WsError> {
            let mut request = url.into_client_request().map_err(|_| WsError::Transport)?;
            {
                let hmap = request.headers_mut();
                for (k, v) in headers {
                    let name =
                        tokio_tungstenite::tungstenite::http::header::HeaderName::from_bytes(
                            k.as_bytes(),
                        )
                        .map_err(|_| WsError::Transport)?;
                    let val =
                        tokio_tungstenite::tungstenite::http::header::HeaderValue::from_str(v)
                            .map_err(|_| WsError::Transport)?;
                    hmap.insert(name, val);
                }
            }
            let (stream, _resp) = tokio_tungstenite::connect_async(request)
                .await
                .map_err(|_| WsError::Transport)?;
            let (sink, source) = stream.split();
            Ok(Box::new(TungsteniteWs {
                sink: tokio::sync::Mutex::new(sink),
                source: tokio::sync::Mutex::new(source),
                closed: AtomicBool::new(false),
                close_code: StdMutex::new(None),
            }))
        }
    }

    struct TungsteniteWs {
        sink: tokio::sync::Mutex<WsSink>,
        source: tokio::sync::Mutex<WsSource>,
        closed: AtomicBool,
        close_code: StdMutex<Option<i64>>,
    }

    #[async_trait]
    impl WsTransport for TungsteniteWs {
        async fn send_bytes(&self, data: &[u8]) -> Result<(), WsError> {
            let mut sink = self.sink.lock().await;
            sink.send(Message::Binary(data.to_vec()))
                .await
                .map_err(|_| WsError::Transport)
        }

        async fn send_text(&self, text: &str) -> Result<(), WsError> {
            let mut sink = self.sink.lock().await;
            sink.send(Message::Text(text.to_owned()))
                .await
                .map_err(|_| WsError::Transport)
        }

        async fn receive(&self) -> WsEvent {
            let mut source = self.source.lock().await;
            loop {
                match source.next().await {
                    Some(Ok(Message::Text(t))) => return WsEvent::Text(t),
                    Some(Ok(Message::Binary(b))) => return WsEvent::Binary(b),
                    Some(Ok(Message::Close(frame))) => {
                        let code = frame.as_ref().map(|f| {
                            let c: u16 = f.code.into();
                            i64::from(c)
                        });
                        let reason = frame.as_ref().map(|f| f.reason.to_string());
                        self.closed.store(true, Ordering::Relaxed);
                        *self.close_code.lock().unwrap() = code;
                        return WsEvent::Close { code, reason };
                    }
                    Some(Ok(_)) => {} // Ping / Pong / Frame — skip, keep reading.
                    Some(Err(_)) => {
                        self.closed.store(true, Ordering::Relaxed);
                        return WsEvent::Error;
                    }
                    None => {
                        self.closed.store(true, Ordering::Relaxed);
                        return WsEvent::Closed;
                    }
                }
            }
        }

        async fn close(&self) {
            let _ = self.sink.lock().await.close().await;
            self.closed.store(true, Ordering::Relaxed);
        }

        fn is_closed(&self) -> bool {
            self.closed.load(Ordering::Relaxed)
        }

        fn close_code(&self) -> Option<i64> {
            *self.close_code.lock().unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    use super::*;

    // ------------------------------------------------------------------
    // Mock transport + connector
    // ------------------------------------------------------------------

    #[derive(Clone, Copy)]
    enum DrainMode {
        /// After queued events drain, return `Closed` forever (dying socket).
        Closed,
        /// After queued events drain, block until an event is pushed (open /
        /// gated socket).
        Block,
    }

    struct MockWsState {
        events: StdMutex<VecDeque<WsEvent>>,
        drain: DrainMode,
        notify: tokio::sync::Notify,
        closed: AtomicBool,
        close_code: Option<i64>,
        sent_bytes: StdMutex<Vec<Vec<u8>>>,
        sent_text: StdMutex<Vec<String>>,
        send_text_fail_remaining: AtomicUsize,
        close_calls: AtomicUsize,
    }

    #[derive(Clone)]
    struct MockWs(Arc<MockWsState>);

    impl MockWs {
        fn new(events: Vec<WsEvent>, drain: DrainMode, close_code: Option<i64>) -> Self {
            Self(Arc::new(MockWsState {
                events: StdMutex::new(events.into_iter().collect()),
                drain,
                notify: tokio::sync::Notify::new(),
                closed: AtomicBool::new(false),
                close_code,
                sent_bytes: StdMutex::new(Vec::new()),
                sent_text: StdMutex::new(Vec::new()),
                send_text_fail_remaining: AtomicUsize::new(0),
                close_calls: AtomicUsize::new(0),
            }))
        }

        fn dying(close_code: Option<i64>) -> Self {
            Self::new(Vec::new(), DrainMode::Closed, close_code)
        }

        fn dying_with(events: Vec<WsEvent>, close_code: Option<i64>) -> Self {
            Self::new(events, DrainMode::Closed, close_code)
        }

        fn open() -> Self {
            Self::new(Vec::new(), DrainMode::Block, None)
        }

        fn set_closed(&self, v: bool) {
            self.0.closed.store(v, Ordering::Relaxed);
        }

        fn release_with_closed(&self) {
            self.0.events.lock().unwrap().push_back(WsEvent::Closed);
            self.0.notify.notify_one();
        }

        fn sent_bytes(&self) -> Vec<Vec<u8>> {
            self.0.sent_bytes.lock().unwrap().clone()
        }

        fn sent_text(&self) -> Vec<String> {
            self.0.sent_text.lock().unwrap().clone()
        }

        fn keepalive_count(&self) -> usize {
            self.sent_text()
                .iter()
                .filter(|s| s.as_str() == KEEPALIVE_JSON)
                .count()
        }

        fn close_calls(&self) -> usize {
            self.0.close_calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl WsTransport for MockWs {
        async fn send_bytes(&self, data: &[u8]) -> Result<(), WsError> {
            self.0.sent_bytes.lock().unwrap().push(data.to_vec());
            Ok(())
        }

        async fn send_text(&self, text: &str) -> Result<(), WsError> {
            let remaining = self.0.send_text_fail_remaining.load(Ordering::Relaxed);
            if remaining > 0 {
                self.0
                    .send_text_fail_remaining
                    .store(remaining - 1, Ordering::Relaxed);
                return Err(WsError::Transport);
            }
            self.0.sent_text.lock().unwrap().push(text.to_string());
            Ok(())
        }

        async fn receive(&self) -> WsEvent {
            loop {
                let next = self.0.events.lock().unwrap().pop_front();
                if let Some(ev) = next {
                    return ev;
                }
                match self.0.drain {
                    DrainMode::Closed => return WsEvent::Closed,
                    DrainMode::Block => self.0.notify.notified().await,
                }
            }
        }

        async fn close(&self) {
            self.0.close_calls.fetch_add(1, Ordering::Relaxed);
            self.0.closed.store(true, Ordering::Relaxed);
        }

        fn is_closed(&self) -> bool {
            self.0.closed.load(Ordering::Relaxed)
        }

        fn close_code(&self) -> Option<i64> {
            self.0.close_code
        }
    }

    struct MockConnector {
        fixed: Option<MockWs>,
        queue: StdMutex<VecDeque<MockWs>>,
        connect_count: Arc<AtomicUsize>,
    }

    impl MockConnector {
        fn fixed(ws: MockWs) -> Arc<Self> {
            Arc::new(Self {
                fixed: Some(ws),
                queue: StdMutex::new(VecDeque::new()),
                connect_count: Arc::new(AtomicUsize::new(0)),
            })
        }

        fn sequence(list: Vec<MockWs>) -> Arc<Self> {
            Arc::new(Self {
                fixed: None,
                queue: StdMutex::new(list.into_iter().collect()),
                connect_count: Arc::new(AtomicUsize::new(0)),
            })
        }

        fn count(&self) -> usize {
            self.connect_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl WsConnector for MockConnector {
        async fn connect(
            &self,
            _url: &str,
            _headers: &[(String, String)],
        ) -> Result<Box<dyn WsTransport>, WsError> {
            self.connect_count.fetch_add(1, Ordering::Relaxed);
            if let Some(f) = &self.fixed {
                return Ok(Box::new(f.clone()));
            }
            let popped = self.queue.lock().unwrap().pop_front();
            let Some(ws) = popped else {
                return Err(WsError::Transport); // simulates StopIteration
            };
            Ok(Box::new(ws))
        }
    }

    fn client_with(conn: Arc<dyn WsConnector>) -> DeepgramTranscriber {
        DeepgramTranscriber::with_connector("test-key", conn)
    }

    async fn yield_until(pred: impl Fn() -> bool) {
        for _ in 0..2000 {
            if pred() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("condition not reached");
    }

    fn results_json(transcript: &str, is_final: bool) -> String {
        serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.99}]},
            "is_final": is_final,
            "start": 0.0,
            "duration": 1.0,
        })
        .to_string()
    }

    // Minimal query parser for URL assertions (percent-decode, group repeats).
    fn parse_query(url: &str) -> std::collections::HashMap<String, Vec<String>> {
        let mut map: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let query = url.split_once('?').map_or("", |(_, q)| q);
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            map.entry(decode(k)).or_default().push(decode(v));
        }
        map
    }

    fn decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap();
                out.push(u8::from_str_radix(hex, 16).unwrap());
                i += 3;
            } else if bytes[i] == b'+' {
                out.push(b' ');
                i += 1;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }

    // ------------------------------------------------------------------
    // Construction / config
    // ------------------------------------------------------------------

    #[test]
    fn init_defaults() {
        let c = DeepgramTranscriber::new("test-key");
        assert_eq!(c.api_key, "test-key");
        assert_eq!(c.model, DEFAULT_MODEL);
        assert_eq!(c.model, "nova-3");
        assert_eq!(c.language, DEFAULT_LANGUAGE);
        assert_eq!(c.language, "en");
        assert_eq!(c.sample_rate, 48000);
    }

    #[test]
    fn init_custom_model() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.model = "nova-2".to_string();
        assert_eq!(c.model, "nova-2");
    }

    #[test]
    fn default_endpointing_widened() {
        assert!(DeepgramTranscriber::new("k").endpointing_ms >= 500);
    }

    #[test]
    fn default_utterance_end_widened() {
        assert!(DeepgramTranscriber::new("k").utterance_end_ms >= 1500);
    }

    // ------------------------------------------------------------------
    // URL / headers
    // ------------------------------------------------------------------

    #[test]
    fn builds_ws_url() {
        let c = DeepgramTranscriber::new("test-key");
        let url = c.build_ws_url();
        let p = parse_query(&url);
        assert!(url.starts_with(DEEPGRAM_WS_URL));
        assert_eq!(p["model"], vec!["nova-3"]);
        assert_eq!(p["language"], vec!["en"]);
        assert_eq!(p["sample_rate"], vec!["48000"]);
        assert_eq!(p["channels"], vec!["1"]);
        assert_eq!(p["encoding"], vec!["linear16"]);
        assert_eq!(p["vad_events"], vec!["false"]);
        assert!(!p.contains_key("diarize"));
    }

    #[test]
    fn builds_ws_url_emits_interim_results_by_default() {
        let c = DeepgramTranscriber::new("test-key");
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["interim_results"], vec!["true"]);
        assert!(p.contains_key("utterance_end_ms"));
    }

    #[test]
    fn builds_ws_url_emits_endpointing() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.endpointing_ms = 300;
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["endpointing"], vec!["300"]);
    }

    #[test]
    fn builds_ws_url_emits_utterance_end_when_interim_on() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.interim_results = true;
        c.utterance_end_ms = 1500;
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["interim_results"], vec!["true"]);
        assert_eq!(p["utterance_end_ms"], vec!["1500"]);
    }

    #[test]
    fn builds_ws_url_omits_interim_and_utterance_when_off() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.interim_results = false;
        let p = parse_query(&c.build_ws_url());
        assert!(!p.contains_key("interim_results"));
        assert!(!p.contains_key("utterance_end_ms"));
    }

    #[test]
    fn builds_ws_url_with_diarize() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.diarize = true;
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["diarize"], vec!["true"]);
    }

    #[test]
    fn smart_format_and_punctuate_enabled_by_default() {
        let c = DeepgramTranscriber::new("test-key");
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["smart_format"], vec!["true"]);
        assert_eq!(p["punctuate"], vec!["true"]);
    }

    #[test]
    fn keyterms_threaded_into_ws_url() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.keyterms = vec![
            "rebasing".to_string(),
            "lifecycle mesh".to_string(),
            "Tam".to_string(),
        ];
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["keyterm"], vec!["rebasing", "lifecycle mesh", "Tam"]);
    }

    #[test]
    fn no_keyterms_omits_param() {
        let c = DeepgramTranscriber::new("test-key");
        assert!(!c.build_ws_url().contains("keyterm="));
    }

    #[test]
    fn keyterm_space_encodes_as_plus_like_python_quote_plus() {
        // Python urlencode/quote_plus renders a space as `+`, not `%20`; the
        // produced URL string must be byte-identical to Python's.
        let mut c = DeepgramTranscriber::new("test-key");
        c.keyterms = vec!["lifecycle mesh".to_string()];
        let url = c.build_ws_url();
        assert!(url.contains("keyterm=lifecycle+mesh"), "url = {url}");
        assert!(!url.contains("%20"), "url = {url}");
        // Still decodes back to the original term.
        assert_eq!(parse_query(&url)["keyterm"], vec!["lifecycle mesh"]);
    }

    // ------------------------------------------------------------------
    // set_keyterms merge/normalize (#198)
    // ------------------------------------------------------------------

    #[test]
    fn set_keyterms_merges_config_and_member_names_into_ws_url() {
        // Config keyterms come first, then the voice-member names get appended;
        // the merged set flows into the connect URL.
        let mut c = DeepgramTranscriber::new("test-key");
        c.keyterms = vec!["lifecycle mesh".to_string(), "Tam".to_string()];
        assert!(c.set_keyterms(vec!["BlueSheep".to_string(), "Ada".to_string()]));
        let p = parse_query(&c.build_ws_url());
        assert_eq!(
            p["keyterm"],
            vec!["lifecycle mesh", "Tam", "BlueSheep", "Ada"]
        );
    }

    #[test]
    fn set_keyterms_dedupes_case_insensitively_config_wins() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.keyterms = vec!["Tam".to_string()];
        c.set_keyterms(vec![
            "tam".to_string(),
            "Ada".to_string(),
            "ADA".to_string(),
        ]);
        // First spelling of each case-insensitive key survives, in order.
        assert_eq!(c.keyterms, vec!["Tam".to_string(), "Ada".to_string()]);
    }

    #[test]
    fn set_keyterms_drops_junk_and_letter_free_tokens() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.set_keyterms(vec![
            "  Ada  ".to_string(),   // trimmed
            String::new(),           // empty
            "   ".to_string(),       // whitespace-only
            "\u{1f984}".to_string(), // emoji-only handle
            "!!!".to_string(),       // punctuation-only
            "2024".to_string(),      // digits-only
        ]);
        assert_eq!(c.keyterms, vec!["Ada".to_string()]);
    }

    #[test]
    fn set_keyterms_caps_total_at_max_keyterms() {
        let mut c = DeepgramTranscriber::new("test-key");
        let many: Vec<String> = (0..(MAX_KEYTERMS + 50))
            .map(|i| format!("name{i}"))
            .collect();
        c.set_keyterms(many);
        assert_eq!(c.keyterms.len(), MAX_KEYTERMS);
        // A per-user clone starts from config keyterms, so an empty config +
        // capped names never exceeds the ceiling in the URL either.
        let count = c.build_ws_url().matches("keyterm=").count();
        assert_eq!(count, MAX_KEYTERMS);
    }

    #[test]
    fn normalize_keyterms_preserves_config_order_ahead_of_names() {
        // Config keyterms (passed first) survive the cap ahead of member names.
        let mut input: Vec<String> = vec!["jargon".to_string()];
        input.extend((0..MAX_KEYTERMS).map(|i| format!("member{i}")));
        let out = normalize_keyterms(input);
        assert_eq!(out.len(), MAX_KEYTERMS);
        assert_eq!(out[0], "jargon");
    }

    #[test]
    fn encode_query_matches_quote_plus() {
        // Unreserved kept verbatim; space → `+`; everything else `%XX` upper-hex.
        assert_eq!(encode_query("aZ0-._~"), "aZ0-._~");
        assert_eq!(encode_query("a b"), "a+b");
        assert_eq!(encode_query("a+b"), "a%2Bb");
        assert_eq!(encode_query("é"), "%C3%A9");
    }

    #[test]
    fn builds_headers() {
        let c = DeepgramTranscriber::new("sk-deepgram-test-123");
        let h = c.build_headers();
        assert_eq!(
            h,
            vec![(
                "Authorization".to_string(),
                "Token sk-deepgram-test-123".to_string()
            )]
        );
    }

    // ------------------------------------------------------------------
    // clone
    // ------------------------------------------------------------------

    #[test]
    fn clone_preserves_keyterms_and_format_flags() {
        let mut orig = DeepgramTranscriber::new("test-key");
        orig.smart_format = false;
        orig.punctuate = false;
        orig.keyterms = vec!["alpha".to_string(), "beta".to_string()];
        let cloned = orig.clone_config();
        assert!(!cloned.smart_format);
        assert!(!cloned.punctuate);
        assert_eq!(
            cloned.keyterms,
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[test]
    fn clone_returns_new_instance_with_same_config() {
        let mut orig = DeepgramTranscriber::new("test-key");
        orig.model = "nova-2".to_string();
        orig.language = "es".to_string();
        orig.sample_rate = 16000;
        orig.channels = 2;
        orig.diarize = true;
        orig.interim_results = false;
        orig.utterance_end_ms = 500;
        orig.vad_events = false;
        orig.endpointing_ms = 450;
        orig.replay_buffer_s = 8.0;
        let cloned = orig.clone_config();
        assert_eq!(cloned.api_key, orig.api_key);
        assert_eq!(cloned.model, orig.model);
        assert_eq!(cloned.language, orig.language);
        assert_eq!(cloned.sample_rate, orig.sample_rate);
        assert_eq!(cloned.channels, orig.channels);
        assert_eq!(cloned.diarize, orig.diarize);
        assert_eq!(cloned.interim_results, orig.interim_results);
        assert_eq!(cloned.utterance_end_ms, orig.utterance_end_ms);
        assert_eq!(cloned.vad_events, orig.vad_events);
        assert_eq!(cloned.endpointing_ms, orig.endpointing_ms);
        assert_eq!(cloned.replay_buffer_s, orig.replay_buffer_s);
    }

    #[test]
    fn clone_carries_env_tuned_attrs() {
        let mut orig = DeepgramTranscriber::new("test-key");
        orig.keepalive_interval_s = 2.5;
        orig.max_reconnects = 7;
        orig.reconnect_backoff_cap = 20.0;
        orig.idle_close_s = 45.0;
        let cloned = orig.clone_config();
        assert_eq!(cloned.keepalive_interval_s, 2.5);
        assert_eq!(cloned.max_reconnects, 7);
        assert_eq!(cloned.reconnect_backoff_cap, 20.0);
        assert_eq!(cloned.idle_close_s, 45.0);
    }

    // ------------------------------------------------------------------
    // parse_response
    // ------------------------------------------------------------------

    #[test]
    fn parse_final_result() {
        let data = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "hello world", "confidence": 0.98}]},
            "is_final": true, "start": 0.0, "duration": 1.5,
        });
        let r = parse_response(&data).unwrap();
        assert_eq!(r.text, "hello world");
        assert!(r.is_final);
        assert_eq!(r.confidence, 0.98);
        assert_eq!(r.start, 0.0);
        assert_eq!(r.end, 1.5);
    }

    #[test]
    fn parse_interim_result() {
        let data = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "hel", "confidence": 0.75}]},
            "is_final": false, "start": 0.0, "duration": 0.5,
        });
        assert!(!parse_response(&data).unwrap().is_final);
    }

    #[test]
    fn parse_empty_transcript_is_none() {
        let data = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "", "confidence": 0.0}]},
            "is_final": true, "start": 0.0, "duration": 1.0,
        });
        assert!(parse_response(&data).is_none());
    }

    #[test]
    fn parse_no_alternatives_is_none() {
        let data = serde_json::json!({
            "type": "Results", "channel": {"alternatives": []},
            "is_final": true, "start": 0.0, "duration": 1.0,
        });
        assert!(parse_response(&data).is_none());
    }

    #[test]
    fn parse_non_results_is_none() {
        let data = serde_json::json!({"type": "Metadata", "request_id": "abc123"});
        assert!(parse_response(&data).is_none());
    }

    #[test]
    fn parse_with_diarize_speaker() {
        let data = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{
                "transcript": "good morning", "confidence": 0.95,
                "words": [{"word": "good", "speaker": 1}, {"word": "morning", "speaker": 1}],
            }]},
            "is_final": true, "start": 2.0, "duration": 1.0,
        });
        assert_eq!(parse_response(&data).unwrap().speaker, Some(1));
    }

    #[test]
    fn parse_speaker_coerces_float_and_string_like_python_int() {
        // Python `int(words[0]["speaker"])` coerces a JSON float (truncating
        // toward zero) and a JSON string; mirror both.
        let float_speaker = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{
                "transcript": "hi", "confidence": 0.9,
                "words": [{"word": "hi", "speaker": 2.0}],
            }]},
            "is_final": true, "start": 0.0, "duration": 1.0,
        });
        assert_eq!(parse_response(&float_speaker).unwrap().speaker, Some(2));

        let string_speaker = serde_json::json!({
            "type": "Results",
            "channel": {"alternatives": [{
                "transcript": "hi", "confidence": 0.9,
                "words": [{"word": "hi", "speaker": "1"}],
            }]},
            "is_final": true, "start": 0.0, "duration": 1.0,
        });
        assert_eq!(parse_response(&string_speaker).unwrap().speaker, Some(1));
    }

    // ------------------------------------------------------------------
    // Pure policy helpers
    // ------------------------------------------------------------------

    #[test]
    fn should_reconnect_classification() {
        assert!(should_reconnect(None)); // transport drop / non-int
        assert!(!should_reconnect(Some(1008))); // policy
        assert!(!should_reconnect(Some(4001))); // deepgram auth
        assert!(!should_reconnect(Some(4000)));
        assert!(!should_reconnect(Some(4999)));
        assert!(should_reconnect(Some(5000)));
        assert!(should_reconnect(Some(1006))); // transport
        assert!(should_reconnect(Some(1000))); // clean, still retried
    }

    #[test]
    fn backoff_sequence() {
        // Attempt 1 immediate; subsequent grow 1x, 2x, 4x... capped.
        assert_eq!(compute_backoff(1, 1.0, 16.0), 0.0);
        assert_eq!(compute_backoff(2, 1.0, 16.0), 1.0);
        assert_eq!(compute_backoff(3, 1.0, 16.0), 2.0);
        assert_eq!(compute_backoff(4, 1.0, 16.0), 4.0);
        assert_eq!(compute_backoff(5, 1.0, 16.0), 8.0);
        assert_eq!(compute_backoff(6, 1.0, 16.0), 16.0);
        assert_eq!(compute_backoff(7, 1.0, 16.0), 16.0); // capped
        // Non-decreasing.
        let seq: Vec<f64> = (1..=7).map(|n| compute_backoff(n, 1.0, 16.0)).collect();
        let mut sorted = seq.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(seq, sorted);
        assert!(seq[6] > seq[1]);
    }

    #[test]
    fn replay_finalize_delay_matches_python() {
        // 48000 bytes at 48 kHz mono s16le == 0.5 s; delay = replay + cushion.
        let replay_s = replay_seconds(48000, 48000, 1);
        assert_eq!(replay_s, 0.5);
        assert_eq!(replay_s + DEFAULT_IDLE_FINALIZE_S, 1.0);
    }

    // ------------------------------------------------------------------
    // Lifecycle (mock transport)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn start_connects_websocket() {
        let conn = MockConnector::fixed(MockWs::open());
        let mut c = client_with(conn.clone());
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        assert_eq!(conn.count(), 1);
        c.stop().await;
    }

    #[tokio::test]
    async fn send_audio_sends_bytes() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.send_audio(b"\x00\x01\x02\x03").await.unwrap();
        assert_eq!(ws.sent_bytes(), vec![b"\x00\x01\x02\x03".to_vec()]);
        c.stop().await;
    }

    #[tokio::test]
    async fn send_audio_before_start_raises() {
        let mut c = DeepgramTranscriber::new("test-key");
        let err = c.send_audio(b"\x00\x01").await.unwrap_err();
        assert!(format!("{err}").contains("not connected"));
    }

    #[tokio::test]
    async fn send_audio_skips_when_ws_closed() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        ws.set_closed(true);
        c.send_audio(b"\x00\x01").await.unwrap();
        assert!(ws.sent_bytes().is_empty());
        c.stop().await;
    }

    #[tokio::test]
    async fn finalize_sends_finalize_message() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.finalize().await;
        assert!(ws.sent_text().iter().any(|s| s == FINALIZE_JSON));
        c.stop().await;
    }

    #[tokio::test]
    async fn finalize_skips_when_ws_closed() {
        let ws = MockWs::open();
        ws.set_closed(true);
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.finalize().await;
        assert!(ws.sent_text().is_empty());
        c.stop().await;
    }

    #[tokio::test]
    async fn finalize_before_start_is_noop() {
        let mut c = DeepgramTranscriber::new("test-key");
        c.finalize().await; // must not panic
    }

    #[tokio::test]
    async fn stop_sends_close_stream() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.stop().await;
        assert_eq!(ws.sent_text(), vec![CLOSESTREAM_JSON.to_string()]);
        assert_eq!(ws.close_calls(), 1);
    }

    #[tokio::test]
    async fn stop_is_idempotent() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.stop().await;
        c.stop().await; // must not panic
        assert_eq!(ws.sent_text(), vec![CLOSESTREAM_JSON.to_string()]);
    }

    // ------------------------------------------------------------------
    // Receive loop
    // ------------------------------------------------------------------

    fn drain_all(rx: &mut UnboundedReceiver<TranscriptionResult>) -> Vec<TranscriptionResult> {
        let mut out = Vec::new();
        while let Ok(r) = rx.try_recv() {
            out.push(r);
        }
        out
    }

    #[tokio::test]
    async fn receive_loop_puts_results_on_queue() {
        let ws = MockWs::dying_with(vec![WsEvent::Text(results_json("hello", true))], Some(1006));
        let conn = MockConnector::fixed(ws);
        let mut c = client_with(conn);
        c.reconnect_delay = 0.0;
        let (tx, mut rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok(); // run to completion
        let results = drain_all(&mut rx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hello");
        assert!(results[0].is_final);
        c.stop().await;
    }

    #[tokio::test]
    async fn receive_loop_ignores_non_result_messages() {
        let meta = serde_json::json!({"type": "Metadata", "request_id": "abc123"}).to_string();
        let ws = MockWs::dying_with(vec![WsEvent::Text(meta)], Some(1006));
        let conn = MockConnector::fixed(ws);
        let mut c = client_with(conn);
        c.reconnect_delay = 0.0;
        let (tx, mut rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        assert!(drain_all(&mut rx).is_empty());
        c.stop().await;
    }

    #[tokio::test]
    async fn receive_loop_drops_vad_events() {
        let s1 = serde_json::json!({"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.1})
            .to_string();
        let s2 =
            serde_json::json!({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.0})
                .to_string();
        let ws = MockWs::dying_with(vec![WsEvent::Text(s1), WsEvent::Text(s2)], Some(1006));
        let conn = MockConnector::fixed(ws);
        let mut c = client_with(conn);
        c.reconnect_delay = 0.0;
        let (tx, mut rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        assert!(drain_all(&mut rx).is_empty());
        c.stop().await;
    }

    #[tokio::test]
    async fn receive_loop_passes_results_through() {
        let s1 = serde_json::json!({"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.1})
            .to_string();
        let s3 =
            serde_json::json!({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.0})
                .to_string();
        let ws = MockWs::dying_with(
            vec![
                WsEvent::Text(s1),
                WsEvent::Text(results_json("hello", true)),
                WsEvent::Text(s3),
            ],
            Some(1006),
        );
        let conn = MockConnector::fixed(ws);
        let mut c = client_with(conn);
        c.reconnect_delay = 0.0;
        let (tx, mut rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        let results = drain_all(&mut rx);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_final);
        c.stop().await;
    }

    // ------------------------------------------------------------------
    // Reconnect
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn reconnects_when_ws_closes() {
        let conn =
            MockConnector::sequence(vec![MockWs::dying(Some(1006)), MockWs::dying(Some(1006))]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        assert!(conn.count() >= 2);
        c.stop().await;
    }

    #[tokio::test]
    async fn send_audio_works_after_reconnect() {
        let ws1 = MockWs::dying(Some(1006));
        ws1.set_closed(true);
        let ws2 = MockWs::open();
        let conn = MockConnector::sequence(vec![ws1, ws2.clone()]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        yield_until(|| conn.count() >= 2).await;
        c.send_audio(b"\x01\x02").await.unwrap();
        assert_eq!(ws2.sent_bytes(), vec![b"\x01\x02".to_vec()]);
        c.stop().await;
    }

    #[tokio::test]
    async fn gives_up_after_max_reconnects() {
        let list: Vec<MockWs> = (0..20).map(|_| MockWs::dying(Some(1006))).collect();
        let conn = MockConnector::sequence(list);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        // start + 6 reconnects (consecutive 1..=6, loop guard `<= 5`) = 7.
        assert_eq!(conn.count(), 7);
        c.stop().await;
    }

    #[tokio::test]
    async fn does_not_reconnect_when_shutting_down() {
        let ws1 = MockWs::open(); // gated: blocks until released
        let conn = MockConnector::sequence(vec![ws1.clone(), MockWs::dying(Some(1006))]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        // Flip the flag before the loop observes the close, then release.
        c.shared
            .as_ref()
            .unwrap()
            .shutting_down
            .store(true, Ordering::Relaxed);
        ws1.release_with_closed();
        c.receive_task.take().unwrap().await.ok();
        assert_eq!(conn.count(), 1); // no reconnect
        c.stop().await;
    }

    #[tokio::test]
    async fn does_not_reconnect_on_auth_close() {
        let conn =
            MockConnector::sequence(vec![MockWs::dying(Some(4001)), MockWs::dying(Some(1006))]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.receive_task.take().unwrap().await.ok();
        assert_eq!(conn.count(), 1); // auth close not retried
        c.stop().await;
    }

    // ------------------------------------------------------------------
    // Replay buffer
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn send_audio_buffers_chunk_when_ws_closed() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        ws.set_closed(true);
        c.send_audio(b"\x01\x02\x03\x04").await.unwrap();
        assert!(ws.sent_bytes().is_empty());
        let shared = c.shared.clone().unwrap();
        let state = shared.send_lock.lock().await;
        assert!(!state.chunks.is_empty());
        drop(state);
        c.stop().await;
    }

    #[tokio::test]
    async fn replay_buffer_chunks_sent_to_new_ws_after_reconnect() {
        let chunk_a = vec![0xAA_u8; 100];
        let chunk_b = vec![0xBB_u8; 100];
        let ws1 = MockWs::open(); // gated
        ws1.set_closed(true); // send_audio buffers
        let ws2 = MockWs::open();
        let conn = MockConnector::sequence(vec![ws1.clone(), ws2.clone()]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();

        c.send_audio(&chunk_a).await.unwrap();
        c.send_audio(&chunk_b).await.unwrap();
        // Both buffered, nothing sent to ws1.
        assert!(ws1.sent_bytes().is_empty());

        ws1.release_with_closed(); // trigger reconnect -> ws2 + replay
        yield_until(|| ws2.sent_bytes().len() >= 2).await;

        assert_eq!(ws2.sent_bytes(), vec![chunk_a.clone(), chunk_b.clone()]);
        let shared = c.shared.clone().unwrap();
        let state = shared.send_lock.lock().await;
        assert_eq!(state.bytes, 0);
        assert!(state.chunks.is_empty());
        drop(state);
        c.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn replay_sends_finalize_after_drain() {
        let chunk = vec![0xAA_u8; 100];
        let ws1 = MockWs::open();
        ws1.set_closed(true);
        let ws2 = MockWs::open();
        let conn = MockConnector::sequence(vec![ws1.clone(), ws2.clone()]);
        let mut c = client_with(conn);
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        c.send_audio(&chunk).await.unwrap();

        ws1.release_with_closed();
        // Wait until the replay drain reaches ws2 (now blocked in the post-drain sleep).
        yield_until(|| !ws2.sent_bytes().is_empty()).await;
        assert!(ws2.sent_text().is_empty()); // no Finalize before the cushion elapses

        tokio::time::advance(std::time::Duration::from_secs(2)).await;
        yield_until(|| ws2.sent_text().iter().any(|s| s == FINALIZE_JSON)).await;

        let finalizes = ws2
            .sent_text()
            .iter()
            .filter(|s| s.as_str() == FINALIZE_JSON)
            .count();
        assert_eq!(finalizes, 1);
        c.stop().await;
    }

    #[tokio::test]
    async fn empty_replay_does_not_send_finalize() {
        let ws1 = MockWs::dying(Some(1006));
        let ws2 = MockWs::open();
        let conn = MockConnector::sequence(vec![ws1, ws2.clone()]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        yield_until(|| conn.count() >= 2).await;
        // A few extra yields to let any (erroneous) finalize path run.
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert!(ws2.sent_text().iter().all(|s| s != FINALIZE_JSON));
        c.stop().await;
    }

    #[tokio::test]
    async fn replay_buffer_trims_oldest_when_over_budget() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        c.sample_rate = 1;
        c.channels = 1; // cap = 5.0 * 1 * 1 * 2 = 10 bytes
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        ws.set_closed(true); // force buffering

        c.send_audio(&[0xAA; 4]).await.unwrap();
        c.send_audio(&[0xBB; 4]).await.unwrap();
        c.send_audio(&[0xCC; 4]).await.unwrap();

        let shared = c.shared.clone().unwrap();
        let state = shared.send_lock.lock().await;
        let buffered: Vec<Vec<u8>> = state.chunks.iter().cloned().collect();
        assert!(!buffered.contains(&vec![0xAA; 4])); // oldest evicted
        assert!(buffered.contains(&vec![0xCC; 4])); // newest retained
        drop(state);
        c.stop().await;
    }

    // ------------------------------------------------------------------
    // Keepalive (paused clock, manual advance)
    // ------------------------------------------------------------------

    async fn tick(ms: u64) {
        tokio::time::advance(std::time::Duration::from_millis(ms)).await;
        tokio::task::yield_now().await;
    }

    #[tokio::test(start_paused = true)]
    async fn keepalive_sent_while_connected() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        c.keepalive_interval_s = 0.02;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        for _ in 0..6 {
            tick(20).await;
        }
        assert!(ws.keepalive_count() >= 2);
        c.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn keepalive_task_cancelled_on_stop() {
        let ws = MockWs::open();
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        c.keepalive_interval_s = 0.02;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        for _ in 0..3 {
            tick(20).await;
        }
        c.stop().await;
        let before = ws.keepalive_count();
        for _ in 0..5 {
            tick(20).await;
        }
        assert_eq!(ws.keepalive_count(), before);
        assert!(c.keepalive_task.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn keepalive_skips_when_ws_closed() {
        let ws = MockWs::open();
        ws.set_closed(true);
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        c.keepalive_interval_s = 0.02;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        for _ in 0..6 {
            tick(20).await;
        }
        assert_eq!(ws.keepalive_count(), 0);
        c.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn keepalive_survives_send_errors() {
        let ws = MockWs::open();
        ws.0.send_text_fail_remaining.store(1, Ordering::Relaxed); // first send fails
        let conn = MockConnector::fixed(ws.clone());
        let mut c = client_with(conn);
        c.keepalive_interval_s = 0.02;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        for _ in 0..6 {
            tick(20).await;
        }
        // First KeepAlive failed (not recorded) but the loop kept ticking.
        assert!(ws.keepalive_count() >= 2);
        c.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn keepalive_follows_reconnect() {
        let ws1 = MockWs::dying(Some(1006));
        let ws2 = MockWs::open();
        let conn = MockConnector::sequence(vec![ws1, ws2.clone()]);
        let mut c = client_with(conn.clone());
        c.reconnect_delay = 0.0;
        c.keepalive_interval_s = 0.02;
        let (tx, _rx) = unbounded_channel();
        c.start(tx).await.unwrap();
        yield_until(|| conn.count() >= 2).await;
        for _ in 0..6 {
            tick(20).await;
        }
        assert!(ws2.keepalive_count() >= 1);
        c.stop().await;
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    #[test]
    fn factory_uses_default_config() {
        let c = build_deepgram_transcriber(
            &DeepgramSTTConfig::default(),
            Some("sk-deepgram-test-abc".to_string()),
        )
        .unwrap();
        assert_eq!(c.api_key, "sk-deepgram-test-abc");
        assert_eq!(c.model, DEFAULT_MODEL);
        assert_eq!(c.language, DEFAULT_LANGUAGE);
    }

    #[test]
    fn factory_raises_when_api_key_missing() {
        let err = build_deepgram_transcriber(&DeepgramSTTConfig::default(), None)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("DEEPGRAM_API_KEY"));
        // Empty string also treated as missing.
        let err2 = build_deepgram_transcriber(&DeepgramSTTConfig::default(), Some(String::new()))
            .err()
            .unwrap();
        assert!(matches!(err2, SttError::MissingApiKey));
    }

    #[test]
    fn factory_toml_config_flows_through() {
        let cfg = DeepgramSTTConfig {
            model: "nova-2".to_string(),
            language: "es".to_string(),
            endpointing_ms: 275,
            utterance_end_ms: 1750,
            smart_format: false,
            punctuate: false,
            keyterms: vec!["lifecycle mesh".to_string(), "Tam".to_string()],
            replay_buffer_s: 8.0,
            keepalive_interval_s: 2.5,
            reconnect_max_attempts: 7,
            reconnect_backoff_cap_s: 20.0,
            idle_close_s: 45.0,
        };
        let c = build_deepgram_transcriber(&cfg, Some("sk-test".to_string())).unwrap();
        assert_eq!(c.model, "nova-2");
        assert_eq!(c.language, "es");
        assert_eq!(c.endpointing_ms, 275);
        assert_eq!(c.utterance_end_ms, 1750);
        assert!(!c.smart_format);
        assert!(!c.punctuate);
        assert_eq!(
            c.keyterms,
            vec!["lifecycle mesh".to_string(), "Tam".to_string()]
        );
        assert_eq!(c.replay_buffer_s, 8.0);
        assert_eq!(c.keepalive_interval_s, 2.5);
        assert_eq!(c.max_reconnects, 7);
        assert_eq!(c.reconnect_backoff_cap, 20.0);
        assert_eq!(c.idle_close_s, 45.0);
    }

    // ------------------------------------------------------------------
    // endpointing_ms poke seam
    // ------------------------------------------------------------------

    #[test]
    fn set_endpointing_ms_pokes_and_reports_support() {
        let mut c = DeepgramTranscriber::new("k");
        assert!(c.set_endpointing_ms(10));
        assert_eq!(c.endpointing_ms, 10);
        // The poke lands on the URL Deepgram connects with.
        let p = parse_query(&c.build_ws_url());
        assert_eq!(p["endpointing"], vec!["10"]);
    }

    #[test]
    fn trait_seams_report_deepgram() {
        let c = DeepgramTranscriber::new("k");
        assert_eq!(c.backend_name(), "deepgram");
        assert_eq!(c.idle_close_s(), 30.0);
    }
}
