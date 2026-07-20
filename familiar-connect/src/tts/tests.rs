//! Port of `tests/test_tts.py` — Cartesia (WS via fake conn), greeting cache,
//! Azure (buffered logic + streaming bridge via fake reader), Gemini (upsample /
//! estimate / style-prompt / synth via fake backend), and the factory.

// Test ergonomics: holding a `Mutex` guard across an assertion (and briefly past
// its last read) is fine here — no cross-task contention in a single test.
#![allow(clippy::significant_drop_tightening)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use futures::stream::StreamExt as _;
use serde_json::{Value, json};

use super::{
    AzureBoundaryType, AzureBufferedBackend, AzureStreamAborter, AzureStreamReader,
    AzureStreamStatus, AzureSynthOutcome, AzureTTSClient, AzureWordBoundary, CARTESIA_API_VERSION,
    CARTESIA_BASE_URL, CARTESIA_WS_URL, CartesiaConn, CartesiaFrame, CartesiaStreamState,
    CartesiaTTSClient, DEFAULT_AZURE_VOICE, DEFAULT_SAMPLE_RATE, GeminiBackend, JitterHints,
    TTSResult, TtsClient, TtsClientKind, TtsError, WordTimestamp, azure_stream,
    azure_synthesize_with, build_azure_result, build_tts_client, cartesia_stream_from_state,
    compose_gemini_style_prompt, estimate_word_timestamps, gemini_synthesize_with,
    get_cached_greeting_audio_in, upsample_s16le_2x,
};
use crate::config::TTSConfig;

const TEST_VOICE_ID: &str = "test-voice-id";
const TEST_MODEL: &str = "sonic-3";

fn client() -> CartesiaTTSClient {
    CartesiaTTSClient::new("test-key", TEST_VOICE_ID, TEST_MODEL)
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn pcm(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn unpack(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Fake Cartesia connection
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ConnLog {
    sent: Arc<Mutex<Vec<Value>>>,
    closed: Arc<AtomicBool>,
}

impl ConnLog {
    fn new() -> Self {
        Self {
            sent: Arc::new(Mutex::new(Vec::new())),
            closed: Arc::new(AtomicBool::new(false)),
        }
    }
}

struct FakeConn {
    frames: VecDeque<CartesiaFrame>,
    log: ConnLog,
}

impl FakeConn {
    fn new(frames: Vec<CartesiaFrame>) -> (Self, ConnLog) {
        let log = ConnLog::new();
        (
            Self {
                frames: frames.into(),
                log: log.clone(),
            },
            log,
        )
    }
}

#[async_trait::async_trait]
impl CartesiaConn for FakeConn {
    async fn send_json(&mut self, value: &Value) -> Result<(), TtsError> {
        self.log.sent.lock().unwrap().push(value.clone());
        Ok(())
    }

    async fn recv(&mut self) -> Option<CartesiaFrame> {
        self.frames.pop_front()
    }

    async fn close(&mut self) {
        self.log.closed.store(true, Ordering::SeqCst);
    }

    fn is_closed(&self) -> bool {
        self.log.closed.load(Ordering::SeqCst)
    }
}

fn text_frame(obj: &Value) -> CartesiaFrame {
    CartesiaFrame::Text(obj.to_string())
}

fn chunk_frame(bytes: &[u8]) -> CartesiaFrame {
    text_frame(&json!({ "type": "chunk", "data": BASE64_STANDARD.encode(bytes) }))
}

fn done_frame() -> CartesiaFrame {
    text_frame(&json!({ "type": "done" }))
}

async fn collect_stream(mut s: super::TtsStream) -> (Vec<Vec<u8>>, Option<TtsError>) {
    let mut chunks = Vec::new();
    let mut err = None;
    while let Some(item) = s.next().await {
        match item {
            Ok(c) => chunks.push(c),
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }
    (chunks, err)
}

// ---------------------------------------------------------------------------
// Dataclasses
// ---------------------------------------------------------------------------

#[test]
fn word_timestamp_fields() {
    let ts = WordTimestamp::new("hello", 0.0, 420.0);
    assert_eq!(ts.word, "hello");
    assert!(approx(ts.start_ms, 0.0));
    assert!(approx(ts.end_ms, 420.0));
}

#[test]
fn tts_result_audio_only_default_empty_timestamps() {
    let result = TTSResult::audio_only(vec![0x00, 0x01]);
    assert_eq!(result.audio, vec![0x00, 0x01]);
    assert!(result.timestamps.is_empty());
}

// ---------------------------------------------------------------------------
// Cartesia construction + payload
// ---------------------------------------------------------------------------

#[test]
fn cartesia_init_stores_required_fields() {
    let c = CartesiaTTSClient::new("test-key", "custom-uuid-1234", "sonic-3");
    assert_eq!(c.api_key, "test-key");
    assert_eq!(c.model, "sonic-3");
    assert_eq!(c.voice_id, "custom-uuid-1234");
}

#[test]
fn cartesia_init_defaults() {
    let c = client();
    assert_eq!(c.sample_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(c.sample_rate, 48_000);
    assert_eq!(c.base_url, CARTESIA_BASE_URL);
    assert_eq!(c.ws_url, CARTESIA_WS_URL);
    assert!(c.ws_url.starts_with("wss://"));
}

#[test]
fn cartesia_build_ws_url_includes_auth_query() {
    let c = CartesiaTTSClient::new("sk-cart-test-123", TEST_VOICE_ID, TEST_MODEL);
    let url = c.build_ws_url();
    assert!(url.starts_with(&format!("{CARTESIA_WS_URL}?")));
    assert!(url.contains("api_key=sk-cart-test-123"));
    assert!(url.contains(&format!("cartesia_version={CARTESIA_API_VERSION}")));
}

#[test]
fn cartesia_build_headers_for_rest() {
    let c = CartesiaTTSClient::new("sk-cart-test-123", TEST_VOICE_ID, TEST_MODEL);
    let h = c.build_headers();
    assert_eq!(h["X-API-Key"], "sk-cart-test-123");
    assert_eq!(h["Cartesia-Version"], CARTESIA_API_VERSION);
    assert_eq!(h["Content-Type"], "application/json");
}

#[test]
fn cartesia_payload_structure() {
    let payload = client().build_payload("Hello, world!", "ctx-1");
    assert_eq!(payload["context_id"], json!("ctx-1"));
    assert_eq!(payload["transcript"], json!("Hello, world!"));
    assert_eq!(payload["model_id"], json!(TEST_MODEL));
    assert_eq!(payload["voice"]["mode"], json!("id"));
    assert_eq!(payload["voice"]["id"], json!(TEST_VOICE_ID));
    assert_eq!(payload["add_timestamps"], json!(true));
    assert_eq!(payload["continue"], json!(false));
}

#[test]
fn cartesia_payload_output_format() {
    let c = CartesiaTTSClient {
        sample_rate: 22_050,
        ..CartesiaTTSClient::new("k", "v", "m")
    };
    let payload = c.build_payload("test", "ctx");
    let fmt = &payload["output_format"];
    assert_eq!(fmt["container"], json!("raw"));
    assert_eq!(fmt["encoding"], json!("pcm_s16le"));
    assert_eq!(fmt["sample_rate"], json!(22_050));
}

// ---------------------------------------------------------------------------
// Cartesia synthesize (buffered, over fake conn)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cartesia_returns_tts_result() {
    let data = [0x10, 0x20, 0x30, 0x40];
    let (mut conn, _log) = FakeConn::new(vec![chunk_frame(&data), done_frame()]);
    let result = client().synthesize_conn("Hello", &mut conn).await.unwrap();
    assert_eq!(result.audio, data);
    assert!(result.timestamps.is_empty());
}

#[tokio::test]
async fn cartesia_concatenates_multiple_chunks() {
    let a = [0xaa; 4];
    let b = [0xbb; 4];
    let (mut conn, _log) = FakeConn::new(vec![chunk_frame(&a), chunk_frame(&b), done_frame()]);
    let result = client().synthesize_conn("Hi", &mut conn).await.unwrap();
    assert_eq!(result.audio, [a, b].concat());
}

#[tokio::test]
async fn cartesia_parses_word_timestamps_seconds_to_ms() {
    let (mut conn, _log) = FakeConn::new(vec![
        chunk_frame(&[0x00]),
        text_frame(&json!({
            "type": "timestamps",
            "word_timestamps": { "words": ["Hello", "world"], "start": [0.0, 0.5], "end": [0.42, 0.9] },
        })),
        done_frame(),
    ]);
    let result = client()
        .synthesize_conn("Hello world", &mut conn)
        .await
        .unwrap();
    assert_eq!(
        result.timestamps,
        vec![
            WordTimestamp::new("Hello", 0.0, 420.0),
            WordTimestamp::new("world", 500.0, 900.0),
        ]
    );
}

#[tokio::test]
async fn cartesia_multiple_timestamp_events_accumulate() {
    let (mut conn, _log) = FakeConn::new(vec![
        text_frame(
            &json!({"type":"timestamps","word_timestamps":{"words":["A"],"start":[0.0],"end":[0.1]}}),
        ),
        text_frame(
            &json!({"type":"timestamps","word_timestamps":{"words":["B"],"start":[0.1],"end":[0.2]}}),
        ),
        done_frame(),
    ]);
    let result = client().synthesize_conn("A B", &mut conn).await.unwrap();
    let words: Vec<&str> = result.timestamps.iter().map(|t| t.word.as_str()).collect();
    assert_eq!(words, vec!["A", "B"]);
}

#[tokio::test]
async fn cartesia_sends_request_payload() {
    let (mut conn, log) = FakeConn::new(vec![done_frame()]);
    client().synthesize_conn("Hello", &mut conn).await.unwrap();
    let sent = log.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["transcript"], json!("Hello"));
    assert_eq!(sent[0]["add_timestamps"], json!(true));
    assert_eq!(sent[0]["continue"], json!(false));
    assert!(sent[0].get("context_id").is_some());
}

#[tokio::test]
async fn cartesia_error_event_raises() {
    let (mut conn, _log) = FakeConn::new(vec![text_frame(&json!({
        "type": "error", "error": "voice id unknown", "status_code": 400,
    }))]);
    let err = client()
        .synthesize_conn("Hello", &mut conn)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("voice id unknown"));
}

#[tokio::test]
async fn cartesia_unexpected_close_raises() {
    let (mut conn, _log) = FakeConn::new(vec![CartesiaFrame::Closed]);
    let err = client()
        .synthesize_conn("Hello", &mut conn)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("closed unexpectedly"));
}

// ---------------------------------------------------------------------------
// Cartesia synthesize_stream (over fake conn)
// ---------------------------------------------------------------------------

fn stream_over(frames: Vec<CartesiaFrame>) -> (super::TtsStream, ConnLog) {
    let (conn, log) = FakeConn::new(frames);
    let payload = client().build_payload("hi", "ctx");
    let stream = cartesia_stream_from_state(CartesiaStreamState::Sending {
        conn: Box::new(conn),
        payload,
    });
    (stream, log)
}

#[tokio::test]
async fn cartesia_stream_yields_each_chunk_in_order() {
    let chunks = [vec![0x10, 0x20], vec![0x30, 0x40], vec![0x50, 0x60]];
    let mut frames: Vec<CartesiaFrame> = chunks.iter().map(|c| chunk_frame(c)).collect();
    frames.push(done_frame());
    let (stream, _log) = stream_over(frames);
    let (collected, err) = collect_stream(stream).await;
    assert!(err.is_none());
    assert_eq!(collected, chunks);
}

#[tokio::test]
async fn cartesia_stream_empty_chunks_skipped() {
    let frames = vec![
        text_frame(&json!({"type":"chunk","data":""})),
        chunk_frame(&[0xab]),
        done_frame(),
    ];
    let (stream, _log) = stream_over(frames);
    let (collected, _err) = collect_stream(stream).await;
    assert_eq!(collected, vec![vec![0xab]]);
}

#[tokio::test]
async fn cartesia_stream_timestamps_events_silently_dropped() {
    let frames = vec![
        chunk_frame(&[0x01]),
        text_frame(
            &json!({"type":"timestamps","word_timestamps":{"words":["hi"],"start":[0.0],"end":[0.1]}}),
        ),
        chunk_frame(&[0x02]),
        done_frame(),
    ];
    let (stream, _log) = stream_over(frames);
    let (collected, _err) = collect_stream(stream).await;
    assert_eq!(collected, vec![vec![0x01], vec![0x02]]);
}

#[tokio::test]
async fn cartesia_stream_error_event_raises() {
    let frames = vec![text_frame(&json!({
        "type":"error","error":"voice id unknown","status_code":400,
    }))];
    let (stream, _log) = stream_over(frames);
    let (_collected, err) = collect_stream(stream).await;
    assert!(err.unwrap().to_string().contains("voice id unknown"));
}

#[tokio::test]
async fn cartesia_stream_unexpected_close_raises() {
    let (stream, _log) = stream_over(vec![CartesiaFrame::Closed]);
    let (_collected, err) = collect_stream(stream).await;
    assert!(err.unwrap().to_string().contains("closed unexpectedly"));
}

#[tokio::test]
async fn cartesia_stream_sends_request_payload() {
    let (stream, log) = stream_over(vec![done_frame()]);
    let (_collected, _err) = collect_stream(stream).await;
    let sent = log.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["transcript"], json!("hi"));
    assert_eq!(sent[0]["add_timestamps"], json!(true));
}

#[tokio::test]
async fn cartesia_stream_done_terminates_iteration() {
    let frames = vec![chunk_frame(&[0xaa]), done_frame(), chunk_frame(&[0xbb])];
    let (stream, _log) = stream_over(frames);
    let (collected, _err) = collect_stream(stream).await;
    assert_eq!(collected, vec![vec![0xaa]]);
}

#[tokio::test]
async fn cartesia_stream_closes_conn_on_done() {
    let (stream, log) = stream_over(vec![chunk_frame(&[0x01]), done_frame()]);
    let (_collected, _err) = collect_stream(stream).await;
    assert!(log.closed.load(Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// Greeting cache
// ---------------------------------------------------------------------------

struct MockGreetingClient {
    audio: Vec<u8>,
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl TtsClient for MockGreetingClient {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        self.calls.lock().unwrap().push(text.to_owned());
        Ok(TTSResult::audio_only(self.audio.clone()))
    }
}

#[tokio::test]
async fn greeting_cache_miss_synthesizes_and_writes_file() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockGreetingClient {
        audio: b"cached-audio".to_vec(),
        calls: Arc::clone(&calls),
    };
    let result = get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hello!", &client)
        .await
        .unwrap();
    assert_eq!(result.audio, b"cached-audio");
    assert_eq!(*calls.lock().unwrap(), vec!["Hello!".to_owned()]);
}

#[tokio::test]
async fn greeting_cache_hit_reads_file_without_synthesis() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockGreetingClient {
        audio: b"cached-audio".to_vec(),
        calls: Arc::clone(&calls),
    };
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hello!", &client)
        .await
        .unwrap();
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hello!", &client)
        .await
        .unwrap();
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn greeting_cache_different_voice_id_not_cached() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockGreetingClient {
        audio: b"audio".to_vec(),
        calls: Arc::clone(&calls),
    };
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hello!", &client)
        .await
        .unwrap();
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-2", "Hello!", &client)
        .await
        .unwrap();
    assert_eq!(calls.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn greeting_cache_different_greeting_not_cached() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockGreetingClient {
        audio: b"audio".to_vec(),
        calls: Arc::clone(&calls),
    };
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hello!", &client)
        .await
        .unwrap();
    get_cached_greeting_audio_in(dir.path(), "cartesia", "voice-1", "Hi there!", &client)
        .await
        .unwrap();
    assert_eq!(calls.lock().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// Azure buffered
// ---------------------------------------------------------------------------

fn azure_client() -> AzureTTSClient {
    AzureTTSClient::new("sk-az", "eastus", DEFAULT_AZURE_VOICE, DEFAULT_SAMPLE_RATE)
}

fn word(text: &str, ticks: i64, duration_ms: f64) -> AzureWordBoundary {
    AzureWordBoundary {
        text: text.to_owned(),
        boundary_type: AzureBoundaryType::Word,
        audio_offset_ticks: ticks,
        duration_ms,
    }
}

#[test]
fn azure_init_stores_required_fields() {
    let c = AzureTTSClient::new("sk-az", "westus2", DEFAULT_AZURE_VOICE, DEFAULT_SAMPLE_RATE);
    assert_eq!(c.subscription_key, "sk-az");
    assert_eq!(c.region, "westus2");
}

#[test]
fn azure_custom_voice_stored() {
    let c = AzureTTSClient::new("k", "r", "en-US-JennyNeural", DEFAULT_SAMPLE_RATE);
    assert_eq!(c.voice_name, "en-US-JennyNeural");
}

#[test]
fn azure_defaults_and_zero_preroll() {
    let c = azure_client();
    assert_eq!(c.voice_name, DEFAULT_AZURE_VOICE);
    assert_eq!(c.sample_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(c.stream_prebuffer_bytes, 0);
    assert!(!c.stream_pad_underrun);
}

#[test]
fn azure_exposes_streaming() {
    // Mirrors test_streaming_publicly_exposed: the player auto-selects the
    // streaming path via as_streaming() (the typed hasattr(synthesize_stream)).
    let c = azure_client();
    assert!(c.as_streaming().is_some());
    assert_eq!(
        c.as_streaming().unwrap().jitter_hints(),
        JitterHints::default()
    );
}

#[test]
fn azure_returns_tts_result_with_audio() {
    let out = AzureSynthOutcome::Completed {
        audio: vec![0x10, 0x20, 0x30, 0x40],
        words: vec![],
    };
    let result = build_azure_result(out).unwrap();
    assert_eq!(result.audio, vec![0x10, 0x20, 0x30, 0x40]);
}

#[test]
fn azure_empty_timestamps_when_no_word_events() {
    let out = AzureSynthOutcome::Completed {
        audio: vec![0x00],
        words: vec![],
    };
    assert!(build_azure_result(out).unwrap().timestamps.is_empty());
}

#[test]
fn azure_word_boundary_ticks_converted_to_ms() {
    let out = AzureSynthOutcome::Completed {
        audio: vec![0x00],
        words: vec![word("Hello", 100_000, 50.0), word("world", 700_000, 80.0)],
    };
    let result = build_azure_result(out).unwrap();
    assert_eq!(
        result.timestamps,
        vec![
            WordTimestamp::new("Hello", 10.0, 60.0),
            WordTimestamp::new("world", 70.0, 150.0),
        ]
    );
}

#[test]
fn azure_non_word_boundary_events_skipped() {
    let out = AzureSynthOutcome::Completed {
        audio: vec![0x00],
        words: vec![AzureWordBoundary {
            text: ",".to_owned(),
            boundary_type: AzureBoundaryType::Other,
            audio_offset_ticks: 50_000,
            duration_ms: 10.0,
        }],
    };
    assert!(build_azure_result(out).unwrap().timestamps.is_empty());
}

#[test]
fn azure_raises_runtime_error_on_synthesis_failure() {
    let out = AzureSynthOutcome::Canceled {
        reason: "Error".to_owned(),
        error_details: "something went wrong".to_owned(),
    };
    let err = build_azure_result(out).unwrap_err();
    assert!(err.to_string().contains("Azure TTS"));
}

struct FakeAzureBuffered {
    audio: Vec<u8>,
    calls: Arc<Mutex<Vec<String>>>,
}

impl AzureBufferedBackend for FakeAzureBuffered {
    fn speak(&self, text: &str) -> AzureSynthOutcome {
        self.calls.lock().unwrap().push(text.to_owned());
        AzureSynthOutcome::Completed {
            audio: self.audio.clone(),
            words: vec![],
        }
    }
}

#[tokio::test]
async fn azure_synthesize_runs_in_executor() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let backend = Arc::new(FakeAzureBuffered {
        audio: vec![0xab, 0xcd],
        calls: Arc::clone(&calls),
    });
    let result = azure_synthesize_with(backend, "Hello".to_owned())
        .await
        .unwrap();
    assert_eq!(result.audio, vec![0xab, 0xcd]);
    assert_eq!(*calls.lock().unwrap(), vec!["Hello".to_owned()]);
}

// ---------------------------------------------------------------------------
// Azure streaming bridge
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Gate {
    inner: Arc<(Mutex<bool>, std::sync::Condvar)>,
}

impl Gate {
    fn set(&self) {
        let (m, c) = &*self.inner;
        *m.lock().unwrap() = true;
        c.notify_all();
    }

    fn wait(&self) {
        let (m, c) = &*self.inner;
        let mut g = m.lock().unwrap();
        while !*g {
            g = c.wait(g).unwrap();
        }
    }
}

struct FakeAzureReader {
    chunks: VecDeque<Vec<u8>>,
    status: AzureStreamStatus,
    gate: Option<Gate>,
    start_err: Option<TtsError>,
    started: Arc<Mutex<Vec<String>>>,
}

impl AzureStreamReader for FakeAzureReader {
    fn start(&mut self, text: &str) -> Result<(), TtsError> {
        self.started.lock().unwrap().push(text.to_owned());
        self.start_err.take().map_or(Ok(()), Err)
    }

    fn read(&mut self) -> Vec<u8> {
        if let Some(c) = self.chunks.pop_front() {
            return c;
        }
        if let Some(g) = &self.gate {
            g.wait();
        }
        Vec::new()
    }

    fn status(&self) -> AzureStreamStatus {
        self.status
    }

    fn cancel_error(&self) -> TtsError {
        super::azure_error("Error", "mid-stream cancel")
    }
}

struct FakeAzureAborter {
    gate: Gate,
    count: Arc<AtomicUsize>,
}

impl AzureStreamAborter for FakeAzureAborter {
    fn stop_speaking(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.gate.set();
    }
}

fn azure_reader(
    chunks: Vec<Vec<u8>>,
    status: AzureStreamStatus,
    start_err: Option<TtsError>,
) -> (Box<dyn AzureStreamReader>, Arc<Mutex<Vec<String>>>) {
    let started = Arc::new(Mutex::new(Vec::new()));
    let reader = Box::new(FakeAzureReader {
        chunks: chunks.into(),
        status,
        gate: None,
        start_err,
        started: Arc::clone(&started),
    });
    (reader, started)
}

fn noop_aborter() -> Arc<dyn AzureStreamAborter> {
    Arc::new(FakeAzureAborter {
        gate: Gate::default(),
        count: Arc::new(AtomicUsize::new(0)),
    })
}

#[tokio::test]
async fn azure_stream_yields_each_chunk_in_order() {
    let a = vec![0x10, 0x20, 0x30, 0x40];
    let b = vec![0x50, 0x60];
    let (reader, _started) =
        azure_reader(vec![a.clone(), b.clone()], AzureStreamStatus::AllData, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (collected, err) = collect_stream(stream).await;
    assert!(err.is_none());
    assert_eq!(collected, vec![a, b]);
}

#[tokio::test]
async fn azure_stream_full_buffer_reads_are_distinct() {
    let a = vec![0xaa; super::AZURE_STREAM_BUFFER_BYTES];
    let b = vec![0xbb; super::AZURE_STREAM_BUFFER_BYTES];
    let (reader, _started) =
        azure_reader(vec![a.clone(), b.clone()], AzureStreamStatus::AllData, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (collected, _err) = collect_stream(stream).await;
    assert_eq!(collected, vec![a, b]);
}

#[tokio::test]
async fn azure_stream_uses_start_speaking() {
    let (reader, started) = azure_reader(vec![vec![0x01, 0x02]], AzureStreamStatus::AllData, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let _ = collect_stream(stream).await;
    assert_eq!(*started.lock().unwrap(), vec!["hi".to_owned()]);
}

#[tokio::test]
async fn azure_stream_cancelled_synthesis_raises() {
    let (reader, _started) = azure_reader(
        vec![],
        AzureStreamStatus::AllData,
        Some(super::azure_error("Error", "stream blew up")),
    );
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (_collected, err) = collect_stream(stream).await;
    assert!(err.unwrap().to_string().contains("Azure TTS"));
}

#[tokio::test]
async fn azure_stream_empty_terminates() {
    let (reader, _started) = azure_reader(vec![], AzureStreamStatus::AllData, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (collected, err) = collect_stream(stream).await;
    assert!(err.is_none());
    assert!(collected.is_empty());
}

#[tokio::test]
async fn azure_stream_mid_stream_cancel_status_raises() {
    let (reader, _started) =
        azure_reader(vec![vec![0x01, 0x02]], AzureStreamStatus::Canceled, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (collected, err) = collect_stream(stream).await;
    assert_eq!(collected, vec![vec![0x01, 0x02]]);
    assert!(err.unwrap().to_string().contains("Azure TTS"));
}

#[tokio::test]
async fn azure_stream_clean_eof_does_not_raise() {
    let (reader, _started) = azure_reader(vec![vec![0x01, 0x02]], AzureStreamStatus::AllData, None);
    let stream = azure_stream(reader, noop_aborter(), "hi".to_owned());
    let (collected, err) = collect_stream(stream).await;
    assert!(err.is_none());
    assert_eq!(collected, vec![vec![0x01, 0x02]]);
}

#[tokio::test]
async fn azure_stream_early_close_unblocks_in_flight_read() {
    let gate = Gate::default();
    let started = Arc::new(Mutex::new(Vec::new()));
    let reader = Box::new(FakeAzureReader {
        chunks: vec![vec![0xaa, 0xbb]].into(),
        status: AzureStreamStatus::AllData,
        gate: Some(gate.clone()),
        start_err: None,
        started,
    });
    let count = Arc::new(AtomicUsize::new(0));
    let aborter = Arc::new(FakeAzureAborter {
        gate,
        count: Arc::clone(&count),
    });
    let mut stream = azure_stream(reader, aborter, "hi".to_owned());

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, vec![0xaa, 0xbb]);

    let start = std::time::Instant::now();
    drop(stream);
    let elapsed = start.elapsed();
    assert!(elapsed < std::time::Duration::from_secs(2));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Gemini helpers
// ---------------------------------------------------------------------------

#[test]
fn upsample_doubles_length() {
    let audio = pcm(&[100, 200]);
    assert_eq!(upsample_s16le_2x(&audio).len(), audio.len() * 2);
}

#[test]
fn upsample_first_sample_preserved() {
    let out = upsample_s16le_2x(&pcm(&[1000, 2000]));
    assert_eq!(unpack(&out)[0], 1000);
}

#[test]
fn upsample_interpolates_midpoint() {
    let out = upsample_s16le_2x(&pcm(&[0, 2000]));
    assert_eq!(unpack(&out)[1], 1000);
}

#[test]
fn upsample_last_sample_doubled() {
    let out = upsample_s16le_2x(&pcm(&[500]));
    assert_eq!(unpack(&out), vec![500, 500]);
}

#[test]
fn upsample_empty_input() {
    assert!(upsample_s16le_2x(b"").is_empty());
}

#[test]
fn estimate_empty_text_returns_empty() {
    assert!(estimate_word_timestamps("", 1000.0).is_empty());
}

#[test]
fn estimate_zero_duration_returns_empty() {
    assert!(estimate_word_timestamps("hello world", 0.0).is_empty());
}

#[test]
fn estimate_single_word_spans_full_duration() {
    let out = estimate_word_timestamps("hello", 500.0);
    assert_eq!(out, vec![WordTimestamp::new("hello", 0.0, 500.0)]);
}

#[test]
fn estimate_uniform_distribution() {
    let out = estimate_word_timestamps("one two three four", 400.0);
    assert_eq!(
        out,
        vec![
            WordTimestamp::new("one", 0.0, 100.0),
            WordTimestamp::new("two", 100.0, 200.0),
            WordTimestamp::new("three", 200.0, 300.0),
            WordTimestamp::new("four", 300.0, 400.0),
        ]
    );
}

// ---------------------------------------------------------------------------
// Gemini style-prompt composer
// ---------------------------------------------------------------------------

fn gemini_cfg() -> TTSConfig {
    TTSConfig {
        provider: "gemini".to_owned(),
        ..TTSConfig::default()
    }
}

#[test]
fn compose_returns_none_when_all_fields_empty() {
    assert!(compose_gemini_style_prompt(&gemini_cfg()).is_none());
}

#[test]
fn compose_audio_profile_only() {
    let cfg = TTSConfig {
        gemini_audio_profile: Some("warm contralto".to_owned()),
        ..gemini_cfg()
    };
    let out = compose_gemini_style_prompt(&cfg).unwrap();
    assert!(out.contains("Audio Profile: warm contralto"));
    assert!(!out.contains("Scene:"));
    assert!(!out.contains("Director"));
    assert!(out.ends_with("\nSay:"));
}

#[test]
fn compose_scene_and_context_joined() {
    let cfg = TTSConfig {
        gemini_scene: Some("quiet room".to_owned()),
        gemini_context: Some("tavern keeper".to_owned()),
        ..gemini_cfg()
    };
    assert!(
        compose_gemini_style_prompt(&cfg)
            .unwrap()
            .contains("Scene: quiet room tavern keeper")
    );
}

#[test]
fn compose_scene_only() {
    let cfg = TTSConfig {
        gemini_scene: Some("foggy docks".to_owned()),
        ..gemini_cfg()
    };
    assert!(
        compose_gemini_style_prompt(&cfg)
            .unwrap()
            .contains("Scene: foggy docks")
    );
}

#[test]
fn compose_director_notes_joined() {
    let cfg = TTSConfig {
        gemini_style: Some("playful".to_owned()),
        gemini_pace: Some("relaxed".to_owned()),
        gemini_accent: Some("Irish lilt".to_owned()),
        ..gemini_cfg()
    };
    let out = compose_gemini_style_prompt(&cfg).unwrap();
    assert!(out.contains("Director's Notes:"));
    assert!(out.contains("Style: playful."));
    assert!(out.contains("Pace: relaxed."));
    assert!(out.contains("Accent: Irish lilt."));
}

#[test]
fn compose_all_fields_ends_with_say() {
    let cfg = TTSConfig {
        gemini_audio_profile: Some("old wizard".to_owned()),
        gemini_scene: Some("dark tower".to_owned()),
        gemini_context: Some("foreboding".to_owned()),
        gemini_style: Some("gravelly".to_owned()),
        gemini_pace: Some("slow".to_owned()),
        gemini_accent: Some("British".to_owned()),
        ..gemini_cfg()
    };
    let out = compose_gemini_style_prompt(&cfg).unwrap();
    assert!(out.ends_with("\nSay:"));
    assert!(out.contains("Audio Profile:"));
    assert!(out.contains("Scene:"));
    assert!(out.contains("Director's Notes:"));
}

// ---------------------------------------------------------------------------
// Gemini synthesize (via fake backend)
// ---------------------------------------------------------------------------

struct FakeGeminiBackend {
    pcm: Option<Vec<u8>>,
    contents: Arc<Mutex<Vec<String>>>,
}

impl GeminiBackend for FakeGeminiBackend {
    fn generate(&self, contents: &str, _voice: &str, _model: &str) -> Option<Vec<u8>> {
        self.contents.lock().unwrap().push(contents.to_owned());
        self.pcm.clone()
    }
}

async fn gemini_synth(
    pcm_24k: Option<Vec<u8>>,
    style_prompt: Option<String>,
    text: &str,
) -> (Result<TTSResult, TtsError>, Arc<Mutex<Vec<String>>>) {
    let contents = Arc::new(Mutex::new(Vec::new()));
    let backend = Arc::new(FakeGeminiBackend {
        pcm: pcm_24k,
        contents: Arc::clone(&contents),
    });
    let result = gemini_synthesize_with(
        backend,
        style_prompt,
        "Kore".to_owned(),
        "m".to_owned(),
        DEFAULT_SAMPLE_RATE,
        text.to_owned(),
    )
    .await;
    (result, contents)
}

#[tokio::test]
async fn gemini_returns_upsampled_audio_and_timestamps() {
    let pcm_24k = pcm(&[100, 200, 300, 400]);
    let (result, _contents) = gemini_synth(Some(pcm_24k.clone()), None, "hello world").await;
    assert_eq!(result.unwrap().audio.len(), pcm_24k.len() * 2);
}

#[tokio::test]
async fn gemini_timestamps_cover_original_words() {
    let pcm_24k = pcm(&vec![0; 96_000]);
    let (result, _contents) = gemini_synth(Some(pcm_24k), None, "one two three").await;
    let result = result.unwrap();
    assert_eq!(result.timestamps.len(), 3);
    assert_eq!(result.timestamps[0].word, "one");
    assert_eq!(result.timestamps[2].word, "three");
    assert!(approx(result.timestamps[0].start_ms, 0.0));
    assert!(result.timestamps[2].end_ms > 0.0);
}

#[tokio::test]
async fn gemini_prepends_style_prompt_when_set() {
    let (result, contents) = gemini_synth(
        Some(pcm(&[0; 4])),
        Some("Audio Profile: wizard\nSay:".to_owned()),
        "hello there",
    )
    .await;
    result.unwrap();
    let contents = contents.lock().unwrap();
    assert!(contents[0].contains("Audio Profile: wizard"));
    assert!(contents[0].contains("hello there"));
}

#[tokio::test]
async fn gemini_no_style_prompt_passes_text_unchanged() {
    let (result, contents) = gemini_synth(Some(pcm(&[0; 4])), None, "just the text").await;
    result.unwrap();
    assert_eq!(contents.lock().unwrap()[0], "just the text");
}

#[tokio::test]
async fn gemini_timestamps_from_original_text_not_prompt() {
    let (result, _contents) = gemini_synth(
        Some(pcm(&[0; 4])),
        Some("Audio Profile: quiet\nSay:".to_owned()),
        "alpha beta",
    )
    .await;
    let words: Vec<String> = result
        .unwrap()
        .timestamps
        .into_iter()
        .map(|t| t.word)
        .collect();
    assert_eq!(words, vec!["alpha".to_owned(), "beta".to_owned()]);
}

#[tokio::test]
async fn gemini_raises_when_no_audio_part() {
    let (result, _contents) = gemini_synth(None, None, "hello").await;
    assert!(result.unwrap_err().to_string().contains("no audio"));
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

#[test]
fn factory_creates_cartesia_from_config() {
    let cfg = TTSConfig {
        provider: "cartesia".to_owned(),
        cartesia_voice_id: Some("some-voice-uuid".to_owned()),
        cartesia_model: Some("sonic-turbo".to_owned()),
        ..TTSConfig::default()
    };
    let kind = build_tts_client(&cfg, |k| {
        (k == "CARTESIA_API_KEY").then(|| "sk-cart-test-abc".to_owned())
    })
    .unwrap();
    match kind {
        TtsClientKind::Cartesia(c) => {
            assert_eq!(c.api_key, "sk-cart-test-abc");
            assert_eq!(c.voice_id, "some-voice-uuid");
            assert_eq!(c.model, "sonic-turbo");
        }
        _ => panic!("expected cartesia"),
    }
}

#[test]
fn factory_cartesia_raises_when_api_key_missing() {
    let cfg = TTSConfig {
        provider: "cartesia".to_owned(),
        cartesia_voice_id: Some("v".to_owned()),
        cartesia_model: Some("m".to_owned()),
        ..TTSConfig::default()
    };
    let err = build_tts_client(&cfg, |_| None).unwrap_err();
    assert!(err.to_string().contains("CARTESIA_API_KEY"));
}

#[test]
fn factory_cartesia_raises_when_voice_id_empty() {
    let cfg = TTSConfig {
        provider: "cartesia".to_owned(),
        cartesia_voice_id: Some(String::new()),
        cartesia_model: Some("sonic-3".to_owned()),
        ..TTSConfig::default()
    };
    let err =
        build_tts_client(&cfg, |k| (k == "CARTESIA_API_KEY").then(|| "sk".to_owned())).unwrap_err();
    assert!(err.to_string().contains("voice_id"));
}

#[test]
fn factory_cartesia_raises_when_model_empty() {
    let cfg = TTSConfig {
        provider: "cartesia".to_owned(),
        cartesia_voice_id: Some("some-voice".to_owned()),
        cartesia_model: Some(String::new()),
        ..TTSConfig::default()
    };
    let err =
        build_tts_client(&cfg, |k| (k == "CARTESIA_API_KEY").then(|| "sk".to_owned())).unwrap_err();
    assert!(err.to_string().contains("model"));
}

#[test]
fn factory_creates_azure() {
    let cfg = TTSConfig {
        provider: "azure".to_owned(),
        azure_voice: "en-US-AmberNeural".to_owned(),
        ..TTSConfig::default()
    };
    let kind = build_tts_client(&cfg, |k| match k {
        "AZURE_SPEECH_KEY" => Some("az-key".to_owned()),
        "AZURE_SPEECH_REGION" => Some("eastus".to_owned()),
        _ => None,
    })
    .unwrap();
    match kind {
        TtsClientKind::Azure(c) => {
            assert_eq!(c.subscription_key, "az-key");
            assert_eq!(c.region, "eastus");
            assert_eq!(c.voice_name, "en-US-AmberNeural");
        }
        _ => panic!("expected azure"),
    }
}

#[test]
fn factory_azure_uses_config_voice() {
    let cfg = TTSConfig {
        provider: "azure".to_owned(),
        azure_voice: "en-US-JennyNeural".to_owned(),
        ..TTSConfig::default()
    };
    let kind = build_tts_client(&cfg, |k| match k {
        "AZURE_SPEECH_KEY" => Some("k".to_owned()),
        "AZURE_SPEECH_REGION" => Some("westus2".to_owned()),
        _ => None,
    })
    .unwrap();
    match kind {
        TtsClientKind::Azure(c) => assert_eq!(c.voice_name, "en-US-JennyNeural"),
        _ => panic!("expected azure"),
    }
}

#[test]
fn factory_azure_raises_when_key_missing() {
    let cfg = TTSConfig {
        provider: "azure".to_owned(),
        ..TTSConfig::default()
    };
    let err = build_tts_client(&cfg, |k| {
        (k == "AZURE_SPEECH_REGION").then(|| "eastus".to_owned())
    })
    .unwrap_err();
    assert!(err.to_string().contains("AZURE_SPEECH_KEY"));
}

#[test]
fn factory_azure_raises_when_region_missing() {
    let cfg = TTSConfig {
        provider: "azure".to_owned(),
        ..TTSConfig::default()
    };
    let err =
        build_tts_client(&cfg, |k| (k == "AZURE_SPEECH_KEY").then(|| "k".to_owned())).unwrap_err();
    assert!(err.to_string().contains("AZURE_SPEECH_REGION"));
}

#[test]
fn factory_gemini_raises_when_no_api_key() {
    let cfg = gemini_cfg();
    let err = build_tts_client(&cfg, |_| None).unwrap_err();
    assert!(err.to_string().contains("GOOGLE_API_KEY"));
}

#[test]
fn factory_gemini_accepts_alias() {
    let cfg = gemini_cfg();
    let kind = build_tts_client(&cfg, |k| {
        (k == "GEMINI_API_KEY").then(|| "g-key".to_owned())
    })
    .unwrap();
    match kind {
        TtsClientKind::Gemini(c) => assert_eq!(c.api_key, "g-key"),
        _ => panic!("expected gemini"),
    }
}

#[test]
fn factory_creates_gemini_from_config() {
    let cfg = TTSConfig {
        provider: "gemini".to_owned(),
        gemini_voice: "Puck".to_owned(),
        gemini_model: "gemini-3.1-flash-tts-preview".to_owned(),
        ..TTSConfig::default()
    };
    let kind = build_tts_client(&cfg, |k| {
        (k == "GOOGLE_API_KEY").then(|| "goog-key".to_owned())
    })
    .unwrap();
    match kind {
        TtsClientKind::Gemini(c) => {
            assert_eq!(c.api_key, "goog-key");
            assert_eq!(c.voice_name, "Puck");
            assert_eq!(c.model, "gemini-3.1-flash-tts-preview");
            assert!(c.style_prompt.is_none());
        }
        _ => panic!("expected gemini"),
    }
}

#[test]
fn factory_gemini_composes_style_prompt() {
    let cfg = TTSConfig {
        provider: "gemini".to_owned(),
        gemini_audio_profile: Some("warm narrator".to_owned()),
        gemini_style: Some("calm".to_owned()),
        ..TTSConfig::default()
    };
    let kind = build_tts_client(&cfg, |k| (k == "GOOGLE_API_KEY").then(|| "k".to_owned())).unwrap();
    match kind {
        TtsClientKind::Gemini(c) => {
            let prompt = c.style_prompt.unwrap();
            assert!(prompt.contains("warm narrator"));
            assert!(prompt.contains("calm"));
        }
        _ => panic!("expected gemini"),
    }
}

#[test]
fn factory_unknown_provider_raises() {
    let cfg = TTSConfig {
        provider: "foo".to_owned(),
        ..TTSConfig::default()
    };
    let err = build_tts_client(&cfg, |_| None).unwrap_err();
    // Python: `f"Unknown TTS provider {provider!r}; ..."` — repr single-quotes the
    // provider, so the message reads `...provider 'foo';...` (never `"foo"`).
    assert_eq!(
        err.to_string(),
        "Unknown TTS provider 'foo'; expected 'azure', 'cartesia', or 'gemini'"
    );
}

#[test]
fn factory_gemini_google_key_preferred_over_alias() {
    let cfg = gemini_cfg();
    let kind = build_tts_client(&cfg, |k| match k {
        "GOOGLE_API_KEY" => Some("primary".to_owned()),
        "GEMINI_API_KEY" => Some("alias".to_owned()),
        _ => None,
    })
    .unwrap();
    match kind {
        TtsClientKind::Gemini(c) => assert_eq!(c.api_key, "primary"),
        _ => panic!("expected gemini"),
    }
}
