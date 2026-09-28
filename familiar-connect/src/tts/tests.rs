//! TTS tests — Cartesia (WS via fake conn), greeting cache, and the factory.

// Test ergonomics: holding a `Mutex` guard across an assertion (and briefly past
// its last read) is fine here — no cross-task contention in a single test.
#![allow(clippy::significant_drop_tightening)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use futures::stream::StreamExt as _;
use serde_json::{Value, json};

use super::{
    CARTESIA_API_VERSION, CARTESIA_BASE_URL, CARTESIA_WS_URL, CartesiaConn, CartesiaFrame,
    CartesiaStreamState, CartesiaTTSClient, DEFAULT_SAMPLE_RATE, JitterHints, TTSResult, TtsClient,
    TtsClientKind, TtsError, WordTimestamp, build_tts_client, cartesia_stream_from_state,
    get_cached_greeting_audio_in,
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

#[test]
fn cartesia_exposes_the_streaming_seam_with_default_hints() {
    let c = client();
    let streaming = c.as_streaming().expect("cartesia streams");
    // Steady cadence: no pre-roll, no underrun padding.
    assert_eq!(streaming.jitter_hints(), JitterHints::default());
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
    let TtsClientKind::Cartesia(c) = kind;
    assert_eq!(c.api_key, "sk-cart-test-abc");
    assert_eq!(c.voice_id, "some-voice-uuid");
    assert_eq!(c.model, "sonic-turbo");
}

#[test]
fn the_shipped_default_provider_is_wired() {
    // Guards the #N1 regression: the shipped default must be synthesizable.
    let cfg = TTSConfig {
        cartesia_voice_id: Some("v".to_owned()),
        cartesia_model: Some("m".to_owned()),
        ..TTSConfig::default()
    };
    build_tts_client(&cfg, |k| (k == "CARTESIA_API_KEY").then(|| "sk".to_owned()))
        .expect("shipped default builds");
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
fn factory_unknown_provider_raises() {
    let cfg = TTSConfig {
        provider: "foo".to_owned(),
        ..TTSConfig::default()
    };
    let err = build_tts_client(&cfg, |_| None).unwrap_err();
    // The message single-quotes the
    // provider, so the message reads `...provider 'foo';...` (never `"foo"`).
    assert_eq!(
        err.to_string(),
        "Unknown TTS provider 'foo'; expected 'cartesia'"
    );
}

#[test]
fn factory_rejects_the_removed_stub_providers() {
    // #N1: azure/gemini were unwired stubs, deleted outright.
    for provider in ["azure", "gemini"] {
        let cfg = TTSConfig {
            provider: provider.to_owned(),
            ..TTSConfig::default()
        };
        let err = build_tts_client(&cfg, |_| None).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("Unknown TTS provider '{provider}'; expected 'cartesia'")
        );
    }
}
