//! Azure tests — request shape, credentials, buffered + streaming over a
//! scripted backend. No network: the SDK adapter is exercised only through
//! its pure event mapping.

#![allow(
    clippy::significant_drop_tightening,
    clippy::unnecessary_wraps,
    reason = "script helpers build `Result` items; guards held across single-task asserts"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::stream::StreamExt as _;

use super::{
    AZURE_OUTPUT_FORMAT, AzureBackend, AzureEvent, AzureEventStream, AzureRequest, AzureTTSClient,
    azure_credentials, azure_endpoint, build_ssml,
};
use crate::tts::{
    JitterHints, StreamingTtsClient as _, TtsClient, TtsError, TtsStream, WordTimestamp,
};

// ---------------------------------------------------------------------------
// Scripted backend
// ---------------------------------------------------------------------------

/// Sets its flag on drop — observes stream teardown.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

type Script = Vec<Result<AzureEvent, TtsError>>;

struct FakeBackend {
    /// `Err` fails `open` itself.
    script: Mutex<Option<Result<Script, TtsError>>>,
    requests: Mutex<Vec<AzureRequest>>,
    dropped: Arc<AtomicBool>,
}

impl FakeBackend {
    fn new(script: Script) -> Arc<Self> {
        Self::with(Ok(script))
    }

    fn with(script: Result<Script, TtsError>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(Some(script)),
            requests: Mutex::new(Vec::new()),
            dropped: Arc::new(AtomicBool::new(false)),
        })
    }

    fn opened(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

#[async_trait]
impl AzureBackend for FakeBackend {
    async fn open(&self, request: AzureRequest) -> Result<AzureEventStream, TtsError> {
        self.requests.lock().unwrap().push(request);
        let script = self.script.lock().unwrap().take().expect("opened once")?;
        let flag = DropFlag(Arc::clone(&self.dropped));
        // The flag rides in the stream state, so it drops with the stream.
        Ok(
            futures::stream::unfold((script.into_iter(), flag), |(mut it, flag)| async move {
                it.next().map(|ev| (ev, (it, flag)))
            })
            .boxed(),
        )
    }
}

fn client(backend: Arc<FakeBackend>) -> AzureTTSClient {
    AzureTTSClient::with_backend("eastus", "en-US-AmberNeural", backend)
}

fn audio(bytes: &[u8]) -> Result<AzureEvent, TtsError> {
    Ok(AzureEvent::Audio(bytes.to_vec()))
}

fn word(text: &str, offset_ticks: i64, duration_ticks: i64) -> Result<AzureEvent, TtsError> {
    Ok(AzureEvent::Word {
        text: text.to_owned(),
        offset_ticks,
        duration_ticks,
    })
}

const fn end() -> Result<AzureEvent, TtsError> {
    Ok(AzureEvent::End)
}

async fn collect_stream(mut s: TtsStream) -> (Vec<Vec<u8>>, Option<TtsError>) {
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
// Request shape
// ---------------------------------------------------------------------------

#[test]
fn endpoint_follows_region_host_rule() {
    assert_eq!(
        azure_endpoint("eastus"),
        "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
    );
    assert_eq!(
        azure_endpoint("chinaeast2"),
        "wss://chinaeast2.tts.speech.azure.cn/cognitiveservices/websocket/v1"
    );
    assert_eq!(
        azure_endpoint("usgovvirginia"),
        "wss://usgovvirginia.tts.speech.azure.us/cognitiveservices/websocket/v1"
    );
}

#[test]
fn ssml_selects_voice_and_locale() {
    assert_eq!(
        build_ssml("en-GB-SoniaNeural", "Hello there"),
        "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" \
         xml:lang=\"en-GB\"><voice name=\"en-GB-SoniaNeural\">Hello there</voice></speak>"
    );
}

#[test]
fn ssml_locale_falls_back_for_unprefixed_voice() {
    assert!(build_ssml("CustomVoice", "hi").contains("xml:lang=\"en-US\""));
}

#[test]
fn ssml_escapes_text_and_voice() {
    let ssml = build_ssml("a\"b", "Tom & Jerry <3 'quoted' \"x\"");
    assert!(ssml.contains("<voice name=\"a&quot;b\">"));
    assert!(ssml.contains("Tom &amp; Jerry &lt;3 &apos;quoted&apos; &quot;x&quot;</voice>"));
}

#[test]
fn build_request_carries_voice_format_region() {
    let c = client(FakeBackend::new(vec![]));
    let req = c.build_request("Hi & bye");
    assert_eq!(req.region, "eastus");
    assert_eq!(req.voice, "en-US-AmberNeural");
    assert_eq!(req.output_format, "raw-48khz-16bit-mono-pcm");
    assert_eq!(req.output_format, AZURE_OUTPUT_FORMAT);
    assert_eq!(req.ssml, build_ssml("en-US-AmberNeural", "Hi & bye"));
    assert_eq!(
        req.endpoint(),
        "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
    );
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

fn creds(key: Option<&str>, region: Option<&str>) -> Result<(String, String), TtsError> {
    let key = key.map(str::to_owned);
    let region = region.map(str::to_owned);
    azure_credentials(move |k| match k {
        "AZURE_SPEECH_KEY" => key.clone(),
        "AZURE_SPEECH_REGION" => region.clone(),
        _ => None,
    })
}

#[test]
fn credentials_read_key_and_region() {
    let (key, region) = creds(Some("abc123"), Some("westeurope")).unwrap();
    assert_eq!(key, "abc123");
    assert_eq!(region, "westeurope");
}

#[test]
fn credentials_missing_or_empty_key() {
    for key in [None, Some("")] {
        let err = creds(key, Some("eastus")).unwrap_err();
        assert!(matches!(err, TtsError::Config(_)));
        assert_eq!(
            err.to_string(),
            "AZURE_SPEECH_KEY environment variable is required for Azure TTS"
        );
    }
}

#[test]
fn credentials_missing_or_empty_region() {
    for region in [None, Some("")] {
        let err = creds(Some("k"), region).unwrap_err();
        assert_eq!(
            err.to_string(),
            "AZURE_SPEECH_REGION environment variable is required for Azure TTS"
        );
    }
}

#[test]
fn credentials_reject_region_the_sdk_would_panic_on() {
    // The SDK `unwrap`s the URI built from the region.
    for region in ["east us", "eastus/evil", "eastus\n"] {
        let err = creds(Some("k"), Some(region)).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "AZURE_SPEECH_REGION '{}' is not an Azure region name \
                 (letters and digits only, e.g. eastus)",
                region.escape_debug()
            )
        );
    }
}

#[test]
fn credentials_reject_key_the_sdk_would_panic_on() {
    // The SDK `unwrap`s the key into a header value; a stray CR/LF from a
    // Windows-edited .env would panic mid-conversation.
    for key in ["abc\r", "ab c", "abc\n"] {
        let err = creds(Some(key), Some("eastus")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "AZURE_SPEECH_KEY contains whitespace or control characters; \
             check .env for stray spaces or line endings"
        );
    }
}

// ---------------------------------------------------------------------------
// Buffered synthesize
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synthesize_concatenates_audio() {
    let backend = FakeBackend::new(vec![audio(&[1, 2, 3]), audio(&[4, 5, 6]), end()]);
    let result = client(Arc::clone(&backend)).synthesize("hi").await.unwrap();
    assert_eq!(result.audio, vec![1, 2, 3, 4, 5, 6]);
    assert!(result.timestamps.is_empty());
}

#[tokio::test]
async fn synthesize_sends_built_request() {
    let backend = FakeBackend::new(vec![end()]);
    let c = client(Arc::clone(&backend));
    c.synthesize("hello").await.unwrap();
    assert_eq!(
        *backend.requests.lock().unwrap(),
        vec![c.build_request("hello")]
    );
}

#[tokio::test]
async fn synthesize_converts_word_ticks_to_ms() {
    let backend = FakeBackend::new(vec![
        audio(&[0, 0]),
        word("Hello", 500_000, 5_125_000),
        word("world", 6_000_000, 4_000_000),
        end(),
    ]);
    let result = client(backend).synthesize("Hello world").await.unwrap();
    assert_eq!(
        result.timestamps,
        vec![
            WordTimestamp::new("Hello", 50.0, 562.5),
            WordTimestamp::new("world", 600.0, 1000.0),
        ]
    );
}

#[tokio::test]
async fn synthesize_trims_trailing_half_sample() {
    let backend = FakeBackend::new(vec![audio(&[1, 2, 3]), end()]);
    let result = client(backend).synthesize("x").await.unwrap();
    assert_eq!(result.audio, vec![1, 2]);
}

#[tokio::test]
async fn synthesize_open_failure_propagates() {
    let backend = FakeBackend::with(Err(TtsError::Transport(
        "Azure WS connect failed: 401".to_owned(),
    )));
    let err = client(backend).synthesize("x").await.unwrap_err();
    assert!(matches!(err, TtsError::Transport(_)));
    assert_eq!(err.to_string(), "Azure WS connect failed: 401");
}

#[tokio::test]
async fn synthesize_mid_stream_error_propagates() {
    let backend = FakeBackend::new(vec![
        audio(&[1, 2]),
        Err(TtsError::Runtime("Azure TTS error: boom".to_owned())),
    ]);
    let err = client(backend).synthesize("x").await.unwrap_err();
    assert_eq!(err.to_string(), "Azure TTS error: boom");
}

#[tokio::test]
async fn synthesize_without_turn_end_is_truncation() {
    let backend = FakeBackend::new(vec![audio(&[1, 2])]);
    let err = client(backend).synthesize("x").await.unwrap_err();
    assert!(matches!(err, TtsError::Runtime(_)));
    assert_eq!(err.to_string(), "Azure TTS stream ended before turn.end");
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_yields_audio_in_order_skipping_metadata() {
    let backend = FakeBackend::new(vec![
        audio(&[1, 2]),
        word("hi", 0, 10),
        audio(&[]),
        audio(&[3, 4, 5, 6]),
        end(),
    ]);
    let (chunks, err) = collect_stream(client(backend).synthesize_stream("hi")).await;
    assert!(err.is_none());
    assert_eq!(chunks, vec![vec![1, 2], vec![3, 4, 5, 6]]);
}

#[tokio::test]
async fn stream_realigns_odd_chunks_to_whole_samples() {
    // The player's mono→stereo rejects odd lengths; carry the half sample.
    let backend = FakeBackend::new(vec![
        audio(&[1, 2, 3]),
        audio(&[4]),
        audio(&[5, 6, 7]),
        end(),
    ]);
    let (chunks, err) = collect_stream(client(backend).synthesize_stream("x")).await;
    assert!(err.is_none());
    assert_eq!(chunks, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
    assert!(chunks.iter().all(|c| c.len() % 2 == 0));
}

#[tokio::test]
async fn stream_opens_lazily_on_first_poll() {
    let backend = FakeBackend::new(vec![audio(&[1, 2]), end()]);
    let c = client(Arc::clone(&backend));
    let mut s = c.synthesize_stream("x");
    assert_eq!(backend.opened(), 0);
    assert_eq!(s.next().await.unwrap().unwrap(), vec![1, 2]);
    assert_eq!(backend.opened(), 1);
    assert_eq!(backend.requests.lock().unwrap()[0], c.build_request("x"));
}

#[tokio::test]
async fn stream_open_failure_is_first_item() {
    let backend = FakeBackend::with(Err(TtsError::Transport(
        "Azure WS connect failed: dns".to_owned(),
    )));
    let (chunks, err) = collect_stream(client(backend).synthesize_stream("x")).await;
    assert!(chunks.is_empty());
    assert_eq!(err.unwrap().to_string(), "Azure WS connect failed: dns");
}

#[tokio::test]
async fn stream_mid_stream_error_surfaces_after_chunks() {
    let backend = FakeBackend::new(vec![
        audio(&[1, 2]),
        Err(TtsError::Runtime("Azure TTS error: gone".to_owned())),
        audio(&[3, 4]),
    ]);
    let (chunks, err) = collect_stream(client(backend).synthesize_stream("x")).await;
    assert_eq!(chunks, vec![vec![1, 2]]);
    assert_eq!(err.unwrap().to_string(), "Azure TTS error: gone");
}

#[tokio::test]
async fn stream_without_turn_end_is_truncation() {
    let backend = FakeBackend::new(vec![audio(&[1, 2])]);
    let (chunks, err) = collect_stream(client(backend).synthesize_stream("x")).await;
    assert_eq!(chunks, vec![vec![1, 2]]);
    assert_eq!(
        err.unwrap().to_string(),
        "Azure TTS stream ended before turn.end"
    );
}

#[tokio::test]
async fn stream_terminates_after_error() {
    let backend = FakeBackend::new(vec![Err(TtsError::Runtime("x".to_owned())), end()]);
    let mut s = client(backend).synthesize_stream("x");
    assert!(s.next().await.unwrap().is_err());
    assert!(s.next().await.is_none());
}

#[tokio::test]
async fn dropping_stream_tears_down_backend() {
    // Barge-in: consumer drops mid-flight.
    let backend = FakeBackend::new(vec![audio(&[1, 2]), audio(&[3, 4]), end()]);
    let c = client(Arc::clone(&backend));
    let mut s = c.synthesize_stream("x");
    s.next().await.unwrap().unwrap();
    assert!(!backend.dropped.load(Ordering::SeqCst));
    drop(s);
    assert!(backend.dropped.load(Ordering::SeqCst));
}

#[test]
fn exposes_streaming_seam_with_default_hints() {
    let c = client(FakeBackend::new(vec![]));
    let streaming = c.as_streaming().expect("azure streams");
    assert_eq!(streaming.jitter_hints(), JitterHints::default());
}

// ---------------------------------------------------------------------------
// SDK adapter (pure parts only — no connection)
// ---------------------------------------------------------------------------

#[cfg(feature = "azure-tts")]
mod sdk {
    use azure_speech::synthesizer::message::{BoundaryType, Metadata, Text};
    use azure_speech::synthesizer::ssml::ToSSML as _;
    use azure_speech::synthesizer::{Event, Language, Voice};

    use std::time::Duration;

    use futures::{SinkExt as _, StreamExt as _};
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message as WsMessage;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    use super::super::{
        AZURE_OUTPUT_FORMAT, PreparedSsml, build_ssml, map_sdk_event, open_at, sdk_audio_format,
    };
    use super::{AzureEvent, AzureRequest};

    fn rid() -> azure_speech::RequestId {
        uuid::Uuid::nil()
    }

    fn text(t: &str, kind: BoundaryType) -> Text {
        Text {
            text: t.to_owned(),
            length: i64::try_from(t.len()).unwrap(),
            boundary_type: kind,
        }
    }

    #[test]
    fn requested_format_is_raw_48k_mono_pcm() {
        assert_eq!(sdk_audio_format().as_str(), AZURE_OUTPUT_FORMAT);
    }

    #[test]
    fn prepared_ssml_is_sent_verbatim() {
        let doc = "<speak>x</speak>".to_owned();
        let got = PreparedSsml(doc.clone())
            .to_ssml(Language::default(), Voice::default())
            .unwrap();
        assert_eq!(got, doc);
    }

    #[test]
    fn maps_audio_and_end() {
        assert_eq!(
            map_sdk_event(Ok(Event::Synthesising(rid(), vec![1, 2])))
                .into_iter()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
            vec![AzureEvent::Audio(vec![1, 2])]
        );
        assert_eq!(
            map_sdk_event(Ok(Event::SessionEnded(rid())))
                .into_iter()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
            vec![AzureEvent::End]
        );
    }

    #[test]
    fn ignores_session_bookkeeping() {
        assert!(map_sdk_event(Ok(Event::SessionStarted(rid()))).is_empty());
        assert!(map_sdk_event(Ok(Event::Synthesised(rid()))).is_empty());
    }

    #[test]
    fn maps_word_boundaries_only() {
        let events = map_sdk_event(Ok(Event::AudioMetadata(
            rid(),
            vec![
                Metadata::WordBoundary {
                    offset: 500_000,
                    duration: 1_000,
                    text: text("Hello", BoundaryType::Word),
                },
                Metadata::WordBoundary {
                    offset: 600_000,
                    duration: 1_000,
                    text: text("!", BoundaryType::Punctuation),
                },
                Metadata::SessionEnd { offset: 9 },
            ],
        )));
        assert_eq!(
            events.into_iter().map(Result::unwrap).collect::<Vec<_>>(),
            vec![AzureEvent::Word {
                text: "Hello".to_owned(),
                offset_ticks: 500_000,
                duration_ticks: 1_000,
            }]
        );
    }

    #[test]
    fn maps_sdk_error_to_runtime() {
        let events = map_sdk_event(Err(azure_speech::Error::Timeout));
        assert_eq!(events.len(), 1);
        let err = events.into_iter().next().unwrap().unwrap_err();
        assert!(matches!(err, crate::tts::TtsError::Runtime(_)));
        assert_eq!(
            err.to_string(),
            "Azure TTS error: Timed out waiting for server message"
        );
    }

    // --- Loopback: real SDK protocol path against a local fake service -----

    /// `headers\r\n\r\nbody` text frame, SDK wire shape.
    fn text_frame(request_id: &str, path: &str, body: &str) -> WsMessage {
        WsMessage::Text(format!(
            "X-RequestId:{request_id}\r\nPath:{path}\r\nContent-Type:application/json\r\n\r\n{body}"
        ))
    }

    /// Binary frame: u16-BE header length, headers, payload.
    fn audio_frame(request_id: &str, stream_id: &str, pcm: &[u8]) -> WsMessage {
        let headers =
            format!("X-RequestId:{request_id}\r\nPath:audio\r\nX-StreamId:{stream_id}\r\n");
        let len = u16::try_from(headers.len()).unwrap();
        let mut out = len.to_be_bytes().to_vec();
        out.extend_from_slice(headers.as_bytes());
        out.extend_from_slice(pcm);
        WsMessage::Binary(out)
    }

    fn header<'a>(frame: &'a str, name: &str) -> Option<&'a str> {
        frame
            .split("\r\n\r\n")
            .next()?
            .split("\r\n")
            .find_map(|l| l.strip_prefix(&format!("{name}:")))
    }

    fn body(frame: &str) -> &str {
        frame.split_once("\r\n\r\n").map_or("", |(_, b)| b)
    }

    /// What the fake service saw.
    #[derive(Debug, Default)]
    struct Seen {
        subscription_key: Option<String>,
        context: Option<Value>,
        ssml: Option<String>,
    }

    #[allow(
        clippy::result_large_err,
        reason = "tungstenite's handshake callback signature"
    )]
    /// Serve one synthesis: capture the handshake + three request frames,
    /// then play `reply(request_id)` (or close early when `truncate`).
    async fn fake_service(truncate: bool) -> (String, tokio::task::JoinHandle<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "ws://{}/cognitiveservices/websocket/v1",
            listener.local_addr().unwrap()
        );
        let handle = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut seen = Seen::default();
            let mut key = None;
            let mut ws =
                tokio_tungstenite::accept_hdr_async(tcp, |req: &Request, resp: Response| {
                    key = req
                        .headers()
                        .get("ocp-apim-subscription-key")
                        .map(|v| v.to_str().unwrap().to_owned());
                    Ok(resp)
                })
                .await
                .unwrap();
            seen.subscription_key = key;
            let mut request_id = String::new();
            while seen.ssml.is_none() {
                let Some(Ok(WsMessage::Text(frame))) = ws.next().await else {
                    panic!("expected request text frame");
                };
                request_id = header(&frame, "X-RequestId").unwrap().to_owned();
                match header(&frame, "Path").unwrap() {
                    "synthesis.context" => {
                        seen.context = Some(serde_json::from_str(body(&frame)).unwrap());
                    }
                    "ssml" => seen.ssml = Some(body(&frame).to_owned()),
                    _ => {}
                }
            }
            let rid = request_id.as_str();
            let mut frames = vec![
                text_frame(rid, "turn.start", "{}"),
                text_frame(
                    rid,
                    "response",
                    r#"{"audio":{"type":"inline","streamId":"s1"}}"#,
                ),
                audio_frame(rid, "s1", &[1, 2, 3, 4]),
            ];
            if !truncate {
                frames.extend([
                    text_frame(
                        rid,
                        "audio.metadata",
                        r#"{"Metadata":[{"Type":"WordBoundary","Data":{"Offset":500000,"Duration":2000000,"text":{"Text":"Hello","Length":5,"BoundaryType":"WordBoundary"}}}]}"#,
                    ),
                    audio_frame("someone-else", "s1", &[9, 9]),
                    audio_frame(rid, "s1", &[5, 6]),
                    text_frame(rid, "turn.end", "{}"),
                ]);
            }
            for f in frames {
                ws.send(f).await.unwrap();
            }
            if truncate {
                let _ = ws.close(None).await;
            } else {
                // Hold open until the client closes.
                while let Some(Ok(_)) = ws.next().await {}
            }
            seen
        });
        (url, handle)
    }

    fn request() -> AzureRequest {
        AzureRequest {
            region: "eastus".to_owned(),
            voice: "en-US-AmberNeural".to_owned(),
            output_format: AZURE_OUTPUT_FORMAT,
            ssml: build_ssml("en-US-AmberNeural", "Hello"),
        }
    }

    async fn drain_events(
        mut events: super::AzureEventStream,
    ) -> Vec<Result<AzureEvent, crate::tts::TtsError>> {
        let mut out = Vec::new();
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(10), events.next())
            .await
            .expect("event within 10s")
        {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn loopback_sends_request_and_relays_events() {
        let (url, service) = fake_service(false).await;
        let events = open_at(&url, "loopback-key", request()).await.unwrap();
        let got = drain_events(events).await;
        let got: Vec<AzureEvent> = got.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            got,
            vec![
                AzureEvent::Audio(vec![1, 2, 3, 4]),
                AzureEvent::Word {
                    text: "Hello".to_owned(),
                    offset_ticks: 500_000,
                    duration_ticks: 2_000_000,
                },
                AzureEvent::Audio(vec![5, 6]),
                AzureEvent::End,
            ]
        );

        let seen = tokio::time::timeout(Duration::from_secs(10), service)
            .await
            .expect("service sees the close")
            .unwrap();
        assert_eq!(seen.subscription_key.as_deref(), Some("loopback-key"));
        assert_eq!(seen.ssml.as_deref(), Some(request().ssml.as_str()));
        let audio = &seen.context.unwrap()["synthesis"]["audio"];
        assert_eq!(audio["outputFormat"], "raw-48khz-16bit-mono-pcm");
        assert_eq!(audio["metadataOptions"]["wordBoundaryEnabled"], true);
    }

    #[tokio::test]
    async fn loopback_server_close_mid_synthesis_is_an_error() {
        let (url, _service) = fake_service(true).await;
        let events = open_at(&url, "k", request()).await.unwrap();
        let got = drain_events(events).await;
        assert_eq!(
            got[0].as_ref().unwrap(),
            &AzureEvent::Audio(vec![1, 2, 3, 4])
        );
        let err = got.last().unwrap().as_ref().unwrap_err();
        assert!(
            err.to_string().starts_with("Azure TTS error: "),
            "got {err}"
        );
        assert!(!got.iter().any(|e| matches!(e, Ok(AzureEvent::End))));
    }

    #[tokio::test]
    async fn loopback_connect_refused_is_transport_error() {
        // Bind then drop: nothing listens on the port.
        let addr = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap();
        let url = format!("ws://{addr}/");
        let Err(err) = open_at(&url, "k", request()).await else {
            panic!("connect must fail");
        };
        assert!(matches!(err, crate::tts::TtsError::Transport(_)));
        assert!(
            err.to_string()
                .starts_with(&format!("Azure WS connect failed ({url}): ")),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn malformed_key_is_config_error_not_panic() {
        let Err(err) = open_at("ws://127.0.0.1:1/", "bad\nkey", request()).await else {
            panic!("bad key must fail");
        };
        assert!(matches!(err, crate::tts::TtsError::Config(_)));
    }
}
