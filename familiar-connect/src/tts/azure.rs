//! Azure Speech TTS client (subsystem 09).
//!
//! One `azure-speech` WebSocket synthesis per utterance. Requests
//! [`AZURE_OUTPUT_FORMAT`] — the same mono `pcm_s16le` @ 48 kHz the player
//! takes from Cartesia, so no resampling. The SDK yields audio chunks as they
//! arrive, so the client is streaming as well as buffered.
//!
//! Logic runs against the [`AzureBackend`] seam; only the SDK adapter needs the
//! `azure-tts` feature. Without it the factory refuses `provider = "azure"` with
//! [`AZURE_TTS_FEATURE_MISSING`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt as _};

use super::{
    DEFAULT_SAMPLE_RATE, StreamTel, StreamingTtsClient, TTSResult, TtsClient, TtsError, TtsStream,
    WordTimestamp, log_buffered_synth,
};

/// Output format requested from Azure: raw mono s16le @ [`DEFAULT_SAMPLE_RATE`].
pub const AZURE_OUTPUT_FORMAT: &str = "raw-48khz-16bit-mono-pcm";
/// 100 ns ticks per millisecond — Azure word-boundary offset unit.
pub const AZURE_TICKS_PER_MS: f64 = 10_000.0;
/// `xml:lang` fallback when the voice name carries no locale prefix.
const AZURE_FALLBACK_LANG: &str = "en-US";

/// Factory refusal when `provider = "azure"` but the binary lacks the SDK.
pub const AZURE_TTS_FEATURE_MISSING: &str = "TTS provider 'azure' requires the 'azure-tts' \
     feature. Rebuild with `azure-tts` added to --features \
     (e.g. `cargo build --release --features discord,discord-voice,azure-tts`).";

const _: () = assert!(
    DEFAULT_SAMPLE_RATE == 48_000,
    "AZURE_OUTPUT_FORMAT pins 48 kHz"
);

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Everything one synthesis sends; built by the client, consumed by the backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AzureRequest {
    /// Azure region (`eastus`, …); selects the endpoint host.
    pub region: String,
    /// Neural voice name.
    pub voice: String,
    /// Azure `outputFormat` string (SDK adapter sends the matching enum).
    pub output_format: &'static str,
    /// Complete `<speak>` document.
    pub ssml: String,
}

impl AzureRequest {
    /// WS endpoint dialed for this request's region.
    #[must_use]
    pub fn endpoint(&self) -> String {
        azure_endpoint(&self.region)
    }
}

/// TTS WebSocket URL for `region`. Same host rule as `azure-speech` 0.10
/// (contains `china` → `azure.cn`, `usgov*` → `azure.us`, else `microsoft.com`).
#[must_use]
pub fn azure_endpoint(region: &str) -> String {
    let host = if region.contains("china") {
        ".azure.cn"
    } else if region.to_lowercase().starts_with("usgov") {
        ".azure.us"
    } else {
        ".microsoft.com"
    };
    format!("wss://{region}.tts.speech{host}/cognitiveservices/websocket/v1")
}

/// `<speak>` document selecting `voice`, with `text` XML-escaped.
#[must_use]
pub fn build_ssml(voice: &str, text: &str) -> String {
    format!(
        "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" \
         xml:lang=\"{}\"><voice name=\"{}\">{}</voice></speak>",
        xml_escape(voice_locale(voice)),
        xml_escape(voice),
        xml_escape(text),
    )
}

/// Locale prefix of a `ll-CC-NameNeural` voice, else [`AZURE_FALLBACK_LANG`].
fn voice_locale(voice: &str) -> &str {
    let mut dashes = voice.match_indices('-').map(|(i, _)| i);
    match (dashes.next(), dashes.next()) {
        (Some(_), Some(second)) => &voice[..second],
        _ => AZURE_FALLBACK_LANG,
    }
}

fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Backend seam
// ---------------------------------------------------------------------------

/// One event from an in-flight synthesis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AzureEvent {
    /// Raw PCM in [`AZURE_OUTPUT_FORMAT`]; not necessarily sample-aligned.
    Audio(Vec<u8>),
    /// Word boundary (punctuation / sentence boundaries never surface).
    Word {
        /// Spoken token.
        text: String,
        /// Audio offset, 100 ns ticks.
        offset_ticks: i64,
        /// Duration, 100 ns ticks.
        duration_ticks: i64,
    },
    /// Service sent `turn.end`; synthesis complete.
    End,
}

/// Events of one synthesis. Ending without [`AzureEvent::End`] is a truncation.
pub type AzureEventStream = BoxStream<'static, Result<AzureEvent, TtsError>>;

/// Synthesis transport: the SDK adapter, or a scripted fake in tests.
#[async_trait]
pub trait AzureBackend: Send + Sync {
    /// Connect, send `request`, return its event stream. Dropping the stream
    /// tears the connection down.
    async fn open(&self, request: AzureRequest) -> Result<AzureEventStream, TtsError>;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Azure Speech TTS client; one connection per synthesis.
#[derive(Clone)]
pub struct AzureTTSClient {
    /// Azure region.
    pub region: String,
    /// Neural voice name.
    pub voice: String,
    backend: Arc<dyn AzureBackend>,
}

impl std::fmt::Debug for AzureTTSClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureTTSClient")
            .field("region", &self.region)
            .field("voice", &self.voice)
            .finish_non_exhaustive()
    }
}

impl AzureTTSClient {
    /// Client over an explicit backend (tests; the SDK path is [`Self::new`]).
    #[must_use]
    pub fn with_backend(
        region: impl Into<String>,
        voice: impl Into<String>,
        backend: Arc<dyn AzureBackend>,
    ) -> Self {
        Self {
            region: region.into(),
            voice: voice.into(),
            backend,
        }
    }

    /// Request for one utterance.
    #[must_use]
    pub fn build_request(&self, text: &str) -> AzureRequest {
        AzureRequest {
            region: self.region.clone(),
            voice: self.voice.clone(),
            output_format: AZURE_OUTPUT_FORMAT,
            ssml: build_ssml(&self.voice, text),
        }
    }
}

#[cfg(feature = "azure-tts")]
impl AzureTTSClient {
    /// Client over the `azure-speech` SDK.
    #[must_use]
    pub fn new(
        subscription_key: impl Into<String>,
        region: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        Self::with_backend(
            region,
            voice,
            Arc::new(sdk::SdkAzureBackend {
                subscription_key: subscription_key.into(),
            }),
        )
    }
}

/// Truncation: event stream ended without `turn.end`.
fn truncated() -> TtsError {
    TtsError::Runtime("Azure TTS stream ended before turn.end".to_owned())
}

/// Prepend the carried half sample; carry a trailing odd byte forward.
fn align_samples(carry: &mut Option<u8>, chunk: Vec<u8>) -> Vec<u8> {
    let mut out = match carry.take() {
        Some(b) => {
            let mut v = Vec::with_capacity(chunk.len() + 1);
            v.push(b);
            v.extend_from_slice(&chunk);
            v
        }
        None => chunk,
    };
    if out.len() % 2 == 1 {
        *carry = out.pop();
    }
    out
}

#[allow(
    clippy::cast_precision_loss,
    reason = "tick counts for one utterance fit the f64 mantissa exactly"
)]
fn ticks_to_ms(ticks: i64) -> f64 {
    ticks as f64 / AZURE_TICKS_PER_MS
}

/// Streaming state across `unfold` polls.
enum AzureStreamState {
    /// Lazily open on first poll.
    Opening {
        backend: Arc<dyn AzureBackend>,
        request: AzureRequest,
    },
    /// Relaying audio.
    Receiving {
        events: AzureEventStream,
        tel: StreamTel,
        carry: Option<u8>,
    },
    /// Terminal.
    Done,
}

type AzureYield = Option<(Result<Vec<u8>, TtsError>, AzureStreamState)>;

/// Pull events until a non-empty aligned chunk or a terminal.
async fn azure_stream_step(
    mut events: AzureEventStream,
    mut tel: StreamTel,
    mut carry: Option<u8>,
) -> AzureYield {
    loop {
        match events.next().await {
            Some(Ok(AzureEvent::Audio(bytes))) => {
                let chunk = align_samples(&mut carry, bytes);
                if !chunk.is_empty() {
                    tel.record(chunk.len());
                    return Some((
                        Ok(chunk),
                        AzureStreamState::Receiving { events, tel, carry },
                    ));
                }
            }
            Some(Ok(AzureEvent::Word { .. })) => {}
            Some(Ok(AzureEvent::End)) => {
                tel.log("Azure/stream");
                return None;
            }
            Some(Err(e)) => return Some((Err(e), AzureStreamState::Done)),
            None => return Some((Err(truncated()), AzureStreamState::Done)),
        }
    }
}

#[async_trait]
impl TtsClient for AzureTTSClient {
    async fn synthesize(&self, text: &str) -> Result<TTSResult, TtsError> {
        let mut events = self.backend.open(self.build_request(text)).await?;
        let mut audio: Vec<u8> = Vec::new();
        let mut timestamps: Vec<WordTimestamp> = Vec::new();
        loop {
            match events.next().await {
                Some(Ok(AzureEvent::Audio(bytes))) => audio.extend_from_slice(&bytes),
                Some(Ok(AzureEvent::Word {
                    text,
                    offset_ticks,
                    duration_ticks,
                })) => {
                    let start_ms = ticks_to_ms(offset_ticks);
                    timestamps.push(WordTimestamp {
                        word: text,
                        start_ms,
                        end_ms: start_ms + ticks_to_ms(duration_ticks),
                    });
                }
                Some(Ok(AzureEvent::End)) => break,
                Some(Err(e)) => return Err(e),
                None => return Err(truncated()),
            }
        }
        // Whole samples only; the player rejects odd-length PCM.
        audio.truncate(audio.len() & !1);
        log_buffered_synth("Azure", &audio, &timestamps);
        Ok(TTSResult { audio, timestamps })
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTtsClient> {
        Some(self)
    }
}

impl StreamingTtsClient for AzureTTSClient {
    fn synthesize_stream(&self, text: &str) -> TtsStream {
        let state = AzureStreamState::Opening {
            backend: Arc::clone(&self.backend),
            request: self.build_request(text),
        };
        futures::stream::unfold(state, |state| async move {
            match state {
                AzureStreamState::Done => None,
                AzureStreamState::Opening { backend, request } => {
                    match backend.open(request).await {
                        Ok(events) => azure_stream_step(events, StreamTel::default(), None).await,
                        Err(e) => Some((Err(e), AzureStreamState::Done)),
                    }
                }
                AzureStreamState::Receiving { events, tel, carry } => {
                    azure_stream_step(events, tel, carry).await
                }
            }
        })
        .boxed()
    }
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// `(key, region)` from `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION`, validated
/// so the SDK's internal `unwrap`s on URI / header construction cannot panic.
///
/// # Errors
/// [`TtsError::Config`] naming the offending variable.
#[cfg_attr(
    not(feature = "azure-tts"),
    allow(
        dead_code,
        reason = "factory calls it only with the SDK compiled in; unit-tested under default features"
    )
)]
pub(crate) fn azure_credentials(
    env: impl Fn(&str) -> Option<String>,
) -> Result<(String, String), TtsError> {
    let key = env("AZURE_SPEECH_KEY")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            TtsError::Config(
                "AZURE_SPEECH_KEY environment variable is required for Azure TTS".to_owned(),
            )
        })?;
    let region = env("AZURE_SPEECH_REGION")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            TtsError::Config(
                "AZURE_SPEECH_REGION environment variable is required for Azure TTS".to_owned(),
            )
        })?;
    if !key.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(TtsError::Config(
            "AZURE_SPEECH_KEY contains whitespace or control characters; \
             check .env for stray spaces or line endings"
                .to_owned(),
        ));
    }
    if !region.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(TtsError::Config(format!(
            "AZURE_SPEECH_REGION '{}' is not an Azure region name \
             (letters and digits only, e.g. eastus)",
            region.escape_debug()
        )));
    }
    Ok((key, region))
}

// ---------------------------------------------------------------------------
// SDK adapter (feature `azure-tts`)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "azure-tts"))]
use sdk::{PreparedSsml, map_sdk_event, open_at, sdk_audio_format};

#[cfg(feature = "azure-tts")]
mod sdk {
    use std::time::Duration;

    use async_trait::async_trait;
    use azure_speech::synthesizer::message::{BoundaryType, Metadata};
    use azure_speech::synthesizer::ssml::ToSSML;
    use azure_speech::synthesizer::{self, AudioFormat, Event, Language, Voice};
    use futures::stream::StreamExt as _;

    use super::{AzureBackend, AzureEvent, AzureEventStream, AzureRequest, TtsError};

    /// Bound on the graceful WS close after a synthesis (never wedge the drain).
    const DISCONNECT_TIMEOUT: Duration = Duration::from_secs(2);

    /// Pre-built `<speak>` document handed to the SDK verbatim (the SDK's own
    /// `&str` path only accepts its closed `Voice` enum).
    #[derive(Debug)]
    pub struct PreparedSsml(pub String);

    impl ToSSML for PreparedSsml {
        fn to_ssml(&self, _language: Language, _voice: Voice) -> azure_speech::Result<String> {
            Ok(self.0.clone())
        }
    }

    /// SDK format enum for [`super::AZURE_OUTPUT_FORMAT`].
    pub const fn sdk_audio_format() -> AudioFormat {
        AudioFormat::Raw48Khz16BitMonoPcm
    }

    /// SDK event → zero or more seam events. Only `Word` boundaries surface.
    pub fn map_sdk_event(event: azure_speech::Result<Event>) -> Vec<Result<AzureEvent, TtsError>> {
        match event {
            Ok(Event::Synthesising(_, audio)) => vec![Ok(AzureEvent::Audio(audio))],
            Ok(Event::SessionEnded(_)) => vec![Ok(AzureEvent::End)],
            Ok(Event::AudioMetadata(_, metadata)) => metadata
                .into_iter()
                .filter_map(|m| match m {
                    Metadata::WordBoundary {
                        offset,
                        duration,
                        text,
                    } if text.boundary_type == BoundaryType::Word => Some(Ok(AzureEvent::Word {
                        text: text.text,
                        offset_ticks: offset,
                        duration_ticks: duration,
                    })),
                    _ => None,
                })
                .collect(),
            Ok(Event::SessionStarted(_) | Event::Synthesised(_)) => Vec::new(),
            Err(e) => vec![Err(TtsError::Runtime(format!("Azure TTS error: {e}")))],
        }
    }

    /// Live backend: one SDK connection per synthesis.
    pub(super) struct SdkAzureBackend {
        pub(super) subscription_key: String,
    }

    #[async_trait]
    impl AzureBackend for SdkAzureBackend {
        async fn open(&self, request: AzureRequest) -> Result<AzureEventStream, TtsError> {
            let url = request.endpoint();
            open_at(&url, &self.subscription_key, request).await
        }
    }

    /// WS handshake for `url`, built fallibly. Mirrors the SDK's `connect`
    /// (subscription-key + connection-id headers), whose `unwrap`s would panic
    /// on a malformed region or key.
    fn client_builder(
        url: &str,
        subscription_key: &str,
    ) -> Result<tokio_websockets::ClientBuilder<'static>, TtsError> {
        let bad = |what: &str, e: &dyn std::fmt::Display| {
            TtsError::Config(format!("Azure WS {what} invalid: {e}"))
        };
        let key = http::HeaderValue::from_str(subscription_key)
            .map_err(|e| bad("subscription key", &e))?;
        let connection_id = http::HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
            .map_err(|e| bad("connection id", &e))?;
        tokio_websockets::ClientBuilder::new()
            .uri(url)
            .map_err(|e| bad("endpoint", &e))?
            .add_header(
                http::HeaderName::from_static("ocp-apim-subscription-key"),
                key,
            )
            .and_then(|b| {
                b.add_header(
                    http::HeaderName::from_static("x-connectionid"),
                    connection_id,
                )
            })
            .map_err(|e| bad("header", &e))
    }

    /// Connect to `url`, send `request`, relay its events.
    pub(super) async fn open_at(
        url: &str,
        subscription_key: &str,
        request: AzureRequest,
    ) -> Result<AzureEventStream, TtsError> {
        let builder = client_builder(url, subscription_key)?;
        let base = azure_speech::connector::Client::connect(builder)
            .await
            .map_err(|e| TtsError::Transport(format!("Azure WS connect failed ({url}): {e}")))?;
        let config = synthesizer::Config::new()
            .with_audio_format(sdk_audio_format())
            .enable_word_boundary();
        let client = synthesizer::Client::new(base, config);
        let events = client
            .synthesize(PreparedSsml(request.ssml))
            .await
            .map_err(|e| TtsError::Transport(format!("Azure synthesis request failed: {e}")))?;
        // The SDK fans frames out over a bounded broadcast channel and silently
        // drops lagged ones, so drain it eagerly into an unbounded queue
        // decoupled from the player's pace.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut events = Box::pin(events);
            'drain: loop {
                tokio::select! {
                    // Consumer dropped (barge-in): stop and close.
                    () = tx.closed() => break,
                    next = events.next() => {
                        let Some(event) = next else { break };
                        for item in map_sdk_event(event) {
                            let terminal = !matches!(
                                item,
                                Ok(AzureEvent::Audio(_) | AzureEvent::Word { .. })
                            );
                            if tx.send(item).is_err() || terminal {
                                break 'drain;
                            }
                        }
                    }
                }
            }
            let _ = tokio::time::timeout(DISCONNECT_TIMEOUT, client.disconnect()).await;
        });
        Ok(futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed())
    }
}

#[cfg(test)]
mod tests;
