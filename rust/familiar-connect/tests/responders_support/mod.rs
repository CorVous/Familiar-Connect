//! Shared test doubles + fixtures for the responder integration tests
//! (subsystem 06). Each `responders_*.rs` file pulls this in via
//! `#[path = "responders_support/mod.rs"] mod support;`.

#![allow(
    dead_code,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::option_if_let_else,
    clippy::significant_drop_tightening
)]

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::{Value, json};

use familiar_connect::bus::envelope::{Event, payload as wrap_payload};
use familiar_connect::bus::in_process::Subscription;
use familiar_connect::bus::protocols::{BackpressurePolicy, EventBus};
use familiar_connect::bus::topics::{
    TOPIC_DISCORD_TEXT, TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL,
};
use familiar_connect::context::{Assembler, CharacterCardLayer, RecentHistoryLayer};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::HistoryStore;
use familiar_connect::identity::Author;
use familiar_connect::llm::{LlmClient, LlmDelta, Message};
use familiar_connect::processors::{
    ActivityGate, DiscordTextPayload, FocusManagerApi, GateDecision, ResponderLlm, SendText,
    ToolContextFactory, TriggerTyping, TypingIndicator, VoiceActivityStart, VoiceTranscriptFinal,
};
use familiar_connect::tools::registry::ToolContext;

// ---------------------------------------------------------------------------
// Store / assembler fixtures
// ---------------------------------------------------------------------------

/// A fresh in-memory async store.
pub fn store() -> Arc<AsyncHistoryStore> {
    Arc::new(AsyncHistoryStore::new(
        HistoryStore::open(":memory:").expect("open :memory:"),
    ))
}

/// A persisted temp character-card file (content is irrelevant to the tests; a
/// missing card would also assemble to an empty system prompt).
pub fn make_card() -> PathBuf {
    let mut f = tempfile::Builder::new()
        .suffix(".md")
        .tempfile()
        .expect("tempfile");
    writeln!(f, "You are a familiar.").expect("write card");
    let (_file, path) = f.keep().expect("persist card");
    path
}

/// An assembler with a character-card layer + a recent-history slot (window 20).
pub fn make_assembler(store: Arc<AsyncHistoryStore>) -> Arc<Assembler> {
    let card = make_card();
    Arc::new(
        Assembler::builder()
            .layer(Arc::new(CharacterCardLayer::new(card)))
            .recent_history(RecentHistoryLayer::builder(store).window_size(20).build())
            .build(),
    )
}

// ---------------------------------------------------------------------------
// Event builders
// ---------------------------------------------------------------------------

/// A `discord.text` event with the given author + channel.
pub fn discord_text_event(payload: DiscordTextPayload, event_id: &str) -> Event {
    let channel_id = payload.channel_id;
    Event {
        event_id: event_id.to_owned(),
        turn_id: event_id.to_owned(),
        session_id: format!("discord:{channel_id}"),
        parent_event_ids: Vec::new(),
        topic: TOPIC_DISCORD_TEXT.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: 1,
        payload: wrap_payload(payload),
    }
}

/// The default alice author used across text tests.
pub fn alice() -> Author {
    Author::new(
        "discord",
        "1",
        Some("alice".to_owned()),
        Some("Alice".to_owned()),
    )
}

/// A basic text payload for `fam` on `channel_id`.
pub fn text_payload(channel_id: i64, content: &str) -> DiscordTextPayload {
    DiscordTextPayload {
        familiar_id: "fam".to_owned(),
        channel_id,
        guild_id: Some(99),
        author: Some(alice()),
        content: content.to_owned(),
        ..Default::default()
    }
}

/// A `voice.activity.start` event.
pub fn activity_start(session_id: &str, turn_id: &str, user_id: Option<i64>) -> Event {
    Event {
        event_id: format!("act-{turn_id}"),
        turn_id: turn_id.to_owned(),
        session_id: session_id.to_owned(),
        parent_event_ids: Vec::new(),
        topic: TOPIC_VOICE_ACTIVITY_START.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: 1,
        payload: wrap_payload(VoiceActivityStart { user_id }),
    }
}

/// A `voice.transcript.final` event.
pub fn voice_final(text: &str, session_id: &str, turn_id: &str, user_id: Option<i64>) -> Event {
    Event {
        event_id: format!("final-{turn_id}"),
        turn_id: turn_id.to_owned(),
        session_id: session_id.to_owned(),
        parent_event_ids: Vec::new(),
        topic: TOPIC_VOICE_TRANSCRIPT_FINAL.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: 2,
        payload: wrap_payload(VoiceTranscriptFinal {
            text: text.to_owned(),
            user_id,
        }),
    }
}

// ---------------------------------------------------------------------------
// LlmDelta helpers
// ---------------------------------------------------------------------------

/// A content-only delta.
pub fn text_delta(s: &str) -> LlmDelta {
    LlmDelta {
        content: s.to_owned(),
        ..Default::default()
    }
}

/// A tool-call delta (`index 0`).
pub fn tc_delta(call_id: &str, name: &str, args: Value) -> LlmDelta {
    LlmDelta {
        content: String::new(),
        tool_calls: vec![json!({
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args.to_string()},
        })],
        finish_reason: None,
    }
}

/// A terminal finish-reason delta.
pub fn finish(reason: &str) -> LlmDelta {
    LlmDelta {
        content: String::new(),
        tool_calls: Vec::new(),
        finish_reason: Some(reason.to_owned()),
    }
}

// ---------------------------------------------------------------------------
// ScriptedLlm — bare content stream with an optional per-delta delay
// ---------------------------------------------------------------------------

/// Yields a fixed content-delta sequence (optionally paced by `delay_ms`).
pub struct ScriptedLlm {
    deltas: Vec<String>,
    delay_ms: u64,
    tool_calling: bool,
    image_tools: bool,
}

impl ScriptedLlm {
    pub fn new(deltas: &[&str]) -> Self {
        Self {
            deltas: deltas.iter().map(|s| (*s).to_owned()).collect(),
            delay_ms: 0,
            tool_calling: false,
            image_tools: false,
        }
    }
    pub fn with_delay(deltas: &[&str], delay_ms: u64) -> Self {
        let mut s = Self::new(deltas);
        s.delay_ms = delay_ms;
        s
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", self.deltas.concat()))
    }
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        let deltas = self.deltas.clone();
        let delay = self.delay_ms;
        let s = stream::unfold(deltas.into_iter(), move |mut it| async move {
            let d = it.next()?;
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Some((Ok(text_delta(&d)), it))
        });
        Ok(Box::pin(s))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        self.tool_calling
    }
}
impl ResponderLlm for ScriptedLlm {
    fn image_tools_enabled(&self) -> bool {
        self.image_tools
    }
}

// ---------------------------------------------------------------------------
// CapturingLlm — records every message list, yields a fixed reply
// ---------------------------------------------------------------------------

pub struct CapturingLlm {
    reply: String,
    captured: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl CapturingLlm {
    pub fn new(reply: &str) -> Self {
        Self {
            reply: reply.to_owned(),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }
    pub fn captured(&self) -> Vec<Vec<Message>> {
        self.captured.lock().expect("captured").clone()
    }
}

#[async_trait]
impl LlmClient for CapturingLlm {
    async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message> {
        self.captured.lock().expect("captured").push(messages);
        Ok(Message::new("assistant", self.reply.clone()))
    }
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        self.captured.lock().expect("captured").push(messages);
        let reply = self.reply.clone();
        Ok(Box::pin(stream::once(
            async move { Ok(text_delta(&reply)) },
        )))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}
impl ResponderLlm for CapturingLlm {}

// ---------------------------------------------------------------------------
// ScriptedToolLlm — per-call scripts of LlmDeltas (tool mode)
// ---------------------------------------------------------------------------

pub struct ScriptedToolLlm {
    scripts: Mutex<Vec<Vec<LlmDelta>>>,
    calls: Arc<Mutex<Vec<Vec<Message>>>>,
    tool_calling: bool,
    image_tools: bool,
}

impl ScriptedToolLlm {
    pub fn new(scripts: Vec<Vec<LlmDelta>>) -> Self {
        Self {
            scripts: Mutex::new(scripts),
            calls: Arc::new(Mutex::new(Vec::new())),
            tool_calling: true,
            image_tools: false,
        }
    }
    pub const fn with_flags(mut self, tool_calling: bool, image_tools: bool) -> Self {
        self.tool_calling = tool_calling;
        self.image_tools = image_tools;
        self
    }
    pub fn call_count(&self) -> usize {
        self.calls.lock().expect("calls").len()
    }
    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().expect("calls").clone()
    }
}

#[async_trait]
impl LlmClient for ScriptedToolLlm {
    async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "x"))
    }
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        self.calls.lock().expect("calls").push(messages);
        let script = {
            let mut scripts = self.scripts.lock().expect("scripts");
            if scripts.is_empty() {
                Vec::new()
            } else {
                scripts.remove(0)
            }
        };
        Ok(Box::pin(stream::iter(script.into_iter().map(Ok))))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        self.tool_calling
    }
}
impl ResponderLlm for ScriptedToolLlm {
    fn image_tools_enabled(&self) -> bool {
        self.image_tools
    }
}

// ---------------------------------------------------------------------------
// CapturingSend
// ---------------------------------------------------------------------------

/// Records `send_text` invocations; returns a configurable platform id.
pub struct CapturingSend {
    calls: Mutex<Vec<(i64, String, Option<String>, Vec<i64>)>>,
    returned_id: Option<String>,
}

impl CapturingSend {
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            returned_id: Some("bot-msg-1".to_owned()),
        }
    }
    pub fn with_id(id: &str) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            returned_id: Some(id.to_owned()),
        }
    }
    pub fn calls(&self) -> Vec<(i64, String, Option<String>, Vec<i64>)> {
        self.calls.lock().expect("send calls").clone()
    }
}

impl Default for CapturingSend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SendText for CapturingSend {
    async fn send(
        &self,
        channel_id: i64,
        content: &str,
        reply_to_message_id: Option<&str>,
        mention_user_ids: &[i64],
    ) -> anyhow::Result<Option<String>> {
        self.calls.lock().expect("send calls").push((
            channel_id,
            content.to_owned(),
            reply_to_message_id.map(str::to_owned),
            mention_user_ids.to_vec(),
        ));
        Ok(self.returned_id.clone())
    }
}

// ---------------------------------------------------------------------------
// RecordingTyping + RecordingIndicator
// ---------------------------------------------------------------------------

struct RecordingTypingInner {
    calls: Mutex<Vec<i64>>,
    entered: AtomicUsize,
    exited: AtomicUsize,
}

/// A typing-indicator factory recording `(channel_id, entered, exited)`.
pub struct RecordingTyping {
    inner: Arc<RecordingTypingInner>,
}

impl RecordingTyping {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RecordingTypingInner {
                calls: Mutex::new(Vec::new()),
                entered: AtomicUsize::new(0),
                exited: AtomicUsize::new(0),
            }),
        }
    }
    pub fn calls(&self) -> Vec<i64> {
        self.inner.calls.lock().expect("typing calls").clone()
    }
    pub fn entered(&self) -> usize {
        self.inner.entered.load(Ordering::SeqCst)
    }
    pub fn exited(&self) -> usize {
        self.inner.exited.load(Ordering::SeqCst)
    }
}

impl Default for RecordingTyping {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TriggerTyping for RecordingTyping {
    async fn open(&self, channel_id: i64) -> Box<dyn TypingIndicator> {
        self.inner
            .calls
            .lock()
            .expect("typing calls")
            .push(channel_id);
        self.inner.entered.fetch_add(1, Ordering::SeqCst);
        Box::new(RecordingIndicator {
            inner: Arc::clone(&self.inner),
        })
    }
}

struct RecordingIndicator {
    inner: Arc<RecordingTypingInner>,
}

#[async_trait]
impl TypingIndicator for RecordingIndicator {
    async fn close(&self) {
        self.inner.exited.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// FakeActivityEngine
// ---------------------------------------------------------------------------

/// Records gate/notify/end_turn calls; returns a fixed decision.
pub struct FakeActivityEngine {
    decision: GateDecision,
    replies_notified: AtomicUsize,
    turns_ended: AtomicUsize,
    missed_pings: Mutex<Vec<i64>>,
    traffic_notes: AtomicUsize,
}

impl FakeActivityEngine {
    pub const fn new(decision: GateDecision) -> Self {
        Self {
            decision,
            replies_notified: AtomicUsize::new(0),
            turns_ended: AtomicUsize::new(0),
            missed_pings: Mutex::new(Vec::new()),
            traffic_notes: AtomicUsize::new(0),
        }
    }
    pub fn replies_notified(&self) -> usize {
        self.replies_notified.load(Ordering::SeqCst)
    }
    pub fn turns_ended(&self) -> usize {
        self.turns_ended.load(Ordering::SeqCst)
    }
    pub fn missed_pings(&self) -> Vec<i64> {
        self.missed_pings.lock().expect("missed").clone()
    }
    pub fn traffic_notes(&self) -> usize {
        self.traffic_notes.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ActivityGate for FakeActivityEngine {
    fn note_traffic(&self) {
        self.traffic_notes.fetch_add(1, Ordering::SeqCst);
    }
    fn gate(&self, _payload: &DiscordTextPayload) -> GateDecision {
        self.decision.clone()
    }
    fn note_missed_ping(&self, turn_id: i64) {
        self.missed_pings.lock().expect("missed").push(turn_id);
    }
    async fn notify_reply_sent(&self) {
        self.replies_notified.fetch_add(1, Ordering::SeqCst);
    }
    async fn end_turn(&self) {
        self.turns_ended.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// TestFocusManager — scriptable focus double
// ---------------------------------------------------------------------------

/// A scriptable [`FocusManagerApi`] double.
pub struct TestFocusManager {
    is_focused: bool,
    text_focus: Option<i64>,
    should_wake: bool,
    channel_names: HashMap<i64, String>,
    guild_names: HashMap<i64, String>,
    end_turn_count: AtomicUsize,
    nudge_count: AtomicUsize,
}

impl TestFocusManager {
    pub fn focused(channel_id: i64) -> Self {
        Self {
            is_focused: true,
            text_focus: Some(channel_id),
            should_wake: false,
            channel_names: HashMap::new(),
            guild_names: HashMap::new(),
            end_turn_count: AtomicUsize::new(0),
            nudge_count: AtomicUsize::new(0),
        }
    }
    pub fn unfocused() -> Self {
        Self {
            is_focused: false,
            text_focus: Some(999),
            should_wake: false,
            channel_names: HashMap::new(),
            guild_names: HashMap::new(),
            end_turn_count: AtomicUsize::new(0),
            nudge_count: AtomicUsize::new(0),
        }
    }
    pub const fn with_should_wake(mut self, v: bool) -> Self {
        self.should_wake = v;
        self
    }
    pub const fn with_text_focus(mut self, v: Option<i64>) -> Self {
        self.text_focus = v;
        self
    }
    pub fn with_channel_name(mut self, cid: i64, name: &str) -> Self {
        self.channel_names.insert(cid, name.to_owned());
        self
    }
    pub fn with_guild_name(mut self, cid: i64, name: &str) -> Self {
        self.guild_names.insert(cid, name.to_owned());
        self
    }
    pub fn end_turn_count(&self) -> usize {
        self.end_turn_count.load(Ordering::SeqCst)
    }
    pub fn nudge_count(&self) -> usize {
        self.nudge_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl FocusManagerApi for TestFocusManager {
    fn is_focused(&self, _channel_id: i64) -> bool {
        self.is_focused
    }
    fn should_wake(&self, _channel_id: i64) -> bool {
        self.should_wake
    }
    fn get_focus(&self, modality: &str) -> Option<i64> {
        if modality == "text" {
            self.text_focus
        } else {
            None
        }
    }
    fn mark_nudge_pending(&self) {
        self.nudge_count.fetch_add(1, Ordering::SeqCst);
    }
    fn channel_names(&self) -> HashMap<i64, String> {
        self.channel_names.clone()
    }
    fn guild_name_for(&self, channel_id: Option<i64>) -> Option<String> {
        channel_id.and_then(|c| self.guild_names.get(&c).cloned())
    }
    fn channel_label(&self, channel_id: Option<i64>) -> String {
        match channel_id {
            Some(c) => match self.channel_names.get(&c) {
                Some(n) => format!("#{n}({c})"),
                None => format!("#{c}"),
            },
            None => "none".to_owned(),
        }
    }
    async fn end_turn(&self) {
        self.end_turn_count.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// RecordingBus — captures publish, never delivers
// ---------------------------------------------------------------------------

/// An [`EventBus`] double that records published events (for nudge tests).
pub struct RecordingBus {
    published: Mutex<Vec<Event>>,
}

impl RecordingBus {
    pub const fn new() -> Self {
        Self {
            published: Mutex::new(Vec::new()),
        }
    }
    pub fn published(&self) -> Vec<Event> {
        self.published.lock().expect("published").clone()
    }
}

impl Default for RecordingBus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EventBus for RecordingBus {
    async fn start(&self) {}
    async fn shutdown(&self) {}
    async fn publish(&self, event: Event) {
        self.published.lock().expect("published").push(event);
    }
    fn subscribe(
        &self,
        _topics: &[&str],
        _policy: BackpressurePolicy,
        _maxsize: usize,
    ) -> Subscription {
        Subscription::closed()
    }
}

// ---------------------------------------------------------------------------
// Tool-context factory
// ---------------------------------------------------------------------------

/// A minimal tool-context factory (`fam`, text kind, images threaded).
pub fn simple_ctx_factory() -> ToolContextFactory {
    Arc::new(|channel_id, turn_id, images| {
        ToolContext::new("fam", channel_id, "text", turn_id).with_images(images)
    })
}

// ---------------------------------------------------------------------------
// Log capture
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct VecWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buf").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for VecWriter {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A thread-local tracing capture; hold it for the duration of the assertion.
pub struct LogCapture {
    buf: Arc<Mutex<Vec<u8>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

impl LogCapture {
    /// Install a thread-local capturing subscriber.
    pub fn install() -> Self {
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(VecWriter(buf.clone()))
            .with_ansi(true)
            // The module target `…voice_responder` contains the substring
            // "respond"; drop it so decision-line filters aren't polluted.
            .with_target(false)
            .with_max_level(tracing::Level::TRACE)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        Self { buf, _guard: guard }
    }
    /// The captured text so far.
    pub fn contents(&self) -> String {
        String::from_utf8_lossy(&self.buf.lock().expect("log buf")).into_owned()
    }
    /// Count of lines containing `needle`.
    pub fn count_lines(&self, needle: &str) -> usize {
        self.contents()
            .lines()
            .filter(|l| l.contains(needle))
            .count()
    }
}
