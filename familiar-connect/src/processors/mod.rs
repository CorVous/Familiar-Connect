//! Bus processors: reply loops (subsystem 06) + watermark-driven memory
//! projectors / background workers (subsystem 07). Python `processors/`.
//!
//! This module root also owns the shared **seam traits** and **event payload
//! types** the two responders consume. Python passed a plain `dict` payload and
//! duck-typed collaborators (an assembler, a `send_text` callable, a
//! `FocusManager`, an `ActivityEngine`); Rust replaces each monkeypatch seam
//! with a trait object (DESIGN §4.8) and each `dict` payload with a typed
//! struct the future producers (subsystem 10) and the tests construct.

// subsystem 06 — reply loops + projector registry + debug logger
pub mod debug_logger;
pub mod history_writer;
pub mod projectors;
pub mod text_responder;
pub mod voice_responder;

// subsystem 07 — the six memory-projector workers
pub mod fact_embedding_worker;
pub mod fact_extractor;
pub mod fact_supersede_worker;
pub mod people_dossier_worker;
pub mod reflection_worker;
pub mod summary_worker;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::identity::Author;
use crate::llm::LlmClient;
use crate::tools::registry::ToolContext;

// ---------------------------------------------------------------------------
// Event payloads (typed replacements for Python's `dict` payloads)
// ---------------------------------------------------------------------------

/// `discord.text` payload (producer: the Discord text source, subsystem 10).
///
/// Field types encode the Python defensive `.get()` / `isinstance` checks: a
/// non-`DiscordTextPayload` payload fails to downcast (the "not a dict" drop
/// rule) and `channel_id` is statically an `i64` (Python's "channel_id not int"
/// drop rule becomes unrepresentable — see the port summary).
#[derive(Clone, Debug, Default)]
pub struct DiscordTextPayload {
    /// The responder filters on `familiar_id == self.familiar_id`.
    pub familiar_id: String,
    /// Required channel id.
    pub channel_id: i64,
    /// Discord guild id, when known.
    pub guild_id: Option<i64>,
    /// Message author (`None` for wake events / anonymous).
    pub author: Option<Author>,
    /// Message body (`""` allowed only with `wake`).
    pub content: String,
    /// Platform-native (Discord snowflake) message id.
    pub message_id: Option<String>,
    /// The message this one replies to, when threaded.
    pub reply_to_message_id: Option<String>,
    /// Resolved mentioned authors.
    pub mentions: Vec<Author>,
    /// `img_N` → URL map threaded into the tool context.
    pub images: HashMap<String, String>,
    /// Whether the incoming message pinged the bot.
    pub pings_bot: bool,
    /// Synthetic unread-nudge wake (no real user content).
    pub wake: bool,
}

/// `voice.activity.start` payload — the responder reads only `user_id`.
#[derive(Clone, Debug, Default)]
pub struct VoiceActivityStart {
    /// Speaker Discord user id, when the source knows it.
    pub user_id: Option<i64>,
}

/// `voice.transcript.final` payload — the responder reads only `text` +
/// `user_id`.
#[derive(Clone, Debug, Default)]
pub struct VoiceTranscriptFinal {
    /// Finalized transcript text.
    pub text: String,
    /// Speaker Discord user id, when known.
    pub user_id: Option<i64>,
}

// ---------------------------------------------------------------------------
// Seam traits
// ---------------------------------------------------------------------------

/// The LLM client the responders type against.
///
/// Extends [`LlmClient`] with the `image_tools_enabled` flag the text
/// responder consults to gate the agentic loop (Python read it via
/// `getattr(llm, "image_tools_enabled", False)`; DESIGN calls for a
/// default-false trait method). Voice never consults it. The default keeps a
/// bare [`LlmClient`] stub a two-line implementation.
pub trait ResponderLlm: LlmClient {
    /// Whether image-tool calling is enabled for this slot (gates the text
    /// agentic loop even when `tool_calling_enabled` is false).
    fn image_tools_enabled(&self) -> bool {
        false
    }
}

/// Post a text reply. `(channel_id, content, reply_to_message_id,
/// mention_user_ids) -> platform message id`.
///
/// A delivery fault is an `Err` (the responder logs `send_error` and bails
/// without persisting the assistant turn, mirroring the Python `try/except`
/// around `send_text`).
#[async_trait]
pub trait SendText: Send + Sync {
    /// Deliver `content` to `channel_id`; returns the platform message id.
    ///
    /// # Errors
    /// A transport/delivery failure the responder should surface but not crash on.
    async fn send(
        &self,
        channel_id: i64,
        content: &str,
        reply_to_message_id: Option<&str>,
        mention_user_ids: &[i64],
    ) -> anyhow::Result<Option<String>>;
}

/// Factory for the Discord "Bot is typing…" indicator (opened lazily mid-stream).
#[async_trait]
pub trait TriggerTyping: Send + Sync {
    /// Open the typing indicator for `channel_id`; the returned guard's
    /// [`close`](TypingIndicator::close) is awaited on every stream exit path.
    async fn open(&self, channel_id: i64) -> Box<dyn TypingIndicator>;
}

/// An open typing-indicator session (Rust has no async `Drop`, so exit is an
/// explicit `close`).
#[async_trait]
pub trait TypingIndicator: Send + Sync {
    /// Close the indicator (idempotent from the responder's perspective — it
    /// is called exactly once per open).
    async fn close(&self);
}

/// What the absence gate decided for a turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GateAction {
    /// Away from the screen — stage the user turn and stop.
    Suppress,
    /// Reply normally, but inject the engine's `state_line` this turn.
    Judgment,
    /// Reply normally; prompt untouched.
    #[default]
    Normal,
}

/// The absence gate's decision for one turn.
#[derive(Clone, Debug, Default)]
pub struct GateDecision {
    /// The action to take.
    pub action: GateAction,
    /// Judgment-turn state line appended (deepest) to the trailing reminder.
    pub state_line: Option<String>,
}

/// The absence controller seam (subsystem 11 `ActivityEngine`). `None` on the
/// responder means "feature off, zero behavior change".
#[async_trait]
pub trait ActivityGate: Send + Sync {
    /// Feed the quiet-clock — called on every handled event.
    fn note_traffic(&self);
    /// Decide how to gate this turn.
    fn gate(&self, payload: &DiscordTextPayload) -> GateDecision;
    /// A suppressed turn that pinged the bot (live missed-ping capture).
    fn note_missed_ping(&self, turn_id: i64);
    /// A judgment-turn reply actually shipped (came back early).
    async fn notify_reply_sent(&self);
    /// End of turn — applies any tool-deferred activity start.
    async fn end_turn(&self);
}

/// The attentional focus controller seam (subsystem 05 `FocusManager`).
#[async_trait]
pub trait FocusManagerApi: Send + Sync {
    /// Whether `channel_id` is the active text or voice focus.
    fn is_focused(&self, channel_id: i64) -> bool;
    /// Whether a non-focused arrival warrants a nudge (debounced).
    fn should_wake(&self, channel_id: i64) -> bool;
    /// The current focus channel for `modality` (`"text"` → text pointer).
    fn get_focus(&self, modality: &str) -> Option<i64>;
    /// Record a nudge timestamp to start the debounce window.
    fn mark_nudge_pending(&self);
    /// The channel-name map (for `#name` rendering in the final reminder).
    fn channel_names(&self) -> HashMap<i64, String> {
        HashMap::new()
    }
    /// Server name for a channel; `None` for `None` / unknown.
    fn guild_name_for(&self, channel_id: Option<i64>) -> Option<String>;
    /// Render a channel as `#name(id)` / `#id` (`"none"` for `None`).
    fn channel_label(&self, channel_id: Option<i64>) -> String;
    /// Responder end-of-turn hook.
    async fn end_turn(&self);
}

#[async_trait]
impl FocusManagerApi for crate::focus::FocusManager {
    fn is_focused(&self, channel_id: i64) -> bool {
        Self::is_focused(self, channel_id)
    }
    fn should_wake(&self, channel_id: i64) -> bool {
        Self::should_wake(self, channel_id)
    }
    fn get_focus(&self, modality: &str) -> Option<i64> {
        Self::get_focus(self, modality)
    }
    fn mark_nudge_pending(&self) {
        Self::mark_nudge_pending(self);
    }
    // NOTE: `channel_names` falls back to the trait default (empty map). The
    // real `FocusManager` keeps its name map private with no getter; a
    // `pub fn channel_names(&self) -> HashMap<i64, String>` getter is requested
    // as a shared-file change so this delegation can be completed. In the only
    // test that wires a real `FocusManager` (immediate shift_focus) no channel
    // names are set, so the empty map is observationally correct there.
    fn guild_name_for(&self, channel_id: Option<i64>) -> Option<String> {
        Self::guild_name_for(self, channel_id)
    }
    fn channel_label(&self, channel_id: Option<i64>) -> String {
        Self::channel_label(self, channel_id)
    }
    async fn end_turn(&self) {
        Self::end_turn(self).await;
    }
}

/// Voice member resolver: `(channel_id, user_id) -> Author | None` (sync).
pub type MemberResolver = Arc<dyn Fn(i64, i64) -> Option<Author> + Send + Sync>;

/// Per-turn tool-context factory: `(channel_id, turn_id, images) -> ToolContext`.
pub type ToolContextFactory =
    Arc<dyn Fn(i64, &str, HashMap<String, String>) -> ToolContext + Send + Sync>;
