//! Text reply orchestrator (subsystem 06; Python `processors/text_responder.py`).
//!
//! Consumes `discord.text` events, assembles a layered prompt (05), streams an
//! LLM reply (08), gates it through the `<silent>` sentinel, rewrites its
//! ping/thread markers, and delivers it via the injected [`SendText`] callback,
//! persisting user + assistant turns to history (03). Everything runs inline
//! under a per-turn [`TurnScope`] so a typing-cancel or a newer event's
//! `begin_turn` supersedes in-flight work cooperatively.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use regex::{Captures, Regex};
use std::sync::LazyLock;
use uuid::Uuid;

use crate::bus::envelope::{Event, TurnScope, payload as wrap_payload};
use crate::bus::protocols::EventBus;
use crate::bus::router::TurnRouter;
use crate::bus::topics::TOPIC_DISCORD_TEXT;
use crate::context::assembler::{Assembler, AssemblyContext};
use crate::context::final_reminder::FinalReminder;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::AppendTurn;
use crate::identity::Author;
use crate::llm::{LlmClient, LlmDelta, Message};
use crate::log_style as ls;
use crate::processors::{
    ActivityGate, DiscordTextPayload, FocusManagerApi, GateAction, ResponderLlm, SendText,
    ToolContextFactory, TriggerTyping, TypingIndicator,
};
use crate::silence::SilentDetector;
use crate::tools::agentic::{
    AgenticHooks, DEFAULT_MAX_ITERATIONS, agentic_loop, tool_content_as_text,
};
use crate::tools::registry::ToolRegistry;
use crate::typing_interrupt::TypingInterruptHandler;

// ---------------------------------------------------------------------------
// LLM output vocabulary regexes + head addendum
// ---------------------------------------------------------------------------

/// `[@DisplayName]` ping markers in the model's output.
static PING_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[@([^\]\n]+)\]").expect("valid ping regex"));

/// Thread-reply markers: `[↩]` / `[reply]`, optionally `[↩ <id>]`.
static THREAD_MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(?:↩|reply)(?:\s+([^\]\n]+))?\]").expect("valid thread regex"));

/// Leaked `[#id]` / `[H:MMpm]`-shaped prefix at the head of a reply.
static LEAKED_META_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:\[[^\]\n]*(?:#\d|\d:\d|[AP]M)[^\]\n]*\]\s*)+")
        .expect("valid leaked-metadata regex")
});

/// The "## Output controls" system-prompt addendum (byte-stable cache prefix).
const BOT_OUTPUT_INSTRUCTIONS: &str = "## Output controls\n\n\
- The `[H:MM Name #id]` prefix on each user message is read-only metadata the \
system adds for you. **Do not** start your replies with that shape — just write \
the message body.\n\
- Ping a user by writing `[@DisplayName]` using a name that appears in recent \
messages. Unrecognised names render as plain text without pinging.\n\
- Optionally prefix your message with `[↩]` to thread it as a reply to the \
message you're responding to. Useful when the channel is busy and it isn't \
obvious who you're addressing. Without `[↩]`, your message posts normally.\n\
- To reply to a *specific* earlier message, write `[↩ <message_id>]` using the \
`#<id>` shown next to that message in recent history. Unknown ids fall back to \
the triggering message.";

// ---------------------------------------------------------------------------
// Pure output-rewrite helpers (unit-tested)
// ---------------------------------------------------------------------------

/// Rewrite `[@DisplayName]` markers to `<@user_id>` for known discord labels;
/// collect the resolved user ids in occurrence order.
///
/// Unknown labels, non-discord canonical keys, and non-integer ids degrade to
/// plain `@DisplayName` text (no ping, no error).
#[must_use]
pub fn rewrite_pings<S: std::hash::BuildHasher>(
    content: &str,
    label_to_key: &HashMap<String, String, S>,
) -> (String, Vec<i64>) {
    let mut resolved: Vec<i64> = Vec::new();
    let rewritten = PING_MARKER_RE
        .replace_all(content, |caps: &Captures| {
            let label = &caps[1];
            label_to_key.get(label).map_or_else(
                || format!("@{label}"),
                |key| {
                    let (platform, user_id) = key.split_once(':').unwrap_or((key.as_str(), ""));
                    if platform != "discord" {
                        return format!("@{label}");
                    }
                    user_id.parse::<i64>().map_or_else(
                        |_| format!("@{label}"),
                        |id| {
                            resolved.push(id);
                            format!("<@{user_id}>")
                        },
                    )
                },
            )
        })
        .into_owned();
    (rewritten, resolved)
}

/// Drop a leaked `[#id]` / `[H:MMpm]`-style prefix from the head (applied once).
#[must_use]
pub fn strip_leaked_metadata_prefix(content: &str) -> String {
    LEAKED_META_PREFIX_RE.replacen(content, 1, "").into_owned()
}

/// Strip thread markers; return `(stripped, wanted_thread, target_id)`.
///
/// Any occurrence anywhere triggers threading; the first non-empty captured id
/// wins (with a leading `#` sigil stripped); the result is left-trimmed.
#[must_use]
pub fn consume_thread_marker(content: &str) -> (String, bool, Option<String>) {
    let mut target_id: Option<String> = None;
    let mut any = false;
    for caps in THREAD_MARKER_RE.captures_iter(content) {
        any = true;
        if target_id.is_none() {
            if let Some(cap) = caps.get(1) {
                let trimmed = cap.as_str().trim();
                if !trimmed.is_empty() {
                    let cleaned = trimmed.trim_start_matches('#').trim();
                    if !cleaned.is_empty() {
                        target_id = Some(cleaned.to_owned());
                    }
                }
            }
        }
    }
    if !any {
        return (content.to_owned(), false, None);
    }
    let stripped = THREAD_MARKER_RE
        .replace_all(content, "")
        .trim_start()
        .to_owned();
    (stripped, true, target_id)
}

fn join_nonempty(parts: &[&str], sep: &str) -> String {
    parts
        .iter()
        .copied()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(sep)
}

// ---------------------------------------------------------------------------
// TextResponder
// ---------------------------------------------------------------------------

/// Streams LLM replies for `discord.text` events; posts via [`SendText`].
pub struct TextResponder {
    assembler: Arc<Assembler>,
    llm: Arc<dyn ResponderLlm>,
    send_text: Arc<dyn SendText>,
    history: Arc<AsyncHistoryStore>,
    router: Arc<TurnRouter>,
    familiar_id: String,
    trigger_typing: Option<Arc<dyn TriggerTyping>>,
    typing_handler: Option<Arc<TypingInterruptHandler>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    tool_context_factory: Option<ToolContextFactory>,
    post_history_instructions: String,
    display_tz: String,
    focus_manager: Option<Arc<dyn FocusManagerApi>>,
    loop_max_iterations: usize,
    activity_engine: Option<Arc<dyn ActivityGate>>,
    seen: Mutex<HashSet<String>>,
}

impl TextResponder {
    /// The processor's human label.
    pub const NAME: &'static str = "text-responder";

    /// Construct a responder with the required collaborators; optional seams
    /// default off (`with_*` setters wire them).
    #[must_use]
    pub fn new(
        assembler: Arc<Assembler>,
        llm: Arc<dyn ResponderLlm>,
        send_text: Arc<dyn SendText>,
        history: Arc<AsyncHistoryStore>,
        router: Arc<TurnRouter>,
        familiar_id: impl Into<String>,
    ) -> Self {
        Self {
            assembler,
            llm,
            send_text,
            history,
            router,
            familiar_id: familiar_id.into(),
            trigger_typing: None,
            typing_handler: None,
            tool_registry: None,
            tool_context_factory: None,
            post_history_instructions: String::new(),
            display_tz: "UTC".to_owned(),
            focus_manager: None,
            loop_max_iterations: 5,
            activity_engine: None,
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Wire the Discord typing indicator.
    #[must_use]
    pub fn with_trigger_typing(mut self, trigger: Arc<dyn TriggerTyping>) -> Self {
        self.trigger_typing = Some(trigger);
        self
    }
    /// Wire the typing-event policy (backoff + user-typing cancel).
    #[must_use]
    pub fn with_typing_handler(mut self, handler: Arc<TypingInterruptHandler>) -> Self {
        self.typing_handler = Some(handler);
        self
    }
    /// Wire the agentic-loop tool registry + context factory.
    #[must_use]
    pub fn with_tools(mut self, registry: Arc<ToolRegistry>, factory: ToolContextFactory) -> Self {
        self.tool_registry = Some(registry);
        self.tool_context_factory = Some(factory);
        self
    }
    /// Set the deepest trailing-reminder etiquette (empty = omitted).
    #[must_use]
    pub fn with_post_history_instructions(mut self, text: impl Into<String>) -> Self {
        self.post_history_instructions = text.into();
        self
    }
    /// Set the trailing-reminder clock timezone.
    #[must_use]
    pub fn with_display_tz(mut self, tz: impl Into<String>) -> Self {
        self.display_tz = tz.into();
        self
    }
    /// Wire the attentional focus controller.
    #[must_use]
    pub fn with_focus_manager(mut self, fm: Arc<dyn FocusManagerApi>) -> Self {
        self.focus_manager = Some(fm);
        self
    }
    /// Set the agentic-loop iteration cap.
    #[must_use]
    pub const fn with_loop_max_iterations(mut self, n: usize) -> Self {
        self.loop_max_iterations = n;
        self
    }
    /// Wire the absence gate.
    #[must_use]
    pub fn with_activity_engine(mut self, engine: Arc<dyn ActivityGate>) -> Self {
        self.activity_engine = Some(engine);
        self
    }

    /// The processor name (`"text-responder"`).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        Self::NAME
    }

    /// The subscribed topics (`discord.text`).
    #[must_use]
    pub const fn topics(&self) -> [&'static str; 1] {
        [TOPIC_DISCORD_TEXT]
    }

    /// Handle one `discord.text` event: the whole turn, inline.
    ///
    /// # Errors
    /// Propagates a store failure on the main persistence path.
    #[allow(clippy::too_many_lines, reason = "one cohesive turn pipeline")]
    pub async fn handle(&self, event: &Event, bus: &dyn EventBus) -> anyhow::Result<()> {
        if event.topic != TOPIC_DISCORD_TEXT {
            return Ok(());
        }
        if self
            .seen
            .lock()
            .expect("text responder seen mutex")
            .contains(&event.event_id)
        {
            return Ok(());
        }
        let Some(payload) = event.payload.downcast_ref::<DiscordTextPayload>() else {
            return Ok(());
        };
        if payload.familiar_id != self.familiar_id {
            return Ok(());
        }
        let channel_id = payload.channel_id;
        let content = payload.content.clone();
        let is_wake = payload.wake;
        if content.is_empty() && !is_wake {
            return Ok(());
        }
        let author = payload.author.clone();
        let guild_id = payload.guild_id;
        let message_id = payload.message_id.clone();
        let reply_to_message_id = payload.reply_to_message_id.clone();
        let mentions = payload.mentions.clone();
        let images = payload.images.clone();
        let pings_bot = payload.pings_bot;

        self.seen
            .lock()
            .expect("text responder seen mutex")
            .insert(event.event_id.clone());

        if let Some(engine) = &self.activity_engine {
            engine.note_traffic();
        }
        if let Some(handler) = &self.typing_handler {
            handler.wait_for_backoff(channel_id).await;
            handler.notify_user_message(channel_id);
        }
        let scope = self.router.begin_turn(&event.session_id, &event.turn_id);

        // Identity upserts (soft "most recently seen" cache writes).
        if let Some(a) = &author {
            self.history.upsert_account(a.clone()).await?;
            if let (Some(g), Some(nick)) = (guild_id, a.guild_nick.clone()) {
                self.history
                    .upsert_guild_nick(a.canonical_key(), g, Some(nick))
                    .await?;
            }
        }
        for m in &mentions {
            self.history.upsert_account(m.clone()).await?;
            if let (Some(g), Some(nick)) = (guild_id, m.guild_nick.clone()) {
                self.history
                    .upsert_guild_nick(m.canonical_key(), g, Some(nick))
                    .await?;
            }
        }

        // Activity gate.
        let gate = self.activity_engine.as_ref().map(|e| e.gate(payload));
        let suppressed = matches!(&gate, Some(g) if g.action == GateAction::Suppress);
        let judgment = matches!(&gate, Some(g) if g.action == GateAction::Judgment);

        let focused = self
            .focus_manager
            .as_ref()
            .is_none_or(|fm| fm.is_focused(channel_id));

        if !is_wake {
            let mut append = AppendTurn::new(&self.familiar_id, channel_id, "user", &content)
                .consumed(focused && !suppressed)
                .pings_bot(pings_bot);
            if let Some(a) = &author {
                append = append.author(a.clone());
            }
            if let Some(g) = guild_id {
                append = append.guild_id(g);
            }
            if let Some(mid) = &message_id {
                append = append.platform_message_id(mid);
            }
            if let Some(rid) = &reply_to_message_id {
                append = append.reply_to_message_id(rid);
            }
            let user_turn = self.history.append_turn(append).await?;
            if !mentions.is_empty() {
                let keys: Vec<String> = mentions.iter().map(Author::canonical_key).collect();
                self.history.record_mentions(user_turn.id, keys).await?;
            }
            if suppressed {
                if pings_bot {
                    if let Some(engine) = &self.activity_engine {
                        engine.note_missed_ping(user_turn.id);
                    }
                }
                let (ch, srv) = self.origin_fields(channel_id);
                let label = author.as_ref().map_or("unknown", author_display);
                tracing::info!(
                    "{} suppressed {}{}{} {}",
                    ls::tag("Activity", ls::G),
                    ch,
                    srv,
                    ls::kv_styled("from", label, ls::W, ls::LW),
                    ls::kv_styled("text", &ls::trunc(&content, 200), ls::W, ls::LW),
                );
                return Ok(());
            }
            if !focused {
                let (ch, srv) = self.origin_fields(channel_id);
                let label = author.as_ref().map_or("unknown", author_display);
                tracing::info!(
                    "{} {}{}{} {}",
                    ls::tag("\u{1f4e5} Staged", ls::Y),
                    ch,
                    srv,
                    ls::kv_styled("from", label, ls::W, ls::LW),
                    ls::kv_styled("text", &content, ls::W, ls::LW),
                );
                if let Some(fm) = &self.focus_manager {
                    if fm.should_wake(channel_id) {
                        self.emit_unread_nudge(bus).await;
                    }
                }
                return Ok(());
            }
        } else if suppressed {
            return Ok(());
        }

        // Seed retrieval before assembly.
        self.assembler.set_rag_cue(&content);
        let label_to_key = self.build_ping_resolver(channel_id, guild_id).await?;

        let activity_state_line = if judgment {
            gate.as_ref().and_then(|g| g.state_line.clone())
        } else {
            None
        };
        let reply = self
            .stream_reply(&scope, channel_id, guild_id, images, activity_state_line)
            .await;

        let Some(reply) = reply else {
            if let Some(engine) = &self.activity_engine {
                engine.end_turn().await;
            }
            return Ok(());
        };
        if scope.is_cancelled() {
            if let Some(engine) = &self.activity_engine {
                engine.end_turn().await;
            }
            return Ok(());
        }
        if reply.trim().is_empty() {
            tracing::warn!(
                "{} {} {}",
                ls::tag("Text", ls::Y),
                ls::kv_styled("skip", "empty_reply", ls::W, ls::LY),
                ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
            );
            if let Some(engine) = &self.activity_engine {
                engine.end_turn().await;
            }
            return Ok(());
        }

        // Output rewriting: thread marker → leaked-metadata strip → pings.
        let (unthreaded, wants_thread, target_id) = consume_thread_marker(&reply);
        let unthreaded = strip_leaked_metadata_prefix(&unthreaded);
        let (rewritten, mention_user_ids) = rewrite_pings(&unthreaded, &label_to_key);
        let thread_target: Option<String> = if wants_thread {
            if let Some(tid) = &target_id {
                let found = self
                    .history
                    .lookup_turn_by_platform_message_id(self.familiar_id.clone(), tid.clone())
                    .await?;
                if found.is_some() {
                    Some(tid.clone())
                } else {
                    message_id.clone()
                }
            } else {
                message_id.clone()
            }
        } else {
            None
        };

        // A mid-turn shift_focus already moved the pointer — reply follows focus.
        let mut send_channel_id = channel_id;
        if let Some(fm) = &self.focus_manager {
            if let Some(focus) = fm.get_focus("text") {
                send_channel_id = focus;
            }
        }

        let sent_message_id = match self
            .send_text
            .send(
                send_channel_id,
                &rewritten,
                thread_target.as_deref(),
                &mention_user_ids,
            )
            .await
        {
            Ok(id) => id,
            Err(exc) => {
                tracing::warn!(
                    "{} {}",
                    ls::tag("Text", ls::R),
                    ls::kv_styled("send_error", &format!("{exc:?}"), ls::W, ls::R),
                );
                if let Some(engine) = &self.activity_engine {
                    engine.end_turn().await;
                }
                return Ok(());
            }
        };
        let (ch, srv) = self.origin_fields(channel_id);
        tracing::info!(
            "{} {}{}{} {} {} {}",
            ls::tag("\u{1f4ac} Text", ls::G),
            ch,
            srv,
            ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
            ls::kv_styled(
                "chars",
                &rewritten.chars().count().to_string(),
                ls::W,
                ls::LW
            ),
            ls::kv_styled(
                "thread",
                if wants_thread { "1" } else { "0" },
                ls::W,
                ls::LB
            ),
            ls::kv_styled("text", &ls::trunc(&rewritten, 200), ls::W, ls::LW),
        );

        if scope.is_cancelled() {
            return Ok(());
        }

        let mut append =
            AppendTurn::new(&self.familiar_id, send_channel_id, "assistant", &rewritten);
        if let Some(g) = guild_id {
            append = append.guild_id(g);
        }
        if let Some(id) = &sent_message_id {
            append = append.platform_message_id(id);
        }
        if let Some(tt) = &thread_target {
            append = append.reply_to_message_id(tt);
        }
        self.history.append_turn(append).await?;
        self.router.end_turn(&scope);
        if let Some(fm) = &self.focus_manager {
            fm.end_turn().await;
        }
        if let Some(engine) = &self.activity_engine {
            if judgment {
                engine.notify_reply_sent().await;
            }
            engine.end_turn().await;
        }
        Ok(())
    }

    fn origin_fields(&self, channel_id: i64) -> (String, String) {
        let (ch_label, guild) = self.focus_manager.as_ref().map_or_else(
            || (format!("#{channel_id}"), None),
            |fm| {
                (
                    fm.channel_label(Some(channel_id)),
                    fm.guild_name_for(Some(channel_id)),
                )
            },
        );
        let ch = format!("{} ", ls::kv_styled("ch", &ch_label, ls::W, ls::LW));
        let srv = match guild {
            Some(g) if !g.is_empty() => format!("{} ", ls::kv_styled("srv", &g, ls::W, ls::LM)),
            _ => String::new(),
        };
        (ch, srv)
    }

    async fn emit_unread_nudge(&self, bus: &dyn EventBus) {
        let Some(fm) = &self.focus_manager else {
            return;
        };
        let Some(focus_ch) = fm.get_focus("text") else {
            return;
        };
        fm.mark_nudge_pending();
        let synth_id = Uuid::new_v4().simple().to_string();
        let turn_id = format!("unread-wake-{synth_id}");
        let payload = DiscordTextPayload {
            familiar_id: self.familiar_id.clone(),
            channel_id: focus_ch,
            content: "[unread messages waiting elsewhere]".to_owned(),
            author: None,
            wake: true,
            ..Default::default()
        };
        let event = Event {
            event_id: synth_id,
            turn_id,
            session_id: focus_ch.to_string(),
            parent_event_ids: Vec::new(),
            topic: TOPIC_DISCORD_TEXT.to_owned(),
            timestamp: chrono::Utc::now(),
            sequence_number: 0,
            payload: wrap_payload(payload),
        };
        bus.publish(event).await;
        tracing::info!(
            "{} {}",
            ls::tag("\u{23f0} Nudge", ls::LC),
            ls::kv_styled("focus", &fm.channel_label(Some(focus_ch)), ls::W, ls::LW),
        );
    }

    async fn build_ping_resolver(
        &self,
        channel_id: i64,
        guild_id: Option<i64>,
    ) -> anyhow::Result<HashMap<String, String>> {
        let authors = self
            .history
            .recent_distinct_authors(self.familiar_id.clone(), channel_id, 20)
            .await?;
        let mut label_to_key: HashMap<String, String> = HashMap::new();
        for a in authors {
            let key = a.canonical_key();
            let label = self
                .history
                .resolve_label(key.clone(), guild_id, Some(self.familiar_id.clone()))
                .await?;
            if let Some(existing) = label_to_key.get(&label) {
                if *existing != key {
                    tracing::warn!(
                        "{} {} {}",
                        ls::tag("Text", ls::Y),
                        ls::kv_styled("ambiguous_label", &label, ls::W, ls::LY),
                        ls::kv_styled("keys", &format!("{existing},{key}"), ls::W, ls::LW),
                    );
                }
                continue;
            }
            label_to_key.insert(label, key);
        }
        Ok(label_to_key)
    }

    async fn stream_reply(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        guild_id: Option<i64>,
        images: HashMap<String, String>,
        activity_state_line: Option<String>,
    ) -> Option<String> {
        let mut ctx =
            AssemblyContext::new(&self.familiar_id, Some(channel_id)).with_viewer_mode("text");
        if let Some(g) = guild_id {
            ctx = ctx.with_guild_id(g);
        }
        let prompt = self.assembler.assemble(&ctx).await;

        let focus_ch = self
            .focus_manager
            .as_ref()
            .and_then(|fm| fm.get_focus("text"));
        let unread_digest: Option<Vec<(i64, (i64, i64))>> = if self.focus_manager.is_some() {
            let staged = self
                .history
                .staged_channels(self.familiar_id.clone())
                .await
                .unwrap_or_default();
            Some(
                staged
                    .into_iter()
                    .map(|(cid, cu)| (cid, (cu.unread(), cu.pings())))
                    .collect(),
            )
        } else {
            None
        };
        let ch_names = self
            .focus_manager
            .as_ref()
            .map(|fm| fm.channel_names())
            .unwrap_or_default();

        let mut head = FinalReminder::new("text")
            .include_time(false)
            .channel_names(ch_names.clone());
        if let Some(fc) = focus_ch {
            head = head.focus_channel_id(fc);
        }
        if let Some(digest) = &unread_digest {
            head = head.unread_digest(digest.clone());
        }
        let reminder = head.render();
        let system = join_nonempty(
            &[&prompt.system_prompt, BOT_OUTPUT_INSTRUCTIONS, &reminder],
            "\n\n",
        );
        let mut messages: Vec<Message> = vec![Message::new("system", system)];
        messages.extend(prompt.recent_history);

        let guild_name = self
            .focus_manager
            .as_ref()
            .and_then(|fm| fm.guild_name_for(focus_ch));
        let mut trailing_b = FinalReminder::new("text")
            .display_tz(&self.display_tz)
            .include_mode_instruction(true)
            .post_history_instructions(&self.post_history_instructions)
            .channel_names(ch_names);
        if let Some(fc) = focus_ch {
            trailing_b = trailing_b.focus_channel_id(fc);
        }
        if let Some(digest) = &unread_digest {
            trailing_b = trailing_b.unread_digest(digest.clone());
        }
        if let Some(gn) = &guild_name {
            trailing_b = trailing_b.guild_name(gn.clone());
        }
        let mut trailing = trailing_b.render();
        if let Some(state_line) = &activity_state_line {
            trailing = format!("{trailing}\n\n{state_line}");
        }
        messages.push(Message::new("system", trailing));

        let tool_mode = self.tool_registry.is_some()
            && self.tool_context_factory.is_some()
            && (self.llm.tool_calling_enabled() || self.llm.image_tools_enabled());
        if tool_mode {
            self.stream_reply_with_tools(scope, channel_id, guild_id, messages, images)
                .await
        } else {
            self.stream_reply_bare(scope, channel_id, messages).await
        }
    }

    async fn stream_reply_bare(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        messages: Vec<Message>,
    ) -> Option<String> {
        let mut typing: Option<Box<dyn TypingIndicator>> = None;
        let out = self
            .stream_bare_inner(scope, channel_id, messages, &mut typing)
            .await;
        if let Some(ind) = typing {
            ind.close().await;
        }
        out
    }

    async fn stream_bare_inner(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        messages: Vec<Message>,
        typing: &mut Option<Box<dyn TypingIndicator>>,
    ) -> Option<String> {
        let mut accumulated = String::new();
        let mut silent = SilentDetector::new();
        let mut stream = match self.llm.stream_completion(messages, None).await {
            Ok(s) => s,
            Err(exc) => {
                tracing::warn!(
                    "{} {}",
                    ls::tag("Text", ls::R),
                    ls::kv_styled("llm_stream_error", &format!("{exc:?}"), ls::W, ls::R),
                );
                return None;
            }
        };
        while let Some(item) = stream.next().await {
            if scope.is_cancelled() {
                return None;
            }
            let delta = match item {
                Ok(d) => d,
                Err(exc) => {
                    tracing::warn!(
                        "{} {}",
                        ls::tag("Text", ls::R),
                        ls::kv_styled("llm_stream_error", &format!("{exc:?}"), ls::W, ls::R),
                    );
                    return None;
                }
            };
            if delta.content.is_empty() {
                continue;
            }
            accumulated.push_str(&delta.content);
            match silent.feed(&delta.content) {
                Some(true) => {
                    tracing::info!(
                        "{} {} {}",
                        ls::tag("\u{1f4a4} Text", ls::B),
                        ls::kv_styled("decision", "silent", ls::W, ls::LB),
                        ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
                    );
                    return None;
                }
                Some(false) => {
                    if typing.is_none()
                        && let Some(trigger) = &self.trigger_typing
                    {
                        *typing = Some(trigger.open(channel_id).await);
                    }
                }
                None => {}
            }
        }
        Some(accumulated)
    }

    async fn stream_reply_with_tools(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        guild_id: Option<i64>,
        mut messages: Vec<Message>,
        images: HashMap<String, String>,
    ) -> Option<String> {
        let (Some(factory), Some(registry)) = (&self.tool_context_factory, &self.tool_registry)
        else {
            return None;
        };
        let registry: &ToolRegistry = registry;
        let ctx = factory(channel_id, &scope.turn_id, images);
        let hooks = TextToolHooks {
            responder: self,
            scope,
            channel_id,
            guild_id,
            silent: Mutex::new(SilentDetector::new()),
            typing: Mutex::new(None),
            typing_started: AtomicBool::new(false),
            bail_silent: AtomicBool::new(false),
        };
        let llm: &dyn LlmClient = self.llm.as_ref();

        let out: Option<String> = 'block: {
            let mut result = match agentic_loop(
                llm,
                &mut messages,
                registry,
                &ctx,
                Some(&hooks),
                self.loop_max_iterations,
            )
            .await
            {
                Ok(r) => r,
                Err(exc) => {
                    tracing::warn!(
                        "{} {}",
                        ls::tag("Text", ls::R),
                        ls::kv_styled("llm_agentic_error", &format!("{exc:?}"), ls::W, ls::R),
                    );
                    break 'block None;
                }
            };
            // qwen leak quirk: an empty completion (no text, no tool, not
            // silent) earns exactly one retry (D17: the retry keeps Python's
            // library-default iteration cap).
            if result.final_content.is_empty()
                && !result.is_silent
                && result.tool_calls_made == 0
                && !hooks.bail_silent.load(Ordering::SeqCst)
                && !scope.is_cancelled()
            {
                tracing::info!(
                    "{} {} {}",
                    ls::tag("Text", ls::Y),
                    ls::kv_styled("retry", "empty_completion", ls::W, ls::LY),
                    ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
                );
                result = match agentic_loop(
                    llm,
                    &mut messages,
                    registry,
                    &ctx,
                    Some(&hooks),
                    DEFAULT_MAX_ITERATIONS,
                )
                .await
                {
                    Ok(r) => r,
                    Err(exc) => {
                        tracing::warn!(
                            "{} {}",
                            ls::tag("Text", ls::R),
                            ls::kv_styled("llm_agentic_error", &format!("{exc:?}"), ls::W, ls::R),
                        );
                        break 'block None;
                    }
                };
            }
            if hooks.bail_silent.load(Ordering::SeqCst) || result.is_silent {
                tracing::info!(
                    "{} {} {}",
                    ls::tag("\u{1f4a4} Text", ls::B),
                    ls::kv_styled("decision", "silent", ls::W, ls::LB),
                    ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
                );
                break 'block None;
            }
            if scope.is_cancelled() {
                break 'block None;
            }
            Some(result.final_content)
        };

        let ind = hooks.typing.lock().expect("typing mutex").take();
        if let Some(ind) = ind {
            ind.close().await;
        }
        out
    }
}

/// Preferred display label for an author (`display_name`, else `"unknown"`).
fn author_display(author: &Author) -> &str {
    author.display_name.as_deref().unwrap_or("unknown")
}

// ---------------------------------------------------------------------------
// Agentic-loop hooks for the tool path
// ---------------------------------------------------------------------------

struct TextToolHooks<'a> {
    responder: &'a TextResponder,
    scope: &'a TurnScope,
    channel_id: i64,
    guild_id: Option<i64>,
    silent: Mutex<SilentDetector>,
    typing: Mutex<Option<Box<dyn TypingIndicator>>>,
    typing_started: AtomicBool,
    bail_silent: AtomicBool,
}

#[async_trait]
impl AgenticHooks for TextToolHooks<'_> {
    async fn on_delta(&self, delta: &LlmDelta) {
        if self.scope.is_cancelled() || self.bail_silent.load(Ordering::SeqCst) {
            return;
        }
        if delta.content.is_empty() {
            return;
        }
        let decision = self
            .silent
            .lock()
            .expect("tool silent mutex")
            .feed(&delta.content);
        match decision {
            Some(true) => {
                self.bail_silent.store(true, Ordering::SeqCst);
            }
            Some(false) => {
                if !self.typing_started.swap(true, Ordering::SeqCst)
                    && let Some(trigger) = &self.responder.trigger_typing
                {
                    let ind = trigger.open(self.channel_id).await;
                    *self.typing.lock().expect("typing mutex") = Some(ind);
                }
            }
            None => {}
        }
    }

    async fn on_iteration_end(&self, assistant: &Message, tool_msgs: &[Message]) {
        let tool_calls = match &assistant.tool_calls {
            Some(tc) if !tc.is_empty() => tc,
            _ => return,
        };
        let tcj = serde_json::to_string(tool_calls).unwrap_or_else(|_| "[]".to_owned());
        let mut append = AppendTurn::new(
            &self.responder.familiar_id,
            self.channel_id,
            "assistant",
            assistant.content_str(),
        )
        .tool_calls_json(tcj);
        if let Some(g) = self.guild_id {
            append = append.guild_id(g);
        }
        // Hook returns `()`; a store failure here is logged and dropped (Python
        // propagated it out of the loop — see the port summary).
        if let Err(e) = self.responder.history.append_turn(append).await {
            tracing::warn!("{} tool-turn persist failed: {e}", ls::tag("Text", ls::R));
        }
        for tm in tool_msgs {
            let mut a = AppendTurn::new(
                &self.responder.familiar_id,
                self.channel_id,
                "tool",
                tool_content_as_text(&tm.content),
            );
            if let Some(id) = &tm.tool_call_id {
                a = a.tool_call_id(id);
            }
            if let Some(g) = self.guild_id {
                a = a.guild_id(g);
            }
            if let Err(e) = self.responder.history.append_turn(a).await {
                tracing::warn!("{} tool-turn persist failed: {e}", ls::tag("Text", ls::R));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{consume_thread_marker, rewrite_pings, strip_leaked_metadata_prefix};
    use std::collections::HashMap;

    #[test]
    fn strips_leading_id_clump() {
        let leaked = "[#1500709436557445449] 4:03AM UTC. I actually know that now!";
        assert_eq!(
            strip_leaked_metadata_prefix(leaked),
            "4:03AM UTC. I actually know that now!"
        );
    }

    #[test]
    fn strips_leading_time_clump() {
        assert_eq!(strip_leaked_metadata_prefix("[4:03AM] hello"), "hello");
    }

    #[test]
    fn strips_chained_metadata_clumps() {
        let leaked = "[14:32 Alice #abc] [↩ msg-1] hi";
        assert!(strip_leaked_metadata_prefix(leaked).starts_with("[↩ msg-1] hi"));
    }

    #[test]
    fn leaves_legitimate_bracketed_text_alone() {
        assert_eq!(
            strip_leaked_metadata_prefix("[note] heads up"),
            "[note] heads up"
        );
    }

    #[test]
    fn passes_through_clean_text() {
        assert_eq!(
            strip_leaked_metadata_prefix("just a normal reply"),
            "just a normal reply"
        );
    }

    #[test]
    fn thread_marker_bare() {
        let (stripped, wants, target) = consume_thread_marker("[↩] sure thing");
        assert!(wants);
        assert_eq!(target, None);
        assert_eq!(stripped, "sure thing");
    }

    #[test]
    fn thread_marker_explicit_id() {
        let (stripped, wants, target) = consume_thread_marker("[↩ older-msg-7] sure");
        assert!(wants);
        assert_eq!(target.as_deref(), Some("older-msg-7"));
        assert_eq!(stripped, "sure");
    }

    #[test]
    fn thread_marker_hash_sigil_stripped() {
        let (_stripped, _wants, target) = consume_thread_marker("[↩ #1500691573262782604] x");
        assert_eq!(target.as_deref(), Some("1500691573262782604"));
    }

    #[test]
    fn thread_marker_reply_word_form() {
        let (stripped, wants, target) = consume_thread_marker("[reply] got it");
        assert!(wants);
        assert_eq!(target, None);
        assert_eq!(stripped, "got it");
    }

    #[test]
    fn thread_marker_absent() {
        let (stripped, wants, target) = consume_thread_marker("no marker here");
        assert!(!wants);
        assert_eq!(target, None);
        assert_eq!(stripped, "no marker here");
    }

    #[test]
    fn ping_known_label_rewritten() {
        let mut map = HashMap::new();
        map.insert("Bob".to_owned(), "discord:222".to_owned());
        let (out, ids) = rewrite_pings("sure thing, [@Bob]", &map);
        assert!(out.contains("<@222>"));
        assert!(!out.contains("[@Bob]"));
        assert_eq!(ids, vec![222]);
    }

    #[test]
    fn ping_unknown_label_plain() {
        let map = HashMap::new();
        let (out, ids) = rewrite_pings("hi [@Nobody], welcome", &map);
        assert!(!out.contains("<@"));
        assert!(out.contains("@Nobody"));
        assert!(ids.is_empty());
    }

    #[test]
    fn ping_non_discord_key_plain() {
        let mut map = HashMap::new();
        map.insert("Ann".to_owned(), "twitch:5".to_owned());
        let (out, ids) = rewrite_pings("[@Ann]", &map);
        assert_eq!(out, "@Ann");
        assert!(ids.is_empty());
    }
}
