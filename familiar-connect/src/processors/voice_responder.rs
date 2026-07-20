//! Voice reply orchestrator (subsystem 06; Python `processors/voice_responder.py`).
//!
//! Consumes `voice.activity.start` + `voice.transcript.final`, produces an LLM
//! reply and speaks it through the injected [`TtsPlayer`]. The dispatcher hands
//! each FINAL to a spawned per-(session, user) task so barge-in (a fresh
//! `activity.start`) can cancel a prior turn parked at an LLM/TTS await. A
//! per-channel reply gate serializes `set_rag_cue → assemble → stream+speak →
//! assistant-commit` across speakers (the user-turn append stays outside it).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;

use crate::bus::envelope::{Event, TurnScope};
use crate::bus::protocols::EventBus;
use crate::bus::router::TurnRouter;
use crate::bus::topics::{TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL};
use crate::context::assembler::{Assembler, AssemblyContext};
use crate::context::final_reminder::FinalReminder;
use crate::diagnostics::cold_cache::log_signals;
use crate::diagnostics::voice_budget::{
    PHASE_LLM_FIRST_TOKEN, PHASE_TTS_FIRST_AUDIO, get_voice_budget_recorder,
};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{AppendTurn, FOCUS_STREAM_CHANNEL_ID};
use crate::identity::Author;
use crate::llm::{LlmClient, LlmDelta, Message};
use crate::log_style as ls;
use crate::processors::{
    FocusManagerApi, MemberResolver, ResponderLlm, ToolContextFactory, VoiceActivityStart,
    VoiceTranscriptFinal,
};
use crate::sentence_streamer::SentenceStreamer;
use crate::silence::{StreamDecision, StreamGate};
use crate::tools::agentic::{
    AgenticHooks, agentic_loop, guard_leaked_content, tool_content_as_text,
};
use crate::tools::registry::ToolRegistry;
use crate::tts_player::protocol::TtsPlayer;

// Cold-cache signal thresholds (Python `log_signals` defaults).
const TOPIC_SHIFT_THRESHOLD: f64 = 0.15;
const TOPIC_SHIFT_MIN_TOKENS: usize = 4;
const SILENCE_GAP_THRESHOLD_S: f64 = 300.0;

/// Default filler phrases spoken on empty-content tool iterations.
pub const DEFAULT_FILLER_PHRASES: [&str; 3] = ["one sec...", "hold on...", "checking..."];

fn join_system(system_prompt: &str, reminder: &str) -> String {
    [system_prompt, reminder]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

struct InflightEntry {
    generation: u64,
    handle: tokio::task::JoinHandle<()>,
}

/// Orchestrates the voice reply loop with turn-scoped cancellation.
pub struct VoiceResponder {
    inner: Arc<VoiceInner>,
}

struct VoiceInner {
    assembler: Arc<Assembler>,
    llm: Arc<dyn ResponderLlm>,
    tts: Arc<dyn TtsPlayer>,
    history: Arc<AsyncHistoryStore>,
    router: Arc<TurnRouter>,
    familiar_id: String,
    member_resolver: Option<MemberResolver>,
    tool_registry: Option<Arc<ToolRegistry>>,
    tool_context_factory: Option<ToolContextFactory>,
    tool_filler_phrases: Vec<String>,
    tool_filler_idx: AtomicUsize,
    post_history_instructions: String,
    display_tz: String,
    focus_manager: Option<Arc<dyn FocusManagerApi>>,
    loop_max_iterations: usize,
    inflight: Mutex<HashMap<String, InflightEntry>>,
    reply_locks: Mutex<HashMap<i64, Arc<tokio::sync::Mutex<()>>>>,
    next_generation: AtomicU64,
}

impl VoiceResponder {
    /// The processor's human label.
    pub const NAME: &'static str = "voice-responder";

    /// Construct a responder with the required collaborators; optional seams
    /// default off (`with_*` setters wire them). `tool_filler_phrases` defaults
    /// to [`DEFAULT_FILLER_PHRASES`].
    #[must_use]
    pub fn new(
        assembler: Arc<Assembler>,
        llm: Arc<dyn ResponderLlm>,
        tts: Arc<dyn TtsPlayer>,
        history: Arc<AsyncHistoryStore>,
        router: Arc<TurnRouter>,
        familiar_id: impl Into<String>,
    ) -> Self {
        Self {
            inner: Arc::new(VoiceInner {
                assembler,
                llm,
                tts,
                history,
                router,
                familiar_id: familiar_id.into(),
                member_resolver: None,
                tool_registry: None,
                tool_context_factory: None,
                tool_filler_phrases: DEFAULT_FILLER_PHRASES
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                tool_filler_idx: AtomicUsize::new(0),
                post_history_instructions: String::new(),
                display_tz: "UTC".to_owned(),
                focus_manager: None,
                loop_max_iterations: 5,
                inflight: Mutex::new(HashMap::new()),
                reply_locks: Mutex::new(HashMap::new()),
                next_generation: AtomicU64::new(0),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut VoiceInner {
        Arc::get_mut(&mut self.inner).expect("no clones during construction")
    }

    /// Wire the sync member resolver (`(channel_id, user_id) -> Author | None`).
    #[must_use]
    pub fn with_member_resolver(mut self, resolver: MemberResolver) -> Self {
        self.inner_mut().member_resolver = Some(resolver);
        self
    }
    /// Wire the agentic-loop tool registry + context factory.
    #[must_use]
    pub fn with_tools(mut self, registry: Arc<ToolRegistry>, factory: ToolContextFactory) -> Self {
        let inner = self.inner_mut();
        inner.tool_registry = Some(registry);
        inner.tool_context_factory = Some(factory);
        self
    }
    /// Set the filler-phrase round-robin (empty disables it).
    #[must_use]
    pub fn with_tool_filler_phrases(mut self, phrases: Vec<String>) -> Self {
        self.inner_mut().tool_filler_phrases = phrases;
        self
    }
    /// Set the deepest trailing-reminder etiquette (empty = omitted).
    #[must_use]
    pub fn with_post_history_instructions(mut self, text: impl Into<String>) -> Self {
        self.inner_mut().post_history_instructions = text.into();
        self
    }
    /// Set the trailing-reminder clock timezone.
    #[must_use]
    pub fn with_display_tz(mut self, tz: impl Into<String>) -> Self {
        self.inner_mut().display_tz = tz.into();
        self
    }
    /// Wire the attentional focus controller.
    #[must_use]
    pub fn with_focus_manager(mut self, fm: Arc<dyn FocusManagerApi>) -> Self {
        self.inner_mut().focus_manager = Some(fm);
        self
    }
    /// Set the agentic-loop iteration cap.
    #[must_use]
    pub fn with_loop_max_iterations(mut self, n: usize) -> Self {
        self.inner_mut().loop_max_iterations = n;
        self
    }

    /// The processor name (`"voice-responder"`).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        Self::NAME
    }

    /// The subscribed topics.
    #[must_use]
    pub const fn topics(&self) -> [&'static str; 2] {
        [TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL]
    }

    /// Dispatch: `activity.start` runs inline; `transcript.final` spawns.
    ///
    /// # Errors
    /// Never fails at dispatch (spawned work handles its own errors).
    #[allow(clippy::unused_async, reason = "processor contract is async")]
    pub async fn handle(&self, event: &Event, _bus: &dyn EventBus) -> anyhow::Result<()> {
        if event.topic == TOPIC_VOICE_ACTIVITY_START {
            self.inner.on_activity_start(event);
        } else if event.topic == TOPIC_VOICE_TRANSCRIPT_FINAL {
            self.inner.spawn_final(event);
        }
        Ok(())
    }

    /// Await every in-flight final-handling task (tests + graceful shutdown).
    ///
    /// Never raises: a task aborted mid-flight yields a `JoinError` that is
    /// swallowed, mirroring the Python `gather(return_exceptions=True)`.
    pub async fn wait_until_idle(&self) {
        let handles: Vec<tokio::task::JoinHandle<()>> = {
            let mut inflight = self.inner.inflight.lock().expect("inflight mutex");
            inflight.drain().map(|(_, e)| e.handle).collect()
        };
        for handle in handles {
            let _ = handle.await;
        }
    }
}

impl VoiceInner {
    fn on_activity_start(&self, event: &Event) {
        let user_id = user_id_from_event(event);
        let scope_key = scope_key(&event.session_id, user_id);
        // begin_turn cancels any prior scope for this (session, user). No
        // global tts.stop(): the shared voice client would cut a *different*
        // user's reply — the player's poll loop halts same-speaker playback.
        self.router.begin_turn(&scope_key, &event.turn_id);
    }

    fn spawn_final(self: &Arc<Self>, event: &Event) {
        let user_id = user_id_from_event(event);
        let scope_key = scope_key(&event.session_id, user_id);
        let event = event.clone();
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let inner = Arc::clone(self);
        let key = scope_key.clone();
        // Take the inflight lock BEFORE spawning and hold it across the insert.
        // Python installs `self._inflight[key] = task` synchronously (asyncio's
        // create_task cannot run the coroutine until the current one awaits), so
        // the done-callback never fires before the entry exists. On a
        // multi-thread tokio runtime the spawned task can begin — and, for a
        // fast stale-turn/empty-text drop, finish — on another worker thread
        // before this function returns; holding the lock across spawn+insert
        // blocks the task's completion cleanup (which locks the same mutex)
        // until our entry is installed, restoring the V5 ownership ordering (no
        // await occurs while the lock is held, so this cannot deadlock).
        let mut inflight = self.inflight.lock().expect("inflight mutex");
        let handle = tokio::spawn(async move {
            // Rust task-abort drops the future silently on barge-in (no
            // CancelledError to swallow).
            let _ = inner.on_final(event).await;
            // Clear the slot only if we still own it — a newer task may have
            // replaced our entry.
            let mut inflight = inner.inflight.lock().expect("inflight mutex");
            if inflight
                .get(&key)
                .is_some_and(|e| e.generation == generation)
            {
                inflight.remove(&key);
            }
        });
        if let Some(prev) = inflight.insert(scope_key, InflightEntry { generation, handle }) {
            // A newer FINAL for the same speaker without an intervening
            // activity.start: hard-cancel the prior task (V5).
            prev.handle.abort();
        }
    }

    async fn on_final(&self, event: Event) -> anyhow::Result<()> {
        let user_id = user_id_from_event(&event);
        let scope_key = scope_key(&event.session_id, user_id);
        let Some(scope) = self.router.active_scope(&scope_key) else {
            return Ok(());
        };
        if scope.turn_id != event.turn_id {
            return Ok(());
        }
        let Some(channel_id) = parse_voice_session(&event.session_id) else {
            return Ok(());
        };
        let text = event
            .payload
            .downcast_ref::<VoiceTranscriptFinal>()
            .map(|p| p.text.clone())
            .unwrap_or_default();
        if text.is_empty() {
            return Ok(());
        }

        let author = self.resolve_author(channel_id, user_id);
        // Cold-cache instrumentation runs before the user turn is appended so
        // `prev_turn_at` reflects the real gap.
        self.emit_cold_cache_signals(channel_id, &scope.turn_id, &text)
            .await;

        // User turn appended OUTSIDE the reply gate — observation is never gated.
        let mut append = AppendTurn::new(&self.familiar_id, channel_id, "user", &text);
        if let Some(a) = &author {
            append = append.author(a.clone());
        }
        self.history.append_turn(append).await?;

        // Per-channel reply gate.
        let gate = self.gate_for(channel_id);
        let _guard = gate.lock().await;
        self.assembler.set_rag_cue(&text);

        let reply = self.stream_and_speak(&scope, channel_id).await;
        let Some(reply) = reply else {
            return Ok(());
        };
        if scope.is_cancelled() {
            return Ok(());
        }
        if reply.trim().is_empty() {
            tracing::warn!(
                "{} {} {}",
                ls::tag("Voice", ls::Y),
                ls::kv_styled("skip", "empty_reply", ls::W, ls::LY),
                ls::kv_styled("turn", &scope.turn_id, ls::W, ls::LC),
            );
            return Ok(());
        }

        self.history
            .append_turn(AppendTurn::new(
                &self.familiar_id,
                channel_id,
                "assistant",
                &reply,
            ))
            .await?;
        self.router.end_turn(&scope);
        if let Some(fm) = &self.focus_manager {
            fm.end_turn().await;
        }
        Ok(())
    }

    fn gate_for(&self, channel_id: i64) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.reply_locks.lock().expect("reply_locks mutex");
        Arc::clone(
            locks
                .entry(channel_id)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    fn resolve_author(&self, channel_id: i64, user_id: Option<i64>) -> Option<Author> {
        let uid = user_id?;
        let resolver = self.member_resolver.as_ref()?;
        resolver(channel_id, uid)
    }

    async fn stream_and_speak(&self, scope: &TurnScope, channel_id: i64) -> Option<String> {
        let ctx =
            AssemblyContext::new(&self.familiar_id, Some(channel_id)).with_viewer_mode("voice");
        let prompt = self.assembler.assemble(&ctx).await;
        let tool_mode = self.tool_registry.is_some()
            && self.tool_context_factory.is_some()
            && self.llm.tool_calling_enabled();

        // Head reminder INCLUDES the timestamp (asymmetric with text).
        let reminder = FinalReminder::new("voice")
            .display_tz(&self.display_tz)
            .tools_enabled(tool_mode)
            .render();
        let system = join_system(&prompt.system_prompt, &reminder);
        let mut messages: Vec<Message> = Vec::new();
        if !system.is_empty() {
            messages.push(Message::new("system", system));
        }
        messages.extend(prompt.recent_history);

        let mut trailing_b = FinalReminder::new("voice")
            .display_tz(&self.display_tz)
            .include_mode_instruction(true)
            .tools_enabled(tool_mode)
            .post_history_instructions(&self.post_history_instructions);
        if let Some(fm) = &self.focus_manager {
            trailing_b = trailing_b
                .focus_channel_id(channel_id)
                .channel_names(fm.channel_names());
            if let Some(gn) = fm.guild_name_for(Some(channel_id)) {
                trailing_b = trailing_b.guild_name(gn);
            }
        }
        messages.push(Message::new("system", trailing_b.render()));

        if tool_mode {
            self.stream_and_speak_with_tools(scope, channel_id, messages)
                .await
        } else {
            self.stream_and_speak_bare(scope, channel_id, messages)
                .await
        }
    }

    #[allow(clippy::too_many_lines, reason = "one cohesive streaming pipeline")]
    async fn stream_and_speak_bare(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        messages: Vec<Message>,
    ) -> Option<String> {
        let mut accumulated = String::new();
        let mut streamer = SentenceStreamer::new();
        let mut gate = StreamGate::new();
        let mut pending: VecDeque<String> = VecDeque::new();
        let mut gate_open = false;
        let budget = get_voice_budget_recorder();
        let mut first_delta_seen = false;
        let mut decision_logged = false;

        let mut stream = match self.llm.stream_completion(messages, None).await {
            Ok(s) => s,
            Err(exc) => {
                log_stream_error(&exc);
                return None;
            }
        };
        while let Some(item) = stream.next().await {
            let delta = match item {
                Ok(d) => d,
                Err(exc) => {
                    log_stream_error(&exc);
                    return None;
                }
            };
            // Cancellation check FIRST, unconditionally (spec V12): a barge-in
            // that lands while the stream is emitting content-empty deltas
            // (finish/role-only frames) must still log `decision=preempted` and
            // bail, exactly as Python's `chat_stream` loop checks
            // `scope.is_cancelled()` as its first per-delta statement. Skipping
            // empty deltas ahead of this check would let a cancelled turn slip
            // past without the preempted marker.
            if scope.is_cancelled() {
                if !decision_logged {
                    log_preempted(&scope.turn_id);
                }
                return None;
            }
            if delta.content.is_empty() {
                continue;
            }
            if !first_delta_seen {
                budget.record(&scope.turn_id, PHASE_LLM_FIRST_TOKEN, None);
                first_delta_seen = true;
            }
            accumulated.push_str(&delta.content);

            if !gate_open {
                match gate.feed(&delta.content) {
                    StreamDecision::Silent => {
                        self.log_silent(&scope.turn_id, channel_id);
                        return None;
                    }
                    StreamDecision::Suppress => {
                        // A leaked tool-call block must never reach TTS or the
                        // persisted turn (issue #109); drop the buffered
                        // sentences and abandon the turn as empty.
                        self.log_leak(&scope.turn_id, channel_id);
                        return None;
                    }
                    StreamDecision::Speak => {
                        gate_open = true;
                        self.log_respond(&scope.turn_id, channel_id);
                        decision_logged = true;
                    }
                    StreamDecision::Pending => {}
                }
            }
            for sentence in streamer.feed(&delta.content) {
                pending.push_back(sentence);
            }
            if gate_open {
                while let Some(sentence) = pending.pop_front() {
                    if scope.is_cancelled() {
                        if !decision_logged {
                            log_preempted(&scope.turn_id);
                        }
                        return None;
                    }
                    self.speak(&sentence, scope).await;
                }
            }
        }

        // Stream ended undecided with non-whitespace content (a very short
        // reply) → treat as the speak path.
        if !gate_open && gate.decided().is_none() && !accumulated.trim().is_empty() {
            self.log_respond(&scope.turn_id, channel_id);
            decision_logged = true;
            gate_open = true;
        }
        if gate_open {
            let tail = streamer.flush();
            if !tail.trim().is_empty() {
                pending.push_back(tail);
            }
            while let Some(sentence) = pending.pop_front() {
                if scope.is_cancelled() {
                    if !decision_logged {
                        log_preempted(&scope.turn_id);
                    }
                    return None;
                }
                self.speak(&sentence, scope).await;
            }
        }
        // Belt-and-suspenders: strip any leaked leading block from the persisted
        // string too, so history never keeps the raw leak (the streaming gate
        // above already kept it out of TTS).
        Some(guard_leaked_content(&accumulated))
    }

    async fn stream_and_speak_with_tools(
        &self,
        scope: &TurnScope,
        channel_id: i64,
        mut messages: Vec<Message>,
    ) -> Option<String> {
        let (Some(factory), Some(registry)) = (&self.tool_context_factory, &self.tool_registry)
        else {
            return None;
        };
        let registry: &ToolRegistry = registry;
        let ctx = factory(channel_id, &scope.turn_id, HashMap::new());
        let hooks = VoiceToolHooks {
            inner: self,
            scope,
            channel_id,
            state: tokio::sync::Mutex::new(VoiceToolState {
                accumulated: String::new(),
                streamer: SentenceStreamer::new(),
                gate: StreamGate::new(),
                pending: VecDeque::new(),
                gate_open: false,
                first_delta_seen: false,
                decision_logged: false,
            }),
        };
        let llm: &dyn LlmClient = self.llm.as_ref();

        if let Err(exc) = agentic_loop(
            llm,
            &mut messages,
            registry,
            &ctx,
            Some(&hooks),
            self.loop_max_iterations,
        )
        .await
        {
            log_agentic_error(&exc);
            return None;
        }

        let mut st = hooks.state.lock().await;
        // A leaked `<silent>` / tool-call block that led the stream (issue #109):
        // abandon the turn as silent/empty so nothing is spoken or persisted.
        if matches!(
            st.gate.decided(),
            Some(StreamDecision::Silent | StreamDecision::Suppress)
        ) {
            return None;
        }
        if st.gate_open {
            let tail = st.streamer.flush();
            if !tail.trim().is_empty() {
                st.pending.push_back(tail);
            }
            while let Some(sentence) = st.pending.pop_front() {
                if scope.is_cancelled() {
                    break;
                }
                self.speak(&sentence, scope).await;
            }
        }
        if !st.gate_open && st.gate.decided().is_none() && !st.accumulated.trim().is_empty() {
            self.log_respond(&scope.turn_id, channel_id);
            st.decision_logged = true;
        }
        if scope.is_cancelled() && !st.decision_logged {
            log_preempted(&scope.turn_id);
            return None;
        }
        // Belt-and-suspenders: strip any leaked leading block from the persisted
        // string too (the streaming gate already kept it out of TTS).
        Some(guard_leaked_content(&st.accumulated))
    }

    fn next_filler_phrase(&self) -> String {
        if self.tool_filler_phrases.is_empty() {
            return String::new();
        }
        let idx = self.tool_filler_idx.fetch_add(1, Ordering::Relaxed);
        self.tool_filler_phrases[idx % self.tool_filler_phrases.len()].clone()
    }

    async fn speak(&self, text: &str, scope: &TurnScope) {
        if text.trim().is_empty() {
            return;
        }
        // First call per turn marks tts_first_audio (the recorder dedupes).
        get_voice_budget_recorder().record(&scope.turn_id, PHASE_TTS_FIRST_AUDIO, None);
        self.tts.speak(text, scope).await;
    }

    async fn emit_cold_cache_signals(&self, channel_id: i64, turn_id: &str, text: &str) {
        let summary = self
            .history
            .get_summary(self.familiar_id.clone(), FOCUS_STREAM_CHANNEL_ID)
            .await
            .ok()
            .flatten();
        let prior_context = summary.map(|s| s.summary_text).unwrap_or_default();
        let recent = self
            .history
            .recent(self.familiar_id.clone(), channel_id, 1, None, None)
            .await
            .unwrap_or_default();
        let prev_at = recent.first().map(|t| t.timestamp);
        let _fired = log_signals(
            channel_id,
            turn_id,
            text,
            &prior_context,
            prev_at,
            chrono::Utc::now(),
            TOPIC_SHIFT_THRESHOLD,
            TOPIC_SHIFT_MIN_TOKENS,
            SILENCE_GAP_THRESHOLD_S,
        );
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

    fn log_respond(&self, turn_id: &str, channel_id: i64) {
        let (ch, srv) = self.origin_fields(channel_id);
        tracing::info!(
            "{} {}{}{} {}",
            ls::tag("Voice", ls::G),
            ch,
            srv,
            ls::kv_styled("decision", "respond", ls::W, ls::LG),
            ls::kv_styled("turn", turn_id, ls::W, ls::LC),
        );
    }

    fn log_silent(&self, turn_id: &str, channel_id: i64) {
        let (ch, srv) = self.origin_fields(channel_id);
        tracing::info!(
            "{} {}{}{} {}",
            ls::tag("\u{1f4a4} Voice", ls::B),
            ch,
            srv,
            ls::kv_styled("decision", "silent", ls::W, ls::LB),
            ls::kv_styled("turn", turn_id, ls::W, ls::LC),
        );
    }

    fn log_leak(&self, turn_id: &str, channel_id: i64) {
        let (ch, srv) = self.origin_fields(channel_id);
        tracing::warn!(
            "{} {}{}{} {}",
            ls::tag("Voice", ls::Y),
            ch,
            srv,
            ls::kv_styled("decision", "leaked_tool_suppressed", ls::W, ls::LY),
            ls::kv_styled("turn", turn_id, ls::W, ls::LC),
        );
    }
}

fn log_preempted(turn_id: &str) {
    tracing::info!(
        "{} {} {}",
        ls::tag("Voice", ls::Y),
        ls::kv_styled("decision", "preempted", ls::W, ls::LY),
        ls::kv_styled("turn", turn_id, ls::W, ls::LC),
    );
}

fn log_stream_error(exc: &anyhow::Error) {
    tracing::warn!(
        "{} {}",
        ls::tag("Voice", ls::R),
        ls::kv_styled("llm_stream_error", &format!("{exc:?}"), ls::W, ls::R),
    );
}

fn log_agentic_error(exc: &anyhow::Error) {
    tracing::warn!(
        "{} {}",
        ls::tag("Voice", ls::R),
        ls::kv_styled("llm_agentic_error", &format!("{exc:?}"), ls::W, ls::R),
    );
}

// ---------------------------------------------------------------------------
// Voice tool-path hooks
// ---------------------------------------------------------------------------

struct VoiceToolState {
    accumulated: String,
    streamer: SentenceStreamer,
    gate: StreamGate,
    pending: VecDeque<String>,
    gate_open: bool,
    first_delta_seen: bool,
    decision_logged: bool,
}

struct VoiceToolHooks<'a> {
    inner: &'a VoiceInner,
    scope: &'a TurnScope,
    channel_id: i64,
    state: tokio::sync::Mutex<VoiceToolState>,
}

#[async_trait]
impl AgenticHooks for VoiceToolHooks<'_> {
    async fn on_delta(&self, delta: &LlmDelta) {
        if self.scope.is_cancelled() || delta.content.is_empty() {
            return;
        }
        let mut st = self.state.lock().await;
        if !st.first_delta_seen {
            get_voice_budget_recorder().record(&self.scope.turn_id, PHASE_LLM_FIRST_TOKEN, None);
            st.first_delta_seen = true;
        }
        st.accumulated.push_str(&delta.content);
        if !st.gate_open {
            match st.gate.feed(&delta.content) {
                StreamDecision::Silent => {
                    self.inner.log_silent(&self.scope.turn_id, self.channel_id);
                    st.pending.clear();
                    return;
                }
                StreamDecision::Suppress => {
                    // Leaked tool-call block (issue #109): drop the buffered
                    // sentences; the latched gate decision below abandons the
                    // turn so nothing reaches TTS or the persisted turn.
                    self.inner.log_leak(&self.scope.turn_id, self.channel_id);
                    st.pending.clear();
                    return;
                }
                StreamDecision::Speak => {
                    st.gate_open = true;
                    self.inner.log_respond(&self.scope.turn_id, self.channel_id);
                    st.decision_logged = true;
                }
                StreamDecision::Pending => {}
            }
        }
        for sentence in st.streamer.feed(&delta.content) {
            st.pending.push_back(sentence);
        }
        if st.gate_open {
            while let Some(sentence) = st.pending.pop_front() {
                if self.scope.is_cancelled() {
                    return;
                }
                self.inner.speak(&sentence, self.scope).await;
            }
        }
    }

    async fn on_before_tools(&self, assistant: &Message) {
        {
            let mut st = self.state.lock().await;
            if st.gate_open {
                let tail = st.streamer.flush();
                if !tail.trim().is_empty() {
                    st.pending.push_back(tail);
                }
                while let Some(sentence) = st.pending.pop_front() {
                    if self.scope.is_cancelled() {
                        break;
                    }
                    self.inner.speak(&sentence, self.scope).await;
                }
            }
        }
        // Filler backstop: an imminent tool call with no spoken content →
        // speak the next filler phrase BEFORE the handler runs.
        let has_calls = assistant
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty());
        if has_calls && assistant.content_str().trim().is_empty() {
            let phrase = self.inner.next_filler_phrase();
            if !phrase.is_empty() && !self.scope.is_cancelled() {
                self.inner.speak(&phrase, self.scope).await;
            }
        }
    }

    async fn on_iteration_end(&self, assistant: &Message, tool_msgs: &[Message]) {
        let tool_calls = match &assistant.tool_calls {
            Some(tc) if !tc.is_empty() => tc,
            _ => return,
        };
        let tcj = serde_json::to_string(tool_calls).unwrap_or_else(|_| "[]".to_owned());
        let append = AppendTurn::new(
            &self.inner.familiar_id,
            self.channel_id,
            "assistant",
            assistant.content_str(),
        )
        .tool_calls_json(tcj);
        if let Err(e) = self.inner.history.append_turn(append).await {
            tracing::warn!("{} tool-turn persist failed: {e}", ls::tag("Voice", ls::R));
        }
        for tm in tool_msgs {
            let mut a = AppendTurn::new(
                &self.inner.familiar_id,
                self.channel_id,
                "tool",
                tool_content_as_text(&tm.content),
            );
            if let Some(id) = &tm.tool_call_id {
                a = a.tool_call_id(id);
            }
            if let Err(e) = self.inner.history.append_turn(a).await {
                tracing::warn!("{} tool-turn persist failed: {e}", ls::tag("Voice", ls::R));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn user_id_from_event(event: &Event) -> Option<i64> {
    if let Some(p) = event.payload.downcast_ref::<VoiceActivityStart>() {
        return p.user_id;
    }
    if let Some(p) = event.payload.downcast_ref::<VoiceTranscriptFinal>() {
        return p.user_id;
    }
    None
}

fn scope_key(session_id: &str, user_id: Option<i64>) -> String {
    user_id.map_or_else(
        || session_id.to_owned(),
        |uid| format!("{session_id}:user:{uid}"),
    )
}

fn parse_voice_session(session_id: &str) -> Option<i64> {
    session_id.strip_prefix("voice:")?.parse::<i64>().ok()
}
