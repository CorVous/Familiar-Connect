//! Attentional focus controller for a single familiar (subsystem 05; Python
//! `focus.py`).
//!
//! Two independent focus pointers (text, voice). Focus shifts are model-decided
//! (via the `shift_focus` tool) and applied immediately at tool-call time — no
//! deferral (behavior 50; `end_turn` is a no-op). An unread nudge fires when a
//! non-focused channel receives traffic, throttled by a debounce window with an
//! injectable clock (DESIGN §4.8).
//!
//! Per DESIGN port notes the two Python `asyncio.Lock`s collapse into a single
//! [`std::sync::Mutex`] over the focus state — an acceptable simplification that
//! also closes the benign cross-modal double-pointer persist race (behavior 49).
//! `on_shift` is invoked outside the lock. The store dependency is a narrow
//! [`FocusStore`] trait so tests inject a scripted double.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use futures::future::BoxFuture;

use crate::history::StoreError;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{FocusPointers, Promotion};
use crate::log_style as ls;
use crate::subscriptions::{SubscriptionKind, SubscriptionView};

/// Default debounce window: rapid arrivals within it share one nudge.
const DEFAULT_NUDGE_DEBOUNCE_S: f64 = 30.0;

/// Default catch-up window — staged turns taken in on a shift.
const DEFAULT_CATCH_UP_LIMIT: usize = 20;

/// Server label for DM channels admitted via the allowlist. This module owns
/// `guild_names` / `guild_name_for`, so the constant lives here.
pub const PRIVATE_MESSAGE_GUILD_NAME: &str = "Private Message";

/// Injectable monotonic clock returning seconds.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Presence-refresh hook awaited once after each applied shift.
pub type OnShift = Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>;

/// The narrow store seam [`FocusManager`] needs (DESIGN §4.8: tests inject a
/// scripted double). Implemented for [`AsyncHistoryStore`].
#[async_trait]
pub trait FocusStore: Send + Sync {
    /// Load the persisted focus pointers row, if any.
    async fn get_focus_pointers(
        &self,
        familiar_id: &str,
    ) -> Result<Option<FocusPointers>, StoreError>;
    /// Upsert both pointers.
    async fn set_focus_pointers(
        &self,
        familiar_id: &str,
        text_channel_id: Option<i64>,
        voice_channel_id: Option<i64>,
    ) -> Result<(), StoreError>;
    /// Promote a text channel's staged backlog to consumed on shift.
    async fn promote_staged_turns(
        &self,
        familiar_id: &str,
        channel_id: i64,
        catch_up_limit: usize,
    ) -> Result<Promotion, StoreError>;
}

#[async_trait]
impl FocusStore for AsyncHistoryStore {
    async fn get_focus_pointers(
        &self,
        familiar_id: &str,
    ) -> Result<Option<FocusPointers>, StoreError> {
        Self::get_focus_pointers(self, familiar_id.to_owned()).await
    }

    async fn set_focus_pointers(
        &self,
        familiar_id: &str,
        text_channel_id: Option<i64>,
        voice_channel_id: Option<i64>,
    ) -> Result<(), StoreError> {
        Self::set_focus_pointers(
            self,
            familiar_id.to_owned(),
            text_channel_id,
            voice_channel_id,
        )
        .await
    }

    async fn promote_staged_turns(
        &self,
        familiar_id: &str,
        channel_id: i64,
        catch_up_limit: usize,
    ) -> Result<Promotion, StoreError> {
        Self::promote_staged_turns(
            self,
            familiar_id.to_owned(),
            channel_id,
            Some(catch_up_limit),
        )
        .await
    }
}

/// Interior mutable focus state (single-mutex design; DESIGN port notes).
struct FocusState {
    text_focus: Option<i64>,
    voice_focus: Option<i64>,
    channel_names: HashMap<i64, String>,
    guild_names: HashMap<i64, String>,
    last_nudge: f64,
}

/// Per-familiar attentional focus controller.
pub struct FocusManager {
    familiar_id: String,
    store: Arc<dyn FocusStore>,
    subscriptions: Arc<dyn SubscriptionView>,
    clock: Clock,
    unread_nudge_enabled: bool,
    nudge_debounce_seconds: f64,
    catch_up_limit: usize,
    state: Mutex<FocusState>,
    on_shift: Mutex<Option<OnShift>>,
}

fn monotonic_clock() -> Clock {
    let start = Instant::now();
    Arc::new(move || start.elapsed().as_secs_f64())
}

impl FocusManager {
    /// New manager with default clock, nudge, and catch-up settings.
    #[must_use]
    pub fn new(
        familiar_id: impl Into<String>,
        store: Arc<dyn FocusStore>,
        subscriptions: Arc<dyn SubscriptionView>,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            store,
            subscriptions,
            clock: monotonic_clock(),
            unread_nudge_enabled: true,
            nudge_debounce_seconds: DEFAULT_NUDGE_DEBOUNCE_S,
            catch_up_limit: DEFAULT_CATCH_UP_LIMIT,
            state: Mutex::new(FocusState {
                text_focus: None,
                voice_focus: None,
                channel_names: HashMap::new(),
                guild_names: HashMap::new(),
                last_nudge: f64::NEG_INFINITY,
            }),
            on_shift: Mutex::new(None),
        }
    }

    /// Inject a clock (default `Instant`-based monotonic seconds).
    #[must_use]
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// Enable / disable the unread nudge.
    #[must_use]
    pub const fn with_unread_nudge_enabled(mut self, enabled: bool) -> Self {
        self.unread_nudge_enabled = enabled;
        self
    }

    /// Set the nudge debounce window (seconds).
    #[must_use]
    pub const fn with_nudge_debounce_seconds(mut self, seconds: f64) -> Self {
        self.nudge_debounce_seconds = seconds;
        self
    }

    /// Set the catch-up window.
    #[must_use]
    pub const fn with_catch_up_limit(mut self, limit: usize) -> Self {
        self.catch_up_limit = limit;
        self
    }

    /// Load persisted focus pointers, dropping any no-longer-subscribed pointer.
    pub async fn initialize(&self) {
        let ptrs = self
            .store
            .get_focus_pointers(&self.familiar_id)
            .await
            .ok()
            .flatten();
        if let Some(ptrs) = ptrs {
            let text = self.keep_if_subscribed(ptrs.text_channel_id);
            let voice = self.keep_if_subscribed(ptrs.voice_channel_id);
            {
                let mut state = self.state.lock().expect("focus state mutex");
                state.text_focus = text;
                state.voice_focus = voice;
            }
            tracing::info!(
                "{} loaded {} {}",
                ls::tag("Focus", ls::LC),
                ls::kv_styled("text", &self.channel_label(text), ls::W, ls::LW),
                ls::kv_styled("voice", &self.channel_label(voice), ls::W, ls::LW),
            );
        }
    }

    fn keep_if_subscribed(&self, channel_id: Option<i64>) -> Option<i64> {
        match channel_id {
            None => None,
            Some(cid) if self.is_subscribed(cid) => Some(cid),
            Some(cid) => {
                tracing::warn!(
                    "{} dropped stale pointer {} (no longer subscribed)",
                    ls::tag("Focus", ls::LC),
                    ls::kv_styled("channel", &self.channel_label(Some(cid)), ls::W, ls::LW),
                );
                None
            }
        }
    }

    /// The staged-turn catch-up window (also the `shift_focus` preview size).
    #[must_use]
    pub const fn catch_up_limit(&self) -> usize {
        self.catch_up_limit
    }

    /// Current focus channel for `modality` (`"text"` → text pointer; any other
    /// string → voice pointer).
    #[must_use]
    pub fn get_focus(&self, modality: &str) -> Option<i64> {
        let state = self.state.lock().expect("focus state mutex");
        if modality == "text" {
            state.text_focus
        } else {
            state.voice_focus
        }
    }

    /// Whether `channel_id` is a known text/voice subscription.
    #[must_use]
    pub fn is_subscribed(&self, channel_id: i64) -> bool {
        u64::try_from(channel_id)
            .ok()
            .and_then(|cid| self.subscriptions.kind_for(cid))
            .is_some()
    }

    /// Sorted, deduped channel ids across text + voice subscriptions.
    #[must_use]
    pub fn subscribed_channels(&self) -> Vec<i64> {
        let mut set: BTreeSet<i64> = BTreeSet::new();
        for sub in self.subscriptions.all() {
            set.insert(i64::try_from(sub.channel_id).unwrap_or(i64::MAX));
        }
        set.into_iter().collect()
    }

    /// Whether `channel_id` is the active text or voice focus.
    #[must_use]
    pub fn is_focused(&self, channel_id: i64) -> bool {
        let state = self.state.lock().expect("focus state mutex");
        Some(channel_id) == state.text_focus || Some(channel_id) == state.voice_focus
    }

    /// Apply a focus shift immediately (at tool-call time).
    ///
    /// Modality is inferred from subscriptions — `voice` kind → voice; anything
    /// else (including unsubscribed) → text. A text shift promotes the target's
    /// staged backlog; both modalities move the pointer and persist both. The
    /// `on_shift` hook is awaited once, after the state lock is released.
    pub async fn shift_now(&self, channel_id: i64) {
        let kind = u64::try_from(channel_id)
            .ok()
            .and_then(|cid| self.subscriptions.kind_for(cid));
        let is_voice = kind == Some(SubscriptionKind::Voice);

        if is_voice {
            let (text, voice) = {
                let mut state = self.state.lock().expect("focus state mutex");
                state.voice_focus = Some(channel_id);
                (state.text_focus, state.voice_focus)
            };
            let _ = self
                .store
                .set_focus_pointers(&self.familiar_id, text, voice)
                .await;
            tracing::info!(
                "{} {}",
                ls::tag("\u{1f500} Focus", ls::LC),
                ls::kv_styled(
                    "voice",
                    &self.channel_label(Some(channel_id)),
                    ls::W,
                    ls::LW
                ),
            );
        } else {
            let promo = self
                .store
                .promote_staged_turns(&self.familiar_id, channel_id, self.catch_up_limit)
                .await
                .unwrap_or(Promotion {
                    consumed: 0,
                    missed: 0,
                });
            let (text, voice) = {
                let mut state = self.state.lock().expect("focus state mutex");
                state.text_focus = Some(channel_id);
                (state.text_focus, state.voice_focus)
            };
            let _ = self
                .store
                .set_focus_pointers(&self.familiar_id, text, voice)
                .await;
            tracing::info!(
                "{} {} {} {}",
                ls::tag("\u{1f500} Focus", ls::LC),
                ls::kv_styled("text", &self.channel_label(Some(channel_id)), ls::W, ls::LW),
                ls::kv_styled("promoted", &promo.consumed.to_string(), ls::W, ls::LG),
                ls::kv_styled("missed", &promo.missed.to_string(), ls::W, ls::LY),
            );
        }

        let hook = self.on_shift.lock().expect("on_shift mutex").clone();
        if let Some(hook) = hook {
            hook().await;
        }
    }

    /// Whether a non-focused arrival warrants a nudge (behavior 51).
    #[must_use]
    pub fn should_wake(&self, channel_id: i64) -> bool {
        if !self.unread_nudge_enabled {
            return false;
        }
        let now = (self.clock)();
        let state = self.state.lock().expect("focus state mutex");
        let focused = Some(channel_id) == state.text_focus || Some(channel_id) == state.voice_focus;
        if focused {
            return false;
        }
        (now - state.last_nudge) >= self.nudge_debounce_seconds
    }

    /// Record a nudge timestamp to start the debounce window.
    pub fn mark_nudge_pending(&self) {
        let now = (self.clock)();
        self.state.lock().expect("focus state mutex").last_nudge = now;
    }

    /// Responder end-of-turn hook — intentionally a no-op (behavior 50).
    #[allow(
        clippy::unused_async,
        reason = "kept async so both responders can await it uniformly"
    )]
    pub async fn end_turn(&self) {}

    /// Format a channel id as `#name(id)` or `#id` (`"none"` for `None`).
    #[must_use]
    pub fn channel_label(&self, channel_id: Option<i64>) -> String {
        channel_id.map_or_else(
            || "none".to_owned(),
            |cid| {
                let state = self.state.lock().expect("focus state mutex");
                state
                    .channel_names
                    .get(&cid)
                    .map_or_else(|| format!("#{cid}"), |name| format!("#{name}({cid})"))
            },
        )
    }

    /// Server name for `channel_id`; `None` for `None` input or unknown channel.
    #[must_use]
    pub fn guild_name_for(&self, channel_id: Option<i64>) -> Option<String> {
        let cid = channel_id?;
        self.state
            .lock()
            .expect("focus state mutex")
            .guild_names
            .get(&cid)
            .cloned()
    }

    /// Guild name for the current text focus; `None` when unset or unknown.
    #[must_use]
    pub fn presence_guild(&self) -> Option<String> {
        let state = self.state.lock().expect("focus state mutex");
        let cid = state.text_focus?;
        state.guild_names.get(&cid).cloned()
    }

    /// `#channel-name` (or `#<id>`) for the current text focus; `None` when unset.
    #[must_use]
    pub fn presence_text(&self) -> Option<String> {
        let state = self.state.lock().expect("focus state mutex");
        let cid = state.text_focus?;
        let name = state
            .channel_names
            .get(&cid)
            .cloned()
            .unwrap_or_else(|| cid.to_string());
        drop(state);
        Some(format!("#{name}"))
    }

    /// Seed a focus pointer without deferral, promotion, or persistence (startup).
    pub fn set_focus_immediately(&self, channel_id: i64, modality: &str) {
        {
            let mut state = self.state.lock().expect("focus state mutex");
            if modality == "text" {
                state.text_focus = Some(channel_id);
            } else {
                state.voice_focus = Some(channel_id);
            }
        }
        tracing::info!(
            "{} default {}",
            ls::tag("Focus", ls::LC),
            ls::kv_styled(
                modality,
                &self.channel_label(Some(channel_id)),
                ls::W,
                ls::LW
            ),
        );
    }

    /// Snapshot of the channel→display-name map (for the final-reminder digest).
    #[must_use]
    pub fn channel_names(&self) -> HashMap<i64, String> {
        self.state
            .lock()
            .expect("focus state mutex")
            .channel_names
            .clone()
    }

    /// Snapshot of the channel→server-name map (for the final-reminder digest;
    /// a [`PRIVATE_MESSAGE_GUILD_NAME`] entry marks a DM).
    #[must_use]
    pub fn guild_names(&self) -> HashMap<i64, String> {
        self.state
            .lock()
            .expect("focus state mutex")
            .guild_names
            .clone()
    }

    /// Set a channel's display name (populated by the Discord shell on ready).
    pub fn set_channel_name(&self, channel_id: i64, name: impl Into<String>) {
        self.state
            .lock()
            .expect("focus state mutex")
            .channel_names
            .insert(channel_id, name.into());
    }

    /// Set a channel's guild name (populated by the Discord shell on ready).
    pub fn set_guild_name(&self, channel_id: i64, name: impl Into<String>) {
        self.state
            .lock()
            .expect("focus state mutex")
            .guild_names
            .insert(channel_id, name.into());
    }

    /// Install the presence-refresh hook awaited after each shift.
    pub fn set_on_shift(&self, hook: OnShift) {
        *self.on_shift.lock().expect("on_shift mutex") = Some(hook);
    }

    /// Whether an `on_shift` hook is installed.
    #[must_use]
    pub fn has_on_shift(&self) -> bool {
        self.on_shift.lock().expect("on_shift mutex").is_some()
    }
}
