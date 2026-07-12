//! Typing-event policy: interruption + bot-pingpong backoff (subsystem 06;
//! Python `typing_interrupt.py`).
//!
//! Discord `on_typing` fires when any user (or bot) starts typing in a channel.
//! [`TypingInterruptHandler`] translates events into:
//!
//! * an immediate [`TurnRouter`] cancellation when a real user is typing
//!   (`[discord.text].respond_to_typing` ON; default), so the bot stops
//!   generating instead of speaking over the user;
//! * exponential backoff when *another bot* is typing — protects against
//!   pingpong with another familiar-connect instance whose typing indicator we
//!   would otherwise mirror by replying.
//!
//! Wired by the Discord shell (`on_typing`); queried by the
//! [`TextResponder`](crate::processors::text_responder::TextResponder) before
//! assembling a reply.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use tokio::time::{Duration, Instant};

use crate::bus::router::TurnRouter;
use crate::config::DiscordTextConfig;
use crate::log_style as ls;

/// Predicate: is `channel_id` a subscribed channel?
pub type IsSubscribed = Arc<dyn Fn(i64) -> bool + Send + Sync>;
/// Provider for the bot's own Discord user id (`None` before the gateway is up).
pub type BotUserIdProvider = Arc<dyn Fn() -> Option<i64> + Send + Sync>;

/// Per-channel typing-event policy.
///
/// Stateless across processes — the backoff ladder lives in memory and resets
/// on user activity. Safe to share across the async runtime (all interior state
/// is behind mutexes; the Python original relied on the single event loop).
pub struct TypingInterruptHandler {
    config: DiscordTextConfig,
    router: Arc<TurnRouter>,
    is_subscribed: IsSubscribed,
    bot_user_id: BotUserIdProvider,
    /// Value to apply on the *next* bot-typing event; doubles after each.
    next_backoff_s: Mutex<HashMap<i64, f64>>,
    /// Monotonic clock time each channel is parked until.
    deadline: Mutex<HashMap<i64, Instant>>,
    /// Last window actually applied per channel (test seam; survives until the
    /// next user message resets the ladder, then until the next apply).
    applied: Mutex<HashMap<i64, f64>>,
}

impl TypingInterruptHandler {
    /// Construct a handler over `config`, cancelling turns on `router`.
    #[must_use]
    pub fn new(
        config: DiscordTextConfig,
        router: Arc<TurnRouter>,
        is_subscribed: IsSubscribed,
        bot_user_id: BotUserIdProvider,
    ) -> Self {
        Self {
            config,
            router,
            is_subscribed,
            bot_user_id,
            next_backoff_s: Mutex::new(HashMap::new()),
            deadline: Mutex::new(HashMap::new()),
            applied: Mutex::new(HashMap::new()),
        }
    }

    /// Translate a Discord `on_typing` event into policy actions.
    pub fn notify_typing(&self, channel_id: i64, user_id: i64, is_bot: bool) {
        if !self.config.respond_to_typing {
            return;
        }
        if !(self.is_subscribed)(channel_id) {
            return;
        }
        if let Some(bot_id) = (self.bot_user_id)() {
            if user_id == bot_id {
                return;
            }
        }
        if is_bot {
            self.apply_bot_backoff(channel_id);
            return;
        }
        self.cancel_active_turn(channel_id, user_id);
    }

    /// Reset the backoff ladder — a real user message means the lane is live.
    pub fn notify_user_message(&self, channel_id: i64) {
        self.next_backoff_s
            .lock()
            .expect("typing next_backoff mutex")
            .remove(&channel_id);
        self.deadline
            .lock()
            .expect("typing deadline mutex")
            .remove(&channel_id);
    }

    /// Monotonic deadline when parked; `None` when free (lazily expires).
    #[must_use]
    pub fn backoff_deadline(&self, channel_id: i64) -> Option<Instant> {
        let mut deadlines = self.deadline.lock().expect("typing deadline mutex");
        let deadline = *deadlines.get(&channel_id)?;
        if deadline <= Instant::now() {
            deadlines.remove(&channel_id);
            return None;
        }
        drop(deadlines);
        Some(deadline)
    }

    /// Backoff window most recently applied to `channel_id` (`0.0` when never
    /// applied). Test seam — production callers use [`wait_for_backoff`](Self::wait_for_backoff).
    #[must_use]
    pub fn current_backoff_s(&self, channel_id: i64) -> f64 {
        self.applied
            .lock()
            .expect("typing applied mutex")
            .get(&channel_id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Sleep until the channel's backoff deadline (no-op when idle).
    pub async fn wait_for_backoff(&self, channel_id: i64) {
        let Some(deadline) = self.backoff_deadline(channel_id) else {
            return;
        };
        let now = Instant::now();
        if deadline <= now {
            return;
        }
        let delay = deadline - now;
        tracing::info!(
            "{} {} {}",
            ls::tag("\u{1f4ac} Text", ls::B),
            ls::kv_styled(
                "typing_backoff",
                &format!("{:.2}s", delay.as_secs_f64()),
                ls::W,
                ls::LB
            ),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
        );
        tokio::time::sleep(delay).await;
    }

    fn apply_bot_backoff(&self, channel_id: i64) {
        let next_s = self
            .next_backoff_s
            .lock()
            .expect("typing next_backoff mutex")
            .get(&channel_id)
            .copied()
            .unwrap_or(self.config.typing_backoff_initial_s);
        let applied = next_s.min(self.config.typing_backoff_max_s);
        self.applied
            .lock()
            .expect("typing applied mutex")
            .insert(channel_id, applied);
        self.deadline.lock().expect("typing deadline mutex").insert(
            channel_id,
            Instant::now() + Duration::from_secs_f64(applied),
        );
        // Double for the next event; cap on read so the ladder can't overflow.
        self.next_backoff_s
            .lock()
            .expect("typing next_backoff mutex")
            .insert(
                channel_id,
                (applied * 2.0).min(self.config.typing_backoff_max_s),
            );
        tracing::info!(
            "{} {} {}",
            ls::tag("\u{1f4ac} Text", ls::Y),
            ls::kv_styled("bot_typing", &format!("{applied:.2}s"), ls::W, ls::LY),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
        );
    }

    fn cancel_active_turn(&self, channel_id: i64, user_id: i64) {
        let session_id = format!("discord:{channel_id}");
        let Some(scope) = self.router.active_scope(&session_id) else {
            return;
        };
        scope.cancel();
        tracing::info!(
            "{} {} {} {}",
            ls::tag("\u{1f4ac} Text", ls::Y),
            ls::kv_styled("typing_cancel", "user", ls::W, ls::LY),
            ls::kv_styled("user", &user_id.to_string(), ls::W, ls::LC),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{BotUserIdProvider, IsSubscribed, TypingInterruptHandler};
    use crate::bus::router::TurnRouter;
    use crate::config::DiscordTextConfig;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;

    fn make_handler(
        config: Option<DiscordTextConfig>,
        bot_user_id: Option<i64>,
        subscribed: Option<HashSet<i64>>,
        router: Option<Arc<TurnRouter>>,
    ) -> (TypingInterruptHandler, Arc<TurnRouter>) {
        let config = config.unwrap_or_default();
        let router = router.unwrap_or_else(|| Arc::new(TurnRouter::new()));
        let subscribed = subscribed.unwrap_or_else(|| HashSet::from([42]));
        let is_subscribed: IsSubscribed = Arc::new(move |ch| subscribed.contains(&ch));
        let bot_provider: BotUserIdProvider = Arc::new(move || bot_user_id);
        let handler =
            TypingInterruptHandler::new(config, Arc::clone(&router), is_subscribed, bot_provider);
        (handler, router)
    }

    #[tokio::test]
    async fn user_typing_cancels_scope() {
        let (handler, router) = make_handler(None, Some(999), None, None);
        let scope = router.begin_turn("discord:42", "t-1");
        handler.notify_typing(42, 7, false);
        assert!(scope.is_cancelled());
    }

    #[tokio::test]
    async fn no_active_scope_is_noop() {
        let (handler, _) = make_handler(None, Some(999), None, None);
        handler.notify_typing(42, 7, false); // nothing raised
    }

    #[tokio::test]
    async fn bot_self_typing_ignored() {
        let (handler, router) = make_handler(None, Some(999), None, None);
        let scope = router.begin_turn("discord:42", "t-1");
        handler.notify_typing(42, 999, true);
        assert!(!scope.is_cancelled());
    }

    #[tokio::test]
    async fn unsubscribed_channel_ignored() {
        let (handler, router) = make_handler(None, Some(999), Some(HashSet::new()), None);
        let scope = router.begin_turn("discord:42", "t-1");
        handler.notify_typing(42, 7, false);
        assert!(!scope.is_cancelled());
    }

    #[tokio::test]
    async fn disabled_via_config() {
        let cfg = DiscordTextConfig {
            respond_to_typing: false,
            ..Default::default()
        };
        let (handler, router) = make_handler(Some(cfg), Some(999), None, None);
        let scope = router.begin_turn("discord:42", "t-1");
        handler.notify_typing(42, 7, false);
        assert!(!scope.is_cancelled());
    }

    #[tokio::test(start_paused = true)]
    async fn first_bot_typing_sets_initial_backoff() {
        let cfg = DiscordTextConfig {
            typing_backoff_initial_s: 2.0,
            typing_backoff_max_s: 8.0,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        handler.notify_typing(42, 123, true);
        // A deadline exists and the applied window is the initial 2.0s.
        assert!(handler.backoff_deadline(42).is_some());
        assert!((handler.current_backoff_s(42) - 2.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn repeated_bot_typing_doubles_backoff() {
        let cfg = DiscordTextConfig {
            typing_backoff_initial_s: 1.0,
            typing_backoff_max_s: 16.0,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        handler.notify_typing(42, 123, true);
        let first = handler.current_backoff_s(42);
        handler.notify_typing(42, 123, true);
        let second = handler.current_backoff_s(42);
        handler.notify_typing(42, 123, true);
        let third = handler.current_backoff_s(42);
        assert!((first - 1.0).abs() < 1e-9);
        assert!((second - 2.0).abs() < 1e-9);
        assert!((third - 4.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn backoff_caps_at_max() {
        let cfg = DiscordTextConfig {
            typing_backoff_initial_s: 4.0,
            typing_backoff_max_s: 8.0,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        for _ in 0..5 {
            handler.notify_typing(42, 123, true);
        }
        assert!((handler.current_backoff_s(42) - 8.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn user_message_resets_backoff() {
        let cfg = DiscordTextConfig {
            typing_backoff_initial_s: 1.0,
            typing_backoff_max_s: 16.0,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        handler.notify_typing(42, 123, true);
        handler.notify_typing(42, 123, true);
        assert!((handler.current_backoff_s(42) - 2.0).abs() < 1e-9);
        handler.notify_user_message(42);
        handler.notify_typing(42, 123, true);
        assert!((handler.current_backoff_s(42) - 1.0).abs() < 1e-9);
    }

    #[tokio::test(start_paused = true)]
    async fn wait_for_backoff_sleeps_until_deadline() {
        let cfg = DiscordTextConfig {
            typing_backoff_initial_s: 0.05,
            typing_backoff_max_s: 0.1,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        handler.notify_typing(42, 123, true);
        let start = tokio::time::Instant::now();
        handler.wait_for_backoff(42).await;
        assert!(start.elapsed() >= Duration::from_secs_f64(0.04));
    }

    #[tokio::test(start_paused = true)]
    async fn wait_for_backoff_returns_immediately_when_idle() {
        let (handler, _) = make_handler(None, Some(999), None, None);
        let start = tokio::time::Instant::now();
        handler.wait_for_backoff(42).await;
        assert!(start.elapsed() < Duration::from_secs_f64(0.05));
    }

    #[tokio::test]
    async fn disabled_via_config_skips_backoff() {
        let cfg = DiscordTextConfig {
            respond_to_typing: false,
            ..Default::default()
        };
        let (handler, _) = make_handler(Some(cfg), Some(999), None, None);
        handler.notify_typing(42, 123, true);
        assert!(handler.backoff_deadline(42).is_none());
    }
}
