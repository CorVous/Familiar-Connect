//! Twitch EventSub WebSocket watcher (subsystem 11; Python `twitch_watcher.py`).
//!
//! Converts EventSub callbacks (follows, subs, gift subs, resubs, cheers,
//! channel-point redemptions, ad breaks) into normalized [`TwitchEvent`] values
//! on a channel. Gated behind the `twitch` feature — the whole pipeline is
//! dormant (no production wiring constructs the watcher), so the concrete
//! `twitch_api` EventSub WS session is deferred; this module ports the pure
//! handler logic, the listener-registration matrix, and the run/stop lifecycle
//! against an abstract [`EventSub`] seam (the Python code duck-typed against
//! twitchAPI's `EventSubWebsocket`; tests pass mocks).
//!
//! Rust cancellation model (DESIGN §4.4): the Python `run` slept on
//! `asyncio.Event().wait()` with a `finally: await eventsub.stop()`. Rust has no
//! async `Drop`, so [`TwitchWatcher::run`] takes a
//! [`CancellationToken`](tokio_util::sync::CancellationToken) instead — it
//! `start()`s, waits on the token, then `stop()`s. Task-abort alone would skip
//! the stop, so callers must cancel the token to unwind cleanly.
#![cfg(feature = "twitch")]

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::identity::Author;
use crate::twitch::{
    TwitchEvent, TwitchWatcherConfig, build_ad_start_event, build_channel_point_event,
    build_cheer_event, build_follow_event, build_gift_subscription_event,
    build_resubscription_event, build_subscription_event,
};

// ---------------------------------------------------------------------------
// Duck-typed EventSub data objects (mirror the twitchAPI field shapes)
// ---------------------------------------------------------------------------

/// The three `user_*` fields every twitchAPI event data object carries.
#[derive(Clone, Debug, Default)]
pub struct UserFields {
    /// Immutable numeric id.
    pub user_id: Option<String>,
    /// Immutable lowercase login.
    pub user_login: Option<String>,
    /// Mutable display name.
    pub user_name: Option<String>,
}

/// `ChannelFollowData` fields.
#[derive(Clone, Debug, Default)]
pub struct FollowData {
    /// Follower identity.
    pub user: UserFields,
}

/// `ChannelSubscribeData` fields.
#[derive(Clone, Debug, Default)]
pub struct SubscribeData {
    /// Subscriber identity.
    pub user: UserFields,
    /// Tier string (`"1000"`/`"2000"`/`"3000"`).
    pub tier: String,
    /// Whether the sub arrived as a gift (handled by the gift handler instead).
    pub is_gift: bool,
}

/// `ChannelSubscriptionGiftData` fields.
#[derive(Clone, Debug, Default)]
pub struct GiftSubData {
    /// Gifter identity (ignored when anonymous).
    pub user: UserFields,
    /// Number of subs gifted.
    pub total: i64,
    /// Tier string.
    pub tier: String,
    /// Whether the gifter is anonymous.
    pub is_anonymous: bool,
}

/// `ChannelSubscriptionMessageData` fields.
#[derive(Clone, Debug, Default)]
pub struct ResubData {
    /// Resubscriber identity.
    pub user: UserFields,
    /// Cumulative months, when present (preferred).
    pub cumulative_months: Option<i64>,
    /// Duration months (always present; the fallback).
    pub duration_months: i64,
    /// Tier string.
    pub tier: String,
    /// The resub message text (nested `message.text` on twitchAPI).
    pub message_text: String,
}

/// `ChannelCheerData` fields.
#[derive(Clone, Debug, Default)]
pub struct CheerData {
    /// Cheerer identity (ignored when anonymous).
    pub user: UserFields,
    /// Bits cheered.
    pub bits: i64,
    /// The cheer message.
    pub message: String,
    /// Whether the cheerer is anonymous.
    pub is_anonymous: bool,
}

/// `ChannelPointsCustomRewardRedemptionData` fields.
#[derive(Clone, Debug, Default)]
pub struct RedemptionData {
    /// Redeemer identity.
    pub user: UserFields,
    /// The redeemed reward's title (nested `reward.title` on twitchAPI).
    pub reward_title: String,
    /// Optional viewer input (empty coerces to no `and says:` clause).
    pub user_input: String,
}

/// `ChannelAdBreakBeginData` fields (unused by the handler).
#[derive(Clone, Debug, Default)]
pub struct AdBreakData {
    /// Ad break length in seconds.
    pub duration_seconds: i64,
}

/// twitchAPI wraps each data object in a `.event` field (`SimpleNamespace(event=...)`).
#[derive(Clone, Debug)]
pub struct EventWrapper<T> {
    /// The wrapped data object.
    pub event: T,
}

// ---------------------------------------------------------------------------
// Tier mapping
// ---------------------------------------------------------------------------

fn tier(tier_str: &str) -> u32 {
    match tier_str {
        "2000" => 2,
        "3000" => 3,
        // "1000" and any unknown string map to 1.
        _ => 1,
    }
}

fn build_author(user: &UserFields) -> Author {
    // Author is immutably keyed on user_id so repeat viewers resolve to the same
    // `people/twitch-<id>.md`. Optional fields coerce to "" — anonymous-capable
    // flows guard before calling.
    Author::from_twitch(
        user.user_id.clone().unwrap_or_default(),
        Some(user.user_login.clone().unwrap_or_default()),
        Some(user.user_name.clone().unwrap_or_default()),
    )
}

// ---------------------------------------------------------------------------
// Callback seam
// ---------------------------------------------------------------------------

/// Boxed async callback for a follow event.
pub type FollowCb = Box<dyn Fn(EventWrapper<FollowData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for a subscription event.
pub type SubscribeCb =
    Box<dyn Fn(EventWrapper<SubscribeData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for a gift-subscription event.
pub type GiftCb = Box<dyn Fn(EventWrapper<GiftSubData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for a resubscription event.
pub type ResubCb = Box<dyn Fn(EventWrapper<ResubData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for a cheer event.
pub type CheerCb = Box<dyn Fn(EventWrapper<CheerData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for an ad-break event.
pub type AdCb = Box<dyn Fn(EventWrapper<AdBreakData>) -> BoxFuture<'static, ()> + Send + Sync>;
/// Boxed async callback for a channel-point redemption event.
pub type RedemptionCb =
    Box<dyn Fn(EventWrapper<RedemptionData>) -> BoxFuture<'static, ()> + Send + Sync>;

/// The subset of twitchAPI's `EventSubWebsocket` the watcher registers against.
///
/// Real EventSub glue (the `twitch_api` crate + `tokio-tungstenite`) would
/// implement this; tests pass mocks that record which `listen_*` were called.
#[async_trait]
pub trait EventSub: Send + Sync {
    /// Register the follow (v2) listener.
    async fn listen_channel_follow_v2(
        &self,
        broadcaster_id: String,
        moderator_id: String,
        cb: FollowCb,
    );
    /// Register the new-subscription listener.
    async fn listen_channel_subscribe(&self, broadcaster_id: String, cb: SubscribeCb);
    /// Register the gift-subscription listener.
    async fn listen_channel_subscription_gift(&self, broadcaster_id: String, cb: GiftCb);
    /// Register the resubscription-message listener.
    async fn listen_channel_subscription_message(&self, broadcaster_id: String, cb: ResubCb);
    /// Register the cheer listener.
    async fn listen_channel_cheer(&self, broadcaster_id: String, cb: CheerCb);
    /// Register the ad-break-begin listener.
    async fn listen_channel_ad_break_begin(&self, broadcaster_id: String, cb: AdCb);
    /// Register the channel-point redemption listener.
    async fn listen_channel_points_custom_reward_redemption_add(
        &self,
        broadcaster_id: String,
        cb: RedemptionCb,
    );
    /// Open the EventSub connection.
    async fn start(&self);
    /// Close the EventSub connection.
    async fn stop(&self);
}

// ---------------------------------------------------------------------------
// Watcher
// ---------------------------------------------------------------------------

/// Converts EventSub callbacks into [`TwitchEvent`] values.
pub struct TwitchWatcher {
    /// Which events produce messages.
    pub config: TwitchWatcherConfig,
    /// Broadcaster id (first arg to every `listen_*`).
    pub broadcaster_id: String,
    /// Channel name stamped on every event.
    pub channel: String,
    /// Moderator id for the follow-v2 listener (defaults to `broadcaster_id`).
    pub moderator_id: String,
}

impl TwitchWatcher {
    /// Build a watcher; `moderator_id` falls back to `broadcaster_id`.
    #[must_use]
    pub fn new(
        config: TwitchWatcherConfig,
        broadcaster_id: impl Into<String>,
        channel: impl Into<String>,
        moderator_id: Option<String>,
    ) -> Self {
        let broadcaster_id = broadcaster_id.into();
        let moderator_id = moderator_id.unwrap_or_else(|| broadcaster_id.clone());
        Self {
            config,
            broadcaster_id,
            channel: channel.into(),
            moderator_id,
        }
    }

    // -- synchronous handlers (pure given the data object) ------------------

    /// Follow → event (or `None` when follows disabled).
    #[must_use]
    pub fn handle_follow(&self, data: &FollowData) -> Option<TwitchEvent> {
        build_follow_event(&self.config, &self.channel, build_author(&data.user))
    }

    /// New subscription → event; `None` for gift subs (they have their own
    /// handler) or when subscriptions are disabled.
    #[must_use]
    pub fn handle_subscription(&self, data: &SubscribeData) -> Option<TwitchEvent> {
        if data.is_gift {
            return None;
        }
        build_subscription_event(
            &self.config,
            &self.channel,
            build_author(&data.user),
            tier(&data.tier),
        )
    }

    /// Gift subscription → event (or `None` when subscriptions disabled).
    #[must_use]
    pub fn handle_gift_subscription(&self, data: &GiftSubData) -> Option<TwitchEvent> {
        let gifter = if data.is_anonymous {
            None
        } else {
            Some(build_author(&data.user))
        };
        build_gift_subscription_event(
            &self.config,
            &self.channel,
            gifter,
            u32::try_from(data.total.max(0)).unwrap_or(0),
            tier(&data.tier),
        )
    }

    /// Resubscription → event (or `None` when subscriptions disabled).
    ///
    /// Months = `cumulative_months` unless absent, then `duration_months`.
    #[must_use]
    pub fn handle_resubscription(&self, data: &ResubData) -> Option<TwitchEvent> {
        let months = data.cumulative_months.unwrap_or(data.duration_months);
        build_resubscription_event(
            &self.config,
            &self.channel,
            build_author(&data.user),
            u32::try_from(months.max(0)).unwrap_or(0),
            tier(&data.tier),
            &data.message_text,
        )
    }

    /// Cheer → event (or `None` when cheers disabled).
    #[must_use]
    pub fn handle_cheer(&self, data: &CheerData) -> Option<TwitchEvent> {
        let viewer = if data.is_anonymous {
            None
        } else {
            Some(build_author(&data.user))
        };
        build_cheer_event(
            &self.config,
            &self.channel,
            viewer,
            u32::try_from(data.bits.max(0)).unwrap_or(0),
            &data.message,
        )
    }

    /// Channel-point redemption → event (or `None` when not in the allow-list).
    #[must_use]
    pub fn handle_channel_point_redemption(&self, data: &RedemptionData) -> Option<TwitchEvent> {
        // `data.user_input or None` — empty input drops the "and says:" clause.
        let user_input = if data.user_input.is_empty() {
            None
        } else {
            Some(data.user_input.as_str())
        };
        build_channel_point_event(
            &self.config,
            &self.channel,
            build_author(&data.user),
            &data.reward_title,
            user_input,
        )
    }

    /// Ad-break begin → event (or `None` when ads disabled).
    #[must_use]
    pub fn handle_ad_break_begin(&self, _data: &AdBreakData) -> Option<TwitchEvent> {
        build_ad_start_event(&self.config, &self.channel)
    }

    // -- listener registration ---------------------------------------------

    /// Register EventSub callbacks for every enabled event type.
    ///
    /// `send` is the channel each callback forwards events to (always provided
    /// from [`run`](Self::run); `None` for standalone registration tests). A
    /// `None` handler result (disabled/filtered) or a `None` `send` forwards
    /// nothing.
    pub async fn register_listeners(
        self: &Arc<Self>,
        eventsub: &dyn EventSub,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) {
        if self.config.follows_enabled {
            eventsub
                .listen_channel_follow_v2(
                    self.broadcaster_id.clone(),
                    self.moderator_id.clone(),
                    self.follow_callback(send.clone()),
                )
                .await;
        }
        if self.config.subscriptions_enabled {
            eventsub
                .listen_channel_subscribe(
                    self.broadcaster_id.clone(),
                    self.subscription_callback(send.clone()),
                )
                .await;
            eventsub
                .listen_channel_subscription_gift(
                    self.broadcaster_id.clone(),
                    self.gift_subscription_callback(send.clone()),
                )
                .await;
            eventsub
                .listen_channel_subscription_message(
                    self.broadcaster_id.clone(),
                    self.resubscription_callback(send.clone()),
                )
                .await;
        }
        if self.config.cheers_enabled {
            eventsub
                .listen_channel_cheer(
                    self.broadcaster_id.clone(),
                    self.cheer_callback(send.clone()),
                )
                .await;
        }
        if self.config.ads_enabled {
            eventsub
                .listen_channel_ad_break_begin(
                    self.broadcaster_id.clone(),
                    self.ad_break_begin_callback(send.clone()),
                )
                .await;
        }
        if !self.config.redemption_names.is_empty() {
            eventsub
                .listen_channel_points_custom_reward_redemption_add(
                    self.broadcaster_id.clone(),
                    self.channel_point_redemption_callback(send.clone()),
                )
                .await;
        }
    }

    fn follow_callback(self: &Arc<Self>, send: Option<UnboundedSender<TwitchEvent>>) -> FollowCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_follow(&ev.event), send.as_ref());
            })
        })
    }

    fn subscription_callback(
        self: &Arc<Self>,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) -> SubscribeCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_subscription(&ev.event), send.as_ref());
            })
        })
    }

    fn gift_subscription_callback(
        self: &Arc<Self>,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) -> GiftCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_gift_subscription(&ev.event), send.as_ref());
            })
        })
    }

    fn resubscription_callback(
        self: &Arc<Self>,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) -> ResubCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_resubscription(&ev.event), send.as_ref());
            })
        })
    }

    fn cheer_callback(self: &Arc<Self>, send: Option<UnboundedSender<TwitchEvent>>) -> CheerCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_cheer(&ev.event), send.as_ref());
            })
        })
    }

    fn ad_break_begin_callback(
        self: &Arc<Self>,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) -> AdCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(watcher.handle_ad_break_begin(&ev.event), send.as_ref());
            })
        })
    }

    fn channel_point_redemption_callback(
        self: &Arc<Self>,
        send: Option<UnboundedSender<TwitchEvent>>,
    ) -> RedemptionCb {
        let watcher = Arc::clone(self);
        Box::new(move |ev| {
            let watcher = Arc::clone(&watcher);
            let send = send.clone();
            Box::pin(async move {
                send_if_present(
                    watcher.handle_channel_point_redemption(&ev.event),
                    send.as_ref(),
                );
            })
        })
    }

    /// Run the watcher: register listeners → `start()` → wait on `cancel` →
    /// `stop()`. (Python slept on `Event().wait()` with a `finally` stop; the
    /// Rust cancellation seam replaces both.)
    pub async fn run(
        self: &Arc<Self>,
        send: UnboundedSender<TwitchEvent>,
        eventsub: Arc<dyn EventSub>,
        cancel: CancellationToken,
    ) {
        self.register_listeners(eventsub.as_ref(), Some(send)).await;
        eventsub.start().await;
        cancel.cancelled().await;
        eventsub.stop().await;
    }
}

fn send_if_present(event: Option<TwitchEvent>, send: Option<&UnboundedSender<TwitchEvent>>) {
    if let (Some(event), Some(sender)) = (event, send) {
        let _ = sender.send(event);
    }
}

#[cfg(test)]
#[allow(clippy::default_trait_access)]
mod tests {
    use super::{
        AdBreakData, CheerData, EventSub, EventWrapper, FollowCb, FollowData, GiftSubData,
        RedemptionData, ResubData, SubscribeData, TwitchWatcher, UserFields,
    };
    use crate::identity::Author;
    use crate::twitch::{TwitchEvent, TwitchWatcherConfig};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc::{self, UnboundedReceiver};
    use tokio_util::sync::CancellationToken;

    // --- data builders -----------------------------------------------------

    fn user_fields(name: Option<&str>) -> UserFields {
        let user_id = name.map_or_else(|| "uid-anon".to_owned(), |n| format!("uid-{n}"));
        UserFields {
            user_id: Some(user_id),
            user_login: name.map(str::to_lowercase),
            user_name: name.map(ToOwned::to_owned),
        }
    }

    fn follow_data(name: &str) -> FollowData {
        FollowData {
            user: user_fields(Some(name)),
        }
    }

    fn subscribe_data(name: &str, tier: &str, is_gift: bool) -> SubscribeData {
        SubscribeData {
            user: user_fields(Some(name)),
            tier: tier.to_owned(),
            is_gift,
        }
    }

    fn gift_sub_data(
        name: Option<&str>,
        total: i64,
        tier: &str,
        is_anonymous: bool,
    ) -> GiftSubData {
        GiftSubData {
            user: user_fields(name),
            total,
            tier: tier.to_owned(),
            is_anonymous,
        }
    }

    fn resub_data(name: &str, cumulative_months: i64, tier: &str, message: &str) -> ResubData {
        ResubData {
            user: user_fields(Some(name)),
            cumulative_months: Some(cumulative_months),
            duration_months: 0,
            tier: tier.to_owned(),
            message_text: message.to_owned(),
        }
    }

    fn cheer_data(name: Option<&str>, bits: i64, message: &str, is_anonymous: bool) -> CheerData {
        CheerData {
            user: user_fields(name),
            bits,
            message: message.to_owned(),
            is_anonymous,
        }
    }

    fn redemption_data(name: &str, reward_title: &str, user_input: &str) -> RedemptionData {
        RedemptionData {
            user: user_fields(Some(name)),
            reward_title: reward_title.to_owned(),
            user_input: user_input.to_owned(),
        }
    }

    fn watcher(config: TwitchWatcherConfig) -> TwitchWatcher {
        TwitchWatcher::new(config, "1", "ch", None)
    }

    // --- construction ------------------------------------------------------

    #[test]
    fn stores_config_and_ids() {
        let cfg = TwitchWatcherConfig::default();
        let w = TwitchWatcher::new(cfg.clone(), "99999", "sapphire-stream", None);
        assert_eq!(w.config, cfg);
        assert_eq!(w.broadcaster_id, "99999");
        assert_eq!(w.channel, "sapphire-stream");
        assert_eq!(w.moderator_id, "99999");
    }

    #[test]
    fn moderator_id_can_be_overridden() {
        let w = TwitchWatcher::new(
            TwitchWatcherConfig::default(),
            "123",
            "ch",
            Some("456".to_owned()),
        );
        assert_eq!(w.moderator_id, "456");
    }

    // --- handle_follow -----------------------------------------------------

    #[test]
    fn follow_returns_event_when_enabled() {
        let w = watcher(TwitchWatcherConfig {
            follows_enabled: true,
            ..Default::default()
        });
        let e = w.handle_follow(&follow_data("Alice")).unwrap();
        assert_eq!(
            e.viewer.as_ref().unwrap().display_name.as_deref(),
            Some("Alice")
        );
        assert_eq!(e.viewer.as_ref().unwrap().platform, "twitch");
    }

    #[test]
    fn follow_returns_none_when_disabled() {
        let w = watcher(TwitchWatcherConfig {
            follows_enabled: false,
            ..Default::default()
        });
        assert!(w.handle_follow(&follow_data("Alice")).is_none());
    }

    #[test]
    fn follow_channel_is_set() {
        let w = TwitchWatcher::new(
            TwitchWatcherConfig {
                follows_enabled: true,
                ..Default::default()
            },
            "1",
            "my-stream",
            None,
        );
        assert_eq!(
            w.handle_follow(&follow_data("Alice")).unwrap().channel,
            "my-stream"
        );
    }

    // --- handle_subscription ----------------------------------------------

    #[test]
    fn subscription_enabled_and_disabled() {
        let on = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        assert!(
            on.handle_subscription(&subscribe_data("Alice", "1000", false))
                .is_some()
        );
        let off = watcher(TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        });
        assert!(
            off.handle_subscription(&subscribe_data("Alice", "1000", false))
                .is_none()
        );
    }

    #[test]
    fn subscription_none_for_gift_sub() {
        let w = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        assert!(
            w.handle_subscription(&subscribe_data("Alice", "1000", true))
                .is_none()
        );
    }

    #[test]
    fn subscription_tier_mapping() {
        let w = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        for (tier_str, label) in [("1000", "tier 1"), ("2000", "tier 2"), ("3000", "tier 3")] {
            let e = w
                .handle_subscription(&subscribe_data("Alice", tier_str, false))
                .unwrap();
            assert!(e.text.contains(label), "{tier_str}: {}", e.text);
        }
    }

    // --- handle_gift_subscription -----------------------------------------

    #[test]
    fn gift_sub_enabled_disabled_and_anon() {
        let on = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        assert!(
            on.handle_gift_subscription(&gift_sub_data(Some("Bob"), 5, "1000", false))
                .is_some()
        );
        let off = watcher(TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        });
        assert!(
            off.handle_gift_subscription(&gift_sub_data(Some("Bob"), 5, "1000", false))
                .is_none()
        );

        let anon = on
            .handle_gift_subscription(&gift_sub_data(None, 3, "1000", true))
            .unwrap();
        assert!(anon.text.to_lowercase().contains("anonymous"));
        assert!(anon.viewer.is_none());

        let named = on
            .handle_gift_subscription(&gift_sub_data(Some("Bob"), 5, "1000", false))
            .unwrap();
        assert_eq!(
            named.viewer.as_ref().unwrap().display_name.as_deref(),
            Some("Bob")
        );

        let t2 = on
            .handle_gift_subscription(&gift_sub_data(Some("Bob"), 5, "2000", false))
            .unwrap();
        assert!(t2.text.contains("tier 2"));
    }

    // --- handle_resubscription --------------------------------------------

    #[test]
    fn resub_enabled_disabled_and_text() {
        let on = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        assert!(
            on.handle_resubscription(&resub_data("Alice", 6, "2000", "love this stream"))
                .is_some()
        );
        let off = watcher(TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        });
        assert!(
            off.handle_resubscription(&resub_data("Alice", 6, "2000", "hi"))
                .is_none()
        );

        let e = on
            .handle_resubscription(&resub_data("Alice", 6, "2000", "love this stream"))
            .unwrap();
        assert_eq!(
            e.text,
            "Alice has subscribed for 6 months at tier 2 and says: love this stream"
        );
        let t3 = on
            .handle_resubscription(&resub_data("Alice", 6, "3000", "hi"))
            .unwrap();
        assert!(t3.text.contains("tier 3"));
    }

    #[test]
    fn resub_falls_back_to_duration_months_when_cumulative_absent() {
        let w = watcher(TwitchWatcherConfig {
            subscriptions_enabled: true,
            ..Default::default()
        });
        let data = ResubData {
            user: user_fields(Some("Alice")),
            cumulative_months: None,
            duration_months: 9,
            tier: "1000".to_owned(),
            message_text: "hi".to_owned(),
        };
        assert!(
            w.handle_resubscription(&data)
                .unwrap()
                .text
                .contains("9 months")
        );
    }

    // --- handle_cheer ------------------------------------------------------

    #[test]
    fn cheer_enabled_disabled_named_anon() {
        let on = watcher(TwitchWatcherConfig {
            cheers_enabled: true,
            ..Default::default()
        });
        assert!(
            on.handle_cheer(&cheer_data(Some("Bob"), 100, "poggers", false))
                .is_some()
        );
        let off = watcher(TwitchWatcherConfig {
            cheers_enabled: false,
            ..Default::default()
        });
        assert!(
            off.handle_cheer(&cheer_data(Some("Bob"), 100, "poggers", false))
                .is_none()
        );

        let anon = on
            .handle_cheer(&cheer_data(None, 500, "hype", true))
            .unwrap();
        assert_eq!(
            anon.text,
            "An anonymous cheerer has cheered with 500 bits and says: hype"
        );
        assert!(anon.viewer.is_none());

        let named = on
            .handle_cheer(&cheer_data(Some("Bob"), 100, "poggers", false))
            .unwrap();
        assert_eq!(
            named.viewer.as_ref().unwrap().display_name.as_deref(),
            Some("Bob")
        );
    }

    // --- handle_channel_point_redemption ----------------------------------

    #[test]
    fn redemption_allow_list_and_text() {
        let w = watcher(TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned()],
            ..Default::default()
        });
        assert!(
            w.handle_channel_point_redemption(&redemption_data("Alice", "Talk to Sapphire", ""))
                .is_some()
        );
        assert!(
            w.handle_channel_point_redemption(&redemption_data("Alice", "Something Else", ""))
                .is_none()
        );

        let empty = watcher(TwitchWatcherConfig {
            redemption_names: vec![],
            ..Default::default()
        });
        assert!(
            empty
                .handle_channel_point_redemption(&redemption_data("Alice", "Talk to Sapphire", ""))
                .is_none()
        );

        let with_input = w
            .handle_channel_point_redemption(&redemption_data(
                "Alice",
                "Talk to Sapphire",
                "hello!",
            ))
            .unwrap();
        assert_eq!(
            with_input.text,
            "Alice has redeemed Talk to Sapphire and says: hello!"
        );

        let hydrate = watcher(TwitchWatcherConfig {
            redemption_names: vec!["Hydrate".to_owned()],
            ..Default::default()
        });
        let no_input = hydrate
            .handle_channel_point_redemption(&redemption_data("Bob", "Hydrate", ""))
            .unwrap();
        assert_eq!(no_input.text, "Bob has redeemed Hydrate");
    }

    // --- handle_ad_break_begin --------------------------------------------

    #[test]
    fn ad_break_enabled_disabled() {
        let on = watcher(TwitchWatcherConfig {
            ads_enabled: true,
            ..Default::default()
        });
        assert!(
            on.handle_ad_break_begin(&AdBreakData {
                duration_seconds: 60
            })
            .is_some()
        );
        let off = watcher(TwitchWatcherConfig {
            ads_enabled: false,
            ..Default::default()
        });
        assert!(
            off.handle_ad_break_begin(&AdBreakData {
                duration_seconds: 60
            })
            .is_none()
        );
    }

    // --- register_listeners: recording mock --------------------------------

    #[derive(Default)]
    struct MockEventSub {
        calls: Mutex<Vec<(&'static str, String, Option<String>)>>,
        follow_cb: Mutex<Option<FollowCb>>,
        started: Mutex<usize>,
        stopped: Mutex<usize>,
    }

    impl MockEventSub {
        fn count(&self, method: &str) -> usize {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|c| c.0 == method)
                .count()
        }
        fn broadcaster_of(&self, method: &str) -> Option<String> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .find(|c| c.0 == method)
                .map(|c| c.1.clone())
        }
        fn record(&self, method: &'static str, b: String, m: Option<String>) {
            self.calls.lock().unwrap().push((method, b, m));
        }
    }

    #[async_trait]
    impl EventSub for MockEventSub {
        async fn listen_channel_follow_v2(&self, b: String, m: String, cb: FollowCb) {
            self.record("follow", b, Some(m));
            *self.follow_cb.lock().unwrap() = Some(cb);
        }
        async fn listen_channel_subscribe(&self, b: String, _cb: super::SubscribeCb) {
            self.record("subscribe", b, None);
        }
        async fn listen_channel_subscription_gift(&self, b: String, _cb: super::GiftCb) {
            self.record("gift", b, None);
        }
        async fn listen_channel_subscription_message(&self, b: String, _cb: super::ResubCb) {
            self.record("resub", b, None);
        }
        async fn listen_channel_cheer(&self, b: String, _cb: super::CheerCb) {
            self.record("cheer", b, None);
        }
        async fn listen_channel_ad_break_begin(&self, b: String, _cb: super::AdCb) {
            self.record("ad", b, None);
        }
        async fn listen_channel_points_custom_reward_redemption_add(
            &self,
            b: String,
            _cb: super::RedemptionCb,
        ) {
            self.record("redemption", b, None);
        }
        async fn start(&self) {
            *self.started.lock().unwrap() += 1;
        }
        async fn stop(&self) {
            *self.stopped.lock().unwrap() += 1;
        }
    }

    #[tokio::test]
    async fn follow_listener_registered_only_when_enabled() {
        let mock = MockEventSub::default();
        let w = Arc::new(TwitchWatcher::new(
            TwitchWatcherConfig {
                follows_enabled: true,
                ..Default::default()
            },
            "123",
            "ch",
            Some("456".to_owned()),
        ));
        w.register_listeners(&mock, None).await;
        assert_eq!(mock.count("follow"), 1);
        let call = mock.calls.lock().unwrap()[0].clone();
        assert_eq!(call.1, "123");
        assert_eq!(call.2.as_deref(), Some("456"));

        let mock2 = MockEventSub::default();
        let w2 = Arc::new(watcher(TwitchWatcherConfig {
            follows_enabled: false,
            ..Default::default()
        }));
        w2.register_listeners(&mock2, None).await;
        assert_eq!(mock2.count("follow"), 0);
    }

    #[tokio::test]
    async fn subscription_and_cheer_and_ad_listeners_matrix() {
        let mock = MockEventSub::default();
        let w = Arc::new(TwitchWatcher::new(
            TwitchWatcherConfig {
                subscriptions_enabled: true,
                cheers_enabled: true,
                ads_enabled: true,
                redemption_names: vec!["Test".to_owned()],
                ..Default::default()
            },
            "999",
            "ch",
            None,
        ));
        w.register_listeners(&mock, None).await;
        assert_eq!(mock.count("subscribe"), 1);
        assert_eq!(mock.count("gift"), 1);
        assert_eq!(mock.count("resub"), 1);
        assert_eq!(mock.count("cheer"), 1);
        assert_eq!(mock.count("ad"), 1);
        assert_eq!(mock.count("redemption"), 1);
        for method in ["subscribe", "gift", "resub", "cheer", "ad", "redemption"] {
            assert_eq!(
                mock.broadcaster_of(method).as_deref(),
                Some("999"),
                "{method}"
            );
        }
    }

    #[tokio::test]
    async fn disabled_config_registers_nothing_but_follow_default_off() {
        let mock = MockEventSub::default();
        let w = Arc::new(watcher(TwitchWatcherConfig {
            follows_enabled: false,
            subscriptions_enabled: false,
            cheers_enabled: false,
            ads_enabled: false,
            redemption_names: vec![],
            ..Default::default()
        }));
        w.register_listeners(&mock, None).await;
        assert_eq!(mock.calls.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn redemption_listener_gated_on_nonempty_list() {
        let mock = MockEventSub::default();
        let w = Arc::new(watcher(TwitchWatcherConfig {
            redemption_names: vec![],
            ..Default::default()
        }));
        w.register_listeners(&mock, None).await;
        assert_eq!(mock.count("redemption"), 0);
    }

    // --- run + callback delivery ------------------------------------------

    async fn run_briefly(
        w: &Arc<TwitchWatcher>,
        mock: Arc<MockEventSub>,
    ) -> (UnboundedReceiver<TwitchEvent>, Arc<MockEventSub>) {
        let (tx, rx) = mpsc::unbounded_channel::<TwitchEvent>();
        let cancel = CancellationToken::new();
        let handle = tokio::spawn({
            let w = Arc::clone(w);
            let mock_dyn: Arc<dyn EventSub> = Arc::clone(&mock) as Arc<dyn EventSub>;
            let cancel = cancel.clone();
            async move { w.run(tx, mock_dyn, cancel).await }
        });
        // Let register_listeners + start resolve, then cancel so `stop` runs.
        tokio::task::yield_now().await;
        cancel.cancel();
        handle.await.unwrap();
        (rx, mock)
    }

    #[tokio::test]
    async fn run_calls_start_then_stop_and_registers() {
        let mock = Arc::new(MockEventSub::default());
        let w = Arc::new(watcher(TwitchWatcherConfig {
            follows_enabled: true,
            ..Default::default()
        }));
        let (_rx, mock) = run_briefly(&w, mock).await;
        assert_eq!(*mock.started.lock().unwrap(), 1);
        assert_eq!(*mock.stopped.lock().unwrap(), 1);
        assert_eq!(mock.count("follow"), 1);
    }

    #[tokio::test]
    async fn callback_sends_event_to_channel() {
        let mock = Arc::new(MockEventSub::default());
        let w = Arc::new(watcher(TwitchWatcherConfig {
            follows_enabled: true,
            ..Default::default()
        }));
        let (mut rx, mock) = run_briefly(&w, mock).await;
        // Fire the captured follow callback with a wrapped data object.
        let cb = mock
            .follow_cb
            .lock()
            .unwrap()
            .take()
            .expect("follow cb captured");
        cb(EventWrapper {
            event: follow_data("Alice"),
        })
        .await;
        let result = rx.try_recv().expect("event queued");
        assert_eq!(result.text, "Alice has followed the channel");
        assert_eq!(
            result.viewer.as_ref().map(Author::label).as_deref(),
            Some("Alice")
        );
    }

    #[tokio::test]
    async fn callback_for_disabled_event_does_not_send() {
        // A disabled handler returns None → nothing queued.
        let w = watcher(TwitchWatcherConfig {
            follows_enabled: false,
            ..Default::default()
        });
        assert!(w.handle_follow(&follow_data("Alice")).is_none());
    }
}
