//! Twitch event types + formatters + builders (subsystem 11; Python `twitch.py`).
//!
//! The **pure** half of the Twitch pipeline: normalized [`TwitchEvent`] values,
//! the exact-string text [formatters](format_follow), and the config-gated
//! [event builders](build_follow_event). No IO, no `twitch_api` dependency —
//! that lives in `twitch_watcher` (Layer 3, feature `twitch`, dormant).
//!
//! The formatter output strings are a conformance contract (spec 11 §12, Data
//! formats table): every string here is byte-for-byte pinned by tests, so a
//! future consumer that surfaces Twitch events into the conversation gets the
//! exact phrasings the model was tuned against. The whole pipeline is currently
//! dormant — nothing constructs a watcher and nothing subscribes to
//! `twitch.event` — but the pure layer is a cheap, self-contained conformance
//! win, ported first.

use chrono::{DateTime, Utc};

use crate::identity::Author;
use crate::llm::Message;

// ---------------------------------------------------------------------------
// Event shape
// ---------------------------------------------------------------------------

/// Delivery priority for a [`TwitchEvent`].
///
/// Everything is [`Normal`](Priority::Normal) except ad break start/end, which
/// are [`Immediate`](Priority::Immediate) when the watcher config's
/// `ads_immediate` flag is set (spec 11 §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// Batched with the normal message flow.
    Normal,
    /// Surfaced immediately (ad breaks, when `ads_immediate`).
    Immediate,
}

impl Priority {
    /// The wire string (`"normal"` / `"immediate"`), matching Python's
    /// `Literal["normal", "immediate"]`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Immediate => "immediate",
        }
    }
}

/// Single Twitch channel event ready for an LLM message batch.
///
/// `viewer` is `None` for events with no associated person (ad breaks) and for
/// anonymous gift subs / cheers (spec 11 §4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwitchEvent {
    /// The channel the event fired on.
    pub channel: String,
    /// Rendered event text (one of the formatter outputs).
    pub text: String,
    /// Delivery priority.
    pub priority: Priority,
    /// UTC-aware build-time timestamp.
    pub timestamp: DateTime<Utc>,
    /// The associated viewer, when the event has one.
    pub viewer: Option<Author>,
}

impl TwitchEvent {
    /// Convert to an LLM [`Message`].
    ///
    /// Content is prefixed with `"[Twitch] "` so the model identifies the
    /// source. The message `name` is the viewer's `openai_name` when present,
    /// else the literal `"Twitch"` (spec 11 Data formats).
    #[must_use]
    pub fn to_message(&self) -> Message {
        let name = self
            .viewer
            .as_ref()
            .map_or_else(|| Some("Twitch".to_owned()), Author::openai_name);
        let mut message = Message::new("user", format!("[Twitch] {}", self.text));
        message.name = name;
        message
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-watcher config — which Twitch events produce messages.
///
/// All boolean flags default to `true`; `redemption_names` defaults to empty
/// (channel-point redemptions are an opt-in allow-list). Programmatic only — no
/// TOML/env source (spec 11 Config knobs).
// The five independent on/off toggles are the faithful port of the Python
// `TwitchWatcherConfig` dataclass fields; each gates a distinct event type, so
// collapsing them into a bitflag/enum would diverge from the pinned surface.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwitchWatcherConfig {
    /// Emit new/gift/resub subscription events.
    pub subscriptions_enabled: bool,
    /// Emit cheer (bits) events.
    pub cheers_enabled: bool,
    /// Emit follow events.
    pub follows_enabled: bool,
    /// Emit ad break start/end events.
    pub ads_enabled: bool,
    /// Mark ad break events `Immediate` (else `Normal`).
    pub ads_immediate: bool,
    /// Allow-list of channel-point redemption names that produce events.
    pub redemption_names: Vec<String>,
}

impl Default for TwitchWatcherConfig {
    fn default() -> Self {
        Self {
            subscriptions_enabled: true,
            cheers_enabled: true,
            follows_enabled: true,
            ads_enabled: true,
            ads_immediate: true,
            redemption_names: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Text formatters
// ---------------------------------------------------------------------------

/// Channel-point redemption text. Empty/absent `user_input` drops the
/// `" and says: …"` clause (spec 11 §5).
#[must_use]
pub fn format_channel_point_redemption(
    viewer: &str,
    redemption_name: &str,
    user_input: Option<&str>,
) -> String {
    let base = format!("{viewer} has redeemed {redemption_name}");
    match user_input {
        Some(input) if !input.is_empty() => format!("{base} and says: {input}"),
        _ => base,
    }
}

/// First-time subscription text.
#[must_use]
pub fn format_subscription(viewer: &str, tier: u32) -> String {
    format!("{viewer} has subscribed at tier {tier}")
}

/// Gift-subscription text. `None` gifter → `"An anonymous gifter"`; the noun is
/// singular only when `count == 1`.
#[must_use]
pub fn format_gift_subscription(gifter: Option<&str>, count: u32, tier: u32) -> String {
    let name = gifter.unwrap_or("An anonymous gifter");
    let noun = if count == 1 {
        "subscription"
    } else {
        "subscriptions"
    };
    format!("{name} has gifted {count} tier {tier} {noun}")
}

/// Resubscription text (always carries the resubscriber's message).
#[must_use]
pub fn format_resubscription(viewer: &str, months: u32, tier: u32, message: &str) -> String {
    format!("{viewer} has subscribed for {months} months at tier {tier} and says: {message}")
}

/// Cheer (bits) text. `None` viewer → `"An anonymous cheerer"`.
#[must_use]
pub fn format_cheer(viewer: Option<&str>, bits: u32, message: &str) -> String {
    let name = viewer.unwrap_or("An anonymous cheerer");
    format!("{name} has cheered with {bits} bits and says: {message}")
}

/// Follow text.
#[must_use]
pub fn format_follow(viewer: &str) -> String {
    format!("{viewer} has followed the channel")
}

/// Ad break start text.
#[must_use]
pub fn format_ad_start() -> String {
    "An ad has begun on the channel".to_owned()
}

/// Ad break end text.
#[must_use]
pub fn format_ad_end() -> String {
    "Ads have ended".to_owned()
}

// ---------------------------------------------------------------------------
// Event builders
// ---------------------------------------------------------------------------

/// The ad-event priority given the config (`Immediate` iff `ads_immediate`).
const fn ad_priority(config: &TwitchWatcherConfig) -> Priority {
    if config.ads_immediate {
        Priority::Immediate
    } else {
        Priority::Normal
    }
}

/// Channel-point redemption event; `None` unless `redemption_name` is (exact
/// string) in `config.redemption_names`.
#[must_use]
pub fn build_channel_point_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    viewer: Author,
    redemption_name: &str,
    user_input: Option<&str>,
) -> Option<TwitchEvent> {
    if !config
        .redemption_names
        .iter()
        .any(|name| name == redemption_name)
    {
        return None;
    }
    let text = format_channel_point_redemption(&viewer.label(), redemption_name, user_input);
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer: Some(viewer),
    })
}

/// New subscription event; `None` if subscriptions disabled.
#[must_use]
pub fn build_subscription_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    viewer: Author,
    tier: u32,
) -> Option<TwitchEvent> {
    if !config.subscriptions_enabled {
        return None;
    }
    let text = format_subscription(&viewer.label(), tier);
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer: Some(viewer),
    })
}

/// Gift-subscription event; `None` if subscriptions disabled. Anonymous gifters
/// (`gifter == None`) render as `"An anonymous gifter"` and leave `viewer` unset.
#[must_use]
pub fn build_gift_subscription_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    gifter: Option<Author>,
    count: u32,
    tier: u32,
) -> Option<TwitchEvent> {
    if !config.subscriptions_enabled {
        return None;
    }
    let gifter_label = gifter.as_ref().map(Author::label);
    let text = format_gift_subscription(gifter_label.as_deref(), count, tier);
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer: gifter,
    })
}

/// Resubscription event; `None` if subscriptions disabled.
#[must_use]
pub fn build_resubscription_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    viewer: Author,
    months: u32,
    tier: u32,
    message: &str,
) -> Option<TwitchEvent> {
    if !config.subscriptions_enabled {
        return None;
    }
    let text = format_resubscription(&viewer.label(), months, tier, message);
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer: Some(viewer),
    })
}

/// Cheer (bits) event; `None` if cheers disabled. Anonymous cheerers
/// (`viewer == None`) render as `"An anonymous cheerer"` and leave `viewer` unset.
#[must_use]
pub fn build_cheer_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    viewer: Option<Author>,
    bits: u32,
    message: &str,
) -> Option<TwitchEvent> {
    if !config.cheers_enabled {
        return None;
    }
    let viewer_label = viewer.as_ref().map(Author::label);
    let text = format_cheer(viewer_label.as_deref(), bits, message);
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer,
    })
}

/// Follow event; `None` if follows disabled.
#[must_use]
pub fn build_follow_event(
    config: &TwitchWatcherConfig,
    channel: &str,
    viewer: Author,
) -> Option<TwitchEvent> {
    if !config.follows_enabled {
        return None;
    }
    let text = format_follow(&viewer.label());
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text,
        priority: Priority::Normal,
        timestamp: Utc::now(),
        viewer: Some(viewer),
    })
}

/// Ad break start event; `None` if ads disabled.
#[must_use]
pub fn build_ad_start_event(config: &TwitchWatcherConfig, channel: &str) -> Option<TwitchEvent> {
    if !config.ads_enabled {
        return None;
    }
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text: format_ad_start(),
        priority: ad_priority(config),
        timestamp: Utc::now(),
        viewer: None,
    })
}

/// Ad break end event; `None` if ads disabled.
///
/// Public and tested but currently has no watcher caller (ad end is not an
/// EventSub subscription in v1; spec 11 §Public API).
#[must_use]
pub fn build_ad_end_event(config: &TwitchWatcherConfig, channel: &str) -> Option<TwitchEvent> {
    if !config.ads_enabled {
        return None;
    }
    Some(TwitchEvent {
        channel: channel.to_owned(),
        text: format_ad_end(),
        priority: ad_priority(config),
        timestamp: Utc::now(),
        viewer: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Priority, TwitchEvent, TwitchWatcherConfig, build_ad_end_event, build_ad_start_event,
        build_channel_point_event, build_cheer_event, build_follow_event,
        build_gift_subscription_event, build_resubscription_event, build_subscription_event,
        format_ad_end, format_ad_start, format_channel_point_redemption, format_cheer,
        format_follow, format_gift_subscription, format_resubscription, format_subscription,
    };
    use crate::identity::Author;
    use crate::llm::Message;
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    fn alice() -> Author {
        Author::from_twitch(
            "uid-Alice",
            Some("alice".to_owned()),
            Some("Alice".to_owned()),
        )
    }

    fn bob() -> Author {
        Author::from_twitch("uid-Bob", Some("bob".to_owned()), Some("Bob".to_owned()))
    }

    fn carol() -> Author {
        Author::from_twitch(
            "uid-Carol",
            Some("carol".to_owned()),
            Some("Carol".to_owned()),
        )
    }

    // --- TwitchEvent shape -------------------------------------------------

    #[test]
    fn twitch_event_required_fields() {
        let event = TwitchEvent {
            channel: "sapphire-stream".to_owned(),
            text: "Alice has followed the channel".to_owned(),
            priority: Priority::Normal,
            timestamp: Utc::now(),
            viewer: None,
        };
        assert_eq!(event.channel, "sapphire-stream");
        assert_eq!(event.text, "Alice has followed the channel");
        assert_eq!(event.priority, Priority::Normal);
    }

    #[test]
    fn twitch_event_priority_immediate() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "ad started".to_owned(),
            priority: Priority::Immediate,
            timestamp: Utc::now(),
            viewer: None,
        };
        assert_eq!(event.priority, Priority::Immediate);
    }

    #[test]
    fn twitch_event_timestamp_is_utc() {
        let ts = Utc.with_ymd_and_hms(2026, 4, 6, 12, 0, 0).unwrap();
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "something".to_owned(),
            priority: Priority::Normal,
            timestamp: ts,
            viewer: None,
        };
        assert_eq!(event.timestamp, ts);
    }

    #[test]
    fn twitch_event_viewer_present_when_set() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "Alice has followed the channel".to_owned(),
            priority: Priority::Normal,
            timestamp: Utc::now(),
            viewer: Some(alice()),
        };
        assert_eq!(event.viewer, Some(alice()));
    }

    #[test]
    fn twitch_event_viewer_defaults_to_none() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "An ad has begun on the channel".to_owned(),
            priority: Priority::Immediate,
            timestamp: Utc::now(),
            viewer: None,
        };
        assert!(event.viewer.is_none());
    }

    // --- text formatters ---------------------------------------------------

    #[test]
    fn channel_point_redemption_without_input() {
        let text = format_channel_point_redemption("Alice", "Talk to Sapphire", None);
        assert_eq!(text, "Alice has redeemed Talk to Sapphire");
    }

    #[test]
    fn channel_point_redemption_with_input() {
        let text = format_channel_point_redemption("Alice", "Talk to Sapphire", Some("hello!"));
        assert_eq!(text, "Alice has redeemed Talk to Sapphire and says: hello!");
    }

    #[test]
    fn channel_point_redemption_empty_input_treated_as_no_input() {
        let text = format_channel_point_redemption("Alice", "Talk to Sapphire", Some(""));
        assert!(!text.contains("says"));
    }

    #[test]
    fn subscription_tiers() {
        assert_eq!(
            format_subscription("Alice", 1),
            "Alice has subscribed at tier 1"
        );
        assert_eq!(
            format_subscription("Alice", 2),
            "Alice has subscribed at tier 2"
        );
        assert_eq!(
            format_subscription("Bob", 3),
            "Bob has subscribed at tier 3"
        );
    }

    #[test]
    fn gift_subscription_named_single() {
        assert_eq!(
            format_gift_subscription(Some("Bob"), 1, 1),
            "Bob has gifted 1 tier 1 subscription"
        );
    }

    #[test]
    fn gift_subscription_named_multiple() {
        assert_eq!(
            format_gift_subscription(Some("Bob"), 5, 1),
            "Bob has gifted 5 tier 1 subscriptions"
        );
    }

    #[test]
    fn gift_subscription_anonymous() {
        assert_eq!(
            format_gift_subscription(None, 5, 1),
            "An anonymous gifter has gifted 5 tier 1 subscriptions"
        );
    }

    #[test]
    fn gift_subscription_tier_2() {
        assert!(format_gift_subscription(Some("Alice"), 3, 2).contains("tier 2"));
    }

    #[test]
    fn resubscription_basic() {
        let text = format_resubscription("Alice", 6, 2, "love this stream");
        assert_eq!(
            text,
            "Alice has subscribed for 6 months at tier 2 and says: love this stream"
        );
    }

    #[test]
    fn resubscription_one_month() {
        // Python keeps plural month phrasing regardless of count; the substring
        // `"1 month"` is present within `"1 months"`.
        let text = format_resubscription("Bob", 1, 1, "woo");
        assert!(text.contains("1 month"));
    }

    #[test]
    fn cheer_named_with_message() {
        assert_eq!(
            format_cheer(Some("Bob"), 100, "poggers"),
            "Bob has cheered with 100 bits and says: poggers"
        );
    }

    #[test]
    fn cheer_anonymous() {
        assert_eq!(
            format_cheer(None, 500, "hype"),
            "An anonymous cheerer has cheered with 500 bits and says: hype"
        );
    }

    #[test]
    fn follow_message() {
        assert_eq!(format_follow("Alice"), "Alice has followed the channel");
    }

    #[test]
    fn ad_break_messages() {
        assert_eq!(format_ad_start(), "An ad has begun on the channel");
        assert_eq!(format_ad_end(), "Ads have ended");
    }

    // --- TwitchWatcherConfig ----------------------------------------------

    #[test]
    fn config_defaults() {
        let config = TwitchWatcherConfig::default();
        assert!(config.subscriptions_enabled);
        assert!(config.cheers_enabled);
        assert!(config.follows_enabled);
        assert!(config.ads_enabled);
        assert!(config.ads_immediate);
        assert!(config.redemption_names.is_empty());
    }

    #[test]
    fn config_can_disable_each_flag() {
        assert!(
            !TwitchWatcherConfig {
                subscriptions_enabled: false,
                ..Default::default()
            }
            .subscriptions_enabled
        );
        assert!(
            !TwitchWatcherConfig {
                cheers_enabled: false,
                ..Default::default()
            }
            .cheers_enabled
        );
        assert!(
            !TwitchWatcherConfig {
                follows_enabled: false,
                ..Default::default()
            }
            .follows_enabled
        );
        assert!(
            !TwitchWatcherConfig {
                ads_enabled: false,
                ..Default::default()
            }
            .ads_enabled
        );
        assert!(
            !TwitchWatcherConfig {
                ads_immediate: false,
                ..Default::default()
            }
            .ads_immediate
        );
    }

    #[test]
    fn config_can_set_redemption_names() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned(), "Hydrate".to_owned()],
            ..Default::default()
        };
        assert!(
            config
                .redemption_names
                .iter()
                .any(|n| n == "Talk to Sapphire")
        );
        assert!(config.redemption_names.iter().any(|n| n == "Hydrate"));
    }

    #[test]
    fn config_redemption_names_independent_per_instance() {
        // Rust `Vec` is owned per value — no shared mutable default (the Python
        // `field(default_factory=list)` bug class cannot occur here).
        let mut a = TwitchWatcherConfig::default();
        let b = TwitchWatcherConfig::default();
        a.redemption_names.push("Test".to_owned());
        assert!(!b.redemption_names.iter().any(|n| n == "Test"));
    }

    // --- build_channel_point_event ----------------------------------------

    #[test]
    fn channel_point_event_for_allowed_redemption() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned()],
            ..Default::default()
        };
        let event =
            build_channel_point_event(&config, "ch", alice(), "Talk to Sapphire", None).unwrap();
        assert_eq!(event.priority, Priority::Normal);
    }

    #[test]
    fn channel_point_event_none_for_unlisted() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned()],
            ..Default::default()
        };
        assert!(
            build_channel_point_event(&config, "ch", alice(), "Something Else", None).is_none()
        );
    }

    #[test]
    fn channel_point_event_none_when_list_empty() {
        let config = TwitchWatcherConfig::default();
        assert!(
            build_channel_point_event(&config, "ch", alice(), "Talk to Sapphire", None).is_none()
        );
    }

    #[test]
    fn channel_point_event_text_matches_formatter() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned()],
            ..Default::default()
        };
        let event =
            build_channel_point_event(&config, "ch", alice(), "Talk to Sapphire", Some("hello!"))
                .unwrap();
        assert_eq!(
            event.text,
            format_channel_point_redemption("Alice", "Talk to Sapphire", Some("hello!"))
        );
    }

    #[test]
    fn channel_point_event_viewer_and_channel() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Test".to_owned()],
            ..Default::default()
        };
        let event =
            build_channel_point_event(&config, "my-channel", alice(), "Test", None).unwrap();
        assert_eq!(event.viewer, Some(alice()));
        assert_eq!(event.channel, "my-channel");
    }

    // --- build_subscription_event -----------------------------------------

    #[test]
    fn subscription_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_subscription_event(&config, "ch", alice(), 1).unwrap();
        assert_eq!(event.priority, Priority::Normal);
        assert_eq!(event.viewer, Some(alice()));
    }

    #[test]
    fn subscription_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        };
        assert!(build_subscription_event(&config, "ch", alice(), 1).is_none());
    }

    #[test]
    fn subscription_event_text_matches_formatter() {
        let config = TwitchWatcherConfig::default();
        let event = build_subscription_event(&config, "ch", alice(), 2).unwrap();
        assert_eq!(event.text, format_subscription("Alice", 2));
    }

    // --- build_gift_subscription_event ------------------------------------

    #[test]
    fn gift_subscription_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_gift_subscription_event(&config, "ch", Some(bob()), 5, 1).unwrap();
        assert_eq!(event.text, format_gift_subscription(Some("Bob"), 5, 1));
        assert_eq!(event.viewer, Some(bob()));
    }

    #[test]
    fn gift_subscription_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        };
        assert!(build_gift_subscription_event(&config, "ch", Some(bob()), 5, 1).is_none());
    }

    #[test]
    fn gift_subscription_event_anonymous() {
        let config = TwitchWatcherConfig::default();
        let event = build_gift_subscription_event(&config, "ch", None, 3, 1).unwrap();
        assert!(event.text.to_lowercase().contains("anonymous"));
        assert!(event.viewer.is_none());
    }

    // --- build_resubscription_event ---------------------------------------

    #[test]
    fn resubscription_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event =
            build_resubscription_event(&config, "ch", alice(), 6, 2, "love this stream").unwrap();
        assert_eq!(event.priority, Priority::Normal);
        assert_eq!(event.viewer, Some(alice()));
        assert_eq!(
            event.text,
            format_resubscription("Alice", 6, 2, "love this stream")
        );
    }

    #[test]
    fn resubscription_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            subscriptions_enabled: false,
            ..Default::default()
        };
        assert!(
            build_resubscription_event(&config, "ch", alice(), 6, 2, "love this stream").is_none()
        );
    }

    // --- build_cheer_event -------------------------------------------------

    #[test]
    fn cheer_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_cheer_event(&config, "ch", Some(bob()), 100, "poggers").unwrap();
        assert_eq!(event.priority, Priority::Normal);
        assert_eq!(event.viewer, Some(bob()));
        assert_eq!(event.text, format_cheer(Some("Bob"), 100, "poggers"));
    }

    #[test]
    fn cheer_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            cheers_enabled: false,
            ..Default::default()
        };
        assert!(build_cheer_event(&config, "ch", Some(bob()), 100, "poggers").is_none());
    }

    #[test]
    fn cheer_event_anonymous() {
        let config = TwitchWatcherConfig::default();
        let event = build_cheer_event(&config, "ch", None, 500, "hype").unwrap();
        assert!(event.text.to_lowercase().contains("anonymous"));
        assert!(event.viewer.is_none());
    }

    // --- build_follow_event ------------------------------------------------

    #[test]
    fn follow_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_follow_event(&config, "ch", alice()).unwrap();
        assert_eq!(event.priority, Priority::Normal);
        assert_eq!(event.viewer, Some(alice()));
        assert_eq!(event.text, format_follow("Alice"));
    }

    #[test]
    fn follow_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            follows_enabled: false,
            ..Default::default()
        };
        assert!(build_follow_event(&config, "ch", alice()).is_none());
    }

    // --- build_ad_start_event / build_ad_end_event ------------------------

    #[test]
    fn ad_start_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_ad_start_event(&config, "ch").unwrap();
        assert_eq!(event.text, format_ad_start());
        assert!(event.viewer.is_none());
    }

    #[test]
    fn ad_start_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            ads_enabled: false,
            ..Default::default()
        };
        assert!(build_ad_start_event(&config, "ch").is_none());
    }

    #[test]
    fn ad_start_priority_immediate_when_ads_immediate() {
        let config = TwitchWatcherConfig {
            ads_immediate: true,
            ..Default::default()
        };
        assert_eq!(
            build_ad_start_event(&config, "ch").unwrap().priority,
            Priority::Immediate
        );
    }

    #[test]
    fn ad_start_priority_normal_when_not_immediate() {
        let config = TwitchWatcherConfig {
            ads_immediate: false,
            ..Default::default()
        };
        assert_eq!(
            build_ad_start_event(&config, "ch").unwrap().priority,
            Priority::Normal
        );
    }

    #[test]
    fn ad_end_event_when_enabled() {
        let config = TwitchWatcherConfig::default();
        let event = build_ad_end_event(&config, "ch").unwrap();
        assert_eq!(event.text, format_ad_end());
        assert!(event.viewer.is_none());
    }

    #[test]
    fn ad_end_event_none_when_disabled() {
        let config = TwitchWatcherConfig {
            ads_enabled: false,
            ..Default::default()
        };
        assert!(build_ad_end_event(&config, "ch").is_none());
    }

    #[test]
    fn ad_end_priority_matches_ads_immediate() {
        let immediate = TwitchWatcherConfig {
            ads_immediate: true,
            ..Default::default()
        };
        assert_eq!(
            build_ad_end_event(&immediate, "ch").unwrap().priority,
            Priority::Immediate
        );
        let normal = TwitchWatcherConfig {
            ads_immediate: false,
            ..Default::default()
        };
        assert_eq!(
            build_ad_end_event(&normal, "ch").unwrap().priority,
            Priority::Normal
        );
    }

    // --- TwitchEvent → Message conversion ---------------------------------

    #[test]
    fn to_message_role_and_prefix() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "Alice has subscribed at tier 1".to_owned(),
            priority: Priority::Normal,
            timestamp: Utc::now(),
            viewer: None,
        };
        let msg = event.to_message();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "[Twitch] Alice has subscribed at tier 1");
    }

    #[test]
    fn to_message_name_is_viewer_when_present() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "Alice has followed the channel".to_owned(),
            priority: Priority::Normal,
            timestamp: Utc::now(),
            viewer: Some(alice()),
        };
        assert_eq!(event.to_message().name.as_deref(), Some("Alice"));
    }

    #[test]
    fn to_message_name_falls_back_to_twitch_without_viewer() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "An ad has begun on the channel".to_owned(),
            priority: Priority::Immediate,
            timestamp: Utc::now(),
            viewer: None,
        };
        assert_eq!(event.to_message().name.as_deref(), Some("Twitch"));
    }

    #[test]
    fn to_message_serialisable_with_viewer() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "Alice has followed the channel".to_owned(),
            priority: Priority::Normal,
            timestamp: Utc::now(),
            viewer: Some(alice()),
        };
        assert_eq!(
            event.to_message().to_dict(),
            json!({
                "role": "user",
                "content": "[Twitch] Alice has followed the channel",
                "name": "Alice",
            })
        );
    }

    #[test]
    fn to_message_serialisable_without_viewer() {
        let event = TwitchEvent {
            channel: "ch".to_owned(),
            text: "An ad has begun on the channel".to_owned(),
            priority: Priority::Immediate,
            timestamp: Utc::now(),
            viewer: None,
        };
        assert_eq!(
            event.to_message().to_dict(),
            json!({
                "role": "user",
                "content": "[Twitch] An ad has begun on the channel",
                "name": "Twitch",
            })
        );
    }

    #[test]
    fn immediate_event_priority_readable_before_conversion() {
        let config = TwitchWatcherConfig::default();
        let event = build_ad_start_event(&config, "ch").unwrap();
        assert_eq!(event.priority, Priority::Immediate);
        // Conversion still works regardless of priority.
        let _msg: Message = event.to_message();
    }

    #[test]
    fn normal_event_round_trips_through_message_list() {
        let config = TwitchWatcherConfig {
            redemption_names: vec!["Talk to Sapphire".to_owned()],
            ..Default::default()
        };
        let raw_events = [
            build_follow_event(&config, "ch", alice()),
            build_subscription_event(&config, "ch", bob(), 1),
            build_channel_point_event(&config, "ch", carol(), "Talk to Sapphire", Some("hello!")),
        ];
        let messages: Vec<Message> = raw_events
            .into_iter()
            .flatten()
            .map(|event| event.to_message())
            .collect();
        assert_eq!(messages.len(), 3);
        assert!(messages.iter().all(|m| m.role == "user"));
        let names: Vec<Option<String>> = messages.iter().map(|m| m.name.clone()).collect();
        assert_eq!(
            names,
            vec![
                Some("Alice".to_owned()),
                Some("Bob".to_owned()),
                Some("Carol".to_owned()),
            ]
        );
        let contents: Vec<String> = messages.iter().map(Message::content_str).collect();
        assert!(contents.iter().all(|c| c.starts_with("[Twitch] ")));
        assert!(contents.iter().any(|c| c.contains("Alice")));
        assert!(contents.iter().any(|c| c.contains("Bob")));
        assert!(
            contents
                .iter()
                .any(|c| c.contains("Carol") && c.contains("hello!"))
        );
    }

    #[test]
    fn anonymous_cheer_falls_back_to_twitch() {
        let config = TwitchWatcherConfig::default();
        let event = build_cheer_event(&config, "ch", None, 500, "hype").unwrap();
        assert_eq!(event.to_message().name.as_deref(), Some("Twitch"));
    }

    #[test]
    fn anonymous_gift_sub_falls_back_to_twitch() {
        let config = TwitchWatcherConfig::default();
        let event = build_gift_subscription_event(&config, "ch", None, 3, 1).unwrap();
        assert_eq!(event.to_message().name.as_deref(), Some("Twitch"));
    }

    #[test]
    fn ad_events_use_twitch_as_name() {
        let config = TwitchWatcherConfig::default();
        let start = build_ad_start_event(&config, "ch").unwrap();
        assert_eq!(start.to_message().name.as_deref(), Some("Twitch"));
        let end = build_ad_end_event(&config, "ch").unwrap();
        assert_eq!(end.to_message().name.as_deref(), Some("Twitch"));
    }

    #[test]
    fn priority_as_str() {
        assert_eq!(Priority::Normal.as_str(), "normal");
        assert_eq!(Priority::Immediate.as_str(), "immediate");
    }
}
