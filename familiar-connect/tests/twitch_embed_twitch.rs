//! Public-surface integration tests for the pure Twitch layer (subsystem 11;
//! Python `tests/test_twitch.py`).
//!
//! The exhaustive formatter/builder matrix lives in `twitch.rs`'s in-module
//! tests; this file pins the crate-external public API — the exact-string
//! conformance targets and the `TwitchEvent` → `Message` conversion — the way a
//! future consumer (a source draining `twitch.event`) would use it.

use familiar_connect::identity::Author;
use familiar_connect::twitch::{
    Priority, TwitchWatcherConfig, build_ad_start_event, build_channel_point_event,
    build_follow_event, build_subscription_event, format_ad_end, format_ad_start, format_cheer,
    format_gift_subscription,
};
use serde_json::json;

fn alice() -> Author {
    Author::from_twitch(
        "uid-Alice",
        Some("alice".to_owned()),
        Some("Alice".to_owned()),
    )
}

#[test]
fn exact_conformance_strings() {
    assert_eq!(format_ad_start(), "An ad has begun on the channel");
    assert_eq!(format_ad_end(), "Ads have ended");
    assert_eq!(
        format_cheer(None, 500, "hype"),
        "An anonymous cheerer has cheered with 500 bits and says: hype"
    );
    assert_eq!(
        format_gift_subscription(Some("Bob"), 1, 1),
        "Bob has gifted 1 tier 1 subscription"
    );
    assert_eq!(
        format_gift_subscription(Some("Bob"), 2, 1),
        "Bob has gifted 2 tier 1 subscriptions"
    );
}

#[test]
fn builder_gating_and_to_message_round_trip() {
    let config = TwitchWatcherConfig {
        redemption_names: vec!["Talk to Sapphire".to_owned()],
        ..Default::default()
    };

    // Disabled follows produce nothing.
    let off = TwitchWatcherConfig {
        follows_enabled: false,
        ..Default::default()
    };
    assert!(build_follow_event(&off, "ch", alice()).is_none());

    // Enabled subscription event carries the viewer name into the message.
    let sub = build_subscription_event(&config, "ch", alice(), 2).unwrap();
    assert_eq!(sub.priority, Priority::Normal);
    assert_eq!(
        sub.to_message().to_dict(),
        json!({
            "role": "user",
            "content": "[Twitch] Alice has subscribed at tier 2",
            "name": "Alice",
        })
    );

    // Allow-listed redemption with input surfaces the "and says" clause.
    let redeem =
        build_channel_point_event(&config, "ch", alice(), "Talk to Sapphire", Some("hi")).unwrap();
    assert_eq!(
        redeem.to_message().content_str(),
        "[Twitch] Alice has redeemed Talk to Sapphire and says: hi"
    );

    // Ad events have no viewer, so the message name falls back to "Twitch".
    let ad = build_ad_start_event(&config, "ch").unwrap();
    assert_eq!(ad.priority, Priority::Immediate);
    assert_eq!(ad.to_message().name.as_deref(), Some("Twitch"));
}
