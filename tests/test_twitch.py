"""Tests for Twitch channel event watcher.

Tests are written red-first against the interface described in twitch-features.md.
The module under test is familiar_connect.twitch, which does not exist yet.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from familiar_connect.twitch import (
    TwitchEvent,
    TwitchWatcherConfig,
    format_ad_end,
    format_ad_start,
    format_cheer,
    format_follow,
    format_gift_subscription,
    format_channel_point_redemption,
    format_resubscription,
    format_subscription,
    build_channel_point_event,
    build_subscription_event,
    build_gift_subscription_event,
    build_resubscription_event,
    build_cheer_event,
    build_follow_event,
    build_ad_start_event,
    build_ad_end_event,
)


# ---------------------------------------------------------------------------
# TwitchEvent shape
# ---------------------------------------------------------------------------


class TestTwitchEvent:
    def test_has_channel_field(self) -> None:
        """TwitchEvent carries the channel/session context."""
        event = TwitchEvent(
            channel="sapphire-stream",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.channel == "sapphire-stream"

    def test_has_text_field(self) -> None:
        """TwitchEvent carries a plain-text description."""
        event = TwitchEvent(
            channel="sapphire-stream",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.text == "Alice has followed the channel"

    def test_priority_normal(self) -> None:
        """TwitchEvent accepts 'normal' priority."""
        event = TwitchEvent(
            channel="ch",
            text="something happened",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.priority == "normal"

    def test_priority_immediate(self) -> None:
        """TwitchEvent accepts 'immediate' priority."""
        event = TwitchEvent(
            channel="ch",
            text="ad started",
            priority="immediate",
            timestamp=datetime.now(UTC),
        )
        assert event.priority == "immediate"

    def test_timestamp_is_utc(self) -> None:
        """TwitchEvent stores a UTC-aware datetime."""
        ts = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)
        event = TwitchEvent(
            channel="ch",
            text="something",
            priority="normal",
            timestamp=ts,
        )
        assert event.timestamp.tzinfo is not None
        assert event.timestamp == ts


# ---------------------------------------------------------------------------
# Event text formatters
# ---------------------------------------------------------------------------


class TestFormatChannelPointRedemption:
    def test_redemption_without_input(self) -> None:
        """Redemption with no user text omits the 'and says' clause."""
        text = format_channel_point_redemption("Alice", "Talk to Sapphire")
        assert text == "Alice has redeemed Talk to Sapphire"

    def test_redemption_with_input(self) -> None:
        """Redemption with user text appends 'and says: <input>'."""
        text = format_channel_point_redemption("Alice", "Talk to Sapphire", "hello!")
        assert text == "Alice has redeemed Talk to Sapphire and says: hello!"

    def test_redemption_with_empty_input_treated_as_no_input(self) -> None:
        """An empty string user input is treated the same as None."""
        text = format_channel_point_redemption("Alice", "Talk to Sapphire", "")
        assert "says" not in text

    def test_redemption_viewer_name_in_output(self) -> None:
        """The viewer's name appears in the formatted string."""
        text = format_channel_point_redemption("Bob", "Hydrate!")
        assert "Bob" in text

    def test_redemption_name_in_output(self) -> None:
        """The redemption name appears in the formatted string."""
        text = format_channel_point_redemption("Bob", "Hydrate!")
        assert "Hydrate!" in text


class TestFormatSubscription:
    def test_tier_1(self) -> None:
        """New tier-1 subscription formatted correctly."""
        text = format_subscription("Alice", 1)
        assert text == "Alice has subscribed at tier 1"

    def test_tier_2(self) -> None:
        """New tier-2 subscription formatted correctly."""
        text = format_subscription("Alice", 2)
        assert text == "Alice has subscribed at tier 2"

    def test_tier_3(self) -> None:
        """New tier-3 subscription formatted correctly."""
        text = format_subscription("Bob", 3)
        assert text == "Bob has subscribed at tier 3"

    def test_viewer_name_in_output(self) -> None:
        """The viewer's name appears in the formatted string."""
        text = format_subscription("Charlie", 1)
        assert "Charlie" in text


class TestFormatGiftSubscription:
    def test_named_gifter_single(self) -> None:
        """Named gifter, single gift."""
        text = format_gift_subscription("Bob", 1, 1)
        assert text == "Bob has gifted 1 tier 1 subscription"

    def test_named_gifter_multiple(self) -> None:
        """Named gifter, multiple gifts."""
        text = format_gift_subscription("Bob", 5, 1)
        assert text == "Bob has gifted 5 tier 1 subscriptions"

    def test_anonymous_gifter(self) -> None:
        """Anonymous gifter uses 'An anonymous gifter'."""
        text = format_gift_subscription(None, 5, 1)
        assert text == "An anonymous gifter has gifted 5 tier 1 subscriptions"

    def test_tier_2_gift(self) -> None:
        """Tier-2 gift subscriptions show the correct tier."""
        text = format_gift_subscription("Alice", 3, 2)
        assert "tier 2" in text

    def test_gifter_name_in_output(self) -> None:
        """The gifter's name appears in the formatted string."""
        text = format_gift_subscription("Charlie", 2, 1)
        assert "Charlie" in text


class TestFormatResubscription:
    def test_basic_resub(self) -> None:
        """Resub with all fields formats correctly."""
        text = format_resubscription("Alice", 6, 2, "love this stream")
        assert text == "Alice has subscribed for 6 months at tier 2 and says: love this stream"

    def test_one_month(self) -> None:
        """Singular month phrasing for 1 month."""
        text = format_resubscription("Bob", 1, 1, "woo")
        assert "1 month" in text

    def test_viewer_name_in_output(self) -> None:
        """The viewer's name appears in the formatted string."""
        text = format_resubscription("Dave", 12, 3, "still here!")
        assert "Dave" in text

    def test_message_in_output(self) -> None:
        """The viewer's message appears in the formatted string."""
        text = format_resubscription("Eve", 3, 1, "hi chat")
        assert "hi chat" in text


class TestFormatCheer:
    def test_named_cheerer_with_message(self) -> None:
        """Named cheerer with a message formats correctly."""
        text = format_cheer("Bob", 100, "poggers")
        assert text == "Bob has cheered with 100 bits and says: poggers"

    def test_anonymous_cheerer(self) -> None:
        """Anonymous cheerer uses 'An anonymous cheerer'."""
        text = format_cheer(None, 500, "hype")
        assert text == "An anonymous cheerer has cheered with 500 bits and says: hype"

    def test_bit_count_in_output(self) -> None:
        """The bit count appears in the formatted string."""
        text = format_cheer("Alice", 1000, "nice")
        assert "1000" in text

    def test_message_in_output(self) -> None:
        """The cheerer's message appears in the formatted string."""
        text = format_cheer("Alice", 50, "GG")
        assert "GG" in text

    def test_cheerer_name_in_output(self) -> None:
        """The cheerer's name appears in the formatted string."""
        text = format_cheer("Zara", 200, "woo")
        assert "Zara" in text


class TestFormatFollow:
    def test_follow_message(self) -> None:
        """Follow event formats correctly."""
        text = format_follow("Alice")
        assert text == "Alice has followed the channel"

    def test_viewer_name_in_output(self) -> None:
        """The viewer's name appears in the formatted string."""
        text = format_follow("NewViewer")
        assert "NewViewer" in text


class TestFormatAdBreak:
    def test_ad_start_message(self) -> None:
        """Ad start event formats correctly."""
        text = format_ad_start()
        assert text == "An ad has begun on the channel"

    def test_ad_end_message(self) -> None:
        """Ad end event formats correctly."""
        text = format_ad_end()
        assert text == "Ads have ended"


# ---------------------------------------------------------------------------
# TwitchWatcherConfig
# ---------------------------------------------------------------------------


class TestTwitchWatcherConfig:
    def test_default_subscriptions_enabled(self) -> None:
        """Subscriptions are enabled by default."""
        config = TwitchWatcherConfig()
        assert config.subscriptions_enabled is True

    def test_default_cheers_enabled(self) -> None:
        """Cheers are enabled by default."""
        config = TwitchWatcherConfig()
        assert config.cheers_enabled is True

    def test_default_follows_enabled(self) -> None:
        """Follows are enabled by default."""
        config = TwitchWatcherConfig()
        assert config.follows_enabled is True

    def test_default_ads_enabled(self) -> None:
        """Ads are enabled by default."""
        config = TwitchWatcherConfig()
        assert config.ads_enabled is True

    def test_default_ads_immediate(self) -> None:
        """Ad immediate mode is enabled by default."""
        config = TwitchWatcherConfig()
        assert config.ads_immediate is True

    def test_default_redemption_names_empty(self) -> None:
        """Redemption name list is empty by default (none trigger messages)."""
        config = TwitchWatcherConfig()
        assert config.redemption_names == []

    def test_can_disable_subscriptions(self) -> None:
        """Subscriptions can be toggled off."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        assert config.subscriptions_enabled is False

    def test_can_disable_cheers(self) -> None:
        """Cheers can be toggled off."""
        config = TwitchWatcherConfig(cheers_enabled=False)
        assert config.cheers_enabled is False

    def test_can_disable_follows(self) -> None:
        """Follows can be toggled off."""
        config = TwitchWatcherConfig(follows_enabled=False)
        assert config.follows_enabled is False

    def test_can_disable_ads(self) -> None:
        """Ads can be toggled off."""
        config = TwitchWatcherConfig(ads_enabled=False)
        assert config.ads_enabled is False

    def test_can_disable_ads_immediate(self) -> None:
        """Ad immediate mode can be toggled off."""
        config = TwitchWatcherConfig(ads_immediate=False)
        assert config.ads_immediate is False

    def test_can_set_redemption_names(self) -> None:
        """Redemption name list can be populated."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire", "Hydrate"])
        assert "Talk to Sapphire" in config.redemption_names
        assert "Hydrate" in config.redemption_names

    def test_redemption_names_independent_per_instance(self) -> None:
        """Two configs do not share the same redemption_names list."""
        a = TwitchWatcherConfig()
        b = TwitchWatcherConfig()
        a.redemption_names.append("Test")
        assert "Test" not in b.redemption_names


# ---------------------------------------------------------------------------
# Event builders — filtering and priority assignment
# ---------------------------------------------------------------------------


class TestBuildChannelPointEvent:
    def test_produces_event_for_allowed_redemption(self) -> None:
        """An allowed redemption name produces a TwitchEvent."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer="Alice",
            redemption_name="Talk to Sapphire",
        )
        assert event is not None
        assert isinstance(event, TwitchEvent)

    def test_returns_none_for_unlisted_redemption(self) -> None:
        """A redemption not in the allowed list produces no event."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer="Alice",
            redemption_name="Something Else",
        )
        assert event is None

    def test_returns_none_when_list_is_empty(self) -> None:
        """An empty redemption list means no redemptions produce events."""
        config = TwitchWatcherConfig(redemption_names=[])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer="Alice",
            redemption_name="Talk to Sapphire",
        )
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches what format_channel_point_redemption produces."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer="Alice",
            redemption_name="Talk to Sapphire",
            user_input="hello!",
        )
        assert event is not None
        assert event.text == format_channel_point_redemption(
            "Alice", "Talk to Sapphire", "hello!"
        )

    def test_event_priority_is_normal(self) -> None:
        """Channel point redemption events have normal priority."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer="Alice",
            redemption_name="Talk to Sapphire",
        )
        assert event is not None
        assert event.priority == "normal"

    def test_event_channel_set_correctly(self) -> None:
        """The channel field reflects the provided channel."""
        config = TwitchWatcherConfig(redemption_names=["Test"])
        event = build_channel_point_event(
            config=config, channel="my-channel", viewer="Alice", redemption_name="Test"
        )
        assert event is not None
        assert event.channel == "my-channel"

    def test_event_timestamp_is_utc_aware(self) -> None:
        """The event timestamp is UTC-aware."""
        config = TwitchWatcherConfig(redemption_names=["Test"])
        event = build_channel_point_event(
            config=config, channel="ch", viewer="Alice", redemption_name="Test"
        )
        assert event is not None
        assert event.timestamp.tzinfo is not None


class TestBuildSubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Subscription event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer="Alice", tier=1
        )
        assert event is not None
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when subscriptions are disabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        event = build_subscription_event(
            config=config, channel="ch", viewer="Alice", tier=1
        )
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_subscription output."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer="Alice", tier=2
        )
        assert event is not None
        assert event.text == format_subscription("Alice", 2)

    def test_event_priority_is_normal(self) -> None:
        """Subscription events have normal priority."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer="Alice", tier=1
        )
        assert event is not None
        assert event.priority == "normal"


class TestBuildGiftSubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Gift sub event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter="Bob", count=5, tier=1
        )
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when subscriptions are disabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter="Bob", count=5, tier=1
        )
        assert event is None

    def test_anonymous_gifter(self) -> None:
        """Anonymous gifter (gifter=None) is handled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=None, count=3, tier=1
        )
        assert event is not None
        assert "anonymous" in event.text.lower()

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_gift_subscription output."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter="Bob", count=5, tier=1
        )
        assert event is not None
        assert event.text == format_gift_subscription("Bob", 5, 1)


class TestBuildResubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Resub event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer="Alice",
            months=6,
            tier=2,
            message="love this stream",
        )
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when subscriptions are disabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer="Alice",
            months=6,
            tier=2,
            message="love this stream",
        )
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_resubscription output."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer="Alice",
            months=6,
            tier=2,
            message="love this stream",
        )
        assert event is not None
        assert event.text == format_resubscription("Alice", 6, 2, "love this stream")

    def test_event_priority_is_normal(self) -> None:
        """Resubscription events have normal priority."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer="Alice",
            months=3,
            tier=1,
            message="woo",
        )
        assert event is not None
        assert event.priority == "normal"


class TestBuildCheerEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Cheer event produced when cheers are enabled."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer="Bob", bits=100, message="poggers"
        )
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when cheers are disabled."""
        config = TwitchWatcherConfig(cheers_enabled=False)
        event = build_cheer_event(
            config=config, channel="ch", viewer="Bob", bits=100, message="poggers"
        )
        assert event is None

    def test_anonymous_cheerer(self) -> None:
        """Anonymous cheerer (viewer=None) is handled."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=None, bits=500, message="hype"
        )
        assert event is not None
        assert "anonymous" in event.text.lower()

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_cheer output."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer="Bob", bits=100, message="poggers"
        )
        assert event is not None
        assert event.text == format_cheer("Bob", 100, "poggers")

    def test_event_priority_is_normal(self) -> None:
        """Cheer events have normal priority."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer="Bob", bits=50, message="nice"
        )
        assert event is not None
        assert event.priority == "normal"


class TestBuildFollowEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Follow event produced when follows are enabled."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer="Alice")
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when follows are disabled."""
        config = TwitchWatcherConfig(follows_enabled=False)
        event = build_follow_event(config=config, channel="ch", viewer="Alice")
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_follow output."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer="Alice")
        assert event is not None
        assert event.text == format_follow("Alice")

    def test_event_priority_is_normal(self) -> None:
        """Follow events have normal priority."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer="Alice")
        assert event is not None
        assert event.priority == "normal"


class TestBuildAdStartEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Ad start event produced when ads are enabled."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when ads are disabled."""
        config = TwitchWatcherConfig(ads_enabled=False)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_ad_start output."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.text == format_ad_start()

    def test_priority_immediate_when_ads_immediate_enabled(self) -> None:
        """Ad start event has 'immediate' priority when ads_immediate is True."""
        config = TwitchWatcherConfig(ads_enabled=True, ads_immediate=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.priority == "immediate"

    def test_priority_normal_when_ads_immediate_disabled(self) -> None:
        """Ad start event has 'normal' priority when ads_immediate is False."""
        config = TwitchWatcherConfig(ads_enabled=True, ads_immediate=False)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.priority == "normal"


class TestBuildAdEndEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Ad end event produced when ads are enabled."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when ads are disabled."""
        config = TwitchWatcherConfig(ads_enabled=False)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_ad_end output."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None
        assert event.text == format_ad_end()

    def test_priority_immediate_when_ads_immediate_enabled(self) -> None:
        """Ad end event has 'immediate' priority when ads_immediate is True."""
        config = TwitchWatcherConfig(ads_enabled=True, ads_immediate=True)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None
        assert event.priority == "immediate"

    def test_priority_normal_when_ads_immediate_disabled(self) -> None:
        """Ad end event has 'normal' priority when ads_immediate is False."""
        config = TwitchWatcherConfig(ads_enabled=True, ads_immediate=False)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None
        assert event.priority == "normal"
