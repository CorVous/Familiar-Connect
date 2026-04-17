"""Tests for Twitch channel event watcher.

Tests are written red-first against the interface described in docs/guides/twitch.md.
The module under test is familiar_connect.twitch, which does not exist yet.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from familiar_connect.identity import Author
from familiar_connect.llm import Message
from familiar_connect.twitch import (
    TwitchEvent,
    TwitchWatcherConfig,
    build_ad_end_event,
    build_ad_start_event,
    build_channel_point_event,
    build_cheer_event,
    build_follow_event,
    build_gift_subscription_event,
    build_resubscription_event,
    build_subscription_event,
    format_ad_end,
    format_ad_start,
    format_channel_point_redemption,
    format_cheer,
    format_follow,
    format_gift_subscription,
    format_resubscription,
    format_subscription,
)

_ALICE = Author.from_twitch(user_id="uid-Alice", user_login="alice", user_name="Alice")
_BOB = Author.from_twitch(user_id="uid-Bob", user_login="bob", user_name="Bob")
_CAROL = Author.from_twitch(user_id="uid-Carol", user_login="carol", user_name="Carol")

# ---------------------------------------------------------------------------
# TwitchEvent shape
# ---------------------------------------------------------------------------


class TestTwitchEvent:
    def test_required_fields(self) -> None:
        """TwitchEvent stores channel, text, and priority."""
        event = TwitchEvent(
            channel="sapphire-stream",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.channel == "sapphire-stream"
        assert event.text == "Alice has followed the channel"
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

    def test_viewer_field_present_when_set(self) -> None:
        """TwitchEvent optionally carries the name of the associated viewer."""
        event = TwitchEvent(
            channel="ch",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
            viewer=_ALICE,
        )
        assert event.viewer == _ALICE

    def test_viewer_field_defaults_to_none(self) -> None:
        """Viewer defaults to None for events with no associated person."""
        event = TwitchEvent(
            channel="ch",
            text="An ad has begun on the channel",
            priority="immediate",
            timestamp=datetime.now(UTC),
        )
        assert event.viewer is None


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


class TestFormatResubscription:
    def test_basic_resub(self) -> None:
        """Resub with all fields formats correctly."""
        text = format_resubscription("Alice", 6, 2, "love this stream")
        assert (
            text
            == "Alice has subscribed for 6 months at tier 2 and says: love this stream"
        )

    def test_one_month(self) -> None:
        """Singular month phrasing for 1 month."""
        text = format_resubscription("Bob", 1, 1, "woo")
        assert "1 month" in text


class TestFormatCheer:
    def test_named_cheerer_with_message(self) -> None:
        """Named cheerer with a message formats correctly."""
        text = format_cheer("Bob", 100, "poggers")
        assert text == "Bob has cheered with 100 bits and says: poggers"

    def test_anonymous_cheerer(self) -> None:
        """Anonymous cheerer uses 'An anonymous cheerer'."""
        text = format_cheer(None, 500, "hype")
        assert text == "An anonymous cheerer has cheered with 500 bits and says: hype"


class TestFormatFollow:
    def test_follow_message(self) -> None:
        """Follow event formats correctly."""
        text = format_follow("Alice")
        assert text == "Alice has followed the channel"


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
    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("subscriptions_enabled", True),
            ("cheers_enabled", True),
            ("follows_enabled", True),
            ("ads_enabled", True),
            ("ads_immediate", True),
            ("redemption_names", []),
        ],
    )
    def test_defaults(self, attr: str, expected: object) -> None:
        """All boolean flags default to True; redemption_names defaults to []."""
        config = TwitchWatcherConfig()
        assert getattr(config, attr) == expected

    @pytest.mark.parametrize(
        "field",
        [
            "subscriptions_enabled",
            "cheers_enabled",
            "follows_enabled",
            "ads_enabled",
            "ads_immediate",
        ],
    )
    def test_can_disable(self, field: str) -> None:
        """Each boolean flag can be toggled off at construction."""
        config = TwitchWatcherConfig(**{field: False})  # ty: ignore[invalid-argument-type]
        assert getattr(config, field) is False

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
            viewer=_ALICE,
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
            viewer=_ALICE,
            redemption_name="Something Else",
        )
        assert event is None

    def test_returns_none_when_list_is_empty(self) -> None:
        """An empty redemption list means no redemptions produce events."""
        config = TwitchWatcherConfig(redemption_names=[])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer=_ALICE,
            redemption_name="Talk to Sapphire",
        )
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches what format_channel_point_redemption produces."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer=_ALICE,
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
            viewer=_ALICE,
            redemption_name="Talk to Sapphire",
        )
        assert event is not None
        assert event.priority == "normal"

    def test_event_viewer_is_set(self) -> None:
        """Channel point redemption event carries the viewer's name."""
        config = TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        event = build_channel_point_event(
            config=config,
            channel="ch",
            viewer=_ALICE,
            redemption_name="Talk to Sapphire",
        )
        assert event is not None
        assert event.viewer == _ALICE

    def test_event_channel_set_correctly(self) -> None:
        """The channel field reflects the provided channel."""
        config = TwitchWatcherConfig(redemption_names=["Test"])
        event = build_channel_point_event(
            config=config, channel="my-channel", viewer=_ALICE, redemption_name="Test"
        )
        assert event is not None
        assert event.channel == "my-channel"

    def test_event_timestamp_is_utc_aware(self) -> None:
        """The event timestamp is UTC-aware."""
        config = TwitchWatcherConfig(redemption_names=["Test"])
        event = build_channel_point_event(
            config=config, channel="ch", viewer=_ALICE, redemption_name="Test"
        )
        assert event is not None
        assert event.timestamp.tzinfo is not None


class TestBuildSubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Subscription event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer=_ALICE, tier=1
        )
        assert event is not None
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when subscriptions are disabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        event = build_subscription_event(
            config=config, channel="ch", viewer=_ALICE, tier=1
        )
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_subscription output."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer=_ALICE, tier=2
        )
        assert event is not None
        assert event.text == format_subscription("Alice", 2)

    def test_event_priority_is_normal(self) -> None:
        """Subscription events have normal priority."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer=_ALICE, tier=1
        )
        assert event is not None
        assert event.priority == "normal"

    def test_event_viewer_is_set(self) -> None:
        """Subscription event carries the viewer's name."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_subscription_event(
            config=config, channel="ch", viewer=_ALICE, tier=1
        )
        assert event is not None
        assert event.viewer == _ALICE


class TestBuildGiftSubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Gift sub event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=_BOB, count=5, tier=1
        )
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when subscriptions are disabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=False)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=_BOB, count=5, tier=1
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
            config=config, channel="ch", gifter=_BOB, count=5, tier=1
        )
        assert event is not None
        assert event.text == format_gift_subscription("Bob", 5, 1)

    def test_named_gifter_viewer_is_set(self) -> None:
        """Gift sub event carries the gifter's name when not anonymous."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=_BOB, count=5, tier=1
        )
        assert event is not None
        assert event.viewer == _BOB

    def test_anonymous_gifter_viewer_is_none(self) -> None:
        """Gift sub event has no viewer when the gifter is anonymous."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=None, count=3, tier=1
        )
        assert event is not None
        assert event.viewer is None


class TestBuildResubscriptionEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Resub event produced when subscriptions are enabled."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer=_ALICE,
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
            viewer=_ALICE,
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
            viewer=_ALICE,
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
            viewer=_ALICE,
            months=3,
            tier=1,
            message="woo",
        )
        assert event is not None
        assert event.priority == "normal"

    def test_event_viewer_is_set(self) -> None:
        """Resubscription event carries the viewer's name."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_resubscription_event(
            config=config,
            channel="ch",
            viewer=_ALICE,
            months=6,
            tier=2,
            message="love this stream",
        )
        assert event is not None
        assert event.viewer == _ALICE


class TestBuildCheerEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Cheer event produced when cheers are enabled."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=_BOB, bits=100, message="poggers"
        )
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when cheers are disabled."""
        config = TwitchWatcherConfig(cheers_enabled=False)
        event = build_cheer_event(
            config=config, channel="ch", viewer=_BOB, bits=100, message="poggers"
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
            config=config, channel="ch", viewer=_BOB, bits=100, message="poggers"
        )
        assert event is not None
        assert event.text == format_cheer("Bob", 100, "poggers")

    def test_event_priority_is_normal(self) -> None:
        """Cheer events have normal priority."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=_BOB, bits=50, message="nice"
        )
        assert event is not None
        assert event.priority == "normal"

    def test_named_viewer_is_set(self) -> None:
        """Cheer event carries the viewer's name when not anonymous."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=_BOB, bits=100, message="poggers"
        )
        assert event is not None
        assert event.viewer == _BOB

    def test_anonymous_viewer_is_none(self) -> None:
        """Cheer event has no viewer when the cheerer is anonymous."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=None, bits=500, message="hype"
        )
        assert event is not None
        assert event.viewer is None


class TestBuildFollowEvent:
    def test_produces_event_when_enabled(self) -> None:
        """Follow event produced when follows are enabled."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer=_ALICE)
        assert event is not None

    def test_returns_none_when_disabled(self) -> None:
        """No event produced when follows are disabled."""
        config = TwitchWatcherConfig(follows_enabled=False)
        event = build_follow_event(config=config, channel="ch", viewer=_ALICE)
        assert event is None

    def test_event_text_matches_formatter(self) -> None:
        """Event text matches format_follow output."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer=_ALICE)
        assert event is not None
        assert event.text == format_follow("Alice")

    def test_event_priority_is_normal(self) -> None:
        """Follow events have normal priority."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer=_ALICE)
        assert event is not None
        assert event.priority == "normal"

    def test_event_viewer_is_set(self) -> None:
        """Follow event carries the viewer's name."""
        config = TwitchWatcherConfig(follows_enabled=True)
        event = build_follow_event(config=config, channel="ch", viewer=_ALICE)
        assert event is not None
        assert event.viewer == _ALICE


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

    def test_event_viewer_is_none(self) -> None:
        """Ad start events have no associated person."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.viewer is None


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

    def test_event_viewer_is_none(self) -> None:
        """Ad end events have no associated person."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None
        assert event.viewer is None


# ---------------------------------------------------------------------------
# TwitchEvent → LLM Message conversion
# ---------------------------------------------------------------------------


class TestTwitchEventToMessage:
    def test_to_message_returns_message_instance(self) -> None:
        """to_message() returns a Message compatible with the LLM client."""
        event = TwitchEvent(
            channel="ch",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        msg = event.to_message()
        assert isinstance(msg, Message)

    def test_to_message_role_is_user(self) -> None:
        """Twitch events enter the conversation as user-role messages."""
        event = TwitchEvent(
            channel="ch",
            text="Bob has cheered with 100 bits and says: poggers",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.to_message().role == "user"

    def test_to_message_content_has_twitch_prefix(self) -> None:
        """Content is prefixed with '[Twitch] ' so the LLM knows the source."""
        event = TwitchEvent(
            channel="ch",
            text="Alice has subscribed at tier 1",
            priority="normal",
            timestamp=datetime.now(UTC),
        )
        assert event.to_message().content == "[Twitch] Alice has subscribed at tier 1"

    def test_to_message_name_is_viewer_when_present(self) -> None:
        """Events with a viewer use the viewer's name as the message name."""
        event = TwitchEvent(
            channel="ch",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
            viewer=_ALICE,
        )
        assert event.to_message().name == "Alice"

    def test_to_message_name_falls_back_to_twitch_when_no_viewer(self) -> None:
        """Events without a viewer (ads, anonymous) fall back to 'Twitch'."""
        event = TwitchEvent(
            channel="ch",
            text="An ad has begun on the channel",
            priority="immediate",
            timestamp=datetime.now(UTC),
            viewer=None,
        )
        assert event.to_message().name == "Twitch"

    def test_to_message_is_serialisable_with_viewer(self) -> None:
        """Viewer-named messages serialise correctly for the API."""
        event = TwitchEvent(
            channel="ch",
            text="Alice has followed the channel",
            priority="normal",
            timestamp=datetime.now(UTC),
            viewer=_ALICE,
        )
        d = event.to_message().to_dict()
        assert d == {
            "role": "user",
            "content": "[Twitch] Alice has followed the channel",
            "name": "Alice",
        }

    def test_to_message_is_serialisable_without_viewer(self) -> None:
        """No-viewer messages serialise with 'Twitch' as the name."""
        event = TwitchEvent(
            channel="ch",
            text="An ad has begun on the channel",
            priority="immediate",
            timestamp=datetime.now(UTC),
            viewer=None,
        )
        d = event.to_message().to_dict()
        assert d == {
            "role": "user",
            "content": "[Twitch] An ad has begun on the channel",
            "name": "Twitch",
        }

    def test_to_message_priority_not_leaked_into_message(self) -> None:
        """Priority stays on the event; Message has no priority field."""
        event = TwitchEvent(
            channel="ch",
            text="An ad has begun on the channel",
            priority="immediate",
            timestamp=datetime.now(UTC),
        )
        msg = event.to_message()
        assert not hasattr(msg, "priority")

    def test_immediate_event_priority_readable_before_conversion(self) -> None:
        """Callers can inspect priority on the event before calling to_message()."""
        config = TwitchWatcherConfig(ads_enabled=True, ads_immediate=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.priority == "immediate"
        # Conversion still works regardless of priority
        assert isinstance(event.to_message(), Message)

    def test_normal_event_round_trips_through_llm_message_list(self) -> None:
        """A batch of Twitch events can be converted and included in an LLM call."""
        config = TwitchWatcherConfig(
            subscriptions_enabled=True,
            follows_enabled=True,
            redemption_names=["Talk to Sapphire"],
        )
        raw_events = [
            build_follow_event(config=config, channel="ch", viewer=_ALICE),
            build_subscription_event(config=config, channel="ch", viewer=_BOB, tier=1),
            build_channel_point_event(
                config=config,
                channel="ch",
                viewer=_CAROL,
                redemption_name="Talk to Sapphire",
                user_input="hello!",
            ),
        ]
        messages = [e.to_message() for e in raw_events if e is not None]
        assert len(messages) == 3
        assert all(isinstance(m, Message) for m in messages)
        assert all(m.role == "user" for m in messages)
        # Each viewer-associated event carries that viewer's name
        names = [m.name for m in messages]
        assert names == ["Alice", "Bob", "Carol"]
        # Every content line is prefixed to identify the Twitch source
        contents = [m.content for m in messages]
        assert all(c.startswith("[Twitch] ") for c in contents)
        assert any("Alice" in c for c in contents)
        assert any("Bob" in c for c in contents)
        assert any("Carol" in c and "hello!" in c for c in contents)

    def test_anonymous_cheer_falls_back_to_twitch(self) -> None:
        """An anonymous cheer has no viewer name so it uses 'Twitch'."""
        config = TwitchWatcherConfig(cheers_enabled=True)
        event = build_cheer_event(
            config=config, channel="ch", viewer=None, bits=500, message="hype"
        )
        assert event is not None
        assert event.to_message().name == "Twitch"

    def test_anonymous_gift_sub_falls_back_to_twitch(self) -> None:
        """An anonymous gift sub has no viewer name so it uses 'Twitch'."""
        config = TwitchWatcherConfig(subscriptions_enabled=True)
        event = build_gift_subscription_event(
            config=config, channel="ch", gifter=None, count=3, tier=1
        )
        assert event is not None
        assert event.to_message().name == "Twitch"

    def test_ad_start_uses_twitch_as_name(self) -> None:
        """Ad start events have no person; they use 'Twitch' as the name."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_start_event(config=config, channel="ch")
        assert event is not None
        assert event.to_message().name == "Twitch"

    def test_ad_end_uses_twitch_as_name(self) -> None:
        """Ad end events have no person; they use 'Twitch' as the name."""
        config = TwitchWatcherConfig(ads_enabled=True)
        event = build_ad_end_event(config=config, channel="ch")
        assert event is not None
        assert event.to_message().name == "Twitch"
