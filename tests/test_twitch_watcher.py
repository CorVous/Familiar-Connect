"""Tests for TwitchWatcher — EventSub WebSocket event handling.

Tests are written red-first. The module under test is
familiar_connect.twitch_watcher, which does not exist yet.

Handler methods accept duck-typed data objects matching the twitchAPI
field shapes (ChannelFollowData, ChannelSubscribeData, etc.) and are
synchronous — all IO lives in run().
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.identity import Author
from familiar_connect.twitch import TwitchEvent, TwitchWatcherConfig
from familiar_connect.twitch_watcher import TwitchWatcher

if TYPE_CHECKING:
    from twitchAPI.object.eventsub import (
        ChannelAdBreakBeginData,
        ChannelCheerData,
        ChannelFollowData,
        ChannelPointsCustomRewardRedemptionData,
        ChannelSubscribeData,
        ChannelSubscriptionGiftData,
        ChannelSubscriptionMessageData,
    )

# ---------------------------------------------------------------------------
# Helpers — duck-typed stand-ins for twitchAPI event data objects
# ---------------------------------------------------------------------------


def _user_fields(
    user_name: str | None,
    user_id: str | None = None,
    user_login: str | None = None,
) -> dict[str, str | None]:
    """Return the three ``user_*`` fields every twitchAPI event data object carries.

    Defaults to a stable ``user_id`` derived from the name so each
    named viewer resolves to the same Author slug across tests.
    """
    if user_id is None:
        user_id = f"uid-{user_name}" if user_name else "uid-anon"
    if user_login is None:
        user_login = user_name.lower() if user_name else None
    return {
        "user_id": user_id,
        "user_login": user_login,
        "user_name": user_name,
    }


def follow_data(user_name: str = "Alice") -> ChannelFollowData:
    return cast("ChannelFollowData", SimpleNamespace(**_user_fields(user_name)))


def subscribe_data(
    user_name: str = "Alice",
    tier: str = "1000",
    *,
    is_gift: bool = False,
) -> ChannelSubscribeData:
    return cast(
        "ChannelSubscribeData",
        SimpleNamespace(**_user_fields(user_name), tier=tier, is_gift=is_gift),
    )


def gift_sub_data(
    user_name: str | None = "Bob",
    total: int = 5,
    tier: str = "1000",
    *,
    is_anonymous: bool = False,
) -> ChannelSubscriptionGiftData:
    return cast(
        "ChannelSubscriptionGiftData",
        SimpleNamespace(
            **_user_fields(user_name),
            total=total,
            tier=tier,
            is_anonymous=is_anonymous,
        ),
    )


def resub_data(
    user_name: str = "Alice",
    cumulative_months: int = 6,
    tier: str = "2000",
    message_text: str = "love this stream",
) -> ChannelSubscriptionMessageData:
    msg = SimpleNamespace(text=message_text)
    return cast(
        "ChannelSubscriptionMessageData",
        SimpleNamespace(
            **_user_fields(user_name),
            cumulative_months=cumulative_months,
            tier=tier,
            message=msg,
        ),
    )


def cheer_data(
    user_name: str | None = "Bob",
    bits: int = 100,
    message: str = "poggers",
    *,
    is_anonymous: bool = False,
) -> ChannelCheerData:
    return cast(
        "ChannelCheerData",
        SimpleNamespace(
            **_user_fields(user_name),
            bits=bits,
            message=message,
            is_anonymous=is_anonymous,
        ),
    )


def redemption_data(
    user_name: str = "Alice",
    reward_title: str = "Talk to Sapphire",
    user_input: str = "",
) -> ChannelPointsCustomRewardRedemptionData:
    reward = SimpleNamespace(title=reward_title)
    return cast(
        "ChannelPointsCustomRewardRedemptionData",
        SimpleNamespace(
            **_user_fields(user_name), reward=reward, user_input=user_input
        ),
    )


def ad_break_data(duration_seconds: int = 60) -> ChannelAdBreakBeginData:
    return cast(
        "ChannelAdBreakBeginData", SimpleNamespace(duration_seconds=duration_seconds)
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTwitchWatcherConstruction:
    def test_stores_config(self) -> None:
        """Watcher stores the provided config."""
        config = TwitchWatcherConfig()
        watcher = TwitchWatcher(config=config, broadcaster_id="123", channel="ch")
        assert watcher.config is config

    def test_stores_broadcaster_id(self) -> None:
        """Watcher stores the broadcaster ID used for EventSub subscriptions."""
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(), broadcaster_id="99999", channel="ch"
        )
        assert watcher.broadcaster_id == "99999"

    def test_stores_channel(self) -> None:
        """Watcher stores the channel name stamped on every TwitchEvent."""
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(),
            broadcaster_id="123",
            channel="sapphire-stream",
        )
        assert watcher.channel == "sapphire-stream"

    def test_moderator_id_defaults_to_broadcaster_id(self) -> None:
        """When no moderator_id is given it falls back to broadcaster_id."""
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(), broadcaster_id="123", channel="ch"
        )
        assert watcher.moderator_id == "123"

    def test_moderator_id_can_be_overridden(self) -> None:
        """A separate moderator_id can be supplied (needed for follow v2)."""
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(),
            broadcaster_id="123",
            channel="ch",
            moderator_id="456",
        )
        assert watcher.moderator_id == "456"


# ---------------------------------------------------------------------------
# handle_follow
# ---------------------------------------------------------------------------


class TestHandleFollow:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Follow handler produces a TwitchEvent when follows are enabled."""
        event = self._watcher(TwitchWatcherConfig(follows_enabled=True)).handle_follow(
            follow_data()
        )
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Follow handler returns None when follows are disabled."""
        event = self._watcher(TwitchWatcherConfig(follows_enabled=False)).handle_follow(
            follow_data()
        )
        assert event is None

    def test_viewer_is_set(self) -> None:
        """Follow event carries the viewer's Author."""
        event = self._watcher(TwitchWatcherConfig(follows_enabled=True)).handle_follow(
            follow_data("Alice")
        )
        assert event is not None
        assert isinstance(event.viewer, Author)
        assert event.viewer.display_name == "Alice"
        assert event.viewer.platform == "twitch"

    def test_channel_is_set(self) -> None:
        """Follow event carries the watcher's channel name."""
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(follows_enabled=True),
            broadcaster_id="1",
            channel="my-stream",
        )
        event = watcher.handle_follow(follow_data())
        assert event is not None
        assert event.channel == "my-stream"


# ---------------------------------------------------------------------------
# handle_subscription
# ---------------------------------------------------------------------------


class TestHandleSubscription:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Subscription handler produces an event when enabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_subscription(subscribe_data())
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Subscription handler returns None when disabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=False)
        ).handle_subscription(subscribe_data())
        assert event is None

    def test_returns_none_for_gift_sub(self) -> None:
        """Gift subs (is_gift=True) are ignored here; they have their own handler."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_subscription(subscribe_data(is_gift=True))
        assert event is None

    def test_tier_string_mapping(self) -> None:
        """Twitch tier strings '1000'/'2000'/'3000' map to integer tiers 1/2/3."""
        watcher = self._watcher(TwitchWatcherConfig(subscriptions_enabled=True))
        for tier_str, label in [
            ("1000", "tier 1"),
            ("2000", "tier 2"),
            ("3000", "tier 3"),
        ]:
            event = watcher.handle_subscription(subscribe_data(tier=tier_str))
            assert event is not None, f"expected event for {tier_str}"
            assert label in event.text


# ---------------------------------------------------------------------------
# handle_gift_subscription
# ---------------------------------------------------------------------------


class TestHandleGiftSubscription:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Gift sub handler produces an event when enabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_gift_subscription(gift_sub_data())
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Gift sub handler returns None when subscriptions are disabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=False)
        ).handle_gift_subscription(gift_sub_data())
        assert event is None

    def test_anonymous_gifter_event_text(self) -> None:
        """Anonymous gift sub uses the anonymous formatter."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_gift_subscription(
            gift_sub_data(user_name=None, total=3, tier="1000", is_anonymous=True)
        )
        assert event is not None
        assert "anonymous" in event.text.lower()

    def test_named_gifter_viewer_is_set(self) -> None:
        """Named gifter event carries the gifter's Author as viewer."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_gift_subscription(gift_sub_data(user_name="Bob"))
        assert event is not None
        assert isinstance(event.viewer, Author)
        assert event.viewer.display_name == "Bob"

    def test_anonymous_gifter_viewer_is_none(self) -> None:
        """Anonymous gift sub has no viewer."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_gift_subscription(gift_sub_data(user_name=None, is_anonymous=True))
        assert event is not None
        assert event.viewer is None

    def test_tier_mapping(self) -> None:
        """Gift sub tier string is mapped to an integer."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_gift_subscription(gift_sub_data(tier="2000"))
        assert event is not None
        assert "tier 2" in event.text


# ---------------------------------------------------------------------------
# handle_resubscription
# ---------------------------------------------------------------------------


class TestHandleResubscription:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Resub handler produces an event when enabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_resubscription(resub_data())
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Resub handler returns None when subscriptions are disabled."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=False)
        ).handle_resubscription(resub_data())
        assert event is None

    def test_event_text(self) -> None:
        """Resub event text matches the formatter output."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_resubscription(
            resub_data(
                user_name="Alice",
                cumulative_months=6,
                tier="2000",
                message_text="love this stream",
            )
        )
        assert event is not None
        assert (
            event.text
            == "Alice has subscribed for 6 months at tier 2 and says: love this stream"
        )

    def test_tier_mapping(self) -> None:
        """Resub tier string is mapped to an integer."""
        event = self._watcher(
            TwitchWatcherConfig(subscriptions_enabled=True)
        ).handle_resubscription(resub_data(tier="3000"))
        assert event is not None
        assert "tier 3" in event.text


# ---------------------------------------------------------------------------
# handle_cheer
# ---------------------------------------------------------------------------


class TestHandleCheer:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Cheer handler produces an event when enabled."""
        event = self._watcher(TwitchWatcherConfig(cheers_enabled=True)).handle_cheer(
            cheer_data()
        )
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Cheer handler returns None when cheers are disabled."""
        event = self._watcher(TwitchWatcherConfig(cheers_enabled=False)).handle_cheer(
            cheer_data()
        )
        assert event is None

    def test_anonymous_cheerer_event_text(self) -> None:
        """Anonymous cheer uses the anonymous formatter."""
        event = self._watcher(TwitchWatcherConfig(cheers_enabled=True)).handle_cheer(
            cheer_data(user_name=None, bits=500, message="hype", is_anonymous=True)
        )
        assert event is not None
        assert (
            event.text
            == "An anonymous cheerer has cheered with 500 bits and says: hype"
        )

    def test_named_viewer_is_set(self) -> None:
        """Named cheer event carries the viewer's Author."""
        event = self._watcher(TwitchWatcherConfig(cheers_enabled=True)).handle_cheer(
            cheer_data(user_name="Bob")
        )
        assert event is not None
        assert isinstance(event.viewer, Author)
        assert event.viewer.display_name == "Bob"

    def test_anonymous_viewer_is_none(self) -> None:
        """Anonymous cheer has no viewer."""
        event = self._watcher(TwitchWatcherConfig(cheers_enabled=True)).handle_cheer(
            cheer_data(user_name=None, is_anonymous=True)
        )
        assert event is not None
        assert event.viewer is None


# ---------------------------------------------------------------------------
# handle_channel_point_redemption
# ---------------------------------------------------------------------------


class TestHandleChannelPointRedemption:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_for_listed_redemption(self) -> None:
        """An allowed redemption name produces an event."""
        event = self._watcher(
            TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        ).handle_channel_point_redemption(
            redemption_data(reward_title="Talk to Sapphire")
        )
        assert isinstance(event, TwitchEvent)

    def test_returns_none_for_unlisted_redemption(self) -> None:
        """An unlisted redemption name produces no event."""
        event = self._watcher(
            TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        ).handle_channel_point_redemption(
            redemption_data(reward_title="Something Else")
        )
        assert event is None

    def test_returns_none_when_list_empty(self) -> None:
        """An empty allow-list suppresses all redemptions."""
        event = self._watcher(
            TwitchWatcherConfig(redemption_names=[])
        ).handle_channel_point_redemption(
            redemption_data(reward_title="Talk to Sapphire")
        )
        assert event is None

    def test_event_text_with_user_input(self) -> None:
        """User input text is included in the event text."""
        event = self._watcher(
            TwitchWatcherConfig(redemption_names=["Talk to Sapphire"])
        ).handle_channel_point_redemption(
            redemption_data(
                user_name="Alice",
                reward_title="Talk to Sapphire",
                user_input="hello!",
            )
        )
        assert event is not None
        assert event.text == "Alice has redeemed Talk to Sapphire and says: hello!"

    def test_event_text_without_user_input(self) -> None:
        """Empty user_input omits the 'and says' clause (coerced to None)."""
        event = self._watcher(
            TwitchWatcherConfig(redemption_names=["Hydrate"])
        ).handle_channel_point_redemption(
            redemption_data(user_name="Bob", reward_title="Hydrate", user_input="")
        )
        assert event is not None
        assert event.text == "Bob has redeemed Hydrate"


# ---------------------------------------------------------------------------
# Ad break begin handling
# ---------------------------------------------------------------------------


class TestHandleAdBreakBegin:
    def _watcher(self, config: TwitchWatcherConfig) -> TwitchWatcher:
        return TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

    def test_returns_event_when_enabled(self) -> None:
        """Ad break handler produces an event when ads are enabled."""
        event = self._watcher(
            TwitchWatcherConfig(ads_enabled=True)
        ).handle_ad_break_begin(ad_break_data())
        assert isinstance(event, TwitchEvent)

    def test_returns_none_when_disabled(self) -> None:
        """Ad break handler returns None when ads are disabled."""
        event = self._watcher(
            TwitchWatcherConfig(ads_enabled=False)
        ).handle_ad_break_begin(ad_break_data())
        assert event is None


# ---------------------------------------------------------------------------
# register_listeners — wires twitchAPI callbacks based on config
# ---------------------------------------------------------------------------


class TestRegisterListeners:
    def _mock_eventsub(self) -> MagicMock:
        """Return a mock EventSubWebsocket with all listen_* as AsyncMocks."""
        mock = MagicMock()
        mock.listen_channel_follow_v2 = AsyncMock(return_value="sub-id-follow")
        mock.listen_channel_subscribe = AsyncMock(return_value="sub-id-sub")
        mock.listen_channel_subscription_gift = AsyncMock(return_value="sub-id-gift")
        mock.listen_channel_subscription_message = AsyncMock(
            return_value="sub-id-resub"
        )
        mock.listen_channel_cheer = AsyncMock(return_value="sub-id-cheer")
        mock.listen_channel_points_custom_reward_redemption_add = AsyncMock(
            return_value="sub-id-redeem"
        )
        mock.listen_channel_ad_break_begin = AsyncMock(return_value="sub-id-ad")
        return mock

    @pytest.mark.asyncio
    async def test_follow_listener_registered_when_enabled(self) -> None:
        """listen_channel_follow_v2 is called when follows_enabled=True."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(follows_enabled=True),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_follow_v2.assert_called_once()

    @pytest.mark.asyncio
    async def test_follow_listener_not_registered_when_disabled(self) -> None:
        """listen_channel_follow_v2 is NOT called when follows_enabled=False."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(follows_enabled=False),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_follow_v2.assert_not_called()

    @pytest.mark.asyncio
    async def test_follow_listener_uses_broadcaster_and_moderator_ids(self) -> None:
        """Follow listener is called with broadcaster_id and moderator_id."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(follows_enabled=True),
            broadcaster_id="123",
            channel="ch",
            moderator_id="456",
        )
        await watcher.register_listeners(eventsub)
        args = eventsub.listen_channel_follow_v2.call_args
        assert args[0][0] == "123"
        assert args[0][1] == "456"

    @pytest.mark.asyncio
    async def test_subscription_listeners_registered_when_enabled(self) -> None:
        """All three sub listeners are registered when subscriptions_enabled=True."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(subscriptions_enabled=True),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_subscribe.assert_called_once()
        eventsub.listen_channel_subscription_gift.assert_called_once()
        eventsub.listen_channel_subscription_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscription_listeners_not_registered_when_disabled(self) -> None:
        """Sub listeners are NOT called when subscriptions_enabled=False."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(subscriptions_enabled=False),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_subscribe.assert_not_called()
        eventsub.listen_channel_subscription_gift.assert_not_called()
        eventsub.listen_channel_subscription_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_cheer_listener_registered_when_enabled(self) -> None:
        """listen_channel_cheer is called when cheers_enabled=True."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(cheers_enabled=True),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_cheer.assert_called_once()

    @pytest.mark.asyncio
    async def test_cheer_listener_not_registered_when_disabled(self) -> None:
        """listen_channel_cheer is NOT called when cheers_enabled=False."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(cheers_enabled=False),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_cheer.assert_not_called()

    @pytest.mark.asyncio
    async def test_ad_listener_registered_when_enabled(self) -> None:
        """listen_channel_ad_break_begin is called when ads_enabled=True."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(ads_enabled=True),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_ad_break_begin.assert_called_once()

    @pytest.mark.asyncio
    async def test_ad_listener_not_registered_when_disabled(self) -> None:
        """listen_channel_ad_break_begin is NOT called when ads_enabled=False."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(ads_enabled=False),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_ad_break_begin.assert_not_called()

    @pytest.mark.asyncio
    async def test_redemption_listener_registered_when_list_nonempty(self) -> None:
        """Redemption listener is registered when redemption_names is non-empty."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(redemption_names=["Talk to Sapphire"]),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_points_custom_reward_redemption_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_redemption_listener_not_registered_when_list_empty(self) -> None:
        """Redemption listener is NOT registered when redemption_names is empty."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(redemption_names=[]),
            broadcaster_id="123",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        eventsub.listen_channel_points_custom_reward_redemption_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_listeners_use_broadcaster_id(self) -> None:
        """Every registered listener is called with the broadcaster_id as first arg."""
        eventsub = self._mock_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(
                subscriptions_enabled=True,
                cheers_enabled=True,
                ads_enabled=True,
                redemption_names=["Test"],
            ),
            broadcaster_id="999",
            channel="ch",
        )
        await watcher.register_listeners(eventsub)
        for mock_method in [
            eventsub.listen_channel_subscribe,
            eventsub.listen_channel_subscription_gift,
            eventsub.listen_channel_subscription_message,
            eventsub.listen_channel_cheer,
            eventsub.listen_channel_ad_break_begin,
            eventsub.listen_channel_points_custom_reward_redemption_add,
        ]:
            assert mock_method.call_args[0][0] == "999", (
                f"{mock_method} not called with broadcaster_id"
            )


# ---------------------------------------------------------------------------
# run() — starts eventsub, forwards events to the asyncio.Queue,
#          stops cleanly on cancellation
# ---------------------------------------------------------------------------


# Deadline used by _run_briefly — long enough for the AsyncMock listeners +
# start() to resolve and for the watcher to reach Event().wait(), short enough
# to keep the test fast. Tune up if tests flake on very slow runners.
_BRIEF_RUN_DEADLINE_S = 0.05


async def _run_briefly(
    watcher: TwitchWatcher,
    send: asyncio.Queue[TwitchEvent],
    eventsub: MagicMock,
) -> None:
    """Run watcher.run() long enough to register listeners, then cancel.

    Replaces the earlier ``with trio.move_on_after(0)`` pattern. The
    AsyncMock listeners + start() return immediately, so the watcher
    reaches its ``Event().wait()`` sleep well within the deadline; the
    deadline then triggers the ``finally: await eventsub.stop()`` path.
    """
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(_BRIEF_RUN_DEADLINE_S):
            await watcher.run(send, eventsub)


class TestRun:
    def _make_eventsub(self) -> MagicMock:
        """Mock EventSubWebsocket with async listen_*, start, and stop."""
        mock = MagicMock()
        mock.listen_channel_follow_v2 = AsyncMock(return_value="id")
        mock.listen_channel_subscribe = AsyncMock(return_value="id")
        mock.listen_channel_subscription_gift = AsyncMock(return_value="id")
        mock.listen_channel_subscription_message = AsyncMock(return_value="id")
        mock.listen_channel_cheer = AsyncMock(return_value="id")
        mock.listen_channel_points_custom_reward_redemption_add = AsyncMock(
            return_value="id"
        )
        mock.listen_channel_ad_break_begin = AsyncMock(return_value="id")
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_run_calls_eventsub_start(self) -> None:
        """run() calls eventsub.start() to begin receiving events."""
        eventsub = self._make_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(),
            broadcaster_id="1",
            channel="ch",
        )
        send: asyncio.Queue[TwitchEvent] = asyncio.Queue(maxsize=16)
        await _run_briefly(watcher, send, eventsub)
        eventsub.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_calls_eventsub_stop_on_cancel(self) -> None:
        """run() calls eventsub.stop() when cancelled."""
        eventsub = self._make_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(),
            broadcaster_id="1",
            channel="ch",
        )
        send: asyncio.Queue[TwitchEvent] = asyncio.Queue(maxsize=16)
        await _run_briefly(watcher, send, eventsub)
        eventsub.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_registers_listeners(self) -> None:
        """run() calls register_listeners before starting."""
        eventsub = self._make_eventsub()
        watcher = TwitchWatcher(
            config=TwitchWatcherConfig(follows_enabled=True),
            broadcaster_id="1",
            channel="ch",
        )
        send: asyncio.Queue[TwitchEvent] = asyncio.Queue(maxsize=16)
        await _run_briefly(watcher, send, eventsub)
        eventsub.listen_channel_follow_v2.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_sends_event_to_channel(self) -> None:
        """When a registered callback fires, its event reaches the queue."""
        eventsub = self._make_eventsub()
        config = TwitchWatcherConfig(follows_enabled=True)
        watcher = TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

        send: asyncio.Queue[TwitchEvent] = asyncio.Queue(maxsize=16)

        # Capture the callback registered for follows
        captured_callback: list = []

        def capture_follow_cb(
            _broadcaster_id: str, _moderator_id: str, cb: object
        ) -> str:
            captured_callback.append(cb)
            return "sub-id"

        eventsub.listen_channel_follow_v2.side_effect = capture_follow_cb

        # Start the watcher and immediately cancel it; callbacks are captured first
        await _run_briefly(watcher, send, eventsub)

        assert captured_callback, "no follow callback was registered"

        # Fire the callback with a mock event (twitchAPI wraps data in .event)
        mock_event = SimpleNamespace(event=follow_data("Alice"))
        await captured_callback[0](mock_event)

        result = send.get_nowait()
        assert isinstance(result, TwitchEvent)
        assert isinstance(result.viewer, Author)
        assert result.viewer.display_name == "Alice"
        assert result.text == "Alice has followed the channel"

    @pytest.mark.asyncio
    async def test_callback_for_disabled_event_does_not_send(self) -> None:
        """A callback that returns None (disabled config) sends nothing."""
        config = TwitchWatcherConfig(follows_enabled=False)
        watcher = TwitchWatcher(config=config, broadcaster_id="1", channel="ch")

        send: asyncio.Queue[TwitchEvent] = asyncio.Queue(maxsize=16)

        # Manually invoke the follow handler (disabled) and verify the queue is empty
        result = watcher.handle_follow(follow_data("Alice"))
        assert result is None

        with pytest.raises(asyncio.QueueEmpty):
            send.get_nowait()
