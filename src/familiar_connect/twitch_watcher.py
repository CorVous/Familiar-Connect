"""Twitch EventSub WebSocket watcher.

Connects to Twitch's EventSub WebSocket (via twitchAPI), registers
listeners for the event types enabled in TwitchWatcherConfig, and
forwards resulting TwitchEvent objects to a trio MemorySendChannel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Protocol

    class _EventWrapper(Protocol):
        """Common structural type of all twitchAPI EventSub event wrappers.

        Every wrapper class (ChannelFollowEvent, ChannelSubscribeEvent, …) has
        a typed .event attribute holding the actual event data.  We can't use
        a concrete base class because twitchAPI's TwitchObject has no .event
        member — it's added per-subclass.  A Protocol lets ty verify the
        attribute access without importing every event class at runtime.
        """

        event: object

import trio

from familiar_connect.twitch import (
    TwitchEvent,
    TwitchWatcherConfig,
    build_ad_start_event,
    build_channel_point_event,
    build_cheer_event,
    build_follow_event,
    build_gift_subscription_event,
    build_resubscription_event,
    build_subscription_event,
)

if TYPE_CHECKING:
    from twitchAPI.eventsub.websocket import EventSubWebsocket
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
# Tier mapping
# ---------------------------------------------------------------------------

_TIER_MAP: dict[str, int] = {"1000": 1, "2000": 2, "3000": 3}


def _tier(tier_str: str) -> int:
    return _TIER_MAP.get(tier_str, 1)


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class TwitchWatcher:
    """Converts twitchAPI EventSub callbacks into TwitchEvent objects."""

    def __init__(
        self,
        *,
        config: TwitchWatcherConfig,
        broadcaster_id: str,
        channel: str,
        moderator_id: str | None = None,
    ) -> None:
        self.config = config
        self.broadcaster_id = broadcaster_id
        self.channel = channel
        self.moderator_id = moderator_id if moderator_id is not None else broadcaster_id

    # ------------------------------------------------------------------
    # Synchronous handlers — accept the .event data object from twitchAPI
    # ------------------------------------------------------------------

    def handle_follow(self, data: ChannelFollowData) -> TwitchEvent | None:
        """Convert a ChannelFollowData into a TwitchEvent."""
        return build_follow_event(
            config=self.config,
            channel=self.channel,
            viewer=data.user_name,
        )

    def handle_subscription(self, data: ChannelSubscribeData) -> TwitchEvent | None:
        """Convert a ChannelSubscribeData into a TwitchEvent.

        Gift subs arrive via handle_gift_subscription; is_gift=True is ignored here.
        """
        if data.is_gift:
            return None
        return build_subscription_event(
            config=self.config,
            channel=self.channel,
            viewer=data.user_name,
            tier=_tier(data.tier),
        )

    def handle_gift_subscription(
        self, data: ChannelSubscriptionGiftData
    ) -> TwitchEvent | None:
        """Convert a ChannelSubscriptionGiftData into a TwitchEvent."""
        gifter = None if data.is_anonymous else data.user_name
        return build_gift_subscription_event(
            config=self.config,
            channel=self.channel,
            gifter=gifter,
            count=data.total,
            tier=_tier(data.tier),
        )

    def handle_resubscription(
        self, data: ChannelSubscriptionMessageData
    ) -> TwitchEvent | None:
        """Convert a ChannelSubscriptionMessageData into a TwitchEvent."""
        return build_resubscription_event(
            config=self.config,
            channel=self.channel,
            viewer=data.user_name,
            months=data.cumulative_months,
            tier=_tier(data.tier),
            message=data.message.text,
        )

    def handle_cheer(self, data: ChannelCheerData) -> TwitchEvent | None:
        """Convert a ChannelCheerData into a TwitchEvent."""
        viewer = None if data.is_anonymous else data.user_name
        return build_cheer_event(
            config=self.config,
            channel=self.channel,
            viewer=viewer,
            bits=data.bits,
            message=data.message,
        )

    def handle_channel_point_redemption(
        self, data: ChannelPointsCustomRewardRedemptionData
    ) -> TwitchEvent | None:
        """Convert a ChannelPointsCustomRewardRedemptionData into a TwitchEvent."""
        return build_channel_point_event(
            config=self.config,
            channel=self.channel,
            viewer=data.user_name,
            redemption_name=data.reward.title,
            user_input=data.user_input or None,
        )

    def handle_ad_break_begin(
        self, _data: ChannelAdBreakBeginData
    ) -> TwitchEvent | None:
        """Convert a ChannelAdBreakBeginData into a TwitchEvent."""
        return build_ad_start_event(config=self.config, channel=self.channel)

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------

    def register_listeners(
        self,
        eventsub: EventSubWebsocket,
        send: trio.MemorySendChannel[TwitchEvent] | None = None,
    ) -> None:
        """Register EventSub callbacks on *eventsub* for all enabled event types.

        *send* is the trio channel that callbacks will forward events to.
        When called from run() this is always provided; the parameter is
        optional only so the method can be called standalone in tests that
        only check which listen_* methods are invoked.
        """
        if self.config.follows_enabled:
            eventsub.listen_channel_follow_v2(
                self.broadcaster_id,
                self.moderator_id,
                self._make_callback(self.handle_follow, send),
            )

        if self.config.subscriptions_enabled:
            eventsub.listen_channel_subscribe(
                self.broadcaster_id,
                self._make_callback(self.handle_subscription, send),
            )
            eventsub.listen_channel_subscription_gift(
                self.broadcaster_id,
                self._make_callback(self.handle_gift_subscription, send),
            )
            eventsub.listen_channel_subscription_message(
                self.broadcaster_id,
                self._make_callback(self.handle_resubscription, send),
            )

        if self.config.cheers_enabled:
            eventsub.listen_channel_cheer(
                self.broadcaster_id,
                self._make_callback(self.handle_cheer, send),
            )

        if self.config.ads_enabled:
            eventsub.listen_channel_ad_break_begin(
                self.broadcaster_id,
                self._make_callback(self.handle_ad_break_begin, send),
            )

        if self.config.redemption_names:
            eventsub.listen_channel_points_custom_reward_redemption_add(
                self.broadcaster_id,
                self._make_callback(self.handle_channel_point_redemption, send),
            )

    @staticmethod
    def _make_callback(
        handler: Callable[[object], TwitchEvent | None],
        send: trio.MemorySendChannel[TwitchEvent] | None,
    ) -> Callable[[_EventWrapper], Awaitable[None]]:
        """Wrap a synchronous handler as an async EventSub callback.

        The returned coroutine accepts a twitchAPI event wrapper, extracts
        .event, calls the handler, and sends any resulting TwitchEvent to
        the captured *send* channel.
        """

        async def callback(event: _EventWrapper) -> None:
            result = handler(event.event)
            if result is not None and send is not None:
                await send.send(result)

        return callback

    # ------------------------------------------------------------------
    # Trio task
    # ------------------------------------------------------------------

    async def run(
        self,
        send: trio.MemorySendChannel[TwitchEvent],
        eventsub: EventSubWebsocket,
    ) -> None:
        """Run the watcher as a trio task.

        Registers all listeners, starts the EventSub connection, then
        sleeps until cancelled — at which point it stops the connection.
        """
        self.register_listeners(eventsub, send)
        try:
            await eventsub.start()
            await trio.sleep_forever()
        finally:
            await eventsub.stop()
