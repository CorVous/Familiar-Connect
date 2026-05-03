"""Twitch EventSub WebSocket watcher.

Connects to Twitch EventSub WebSocket (via twitchAPI), registers
listeners for events enabled in TwitchWatcherConfig, forwards
resulting TwitchEvent objects to an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from twitchAPI.eventsub.websocket import EventSubWebsocket
    from twitchAPI.object.eventsub import (
        ChannelAdBreakBeginData,
        ChannelAdBreakBeginEvent,
        ChannelCheerData,
        ChannelCheerEvent,
        ChannelFollowData,
        ChannelFollowEvent,
        ChannelPointsCustomRewardRedemptionAddEvent,
        ChannelPointsCustomRewardRedemptionData,
        ChannelSubscribeData,
        ChannelSubscribeEvent,
        ChannelSubscriptionGiftData,
        ChannelSubscriptionGiftEvent,
        ChannelSubscriptionMessageData,
        ChannelSubscriptionMessageEvent,
    )


from familiar_connect.identity import Author
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
    # sync handlers — accept .event data object from twitchAPI
    # ------------------------------------------------------------------

    def handle_follow(self, data: ChannelFollowData) -> TwitchEvent | None:
        return build_follow_event(
            config=self.config,
            channel=self.channel,
            viewer=_build_author(data.user_id, data.user_login, data.user_name),
        )

    def handle_subscription(self, data: ChannelSubscribeData) -> TwitchEvent | None:
        """Subscription event; gift subs arrive via handle_gift_subscription."""
        if data.is_gift:
            return None
        return build_subscription_event(
            config=self.config,
            channel=self.channel,
            viewer=_build_author(data.user_id, data.user_login, data.user_name),
            tier=_tier(data.tier),
        )

    def handle_gift_subscription(
        self, data: ChannelSubscriptionGiftData
    ) -> TwitchEvent | None:
        gifter = (
            None
            if data.is_anonymous
            else _build_author(data.user_id, data.user_login, data.user_name)
        )
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
        """Resub event.

        cumulative_months Optional on Twitch side; falls back to
        duration_months (always present).
        """
        months = (
            data.cumulative_months
            if data.cumulative_months is not None
            else data.duration_months
        )
        return build_resubscription_event(
            config=self.config,
            channel=self.channel,
            viewer=_build_author(data.user_id, data.user_login, data.user_name),
            months=months,
            tier=_tier(data.tier),
            message=data.message.text,
        )

    def handle_cheer(self, data: ChannelCheerData) -> TwitchEvent | None:
        viewer = (
            None
            if data.is_anonymous
            else _build_author(data.user_id, data.user_login, data.user_name)
        )
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
        return build_channel_point_event(
            config=self.config,
            channel=self.channel,
            viewer=_build_author(data.user_id, data.user_login, data.user_name),
            redemption_name=data.reward.title,
            user_input=data.user_input or None,
        )

    def handle_ad_break_begin(
        self, _data: ChannelAdBreakBeginData
    ) -> TwitchEvent | None:
        return build_ad_start_event(config=self.config, channel=self.channel)

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------

    async def register_listeners(
        self,
        eventsub: EventSubWebsocket,
        send: asyncio.Queue[TwitchEvent] | None = None,
    ) -> None:
        """Register EventSub callbacks on ``eventsub`` for all enabled event types.

        send: asyncio.Queue callbacks forward events to. Always provided
        from :meth:`run`; optional for standalone test calls that only
        check which ``listen_*`` methods are invoked.
        """
        if self.config.follows_enabled:
            await eventsub.listen_channel_follow_v2(
                self.broadcaster_id,
                self.moderator_id,
                self._follow_callback(send),
            )

        if self.config.subscriptions_enabled:
            await eventsub.listen_channel_subscribe(
                self.broadcaster_id,
                self._subscription_callback(send),
            )
            await eventsub.listen_channel_subscription_gift(
                self.broadcaster_id,
                self._gift_subscription_callback(send),
            )
            await eventsub.listen_channel_subscription_message(
                self.broadcaster_id,
                self._resubscription_callback(send),
            )

        if self.config.cheers_enabled:
            await eventsub.listen_channel_cheer(
                self.broadcaster_id,
                self._cheer_callback(send),
            )

        if self.config.ads_enabled:
            await eventsub.listen_channel_ad_break_begin(
                self.broadcaster_id,
                self._ad_break_begin_callback(send),
            )

        if self.config.redemption_names:
            await eventsub.listen_channel_points_custom_reward_redemption_add(
                self.broadcaster_id,
                self._channel_point_redemption_callback(send),
            )

    # ------------------------------------------------------------------
    # Typed callback factories — one per event type so ty can verify the
    # callback signatures match what each listen_* method expects.
    # ------------------------------------------------------------------

    def _follow_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelFollowEvent], Awaitable[None]]:
        async def cb(event: ChannelFollowEvent) -> None:
            await _send_if_present(self.handle_follow(event.event), send)

        return cb

    def _subscription_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelSubscribeEvent], Awaitable[None]]:
        async def cb(event: ChannelSubscribeEvent) -> None:
            await _send_if_present(self.handle_subscription(event.event), send)

        return cb

    def _gift_subscription_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelSubscriptionGiftEvent], Awaitable[None]]:
        async def cb(event: ChannelSubscriptionGiftEvent) -> None:
            await _send_if_present(self.handle_gift_subscription(event.event), send)

        return cb

    def _resubscription_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelSubscriptionMessageEvent], Awaitable[None]]:
        async def cb(event: ChannelSubscriptionMessageEvent) -> None:
            await _send_if_present(self.handle_resubscription(event.event), send)

        return cb

    def _cheer_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelCheerEvent], Awaitable[None]]:
        async def cb(event: ChannelCheerEvent) -> None:
            await _send_if_present(self.handle_cheer(event.event), send)

        return cb

    def _channel_point_redemption_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelPointsCustomRewardRedemptionAddEvent], Awaitable[None]]:
        async def cb(event: ChannelPointsCustomRewardRedemptionAddEvent) -> None:
            await _send_if_present(
                self.handle_channel_point_redemption(event.event), send
            )

        return cb

    def _ad_break_begin_callback(
        self, send: asyncio.Queue[TwitchEvent] | None
    ) -> Callable[[ChannelAdBreakBeginEvent], Awaitable[None]]:
        async def cb(event: ChannelAdBreakBeginEvent) -> None:
            await _send_if_present(self.handle_ad_break_begin(event.event), send)

        return cb

    # ------------------------------------------------------------------
    # Event loop task
    # ------------------------------------------------------------------

    async def run(
        self,
        send: asyncio.Queue[TwitchEvent],
        eventsub: EventSubWebsocket,
    ) -> None:
        """Run the watcher as an asyncio task.

        Registers all listeners, starts the EventSub connection, then
        sleeps until cancelled — at which point it stops the connection.
        """
        await self.register_listeners(eventsub, send)
        try:
            await eventsub.start()
            # Sleep forever; cancellation from the parent TaskGroup ends this.
            await asyncio.Event().wait()
        finally:
            await eventsub.stop()


async def _send_if_present(
    event: TwitchEvent | None,
    send: asyncio.Queue[TwitchEvent] | None,
) -> None:
    """Send *event* to *send* if both are non-None."""
    if event is not None and send is not None:
        await send.put(event)


def _build_author(
    user_id: str | None,
    user_login: str | None,
    user_name: str | None,
) -> Author:
    """Build an Author from raw twitchAPI identity fields.

    :class:`Author` is immutably keyed on ``user_id`` so repeat viewers
    resolve to the same ``people/twitch-<id>.md`` file regardless of
    display-name changes. Optional-typed fields are coerced to empty
    strings; callers in anonymous-capable flows must guard first.
    """
    return Author.from_twitch(
        user_id=str(user_id or ""),
        user_login=user_login or "",
        user_name=user_name or "",
    )
