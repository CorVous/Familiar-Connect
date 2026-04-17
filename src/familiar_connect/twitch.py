"""Twitch channel event watcher — event types, formatters, and builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from typing import Literal, Self

    from familiar_connect.identity import Author


# ---------------------------------------------------------------------------
# Event shape
# ---------------------------------------------------------------------------


@dataclass
class TwitchEvent:
    """A single Twitch channel event ready to feed into the LLM message batch."""

    channel: str
    text: str
    priority: Literal["normal", "immediate"]
    timestamp: datetime
    viewer: Author | None = None

    def to_message(self: Self) -> Message:
        """Convert to an LLM Message.

        Content is prefixed with '[Twitch] ' so the model can identify the
        source. The message name is the viewer's ``openai_name`` when one
        is associated with the event, otherwise 'Twitch'.
        """
        return Message(
            role="user",
            content=f"[Twitch] {self.text}",
            name=self.viewer.openai_name if self.viewer is not None else "Twitch",
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TwitchWatcherConfig:
    """Per-watcher configuration for which Twitch events produce messages."""

    subscriptions_enabled: bool = True
    cheers_enabled: bool = True
    follows_enabled: bool = True
    ads_enabled: bool = True
    ads_immediate: bool = True
    redemption_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------


def format_channel_point_redemption(
    viewer: str,
    redemption_name: str,
    user_input: str | None = None,
) -> str:
    """Format a channel point redemption event as plain text."""
    base = f"{viewer} has redeemed {redemption_name}"
    if user_input:
        return f"{base} and says: {user_input}"
    return base


def format_subscription(viewer: str, tier: int) -> str:
    """Format a new (first-time) subscription event as plain text."""
    return f"{viewer} has subscribed at tier {tier}"


def format_gift_subscription(gifter: str | None, count: int, tier: int) -> str:
    """Format a gift subscription event as plain text."""
    name = gifter if gifter is not None else "An anonymous gifter"
    noun = "subscription" if count == 1 else "subscriptions"
    return f"{name} has gifted {count} tier {tier} {noun}"


def format_resubscription(viewer: str, months: int, tier: int, message: str) -> str:
    """Format a resubscription event as plain text."""
    return (
        f"{viewer} has subscribed for {months} months at tier {tier}"
        f" and says: {message}"
    )


def format_cheer(viewer: str | None, bits: int, message: str) -> str:
    """Format a cheer (bits) event as plain text."""
    name = viewer if viewer is not None else "An anonymous cheerer"
    return f"{name} has cheered with {bits} bits and says: {message}"


def format_follow(viewer: str) -> str:
    """Format a follow event as plain text."""
    return f"{viewer} has followed the channel"


def format_ad_start() -> str:
    """Format an ad break start event as plain text."""
    return "An ad has begun on the channel"


def format_ad_end() -> str:
    """Format an ad break end event as plain text."""
    return "Ads have ended"


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def build_channel_point_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    viewer: Author,
    redemption_name: str,
    user_input: str | None = None,
) -> TwitchEvent | None:
    """Build a channel point redemption event, or None if not in the allow-list."""
    if redemption_name not in config.redemption_names:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_channel_point_redemption(viewer.label, redemption_name, user_input),
        priority="normal",
        timestamp=_now(),
        viewer=viewer,
    )


def build_subscription_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    viewer: Author,
    tier: int,
) -> TwitchEvent | None:
    """Build a new subscription event, or None if subscriptions are disabled."""
    if not config.subscriptions_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_subscription(viewer.label, tier),
        priority="normal",
        timestamp=_now(),
        viewer=viewer,
    )


def build_gift_subscription_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    gifter: Author | None,
    count: int,
    tier: int,
) -> TwitchEvent | None:
    """Build a gift subscription event, or None if subscriptions are disabled."""
    if not config.subscriptions_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_gift_subscription(
            gifter.label if gifter is not None else None, count, tier
        ),
        priority="normal",
        timestamp=_now(),
        viewer=gifter,
    )


def build_resubscription_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    viewer: Author,
    months: int,
    tier: int,
    message: str,
) -> TwitchEvent | None:
    """Build a resubscription event, or None if subscriptions are disabled."""
    if not config.subscriptions_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_resubscription(viewer.label, months, tier, message),
        priority="normal",
        timestamp=_now(),
        viewer=viewer,
    )


def build_cheer_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    viewer: Author | None,
    bits: int,
    message: str,
) -> TwitchEvent | None:
    """Build a cheer (bits) event, or None if cheers are disabled."""
    if not config.cheers_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_cheer(viewer.label if viewer is not None else None, bits, message),
        priority="normal",
        timestamp=_now(),
        viewer=viewer,
    )


def build_follow_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
    viewer: Author,
) -> TwitchEvent | None:
    """Build a follow event, or None if follows are disabled."""
    if not config.follows_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_follow(viewer.label),
        priority="normal",
        timestamp=_now(),
        viewer=viewer,
    )


def build_ad_start_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
) -> TwitchEvent | None:
    """Build an ad break start event, or None if ads are disabled."""
    if not config.ads_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_ad_start(),
        priority="immediate" if config.ads_immediate else "normal",
        timestamp=_now(),
        viewer=None,
    )


def build_ad_end_event(
    *,
    config: TwitchWatcherConfig,
    channel: str,
) -> TwitchEvent | None:
    """Build an ad break end event, or None if ads are disabled."""
    if not config.ads_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_ad_end(),
        priority="immediate" if config.ads_immediate else "normal",
        timestamp=_now(),
        viewer=None,
    )
