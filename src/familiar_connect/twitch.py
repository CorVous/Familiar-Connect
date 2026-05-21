"""Twitch channel event watcher — event types, formatters, builders."""

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
    """Single Twitch channel event ready for LLM message batch."""

    channel: str
    text: str
    priority: Literal["normal", "immediate"]
    timestamp: datetime
    viewer: Author | None = None

    def to_message(self: Self) -> Message:
        """Convert to LLM Message.

        Content prefixed with ``[Twitch] `` so model identifies source.
        Message name is viewer's ``openai_name`` when present, else
        ``Twitch``.
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
    """Per-watcher config — which Twitch events produce messages."""

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
    base = f"{viewer} has redeemed {redemption_name}"
    if user_input:
        return f"{base} and says: {user_input}"
    return base


def format_subscription(viewer: str, tier: int) -> str:
    """First-time subscription text."""
    return f"{viewer} has subscribed at tier {tier}"


def format_gift_subscription(gifter: str | None, count: int, tier: int) -> str:
    name = gifter if gifter is not None else "An anonymous gifter"
    noun = "subscription" if count == 1 else "subscriptions"
    return f"{name} has gifted {count} tier {tier} {noun}"


def format_resubscription(viewer: str, months: int, tier: int, message: str) -> str:
    return (
        f"{viewer} has subscribed for {months} months at tier {tier}"
        f" and says: {message}"
    )


def format_cheer(viewer: str | None, bits: int, message: str) -> str:
    name = viewer if viewer is not None else "An anonymous cheerer"
    return f"{name} has cheered with {bits} bits and says: {message}"


def format_follow(viewer: str) -> str:
    return f"{viewer} has followed the channel"


def format_ad_start() -> str:
    return "An ad has begun on the channel"


def format_ad_end() -> str:
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
    """Channel point redemption event; ``None`` if not in allow-list."""
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
    """Build new subscription event; ``None`` if subscriptions disabled."""
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
    """Gift subscription event; ``None`` if subscriptions disabled."""
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
    """Resubscription event; ``None`` if subscriptions disabled."""
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
    """Cheer (bits) event; ``None`` if cheers disabled."""
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
    """Follow event; ``None`` if follows disabled."""
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
    """Ad break start event; ``None`` if ads disabled."""
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
    """Ad break end event; ``None`` if ads disabled."""
    if not config.ads_enabled:
        return None
    return TwitchEvent(
        channel=channel,
        text=format_ad_end(),
        priority="immediate" if config.ads_immediate else "normal",
        timestamp=_now(),
        viewer=None,
    )
