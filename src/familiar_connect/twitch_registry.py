"""Guild-scoped Twitch watcher registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.twitch import TwitchWatcherConfig
    from familiar_connect.twitch_watcher import TwitchWatcher


class RegistryError(Exception):
    """Raised when a guild registry operation violates a constraint."""


@dataclass
class GuildTwitchState:
    """All runtime state for a guild's active Twitch watcher."""

    guild_id: int
    channel: str
    broadcaster_id: str
    config: TwitchWatcherConfig
    watcher: TwitchWatcher
    task: Any  # asyncio.Task[None] at runtime; Any for testability


_registry: dict[int, GuildTwitchState] = {}


def set_guild_twitch(guild_id: int, state: GuildTwitchState) -> None:
    """Store *state* for *guild_id*.

    :raises RegistryError: If an entry already exists for this guild.
    """
    if guild_id in _registry:
        msg = f"Guild {guild_id} already has an active Twitch watcher"
        raise RegistryError(msg)
    _registry[guild_id] = state


def get_guild_twitch(guild_id: int) -> GuildTwitchState | None:
    """Return the active state for *guild_id*, or None if not connected."""
    return _registry.get(guild_id)


def clear_guild_twitch(guild_id: int) -> None:
    """Remove the entry for *guild_id*.  No-op if absent."""
    _registry.pop(guild_id, None)
