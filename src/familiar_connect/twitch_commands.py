"""Discord slash command handlers for the /twitch command group."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.twitch import Twitch

from familiar_connect.twitch import TwitchWatcherConfig
from familiar_connect.twitch_registry import (
    GuildTwitchState,
    clear_guild_twitch,
    get_guild_twitch,
    set_guild_twitch,
)
from familiar_connect.twitch_watcher import TwitchWatcher

if TYPE_CHECKING:
    import discord

    from familiar_connect.twitch import TwitchEvent


async def _resolve_broadcaster_id(channel: str, client_id: str, token: str) -> str:
    """Resolve a Twitch channel name to its broadcaster ID via the Twitch API.

    This is a thin wrapper so tests can patch it in isolation.
    """
    async with await Twitch(client_id, token) as api:
        async for user in api.get_users(logins=[channel]):
            return user.id
    msg = f"Channel '{channel}' not found on Twitch"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# /twitch connect
# ---------------------------------------------------------------------------


async def connect_cmd(
    ctx: discord.ApplicationContext,
    channel: str,
) -> None:
    """Handle /twitch connect <channel>."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    client_id = os.environ.get("TWITCH_CLIENT_ID")
    token = os.environ.get("TWITCH_ACCESS_TOKEN")
    if not client_id or not token:
        await ctx.respond(
            "Missing Twitch credentials. Set TWITCH_CLIENT_ID and TWITCH_ACCESS_TOKEN."
        )
        return

    existing = get_guild_twitch(guild_id)
    if existing is not None:
        await ctx.respond(
            f"Already connected to **{existing.channel}**. "
            "Use `/twitch disconnect` first."
        )
        return

    broadcaster_id = await _resolve_broadcaster_id(channel, client_id, token)

    config = TwitchWatcherConfig()
    watcher = TwitchWatcher(
        config=config,
        broadcaster_id=broadcaster_id,
        channel=channel,
    )
    queue: asyncio.Queue[TwitchEvent] = asyncio.Queue()
    task = asyncio.create_task(
        _run_watcher(watcher, client_id, token, queue),
        name=f"twitch-watcher-{guild_id}",
    )

    state = GuildTwitchState(
        guild_id=guild_id,
        channel=channel,
        broadcaster_id=broadcaster_id,
        config=config,
        watcher=watcher,
        task=task,
        queue=queue,
    )
    set_guild_twitch(guild_id, state)

    await ctx.respond(f"Connected to **{channel}**. Watching for Twitch events.")


async def _run_watcher(
    watcher: TwitchWatcher,
    client_id: str,
    token: str,
    send: asyncio.Queue[TwitchEvent],
) -> None:
    """Background task: run the EventSub watcher loop.

    Creates an authenticated Twitch client, wires up an EventSubWebsocket,
    and delegates to TwitchWatcher.run() which registers all listeners and
    suspends until cancelled.
    """
    async with await Twitch(client_id, token) as api:
        eventsub = EventSubWebsocket(api)
        await watcher.run(send, eventsub)


# ---------------------------------------------------------------------------
# /twitch disconnect
# ---------------------------------------------------------------------------


async def disconnect_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle /twitch disconnect."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    state.task.cancel()
    clear_guild_twitch(guild_id)
    await ctx.respond(f"Disconnected from **{state.channel}**.")


# ---------------------------------------------------------------------------
# /twitch status
# ---------------------------------------------------------------------------


_TOGGLE: dict[bool, str] = {True: "enabled", False: "disabled"}


def _format_status(state: GuildTwitchState) -> str:
    cfg = state.config
    if cfg.redemption_names:
        redemptions = ", ".join(cfg.redemption_names)
    else:
        redemptions = "None (no redemptions configured)"
    ads_mode = "immediate" if cfg.ads_immediate else "normal"
    return (
        f"**Channel:** {state.channel}\n"
        f"**Subscriptions:** {_TOGGLE[cfg.subscriptions_enabled]}\n"
        f"**Cheers:** {_TOGGLE[cfg.cheers_enabled]}\n"
        f"**Follows:** {_TOGGLE[cfg.follows_enabled]}\n"
        f"**Ads:** {_TOGGLE[cfg.ads_enabled]} ({ads_mode})\n"
        f"**Redemptions:** {redemptions}"
    )


async def status_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle /twitch status."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    await ctx.respond(_format_status(state))


# ---------------------------------------------------------------------------
# /twitch events
# ---------------------------------------------------------------------------


async def events_cmd(
    ctx: discord.ApplicationContext,
    *,
    subscriptions: bool | None = None,
    cheers: bool | None = None,
    follows: bool | None = None,
    ads: bool | None = None,
) -> None:
    """Handle /twitch events — update event type toggles."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    cfg = state.config
    if subscriptions is not None:
        cfg.subscriptions_enabled = subscriptions
    if cheers is not None:
        cfg.cheers_enabled = cheers
    if follows is not None:
        cfg.follows_enabled = follows
    if ads is not None:
        cfg.ads_enabled = ads

    await ctx.respond(_format_status(state))


# ---------------------------------------------------------------------------
# /twitch ads-immediate
# ---------------------------------------------------------------------------


async def ads_immediate_cmd(
    ctx: discord.ApplicationContext,
    *,
    enabled: bool,
) -> None:
    """Handle /twitch ads-immediate."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    state.config.ads_immediate = enabled
    status = "enabled" if enabled else "disabled"
    await ctx.respond(f"Ads immediate mode {status}.")


# ---------------------------------------------------------------------------
# /twitch redemptions add
# ---------------------------------------------------------------------------


async def redemptions_add_cmd(
    ctx: discord.ApplicationContext,
    name: str,
) -> None:
    """Handle /twitch redemptions add <name>."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    if name in state.config.redemption_names:
        await ctx.respond(f"**{name}** is already in the redemption allow-list.")
        return

    state.config.redemption_names.append(name)
    await ctx.respond(f"Added **{name}** to the redemption allow-list.")


# ---------------------------------------------------------------------------
# /twitch redemptions remove
# ---------------------------------------------------------------------------


async def redemptions_remove_cmd(
    ctx: discord.ApplicationContext,
    name: str,
) -> None:
    """Handle /twitch redemptions remove <name>."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    if name not in state.config.redemption_names:
        await ctx.respond(f"**{name}** is not in the redemption allow-list.")
        return

    state.config.redemption_names.remove(name)
    await ctx.respond(f"Removed **{name}** from the redemption allow-list.")


# ---------------------------------------------------------------------------
# /twitch redemptions list
# ---------------------------------------------------------------------------


async def redemptions_list_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle /twitch redemptions list."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    names = state.config.redemption_names
    if not names:
        await ctx.respond("No redemptions configured.")
        return

    listing = "\n".join(f"• {n}" for n in names)
    await ctx.respond(f"**Redemption allow-list:**\n{listing}")


# ---------------------------------------------------------------------------
# /twitch redemptions clear
# ---------------------------------------------------------------------------


async def redemptions_clear_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle /twitch redemptions clear."""
    guild_id = ctx.guild_id
    if guild_id is None:
        await ctx.respond("This command can only be used in a server.")
        return

    state = get_guild_twitch(guild_id)
    if state is None:
        await ctx.respond("Not connected to any Twitch channel.")
        return

    state.config.redemption_names.clear()
    await ctx.respond("Redemption allow-list cleared.")
