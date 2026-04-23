"""Discord bot shell for the re-arch demolition branch.

Reply orchestration is a stub. This module owns:

- Discord client construction (``py-cord`` + DAVE voice)
- Subscribe / unsubscribe slash commands (text + voice)
- ``on_message`` and ``on_voice_state_update`` event handlers
- :func:`ingest_event` — the symmetric log-and-drop stub the text,
  voice, and Twitch event paths all funnel through. Stays symmetric
  on purpose so the future unified event-stream abstraction has one
  seam to hook into.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import discord

from familiar_connect import log_style as ls
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient

if TYPE_CHECKING:
    from familiar_connect.familiar import Familiar

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symmetric event stub — all ingest paths flow through here
# ---------------------------------------------------------------------------


async def ingest_event(  # noqa: RUF029 — async reserved for the future reply path
    *,
    source: str,
    familiar: Familiar,
    channel_id: int | None,
    guild_id: int | None,
    author_label: str,
    text: str,
) -> None:
    """Log and drop. Placeholder for the next reply-path design.

    Text messages, voice utterances, and Twitch events all funnel here
    with the same call signature so the upcoming unified event-stream
    abstraction has one seam to replace.
    """
    del familiar  # will be used by the future reply path
    _logger.info(
        f"{ls.tag('📥 Event', ls.LG)} "
        f"{ls.kv('source', source, vc=ls.LG)} "
        f"{ls.kv('channel', str(channel_id), vc=ls.LC)} "
        f"{ls.kv('guild', str(guild_id), vc=ls.LC)} "
        f"{ls.kv('author', author_label, vc=ls.LC)} "
        f"{ls.kv('text', ls.trunc(text, 200), vc=ls.LW)}"
    )


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(familiar: Familiar) -> discord.Bot:
    """Construct the Discord client and register slash commands + events."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True

    bot = discord.Bot(intents=intents)

    _register_slash_commands(bot, familiar)
    _register_events(bot, familiar)

    return bot


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _register_slash_commands(bot: discord.Bot, familiar: Familiar) -> None:
    @bot.slash_command(
        name="subscribe-text",
        description="Listen for text messages in this channel.",
    )
    async def subscribe_text(ctx: discord.ApplicationContext) -> None:
        if ctx.channel_id is None:
            await ctx.respond("No channel in context.", ephemeral=True)
            return
        familiar.subscriptions.add(
            channel_id=ctx.channel_id,
            kind=SubscriptionKind.text,
            guild_id=ctx.guild_id,
        )
        await ctx.respond("Listening in this channel.", ephemeral=True)

    @bot.slash_command(
        name="unsubscribe-text",
        description="Stop listening for text messages in this channel.",
    )
    async def unsubscribe_text(ctx: discord.ApplicationContext) -> None:
        if ctx.channel_id is None:
            await ctx.respond("No channel in context.", ephemeral=True)
            return
        familiar.subscriptions.remove(
            channel_id=ctx.channel_id,
            kind=SubscriptionKind.text,
        )
        await ctx.respond("No longer listening here.", ephemeral=True)

    @bot.slash_command(
        name="subscribe-voice",
        description="Join your voice channel and listen.",
    )
    async def subscribe_voice(ctx: discord.ApplicationContext) -> None:
        member = ctx.author
        voice_state = getattr(member, "voice", None)
        if voice_state is None or voice_state.channel is None:
            await ctx.respond("You must be in a voice channel.", ephemeral=True)
            return

        channel = voice_state.channel
        try:
            await channel.connect(cls=DaveVoiceClient)
        except discord.DiscordException as exc:
            _logger.warning("voice connect failed: %s", exc)
            await ctx.respond("Could not join voice.", ephemeral=True)
            return

        # Intentional: no start_recording here. The reply path is a
        # stub, so an audio pump would have nowhere to deliver PCM.
        # The RecordingSink + Deepgram wiring returns when the next
        # reply design wants incoming audio.

        familiar.subscriptions.add(
            channel_id=channel.id,
            kind=SubscriptionKind.voice,
            guild_id=ctx.guild_id,
        )
        await ctx.respond(f"Joined {channel.name}.", ephemeral=True)

    @bot.slash_command(
        name="unsubscribe-voice",
        description="Leave the voice channel in this guild.",
    )
    async def unsubscribe_voice(ctx: discord.ApplicationContext) -> None:
        guild = ctx.guild
        if guild is None:
            await ctx.respond("Not in a guild.", ephemeral=True)
            return

        sub = familiar.subscriptions.voice_in_guild(guild.id)
        if sub is None:
            await ctx.respond("Not in a voice channel here.", ephemeral=True)
            return

        vc = guild.voice_client
        if vc is not None:
            with contextlib.suppress(discord.DiscordException):
                await vc.disconnect(force=False)

        familiar.subscriptions.remove(
            channel_id=sub.channel_id,
            kind=SubscriptionKind.voice,
        )
        await ctx.respond("Left voice channel.", ephemeral=True)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _register_events(bot: discord.Bot, familiar: Familiar) -> None:
    @bot.event
    async def on_ready() -> None:  # noqa: RUF029 — Discord event handler contract
        user = bot.user
        if user is not None:
            familiar.bot_user_id = user.id
        _logger.info(
            f"{ls.tag('🤖 Ready', ls.G)} "
            f"{ls.kv('user', str(user), vc=ls.LC)} "
            f"{ls.kv('guilds', str(len(bot.guilds)), vc=ls.LC)}"
        )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if (
            familiar.bot_user_id is not None
            and message.author.id == familiar.bot_user_id
        ):
            return
        if message.author.bot:
            return
        if (
            familiar.subscriptions.get(
                channel_id=message.channel.id,
                kind=SubscriptionKind.text,
            )
            is None
        ):
            return

        await ingest_event(
            source="discord-text",
            familiar=familiar,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            author_label=f"{message.author.name}#{message.author.id}",
            text=message.content,
        )

    @bot.event
    async def on_voice_state_update(  # noqa: RUF029 — Discord event handler contract
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        # Purely informational during the demolition pass. Keeps
        # voice-channel presence visible in the log without wiring any
        # of the prior interjection / interruption behaviour.
        del before
        if familiar.bot_user_id is not None and member.id == familiar.bot_user_id:
            return
        if after.channel is None:
            return
        sub = familiar.subscriptions.voice_in_guild(member.guild.id)
        if sub is None or sub.channel_id != after.channel.id:
            return
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.G)} "
            f"{ls.kv('member', member.display_name, vc=ls.LC)} "
            f"{ls.kv('channel', after.channel.name, vc=ls.LC)}"
        )
