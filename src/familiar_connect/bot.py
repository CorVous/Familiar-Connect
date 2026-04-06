"""Discord bot factory and slash command definitions."""

import logging

import discord

from familiar_connect.voice import DaveVoiceClient

_logger = logging.getLogger(__name__)


async def awaken(ctx: discord.ApplicationContext) -> None:
    """Handle the /awaken slash command — join the user's voice channel.

    :param ctx: The application context for the slash command invocation
    """
    author = ctx.author
    if not isinstance(author, discord.Member) or author.voice is None:
        await ctx.respond(
            "You need to be in a voice channel first.",
            ephemeral=True,
        )
        return

    if ctx.voice_client is not None:
        await ctx.respond("I'm already in a voice channel.")
        return

    channel = author.voice.channel
    if channel is None:
        await ctx.respond(
            "Could not determine your voice channel.",
            ephemeral=True,
        )
        return

    # Voice connection + DAVE handshake takes >3s, so defer the interaction.
    await ctx.defer()
    await channel.connect(cls=DaveVoiceClient)
    _logger.info("Joined voice channel: %s", channel.name)
    await ctx.followup.send(f"Joined **{channel.name}**.")


async def sleep_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle the /sleep slash command — leave the current voice channel.

    :param ctx: The application context for the slash command invocation
    """
    if ctx.voice_client is None:
        await ctx.respond(
            "I'm not in a voice channel.",
            ephemeral=True,
        )
        return

    await ctx.voice_client.disconnect()
    _logger.info("Left voice channel")
    await ctx.respond("Goodnight.")


def create_bot() -> discord.Bot:
    """Create and configure the Discord bot with slash commands.

    :return: A configured discord.Bot with /awaken and /sleep commands
    """
    intents = discord.Intents.default()
    intents.voice_states = True
    bot = discord.Bot(intents=intents)

    bot.slash_command(name="awaken", description="Join your voice channel")(awaken)
    bot.slash_command(name="sleep", description="Leave the voice channel")(sleep_cmd)

    return bot
