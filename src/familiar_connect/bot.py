"""Discord bot factory and slash command definitions."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import discord

from familiar_connect.llm import LLMClient, Message, sanitize_name
from familiar_connect.text_session import (
    SessionError,
    TextSession,
    clear_session,
    get_session,
    set_session,
)
from familiar_connect.twitch_commands import (
    _NurseryProto,
    ads_immediate_cmd,
    connect_cmd,
    disconnect_cmd,
    events_cmd,
    redemptions_add_cmd,
    redemptions_clear_cmd,
    redemptions_list_cmd,
    redemptions_remove_cmd,
    status_cmd,
)
from familiar_connect.voice import DaveVoiceClient
from familiar_connect.voice.audio import mono_to_stereo

if TYPE_CHECKING:
    from familiar_connect.tts import CartesiaTTSClient

_logger = logging.getLogger(__name__)


async def awaken(
    ctx: discord.ApplicationContext,
    system_prompt: str = "",
    tts_client: CartesiaTTSClient | None = None,
) -> None:
    """Handle the /awaken slash command.

    Behaviour depends on where the command is invoked:

    - **Text channel** (user not in voice): binds the bot to the text channel
      and begins listening for messages there.
    - **Voice channel** (user in voice): joins the user's voice channel and
      speaks a greeting if a TTS client is available.

    The bot can only be in one active session at a time (text *or* voice).

    :param ctx: The application context for the slash command invocation.
    :param system_prompt: Pre-assembled system prompt string (injected by the
        run command; empty string disables LLM responses).
    :param tts_client: Optional TTS client used to speak a greeting on join.
    """
    author = ctx.author

    # Determine whether the user wants a voice or text session.
    in_voice = isinstance(author, discord.Member) and author.voice is not None

    if in_voice:
        await _awaken_voice(ctx, tts_client=tts_client)
    else:
        await _awaken_text(ctx, system_prompt)


async def _awaken_voice(
    ctx: discord.ApplicationContext,
    tts_client: CartesiaTTSClient | None = None,
) -> None:
    """Join the invoking user's voice channel.

    If *tts_client* is provided, speaks a brief greeting immediately after
    connecting so the TTS audio pipeline can be verified end-to-end.
    """
    author = ctx.author

    # Refuse if any session (voice or text) is already active.
    if ctx.voice_client is not None or get_session() is not None:
        await ctx.respond("I'm already active somewhere.", ephemeral=True)
        return

    # author.voice is guaranteed non-None here (checked in awaken() before dispatch).
    assert isinstance(author, discord.Member)  # noqa: S101
    assert author.voice is not None  # noqa: S101
    channel = author.voice.channel
    if channel is None:
        await ctx.respond(
            "Could not determine your voice channel.",
            ephemeral=True,
        )
        return

    # Voice connection + DAVE handshake takes >3s, so defer the interaction.
    await ctx.defer()
    vc = await channel.connect(cls=DaveVoiceClient)
    _logger.info("Joined voice channel: %s", channel.name)

    if tts_client is not None:
        try:
            pcm_mono = await tts_client.synthesize("Hello!")
            stereo = mono_to_stereo(pcm_mono)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
        except Exception:
            _logger.exception("Opening greeting TTS failed")

    await ctx.followup.send(f"Joined **{channel.name}**.")


async def _awaken_text(
    ctx: discord.ApplicationContext,
    system_prompt: str,
) -> None:
    """Bind the bot to the invoking text channel."""
    # Refuse if any session (voice or text) is already active.
    if ctx.voice_client is not None or get_session() is not None:
        await ctx.respond("I'm already active somewhere.", ephemeral=True)
        return

    # channel_id is always set for messages sent in a channel.
    if ctx.channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return
    channel_id: int = ctx.channel_id
    session = TextSession(channel_id=channel_id, system_prompt=system_prompt)
    try:
        set_session(session)
    except SessionError:
        await ctx.respond("I'm already active somewhere.", ephemeral=True)
        return

    channel = ctx.channel
    channel_name = getattr(channel, "name", str(channel_id))
    _logger.info("Bound to text channel: %s", channel_name)
    await ctx.respond(f"Listening in **#{channel_name}**.")


async def sleep_cmd(ctx: discord.ApplicationContext) -> None:
    """Handle the /sleep slash command — end whichever session is active.

    Disconnects from a voice channel OR clears a text session, depending on
    which is currently active.

    :param ctx: The application context for the slash command invocation.
    """
    if ctx.voice_client is not None:
        await ctx.voice_client.disconnect()
        _logger.info("Left voice channel")
        await ctx.respond("Goodnight.")
        return

    if get_session() is not None:
        clear_session()
        _logger.info("Left text channel")
        await ctx.respond("Goodnight.")
        return

    await ctx.respond("I'm not active anywhere.", ephemeral=True)


async def on_message(
    message: discord.Message,
    llm_client: LLMClient,
    tts_client: CartesiaTTSClient | None = None,
) -> None:
    """Handle incoming Discord messages for the active text session.

    Ignores bot messages, messages outside the active session's channel, and
    messages when no text session is active.  Otherwise appends the message to
    history, calls the LLM, appends the reply, and posts it to the channel.

    If *tts_client* is provided and the bot is in a voice channel in the same
    guild, the reply is also synthesized and played in that voice channel.

    :param message: The incoming Discord message.
    :param llm_client: LLM client to generate a response.
    :param tts_client: Optional TTS client for voice output.
    """
    if message.author.bot:
        return

    session = get_session()
    if session is None:
        return

    if message.channel.id != session.channel_id:
        return

    user_msg = Message(
        role="user",
        content=message.content,
        name=sanitize_name(message.author.display_name),
    )
    session.history.append(user_msg)

    messages: list[Message] = [
        Message(role="system", content=session.system_prompt),
        *session.history,
    ]

    async with message.channel.typing():
        reply = await llm_client.chat(messages)

    session.history.append(reply)
    await message.channel.send(reply.content)

    # Voice output: synthesize and play if in a voice channel.
    if tts_client is not None and message.guild is not None:
        vc = message.guild.voice_client
        if vc is not None and not vc.is_playing():
            try:
                pcm_mono = await tts_client.synthesize(reply.content)
                stereo = mono_to_stereo(pcm_mono)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            except Exception:
                _logger.exception("TTS synthesis failed")


def create_bot(
    llm_client: LLMClient | None = None,
    system_prompt: str = "",
    tts_client: CartesiaTTSClient | None = None,
    nursery: _NurseryProto | None = None,
) -> discord.Bot:
    """Create and configure the Discord bot with slash commands.

    :param llm_client: Optional LLM client for text-channel responses.
        If None, the bot will still respond to /awaken//sleep but won't
        call the LLM when messages are received.
    :param system_prompt: Pre-assembled system prompt passed to /awaken.
    :param tts_client: Optional TTS client for voice output. When provided
        and the bot is in a voice channel, LLM replies are also spoken aloud.
    :param nursery: Trio nursery used to spawn Twitch watcher tasks. When None,
        /twitch connect will still register the state but cannot spawn the task.
    :return: A configured discord.Bot.
    """
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.message_content = True
    intents.messages = True
    bot = discord.Bot(intents=intents)

    async def _awaken_cmd(ctx: discord.ApplicationContext) -> None:
        await awaken(ctx, system_prompt=system_prompt, tts_client=tts_client)

    async def _on_message(message: discord.Message) -> None:
        if llm_client is not None:
            await on_message(message, llm_client, tts_client=tts_client)

    bot.slash_command(name="awaken", description="Join your voice or text channel")(
        _awaken_cmd
    )
    bot.slash_command(name="sleep", description="Leave the voice or text channel")(
        sleep_cmd
    )
    bot.add_listener(_on_message, name="on_message")

    _register_twitch_commands(bot, nursery)

    return bot


def _register_twitch_commands(
    bot: discord.Bot,
    nursery: _NurseryProto | None,
) -> None:
    """Register the /twitch slash command group on *bot*."""
    twitch = discord.SlashCommandGroup("twitch", "Twitch channel event watcher")

    # /twitch connect
    @twitch.command(name="connect", description="Connect to a Twitch channel")
    @discord.option("channel", description="Twitch channel name", required=True)
    async def _connect(ctx: discord.ApplicationContext, channel: str) -> None:
        await connect_cmd(ctx, channel=channel, nursery=nursery)

    # /twitch disconnect
    @twitch.command(name="disconnect", description="Stop watching the Twitch channel")
    async def _disconnect(ctx: discord.ApplicationContext) -> None:
        await disconnect_cmd(ctx)

    # /twitch status
    @twitch.command(
        name="status",
        description="Show connected channel and event settings",
    )
    async def _status(ctx: discord.ApplicationContext) -> None:
        await status_cmd(ctx)

    # /twitch events
    @twitch.command(
        name="events",
        description="Toggle which event types produce messages",
    )
    @discord.option(
        "subscriptions",
        description="Enable subscription events",
        required=False,
    )
    @discord.option("cheers", description="Enable cheer (bits) events", required=False)
    @discord.option("follows", description="Enable follow events", required=False)
    @discord.option("ads", description="Enable ad break events", required=False)
    async def _events(
        ctx: discord.ApplicationContext,
        *,
        subscriptions: bool | None = None,
        cheers: bool | None = None,
        follows: bool | None = None,
        ads: bool | None = None,
    ) -> None:
        await events_cmd(
            ctx,
            subscriptions=subscriptions,
            cheers=cheers,
            follows=follows,
            ads=ads,
        )

    # /twitch ads-immediate
    @twitch.command(
        name="ads-immediate",
        description="Toggle immediate ad break notifications",
    )
    @discord.option("enabled", description="Enable immediate ads mode", required=True)
    async def _ads_immediate(ctx: discord.ApplicationContext, *, enabled: bool) -> None:
        await ads_immediate_cmd(ctx, enabled=enabled)

    # /twitch redemptions subgroup
    redemptions = twitch.create_subgroup(
        "redemptions",
        "Manage channel point redemption allow-list",
    )

    @redemptions.command(name="add", description="Add a redemption to the allow-list")
    @discord.option("name", description="Redemption title", required=True)
    async def _redemptions_add(ctx: discord.ApplicationContext, name: str) -> None:
        await redemptions_add_cmd(ctx, name=name)

    @redemptions.command(
        name="remove",
        description="Remove a redemption from the allow-list",
    )
    @discord.option("name", description="Redemption title", required=True)
    async def _redemptions_remove(ctx: discord.ApplicationContext, name: str) -> None:
        await redemptions_remove_cmd(ctx, name=name)

    @redemptions.command(
        name="list",
        description="List all redemptions in the allow-list",
    )
    async def _redemptions_list(ctx: discord.ApplicationContext) -> None:
        await redemptions_list_cmd(ctx)

    @redemptions.command(
        name="clear",
        description="Clear the entire redemption allow-list",
    )
    async def _redemptions_clear(ctx: discord.ApplicationContext) -> None:
        await redemptions_clear_cmd(ctx)

    bot.add_application_command(twitch)
