"""Discord bot factory, slash commands, and pipeline-routed message loop.

Step 7 of ``docs/architecture/context-pipeline.md``, plus the
post-merge voice transcription wiring from PR #17. Replaces the
single-slot ``TextSession``/``/awaken`` surface with:

- A multi-channel :class:`SubscriptionRegistry` persisted to disk.
- Explicit ``/subscribe-text``, ``/subscribe-my-voice``,
  ``/unsubscribe-text``, ``/unsubscribe-voice`` commands.
- Three per-channel mode commands (``/channel-full-rp``,
  ``/channel-text-conversation-rp``, ``/channel-imitate-voice``)
  that flip the :class:`ChannelMode` stored in the channel's
  TOML sidecar.
- ``on_message`` that routes every subscribed text message through
  the :class:`ContextPipeline`, lets registered pre/post processors
  run, and persists user + assistant turns to
  :class:`HistoryStore`.
- ``/subscribe-my-voice`` that, when the familiar has a
  :class:`DeepgramTranscriber` configured, starts a per-user
  transcription pipeline and routes every final transcription
  through the **same** :class:`ContextPipeline` text uses. Voice
  turns land in the :class:`HistoryStore` with ``role="user"``,
  a sanitised speaker name, and the channel id of the voice
  channel — so voice and text share memory, speaker prefixing,
  history summaries, and every other pipeline output without the
  voice path carrying its own state.

The bot owns a single :class:`Familiar` bundle for the lifetime of
the process — per ``docs/architecture/configuration-model.md`` one
process runs exactly one active character.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

import discord

from familiar_connect.config import ChannelMode
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import ContextRequest, Modality, PendingTurn
from familiar_connect.llm import sanitize_name
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice_pipeline import get_pipeline, start_pipeline, stop_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from familiar_connect.chattiness import BufferedMessage
    from familiar_connect.context.pipeline import ProviderOutcome
    from familiar_connect.familiar import Familiar
    from familiar_connect.transcription import TranscriptionResult

_logger = logging.getLogger(__name__)


async def _recording_finished_callback(  # noqa: RUF029
    sink: discord.sinks.Sink,
    *args: object,
) -> None:
    """No-op callback required by py-cord's ``start_recording`` API.

    Must be a coroutine even though it awaits nothing — py-cord
    awaits this callback internally when a recording ends. Cleanup
    is handled by :func:`stop_pipeline` inside
    :func:`unsubscribe_voice`, not here.
    """
    del sink, args


def _log_pipeline_outcomes(
    channel_id: int,
    outcomes: list[ProviderOutcome],
) -> None:
    """Emit a structured log entry per provider outcome.

    Kept tiny so the dashboard-backing work can later hook the same
    call site without having to parse the bot's freeform logs.
    """
    for outcome in outcomes:
        _logger.info(
            "pipeline channel=%s provider=%s status=%s duration=%.3fs",
            channel_id,
            outcome.provider_id,
            outcome.status,
            outcome.duration_s,
        )


# ---------------------------------------------------------------------------
# Slash commands — /subscribe-*
# ---------------------------------------------------------------------------


async def subscribe_text(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Register the current text channel as a text subscription."""
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return

    guild = ctx.guild
    channel = ctx.channel
    if guild is not None and isinstance(channel, discord.TextChannel):
        perms = channel.permissions_for(guild.me)
        if not perms.view_channel or not perms.send_messages:
            await ctx.respond(
                "My powers don't extend to this channel"
                " \N{EM DASH} I lack the permissions to speak here.",
                ephemeral=True,
            )
            return

    familiar.subscriptions.add(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
        guild_id=ctx.guild_id,
    )
    name = getattr(ctx.channel, "name", str(channel_id))
    _logger.info("Subscribed to text channel: %s (%s)", name, channel_id)
    await ctx.respond(f"Subscribed to text in **#{name}**.", ephemeral=True)


async def unsubscribe_text(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Remove the text subscription for the current channel."""
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return

    sub = familiar.subscriptions.get(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
    )
    if sub is None:
        await ctx.respond("I'm not listening in this channel.", ephemeral=True)
        return

    familiar.subscriptions.remove(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
    )
    familiar.monitor.clear_channel(channel_id)
    await ctx.respond("No longer listening here.", ephemeral=True)


def _build_voice_response_handler(
    *,
    vc: discord.VoiceClient,
    familiar: Familiar,
    voice_channel_id: int,
    guild_id: int | None,
    user_names: dict[int, str],
) -> Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]]:
    """Build the async callback invoked for each final voice transcription.

    Routes voice turns through the **same** :class:`ContextPipeline`
    the text loop uses: assemble a :class:`ContextRequest` with
    :attr:`Modality.voice` and the voice channel's id, run the
    pipeline, render, call the main LLM, run post-processors,
    persist to :class:`HistoryStore`, fan out to TTS.

    This is the only way voice turns get the same memory search,
    history summary, speaker prefixing, and mode instructions that
    text turns get. The earlier PR #17 implementation stashed voice
    history in a closure-local ``list[Message]`` that bypassed the
    pipeline entirely; that stub is replaced by this handler.
    """

    async def _handle_voice_result(
        user_id: int,
        result: TranscriptionResult,
    ) -> None:
        # Resolve the speaker's display name. ``user_names`` is mutated
        # by voice_pipeline's name resolver when a late joiner shows up,
        # so a lookup failure falls back to a stable ``User-<id>`` form
        # rather than ``None``.
        speaker = user_names.get(user_id, f"User-{user_id}")
        safe_name = sanitize_name(speaker) or speaker

        channel_config = familiar.channel_configs.get(channel_id=voice_channel_id)
        request = ContextRequest(
            familiar_id=familiar.id,
            channel_id=voice_channel_id,
            guild_id=guild_id,
            speaker=safe_name,
            utterance=result.text,
            modality=Modality.voice,
            budget_tokens=channel_config.budget_tokens,
            deadline_s=channel_config.deadline_s,
        )

        pipeline = familiar.build_pipeline(channel_config)
        pipeline_output = await pipeline.assemble(
            request,
            budget_by_layer=channel_config.budget_by_layer,
        )
        _log_pipeline_outcomes(voice_channel_id, pipeline_output.outcomes)

        messages = assemble_chat_messages(
            pipeline_output,
            store=familiar.history_store,
            history_window_size=familiar.config.history_window_size,
            depth_inject_position=familiar.config.depth_inject_position,
            depth_inject_role=familiar.config.depth_inject_role,
            mode=channel_config.mode,
            display_tz=familiar.config.display_tz,
        )

        _logger.info(
            "LLM request channel=%s (voice) messages=%d:\n%s",
            voice_channel_id,
            len(messages),
            "\n".join(
                f"  [{m.role}]{f' ({m.name})' if m.name else ''}: "
                f"{m.content[:200]}{'…' if len(m.content) > 200 else ''}"
                for m in messages
            ),
        )

        reply = await familiar.llm_clients["main_prose"].chat(messages)
        reply_text = await pipeline.run_post_processors(reply.content, request)

        # Persist both turns *after* the LLM call so a mid-turn crash
        # doesn't leave the store with a user turn but no reply.
        familiar.history_store.append_turn(
            familiar_id=familiar.id,
            channel_id=voice_channel_id,
            guild_id=guild_id,
            role="user",
            content=result.text,
            speaker=safe_name,
            mode=channel_config.mode,
        )
        familiar.history_store.append_turn(
            familiar_id=familiar.id,
            channel_id=voice_channel_id,
            guild_id=guild_id,
            role="assistant",
            content=reply_text,
            mode=channel_config.mode,
        )

        _logger.info("[Voice Response] %s", reply_text)

        if familiar.tts_client is not None:
            try:
                pcm_mono = await familiar.tts_client.synthesize(reply_text)
                stereo = mono_to_stereo(pcm_mono)
                # Wait for any currently-playing audio to finish.
                # vc.is_playing() is a third-party poll; no event available.
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            except Exception:
                _logger.exception("Voice response TTS failed")

    return _handle_voice_result


async def subscribe_my_voice(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Join the caller's voice channel and register a voice subscription.

    When ``familiar.transcriber`` is configured, this also starts a
    per-user Deepgram transcription pipeline and wires every final
    transcription through the ContextPipeline via
    :func:`_build_voice_response_handler`. Without a transcriber the
    bot still joins the channel and plays TTS but does not react to
    incoming speech.
    """
    author = ctx.author
    if not isinstance(author, discord.Member) or author.voice is None:
        await ctx.respond(
            "You need to be in a voice channel first.",
            ephemeral=True,
        )
        return

    channel = author.voice.channel
    if channel is None:
        await ctx.respond("Could not determine your voice channel.", ephemeral=True)
        return

    if ctx.voice_client is not None:
        await ctx.respond("I'm already in a voice channel.", ephemeral=True)
        return

    guild = ctx.guild
    if guild is not None:
        perms = channel.permissions_for(guild.me)
        if not perms.connect or not perms.speak:
            await ctx.respond(
                "I can't reach that voice channel"
                " \N{EM DASH} I lack the permissions to enter and speak there.",
                ephemeral=True,
            )
            return

    # Voice connection + DAVE handshake takes >3s, so defer.
    await ctx.defer()
    try:
        vc = await channel.connect(cls=DaveVoiceClient)
    except Exception:
        _logger.exception("Failed to connect to voice channel %s", channel.name)
        await ctx.followup.send(
            "I couldn't enter that voice channel"
            " \N{EM DASH} something went wrong when I tried to connect.",
            ephemeral=True,
        )
        return
    _logger.info("Joined voice channel: %s", channel.name)

    familiar.subscriptions.add(
        channel_id=channel.id,
        kind=SubscriptionKind.voice,
        guild_id=ctx.guild_id,
    )

    if familiar.tts_client is not None:
        try:
            pcm_mono = await familiar.tts_client.synthesize("Hello!")
            stereo = mono_to_stereo(pcm_mono)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
        except Exception:
            _logger.exception("Opening greeting TTS failed")

    if familiar.transcriber is not None:
        user_names = {m.id: m.display_name for m in channel.members}

        def _resolve_from_channel(user_id: int) -> str | None:
            for member in channel.members:
                if member.id == user_id:
                    return member.display_name
            return None

        response_handler = _build_voice_response_handler(
            vc=vc,
            familiar=familiar,
            voice_channel_id=channel.id,
            guild_id=ctx.guild_id,
            user_names=user_names,
        )

        pipeline = await start_pipeline(
            familiar.transcriber,
            user_names=user_names,
            resolve_name=_resolve_from_channel,
            response_handler=response_handler,
            lull_timeout=familiar.config.lull_timeout,
        )
        sink = RecordingSink(
            loop=asyncio.get_running_loop(),
            audio_queue=pipeline.tagged_audio_queue,
        )
        vc.start_recording(sink, _recording_finished_callback)
        _logger.info(
            "Started voice transcription pipeline for channel %s",
            channel.id,
        )

    await ctx.followup.send(f"Joined **{channel.name}**.", ephemeral=True)


async def unsubscribe_voice(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Leave the current voice channel, tear down the pipeline, drop the sub."""
    vc = ctx.voice_client
    guild_id = ctx.guild_id
    sub = (
        familiar.subscriptions.voice_in_guild(guild_id)
        if guild_id is not None
        else None
    )

    if vc is None and sub is None:
        await ctx.respond("I'm not in a voice channel.", ephemeral=True)
        return

    if vc is not None:
        # Stop the transcription pipeline first so audio chunks stop
        # flowing into a voice client that's about to disconnect.
        if get_pipeline() is not None:
            if hasattr(vc, "recording") and vc.recording:
                vc.stop_recording()
            await stop_pipeline()
            _logger.info("Stopped voice transcription pipeline")
        await vc.disconnect()

    if sub is not None:
        familiar.subscriptions.remove(
            channel_id=sub.channel_id,
            kind=SubscriptionKind.voice,
        )

    await ctx.respond("Left voice.", ephemeral=True)


# ---------------------------------------------------------------------------
# Slash commands — /channel-*
# ---------------------------------------------------------------------------


async def set_channel_mode(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
    mode: ChannelMode,
) -> None:
    """Persist *mode* as the channel's :class:`ChannelMode`."""
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return

    familiar.channel_configs.set_mode(channel_id=channel_id, mode=mode)
    _logger.info("Channel %s mode = %s", channel_id, mode.value)
    await ctx.respond(f"Channel mode set to **{mode.value}**.", ephemeral=True)


# ---------------------------------------------------------------------------
# Pipeline response path
# ---------------------------------------------------------------------------


async def _run_text_response(
    channel_id: int,
    guild_id: int | None,
    speaker: str,
    utterance: str,
    buffer: list[BufferedMessage],
    familiar: Familiar,
    channel: discord.TextChannel,
) -> None:
    """Execute the full pipeline → LLM → reply path for a text channel.

    Called by the ``on_respond`` callback built in :func:`create_bot`
    when the :class:`ConversationMonitor` decides the familiar should
    speak. Persists all buffered user messages to history (in order)
    and then the assistant reply, so the history store has a complete
    record of the conversation even though some messages were buffered
    before the pipeline ran.

    :param channel_id: Discord channel id.
    :param guild_id: Discord guild id, or ``None`` for DMs.
    :param speaker: Sanitised display name of the triggering speaker.
    :param utterance: Text of the most recent (trigger) message.
    :param buffer: All messages accumulated since the last response,
        including the trigger. Persisted to history after the LLM call.
    :param familiar: The active :class:`Familiar` bundle.
    :param channel: Discord text channel to send the reply to.
    """
    channel_config = familiar.channel_configs.get(channel_id=channel_id)

    request = ContextRequest(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        speaker=speaker,
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=tuple(
            PendingTurn(speaker=m.speaker, text=m.text) for m in buffer
        ),
    )

    pipeline = familiar.build_pipeline(channel_config)
    pipeline_output = await pipeline.assemble(
        request,
        budget_by_layer=channel_config.budget_by_layer,
    )
    _log_pipeline_outcomes(channel_id, pipeline_output.outcomes)

    messages = assemble_chat_messages(
        pipeline_output,
        store=familiar.history_store,
        history_window_size=familiar.config.history_window_size,
        depth_inject_position=familiar.config.depth_inject_position,
        depth_inject_role=familiar.config.depth_inject_role,
        mode=channel_config.mode,
        display_tz=familiar.config.display_tz,
    )

    _logger.info(
        "LLM request channel=%s messages=%d:\n%s",
        channel_id,
        len(messages),
        "\n".join(
            f"  [{m.role}]{f' ({m.name})' if m.name else ''}: "
            f"{m.content[:200]}{'…' if len(m.content) > 200 else ''}"
            for m in messages
        ),
    )

    async with channel.typing():
        reply = await familiar.llm_clients["main_prose"].chat(messages)

    reply_text = await pipeline.run_post_processors(reply.content, request)

    # Persist all buffered user turns, then the assistant reply. Done
    # after the LLM call so a mid-request crash doesn't leave an
    # orphaned user turn with no reply.
    for msg in buffer:
        familiar.history_store.append_turn(
            familiar_id=familiar.id,
            channel_id=channel_id,
            guild_id=guild_id,
            role="user",
            content=msg.text,
            speaker=msg.speaker,
            mode=channel_config.mode,
        )
    familiar.history_store.append_turn(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        role="assistant",
        content=reply_text,
        mode=channel_config.mode,
    )

    await channel.send(reply_text)

    # TTS fan-out: if a voice sub exists in this guild and a voice
    # client is connected, speak the same reply the text channel saw.
    guild = getattr(channel, "guild", None)
    if (
        familiar.tts_client is not None
        and guild is not None
        and familiar.subscriptions.voice_in_guild(guild.id) is not None
    ):
        vc = guild.voice_client
        if vc is not None and not vc.is_playing():
            try:
                pcm_mono = await familiar.tts_client.synthesize(reply_text)
                stereo = mono_to_stereo(pcm_mono)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            except Exception:
                _logger.exception("TTS synthesis failed")


# ---------------------------------------------------------------------------
# Message loop
# ---------------------------------------------------------------------------


async def on_message(message: discord.Message, familiar: Familiar) -> None:
    """Hand an incoming Discord message to the conversation monitor.

    The monitor decides whether and when the familiar responds. If it
    does, the ``on_respond`` callback (wired up in :func:`create_bot`)
    calls :func:`_run_text_response` with the buffered messages.

    Flow:

    1. Ignore bot messages.
    2. Look up the text subscription for this channel; return if absent.
    3. Detect whether the bot itself is @mentioned.
    4. Delegate to :attr:`familiar.monitor.on_message`.
    """
    if message.author.bot:
        return

    channel_id = message.channel.id
    text_sub = familiar.subscriptions.get(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
    )
    if text_sub is None:
        return

    bot_user = familiar.extras.get("bot_user")
    is_mention = (
        message.guild is not None
        and bot_user is not None
        and bot_user in message.mentions
    )

    raw_name = message.author.display_name
    speaker = sanitize_name(raw_name) or raw_name
    await familiar.monitor.on_message(
        channel_id=channel_id,
        speaker=speaker,
        text=message.content,
        is_mention=is_mention,
    )


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(familiar: Familiar) -> discord.Bot:
    """Create and configure the Discord bot bound to *familiar*.

    Registers the full subscription + channel-mode slash command
    surface and wires ``on_message`` to the monitor-routed handler.
    Also builds and installs the ``on_respond`` callback on the
    :class:`ConversationMonitor` so pipeline responses can reach
    Discord channels.
    """
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.message_content = True
    intents.messages = True
    bot = discord.Bot(intents=intents)

    # Store bot.user in extras once the bot is ready so on_message can
    # detect @mentions by comparing against bot.user in message.mentions.
    @bot.event
    async def on_ready() -> None:  # noqa: RUF029
        familiar.extras["bot_user"] = bot.user

    # Build the on_respond callback that drives the full pipeline path.
    # Captured variables: bot (for channel lookup) and familiar.
    async def _on_respond(
        channel_id: int,
        buffer: list[BufferedMessage],
    ) -> None:
        channel = bot.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            return
        sub = familiar.subscriptions.get(
            channel_id=channel_id,
            kind=SubscriptionKind.text,
        )
        if sub is None:
            return
        last = buffer[-1] if buffer else None
        if last is None:
            return
        await _run_text_response(
            channel_id=channel_id,
            guild_id=sub.guild_id,
            speaker=last.speaker,
            utterance=last.text,
            buffer=buffer,
            familiar=familiar,
            channel=channel,
        )

    familiar.monitor.on_respond = _on_respond

    # --- /subscribe-* / /unsubscribe-* ---
    async def _subscribe_text_cmd(ctx: discord.ApplicationContext) -> None:
        await subscribe_text(ctx, familiar)

    async def _unsubscribe_text_cmd(ctx: discord.ApplicationContext) -> None:
        await unsubscribe_text(ctx, familiar)

    async def _subscribe_my_voice_cmd(ctx: discord.ApplicationContext) -> None:
        await subscribe_my_voice(ctx, familiar)

    async def _unsubscribe_voice_cmd(ctx: discord.ApplicationContext) -> None:
        await unsubscribe_voice(ctx, familiar)

    bot.slash_command(
        name="subscribe-text",
        description="Listen to this text channel",
    )(_subscribe_text_cmd)
    bot.slash_command(
        name="unsubscribe-text",
        description="Stop listening to this text channel",
    )(_unsubscribe_text_cmd)
    bot.slash_command(
        name="subscribe-my-voice",
        description="Join your voice channel and enable voice replies",
    )(_subscribe_my_voice_cmd)
    bot.slash_command(
        name="unsubscribe-voice",
        description="Leave the voice channel",
    )(_unsubscribe_voice_cmd)

    # --- /channel-* ---
    async def _channel_full_rp_cmd(ctx: discord.ApplicationContext) -> None:
        await set_channel_mode(ctx, familiar, ChannelMode.full_rp)

    async def _channel_text_rp_cmd(ctx: discord.ApplicationContext) -> None:
        await set_channel_mode(ctx, familiar, ChannelMode.text_conversation_rp)

    async def _channel_imitate_voice_cmd(ctx: discord.ApplicationContext) -> None:
        await set_channel_mode(ctx, familiar, ChannelMode.imitate_voice)

    bot.slash_command(
        name="channel-full-rp",
        description="Tune this channel for full-roleplay mode",
    )(_channel_full_rp_cmd)
    bot.slash_command(
        name="channel-text-conversation-rp",
        description="Tune this channel for text conversation roleplay",
    )(_channel_text_rp_cmd)
    bot.slash_command(
        name="channel-imitate-voice",
        description="Tune this channel for low-latency voice imitation",
    )(_channel_imitate_voice_cmd)

    # --- message loop ---
    async def _on_message(message: discord.Message) -> None:
        await on_message(message, familiar)

    bot.add_listener(_on_message, name="on_message")

    return bot
