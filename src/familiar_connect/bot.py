"""Discord bot factory and slash command definitions."""

from __future__ import annotations

import asyncio
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
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice_pipeline import get_pipeline, start_pipeline, stop_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from familiar_connect.transcription import DeepgramTranscriber, TranscriptionResult
    from familiar_connect.tts import CartesiaTTSClient

_logger = logging.getLogger(__name__)


async def _recording_finished_callback(sink: discord.sinks.Sink, *args: object) -> None:
    """No-op callback required by py-cord's start_recording.

    Cleanup is handled by :func:`stop_pipeline`, not this callback.
    """


async def awaken(
    ctx: discord.ApplicationContext,
    system_prompt: str = "",
    tts_client: CartesiaTTSClient | None = None,
    transcriber: DeepgramTranscriber | None = None,
    llm_client: LLMClient | None = None,
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
    :param transcriber: Optional Deepgram transcriber template for voice
        transcription. When provided, per-user streams are started and
        speech is debug-logged.
    :param llm_client: Optional LLM client for generating voice responses.
        When combined with a transcriber, the bot will respond to speech
        via LLM + TTS.
    """
    author = ctx.author

    # Determine whether the user wants a voice or text session.
    in_voice = isinstance(author, discord.Member) and author.voice is not None

    if in_voice:
        await _awaken_voice(
            ctx,
            tts_client=tts_client,
            transcriber=transcriber,
            llm_client=llm_client,
            system_prompt=system_prompt,
        )
    else:
        await _awaken_text(ctx, system_prompt)


def _build_voice_response_handler(
    *,
    vc: discord.VoiceClient,
    llm_client: LLMClient | None,
    tts_client: CartesiaTTSClient | None,
    system_prompt: str,
    user_names: dict[int, str],
) -> Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]] | None:
    """Build the async callback invoked for each final transcription.

    Returns ``None`` when no *llm_client* is provided (no responses
    possible without an LLM).
    """
    if llm_client is None:
        return None

    history: list[Message] = []

    async def _handle_voice_result(
        user_id: int,
        result: TranscriptionResult,
    ) -> None:
        speaker = user_names.get(user_id, f"User-{user_id}")
        safe_name = sanitize_name(speaker) or speaker
        user_msg = result.to_message(
            speaker_names={result.speaker or 0: safe_name},
        )
        user_msg.name = sanitize_name(speaker)
        history.append(user_msg)

        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            *history,
        ]

        reply = await llm_client.chat(messages)
        history.append(reply)
        _logger.info("[Voice Response] %s", reply.content)

        if tts_client is not None:
            try:
                pcm_mono = await tts_client.synthesize(reply.content)
                stereo = mono_to_stereo(pcm_mono)
                # Wait for any currently-playing audio to finish.
                # vc.is_playing() is a third-party poll; no event available.
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            except Exception:
                _logger.exception("Voice response TTS failed")

    return _handle_voice_result


async def _awaken_voice(
    ctx: discord.ApplicationContext,
    tts_client: CartesiaTTSClient | None = None,
    transcriber: DeepgramTranscriber | None = None,
    llm_client: LLMClient | None = None,
    system_prompt: str = "",
) -> None:
    """Join the invoking user's voice channel.

    If *tts_client* is provided, speaks a brief greeting immediately after
    connecting so the TTS audio pipeline can be verified end-to-end.

    If *transcriber* is provided, starts per-user transcription streams and
    begins recording voice audio for speech-to-text.

    If *llm_client* is also provided, a response handler is installed that
    sends each final transcription through the LLM and (optionally) speaks
    the reply via TTS.
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

    if transcriber is not None:
        user_names = {m.id: m.display_name for m in channel.members}

        def _resolve_from_channel(user_id: int) -> str | None:
            for member in channel.members:
                if member.id == user_id:
                    return member.display_name
            return None

        # Build voice response handler if LLM is available.
        response_handler = _build_voice_response_handler(
            vc=vc,
            llm_client=llm_client,
            tts_client=tts_client,
            system_prompt=system_prompt,
            user_names=user_names,
        )

        pipeline = await start_pipeline(
            transcriber,
            user_names=user_names,
            resolve_name=_resolve_from_channel,
            response_handler=response_handler,
        )
        sink = RecordingSink(
            loop=asyncio.get_running_loop(),
            audio_queue=pipeline.tagged_audio_queue,
        )
        vc.start_recording(sink, _recording_finished_callback)
        _logger.info("Started voice transcription pipeline")

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
        if get_pipeline() is not None:
            if hasattr(ctx.voice_client, "recording") and ctx.voice_client.recording:
                ctx.voice_client.stop_recording()
            await stop_pipeline()
            _logger.info("Stopped voice transcription pipeline")
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
    transcriber: DeepgramTranscriber | None = None,
) -> discord.Bot:
    """Create and configure the Discord bot with slash commands.

    :param llm_client: Optional LLM client for text-channel responses.
        If None, the bot will still respond to /awaken//sleep but won't
        call the LLM when messages are received.
    :param system_prompt: Pre-assembled system prompt passed to /awaken.
    :param tts_client: Optional TTS client for voice output. When provided
        and the bot is in a voice channel, LLM replies are also spoken aloud.
    :param transcriber: Optional Deepgram transcriber template for voice
        transcription. Cloned per-user when the bot joins a voice channel.
    :return: A configured discord.Bot.
    """
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.message_content = True
    intents.messages = True
    bot = discord.Bot(intents=intents)

    async def _awaken_cmd(ctx: discord.ApplicationContext) -> None:
        await awaken(
            ctx,
            system_prompt=system_prompt,
            tts_client=tts_client,
            transcriber=transcriber,
            llm_client=llm_client,
        )

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

    return bot
