"""Discord bot shell.

Owns:

- Discord client construction (``py-cord`` + DAVE voice)
- Subscribe / unsubscribe slash commands (text + voice)
- ``on_message`` and ``on_voice_state_update`` event handlers
- :func:`ingest_event` — text and voice events publish to the event
  bus via :class:`DiscordTextSource`. Twitch has its own source.
- :class:`BotHandle` — adapter exposed to the lifecycle wiring so
  bus-only processors (e.g. :class:`TextResponder`) can post back
  to Discord without taking a direct ``discord.Bot`` reference.
- :func:`_start_voice_intake` / :func:`_stop_voice_intake` — bring
  up / tear down the per-channel sink + transcriber + voice source
  pipeline that ``/subscribe-voice`` triggers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import discord

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.collector import get_span_collector
from familiar_connect.diagnostics.report import render_summary_table
from familiar_connect.identity import Author
from familiar_connect.sources import DiscordTextSource
from familiar_connect.sources.voice import VoiceSource
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient
from familiar_connect.voice.recording_sink import RecordingSink

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.familiar import Familiar
    from familiar_connect.transcription import DeepgramTranscriber, TranscriptionResult

    # Outbound text callback. Returns the Discord message id of the
    # posted message (str) for reply-chain tracking, or None if the
    # send failed. ``reply_to_message_id`` threads via Discord's
    # ``MessageReference``; ``mention_user_ids`` populates
    # ``AllowedMentions(users=...)`` so only resolved targets get pings.
    SendText = Callable[
        [int, str, str | None, tuple[int, ...]],
        Awaitable[str | None],
    ]

_logger = logging.getLogger(__name__)


@dataclass
class VoiceRuntime:
    """Per-voice-channel pipeline state.

    Holds the live voice client + the recording-sink / pump / source
    tasks so ``/unsubscribe-voice`` can tear them down cleanly.

    ``transcribers`` and ``fanin_tasks`` are keyed by Discord user_id
    — the pump lazily clones a Deepgram stream per speaker (Discord
    delivers per-SSRC audio so each user is naturally isolated). One
    fan-in task per user forwards results onto the shared
    ``result_queue`` after tagging with the originating user_id.
    """

    voice_client: discord.VoiceClient
    sink: RecordingSink
    audio_queue: asyncio.Queue[tuple[int, bytes]]
    result_queue: asyncio.Queue[TranscriptionResult]
    pump_task: asyncio.Task[None]
    source_task: asyncio.Task[None]
    transcribers: dict[int, DeepgramTranscriber] = field(default_factory=dict)
    fanin_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)


@dataclass
class BotHandle:
    """Bundle of bot + outbound seams used by bus processors.

    ``send_text`` is the seam :class:`TextResponder` injects so it can
    post replies without depending on pycord. The callback returns the
    Discord message id of the posted message (as ``str``) so the
    caller can record it for future reply lookups; returns ``None``
    on send failure. ``voice_runtime`` is keyed by voice-channel id;
    populated by ``/subscribe-voice`` and consumed by the active TTS
    player to find the live voice client.
    """

    bot: discord.Bot
    send_text: SendText
    voice_runtime: dict[int, VoiceRuntime] = field(default_factory=dict)


async def _on_recording_done(sink: RecordingSink, *args: object) -> None:  # noqa: RUF029 — pycord requires coroutine fn even when there's nothing to await
    """py-cord ``start_recording`` callback. No-op — sink writes via queue.

    Must be ``async def`` — pycord schedules the return value with
    ``asyncio.run_coroutine_threadsafe`` (voice_client.py:915), which
    requires a coroutine.
    """
    del sink, args


async def _start_voice_intake(  # noqa: RUF029 — called from async slash-command handler
    *,
    handle: BotHandle,
    familiar: Familiar,
    voice_client: discord.VoiceClient,
    channel_id: int,
) -> VoiceRuntime | None:
    """Bring up sink + per-user transcribers + voice source for *channel_id*.

    Returns ``None`` when no transcriber is configured — the bot can
    still join for TTS playback only. Idempotent: a second call for
    the same channel returns the existing runtime.

    ``familiar.transcriber`` is treated as a *template*: a fresh
    Deepgram WS is cloned for each Discord user_id the first time
    audio arrives from that user. Per-user streams kill mixed-stream
    endpointing (one speaker's pause finalizing another's mid-sentence)
    and inherit attribution from Discord's per-SSRC delivery.
    """
    if familiar.transcriber is None:
        return None
    existing = handle.voice_runtime.get(channel_id)
    if existing is not None:
        return existing

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
    result_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()

    sink = RecordingSink(loop=loop, audio_queue=audio_queue)
    voice_client.start_recording(sink, _on_recording_done)

    source = VoiceSource(
        bus=familiar.bus,
        familiar_id=familiar.id,
        voice_channel_id=channel_id,
        queue=result_queue,
    )

    template = familiar.transcriber
    transcribers: dict[int, DeepgramTranscriber] = {}
    fanin_tasks: dict[int, asyncio.Task[None]] = {}

    async def _fanin(user_id: int, q: asyncio.Queue[TranscriptionResult]) -> None:
        """Tag each result with ``user_id`` and forward to shared queue."""
        while True:
            result = await q.get()
            result.user_id = user_id
            await result_queue.put(result)

    async def _ensure_transcriber(user_id: int) -> DeepgramTranscriber:
        existing = transcribers.get(user_id)
        if existing is not None:
            return existing
        clone = template.clone()
        per_user_q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await clone.start(per_user_q)
        transcribers[user_id] = clone
        fanin_tasks[user_id] = asyncio.create_task(
            _fanin(user_id, per_user_q),
            name=f"voice-fanin-{channel_id}-{user_id}",
        )
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.G)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)} "
            f"{ls.kv('transcriber', 'opened', vc=ls.LG)}"
        )
        return clone

    async def _pump_audio() -> None:
        while True:
            user_id, pcm = await audio_queue.get()
            clone = await _ensure_transcriber(user_id)
            await clone.send_audio(pcm)

    pump_task = asyncio.create_task(_pump_audio(), name=f"voice-pump-{channel_id}")
    source_task = asyncio.create_task(source.run(), name=f"voice-source-{channel_id}")
    rt = VoiceRuntime(
        voice_client=voice_client,
        sink=sink,
        audio_queue=audio_queue,
        result_queue=result_queue,
        pump_task=pump_task,
        source_task=source_task,
        transcribers=transcribers,
        fanin_tasks=fanin_tasks,
    )
    handle.voice_runtime[channel_id] = rt
    _logger.info(
        f"{ls.tag('🎙️  Voice', ls.G)} "
        f"{ls.kv('intake', 'started', vc=ls.LG)} "
        f"{ls.kv('channel', str(channel_id), vc=ls.LC)}"
    )
    return rt


async def _stop_voice_intake(
    *,
    handle: BotHandle,
    familiar: Familiar,
    channel_id: int,
) -> None:
    """Cancel pump + source + fan-ins, stop recording, stop every clone."""
    del familiar  # template not stopped — only clones own connections
    rt = handle.voice_runtime.pop(channel_id, None)
    if rt is None:
        return
    with contextlib.suppress(Exception):
        rt.voice_client.stop_recording()
    rt.pump_task.cancel()
    rt.source_task.cancel()
    for t in rt.fanin_tasks.values():
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.pump_task
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.source_task
    for t in rt.fanin_tasks.values():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t
    for clone in rt.transcribers.values():
        with contextlib.suppress(Exception):
            await clone.stop()
    _logger.info(
        f"{ls.tag('🎙️  Voice', ls.Y)} "
        f"{ls.kv('intake', 'stopped', vc=ls.LY)} "
        f"{ls.kv('channel', str(channel_id), vc=ls.LC)}"
    )


# ---------------------------------------------------------------------------
# Event ingest — publishes onto the bus
# ---------------------------------------------------------------------------


async def ingest_event(
    *,
    source: DiscordTextSource,
    familiar: Familiar,
    channel_id: int,
    guild_id: int | None,
    author: Author,
    text: str,
    message_id: str | None = None,
    reply_to_message_id: str | None = None,
    mentions: tuple[Author, ...] = (),
) -> None:
    """Publish a text event onto the bus.

    Kept as a named function so callers (currently only ``on_message``)
    have a single seam to point at. Logging moves into the debug
    processor; this function stays narrow.
    """
    del familiar  # unused — reserved for future per-familiar routing
    await source.publish_text(
        channel_id=channel_id,
        guild_id=guild_id,
        author=author,
        content=text,
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        mentions=mentions,
    )


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(familiar: Familiar) -> BotHandle:
    """Construct the Discord client and register slash commands + events.

    Returns a :class:`BotHandle` carrying the bot plus the
    ``send_text`` callback bus processors use for outbound posts and
    the per-channel voice runtime map.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True

    bot = discord.Bot(intents=intents)

    async def send_text(
        channel_id: int,
        content: str,
        reply_to_message_id: str | None = None,
        mention_user_ids: tuple[int, ...] = (),
    ) -> str | None:
        """Resolve channel by id, post ``content`` via ``channel.send``.

        Resolves on each call — channel cache may miss right after
        startup; ``fetch_channel`` is the fallback.

        ``reply_to_message_id``: when set, threads the post to that
        message via ``discord.MessageReference``.
        ``mention_user_ids``: populates ``AllowedMentions(users=...)``
        so only those user ids actually receive a notification, even
        if the content contains other ``<@…>`` markers.

        Returns the platform message id of the posted message (so
        ``TextResponder`` can persist it for future reply lookups),
        or ``None`` if the send failed.
        """
        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except discord.DiscordException as exc:
                _logger.warning(
                    "send_text fetch_channel failed: channel=%d err=%s",
                    channel_id,
                    exc,
                )
                return None
        if not isinstance(channel, discord.abc.Messageable):
            _logger.warning("send_text: channel %d not messageable", channel_id)
            return None

        # Always restrict who can be pinged. Defers @everyone / role
        # decisions to Discord's bot-and-role permissions.
        allowed = discord.AllowedMentions(
            everyone=False,
            roles=False,
            users=[discord.Object(id=int(u)) for u in mention_user_ids],
        )
        reference: discord.MessageReference | None = None
        if reply_to_message_id:
            try:
                ref_id = int(reply_to_message_id)
            except ValueError:
                ref_id = None
            if ref_id is not None:
                reference = discord.MessageReference(
                    message_id=ref_id,
                    channel_id=channel_id,
                    fail_if_not_exists=False,
                )

        try:
            if reference is not None:
                sent = await channel.send(
                    content, allowed_mentions=allowed, reference=reference
                )
            else:
                sent = await channel.send(content, allowed_mentions=allowed)
        except discord.DiscordException as exc:
            _logger.warning(
                "send_text channel.send failed: channel=%d err=%s",
                channel_id,
                exc,
            )
            return None
        return str(sent.id) if sent is not None else None

    handle = BotHandle(bot=bot, send_text=send_text)

    text_source = DiscordTextSource(bus=familiar.bus, familiar_id=familiar.id)
    _register_slash_commands(handle, familiar)
    _register_events(bot, familiar, text_source)

    return handle


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _register_slash_commands(handle: BotHandle, familiar: Familiar) -> None:
    bot = handle.bot

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
            voice_client = await channel.connect(cls=DaveVoiceClient)
        except discord.DiscordException as exc:
            _logger.warning("voice connect failed: %s", exc)
            await ctx.respond("Could not join voice.", ephemeral=True)
            return

        # Bring up sink + transcriber + voice source. Returns None if
        # no transcriber configured — bot still joined for playback.
        rt = await _start_voice_intake(
            handle=handle,
            familiar=familiar,
            voice_client=voice_client,
            channel_id=channel.id,
        )

        familiar.subscriptions.add(
            channel_id=channel.id,
            kind=SubscriptionKind.voice,
            guild_id=ctx.guild_id,
        )
        suffix = "" if rt is not None else " (playback only — no transcriber)"
        await ctx.respond(f"Joined {channel.name}.{suffix}", ephemeral=True)

    @bot.slash_command(
        name="diagnostics",
        description="Show span timings (last p50/p95 per span).",
    )
    async def diagnostics(ctx: discord.ApplicationContext) -> None:
        summary = get_span_collector().summary()
        text = render_summary_table(summary)
        await ctx.respond(text, ephemeral=True)

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

        await _stop_voice_intake(
            handle=handle,
            familiar=familiar,
            channel_id=sub.channel_id,
        )

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


def _register_events(
    bot: discord.Bot,
    familiar: Familiar,
    text_source: DiscordTextSource,
) -> None:
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

        reply_to: str | None = None
        if message.reference is not None and message.reference.message_id:
            reply_to = str(message.reference.message_id)
        mention_authors = tuple(
            Author.from_discord_member(u)
            for u in message.mentions
            if not getattr(u, "bot", False)
        )
        await ingest_event(
            source=text_source,
            familiar=familiar,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            author=Author.from_discord_member(message.author),
            text=message.content,
            message_id=str(message.id),
            reply_to_message_id=reply_to,
            mentions=mention_authors,
        )

    @bot.event
    async def on_voice_state_update(  # noqa: RUF029 — Discord event handler contract
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
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
