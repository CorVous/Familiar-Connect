"""Discord bot shell.

Owns:

- Discord client construction (``py-cord`` + DAVE voice)
- Subscribe / unsubscribe slash commands (text + voice)
- ``on_message`` / ``on_voice_state_update`` handlers
- :func:`ingest_event` — text + voice events publish to bus via
  :class:`DiscordTextSource`. Twitch has its own source.
- :class:`BotHandle` — adapter for bus-only processors (e.g.
  :class:`TextResponder`) to post back to Discord without holding
  a direct ``discord.Bot`` reference.
- :func:`_start_voice_intake` / :func:`_stop_voice_intake` —
  bring up / tear down per-channel sink + transcriber + voice source
  pipeline behind ``/subscribe-voice``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
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
from familiar_connect.typing_interrupt import TypingInterruptHandler
from familiar_connect.voice import DaveVoiceClient
from familiar_connect.voice.recording_sink import RecordingSink

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager as AsyncContextManager

    from familiar_connect.familiar import Familiar
    from familiar_connect.stt import Transcriber, TranscriptionResult
    from familiar_connect.voice.turn_detection import UtteranceEndpointer

    # outbound text callback. returns posted-message id (str) for
    # reply-chain tracking, or None on send failure.
    # ``reply_to_message_id`` threads via Discord ``MessageReference``;
    # ``mention_user_ids`` → ``AllowedMentions(users=...)`` so only
    # resolved targets ping.
    SendText = Callable[
        [int, str, str | None, tuple[int, ...]],
        Awaitable[str | None],
    ]

    # py-cord ``Bot is typing…`` indicator hook. Returns a fresh async
    # context manager that wraps the bot's response generation so the
    # client shows the indicator for the duration. Implemented in
    # ``create_bot`` via ``channel.typing()``.
    TriggerTyping = Callable[[int], AsyncContextManager[None]]

    # resolves Discord member identity for voice turns so they carry
    # the same author attribution as text turns.
    ResolveMember = Callable[[int, int], "Author | None"]

_logger = logging.getLogger(__name__)


@dataclass
class VoiceRuntime:
    """Per-voice-channel pipeline state.

    Holds live voice client + recording-sink / pump / source tasks for
    clean ``/unsubscribe-voice`` teardown.

    ``transcribers`` / ``fanin_tasks`` keyed by Discord user_id — pump
    lazily clones a Deepgram stream per speaker (per-SSRC audio gives
    natural isolation). One fan-in task per user forwards results onto
    the shared ``result_queue`` tagged with originating user_id.
    """

    voice_client: discord.VoiceClient
    sink: RecordingSink
    audio_queue: asyncio.Queue[tuple[int, bytes]]
    result_queue: asyncio.Queue[TranscriptionResult]
    source: VoiceSource
    pump_task: asyncio.Task[None]
    source_task: asyncio.Task[None]
    transcribers: dict[int, Transcriber] = field(default_factory=dict)
    fanin_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    # idle watchdog closes per-user streams after extended silence so
    # Deepgram doesn't tear them down server-side mid-utterance.
    # reopened lazily on next audio chunk for that user.
    idle_watchdog_task: asyncio.Task[None] | None = None
    # V1 phase 2: per-user local turn-endpointer (TEN-VAD + Smart Turn).
    # empty when ``familiar.local_turn_detector`` unset.
    endpointers: dict[int, UtteranceEndpointer] = field(default_factory=dict)


@dataclass
class BotHandle:
    """Bot + outbound seams for bus processors.

    send_text: seam :class:`TextResponder` injects to post replies
    without depending on pycord. Returns posted-message id (``str``)
    for future reply lookups; ``None`` on send failure.
    voice_runtime: keyed by voice-channel id; populated by
    ``/subscribe-voice``, read by active TTS player to find the live
    voice client.
    resolve_member: ``(channel_id, user_id) → Author``; consumed by
    :class:`VoiceResponder` so voice user turns get the same
    speaker-attributed prefixes as text turns.
    voice_members: side cache for voice-only members. Without the
    privileged ``members`` intent, ``guild.get_member()`` only knows
    users seen via other events (text, voice-state changes); voice-only
    joiners stay invisible. Populated by voice-state events + background
    ``guild.fetch_member()`` triggered on first audio per user_id.
    """

    bot: discord.Bot
    send_text: SendText
    voice_runtime: dict[int, VoiceRuntime] = field(default_factory=dict)
    resolve_member: ResolveMember | None = None
    voice_members: dict[int, Author] = field(default_factory=dict)
    trigger_typing: TriggerTyping | None = None
    typing_interrupt: TypingInterruptHandler | None = None
    """Policy seam for ``on_typing`` events; consumed by ``TextResponder``."""


async def _on_recording_done(sink: RecordingSink, *args: object) -> None:  # noqa: RUF029 — pycord requires coroutine fn even when there's nothing to await
    """py-cord ``start_recording`` callback. No-op — sink writes via queue.

    Must be ``async def`` — pycord schedules the return value with
    ``asyncio.run_coroutine_threadsafe`` (voice_client.py:915), which
    requires a coroutine.
    """
    del sink, args


async def _prefetch_voice_member(
    *, handle: BotHandle, channel_id: int, user_id: int
) -> None:
    """Populate ``handle.voice_members[user_id]``; safe to fire repeatedly.

    Order: cache → ``guild.get_member`` → ``guild.fetch_member`` (REST).
    Each step bails on hit. Errors are swallowed — a missing voice
    name is recoverable; a crashing prefetch task isn't.
    """
    if user_id in handle.voice_members:
        return
    channel = handle.bot.get_channel(channel_id)
    guild = getattr(channel, "guild", None)
    if guild is None:
        return
    try:
        member = guild.get_member(user_id)
        if member is None:
            member = await guild.fetch_member(user_id)
    except Exception as exc:  # noqa: BLE001
        _logger.debug(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('member_prefetch_error', repr(exc), vc=ls.Y)}"
        )
        return
    if member is None:
        return
    handle.voice_members[user_id] = Author.from_discord_member(member)


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
    detector = getattr(familiar, "local_turn_detector", None)
    transcribers: dict[int, Transcriber] = {}
    fanin_tasks: dict[int, asyncio.Task[None]] = {}
    endpointers: dict[int, UtteranceEndpointer] = {}
    # last audio-chunk arrival per user (monotonic seconds). drives the
    # idle watchdog below; entries removed when the stream is closed.
    last_audio_time: dict[int, float] = {}

    async def _fanin(user_id: int, q: asyncio.Queue[TranscriptionResult]) -> None:
        """Tag each result with ``user_id`` and forward to shared queue."""
        while True:
            result = await q.get()
            result.user_id = user_id
            await result_queue.put(result)

    async def _ensure_transcriber(user_id: int) -> Transcriber:
        existing = transcribers.get(user_id)
        if existing is not None:
            return existing
        last_audio_time[user_id] = time.monotonic()
        clone = template.clone()
        # V1 phase 2: when local turn detection owns endpointing, drive
        # Deepgram with a near-zero hosted endpointer so it relies on
        # ``Finalize`` from the local chain. Backend-specific knob —
        # not on the Transcriber Protocol; setattr keeps the typing
        # honest while still no-oping for backends that lack the field
        # (V3 phase 2/3, plus mocked clones in tests).
        if detector is not None and hasattr(clone, "endpointing_ms"):
            setattr(clone, "endpointing_ms", 10)  # noqa: B010
        per_user_q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await clone.start(per_user_q)
        transcribers[user_id] = clone
        if detector is not None:

            async def _on_complete(_audio: bytes, uid: int = user_id) -> None:
                # Park vad_end ahead of finalize so the buffered timestamp
                # reaches VoiceSource before the resulting stt_final.
                source.record_vad_end(user_id=uid)
                target = transcribers.get(uid)
                if target is None:
                    return
                with contextlib.suppress(Exception):
                    await target.finalize()

            endpointers[user_id] = detector.make_endpointer(
                on_turn_complete=_on_complete,
            )
        fanin_tasks[user_id] = asyncio.create_task(
            _fanin(user_id, per_user_q),
            name=f"voice-fanin-{channel_id}-{user_id}",
        )
        # Warm the voice-member cache. Voice-only users who haven't sent
        # text aren't in ``guild._members`` (no privileged ``members``
        # intent), so the resolver would miss them and the bot would
        # log voice turns anonymously. Fire-and-forget — fetch races
        # with the user's first utterance and almost always wins.
        asyncio.create_task(  # noqa: RUF006 — best-effort cache warm
            _prefetch_voice_member(
                handle=handle, channel_id=channel_id, user_id=user_id
            ),
            name=f"voice-prefetch-{channel_id}-{user_id}",
        )
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.G)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)} "
            f"{ls.kv('transcriber', 'opened', vc=ls.LG)}"
        )
        return clone

    async def _close_user_stream(user_id: int, *, reason: str) -> None:
        """Tear down a per-user transcriber; reopened lazily on next audio."""
        clone = transcribers.pop(user_id, None)
        fanin = fanin_tasks.pop(user_id, None)
        endpointers.pop(user_id, None)
        last_audio_time.pop(user_id, None)
        if fanin is not None:
            fanin.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fanin
        if clone is not None:
            with contextlib.suppress(Exception):
                await clone.stop()
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.Y)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)} "
            f"{ls.kv('transcriber', f'closed-{reason}', vc=ls.LY)}"
        )

    async def _idle_watchdog(idle_close_s: float) -> None:
        """Pre-empt Deepgram's silence-based session close.

        Per-user streams that go quiet still send KeepAlive every few
        seconds, but Deepgram closes them anyway after extended silence
        (observed: clean 1000 close after the user's last real audio,
        regardless of KeepAlive flow). Closing proactively avoids the
        reconnect + replay cycle and keeps logs quiet. Stream is
        reopened on the user's next audio chunk via ``_ensure_transcriber``.
        """
        # quarter the idle window so a stale stream is closed within
        # ~25 % of the threshold; floor keeps CPU negligible without
        # forcing slow tests to wait a full second per scan.
        interval = max(idle_close_s / 4.0, 0.01)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale = [
                uid
                for uid, last in list(last_audio_time.items())
                if now - last > idle_close_s
            ]
            for uid in stale:
                await _close_user_stream(uid, reason="idle")

    async def _pump_audio() -> None:
        while True:
            user_id, pcm = await audio_queue.get()
            last_audio_time[user_id] = time.monotonic()
            clone = await _ensure_transcriber(user_id)
            await clone.send_audio(pcm)
            ep = endpointers.get(user_id)
            if ep is not None:
                with contextlib.suppress(Exception):
                    await ep.feed_audio(pcm)

    pump_task = asyncio.create_task(_pump_audio(), name=f"voice-pump-{channel_id}")
    source_task = asyncio.create_task(source.run(), name=f"voice-source-{channel_id}")
    idle_close_s = float(getattr(template, "_IDLE_CLOSE_S", 0.0))
    watchdog_task: asyncio.Task[None] | None = None
    if idle_close_s > 0:
        watchdog_task = asyncio.create_task(
            _idle_watchdog(idle_close_s),
            name=f"voice-idle-watchdog-{channel_id}",
        )
    rt = VoiceRuntime(
        voice_client=voice_client,
        sink=sink,
        audio_queue=audio_queue,
        result_queue=result_queue,
        source=source,
        pump_task=pump_task,
        source_task=source_task,
        transcribers=transcribers,
        fanin_tasks=fanin_tasks,
        idle_watchdog_task=watchdog_task,
        endpointers=endpointers,
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
    if rt.idle_watchdog_task is not None:
        rt.idle_watchdog_task.cancel()
    for t in rt.fanin_tasks.values():
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.pump_task
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.source_task
    if rt.idle_watchdog_task is not None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await rt.idle_watchdog_task
    if rt.fanin_tasks:
        await asyncio.gather(*rt.fanin_tasks.values(), return_exceptions=True)
    # Per-user transcriber teardown in parallel — each WS close is
    # independent. Sequential awaits would N-times the unsubscribe
    # latency and run the slash-command handler past Discord's 3 s
    # interaction-token deadline.
    if rt.transcribers:
        await asyncio.gather(
            *(clone.stop() for clone in rt.transcribers.values()),
            return_exceptions=True,
        )
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

    @contextlib.asynccontextmanager
    async def trigger_typing(channel_id: int):  # noqa: ANN202 — py-cord typing CM
        """Run ``async with channel.typing():`` for *channel_id*.

        Falls through silently when the channel isn't messageable yet
        (cache miss right after startup) or Discord rejects the
        ``typing`` REST call; the bot still posts the reply.
        """
        channel = bot.get_channel(channel_id)
        if channel is None or not isinstance(channel, discord.abc.Messageable):
            yield
            return
        try:
            cm = channel.typing()
        except discord.DiscordException as exc:
            _logger.debug(
                f"{ls.tag('💬 Text', ls.Y)} "
                f"{ls.kv('typing_init_error', repr(exc), vc=ls.Y)}"
            )
            yield
            return
        async with cm:
            yield

    typing_interrupt = TypingInterruptHandler(
        config=familiar.config.discord_text,
        router=familiar.router,
        is_subscribed=lambda ch: (
            familiar.subscriptions.get(channel_id=ch, kind=SubscriptionKind.text)
            is not None
        ),
        bot_user_id_provider=lambda: familiar.bot_user_id,
    )
    handle = BotHandle(
        bot=bot,
        send_text=send_text,
        trigger_typing=trigger_typing,
        typing_interrupt=typing_interrupt,
    )

    def resolve_member(channel_id: int, user_id: int) -> Author | None:
        """Look up Discord member for a voice user_id; return Author.

        Order: voice-member side cache → ``guild.get_member`` (cache).
        Returns ``None`` on miss; the caller treats that as an anonymous
        voice turn rather than blocking on a Discord fetch — the audio
        path can't tolerate REST round-trips. Prefetch warms the cache
        in the background per ``_prefetch_voice_member``.
        """
        cached = handle.voice_members.get(user_id)
        if cached is not None:
            return cached
        channel = bot.get_channel(channel_id)
        guild = getattr(channel, "guild", None)
        if guild is None:
            return None
        member = guild.get_member(user_id)
        if member is None:
            return None
        author = Author.from_discord_member(member)
        handle.voice_members[user_id] = author
        return author

    handle.resolve_member = resolve_member

    text_source = DiscordTextSource(bus=familiar.bus, familiar_id=familiar.id)
    _register_slash_commands(handle, familiar)
    _register_events(bot, familiar, text_source, handle)

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

        # Defer immediately — Discord's interaction token expires after
        # 3 s. With per-user transcribers each having their own WS,
        # teardown easily exceeds that. defer() converts the response
        # to "thinking…"; the followup below replaces it.
        with contextlib.suppress(discord.DiscordException):
            await ctx.defer(ephemeral=True)

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
        with contextlib.suppress(discord.DiscordException):
            await ctx.followup.send("Left voice channel.", ephemeral=True)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _register_events(
    bot: discord.Bot,
    familiar: Familiar,
    text_source: DiscordTextSource,
    handle: BotHandle,
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
    async def on_typing(  # noqa: RUF029 — Discord event handler contract
        channel: discord.abc.Messageable,
        user: discord.User | discord.Member,
        when: object,  # discord passes a datetime; unused here
    ) -> None:
        del when
        if handle.typing_interrupt is None:
            return
        channel_id = getattr(channel, "id", None)
        if not isinstance(channel_id, int):
            return
        handle.typing_interrupt.notify_typing(
            channel_id=channel_id,
            user_id=int(user.id),
            is_bot=bool(getattr(user, "bot", False)),
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
        # Warm the voice-member cache. This is the only reliable place
        # to learn voice-only members without the privileged ``members``
        # intent — message events miss anyone who never types.
        handle.voice_members[member.id] = Author.from_discord_member(member)
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.G)} "
            f"{ls.kv('member', member.display_name, vc=ls.LC)} "
            f"{ls.kv('channel', after.channel.name, vc=ls.LC)}"
        )
