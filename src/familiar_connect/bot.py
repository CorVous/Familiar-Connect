"""Discord bot shell.

Owns:

- Discord client (``py-cord`` + DAVE voice)
- Subscribe / unsubscribe slash commands (text + voice)
- ``on_message`` / ``on_voice_state_update`` handlers
- :func:`ingest_event` — publishes text + voice events to bus via
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
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import discord

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.collector import get_span_collector
from familiar_connect.diagnostics.report import render_summary_table
from familiar_connect.identity import Author
from familiar_connect.sources import DiscordTextSource
from familiar_connect.sources.discord_embed_text import format_embeds
from familiar_connect.sources.voice import VoiceSource
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.typing_interrupt import TypingInterruptHandler
from familiar_connect.voice import DaveVoiceClient
from familiar_connect.voice.recording_sink import RecordingSink

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from contextlib import AbstractAsyncContextManager as AsyncContextManager

    from familiar_connect.activities.engine import ActivityEngine
    from familiar_connect.familiar import Familiar
    from familiar_connect.focus import FocusManager
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.stt import Transcriber, TranscriptionResult
    from familiar_connect.voice.turn_detection import UtteranceEndpointer

    # Outbound text callback. returns posted-message id for reply-chain
    # tracking, None on send failure. ``reply_to_message_id`` threads
    # via ``MessageReference``; ``mention_user_ids`` →
    # ``AllowedMentions(users=...)`` so only resolved targets ping.
    SendText = Callable[
        [int, str, str | None, tuple[int, ...]],
        Awaitable[str | None],
    ]

    # Py-cord ``Bot is typing…`` indicator hook. Returns fresh async
    # context manager wrapping response generation so indicator stays
    # for duration. Implemented in ``create_bot`` via ``channel.typing()``.
    TriggerTyping = Callable[[int], AsyncContextManager[None]]

    # Resolves Discord member for voice turns so they carry same
    # author attribution as text turns.
    ResolveMember = Callable[[int, int], "Author | None"]

_logger = logging.getLogger(__name__)


async def _defer_interaction(ctx: discord.ApplicationContext) -> bool:
    """ACK interaction ASAP to claim Discord's 3s response window.

    Slash handlers that don't defer race the 3s deadline and fail with
    ``NotFound (10062)`` under loop pressure or gateway lag. Returns
    ``False`` when the interaction is already gone.
    """
    try:
        await ctx.defer(ephemeral=True)
    except discord.NotFound:
        name = ctx.command.name if ctx.command else "?"
        _logger.warning(
            f"{ls.tag('Discord', ls.Y)} {ls.kv('stale_interaction', name, vc=ls.LY)}"
        )
        return False
    return True


async def _reply(ctx: discord.ApplicationContext, message: str) -> None:
    """Ephemeral followup reply; swallow ``NotFound`` from a dead interaction.

    Pairs with :func:`_defer_interaction` — the action already ran, so a
    gone interaction just means the confirmation can't be delivered.
    """
    try:
        await ctx.followup.send(message, ephemeral=True)
    except discord.NotFound:
        _logger.warning(
            f"{ls.tag('Discord', ls.Y)} "
            f"{ls.kv('reply_dropped', 'interaction_gone', vc=ls.LY)}"
        )


@dataclass
class VoiceRuntime:
    """Per-voice-channel pipeline state.

    Holds live voice client + recording-sink / pump / source tasks for
    clean ``/unsubscribe-voice`` teardown.

    ``transcribers`` / ``fanin_tasks`` / ``user_pump_tasks`` keyed by
    Discord user_id — shared ``pump_task`` only demuxes sink queue to
    per-user pumps. Per-user pumps own ``send_audio`` + ``feed_audio``
    so one slow speaker (network blip, slow VAD, GC pause) can't stall
    others' audio path. Per-SSRC audio gives natural isolation; one
    fan-in task per user forwards results to shared ``result_queue``
    tagged with originating user_id.
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
    # Per-user audio drain tasks — one ``send_audio`` + one
    # ``feed_audio`` per chunk per user, parallel across users.
    # Populated lazily on first audio per user_id.
    user_pump_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    # Closes per-user streams after extended silence so Deepgram
    # doesn't tear them down server-side mid-utterance. reopened
    # lazily on next audio chunk for that user.
    idle_watchdog_task: asyncio.Task[None] | None = None
    # V1 phase 2: per-user local turn-endpointer (TEN-VAD + Smart Turn).
    # Empty when ``familiar.local_turn_detector`` unset.
    endpointers: dict[int, UtteranceEndpointer] = field(default_factory=dict)


@dataclass
class BotHandle:
    """Bot + outbound seams for bus processors.

    send_text: seam :class:`TextResponder` injects to post replies
    without depending on pycord. Returns posted-message id for future
    reply lookups; ``None`` on send failure.
    voice_runtime: keyed by voice-channel id; populated by
    ``/subscribe-voice``, read by active TTS player to find live voice
    client.
    resolve_member: ``(channel_id, user_id) → Author``; consumed by
    :class:`VoiceResponder` so voice user turns get same
    speaker-attributed prefixes as text turns.
    voice_members: side cache for voice-only members. Without
    privileged ``members`` intent, ``guild.get_member()`` only knows
    users seen via other events (text, voice-state changes); voice-only
    joiners stay invisible. Populated by voice-state events + background
    ``guild.fetch_member()`` on first audio per user_id.
    """

    bot: discord.Bot
    send_text: SendText
    voice_runtime: dict[int, VoiceRuntime] = field(default_factory=dict)
    resolve_member: ResolveMember | None = None
    voice_members: dict[int, Author] = field(default_factory=dict)
    trigger_typing: TriggerTyping | None = None
    typing_interrupt: TypingInterruptHandler | None = None
    """Seam for ``on_typing`` events; consumed by ``TextResponder``."""
    focus_manager: FocusManager | None = None
    """Attentional focus controller; wired in when available."""
    activity_engine: ActivityEngine | None = None
    """Absence controller; ``on_ready`` resyncs away presence via it."""


async def _on_recording_done(sink: RecordingSink, *args: object) -> None:  # noqa: RUF029 — pycord requires coroutine fn even when there's nothing to await
    """py-cord ``start_recording`` callback. No-op — sink writes via queue.

    Must be ``async def`` — pycord schedules return via
    ``asyncio.run_coroutine_threadsafe`` (voice_client.py:915), needs
    coroutine.
    """
    del sink, args


async def _prefetch_voice_member(
    *, handle: BotHandle, channel_id: int, user_id: int
) -> None:
    """Populate ``handle.voice_members[user_id]``; safe to fire repeatedly.

    Order: cache → ``guild.get_member`` → ``guild.fetch_member`` (REST).
    Each step bails on hit. Errors swallowed — missing voice name
    recoverable; crashing prefetch task isn't.
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

    Returns ``None`` when no transcriber configured — bot can still
    join for TTS playback only. Idempotent: second call for same
    channel returns existing runtime.

    ``familiar.transcriber`` treated as *template*: fresh Deepgram WS
    cloned per Discord user_id on first audio from that user. Per-user
    streams kill mixed-stream endpointing (one speaker's pause
    finalizing another's mid-sentence) and inherit attribution from
    Discord's per-SSRC delivery.
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
    # Per-user audio drain tasks + inbound queues. router below
    # demuxes shared ``audio_queue`` into these so one slow user's
    # ``send_audio``/``feed_audio`` can't head-of-line-block the rest.
    user_pump_tasks: dict[int, asyncio.Task[None]] = {}
    user_audio_queues: dict[int, asyncio.Queue[bytes]] = {}
    # Last audio-chunk arrival per user (monotonic seconds). drives
    # idle watchdog below; entries removed when stream closed.
    last_audio_time: dict[int, float] = {}

    async def _fanin(user_id: int, q: asyncio.Queue[TranscriptionResult]) -> None:
        """Tag result with ``user_id``, forward to shared queue."""
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
        # Deepgram with near-zero hosted endpointer so it relies on
        # ``Finalize`` from local chain. Backend-specific knob — not on
        # Transcriber Protocol; setattr keeps typing honest while
        # no-oping for backends lacking field (V3 phase 2/3, plus
        # mocked clones in tests).
        if detector is not None and hasattr(clone, "endpointing_ms"):
            setattr(clone, "endpointing_ms", 10)  # noqa: B010
        per_user_q: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        await clone.start(per_user_q)
        transcribers[user_id] = clone
        if detector is not None:

            async def _on_complete(_audio: bytes, uid: int = user_id) -> None:
                # Park vad_end ahead of finalize so buffered timestamp
                # reaches VoiceSource before resulting stt_final.
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
        # Warm voice-member cache. voice-only users who haven't sent
        # text aren't in ``guild._members`` (no privileged ``members``
        # intent), so resolver would miss them and bot would log
        # voice turns anonymously. fire-and-forget — fetch races with
        # user's first utterance and almost always wins.
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
        """Tear down per-user transcriber; reopened lazily on next audio."""
        clone = transcribers.pop(user_id, None)
        fanin = fanin_tasks.pop(user_id, None)
        pump = user_pump_tasks.pop(user_id, None)
        user_audio_queues.pop(user_id, None)
        endpointers.pop(user_id, None)
        last_audio_time.pop(user_id, None)
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
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
        (observed: clean 1000 close after user's last real audio,
        regardless of KeepAlive flow). Closing proactively avoids
        reconnect + replay cycle, keeps logs quiet. Stream reopened
        on user's next audio chunk via ``_ensure_transcriber``.
        """
        # Quarter idle window so stale stream closed within ~25 % of
        # threshold; floor keeps CPU negligible without forcing slow
        # tests to wait a full second per scan.
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

    async def _user_pump(user_id: int, q: asyncio.Queue[bytes]) -> None:
        """Drain one user's audio: ``send_audio`` + ``feed_audio`` in FIFO.

        Owns per-user ``_ensure_transcriber`` await so slow websocket
        connect for user A doesn't stall router demux for B, C, ….
        Exceptions on inner awaits swallowed (matching prior
        ``feed_audio`` behaviour); broken transcriber recovered by
        idle watchdog reopening stream after silence.
        """
        try:
            clone = await _ensure_transcriber(user_id)
        except Exception as exc:  # noqa: BLE001 — best-effort start
            _logger.warning(
                f"{ls.tag('🎙️  Voice', ls.Y)} "
                f"{ls.kv('user', str(user_id), vc=ls.LC)} "
                f"{ls.kv('transcriber_start_error', repr(exc), vc=ls.Y)}"
            )
            user_pump_tasks.pop(user_id, None)
            user_audio_queues.pop(user_id, None)
            return
        while True:
            pcm = await q.get()
            with contextlib.suppress(Exception):
                await clone.send_audio(pcm)
            ep = endpointers.get(user_id)
            if ep is not None:
                with contextlib.suppress(Exception):
                    await ep.feed_audio(pcm)

    async def _route_audio() -> None:
        """Demux sink → per-user pumps. No per-chunk awaits past dispatch."""
        while True:
            user_id, pcm = await audio_queue.get()
            last_audio_time[user_id] = time.monotonic()
            q = user_audio_queues.get(user_id)
            if q is None:
                q = asyncio.Queue()
                user_audio_queues[user_id] = q
                user_pump_tasks[user_id] = asyncio.create_task(
                    _user_pump(user_id, q),
                    name=f"voice-user-pump-{channel_id}-{user_id}",
                )
            q.put_nowait(pcm)

    pump_task = asyncio.create_task(_route_audio(), name=f"voice-pump-{channel_id}")
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
        user_pump_tasks=user_pump_tasks,
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
    del familiar  # Template not stopped — only clones own connections
    rt = handle.voice_runtime.pop(channel_id, None)
    if rt is None:
        return
    with contextlib.suppress(Exception):
        rt.voice_client.stop_recording()
    rt.pump_task.cancel()
    rt.source_task.cancel()
    if rt.idle_watchdog_task is not None:
        rt.idle_watchdog_task.cancel()
    for t in rt.user_pump_tasks.values():
        t.cancel()
    for t in rt.fanin_tasks.values():
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.pump_task
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await rt.source_task
    if rt.idle_watchdog_task is not None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await rt.idle_watchdog_task
    if rt.user_pump_tasks:
        await asyncio.gather(*rt.user_pump_tasks.values(), return_exceptions=True)
    if rt.fanin_tasks:
        await asyncio.gather(*rt.fanin_tasks.values(), return_exceptions=True)
    # Per-user transcriber teardown in parallel — each WS close is
    # independent. sequential awaits would N-times unsubscribe
    # latency and push slash-command handler past Discord's 3 s
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
    images: dict[str, str] | None = None,
    pings_bot: bool = False,
) -> None:
    """Publish text event onto bus.

    Named function so callers (currently only ``on_message``) have
    one seam to point at. Logging lives in debug processor; this
    function stays narrow.
    """
    del familiar  # Reserved for future per-familiar routing
    await source.publish_text(
        channel_id=channel_id,
        guild_id=guild_id,
        author=author,
        content=text,
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        mentions=mentions,
        images=images or {},
        pings_bot=pings_bot,
    )


def message_pings_bot(
    message: discord.Message,
    bot_user_id: int | None,
) -> bool:
    """Real bot ping: bot user appears in ``message.mentions``.

    py-cord puts both ``<@id>`` mentions and reply-ping targets in
    ``mentions``; role/@everyone mentions live elsewhere and never
    count. Bare name-mentions don't count either.
    """
    if bot_user_id is None:
        return False
    return any(getattr(u, "id", None) == bot_user_id for u in message.mentions)


def compose_content_with_embeds(
    content: str,
    embeds: Iterable[object],
) -> str:
    """Append rendered embed text to ``content``.

    Mirrors what humans see in client — message body plus Discord's
    URL unfurl below. ``embeds`` may be empty (no unfurl yet) or
    contain blank entries; both collapse to original ``content``.
    Empty body + non-empty embeds yields embed text alone (no
    leading blank line).
    """
    embed_text = format_embeds(embeds)
    if not embed_text:
        return content
    if not content:
        return embed_text
    return f"{content}\n\n{embed_text}"


# Regex for inline image URLs not already captured as attachments/embeds.
# Matches http(s) URLs ending with common image extensions (with optional query string)
_IMAGE_URL_RE = re.compile(
    r"https?://\S+\.(?:png|jpe?g|gif|webp|bmp|tiff?)(?:\?\S+)?",
    re.IGNORECASE,
)

_IMAGE_CONTENT_TYPE_PREFIXES = ("image/",)
_IMAGE_EXTENSIONS = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
})


def _is_image_attachment(attachment: object) -> bool:
    """Return True when attachment is an image by content-type or extension."""
    ct = getattr(attachment, "content_type", None) or ""
    if isinstance(ct, str) and ct.lower().startswith(_IMAGE_CONTENT_TYPE_PREFIXES):
        return True
    filename = getattr(attachment, "filename", None) or ""
    if isinstance(filename, str):
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in _IMAGE_EXTENSIONS
    return False


def collect_images(
    *,
    content: str,
    attachments: Iterable[object],
    embeds: Iterable[object],
) -> tuple[str, dict[str, str]]:
    """Return ``(content_with_placeholders, img_id -> url)``.

    Sources in order: attachments, embed.image.url, inline image URLs in
    content. Assigns img_0, img_1 … ; appends ``[image: img_N (filename)]``
    markers to content. Dedupes by URL — same URL only gets one id.
    """
    images: dict[str, str] = {}  # Img_id → url
    seen_urls: dict[str, str] = {}  # Url → img_id
    markers: list[str] = []

    def _add(url: str, filename: str) -> None:
        if url in seen_urls:
            return
        img_id = f"img_{len(images)}"
        images[img_id] = url
        seen_urls[url] = img_id
        markers.append(f"[image: {img_id} ({filename})]")

    for att in attachments:
        if not _is_image_attachment(att):
            continue
        url = getattr(att, "url", None) or ""
        filename = getattr(att, "filename", None) or url.rsplit("/", 1)[-1] or "image"
        if url:
            _add(url, filename)

    for embed in embeds:
        img = getattr(embed, "image", None)
        if img is None:
            continue
        # Prefer proxy_url — Discord's re-hosted copy is more reliably fetchable
        # than the original source URL, which may be unsigned or externally hosted
        url = getattr(img, "proxy_url", None) or getattr(img, "url", None) or ""
        if not url:
            continue
        filename = url.rsplit("/", 1)[-1].split("?")[0] or "embed-image"
        _add(url, filename)

    for match in _IMAGE_URL_RE.finditer(content):
        url = match.group(0)
        filename = url.rsplit("/", 1)[-1].split("?")[0] or "image"
        _add(url, filename)

    if not markers:
        return content, images
    marker_text = "\n".join(markers)
    new_content = f"{content}\n{marker_text}" if content else marker_text
    return new_content, images


def apply_message_edit(
    *,
    store: HistoryStore,
    familiar_id: str,
    is_subscribed: Callable[[int], bool],
    channel_id: int,
    message_id: int | str,
    content: str,
    embeds: Iterable[object],
) -> None:
    """Refresh stored turn's content when Discord adds an embed.

    Pure dispatcher — separated from gateway handler so testable
    without spinning up Discord bot. No-op when:

    - channel isn't text-subscribed (write nothing we can't read)
    - edit carries no embed (pure text edits aren't tracked)
    - no stored turn matches ``message_id`` (bot came up late)

    Embed text merged into ``content`` via
    :func:`compose_content_with_embeds`; original ``message_id``
    column lets FTS update trigger reindex transparently.
    """
    if not is_subscribed(channel_id):
        return
    embed_text = format_embeds(embeds)
    if not embed_text:
        return
    merged = compose_content_with_embeds(content, embeds)
    store.update_turn_content_by_message_id(
        familiar_id=familiar_id,
        platform_message_id=str(message_id),
        content=merged,
    )


def _emoji_repr(emoji: discord.PartialEmoji) -> str:
    """Stable string for :class:`discord.PartialEmoji`.

    Unicode emoji → char itself. Custom emoji → ``<:name:id>``
    (or ``<a:name:id>`` for animated). Empty input returns ``""`` —
    caller short-circuits.
    """
    if emoji.id is None:
        return emoji.name or ""
    if emoji.name is None:
        return ""
    prefix = "a" if emoji.animated else ""
    return f"<{prefix}:{emoji.name}:{emoji.id}>"


def apply_reaction_delta(
    *,
    store: HistoryStore,
    familiar_id: str,
    is_subscribed: Callable[[int], bool],
    channel_id: int,
    message_id: int,
    emoji: discord.PartialEmoji,
    delta: int,
) -> None:
    """Apply ``delta`` to one (message, emoji) row.

    Pure dispatcher — separated from gateway handler so testable
    without spinning up Discord bot. Channel-subscription check up
    front avoids writing rows we'll never read.
    """
    if not is_subscribed(channel_id):
        return
    name = _emoji_repr(emoji)
    if not name:
        return
    store.bump_reaction(
        familiar_id=familiar_id,
        platform_message_id=str(message_id),
        emoji=name,
        delta=delta,
    )


def apply_reaction_clear(
    *,
    store: HistoryStore,
    familiar_id: str,
    is_subscribed: Callable[[int], bool],
    channel_id: int,
    message_id: int,
    emoji: discord.PartialEmoji | None = None,
) -> None:
    """Drop all reactions on message, optionally scoped to one emoji."""
    if not is_subscribed(channel_id):
        return
    name = _emoji_repr(emoji) if emoji is not None else None
    store.clear_reactions(
        familiar_id=familiar_id,
        platform_message_id=str(message_id),
        emoji=name,
    )


# ---------------------------------------------------------------------------
# Presence sync
# ---------------------------------------------------------------------------


async def _sync_presence(bot: discord.Bot, fm: FocusManager) -> None:
    """Update bot presence to reflect current text focus."""
    guild = fm.presence_guild()
    channel = fm.presence_text()
    if guild and channel:
        state = f"✨ {guild} -> {channel}"
    elif channel:
        state = f"✨ {channel}"
    else:
        state = None
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.custom,
            name="Custom Status",
            state=state,
        ),
        status=discord.Status.online,
    )


def build_activity_presence_cb(
    handle: BotHandle,
) -> Callable[[str, str | None], Awaitable[None]]:
    """Presence callback for :class:`ActivityEngine`.

    ``("idle", label)`` while out reachable — yellow dot + activity
    label; ``("dnd", label)`` while out unreachable — red dot, same
    label; ``("online", None)`` on return — restores normal focus
    presence via :func:`_sync_presence`. No-op until the bot is ready
    (headless tests, pre-login engine start).
    """

    async def _cb(status: str, label: str | None) -> None:
        bot = handle.bot
        if not bot.is_ready():
            return
        if status in {"idle", "dnd"}:
            away = discord.Status.idle if status == "idle" else discord.Status.dnd
            await bot.change_presence(
                status=away,
                activity=discord.CustomActivity(name=label or "away"),
            )
            return
        fm = handle.focus_manager
        if fm is not None:
            await _sync_presence(bot, fm)
        else:
            await bot.change_presence(status=discord.Status.online)

    return _cb


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(
    familiar: Familiar,
    *,
    focus_manager: FocusManager | None = None,
) -> BotHandle:
    """Construct Discord client, register slash commands + events.

    Returns :class:`BotHandle` carrying bot plus ``send_text``
    callback (bus processors use for outbound posts) and per-channel
    voice runtime map.
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
        startup; ``fetch_channel`` is fallback.

        ``reply_to_message_id``: when set, threads post to that
        message via ``discord.MessageReference``.
        ``mention_user_ids``: populates ``AllowedMentions(users=...)``
        so only those user ids receive notification, even if content
        contains other ``<@…>`` markers.

        Returns platform message id of posted message (so
        ``TextResponder`` can persist for future reply lookups), or
        ``None`` on send failure.
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

        # Always restrict who can be pinged. defers @everyone / role
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

        Falls through silently when channel isn't messageable yet
        (cache miss right after startup) or Discord rejects the
        ``typing`` REST call; bot still posts the reply.
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
        focus_manager=focus_manager,
    )

    def resolve_member(channel_id: int, user_id: int) -> Author | None:
        """Look up Discord member for voice user_id; return Author.

        Order: voice-member side cache → ``guild.get_member`` (cache).
        Returns ``None`` on miss; caller treats that as anonymous
        voice turn rather than blocking on Discord fetch — audio path
        can't tolerate REST round-trips. Prefetch warms cache in
        background per ``_prefetch_voice_member``.
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
        await _defer_interaction(ctx)
        if ctx.channel_id is None:
            await _reply(ctx, "No channel in context.")
            return
        familiar.subscriptions.add(
            channel_id=ctx.channel_id,
            kind=SubscriptionKind.text,
            guild_id=ctx.guild_id,
        )
        await _reply(ctx, "Listening in this channel.")

    @bot.slash_command(
        name="unsubscribe-text",
        description="Stop listening for text messages in this channel.",
    )
    async def unsubscribe_text(ctx: discord.ApplicationContext) -> None:
        await _defer_interaction(ctx)
        if ctx.channel_id is None:
            await _reply(ctx, "No channel in context.")
            return
        familiar.subscriptions.remove(
            channel_id=ctx.channel_id,
            kind=SubscriptionKind.text,
        )
        await _reply(ctx, "No longer listening here.")

    @bot.slash_command(
        name="subscribe-voice",
        description="Join your voice channel and listen.",
    )
    async def subscribe_voice(ctx: discord.ApplicationContext) -> None:
        await _defer_interaction(ctx)
        member = ctx.author
        voice_state = getattr(member, "voice", None)
        if voice_state is None or voice_state.channel is None:
            await _reply(ctx, "You must be in a voice channel.")
            return

        channel = voice_state.channel
        try:
            voice_client = await channel.connect(cls=DaveVoiceClient)
        except discord.DiscordException as exc:
            _logger.warning("voice connect failed: %s", exc)
            await _reply(ctx, "Could not join voice.")
            return

        # Bring up sink + transcriber + voice source. returns None if
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
        await _reply(ctx, f"Joined {channel.name}.{suffix}")

    @bot.slash_command(
        name="diagnostics",
        description="Show span timings (last p50/p95 per span).",
    )
    async def diagnostics(ctx: discord.ApplicationContext) -> None:
        await _defer_interaction(ctx)
        summary = get_span_collector().summary()
        text = render_summary_table(summary)

        # Focus + unread summary
        fm = handle.focus_manager
        if fm is not None:
            text_focus = fm.get_focus("text")
            voice_focus = fm.get_focus("voice")
            tf_str = f"#{text_focus}" if text_focus is not None else "unset"
            vf_str = f"#{voice_focus}" if voice_focus is not None else "unset"
            focus_line = f"\nFocus: text={tf_str} voice={vf_str}"
            staged = await familiar.history_store.staged_channels(
                familiar_id=familiar.id
            )
            if staged:
                unreads = ", ".join(
                    f"#{ch_id} ({count})" for ch_id, count in sorted(staged.items())
                )
                focus_line += f"\nUnreads: {unreads}"
            text += focus_line

        await _reply(ctx, text)

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
        # 3 s. with per-user transcribers each having their own WS,
        # teardown easily exceeds that. defer() converts response to
        # "thinking…"; followup below replaces it.
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
    async def on_ready() -> None:
        user = bot.user
        if user is not None:
            familiar.bot_user_id = user.id
        # Populate channel name cache so focus logs show names
        fm = handle.focus_manager
        if fm is not None:
            fm.channel_names.update({
                ch.id: ch.name
                for guild in bot.guilds
                for ch in guild.channels
                if hasattr(ch, "name")
            })
            fm.guild_names.update({
                ch.id: guild.name
                for guild in bot.guilds
                for ch in guild.channels
                if hasattr(ch, "name")
            })
        _logger.info(
            f"{ls.tag('🤖 Ready', ls.G)} "
            f"{ls.kv('user', str(user), vc=ls.LC)} "
            f"{ls.kv('guilds', str(len(bot.guilds)), vc=ls.LC)}"
        )
        if fm is not None:
            _logger.info(
                f"{ls.tag('👁️ Focus', ls.LC)} "
                f"{ls.kv('text', fm.channel_label(fm.get_focus('text')), vc=ls.LW)} "
                f"{ls.kv('voice', fm.channel_label(fm.get_focus('voice')), vc=ls.LW)}"
            )
            fm.on_shift = lambda: _sync_presence(bot, fm)
            await _sync_presence(bot, fm)
        # AFTER the focus sync: re-issue away presence if mid-activity —
        # engine.start() ran pre-login (cb dropped its call), and gateway
        # reconnects reset presence. no-op when idle or engine unwired
        engine = handle.activity_engine
        if engine is not None:
            await engine.resync_presence()

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
        # ``message.embeds`` usually empty here — Discord unfurls URLs
        # server-side and fires ``on_message_edit`` a moment later.
        # Pre-cached unfurls (and bot-author embeds, though bots
        # filtered above) do arrive populated, so merge whatever is
        # on inbound message and let edit handler patch the rest.
        text = compose_content_with_embeds(message.content, message.embeds or ())
        text, images = collect_images(
            content=text,
            attachments=message.attachments or (),
            embeds=message.embeds or (),
        )
        await ingest_event(
            source=text_source,
            familiar=familiar,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            author=Author.from_discord_member(message.author),
            text=text,
            message_id=str(message.id),
            reply_to_message_id=reply_to,
            mentions=mention_authors,
            images=images,
            # mention_authors filters bots, so reply-pings at the bot
            # vanish from metadata — pings_bot carries them explicitly
            pings_bot=message_pings_bot(message, familiar.bot_user_id),
        )

    @bot.event
    async def on_message_edit(  # noqa: RUF029 — Discord event handler contract
        before: discord.Message,
        after: discord.Message,
    ) -> None:
        # Discord fires this when embed unfurl finishes (usually
        # 1-2 s after original message). only care about transitions
        # that *added* embed content — pure text edits aren't tracked
        # here. bot-authored edits skip too: responder owns its own
        # turn writes.
        if familiar.bot_user_id is not None and after.author.id == familiar.bot_user_id:
            return
        if after.author.bot:
            return
        before_embeds = list(getattr(before, "embeds", None) or ())
        after_embeds = list(getattr(after, "embeds", None) or ())
        if not after_embeds or before_embeds == after_embeds:
            return
        apply_message_edit(
            store=familiar.history_store.sync,
            familiar_id=familiar.id,
            is_subscribed=_is_text_subscribed,
            channel_id=after.channel.id,
            message_id=after.id,
            content=after.content or "",
            embeds=after_embeds,
        )

    @bot.event
    async def on_typing(  # noqa: RUF029 — Discord event handler contract
        channel: discord.abc.Messageable,
        user: discord.User | discord.Member,
        when: object,  # Discord passes a datetime; unused here
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

    def _is_text_subscribed(channel_id: int) -> bool:
        return (
            familiar.subscriptions.get(
                channel_id=channel_id, kind=SubscriptionKind.text
            )
            is not None
        )

    @bot.event
    async def on_raw_reaction_add(  # noqa: RUF029 — Discord event handler contract
        payload: discord.RawReactionActionEvent,
    ) -> None:
        apply_reaction_delta(
            store=familiar.history_store.sync,
            familiar_id=familiar.id,
            is_subscribed=_is_text_subscribed,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            emoji=payload.emoji,
            delta=1,
        )

    @bot.event
    async def on_raw_reaction_remove(  # noqa: RUF029 — Discord event handler contract
        payload: discord.RawReactionActionEvent,
    ) -> None:
        apply_reaction_delta(
            store=familiar.history_store.sync,
            familiar_id=familiar.id,
            is_subscribed=_is_text_subscribed,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            emoji=payload.emoji,
            delta=-1,
        )

    @bot.event
    async def on_raw_reaction_clear(  # noqa: RUF029 — Discord event handler contract
        payload: discord.RawReactionClearEvent,
    ) -> None:
        apply_reaction_clear(
            store=familiar.history_store.sync,
            familiar_id=familiar.id,
            is_subscribed=_is_text_subscribed,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
        )

    @bot.event
    async def on_raw_reaction_clear_emoji(  # noqa: RUF029 — Discord event handler contract
        payload: discord.RawReactionClearEmojiEvent,
    ) -> None:
        apply_reaction_clear(
            store=familiar.history_store.sync,
            familiar_id=familiar.id,
            is_subscribed=_is_text_subscribed,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            emoji=payload.emoji,
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
        # Warm voice-member cache. only reliable place to learn
        # voice-only members without privileged ``members`` intent —
        # message events miss anyone who never types.
        handle.voice_members[member.id] = Author.from_discord_member(member)
        _logger.info(
            f"{ls.tag('🎙️  Voice', ls.G)} "
            f"{ls.kv('member', member.display_name, vc=ls.LC)} "
            f"{ls.kv('channel', after.channel.name, vc=ls.LC)}"
        )
