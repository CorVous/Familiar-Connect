"""Discord bot factory, slash commands, and pipeline-routed message loop.

- subscription + channel-mode slash command surface
- ``on_message`` routes subscribed text through :class:`ContextPipeline`
- voice transcription pipeline shares the same context path as text
- single :class:`Familiar` bundle per process lifetime
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import io
import logging
import secrets
import time
from typing import TYPE_CHECKING, cast

import discord
import httpx

from familiar_connect import log_style as ls
from familiar_connect.chattiness import (
    BufferedMessage,
    ChannelContext,
    ChannelKind,
    ResponseTrigger,
)
from familiar_connect.config import ChannelMode
from familiar_connect.context.providers.mode_instructions import resolve_mode_default
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import ContextRequest, Modality, PendingTurn
from familiar_connect.identity import Author
from familiar_connect.metrics import TraceBuilder
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.text.delivery import (
    compute_typing_delay,
    split_reply_into_chunks,
)
from familiar_connect.tts import get_cached_greeting_audio
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice.deepgram_vad import DeepgramVoiceActivityDetector
from familiar_connect.voice.interruption import (
    InterruptionClass,
    InterruptionDetector,
    ResponseState,
    split_at_elapsed,
)
from familiar_connect.voice_lull import VoiceLullMonitor
from familiar_connect.voice_pipeline import get_pipeline, start_pipeline, stop_pipeline

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.familiar import Familiar
    from familiar_connect.transcription import TranscriptionResult
    from familiar_connect.tts import WordTimestamp

_logger = logging.getLogger(__name__)


def _timestamps_to_text(timestamps: list[WordTimestamp]) -> str:
    """Join word timestamps into a space-separated string."""
    return " ".join(ts.word for ts in timestamps)


async def _recording_finished_callback(  # noqa: RUF029
    sink: discord.sinks.Sink,
    *args: object,
) -> None:
    """no-op coroutine required by py-cord's ``start_recording`` API.

    py-cord awaits this internally; cleanup lives in :func:`unsubscribe_voice`.
    """
    del sink, args


def _refresh_channel_context(
    familiar: Familiar,
    channel: (
        discord.abc.GuildChannel
        | discord.abc.PrivateChannel
        | discord.Thread
        | discord.PartialMessageable
    ),
) -> ChannelContext | None:
    """Register channel into the monitor from a live Discord object.

    Dispatches on channel type so all call sites share one implementation.
    Returns the freshly-stored :class:`ChannelContext`, or ``None`` for
    unrecognised types (e.g. ``PartialMessageable``).
    """
    name: str
    kind: ChannelKind
    parent_name: str | None = None
    # StageChannel and VoiceChannel are siblings under VocalGuildChannel;
    # ordering is defensive in case py-cord ever merges them.
    match channel:
        case discord.Thread():
            name = getattr(channel, "name", str(channel.id))
            parent = channel.parent
            parent_name = getattr(parent, "name", None)
            kind = (
                "forum_post" if isinstance(parent, discord.ForumChannel) else "thread"
            )
        case discord.DMChannel():
            recipient = getattr(channel, "recipient", None)
            name = getattr(recipient, "display_name", None) or str(channel.id)
            kind = "dm"
        case discord.GroupChannel():
            name = getattr(channel, "name", None) or str(channel.id)
            kind = "group_dm"
        case discord.StageChannel():
            name = getattr(channel, "name", str(channel.id))
            kind = "stage"
        case discord.VoiceChannel():
            name = getattr(channel, "name", str(channel.id))
            kind = "voice"
        case discord.ForumChannel():
            name = getattr(channel, "name", str(channel.id))
            kind = "forum_root"
        case discord.CategoryChannel():
            name = getattr(channel, "name", str(channel.id))
            kind = "category"
        case discord.TextChannel():
            name = getattr(channel, "name", str(channel.id))
            kind = "text"
        case _:
            return None

    familiar.monitor.register_channel_context(
        channel.id,
        name=name,
        kind=kind,
        parent_name=parent_name,
    )
    return ChannelContext(name=name, kind=kind, parent_name=parent_name)


# ---------------------------------------------------------------------------
# Slash commands — /subscribe-*
# ---------------------------------------------------------------------------


async def subscribe_text(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Register the current text channel, thread, or forum post.

    Threads (regular channel threads or forum posts) are accepted — a
    forum post is itself a ``discord.Thread`` in Discord's API, so one
    branch covers both. The forum channel root itself has no
    conversation surface and is rejected.
    """
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return

    guild = ctx.guild
    channel = ctx.channel

    if not isinstance(channel, discord.TextChannel | discord.Thread):
        await ctx.respond(
            "I can only be summoned to a text channel, a thread, or a forum post.",
            ephemeral=True,
        )
        return
    is_thread = isinstance(channel, discord.Thread)

    if guild is not None:
        perms = channel.permissions_for(guild.me)
        can_send = perms.send_messages_in_threads if is_thread else perms.send_messages
        if not perms.view_channel or not can_send:
            await ctx.respond(
                "My powers don't extend to this channel"
                " \N{EM DASH} I lack the permissions to speak here.",
                ephemeral=True,
            )
            return

    # threads must be joined explicitly; no-op for already-joined
    # public threads. Swallow transient HTTP failures — subscription
    # still proceeds, and py-cord auto-joins on first send.
    if isinstance(channel, discord.Thread):
        with contextlib.suppress(discord.HTTPException):
            await channel.join()

    familiar.subscriptions.add(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
        guild_id=ctx.guild_id,
    )
    ctx_info = cast("ChannelContext", _refresh_channel_context(familiar, channel))
    assert ctx_info is not None  # noqa: S101 — isinstance guard above ensures TextChannel | Thread
    kind = ctx_info.kind
    name = ctx_info.name

    channel_config = familiar.channel_configs.get(channel_id=channel_id)
    if channel_config:
        ch_mode = channel_config.mode.value
    else:
        ch_mode = familiar.config.default_mode.value
    label = familiar.monitor.format_channel_context(channel_id)
    _logger.info(
        f"{ls.tag('✨ Summoned', ls.G)} "
        f"{ls.kv('type', kind)} "
        f"{ls.word(label, ls.C)} "
        f"{ls.kv('id', str(channel_id))} "
        f"{ls.kv('mode', ch_mode)}"
    )
    if kind == "text":
        reply = f"Subscribed to text in **#{name}**."
    elif kind == "thread":
        reply = f"Subscribed to thread **{name}**."
    else:
        reply = f"Subscribed to forum post **{name}**."
    await ctx.respond(reply, ephemeral=True)


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
    # flush can exceed Discord's 3s interaction window (LLM call + file writes)
    await ctx.defer(ephemeral=True)
    await familiar.memory_writer_scheduler.flush()
    await ctx.followup.send("No longer listening here.", ephemeral=True)


async def _run_voice_response(
    channel_id: int,
    guild_id: int | None,
    author: Author,
    utterance: str,
    buffer: list[BufferedMessage],
    familiar: Familiar,
    vc: discord.VoiceClient,
    trigger: ResponseTrigger,
    interruption_context: str | None = None,
) -> None:
    """Run full pipeline → LLM → TTS path for voice channel."""
    # resolve per-guild response tracker; mark unsolicited flag
    tracker = familiar.tracker_registry.get(guild_id if guild_id is not None else 0)
    tracker.vc = vc
    tracker.is_unsolicited = trigger.is_unsolicited
    # Cache the mood modifier for the whole turn — should_keep_talking
    # at Moment 1 must use the same value the tracker saw at generation
    # start, not re-roll mid-response.
    tracker.mood_modifier = await familiar.mood_evaluator.evaluate(
        channel_id=channel_id,
        familiar_id=familiar.id,
    )
    tracker.transition(ResponseState.GENERATING)

    channel_config = familiar.channel_configs.get(channel_id=channel_id)

    # disable LLM-calling providers and processors for voice to reduce
    # real-time latency; remove this replace() call to re-enable
    channel_config = dataclasses.replace(
        channel_config,
        providers_enabled=channel_config.providers_enabled
        - {
            "content_search",
            "history",
        },
        preprocessors_enabled=frozenset(),
        postprocessors_enabled=frozenset(),
    )

    request = ContextRequest(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        author=author,
        utterance=utterance,
        modality=Modality.voice,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=tuple(PendingTurn(author=m.author, text=m.text) for m in buffer),
        interruption_context=interruption_context,
    )

    builder = TraceBuilder(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        modality="voice",
    )
    builder.tag("channel_mode", channel_config.mode.value)
    builder.tag("speaker", author.label)

    pipeline = familiar.build_pipeline(channel_config)
    async with builder.span("pipeline_assembly") as pa_meta:
        pipeline_output = await pipeline.assemble(
            request,
            budget_by_layer=channel_config.budget_by_layer,
        )
        builder.add_provider_outcomes(pa_meta, pipeline_output.outcomes)

    async with builder.span("render") as rmeta:
        messages = assemble_chat_messages(
            pipeline_output,
            store=familiar.history_store,
            history_window_size=familiar.config.history_window_size,
            depth_inject_position=familiar.config.depth_inject_position,
            depth_inject_role=familiar.config.depth_inject_role,
            mode=channel_config.mode,
            display_tz=familiar.config.display_tz,
        )
        rmeta["message_count"] = len(messages)
    familiar.last_context_cache.put(
        channel_id=channel_id, messages=messages, modality="voice"
    )

    n_new = len(request.pending_turns) if request.pending_turns else 1
    pending_lines = (
        "\n".join(
            f"  [{pt.author.label if pt.author else '?'}]: "
            f"{pt.text[:200]}{'…' if len(pt.text) > 200 else ''}"
            for pt in request.pending_turns
        )
        if request.pending_turns
        else (
            f"  [{request.author.label if request.author else '?'}]: "
            f"{request.utterance[:200]}{'…' if len(request.utterance) > 200 else ''}"
        )
    )
    last_hist = (
        f"[{messages[-(n_new + 1)].role}]: {messages[-(n_new + 1)].content[:200]}"
        if len(messages) > n_new
        else ""
    )
    hist = f"  {ls.word(ls.trunc(last_hist), ls.LW)}\n" if last_hist else ""
    _logger.info(
        f"{ls.tag('🧠 Generating Voice', ls.G)} "
        f"{ls.word(str(channel_id), ls.C)} "
        f"{ls.kv('messages', str(len(messages)), vc=ls.LG)} "
        f"{ls.kv('new', str(n_new), vc=ls.LG)}\n"
        f"{hist}"
        f"{ls.word(pending_lines, ls.LW)}"
    )

    # cancellable generation task; parked on tracker for interruption path.
    # errors captured inside span so `llm_call` stage lands in trace;
    # re-raise / early return happens after span exits cleanly.
    llm_error: BaseException | None = None
    reply = None
    gen_start = time.monotonic()
    async with builder.span("llm_call") as llm_meta:
        llm_meta["model"] = familiar.llm_clients["main_prose"].model
        generation_task = asyncio.create_task(
            familiar.llm_clients["main_prose"].chat(messages),
        )
        tracker.generation_task = generation_task
        try:
            reply = await generation_task
        except asyncio.CancelledError as exc:
            llm_meta["error"] = "cancelled"
            llm_error = exc
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            llm_meta["error"] = f"{type(exc).__name__}: {exc}"
            llm_error = exc
        else:
            llm_meta["reply_length"] = len(reply.content)
        finally:
            tracker.generation_task = None
    if isinstance(llm_error, asyncio.CancelledError):
        current = asyncio.current_task()
        tracker.transition(ResponseState.IDLE)
        familiar.metrics_collector.record(builder.finalize())
        if current is not None and current.cancelling() > 0:
            # outer task cancelled from above — propagate
            raise llm_error
        # generation task cancelled (interruption path)
        _logger.info(
            f"{ls.tag('❌ Cancelled', ls.Y)} "
            f"{ls.word(str(channel_id), ls.C)} "
            f"{ls.kv('elapsed', f'{time.monotonic() - gen_start:.2f}s')} "
            f"{ls.kv('reason', 'interruption')} "
            f"{ls.kv('speaker', tracker.interrupt_starter_name or 'unknown', vc=ls.LC)}"
        )
        return
    if llm_error is not None:
        _logger.warning(
            "main reply (voice): %s: %s",
            type(llm_error).__name__,
            llm_error,
        )
        tracker.transition(ResponseState.IDLE)
        familiar.metrics_collector.record(builder.finalize())
        return
    if reply is None:
        # unreachable; type-guard
        return

    async with builder.span("post_processing"):
        reply_text = await pipeline.run_post_processors(reply.content, request)
    tracker.response_text = reply_text

    _logger.info(
        f"{ls.tag('🔊 Generated Voice', ls.G)} "
        f"{ls.word(familiar.monitor.format_channel_context(channel_id), ls.C)} "
        f"{ls.kv('chars', str(len(reply_text)), vc=ls.LG)}\n"
        f"  {ls.word(ls.trunc(reply_text, 500), ls.LG)}"
    )

    if familiar.tts_client is not None:
        try:
            tts_result = await familiar.tts_client.synthesize(reply_text)
            tracker.timestamps = list(tts_result.timestamps)
            stereo = mono_to_stereo(tts_result.audio)
            # Step 8 delivery gate: wait for any active burst (pre-playback)
            # to finalize. Keeps _burst_latest_state=GENERATING so a long
            # burst that straddled TTS synthesis fires on_long_during_generating
            # → regen task, and this response is discarded.
            detector = familiar.extras.get("interruption_detector")
            if isinstance(detector, InterruptionDetector):
                gate_result = await detector.wait_for_lull()
                if gate_result is InterruptionClass.long:
                    # Step 9 — any pending interrupter turns belong to a
                    # discarded response; drop them so the regen starts clean.
                    tracker.pending_interrupter_turns.clear()
                    tracker.transition(ResponseState.IDLE)
                    return
            # Persist user turns only after the delivery gate (discarded
            # responses skip these; a long@GENERATING regen writes its own).
            for msg in buffer:
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="user",
                    content=msg.text,
                    author=msg.author,
                    mode=channel_config.mode,
                )
            # Step 9 — flush interrupter turns captured during GENERATING
            # (short@GENERATING dispatch stashed them on the tracker).
            # Ordering: after buffer, before assistant reply → chronology.
            for interrupter_author, text in tracker.pending_interrupter_turns:
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="user",
                    content=text,
                    author=interrupter_author,
                    mode=channel_config.mode,
                )
            tracker.pending_interrupter_turns.clear()
            _logger.info(
                f"{ls.tag('🔊 Voice', ls.G)} "
                f"{ls.word(str(channel_id), ls.C)} "
                f"{ls.kv('text', ls.trunc(reply_text), vc=ls.LG)}"
            )
            # Wait for any currently-playing audio to finish.
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            tracker.transition(ResponseState.SPEAKING)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            # Step 11/12 post-playback: if an interrupt yield was triggered
            # mid-playback, await burst finalization to learn classification.
            interrupt_event = tracker.interrupt_event
            if interrupt_event is not None:
                await interrupt_event.wait()
            # Step 12: long@SPEAKING yield — write the delivered portion
            # and re-generate with an interruption context note.
            if tracker.interrupt_classification is InterruptionClass.long:
                elapsed_ms = tracker.interruption_elapsed_ms or 0.0
                delivered_ts, _ = split_at_elapsed(tracker.timestamps, elapsed_ms)
                delivered_text = _timestamps_to_text(delivered_ts)
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="assistant",
                    content=delivered_text,
                    mode=channel_config.mode,
                )
                interruption_note = (
                    f'You were speaking and said: "{delivered_text}". '
                    f"{tracker.interrupt_starter_name} interrupted you. "
                    f'They said: "{tracker.interrupt_transcript}"'
                )
                _logger.info(
                    f"{ls.tag('⚡ Dispatch', ls.Y)} "
                    f"{ls.word(str(channel_id), ls.C)} "
                    f"{ls.kv('speaker', tracker.interrupt_starter_name, vc=ls.LC)} "
                    f"{ls.kv('event', 'long@SPEAKING→regen', vc=ls.LY)}"
                )
                regen_request = dataclasses.replace(
                    request, interruption_context=interruption_note
                )
                tracker.transition(ResponseState.IDLE)
                tracker.transition(ResponseState.GENERATING)
                regen_pipeline = familiar.build_pipeline(channel_config)
                regen_output = await regen_pipeline.assemble(
                    regen_request,
                    budget_by_layer=channel_config.budget_by_layer,
                )
                regen_messages = assemble_chat_messages(
                    regen_output,
                    store=familiar.history_store,
                    history_window_size=familiar.config.history_window_size,
                    depth_inject_position=familiar.config.depth_inject_position,
                    depth_inject_role=familiar.config.depth_inject_role,
                    mode=channel_config.mode,
                    display_tz=familiar.config.display_tz,
                )
                familiar.last_context_cache.put(
                    channel_id=channel_id,
                    messages=regen_messages,
                    modality="voice-regen",
                )
                regen_task = asyncio.create_task(
                    familiar.llm_clients["main_prose"].chat(regen_messages),
                )
                tracker.generation_task = regen_task
                try:
                    regen_reply = await regen_task
                except asyncio.CancelledError:
                    tracker.generation_task = None
                    tracker.transition(ResponseState.IDLE)
                    return
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    tracker.generation_task = None
                    _logger.warning(
                        "regen reply (long@SPEAKING): %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    tracker.transition(ResponseState.IDLE)
                    return
                tracker.generation_task = None
                regen_text = await regen_pipeline.run_post_processors(
                    regen_reply.content, regen_request
                )
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="assistant",
                    content=regen_text,
                    mode=channel_config.mode,
                )
                await familiar.memory_writer_scheduler.notify_turn()
                _logger.info(
                    f"{ls.tag('🔊 Voice Regen', ls.LG)} "
                    f"{ls.word(str(channel_id), ls.C)} "
                    f"{ls.kv('text', ls.trunc(regen_text))}"
                )
                regen_tts = await familiar.tts_client.synthesize(regen_text)
                tracker.timestamps = list(regen_tts.timestamps)
                regen_stereo = mono_to_stereo(regen_tts.audio)
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                tracker.transition(ResponseState.SPEAKING)
                vc.play(discord.PCMAudio(io.BytesIO(regen_stereo)))
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                tracker.transition(ResponseState.IDLE)
                return
            # Normal completion or short@SPEAKING yield/push-through —
            # write the full assistant turn. Step 11's resume callback
            # handles re-synthesis of the remaining words separately.
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=channel_id,
                guild_id=guild_id,
                role="assistant",
                content=reply_text,
                mode=channel_config.mode,
            )
            await familiar.memory_writer_scheduler.notify_turn()
        except Exception:
            _logger.exception("Voice response TTS failed")
            tracker.transition(ResponseState.IDLE)
            return
    else:
        # No TTS — write user + assistant history directly (no gates).
        for msg in buffer:
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=channel_id,
                guild_id=guild_id,
                role="user",
                content=msg.text,
                author=msg.author,
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
        await familiar.memory_writer_scheduler.notify_turn()
        _logger.info(
            f"{ls.tag('🔊 Voice', ls.G)} "
            f"{ls.word(str(channel_id), ls.C)} "
            f"{ls.kv('text', ls.trunc(reply_text), vc=ls.LG)}"
        )
    tracker.transition(ResponseState.IDLE)
    familiar.metrics_collector.record(builder.finalize())


async def dispatch_interruption_regen(
    channel_id: int,
    guild_id: int | None,
    author: Author,
    transcript: str,
    familiar: Familiar,
    vc: discord.VoiceClient,
) -> None:
    """Re-generate voice reply after a long interruption during GENERATING.

    Cancels the in-flight LLM task and waits for it to finish so
    :func:`_run_voice_response` can run its ``CancelledError`` handler
    (which transitions the tracker back to ``IDLE``), then calls
    :func:`_run_voice_response` again with an ``interruption_context``
    note so the regenerated reply can acknowledge what the user said.

    Called by the per-guild ``_on_long_during_generating`` closure wired
    inside :func:`subscribe_my_voice` via ``asyncio.create_task``.
    :func:`_run_voice_response` registered its awaiter on
    ``generation_task`` first, so asyncio schedules its cleanup before
    resuming this coroutine — the tracker is ``IDLE`` by the time we
    continue after ``await task``.
    """
    tracker = familiar.tracker_registry.get(guild_id if guild_id is not None else 0)
    task = tracker.generation_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    # Safety net: if _run_voice_response's cleanup hasn't run yet
    # (e.g. task was already done when we got here), ensure clean state.
    if tracker.state is not ResponseState.IDLE:
        tracker.transition(ResponseState.IDLE)
    note = (
        f"{author.label} interrupted while you were forming a response."
        f' They said: "{transcript}"'
    )
    _logger.info(
        f"{ls.tag('⚡ Dispatch', ls.Y)} "
        f"{ls.word(str(channel_id), ls.C)} "
        f"{ls.kv('speaker', author.label, vc=ls.LC)} "
        f"{ls.kv('event', 'long@GENERATING→cancel+regen', vc=ls.LY)}"
    )
    await _run_voice_response(
        channel_id=channel_id,
        guild_id=guild_id,
        author=author,
        utterance=transcript,
        buffer=[
            BufferedMessage(
                author=author,
                text=transcript,
                timestamp=time.monotonic(),
            )
        ],
        familiar=familiar,
        vc=vc,
        trigger=ResponseTrigger.direct_address,
        interruption_context=note,
    )


async def subscribe_my_voice(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Join caller's voice channel and register a voice subscription.

    When ``familiar.transcriber`` is set, starts per-user Deepgram
    pipeline → :class:`VoiceLullMonitor` → ``ConversationMonitor``
    (same gate as text). Without transcriber, joins for TTS only.
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

    # voice connection + DAVE handshake takes >3s, so defer
    await ctx.defer(ephemeral=True)
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
    _logger.info(
        f"{ls.tag('✨ Summoned', ls.G)} "
        f"{ls.kv('type', 'voice')} "
        f"{ls.word(channel.name, ls.C)} "
        f"{ls.kv('id', str(channel.id))}"
    )

    familiar.subscriptions.add(
        channel_id=channel.id,
        kind=SubscriptionKind.voice,
        guild_id=ctx.guild_id,
    )

    if familiar.tts_client is not None:
        try:
            greetings = familiar.config.tts.greetings
            greeting_text = secrets.choice(greetings) if greetings else "Hello!"
            tts_cfg = familiar.config.tts
            if tts_cfg.provider == "cartesia":
                voice_id = tts_cfg.cartesia_voice_id
            elif tts_cfg.provider == "gemini":
                voice_id = tts_cfg.gemini_voice
            else:
                voice_id = tts_cfg.azure_voice
            tts_result = await get_cached_greeting_audio(
                provider=tts_cfg.provider,
                voice_id=voice_id or "",
                greeting=greeting_text,
                client=familiar.tts_client,
            )
            stereo = mono_to_stereo(tts_result.audio)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
        except Exception:
            _logger.exception("Opening greeting TTS failed")

    if familiar.transcriber is not None:
        voice_authors: dict[int, Author] = {
            m.id: Author.from_discord_member(m) for m in channel.members
        }
        user_names = {uid: a.label for uid, a in voice_authors.items()}
        voice_channel_id = channel.id
        voice_guild_id = ctx.guild_id

        def _resolve_from_channel(user_id: int) -> str | None:
            for member in channel.members:
                if member.id == user_id:
                    return member.display_name
            return None

        def _author_for(user_id: int) -> Author:
            cached = voice_authors.get(user_id)
            if cached is not None:
                return cached
            # fallback: look up via live channel members (may have joined late)
            for member in channel.members:
                if member.id == user_id:
                    author = Author.from_discord_member(member)
                    voice_authors[user_id] = author
                    return author
            return Author(
                platform="discord",
                user_id=str(user_id),
                username=None,
                display_name=f"User-{user_id}",
            )

        # per-voice-channel response handler; dispatched by on_respond
        async def _voice_response_handler(
            channel_id: int,
            buffer: list[BufferedMessage],
            trigger: ResponseTrigger,
        ) -> None:
            if not buffer:
                return
            last = buffer[-1]
            await _run_voice_response(
                channel_id=channel_id,
                guild_id=voice_guild_id,
                author=last.author,
                utterance=last.text,
                buffer=buffer,
                familiar=familiar,
                vc=vc,
                trigger=trigger,
            )

        voice_response_handlers = cast(
            "dict[int, Callable[[int, list[BufferedMessage], ResponseTrigger], Awaitable[None]]]",  # noqa: E501
            familiar.extras.setdefault("voice_response_handlers", {}),
        )
        voice_response_handlers[voice_channel_id] = _voice_response_handler

        # deliver debounced transcript to ConversationMonitor
        async def _deliver_to_monitor(
            user_id: int,
            merged: TranscriptionResult,
        ) -> None:
            author = _author_for(user_id)
            tracker = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0,
            )
            # Guard 1: a long-interruption regen was dispatched synchronously
            # before this coroutine runs — let the regen path handle the speech.
            # Guard 2: familiar is already generating/speaking (concurrent lull).
            # Guard 3: a short@SPEAKING yield+resume task is in flight — the
            # voice endpointing lull fires in the same event-loop window as
            # _finalize_burst, so the task runs before the resume can claim IDLE.
            if (
                familiar.extras.pop("_regen_pending", False)
                or tracker.state is not ResponseState.IDLE
                or tracker.short_yield_pending
            ):
                return
            # Voice lull dispatch: mark the tracker as GENERATING for the
            # duration of the side-model YES/NO eval so an interruption
            # that arrives while the familiar is "thinking about whether
            # to speak" is detectable. On YES, ``_run_voice_response``
            # also transitions to GENERATING (no-op) and continues into
            # SPEAKING/IDLE on its own. On NO, no on_respond fires, so
            # the tracker is still GENERATING when we get back here and
            # we transition it back to IDLE.
            # is_unsolicited stays False here: lulls are not unsolicited.
            # If monitor decides to respond (on_respond → _run_voice_response),
            # trigger.is_unsolicited drives the tracker flag (bot.py:167).
            tracker.transition(ResponseState.GENERATING)
            try:
                await familiar.monitor.on_message(
                    channel_id=voice_channel_id,
                    author=author,
                    text=merged.text,
                    is_mention=False,
                    # already debounced by VoiceLullMonitor; skip lull timer
                    is_lull_endpoint=True,
                )
            finally:
                # revert to IDLE if monitor said NO (on YES,
                # _run_voice_response already cycled to IDLE)
                if tracker.state is ResponseState.GENERATING:
                    tracker.transition(ResponseState.IDLE)

        # Step 11 dispatch callbacks. On short+yield: re-synth the remaining
        # words (captured at stop-time) and resume playback. On push-through:
        # write the interrupter's transcript to history so it's not lost.

        async def _on_short_yield_resume(remaining: list[WordTimestamp]) -> None:
            t = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0
            )
            # Always clear — we're now past the window where the lull needs
            # to be suppressed (regardless of whether we proceed to play).
            t.short_yield_pending = False
            # _run_voice_response transitions SPEAKING→IDLE after draining.
            # The resume task is scheduled via create_task before that
            # transition, so we may arrive here while state is still SPEAKING.
            # Poll briefly to let _run_voice_response settle; bail if a new
            # response starts (GENERATING) or we time out.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 2.0
            while t.state is ResponseState.SPEAKING:
                if loop.time() >= deadline:
                    _logger.warning(
                        "voice resume: timed out waiting for IDLE; skipping"
                    )
                    return
                await asyncio.sleep(0.05)
            # If a new response started before the lull confirmed, skip.
            if t.state is not ResponseState.IDLE or familiar.tts_client is None:
                return
            remaining_text = " ".join(ts.word for ts in remaining)
            if not remaining_text.strip():
                return
            try:
                tts_result = await familiar.tts_client.synthesize(remaining_text)
                stereo = mono_to_stereo(tts_result.audio)
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                t.transition(ResponseState.SPEAKING)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
            except Exception:
                _logger.exception("Voice resume TTS failed")
            t.transition(ResponseState.IDLE)

        def _on_push_through_transcript(user_id: int, transcript: str) -> None:
            if not transcript.strip():
                return
            author = _author_for(user_id)
            ch_cfg = familiar.channel_configs.get(channel_id=voice_channel_id)
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=voice_guild_id,
                role="user",
                content=transcript,
                author=author,
                mode=ch_cfg.mode,
            )

        # Per-guild interruption detector. Consumes Discord voice-activity
        # events from the lull monitor (no separate Deepgram VAD path)
        # and classifies bursts as discarded/short/long relative to the
        # current ResponseTracker state.
        #
        # Step 8: ``_on_long_during_generating`` fires when a burst
        # finalizes as ``long`` while the tracker is ``GENERATING``.
        # It schedules ``dispatch_interruption_regen`` as an asyncio
        # task so the detector callback stays synchronous.
        # Pending regen tasks — held so they are not garbage-collected before
        # they complete (asyncio only keeps a weak reference to tasks).
        regen_tasks: set[asyncio.Task[None]] = set()

        def _on_long_during_generating(starter_id: int, transcript: str) -> None:
            author = _author_for(starter_id)
            # Set before creating the task so _deliver_to_monitor (which
            # fires in the same event-loop tick via _fire_lull) sees it.
            familiar.extras["_regen_pending"] = True
            t = asyncio.create_task(
                dispatch_interruption_regen(
                    channel_id=voice_channel_id,
                    guild_id=voice_guild_id,
                    author=author,
                    transcript=transcript,
                    familiar=familiar,
                    vc=vc,
                ),
                name="long-interruption-regen",
            )
            regen_tasks.add(t)
            t.add_done_callback(regen_tasks.discard)

        def _on_long_boundary_crossed(starter_id: int, transcript: str) -> None:  # noqa: ARG001
            # Cancel the in-flight LLM task immediately — don't wait for the lull.
            # Regen is still scheduled at lull time via on_long_during_generating.
            tracker = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0,
            )
            task = tracker.generation_task
            if task is not None and not task.done():
                task.cancel()

        interruption_detector = InterruptionDetector(
            tracker_registry=familiar.tracker_registry,
            guild_id=voice_guild_id if voice_guild_id is not None else 0,
            min_interruption_s=familiar.config.min_interruption_s,
            short_long_boundary_s=familiar.config.short_long_boundary_s,
            lull_timeout_s=familiar.config.voice_lull_timeout,
            base_tolerance=(familiar.config.interrupt_tolerance.base_probability),
            on_long_during_generating=_on_long_during_generating,
            on_long_boundary_crossed=_on_long_boundary_crossed,
            on_short_yield_resume=_on_short_yield_resume,
            on_push_through_transcript=_on_push_through_transcript,
            author_resolver=_author_for,
        )
        familiar.extras["interruption_detector"] = interruption_detector

        # debounce per-final Deepgram fragments into a single utterance
        # via VoiceLullMonitor; fires after voice_lull_timeout of silence

        lull_monitor = VoiceLullMonitor(
            lull_timeout=familiar.config.voice_lull_timeout,
            on_utterance_complete=_deliver_to_monitor,
            on_voice_activity=interruption_detector.on_voice_activity,
            channel_name=channel.name,
        )  # voice_lull_timeout = endpointing; conversational lull governed
        # by text_lull_timeout inside ConversationMonitor
        familiar.extras["voice_lull_monitor"] = lull_monitor

        # Deepgram-driven VAD: speech-start / speech-end edges derived from
        # interim word arrivals on the Deepgram socket.
        deepgram_vad = DeepgramVoiceActivityDetector(
            on_speech_start=lull_monitor.on_speech_start,
            on_speech_end=lull_monitor.on_speech_end,
        )
        familiar.extras["deepgram_vad"] = deepgram_vad

        async def _route_transcript_to_monitor(  # noqa: RUF029
            user_id: int,
            result: TranscriptionResult,
        ) -> None:
            # Feed DGVAD first so speech_start fires before transcript fan-out.
            deepgram_vad.feed_transcript(user_id, result)
            lull_monitor.on_transcript(user_id, result)
            # Forward finals to the interruption detector so it can
            # accumulate the burst transcript for dispatch.
            if result.is_final:
                interruption_detector.on_transcript(user_id, result.text)

        pipeline = await start_pipeline(
            familiar.transcriber,
            user_names=user_names,
            channel_name=channel.name,
            resolve_name=_resolve_from_channel,
            response_handler=_route_transcript_to_monitor,
        )
        sink = RecordingSink(
            loop=asyncio.get_running_loop(),
            audio_queue=pipeline.tagged_audio_queue,
        )
        vc.start_recording(sink, _recording_finished_callback)

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

    # disconnect + pipeline teardown + memory flush can exceed Discord's 3s
    # interaction window, so defer before doing any of it
    await ctx.defer(ephemeral=True)

    if vc is not None:
        # stop transcription first so audio stops flowing before disconnect
        if get_pipeline() is not None:
            if hasattr(vc, "recording") and vc.recording:
                vc.stop_recording()
            await stop_pipeline()
            # stt_stop logged inside stop_pipeline()
        lull_monitor = familiar.extras.pop("voice_lull_monitor", None)
        if isinstance(lull_monitor, VoiceLullMonitor):
            lull_monitor.clear()
        familiar.extras.pop("interruption_detector", None)
        dgvad = familiar.extras.pop("deepgram_vad", None)
        if isinstance(dgvad, DeepgramVoiceActivityDetector):
            dgvad.reset()
        await vc.disconnect()

    if sub is not None:
        # drop voice dispatch + monitor state for clean re-subscribe
        voice_response_handlers = cast(
            "dict[int, Callable[[int, list[BufferedMessage]], Awaitable[None]]]",
            familiar.extras.get("voice_response_handlers", {}),
        )
        voice_response_handlers.pop(sub.channel_id, None)
        familiar.monitor.clear_channel(sub.channel_id)
        familiar.subscriptions.remove(
            channel_id=sub.channel_id,
            kind=SubscriptionKind.voice,
        )

    await familiar.memory_writer_scheduler.flush()
    await ctx.followup.send("Left voice.", ephemeral=True)


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
    _logger.info(
        f"{ls.tag('Config', ls.W)} "
        f"{ls.word(familiar.monitor.format_channel_context(channel_id), ls.C)} "
        f"{ls.kv('mode', mode.value)}"
    )
    await ctx.respond(f"Channel mode set to **{mode.value}**.", ephemeral=True)


_PLACEHOLDER_MAX = 100  # Discord hard limit on InputText placeholder length


def _default_backdrop_placeholder(default: str | None) -> str:
    """Build a ≤100-char single-line preview of *default* for the modal."""
    fallback = "Author-note injected on every turn. Replaces the mode default."
    if not default:
        return fallback
    single_line = " ".join(default.split())
    if len(single_line) <= _PLACEHOLDER_MAX:
        return single_line
    return single_line[: _PLACEHOLDER_MAX - 1].rstrip() + "\u2026"


def _make_backdrop_modal(
    familiar: Familiar,
    channel_id: int,
    channel_name: str,
    current: str | None,
    mode_default: str | None,
) -> discord.ui.Modal:
    """Build and return the backdrop-editing modal for *channel_id*."""
    placeholder = _default_backdrop_placeholder(mode_default)

    class ChannelBackdropModal(discord.ui.Modal):
        def __init__(self) -> None:
            super().__init__(title="Channel Backdrop")
            field = discord.ui.InputText(
                label="Backdrop",
                style=discord.InputTextStyle.long,
                placeholder=placeholder,
                value=current or "",
                required=False,
                max_length=4000,
            )
            # py-cord 2.7.1 bug: _generate_underlying collapses
            # ``required=False`` to ``None`` via ``False or self.required``;
            # Discord treats ``null`` as required. Post-set to work around.
            # Drop this line once py-cord ships the fix.
            field.required = False
            self.add_item(field)

        async def callback(self, interaction: discord.Interaction) -> None:
            text: str = self.children[0].value or ""  # type: ignore[union-attr]
            familiar.channel_configs.set_backdrop(
                channel_id=channel_id,
                backdrop=text,
                channel_name=channel_name,
            )
            label = familiar.monitor.format_channel_context(channel_id)
            _logger.info(
                f"{ls.tag('Config', ls.W)} "
                f"{ls.word(label, ls.C)} "
                f"{ls.kv('action', 'backdrop_set')}"
            )
            stripped = text.strip()
            if stripped:
                msg = "Channel backdrop saved."
            else:
                msg = (
                    "Channel backdrop cleared"
                    " — reverting to the mode default on next turn."
                )
            await interaction.response.send_message(msg, ephemeral=True)

    return ChannelBackdropModal()


async def channel_backdrop(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Handle ``/channel-backdrop``. Submit blank in the modal to clear."""
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return

    channel = ctx.channel
    if channel is not None:
        _refresh_channel_context(familiar, channel)
    channel_name = familiar.monitor.format_channel_context(channel_id)
    current = familiar.channel_configs.get_backdrop(channel_id=channel_id)
    channel_cfg = familiar.channel_configs.get(channel_id=channel_id)
    mode_default = resolve_mode_default(
        modes_root=familiar.root / "modes",
        mode=channel_cfg.mode,
        defaults_modes_root=familiar.root.parent / "_default" / "modes",
    )
    modal = _make_backdrop_modal(
        familiar, channel_id, channel_name, current, mode_default
    )
    await ctx.send_modal(modal)


async def context_command(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Handle ``/context``. Posts last-used LLM context as ``context.md``."""
    channel_id = ctx.channel_id
    if channel_id is None:
        await ctx.respond("Cannot determine channel.", ephemeral=True)
        return
    channel = ctx.channel
    if channel is not None:
        _refresh_channel_context(familiar, channel)
    label = familiar.monitor.format_channel_context(channel_id)
    raw = familiar.last_context_cache.get(channel_id=channel_id)
    if raw is None:
        _logger.info(
            f"{ls.tag('Context', ls.W)} "
            f"{ls.word(label, ls.C)} "
            f"{ls.kv('action', 'context_miss')}"
        )
        await ctx.respond("No context cached for this channel yet.", ephemeral=True)
        return
    _logger.info(
        f"{ls.tag('Context', ls.W)} "
        f"{ls.word(label, ls.C)} "
        f"{ls.kv('action', 'context_sent')} "
        f"{ls.kv('bytes', str(len(raw)), vc=ls.LG)}"
    )
    await ctx.respond(file=discord.File(io.BytesIO(raw), filename="context.md"))


# ---------------------------------------------------------------------------
# Pipeline response path
# ---------------------------------------------------------------------------


async def _run_text_response(
    channel_id: int,
    guild_id: int | None,
    author: Author,
    utterance: str,
    buffer: list[BufferedMessage],
    familiar: Familiar,
    channel: discord.TextChannel | discord.Thread,
) -> None:
    """Full pipeline → LLM → reply path for a text channel or thread.

    Persists all buffered user messages to history (in order), then
    the assistant reply.

    :param buffer: messages accumulated since last response, including
        trigger; persisted after LLM call.
    """
    # closed-thread guard: archived+locked == Discord's "thread closed"
    # UX. also covers the offline case (thread closed while bot was
    # down) because we check on every send, not on lifecycle events.
    if isinstance(channel, discord.Thread) and channel.archived and channel.locked:
        _logger.error(
            f"{ls.tag('❌ Send skipped', ls.R)} "
            f"{ls.kv('reason', 'thread_closed')} "
            f"{ls.kv('channel_id', str(channel_id))}"
        )
        return

    channel_config = familiar.channel_configs.get(channel_id=channel_id)

    builder = TraceBuilder(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        modality="text",
    )
    builder.tag("channel_mode", channel_config.mode.value)
    builder.tag("speaker", author.label)

    request = ContextRequest(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        author=author,
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=tuple(PendingTurn(author=m.author, text=m.text) for m in buffer),
    )

    # typing indicator spans pipeline assembly + LLM + post-proc so
    # it shows while context is built, not just during the LLM call
    async with channel.typing():
        pipeline = familiar.build_pipeline(channel_config)
        async with builder.span("pipeline_assembly") as pa_meta:
            pipeline_output = await pipeline.assemble(
                request,
                budget_by_layer=channel_config.budget_by_layer,
            )
            builder.add_provider_outcomes(pa_meta, pipeline_output.outcomes)

        async with builder.span("render") as rmeta:
            messages = assemble_chat_messages(
                pipeline_output,
                store=familiar.history_store,
                history_window_size=familiar.config.history_window_size,
                depth_inject_position=familiar.config.depth_inject_position,
                depth_inject_role=familiar.config.depth_inject_role,
                mode=channel_config.mode,
                display_tz=familiar.config.display_tz,
            )
            rmeta["message_count"] = len(messages)
        familiar.last_context_cache.put(
            channel_id=channel_id, messages=messages, modality="text"
        )

        n_new = len(request.pending_turns) if request.pending_turns else 1
        batch = (
            "\n".join(
                f"  [{pt.author.label if pt.author else '?'}]: "
                f"{pt.text[:200]}{'…' if len(pt.text) > 200 else ''}"
                for pt in request.pending_turns
            )
            if request.pending_turns
            else (
                f"  [{request.author.label if request.author else '?'}]: "
                f"{request.utterance[:200]}"
                f"{'…' if len(request.utterance) > 200 else ''}"
            )
        )
        last_hist_text = messages[-(n_new + 1)].content if len(messages) > n_new else ""
        hist = (
            f"  {ls.word(ls.trunc(last_hist_text), ls.LW)}\n" if last_hist_text else ""
        )
        ch_label = familiar.monitor.format_channel_context(channel_id)
        _logger.info(
            f"{ls.tag('🧠 Generating Text', ls.G)} "
            f"{ls.word(ch_label, ls.C)} "
            f"{ls.kv('messages', str(len(messages)), vc=ls.LG)} "
            f"{ls.kv('new', str(n_new), vc=ls.LG)}\n"
            f"{hist}"
            f"{ls.word(batch, ls.LW)}"
        )

        # main reply isolation: catch ``LLMClient.chat`` raise set;
        # on failure return silently (no history write, no TTS).
        # error captured inside span so llm_call stage lands in the trace.
        llm_error: Exception | None = None
        reply = None
        async with builder.span("llm_call") as llm_meta:
            llm_meta["model"] = familiar.llm_clients["main_prose"].model
            try:
                reply = await familiar.llm_clients["main_prose"].chat(messages)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                llm_meta["error"] = f"{type(exc).__name__}: {exc}"
                llm_error = exc
            else:
                llm_meta["reply_length"] = len(reply.content)
        if llm_error is not None:
            _logger.warning(
                "main reply (text): %s: %s",
                type(llm_error).__name__,
                llm_error,
            )
            familiar.metrics_collector.record(builder.finalize())
            return
        if reply is None:
            # unreachable; type-guard
            return

        async with builder.span("post_processing"):
            reply_text = await pipeline.run_post_processors(reply.content, request)

        _logger.info(
            f"{ls.tag('💬 Generated Text', ls.G)} "
            f"{ls.word(ch_label, ls.C)} "
            f"{ls.kv('chars', str(len(reply_text)), vc=ls.LG)}\n"
            f"  {ls.word(ls.trunc(reply_text, 500), ls.LG)}"
        )

    ts_cfg = channel_config.typing_simulation

    if ts_cfg.enabled:
        # chunked typing-simulation path: user turns persist BEFORE
        # delivery so a mid-flight cancel leaves history in order
        # (user msgs → partial assistant → new user msg → fresh reply).
        async with builder.span("history_write_user") as hw_meta:
            for msg in buffer:
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="user",
                    content=msg.text,
                    author=msg.author,
                    mode=channel_config.mode,
                )
            hw_meta["turns_written"] = len(buffer)

        chunks = split_reply_into_chunks(
            reply_text,
            sentence_split_threshold=ts_cfg.sentence_split_threshold,
        )
        tracker = familiar.text_delivery_registry.get(channel_id)

        async def _deliver() -> None:
            for i, chunk in enumerate(chunks):
                delay = compute_typing_delay(chunk, ts_cfg)
                async with channel.typing():
                    await asyncio.sleep(delay)
                await channel.send(chunk)
                tracker.mark_sent(chunk)
                if i < len(chunks) - 1:
                    await asyncio.sleep(ts_cfg.inter_line_pause_s)

        was_cancelled = False
        async with builder.span("chunked_delivery") as cd_meta:
            cd_meta["chunk_count"] = len(chunks)
            task = asyncio.create_task(_deliver())
            tracker.start(task)
            try:
                await task
            except asyncio.CancelledError:
                was_cancelled = True
                cd_meta["cancelled"] = True
            cd_meta["chunks_sent"] = len(tracker.sent_chunks)

        sent_chunks = list(tracker.sent_chunks)
        tracker.clear()

        async with builder.span("history_write_assistant") as hw_meta:
            if sent_chunks:
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="assistant",
                    content="\n\n".join(sent_chunks),
                    mode=channel_config.mode,
                )
                hw_meta["turns_written"] = 1
            else:
                hw_meta["turns_written"] = 0
        await familiar.memory_writer_scheduler.notify_turn()

        if was_cancelled:
            # cancelled delivery → skip TTS fan-out; caller's new
            # user message will drive a fresh reply through the monitor
            familiar.metrics_collector.record(builder.finalize())
            return
    else:
        # legacy path: user turns + full assistant reply in one history
        # write, then single send — preserves pre-typing-sim behaviour
        async with builder.span("history_write") as hw_meta:
            turns_written = 0
            for msg in buffer:
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="user",
                    content=msg.text,
                    author=msg.author,
                    mode=channel_config.mode,
                )
                turns_written += 1
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=channel_id,
                guild_id=guild_id,
                role="assistant",
                content=reply_text,
                mode=channel_config.mode,
            )
            turns_written += 1
            hw_meta["turns_written"] = turns_written
        await familiar.memory_writer_scheduler.notify_turn()

        async with builder.span("discord_send"):
            await channel.send(reply_text)

    # TTS fan-out: speak reply in voice if a voice sub exists in guild
    guild = getattr(channel, "guild", None)
    if (
        familiar.tts_client is not None
        and guild is not None
        and familiar.subscriptions.voice_in_guild(guild.id) is not None
    ):
        vc = guild.voice_client
        if vc is not None and not vc.is_playing():
            async with builder.span("tts") as tts_meta:
                try:
                    tts_result = await familiar.tts_client.synthesize(reply_text)
                    stereo = mono_to_stereo(tts_result.audio)
                    tts_meta["audio_bytes"] = len(stereo)
                    vc.play(discord.PCMAudio(io.BytesIO(stereo)))
                except Exception as exc:
                    tts_meta["error"] = f"{type(exc).__name__}: {exc}"
                    _logger.exception("TTS synthesis failed")

    familiar.metrics_collector.record(builder.finalize())


# ---------------------------------------------------------------------------
# Message loop
# ---------------------------------------------------------------------------


async def on_message(message: discord.Message, familiar: Familiar) -> None:
    """Route incoming Discord message to the conversation monitor."""
    if message.author.bot:
        return

    channel_id = message.channel.id
    text_sub = familiar.subscriptions.get(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
    )
    if text_sub is None:
        return

    # refresh channel context each message so renames propagate and
    # subscriptions loaded from disk at startup pick up a label.
    msg_channel = message.channel
    if isinstance(msg_channel, discord.TextChannel | discord.Thread):
        _refresh_channel_context(familiar, msg_channel)

    bot_user = familiar.extras.get("bot_user")
    is_mention = (
        message.guild is not None
        and bot_user is not None
        and bot_user in message.mentions
    )

    # mid-delivery cancellation: if a chunked typing-sim delivery is
    # in flight for this channel, cancel it and wait for its partial
    # history write to settle before the new message enters the monitor.
    tracker = familiar.text_delivery_registry.get(channel_id)
    if tracker.is_active():
        await tracker.cancel_and_wait()

    author = Author.from_discord_member(message.author)
    await familiar.monitor.on_message(
        channel_id=channel_id,
        author=author,
        text=message.content,
        is_mention=is_mention,
    )


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def create_bot(familiar: Familiar) -> discord.Bot:
    """Create Discord bot bound to *familiar*.

    Registers slash commands and wires ``on_message`` + ``on_respond``
    callback on :class:`ConversationMonitor`.
    """
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.message_content = True
    intents.messages = True
    bot = discord.Bot(intents=intents)

    # stash bot.user for @mention detection in on_message
    @bot.event
    async def on_ready() -> None:  # noqa: RUF029
        familiar.extras["bot_user"] = bot.user
        familiar.memory_writer_scheduler.start()

    # on_respond callback: drives full pipeline path
    async def _on_respond(
        channel_id: int,
        buffer: list[BufferedMessage],
        trigger: ResponseTrigger,
    ) -> None:
        # voice dispatch first — shared ConversationMonitor gate
        voice_handlers = cast(
            "dict[int, Callable[[int, list[BufferedMessage], ResponseTrigger], Awaitable[None]]]",  # noqa: E501
            familiar.extras.get("voice_response_handlers", {}),
        )
        voice_handler = voice_handlers.get(channel_id)
        if voice_handler is not None:
            await voice_handler(channel_id, buffer, trigger)
            return

        channel = bot.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel | discord.Thread):
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
            author=last.author,
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

    @bot.slash_command(
        name="channel-backdrop",
        description=(
            "Set a per-channel author-note (replaces the mode default)."
            " Submit blank to clear."
        ),
    )
    async def _channel_backdrop_cmd(
        ctx: discord.ApplicationContext,
    ) -> None:
        await channel_backdrop(ctx, familiar)

    @bot.slash_command(
        name="context",
        description="Post a context.md dump of the last LLM context for this channel.",
    )
    async def _context_cmd(ctx: discord.ApplicationContext) -> None:
        await context_command(ctx, familiar)

    # --- message loop ---
    async def _on_message(message: discord.Message) -> None:
        await on_message(message, familiar)

    bot.add_listener(_on_message, name="on_message")

    return bot
