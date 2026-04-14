"""Discord bot factory, slash commands, and pipeline-routed message loop.

- subscription + channel-mode slash command surface
- ``on_message`` routes subscribed text through :class:`ContextPipeline`
- voice transcription pipeline shares the same context path as text
- single :class:`Familiar` bundle per process lifetime
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
from typing import TYPE_CHECKING, cast

import discord
import httpx

from familiar_connect.chattiness import BufferedMessage
from familiar_connect.config import ChannelConfig, ChannelMode
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import ContextRequest, Modality, PendingTurn
from familiar_connect.llm import sanitize_name
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice.interruption import (
    InterruptionDetector,
    ResponseState,
    ResponseTracker,
)
from familiar_connect.voice_lull import VoiceLullMonitor
from familiar_connect.voice_pipeline import get_pipeline, start_pipeline, stop_pipeline

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.chattiness import ResponseTrigger
    from familiar_connect.context.pipeline import ProviderOutcome
    from familiar_connect.familiar import Familiar
    from familiar_connect.transcription import TranscriptionResult

_logger = logging.getLogger(__name__)


async def _recording_finished_callback(  # noqa: RUF029
    sink: discord.sinks.Sink,
    *args: object,
) -> None:
    """no-op coroutine required by py-cord's ``start_recording`` API.

    py-cord awaits this internally; cleanup lives in :func:`unsubscribe_voice`.
    """
    del sink, args


def _log_pipeline_outcomes(
    channel_id: int,
    outcomes: list[ProviderOutcome],
) -> None:
    """Emit structured log entry per provider outcome."""
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


async def _run_voice_response(
    channel_id: int,
    guild_id: int | None,
    speaker: str,
    utterance: str,
    buffer: list[BufferedMessage],
    familiar: Familiar,
    vc: discord.VoiceClient,
    trigger: ResponseTrigger,
) -> None:
    """Run full pipeline → LLM → TTS path for voice channel.

    Thin shim: builds the initial :class:`ContextRequest` and delegates to
    :func:`_run_voice_response_with_request`. The regen-on-interruption
    path (:func:`dispatch_interruption_regen`) shares the same helper.
    """
    tracker = familiar.tracker_registry.get(guild_id if guild_id is not None else 0)
    tracker.vc = vc
    tracker.is_unsolicited = trigger.is_unsolicited
    # cache mood modifier for whole turn (no mid-response re-roll)
    tracker.mood_modifier = familiar.mood_evaluator.evaluate()

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
        speaker=speaker,
        utterance=utterance,
        modality=Modality.voice,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=tuple(
            PendingTurn(speaker=m.speaker, text=m.text) for m in buffer
        ),
    )
    await _run_voice_response_with_request(
        request=request,
        channel_config=channel_config,
        buffer=buffer,
        familiar=familiar,
        vc=vc,
        tracker=tracker,
    )


async def _run_voice_response_with_request(
    *,
    request: ContextRequest,
    channel_config: ChannelConfig,
    buffer: list[BufferedMessage],
    familiar: Familiar,
    vc: discord.VoiceClient,
    tracker: ResponseTracker,
) -> None:
    """Core pipeline → LLM → TTS → play path; shared by initial + regen.

    Caller supplies a fully-formed :class:`ContextRequest`. Step-8 gate +
    commit flag consumed here so the regen path shares this body without
    duplicating orchestration.
    """
    # stash original user buffer so the dispatcher can replay on cancel
    tracker.pending_buffer = list(buffer)
    tracker.transition(ResponseState.GENERATING)

    channel_id = request.channel_id
    guild_id = request.guild_id

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
        "LLM request channel=%s (voice) messages=%d, %d new:\n%s",
        channel_id,
        len(messages),
        len(request.pending_turns) if request.pending_turns else 1,
        "\n".join(
            f"  [{pt.speaker or '?'}]: "
            f"{pt.text[:200]}{'…' if len(pt.text) > 200 else ''}"
            for pt in request.pending_turns
        )
        if request.pending_turns
        else (
            f"  [{request.speaker or '?'}]: "
            f"{request.utterance[:200]}{'…' if len(request.utterance) > 200 else ''}"
        ),
    )

    # cancellable generation task; parked on tracker for interruption path
    generation_task = asyncio.create_task(
        familiar.llm_clients["main_prose"].chat(messages),
    )
    tracker.generation_task = generation_task
    try:
        reply = await generation_task
    except asyncio.CancelledError:
        tracker.generation_task = None
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            # outer task cancelled from above — propagate
            tracker.transition(ResponseState.IDLE)
            raise
        # step 8: long-interruption path — dispatcher owns regen + state.
        # don't write history (buffer forwards to regen); don't reset
        # scratch (dispatcher yields via sleep(0) then transitions IDLE).
        if tracker.cancel_committed:
            _logger.info(
                "voice generation cancelled by long-interruption channel=%s",
                channel_id,
            )
            return
        # unexpected cancellation outside the interruption dispatch path
        _logger.info(
            "voice generation cancelled channel=%s",
            channel_id,
        )
        tracker.transition(ResponseState.IDLE)
        return
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        tracker.generation_task = None
        _logger.warning(
            "main reply (voice): %s: %s",
            type(exc).__name__,
            exc,
        )
        tracker.transition(ResponseState.IDLE)
        return
    tracker.generation_task = None

    # step 8: pause until burst finalizes (short) or commits long.
    # gate is only cleared when min-crossed during GENERATING; no-op fast
    # path in the common case where no interruption registered.
    await tracker.delivery_gate.wait()
    if tracker.cancel_committed:
        # long boundary crossed while we were generating or post-gate;
        # dispatcher owns regen. Skip history + TTS + play.
        _logger.info(
            "voice reply aborted by long-interruption channel=%s",
            channel_id,
        )
        return

    reply_text = await pipeline.run_post_processors(reply.content, request)
    tracker.response_text = reply_text

    # persist buffered user utterances then assistant reply
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
        mode=channel_config.mode,  # type: ignore[attr-defined]
    )

    _logger.info("[Voice Response] %s", reply_text)

    if familiar.tts_client is not None:
        try:
            tts_result = await familiar.tts_client.synthesize(reply_text)
            tracker.timestamps = list(tts_result.timestamps)
            stereo = mono_to_stereo(tts_result.audio)
            # wait for current audio to finish (poll — no event available)
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            tracker.transition(ResponseState.SPEAKING)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            # poll playback to completion for IDLE transition
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
        except Exception:
            _logger.exception("Voice response TTS failed")
    tracker.transition(ResponseState.IDLE)


async def dispatch_interruption_regen(
    *,
    channel_id: int,
    guild_id: int | None,
    speaker_name: str,
    transcript: str,
    original_buffer: list[BufferedMessage],
    familiar: Familiar,
    vc: discord.VoiceClient,
    tracker: ResponseTracker,
) -> None:
    """Re-run voice pipeline after a long interruption commits.

    Preserves ``original_buffer`` as leading pending turns and appends
    the interrupter's transcript as a trailing user turn. Yields once to
    let the cancelled branch of the prior ``_run_voice_response`` settle
    before resetting tracker state.
    """
    # let the cancelled branch of the original call settle first so its
    # IDLE transition (if any) doesn't stomp on the fresh GENERATING
    await asyncio.sleep(0)

    # reset scratch (gate, cancel flag, pending buffer, etc.)
    tracker.transition(ResponseState.IDLE)

    channel_config = familiar.channel_configs.get(channel_id=channel_id)
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

    # build combined buffer: original user turns + interrupter's transcript
    combined_buffer: list[BufferedMessage] = list(original_buffer)
    interrupter_msg = BufferedMessage(
        speaker=speaker_name,
        text=transcript,
        timestamp=0.0,
    )
    combined_buffer.append(interrupter_msg)

    pending_turns = tuple(
        PendingTurn(speaker=m.speaker, text=m.text) for m in combined_buffer
    )

    request = ContextRequest(
        familiar_id=familiar.id,
        channel_id=channel_id,
        guild_id=guild_id,
        speaker=speaker_name,
        utterance=transcript,
        modality=Modality.voice,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=pending_turns,
    )

    # mark unsolicited=False: this is a direct reply to an interrupter,
    # not a chattiness interjection. mood eval re-runs for new turn.
    tracker.is_unsolicited = False
    tracker.mood_modifier = familiar.mood_evaluator.evaluate()
    tracker.vc = vc

    await _run_voice_response_with_request(
        request=request,
        channel_config=channel_config,
        buffer=combined_buffer,
        familiar=familiar,
        vc=vc,
        tracker=tracker,
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
    _logger.info("Joined voice channel: %s", channel.name)

    familiar.subscriptions.add(
        channel_id=channel.id,
        kind=SubscriptionKind.voice,
        guild_id=ctx.guild_id,
    )

    if familiar.tts_client is not None:
        try:
            tts_result = await familiar.tts_client.synthesize("Hello!")
            stereo = mono_to_stereo(tts_result.audio)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
        except Exception:
            _logger.exception("Opening greeting TTS failed")

    if familiar.transcriber is not None:
        user_names = {m.id: m.display_name for m in channel.members}
        voice_channel_id = channel.id
        voice_guild_id = ctx.guild_id

        def _resolve_from_channel(user_id: int) -> str | None:
            for member in channel.members:
                if member.id == user_id:
                    return member.display_name
            return None

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
                speaker=last.speaker,
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
            raw_name = user_names.get(user_id, f"User-{user_id}")
            safe_name = sanitize_name(raw_name) or raw_name
            # mark GENERATING during side-model eval; reverted to IDLE
            # in finally block if monitor says NO
            tracker = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0,
            )
            tracker.is_unsolicited = True  # lull is always unsolicited
            tracker.transition(ResponseState.GENERATING)
            try:
                await familiar.monitor.on_message(
                    channel_id=voice_channel_id,
                    speaker=safe_name,
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

        # per-guild interruption detector (step 8: dispatches regen)
        interruption_detector = InterruptionDetector(
            tracker_registry=familiar.tracker_registry,
            guild_id=voice_guild_id if voice_guild_id is not None else 0,
            min_interruption_s=familiar.config.min_interruption_s,
            short_long_boundary_s=familiar.config.short_long_boundary_s,
            lull_timeout_s=familiar.config.voice_lull_timeout,
            base_tolerance=(familiar.config.interrupt_tolerance.base_probability),
        )

        # step 8: dispatcher fires regen when long boundary crossed
        # during GENERATING. resolves the interrupter's display name from
        # the user_names map or channel members.
        def _resolve_interrupter_name(user_id: int) -> str:
            raw = user_names.get(user_id)
            if raw is None:
                for member in channel.members:
                    if member.id == user_id:
                        raw = member.display_name
                        break
            if raw is None:
                raw = f"User-{user_id}"
            return sanitize_name(raw) or raw

        regen_tasks: set[asyncio.Task[None]] = set()

        def _dispatch_regen(
            *,
            speaker_user_id: int,
            transcript: str,
            original_buffer: list[BufferedMessage],
            tracker: ResponseTracker,
        ) -> None:
            speaker_name = _resolve_interrupter_name(speaker_user_id)
            task = asyncio.create_task(
                dispatch_interruption_regen(
                    channel_id=voice_channel_id,
                    guild_id=voice_guild_id,
                    speaker_name=speaker_name,
                    transcript=transcript,
                    original_buffer=original_buffer,
                    familiar=familiar,
                    vc=vc,
                    tracker=tracker,
                ),
            )
            regen_tasks.add(task)
            task.add_done_callback(regen_tasks.discard)

        familiar.extras["voice_interruption_regen_tasks"] = regen_tasks

        interruption_detector.set_dispatch(_dispatch_regen)
        familiar.extras["interruption_detector"] = interruption_detector

        # debounce per-final Deepgram fragments into a single utterance
        # via VoiceLullMonitor; fires after voice_lull_timeout of silence

        lull_monitor = VoiceLullMonitor(
            lull_timeout=familiar.config.voice_lull_timeout,
            user_silence_s=0.2,
            on_utterance_complete=_deliver_to_monitor,
            on_voice_activity=interruption_detector.on_voice_activity,
        )  # voice_lull_timeout = endpointing; conversational lull governed
        # by text_lull_timeout inside ConversationMonitor
        familiar.extras["voice_lull_monitor"] = lull_monitor

        async def _route_transcript_to_monitor(  # noqa: RUF029
            user_id: int,
            result: TranscriptionResult,
        ) -> None:
            lull_monitor.on_transcript(user_id, result)
            # step 8: feed finals to the detector's burst transcript
            # accumulator; detector ignores when no burst active.
            interruption_detector.on_transcript(user_id, result.text)

        pipeline = await start_pipeline(
            familiar.transcriber,
            user_names=user_names,
            resolve_name=_resolve_from_channel,
            response_handler=_route_transcript_to_monitor,
            on_audio=lull_monitor.on_audio,
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
        # stop transcription first so audio stops flowing before disconnect
        if get_pipeline() is not None:
            if hasattr(vc, "recording") and vc.recording:
                vc.stop_recording()
            await stop_pipeline()
            _logger.info("Stopped voice transcription pipeline")
        lull_monitor = familiar.extras.pop("voice_lull_monitor", None)
        if isinstance(lull_monitor, VoiceLullMonitor):
            lull_monitor.clear()
        familiar.extras.pop("interruption_detector", None)
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
    """Full pipeline → LLM → reply path for a text channel.

    Persists all buffered user messages to history (in order), then
    the assistant reply.

    :param buffer: messages accumulated since last response, including
        trigger; persisted after LLM call.
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

    n_new = len(request.pending_turns) if request.pending_turns else 1
    batch = (
        "\n".join(
            f"  [{pt.speaker or '?'}]: "
            f"{pt.text[:200]}{'…' if len(pt.text) > 200 else ''}"
            for pt in request.pending_turns
        )
        if request.pending_turns
        else (
            f"  [{request.speaker or '?'}]: "
            f"{request.utterance[:200]}{'…' if len(request.utterance) > 200 else ''}"
        )
    )
    _logger.info(
        "LLM request channel=%s messages=%d, %d new:\n%s",
        channel_id,
        len(messages),
        n_new,
        batch,
    )

    # main reply isolation: catch ``LLMClient.chat`` raise set;
    # on failure return silently (no history write, no TTS)
    async with channel.typing():
        try:
            reply = await familiar.llm_clients["main_prose"].chat(messages)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            _logger.warning(
                "main reply (text): %s: %s",
                type(exc).__name__,
                exc,
            )
            return

    reply_text = await pipeline.run_post_processors(reply.content, request)

    # persist buffered user turns then assistant reply (after LLM
    # call so a crash never leaves an orphaned user turn)
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

    # TTS fan-out: speak reply in voice if a voice sub exists in guild
    guild = getattr(channel, "guild", None)
    if (
        familiar.tts_client is not None
        and guild is not None
        and familiar.subscriptions.voice_in_guild(guild.id) is not None
    ):
        vc = guild.voice_client
        if vc is not None and not vc.is_playing():
            try:
                tts_result = await familiar.tts_client.synthesize(reply_text)
                stereo = mono_to_stereo(tts_result.audio)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            except Exception:
                _logger.exception("TTS synthesis failed")


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
