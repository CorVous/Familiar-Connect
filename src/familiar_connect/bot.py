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
import dataclasses
import io
import logging
from typing import TYPE_CHECKING, cast

import discord
import httpx

from familiar_connect.config import ChannelMode
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import ContextRequest, Modality, PendingTurn
from familiar_connect.llm import sanitize_name
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice.interruption import InterruptionDetector, ResponseState
from familiar_connect.voice_lull import VoiceLullMonitor
from familiar_connect.voice_pipeline import get_pipeline, start_pipeline, stop_pipeline

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.chattiness import BufferedMessage, ResponseTrigger
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
    """Execute the full pipeline → LLM → TTS path for a voice channel.

    Called by the ``on_respond`` callback built in :func:`create_bot`
    when the :class:`ConversationMonitor` decides the familiar should
    speak on a voice channel. Persists all buffered user utterances
    to history (in order) and then the assistant reply, so the
    history store has a complete record of the conversation even
    though some utterances were buffered before the pipeline ran.

    Mirrors :func:`_run_text_response` so text and voice share both
    the decision gate (``ConversationMonitor``) and the history
    shape. The voice-specific difference is :attr:`Modality.voice`
    on the :class:`ContextRequest` plus the latency-focused trimming
    of providers and processors (see ``dataclasses.replace`` below).
    """
    # Resolve the per-guild response tracker and mark this response with
    # the trigger's unsolicited flag. The tracker drives the voice
    # interruption state machine; for Step 3 it is observational only —
    # we log the IDLE→GENERATING→SPEAKING→IDLE lifecycle so operators
    # can see that solicited vs. unsolicited replies are tagged right
    # before any interruption logic is wired up.
    tracker = familiar.tracker_registry.get(guild_id if guild_id is not None else 0)
    tracker.vc = vc
    tracker.is_unsolicited = trigger.is_unsolicited
    # Cache the mood modifier for the whole turn — should_keep_talking
    # at Moment 1 must use the same value the tracker saw at generation
    # start, not re-roll mid-response.
    tracker.mood_modifier = familiar.mood_evaluator.evaluate()
    tracker.transition(ResponseState.GENERATING)

    channel_config = familiar.channel_configs.get(channel_id=channel_id)

    # VOICE: Pre- and post-processors are disabled for voice turns to reduce
    # real-time latency. stepped_thinking (reasoning_context LLM) adds
    # chain-of-thought overhead; recast (post_process_style LLM) rewrites
    # text destined for TTS rather than screen reading.
    # Providers that make their own LLM calls (content_search → memory_search,
    # history → history_summary) are also stripped for the same reason.
    # To re-enable: remove this dataclasses.replace() call and its comment.
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

    # Main reply isolation: catch the closed raise set of
    # ``LLMClient.chat`` — httpx transport/status errors, plus the
    # ``ValueError`` / ``KeyError`` branches in ``llm.chat`` for
    # malformed payloads. Log and return cleanly so the transcriber
    # callback stays alive for the next utterance. No TTS, no
    # history write, no post-processing on failure.
    try:
        reply = await familiar.llm_clients["main_prose"].chat(messages)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        _logger.warning(
            "main reply (voice): %s: %s",
            type(exc).__name__,
            exc,
        )
        tracker.transition(ResponseState.IDLE)
        return
    reply_text = await pipeline.run_post_processors(reply.content, request)
    tracker.response_text = reply_text

    # Persist all buffered user utterances, then the assistant reply.
    # Done after the LLM call so a mid-request crash doesn't leave an
    # orphaned user turn with no reply. Mirrors _run_text_response.
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

    _logger.info("[Voice Response] %s", reply_text)

    if familiar.tts_client is not None:
        try:
            tts_result = await familiar.tts_client.synthesize(reply_text)
            tracker.timestamps = list(tts_result.timestamps)
            stereo = mono_to_stereo(tts_result.audio)
            # Wait for any currently-playing audio to finish.
            # vc.is_playing() is a third-party poll; no event available.
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            tracker.transition(ResponseState.SPEAKING)
            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            # Poll playback to completion so the IDLE transition is
            # observable in the log and the tracker state reflects
            # reality for the next turn. No interruption dispatch yet
            # (Step 3 is observational); later steps will replace this
            # poll with a state-machine-aware loop.
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
        except Exception:
            _logger.exception("Voice response TTS failed")
    tracker.transition(ResponseState.IDLE)


async def subscribe_my_voice(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Join the caller's voice channel and register a voice subscription.

    When ``familiar.transcriber`` is configured, this also starts a
    per-user Deepgram transcription pipeline, debounces Deepgram
    finals through a :class:`VoiceLullMonitor`, and feeds the merged
    utterances into :attr:`Familiar.monitor` so voice turns flow
    through the **same** :class:`ConversationMonitor` gate (direct
    address, interjection check, conversational lull) that text
    channels use. The monitor's ``on_respond`` callback dispatches
    back to :func:`_run_voice_response` via the
    ``voice_response_handlers`` registry on :attr:`Familiar.extras`.
    Without a transcriber the bot still joins the channel and plays
    TTS but does not react to incoming speech.
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

        # Register a per-voice-channel response handler on the familiar.
        # The ``on_respond`` callback built in ``create_bot`` looks this
        # up by channel id and dispatches here when the
        # ``ConversationMonitor`` decides the familiar should speak on
        # this voice channel. Captures ``vc`` so the same voice client
        # connection handles playback for the life of the subscription.
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

        # Debounce Deepgram finals into one utterance per speaker turn,
        # then hand the merged transcript to the ConversationMonitor —
        # exactly the same entry point text channels use. The monitor
        # runs direct-address detection on the transcript text, counter-
        # based interjection checks, and a silence-based conversational
        # lull gate. On YES, ``_voice_response_handler`` fires via
        # ``on_respond``.
        async def _deliver_to_monitor(
            user_id: int,
            merged: TranscriptionResult,
        ) -> None:
            raw_name = user_names.get(user_id, f"User-{user_id}")
            safe_name = sanitize_name(raw_name) or raw_name
            # Voice lull dispatch: mark the tracker as GENERATING for the
            # duration of the side-model YES/NO eval so an interruption
            # that arrives while the familiar is "thinking about whether
            # to speak" is detectable. On YES, ``_run_voice_response``
            # also transitions to GENERATING (no-op) and continues into
            # SPEAKING/IDLE on its own. On NO, no on_respond fires, so
            # the tracker is still GENERATING when we get back here and
            # we transition it back to IDLE.
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
                    # The voice pipeline already debounced silence via
                    # VoiceLullMonitor, so this call is itself the lull.
                    # Tell the monitor not to start another lull timer.
                    is_lull_endpoint=True,
                )
            finally:
                # If the side-model said NO, no on_respond fired, the
                # tracker is still GENERATING — revert to IDLE so the
                # next turn starts clean. If YES, _run_voice_response
                # has already cycled the tracker through SPEAKING→IDLE.
                if tracker.state is ResponseState.GENERATING:
                    tracker.transition(ResponseState.IDLE)

        # Per-guild interruption detector. Consumes Discord voice-activity
        # events from the lull monitor (no separate Deepgram VAD path)
        # and classifies bursts as discarded/short/long relative to the
        # current ResponseTracker state. Detect-only for now — dispatch
        # lands in later steps.
        interruption_detector = InterruptionDetector(
            tracker_registry=familiar.tracker_registry,
            guild_id=voice_guild_id if voice_guild_id is not None else 0,
            min_interruption_s=familiar.config.min_interruption_s,
            short_long_boundary_s=familiar.config.short_long_boundary_s,
            lull_timeout_s=familiar.config.voice_lull_timeout,
            base_tolerance=(familiar.config.interrupt_tolerance.base_probability),
        )
        familiar.extras["interruption_detector"] = interruption_detector

        lull_monitor = VoiceLullMonitor(
            lull_timeout=familiar.config.voice_lull_timeout,
            user_silence_s=0.2,
            on_utterance_complete=_deliver_to_monitor,
            on_voice_activity=interruption_detector.on_voice_activity,
        )  # voice_lull_timeout is endpointing only; the conversational
        # lull (side-model YES/NO gate) is governed by text_lull_timeout
        # inside ConversationMonitor.
        familiar.extras["voice_lull_monitor"] = lull_monitor

        async def _route_transcript_to_monitor(  # noqa: RUF029
            user_id: int,
            result: TranscriptionResult,
        ) -> None:
            lull_monitor.on_transcript(user_id, result)

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
        # Stop the transcription pipeline first so audio chunks stop
        # flowing into a voice client that's about to disconnect.
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
        # Drop the per-channel voice response dispatch and clear any
        # monitor state for this voice channel so a later re-subscribe
        # starts fresh (and so a lull timer doesn't fire into a dead
        # voice client).
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

    # Main reply isolation: catch the closed raise set of
    # ``LLMClient.chat`` — ``httpx.HTTPError`` covers transport /
    # status / timeout; ``ValueError`` and ``KeyError`` cover the
    # no-choices / malformed-payload branches inside ``llm.chat``.
    # On failure, return without writing history, without post-
    # processing, without a Discord send, and without TTS fan-out.
    # The user sees silence and can simply retry.
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
                tts_result = await familiar.tts_client.synthesize(reply_text)
                stereo = mono_to_stereo(tts_result.audio)
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
        trigger: ResponseTrigger,
    ) -> None:
        # Voice dispatch first: if this channel has a voice response
        # handler registered by subscribe_my_voice, hand the buffer to
        # it. This is how the ConversationMonitor gate (direct address,
        # interjection check, conversational lull) reaches the voice
        # path now that voice shares the same monitor as text.
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
