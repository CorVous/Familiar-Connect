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
import time
from typing import TYPE_CHECKING, cast

import discord
import httpx

from familiar_connect.chattiness import BufferedMessage, ResponseTrigger
from familiar_connect.config import ChannelMode
from familiar_connect.context.render import assemble_chat_messages
from familiar_connect.context.types import ContextRequest, Modality, PendingTurn
from familiar_connect.llm import sanitize_name
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
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

    from familiar_connect.context.pipeline import ProviderOutcome
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
    interruption_context: str | None = None,
) -> None:
    """Run full pipeline → LLM → TTS path for a voice channel.

    Called by ``on_respond`` when :class:`ConversationMonitor` decides
    the familiar should speak. Mirrors :func:`_run_text_response`;
    voice uses :attr:`Modality.voice` and trims LLM-calling providers
    and processors for lower latency. Persists every buffered user
    utterance in order, then the assistant reply, so the history
    store reflects the full conversation.
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
        speaker=speaker,
        utterance=utterance,
        modality=Modality.voice,
        budget_tokens=channel_config.budget_tokens,
        deadline_s=channel_config.deadline_s,
        pending_turns=tuple(
            PendingTurn(speaker=m.speaker, text=m.text) for m in buffer
        ),
        interruption_context=interruption_context,
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

    # main reply isolation: catch ``LLMClient.chat`` raise set
    # (transport, status, malformed payload) and return cleanly so
    # the monitor callback stays alive for the next utterance.
    #
    # Step 7: the chat call runs as a cancellable asyncio.Task parked
    # on ``tracker.generation_task`` so a later long-interruption
    # handler can call ``tracker.generation_task.cancel()`` to abort
    # mid-generation without wasting tokens. No caller cancels yet —
    # this is plumbing for Steps 8 and 12.
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
            # The outer task was cancelled from above — propagate so
            # we don't swallow someone else's cancellation.
            tracker.transition(ResponseState.IDLE)
            raise
        # The generation task itself was cancelled (interruption path).
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
    reply_text = await pipeline.run_post_processors(reply.content, request)
    tracker.response_text = reply_text

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
                    speaker=msg.speaker,
                    mode=channel_config.mode,
                )
            # Step 9 — flush interrupter turns captured during GENERATING
            # (short@GENERATING dispatch stashed them on the tracker).
            # Ordering: after buffer, before assistant reply → chronology.
            for raw_name, text in tracker.pending_interrupter_turns:
                safe_name = sanitize_name(raw_name) or raw_name
                familiar.history_store.append_turn(
                    familiar_id=familiar.id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    role="user",
                    content=text,
                    speaker=safe_name,
                    mode=channel_config.mode,
                )
            tracker.pending_interrupter_turns.clear()
            _logger.info("[Voice Response] %s", reply_text)
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
                    "dispatch: long@SPEAKING → regen speaker=%s",
                    tracker.interrupt_starter_name,
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
                _logger.info("[Voice Regen Response] %s", regen_text)
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
    tracker.transition(ResponseState.IDLE)


async def dispatch_interruption_regen(
    channel_id: int,
    guild_id: int | None,
    speaker: str,
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
        f"{speaker} interrupted while you were forming a response."
        f' They said: "{transcript}"'
    )
    _logger.info(
        "dispatch: long@GENERATING → cancel+regen speaker=%s",
        speaker,
    )
    await _run_voice_response(
        channel_id=channel_id,
        guild_id=guild_id,
        speaker=speaker,
        utterance=transcript,
        buffer=[
            BufferedMessage(
                speaker=speaker,
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

    When ``familiar.transcriber`` is set, starts a per-user Deepgram
    pipeline, debounces finals through :class:`VoiceLullMonitor`, and
    feeds merged utterances into :attr:`Familiar.monitor` so voice
    turns share the same ``ConversationMonitor`` gate as text. On
    YES, the monitor dispatches via ``voice_response_handlers`` in
    :attr:`Familiar.extras`. Without a transcriber, joins for TTS only.
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
            tracker = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0,
            )
            # Guard 1: a long-interruption regen was dispatched synchronously
            # before this coroutine runs — let the regen path handle the speech.
            # Guard 2: familiar is already generating/speaking (concurrent lull).
            if (
                familiar.extras.pop("_regen_pending", False)
                or tracker.state is not ResponseState.IDLE
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

        # Step 11 dispatch callbacks. On short+yield: re-synth the remaining
        # words (captured at stop-time) and resume playback. On push-through:
        # write the interrupter's transcript to history so it's not lost.

        async def _on_short_yield_resume(remaining: list[WordTimestamp]) -> None:
            t = familiar.tracker_registry.get(
                voice_guild_id if voice_guild_id is not None else 0
            )
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
            raw_name = user_names.get(user_id, f"User-{user_id}")
            safe_name = sanitize_name(raw_name) or raw_name
            ch_cfg = familiar.channel_configs.get(channel_id=voice_channel_id)
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=voice_guild_id,
                role="user",
                content=transcript,
                speaker=safe_name,
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
            raw_name = user_names.get(starter_id, f"User-{starter_id}")
            safe_name = sanitize_name(raw_name) or raw_name
            # Set before creating the task so _deliver_to_monitor (which
            # fires in the same event-loop tick via _fire_lull) sees it.
            familiar.extras["_regen_pending"] = True
            t = asyncio.create_task(
                dispatch_interruption_regen(
                    channel_id=voice_channel_id,
                    guild_id=voice_guild_id,
                    speaker=safe_name,
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
            name_resolver=lambda uid: user_names.get(uid, f"User-{uid}"),
        )
        familiar.extras["interruption_detector"] = interruption_detector

        # debounce per-final Deepgram fragments into a single utterance
        # via VoiceLullMonitor; fires after voice_lull_timeout of silence

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
            # Forward finals to the interruption detector so it can
            # accumulate the burst transcript for dispatch.
            if result.is_final:
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
