"""Discord bot factory, slash commands, and pipeline-routed message loop.

Step 7 of ``future-features/context-management.md``, plus the
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
the process — per ``future-features/configuration-levels.md`` one
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
from familiar_connect.mood import effective_tolerance
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.voice import DaveVoiceClient, RecordingSink
from familiar_connect.voice.audio import mono_to_stereo
from familiar_connect.voice.interruption import (
    InterruptionDetector,
    InterruptionEvent,
    ResponseState,
    ResponseTracker,
    should_keep_talking,
    split_at_elapsed,
)
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


def _format_recent_for_mood(
    familiar: Familiar,
    channel_id: int,
    mode: ChannelMode | None,
) -> str:
    """Format recent history as text for the mood evaluator prompt."""
    turns = familiar.history_store.recent(
        familiar_id=familiar.id,
        channel_id=channel_id,
        limit=10,
        mode=mode,
    )
    lines: list[str] = []
    for turn in turns:
        prefix = turn.speaker or familiar.id if turn.role == "user" else familiar.id
        lines.append(f"{prefix}: {turn.content}")
    return "\n".join(lines) if lines else "(no recent conversation)"


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

    familiar.subscriptions.add(
        channel_id=channel_id,
        kind=SubscriptionKind.text,
        guild_id=ctx.guild_id,
    )
    name = getattr(ctx.channel, "name", str(channel_id))
    _logger.info("Subscribed to text channel: %s (%s)", name, channel_id)
    await ctx.respond(f"Subscribed to text in **#{name}**.")


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
    await ctx.respond("No longer listening here.")


def _build_voice_response_handler(
    *,
    vc: discord.VoiceClient,
    familiar: Familiar,
    voice_channel_id: int,
    guild_id: int | None,
    user_names: dict[int, str],
) -> tuple[
    Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]],
    ResponseTracker,
    InterruptionDetector,
    Callable[[int, str], None],
    Callable[[int, str], None],
]:
    """Build the async callback invoked for each final voice transcription.

    Routes voice turns through the **same** :class:`ContextPipeline`
    the text loop uses: assemble a :class:`ContextRequest` with
    :attr:`Modality.voice` and the voice channel's id, run the
    pipeline, render, call the main LLM, run post-processors,
    persist to :class:`HistoryStore`, fan out to TTS.

    Returns ``(handler, tracker, detector, vad_callback,
    deepgram_vad_callback)`` — the handler plus the per-guild
    :class:`ResponseTracker`, :class:`InterruptionDetector`,
    a synchronous VAD event callback for the voice debounce
    logic (driven by Discord audio), and a Deepgram VAD
    callback for the transcription flush gate.
    """
    tracker = ResponseTracker()

    # ----- Interruption handler callbacks -----

    async def _on_interrupt_start() -> bool:  # noqa: RUF029
        """Moment 1: toll check — yield or keep talking.

        Fires ``min_interruption_s`` after someone starts talking over
        the familiar.  Returns ``True`` if the familiar yields.

        When yielding during SPEAKING, captures the playback position
        and the delivered/remaining text split.  The actual decision
        (truncate or resume) is deferred to moment 2 — see
        :func:`_on_short_during_speaking` and
        :func:`_on_long_during_speaking`.
        """
        nonlocal yield_delivered, yield_remaining

        base = familiar.config.interrupt_tolerance.tolerance
        mood = tracker.mood_modifier
        tol = effective_tolerance(base, mood)
        if should_keep_talking(tol):
            _logger.info(
                "Interruption: pushing through (base=%.2f mood=%+.2f tol=%.2f)",
                base,
                mood,
                tol,
            )
            return False
        _logger.info(
            "Interruption: yielding (base=%.2f mood=%+.2f tol=%.2f)",
            base,
            mood,
            tol,
        )
        # Capture the playback position *before* stopping so the
        # elapsed-time measurement is as accurate as possible.
        if tracker.state is ResponseState.SPEAKING:
            elapsed_ms = tracker.stop_speaking()
            delivered, remaining = split_at_elapsed(
                tracker.word_timestamps,
                elapsed_ms,
            )
            yield_delivered = delivered or ""
            yield_remaining = remaining or ""
            interruption_resolved.clear()
            _logger.info(
                "Yield during SPEAKING (elapsed=%.0fms): "
                "delivered=%d chars, remaining=%d chars",
                elapsed_ms,
                len(yield_delivered),
                len(yield_remaining),
            )
        if vc.is_playing():
            vc.stop()
        return True

    async def _on_short_during_generating(event: InterruptionEvent) -> None:  # noqa: RUF029
        """Short interruption resolved during generation — hold response."""
        _logger.info(
            "Short interruption resolved during generation (%.1fs)",
            event.duration_s,
        )
        # Generation continues; response delivered once complete.

    async def _on_long_during_generating(event: InterruptionEvent) -> None:  # noqa: RUF029
        """Long interruption resolved during generation — cancel task."""
        _logger.info(
            "Long interruption resolved during generation (%.1fs)",
            event.duration_s,
        )
        if tracker.generation_task is not None:
            tracker.generation_task.cancel()

    async def _on_short_during_speaking(event: InterruptionEvent) -> None:  # noqa: RUF029
        """Short interruption resolved — signal resume.

        Leaves ``interrupted_reply`` as ``None`` so the playback loop
        in :func:`_generate_response` re-synthesises the remaining text.
        """
        _logger.info(
            "Short interruption resolved during speaking (%.1fs) "
            "— will resume remaining text",
            event.duration_s,
        )
        # interrupted_reply stays None → resume.
        interruption_resolved.set()

    async def _on_long_during_speaking(event: InterruptionEvent) -> None:  # noqa: RUF029
        """Long interruption resolved — signal truncation.

        Sets ``interrupted_reply`` to the text captured at moment 1
        so :func:`_generate_response` persists only the spoken portion.
        """
        nonlocal interrupted_reply
        interrupted_reply = yield_delivered or ""
        _logger.info(
            "Long interruption resolved during speaking (%.1fs) "
            "— truncating to %d chars",
            event.duration_s,
            len(interrupted_reply),
        )
        interruption_resolved.set()

    detector = InterruptionDetector(
        tracker=tracker,
        min_interruption_s=familiar.config.min_interruption_s,
        short_long_boundary_s=familiar.config.short_long_boundary_s,
        lull_timeout_s=familiar.config.lull_timeout,
        on_interrupt_start=_on_interrupt_start,
        on_short_during_generating=_on_short_during_generating,
        on_long_during_generating=_on_long_during_generating,
        on_short_during_speaking=_on_short_during_speaking,
        on_long_during_speaking=_on_long_during_speaking,
    )

    # Voice debounce state: accumulate transcription results and only
    # generate after lull_timeout seconds of silence.  Uses VAD events
    # to track active speakers so the timer only runs when nobody is
    # talking — this prevents premature generation when Deepgram
    # finalises a segment while the user is still mid-sentence.
    pending_utterances: list[tuple[int, TranscriptionResult]] = []
    lull_gen_task: asyncio.Task[None] | None = None
    debounce_speakers: set[int] = set()

    # Deepgram flush gate: after the lull timer fires, wait for
    # Deepgram to confirm all in-transit transcriptions have been
    # delivered (via UtteranceEnd) before generating.  Prevents
    # generating before a slow final transcript arrives.
    pending_deepgram_speakers: set[int] = set()
    deepgram_ready = asyncio.Event()
    deepgram_ready.set()  # Initially ready (no pending transcripts)

    # Interruption state shared between _on_interrupt_start (moment 1)
    # and _on_short/long_during_speaking (moment 2).
    #
    # At moment 1 (yield during SPEAKING): delivered/remaining text is
    # captured and interruption_resolved is cleared.  _generate_response
    # waits on interruption_resolved.  At moment 2: the handler sets
    # interrupted_reply (long → truncated text, short → None for resume)
    # and sets the event.
    yield_delivered: str | None = None
    yield_remaining: str | None = None
    interruption_resolved = asyncio.Event()
    interruption_resolved.set()  # No pending interruption initially.
    interrupted_reply: str | None = None

    async def _playback_loop(reply_text: str) -> str:
        """Synthesise, play, and handle interruption-resume cycles.

        Returns the text to persist in history — either the full
        ``reply_text`` (no interruption, or short-interrupt resume
        completed) or a truncated prefix (long interruption).
        """
        nonlocal interrupted_reply, yield_delivered, yield_remaining

        assert familiar.tts_client is not None  # noqa: S101

        text_to_play = reply_text
        delivered_so_far = ""
        first_segment = True

        while True:
            tts_result = await familiar.tts_client.synthesize_with_timestamps(
                text_to_play,
            )
            stereo = mono_to_stereo(tts_result.audio)
            # Wait for any currently-playing audio to finish.
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)

            if first_segment:
                # --- State machine: GENERATING → SPEAKING ---
                tracker.start_speaking(
                    word_timestamps=tts_result.timestamps,
                )
                first_segment = False
            else:
                tracker.resume_speaking(
                    word_timestamps=tts_result.timestamps,
                )

            vc.play(discord.PCMAudio(io.BytesIO(stereo)))
            # Wait for playback to finish (or interruption).
            while vc.is_playing():  # noqa: ASYNC110
                await asyncio.sleep(0.1)

            if yield_remaining is None:
                # Playback completed without interruption.
                return reply_text

            # Interrupted — wait for moment 2 classification.
            try:
                await asyncio.wait_for(
                    interruption_resolved.wait(),
                    timeout=30.0,
                )
            except TimeoutError:
                _logger.warning(
                    "Timed out waiting for interruption classification — truncating",
                )
                interrupted_reply = yield_delivered or ""

            if interrupted_reply is not None:
                # Long interruption — return truncated text.
                if delivered_so_far:
                    return f"{delivered_so_far} {interrupted_reply}"
                return interrupted_reply

            # Short interruption — resume remaining text.
            if yield_delivered:
                if delivered_so_far:
                    delivered_so_far = f"{delivered_so_far} {yield_delivered}"
                else:
                    delivered_so_far = yield_delivered

            text_to_play = yield_remaining or ""
            _logger.info(
                "Resuming remaining text after short interrupt: %r",
                text_to_play[:80],
            )
            # Reset for the next segment.
            yield_delivered = None
            yield_remaining = None
            interrupted_reply = None

    async def _generate_response(
        user_id: int,
        combined_text: str,
    ) -> None:
        """Run the full generation pipeline for accumulated voice input.

        Called by :func:`_flush_pending` after the lull timer fires.
        The tracker is already in GENERATING state (set by
        :func:`_lull_then_generate` when the lull expired).
        """
        try:
            # Resolve the speaker's display name. ``user_names`` is
            # mutated by voice_pipeline's name resolver when a late
            # joiner shows up, so a lookup failure falls back to a
            # stable ``User-<id>`` form rather than ``None``.
            speaker = user_names.get(user_id, f"User-{user_id}")
            safe_name = sanitize_name(speaker) or speaker

            channel_config = familiar.channel_configs.get(
                channel_id=voice_channel_id,
            )
            request = ContextRequest(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=guild_id,
                speaker=safe_name,
                utterance=combined_text,
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
                    f"{m.content[:200]}"
                    f"{'…' if len(m.content) > 200 else ''}"
                    for m in messages
                ),
            )

            # Pre-compute mood modifier for the toll check
            if familiar.mood_evaluator is not None and guild_id is not None:
                recent = _format_recent_for_mood(
                    familiar, voice_channel_id, channel_config.mode
                )
                evaluation = await familiar.mood_evaluator.evaluate(
                    recent_context=recent,
                )
                tracker.mood_modifier = evaluation.modifier

            reply = await familiar.llm_client.chat(messages)

            reply_text = await pipeline.run_post_processors(reply.content, request)
            tracker.generation_complete(reply_text)

            # Persist the user turn now — the assistant turn is
            # deferred until after playback so an interruption can
            # truncate the stored text to what was actually spoken.
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=guild_id,
                role="user",
                content=combined_text,
                speaker=safe_name,
                mode=channel_config.mode,
            )

            _logger.info("[Voice Response] %s", reply_text)

            # Reset interruption state before playback starts.
            nonlocal interrupted_reply, yield_delivered, yield_remaining
            interrupted_reply = None
            yield_delivered = None
            yield_remaining = None

            if familiar.tts_client is not None:
                stored_reply = await _playback_loop(reply_text)
            else:
                stored_reply = reply_text

            # Persist assistant turn.
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=guild_id,
                role="assistant",
                content=stored_reply,
                mode=channel_config.mode,
            )
            if stored_reply != reply_text:
                _logger.info(
                    "[Voice Interrupted] stored partial: %s",
                    stored_reply[:120],
                )
        except asyncio.CancelledError:
            _logger.info("Voice generation cancelled by interruption")
            return
        except Exception:
            _logger.exception("Voice response handler failed")
        finally:
            # --- State machine: → IDLE ---
            interrupted_reply = None
            yield_delivered = None
            yield_remaining = None
            tracker.reset()

    async def _flush_pending() -> None:
        """Drain the utterance buffer and generate a single response."""
        if not pending_utterances:
            return
        utterances = list(pending_utterances)
        pending_utterances.clear()

        last_user_id = utterances[-1][0]
        combined_text = " ".join(r.text for _, r in utterances)
        _logger.info(
            "Voice lull expired — generating from %d utterance(s): %r",
            len(utterances),
            combined_text[:120],
        )
        await _generate_response(last_user_id, combined_text)

    async def _lull_then_generate() -> None:
        """Wait for lull_timeout seconds of silence, then flush.

        After the lull timer expires, immediately transitions to
        GENERATING so the InterruptionDetector can handle speech
        during the Deepgram flush wait.  The task is cancellable
        during the sleep phase; during the Deepgram wait phase
        cancellation resets the tracker back to IDLE.
        """
        nonlocal lull_gen_task
        try:
            await asyncio.sleep(familiar.config.lull_timeout)

            # --- State machine: IDLE → GENERATING ---
            # Committed to generating once Deepgram confirms.
            current = asyncio.current_task()
            if current is not None:
                tracker.start_generating(current)

            # Wait for Deepgram to confirm all transcriptions flushed.
            if not deepgram_ready.is_set():
                _logger.info(
                    "Lull expired, waiting for Deepgram to flush (pending=%s)",
                    pending_deepgram_speakers,
                )
                try:
                    await asyncio.wait_for(deepgram_ready.wait(), timeout=5.0)
                except TimeoutError:
                    _logger.warning(
                        "Timed out waiting for Deepgram flush, "
                        "generating anyway (pending=%s)",
                        pending_deepgram_speakers,
                    )
                    pending_deepgram_speakers.clear()
                    deepgram_ready.set()
        except asyncio.CancelledError:
            # If we were already GENERATING (cancelled during flush
            # wait), roll back to IDLE so the next cycle can start.
            if tracker.state is ResponseState.GENERATING:
                tracker.reset()
            return
        # Past the cancellable phases — clear reference so new
        # results don't cancel the generation phase that follows.
        lull_gen_task = None
        await _flush_pending()

        # Check for utterances that arrived during generation.
        # If new speech was buffered while we were busy, kick off
        # another cycle.
        if (
            pending_utterances
            and not debounce_speakers
            and tracker.state is ResponseState.IDLE
        ):
            lull_gen_task = asyncio.create_task(_lull_then_generate())

    def _on_vad_event(user_id: int, event_type: str) -> None:
        """Track active speakers for VAD-gated debounce.

        Called by the voice pipeline's VAD dispatcher for every
        ``SpeechStarted`` / ``UtteranceEnd`` event, regardless of
        tracker state.
        """
        nonlocal lull_gen_task

        speaker = user_names.get(user_id, f"User-{user_id}")

        if event_type == "SpeechStarted":
            debounce_speakers.add(user_id)
            # Mark that Deepgram has pending audio for this user.
            pending_deepgram_speakers.add(user_id)
            deepgram_ready.clear()
            had_timer = lull_gen_task is not None
            # Someone started talking — cancel the lull timer,
            # but only if we haven't committed to generating yet.
            if lull_gen_task is not None and tracker.state is ResponseState.IDLE:
                lull_gen_task.cancel()
                lull_gen_task = None
            _logger.debug(
                "VAD SpeechStarted from %s (active=%s, timer_cancelled=%s)",
                speaker,
                debounce_speakers,
                had_timer and tracker.state is ResponseState.IDLE,
            )
        elif event_type == "UtteranceEnd":
            debounce_speakers.discard(user_id)
            _logger.debug(
                "VAD UtteranceEnd from %s (active=%s, pending=%d)",
                speaker,
                debounce_speakers,
                len(pending_utterances),
            )
            # Everyone stopped — start the lull timer if we have
            # buffered results and aren't already generating.
            if (
                not debounce_speakers
                and pending_utterances
                and tracker.state is ResponseState.IDLE
            ):
                if lull_gen_task is not None:
                    lull_gen_task.cancel()
                lull_gen_task = asyncio.create_task(_lull_then_generate())

    async def _handle_voice_result(  # noqa: RUF029
        user_id: int,
        result: TranscriptionResult,
    ) -> None:
        """Buffer a transcription result for VAD-gated debounce.

        Results are accumulated in ``pending_utterances``.  The lull
        timer is started by :func:`_on_vad_event` when all speakers
        stop.  As a fallback (in case VAD events arrive out of order
        or are missing), a timer is also started here when no speakers
        are currently active.
        """
        nonlocal lull_gen_task

        pending_utterances.append((user_id, result))
        _logger.info(
            "Buffered voice result from %s: %r (pending=%d)",
            user_names.get(user_id, f"User-{user_id}"),
            result.text[:80],
            len(pending_utterances),
        )

        # Each new transcript is direct evidence of recent speech.
        # (Re)start the lull timer so the countdown runs from the
        # latest result, not from a stale UtteranceEnd.  Without
        # this reset, a timer started by an earlier event can fire
        # before the next segment arrives during continuous speech.
        # Only start/restart the timer if we're still IDLE — once
        # GENERATING, the InterruptionDetector handles new speech.
        if not debounce_speakers and tracker.state is ResponseState.IDLE:
            if lull_gen_task is not None:
                lull_gen_task.cancel()
            lull_gen_task = asyncio.create_task(_lull_then_generate())

    def _on_deepgram_vad(user_id: int, event_type: str) -> None:
        """Track Deepgram transcription flush state.

        Called by the voice pipeline's VAD dispatcher for Deepgram
        ``UtteranceEnd`` events.  When all pending speakers have been
        flushed, ``deepgram_ready`` is set so the lull gate can
        proceed to generation.
        """
        if event_type == "UtteranceEnd":
            pending_deepgram_speakers.discard(user_id)
            if not pending_deepgram_speakers:
                deepgram_ready.set()
            _logger.debug(
                "Deepgram UtteranceEnd for %s (pending_deepgram=%s)",
                user_names.get(user_id, f"User-{user_id}"),
                pending_deepgram_speakers,
            )

    return (
        _handle_voice_result,
        tracker,
        detector,
        _on_vad_event,
        _on_deepgram_vad,
    )


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

    # Voice connection + DAVE handshake takes >3s, so defer.
    await ctx.defer()
    vc = await channel.connect(cls=DaveVoiceClient)
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

        response_handler, _tracker, detector, vad_cb, dg_vad_cb = (
            _build_voice_response_handler(
                vc=vc,
                familiar=familiar,
                voice_channel_id=channel.id,
                guild_id=ctx.guild_id,
                user_names=user_names,
            )
        )

        pipeline = await start_pipeline(
            familiar.transcriber,
            user_names=user_names,
            resolve_name=_resolve_from_channel,
            response_handler=response_handler,
            interruption_detector=detector,
            vad_callback=vad_cb,
            deepgram_vad_callback=dg_vad_cb,
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

    await ctx.followup.send(f"Joined **{channel.name}**.")


async def unsubscribe_voice(
    ctx: discord.ApplicationContext,
    familiar: Familiar,
) -> None:
    """Leave the current voice channel, tear down the pipeline, drop the sub."""
    vc = ctx.voice_client
    if vc is not None:
        # Stop the transcription pipeline first so audio chunks stop
        # flowing into a voice client that's about to disconnect.
        if get_pipeline() is not None:
            if hasattr(vc, "recording") and vc.recording:
                vc.stop_recording()
            await stop_pipeline()
            _logger.info("Stopped voice transcription pipeline")
        await vc.disconnect()

    guild_id = ctx.guild_id
    if guild_id is not None:
        sub = familiar.subscriptions.voice_in_guild(guild_id)
        if sub is not None:
            familiar.subscriptions.remove(
                channel_id=sub.channel_id,
                kind=SubscriptionKind.voice,
            )

    await ctx.respond("Left voice.")


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
    await ctx.respond(f"Channel mode set to **{mode.value}**.")


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
        reply = await familiar.llm_client.chat(messages)

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
