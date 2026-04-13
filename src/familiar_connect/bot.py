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
import enum
import io
import logging
from typing import TYPE_CHECKING

import discord
import httpx

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
# Voice response state machine
# ---------------------------------------------------------------------------


class ResponseState(enum.Enum):
    """The phases of a voice response lifecycle relevant to debounce."""

    IDLE = "idle"
    """Not generating. The lull timer may be running."""

    GENERATING = "generating"
    """LLM call is in-flight. New speech is buffered, not acted upon."""


def _idle_event_factory() -> asyncio.Event:
    """Create an :class:`asyncio.Event` that starts **set** (IDLE)."""
    event = asyncio.Event()
    event.set()
    return event


@dataclasses.dataclass
class ResponseTracker:
    """Per-guild state machine for the voice debounce lifecycle.

    :attr:`idle_event` is **set** when IDLE and **cleared** when
    GENERATING so concurrent handlers can ``await idle_event.wait()``.
    """

    state: ResponseState = ResponseState.IDLE
    generation_task: asyncio.Task[object] | None = None
    idle_event: asyncio.Event = dataclasses.field(
        default_factory=_idle_event_factory,
    )

    def start_generating(self, task: asyncio.Task[object]) -> None:
        """Transition IDLE → GENERATING."""
        if self.state is not ResponseState.IDLE:
            msg = f"Cannot start generating from state {self.state.value}"
            raise RuntimeError(msg)
        self.state = ResponseState.GENERATING
        self.generation_task = task
        self.idle_event.clear()

    def reset(self) -> None:
        """Return to IDLE and clear transient state."""
        self.state = ResponseState.IDLE
        self.generation_task = None
        self.idle_event.set()


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
) -> tuple[
    Callable[[int, TranscriptionResult], Coroutine[Any, Any, None]],
    ResponseTracker,
    Callable[[int, str], None],
    Callable[[int, str], None],
]:
    """Build the debounced voice response handler and its VAD callbacks.

    Routes accumulated voice utterances through the :class:`ContextPipeline`
    after a configurable lull timeout, preventing premature generation on
    partial transcripts.

    Returns ``(handler, tracker, vad_callback, deepgram_vad_callback)``:

    - *handler* — async callback for each final transcription result.
    - *tracker* — per-guild :class:`ResponseTracker` (IDLE/GENERATING).
    - *vad_callback* — sync callback for Discord audio packet timing
      events (``SpeechStarted`` / ``UtteranceEnd``). Gates the lull
      timer so it only runs when no speakers are active.
    - *deepgram_vad_callback* — sync callback for Deepgram
      ``UtteranceEnd`` events. Used to wait for in-transit transcripts
      before generating.
    """
    tracker = ResponseTracker()

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
                    f"{m.content[:200]}{'…' if len(m.content) > 200 else ''}"
                    for m in messages
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
                return
            reply_text = await pipeline.run_post_processors(reply.content, request)

            # Persist both turns *after* the LLM call so a mid-turn
            # crash doesn't leave the store with a user turn but no
            # reply.
            familiar.history_store.append_turn(
                familiar_id=familiar.id,
                channel_id=voice_channel_id,
                guild_id=guild_id,
                role="user",
                content=combined_text,
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
                pcm_mono = await familiar.tts_client.synthesize(reply_text)
                stereo = mono_to_stereo(pcm_mono)
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
                vc.play(discord.PCMAudio(io.BytesIO(stereo)))
                while vc.is_playing():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            _logger.info("Voice generation cancelled")
            return
        except Exception:
            _logger.exception("Voice response handler failed")
        finally:
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
        GENERATING so new speech is buffered rather than cancelling
        the in-progress generation.  The task is cancellable during
        the sleep phase; during the Deepgram wait phase cancellation
        resets the tracker back to IDLE.
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

        Called by the voice pipeline for every ``SpeechStarted`` /
        ``UtteranceEnd`` event from Discord audio packet timing,
        regardless of tracker state.
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
        # latest result, not from a stale UtteranceEnd.  Only
        # start/restart the timer if we're still IDLE — once
        # GENERATING, new speech is buffered for the next cycle.
        if not debounce_speakers and tracker.state is ResponseState.IDLE:
            if lull_gen_task is not None:
                lull_gen_task.cancel()
            lull_gen_task = asyncio.create_task(_lull_then_generate())

    def _on_deepgram_vad(user_id: int, event_type: str) -> None:
        """Track Deepgram transcription flush state.

        Called by the voice pipeline for Deepgram ``UtteranceEnd``
        events.  When all pending speakers have been flushed,
        ``deepgram_ready`` is set so the lull gate can proceed to
        generation.
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

        response_handler, _tracker, vad_cb, dg_vad_cb = _build_voice_response_handler(
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
