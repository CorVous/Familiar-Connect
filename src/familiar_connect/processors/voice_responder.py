"""Voice reply orchestrator.

Consumes ``voice.activity.start`` and ``voice.transcript.final``
from bus; produces LLM output via :meth:`LLMClient.chat_stream` and
speaks it through :class:`TTSPlayer`. Every step scoped to current
:class:`TurnScope` — new ``voice.activity.start`` cancels in-flight
work.

See plan § Design.3 (turn scope) and plan § Rollout Phase 2.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.topics import (
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.diagnostics.cold_cache import log_signals
from familiar_connect.diagnostics.voice_budget import (
    PHASE_LLM_FIRST_TOKEN,
    PHASE_TTS_FIRST_AUDIO,
    get_voice_budget_recorder,
)
from familiar_connect.llm import LLMDelta, Message
from familiar_connect.sentence_streamer import SentenceStreamer
from familiar_connect.silence import SilentDetector

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.bus.envelope import Event, TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.identity import Author
    from familiar_connect.llm import LLMClient
    from familiar_connect.tools.registry import ToolContext, ToolRegistry
    from familiar_connect.tts_player.protocol import TTSPlayer

    # ``(channel_id, user_id) -> Author | None``. wired from bot to
    # resolve Discord members live; ``None`` when member can't be
    # resolved (left guild, cache miss, etc.)
    MemberResolver = Callable[[int, int], "Author | None"]

_logger = logging.getLogger("familiar_connect.processors.voice_responder")


class VoiceResponder:
    """Orchestrates voice reply loop with turn-scoped cancellation."""

    name: str = "voice-responder"
    topics: tuple[str, ...] = (
        TOPIC_VOICE_ACTIVITY_START,
        TOPIC_VOICE_TRANSCRIPT_FINAL,
    )

    def __init__(
        self,
        *,
        assembler: Assembler,
        llm_client: LLMClient,
        tts_player: TTSPlayer,
        history_store: AsyncHistoryStore,
        router: TurnRouter,
        familiar_id: str,
        member_resolver: MemberResolver | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_context_factory: Callable[[int, str], ToolContext] | None = None,
        tool_filler_phrases: tuple[str, ...] = (
            "one sec...",
            "hold on...",
            "checking...",
        ),
        post_history_instructions: str = "",
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._tts = tts_player
        self._history = history_store
        self._sync_history = history_store.sync
        self._router = router
        self._familiar_id = familiar_id
        self._member_resolver = member_resolver
        # per-familiar etiquette appended to the trailing reminder
        # (post-history). empty = omitted.
        self._post_history_instructions = post_history_instructions
        # one in-flight final-handling task per (session, user);
        # replaced when newer final from same speaker arrives.
        # cross-user finals coexist — only TTS player serializes playback
        self._inflight: dict[str, asyncio.Task[None]] = {}
        # agentic-loop wiring — see :meth:`_stream_and_speak_with_tools`
        self._tool_registry = tool_registry
        self._tool_context_factory = tool_context_factory
        # short stock phrases spoken before tool execution when
        # iteration produced no spoken content. rotated round-robin
        # so repeat use doesn't always say same word
        self._tool_filler_phrases = tool_filler_phrases
        self._tool_filler_idx = 0

    @staticmethod
    def _user_id_from_event(event: Event) -> int | None:
        """Discord user_id from event payload; ``None`` if absent."""
        if not isinstance(event.payload, dict):
            return None
        raw = event.payload.get("user_id")
        if isinstance(raw, int):
            return raw
        return None

    @staticmethod
    def _scope_key(session_id: str, user_id: int | None) -> str:
        """Per-(session, user) key; falls back to channel-level for legacy events."""
        if user_id is None:
            return session_id
        return f"{session_id}:user:{user_id}"

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        if event.topic == TOPIC_VOICE_ACTIVITY_START:
            self._on_activity_start(event)
            return
        if event.topic == TOPIC_VOICE_TRANSCRIPT_FINAL:
            self._spawn_final(event)
            return

    async def wait_until_idle(self) -> None:
        """Await every in-flight final-handling task.

        Used by tests and graceful shutdown — no-op when nothing
        in flight. Suppresses ``CancelledError`` from spawned tasks
        whose scope was cancelled mid-flight.
        """
        tasks = [t for t in list(self._inflight.values()) if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Activity start — cancel prior, install new scope + stop TTS
    # ------------------------------------------------------------------

    def _on_activity_start(self, event: Event) -> None:
        user_id = self._user_id_from_event(event)
        scope_key = self._scope_key(event.session_id, user_id)
        # begin_turn cancels prior scope for this (session, user).
        # if prior scope is one currently being spoken, player's poll
        # loop catches ``scope.is_cancelled()`` and stops playback
        # within one poll tick. global ``tts.stop()`` here would also
        # cut a *different* user's in-flight reply — Discord gives us
        # one shared voice client, so any continuous speaker would
        # barrage stop() against bot's reply to someone else
        self._router.begin_turn(session_id=scope_key, turn_id=event.turn_id)

    # ------------------------------------------------------------------
    # Final dispatch — spawn so the bus loop keeps pulling events
    # ------------------------------------------------------------------

    def _spawn_final(self, event: Event) -> None:
        """Run ``_on_final`` as per-(session, user) task.

        Decouples ingestion from processing: bus dispatcher returns
        immediately after handing off, so fresh ``activity.start`` can
        call ``prior.cancel()`` and ``tts.stop()`` while this task
        still parked at LLM/TTS await point. Without this, soft
        scope-cancel never fires until prior reply has fully played,
        and user hears it lag behind.
        """
        user_id = self._user_id_from_event(event)
        scope_key = self._scope_key(event.session_id, user_id)
        prior_task = self._inflight.get(scope_key)
        if prior_task is not None and not prior_task.done():
            # newer final from same speaker without intervening
            # activity.start is unusual but defendable: cancel prior
            # so we don't double-speak. cross-user finals don't
            # collide because they live under different scope keys
            prior_task.cancel()
        task = asyncio.create_task(
            self._run_final(event),
            name=f"voice-final-{event.turn_id}",
        )
        self._inflight[scope_key] = task
        task.add_done_callback(lambda t, sid=scope_key: self._on_final_done(sid, t))

    def _on_final_done(self, scope_key: str, task: asyncio.Task[None]) -> None:
        # only clear slot if we still own it — a newer turn may
        # have already replaced our entry
        if self._inflight.get(scope_key) is task:
            self._inflight.pop(scope_key, None)

    async def _run_final(self, event: Event) -> None:
        try:
            await self._on_final(event)
        except asyncio.CancelledError:
            # expected on barge-in: a newer final hard-cancelled us
            return

    # ------------------------------------------------------------------
    # Final transcript — run the reply pipeline
    # ------------------------------------------------------------------

    async def _on_final(self, event: Event) -> None:
        user_id = self._user_id_from_event(event)
        scope_key = self._scope_key(event.session_id, user_id)
        scope = self._router.active_scope(scope_key)
        if scope is None or scope.turn_id != event.turn_id:
            # stale final — newer utterance already started
            return
        channel_id = _parse_voice_session(event.session_id)
        if channel_id is None:
            return

        text = (
            (event.payload or {}).get("text", "")
            if isinstance(event.payload, dict)
            else ""
        )
        if not text:
            return

        author = self._resolve_author(channel_id=channel_id, user_id=user_id)

        # cold-cache signals — instrumentation only (no action yet).
        # runs before user turn appended so ``prev_turn_at`` reflects
        # real gap
        self._emit_cold_cache_signals(
            channel_id=channel_id, turn_id=scope.turn_id, text=text
        )

        # record user turn so RecentHistoryLayer picks it up next time
        await self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="user",
            content=text,
            author=author,
        )

        # seed retrieval cue for RagContextLayer (if wired)
        self._assembler.set_rag_cue(text)

        reply = await self._stream_and_speak(scope, channel_id)
        if reply is None or scope.is_cancelled():
            return
        # Cartesia rejects empty/whitespace ``transcript`` with HTTP 400.
        # empty reply usually means LLM stream emitted no deltas —
        # bad model name, content filter, or upstream error frame
        # parser silently dropped. mirrors ``TextResponder``'s guard.
        # _stream_and_speak gates TTS on this too, so just skip
        # assistant-turn write here
        if not reply.strip():
            _logger.warning(
                f"{ls.tag('Voice', ls.Y)} "
                f"{ls.kv('skip', 'empty_reply', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return

        await self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="assistant",
            content=reply,
            author=None,
        )
        self._router.end_turn(scope)

    def _resolve_author(self, *, channel_id: int, user_id: int | None) -> Author | None:
        """Look up Discord member; swallow resolver errors as anon turn."""
        if user_id is None or self._member_resolver is None:
            return None
        try:
            return self._member_resolver(channel_id, user_id)
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                f"{ls.tag('Voice', ls.Y)} "
                f"{ls.kv('member_resolve_error', repr(exc), vc=ls.Y)}"
            )
            return None

    async def _stream_and_speak(self, scope: TurnScope, channel_id: int) -> str | None:
        """Stream LLM output, speak sentence-by-sentence.

        Returns concatenated reply on speech, ``None`` on silent /
        empty / cancelled / stream error. ``<silent>`` sentinel
        decided before any sentence reaches TTS — buffered sentences
        wait until :class:`SilentDetector` rules in/out.
        """
        ctx = AssemblyContext(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            viewer_mode="voice",
        )
        prompt = await self._assembler.assemble(ctx)
        tool_mode = (
            self._tool_registry is not None
            and self._tool_context_factory is not None
            and self._llm.tool_calling_enabled
        )
        reminder = build_final_reminder(viewer_mode="voice", tools_enabled=tool_mode)
        system = "\n\n".join(s for s in (prompt.system_prompt, reminder) if s)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.extend(prompt.recent_history)
        # trailing reminder: same content + per-mode operating
        # directive ("You are speaking aloud…") so format gate lives
        # at tail of context, where recency-biased models are more
        # likely to honor it. end-placed tool nudge piggybacks on
        # trailing copy so it lands closest to next assistant turn
        trailing = build_final_reminder(
            viewer_mode="voice",
            include_mode_instruction=True,
            tools_enabled=tool_mode,
            post_history_instructions=self._post_history_instructions,
        )
        messages.append(Message(role="system", content=trailing))

        if tool_mode:
            return await self._stream_and_speak_with_tools(
                scope, channel_id=channel_id, messages=messages
            )

        accumulated: list[str] = []
        streamer = SentenceStreamer()
        silent = SilentDetector()
        # sentences buffered while silent gate still pending.
        # drained in arrival order once gate opens; dropped on ``True``
        pending: list[str] = []
        gate_open = False  # SilentDetector returned False — speak path live
        budget = get_voice_budget_recorder()
        first_delta_seen = False
        # exactly one decision line per turn — silent | respond | preempted.
        # tracked so cancel-mid-speak after gate_open doesn't double-log
        decision_logged = False

        try:
            async for delta in self._llm.chat_stream(messages):
                if scope.is_cancelled():
                    if not decision_logged:
                        self._log_preempted(scope.turn_id)
                    return None
                if not first_delta_seen:
                    budget.record(turn_id=scope.turn_id, phase=PHASE_LLM_FIRST_TOKEN)
                    first_delta_seen = True
                accumulated.append(delta)

                if not gate_open:
                    decision = silent.feed(delta)
                    if decision is True:
                        _logger.info(
                            f"{ls.tag('💤 Voice', ls.B)} "
                            f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                        )
                        return None
                    if decision is False:
                        gate_open = True
                        self._log_respond(scope.turn_id)
                        decision_logged = True

                pending.extend(streamer.feed(delta))
                if gate_open:
                    while pending:
                        if scope.is_cancelled():
                            if not decision_logged:
                                self._log_preempted(scope.turn_id)
                            return None
                        await self._speak(pending.pop(0), scope=scope)

            # stream ended undecided (very short / whitespace-only
            # reply). treat non-empty content as speak path so
            # _on_final's empty-reply guard runs symmetrically with
            # streaming case
            if (
                not gate_open
                and silent.decided is None
                and "".join(accumulated).strip()
            ):
                self._log_respond(scope.turn_id)
                decision_logged = True
                gate_open = True

            if gate_open:
                tail = streamer.flush()
                if tail.strip():
                    pending.append(tail)
                while pending:
                    if scope.is_cancelled():
                        if not decision_logged:
                            self._log_preempted(scope.turn_id)
                        return None
                    await self._speak(pending.pop(0), scope=scope)
        except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
            _logger.warning(
                f"{ls.tag('Voice', ls.R)} "
                f"{ls.kv('llm_stream_error', repr(exc), vc=ls.R)}"
            )
            return None

        return "".join(accumulated)

    async def _stream_and_speak_with_tools(
        self,
        scope: TurnScope,
        *,
        channel_id: int,
        messages: list[Message],
    ) -> str | None:
        """Agentic-loop variant of :meth:`_stream_and_speak`.

        Streams content to TTS as it arrives via ``on_delta``,
        executes tools after each stream closes, re-prompts until
        model stops calling tools. When iteration closes with a
        tool_call and no spoken content, a stock filler phrase is
        spoken before handler runs so user never hears long silent
        gap.
        """
        from familiar_connect.tools.loop import agentic_loop  # noqa: PLC0415

        ctx_factory = self._tool_context_factory
        registry = self._tool_registry
        if ctx_factory is None or registry is None:  # pragma: no cover — guard
            return None
        tool_ctx = ctx_factory(channel_id, scope.turn_id)

        accumulated: list[str] = []
        streamer = SentenceStreamer()
        silent = SilentDetector()
        pending: list[str] = []
        gate_open = False
        budget = get_voice_budget_recorder()
        first_delta_seen = False
        decision_logged = False

        async def _drain_pending() -> None:
            while pending:
                if scope.is_cancelled():
                    return
                await self._speak(pending.pop(0), scope=scope)

        async def _on_delta(delta: LLMDelta) -> None:
            nonlocal first_delta_seen, gate_open, decision_logged
            if scope.is_cancelled():
                return
            if not delta.content:
                return
            if not first_delta_seen:
                budget.record(turn_id=scope.turn_id, phase=PHASE_LLM_FIRST_TOKEN)
                first_delta_seen = True
            accumulated.append(delta.content)
            if not gate_open:
                decision = silent.feed(delta.content)
                if decision is True:
                    _logger.info(
                        f"{ls.tag('💤 Voice', ls.B)} "
                        f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                        f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                    )
                    pending.clear()
                    return
                if decision is False:
                    gate_open = True
                    self._log_respond(scope.turn_id)
                    decision_logged = True
            pending.extend(streamer.feed(delta.content))
            if gate_open:
                await _drain_pending()

        async def _on_before_tools(assistant: Message) -> None:
            # flush any in-flight sentence boundary so we don't speak
            # a half-formed clause after tool execution
            if gate_open:
                tail = streamer.flush()
                if tail.strip():
                    pending.append(tail)
                await _drain_pending()
            # filler backstop: empty content with imminent tools →
            # speak short stock phrase so user hears acknowledgement
            # before silent tool window
            if assistant.tool_calls and not (assistant.content or "").strip():
                phrase = self._next_filler_phrase()
                if phrase and not scope.is_cancelled():
                    await self._speak(phrase, scope=scope)

        async def _on_iteration_end(
            assistant: Message,
            tool_msgs: list[Message],
        ) -> None:
            # only persist intermediate iterations — terminal
            # text-only iteration handled by :meth:`_on_final`
            # alongside user turn
            if not assistant.tool_calls:
                return
            import json as _json  # noqa: PLC0415 — keep top minimal

            await self._history.append_turn(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                role="assistant",
                content=assistant.content or "",
                author=None,
                tool_calls_json=_json.dumps(assistant.tool_calls),
            )
            for tm in tool_msgs:
                await self._history.append_turn(
                    familiar_id=self._familiar_id,
                    channel_id=channel_id,
                    role="tool",
                    content=tm.content,
                    tool_call_id=tm.tool_call_id,
                )

        try:
            await agentic_loop(
                llm=self._llm,
                messages=messages,
                registry=registry,
                ctx=tool_ctx,
                on_delta=_on_delta,
                on_before_tools=_on_before_tools,
                on_iteration_end=_on_iteration_end,
            )
        except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
            _logger.warning(
                f"{ls.tag('Voice', ls.R)} "
                f"{ls.kv('llm_agentic_error', repr(exc), vc=ls.R)}"
            )
            return None

        # flush any trailing sentence from terminal iteration
        if gate_open:
            tail = streamer.flush()
            if tail.strip():
                pending.append(tail)
            await _drain_pending()
        # short / whitespace-only terminal content: still mark respond
        # so empty-reply log doesn't double-fire
        if not gate_open and silent.decided is None and "".join(accumulated).strip():
            self._log_respond(scope.turn_id)
            decision_logged = True
        if scope.is_cancelled() and not decision_logged:
            self._log_preempted(scope.turn_id)
            return None
        return "".join(accumulated)

    def _next_filler_phrase(self) -> str:
        """Round-robin select filler from configured list."""
        if not self._tool_filler_phrases:
            return ""
        phrase = self._tool_filler_phrases[
            self._tool_filler_idx % len(self._tool_filler_phrases)
        ]
        self._tool_filler_idx += 1
        return phrase

    def _log_respond(self, turn_id: str) -> None:
        """One ``decision=respond`` line per turn; matches silent log."""
        _logger.info(
            f"{ls.tag('Voice', ls.G)} "
            f"{ls.kv('decision', 'respond', vc=ls.LG)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)}"
        )

    def _log_preempted(self, turn_id: str) -> None:
        """Cancel-before-decision marker.

        Emitted when ``scope.is_cancelled()`` short-circuits stream
        loop before silent/respond latched. Without this line, a
        barge-in chain (continuous speaker) leaves a trail of
        ``[LLM call] status=cancelled`` with no way to tell which
        transcript was preempted.
        """
        _logger.info(
            f"{ls.tag('Voice', ls.Y)} "
            f"{ls.kv('decision', 'preempted', vc=ls.LY)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)}"
        )

    async def _speak(self, text: str, *, scope: TurnScope) -> None:
        """Skip whitespace-only chunks; ``DiscordVoicePlayer`` would too."""
        if not text.strip():
            return
        # first call per turn marks tts_first_audio. recorder dedupes
        # so subsequent sentences don't overwrite
        get_voice_budget_recorder().record(
            turn_id=scope.turn_id, phase=PHASE_TTS_FIRST_AUDIO
        )
        await self._tts.speak(text, scope=scope)

    # ------------------------------------------------------------------
    # Cold-cache signal emission (Phase-3 instrumentation)
    # ------------------------------------------------------------------

    def _emit_cold_cache_signals(
        self, *, channel_id: int, turn_id: str, text: str
    ) -> None:
        """Emit cold-cache spans; swallow errors.

        Best-effort — instrumentation must never block reply path.
        """
        try:
            summary = self._sync_history.get_summary(
                familiar_id=self._familiar_id, channel_id=channel_id
            )
            prior_context = summary.summary_text if summary is not None else ""
            # pull most recent turn's timestamp for silence-gap
            recent = self._sync_history.recent(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                limit=1,
            )
            prev_at = recent[0].timestamp if recent else None
            log_signals(
                channel_id=channel_id,
                turn_id=turn_id,
                new_text=text,
                prior_context=prior_context,
                prev_turn_at=prev_at,
                current_turn_at=datetime.now(tz=UTC),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                f"{ls.tag('ColdCache', ls.Y)} "
                f"{ls.kv('signal_emit_error', repr(exc), vc=ls.Y)}"
            )


def _parse_voice_session(session_id: str) -> int | None:
    """``voice:123`` → ``123``; ``None`` for non-voice sessions."""
    if not session_id.startswith("voice:"):
        return None
    try:
        return int(session_id.split(":", 1)[1])
    except ValueError:
        return None
