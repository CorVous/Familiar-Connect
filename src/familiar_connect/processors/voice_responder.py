"""Voice reply orchestrator.

Consumes ``voice.activity.start`` and ``voice.transcript.final`` from
the bus; produces LLM output via :meth:`LLMClient.chat_stream` and
speaks it through :class:`TTSPlayer`. Every step is scoped to the
current :class:`TurnScope` — a new ``voice.activity.start`` cancels
in-flight work.

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
from familiar_connect.llm import Message
from familiar_connect.sentence_streamer import SentenceStreamer
from familiar_connect.silence import SilentDetector

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.bus.envelope import Event, TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.identity import Author
    from familiar_connect.llm import LLMClient
    from familiar_connect.tts_player.protocol import TTSPlayer

    # ``(channel_id, user_id) -> Author | None``. Wired from the bot to
    # resolve Discord members live; ``None`` when the member can't be
    # resolved (left guild, cache miss, etc.).
    MemberResolver = Callable[[int, int], "Author | None"]

_logger = logging.getLogger("familiar_connect.processors.voice_responder")


class VoiceResponder:
    """Orchestrates the voice reply loop with turn-scoped cancellation."""

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
        history_store: HistoryStore,
        router: TurnRouter,
        familiar_id: str,
        member_resolver: MemberResolver | None = None,
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._tts = tts_player
        self._history = history_store
        self._router = router
        self._familiar_id = familiar_id
        self._member_resolver = member_resolver
        # one in-flight final-handling task per (session, user); replaced
        # when a newer final from the same speaker arrives. cross-user
        # finals coexist — only the TTS player serializes playback.
        self._inflight: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _user_id_from_event(event: Event) -> int | None:
        """Extract Discord user_id from an event payload, if present."""
        if not isinstance(event.payload, dict):
            return None
        raw = event.payload.get("user_id")
        if isinstance(raw, int):
            return raw
        return None

    @staticmethod
    def _scope_key(session_id: str, user_id: int | None) -> str:
        """Per-(session, user) key. Falls back to channel-level for legacy events."""
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

        Used by tests and graceful shutdown — a no-op when nothing is
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
        # begin_turn cancels the prior scope for this (session, user).
        # If the prior scope is the one currently being spoken, the
        # player's poll loop catches ``scope.is_cancelled()`` and stops
        # playback within one poll tick. A global ``tts.stop()`` here
        # would also cut a *different* user's in-flight reply — Discord
        # gives us one shared voice client, so any continuous speaker
        # would barrage stop() against the bot's reply to someone else.
        self._router.begin_turn(session_id=scope_key, turn_id=event.turn_id)

    # ------------------------------------------------------------------
    # Final dispatch — spawn so the bus loop keeps pulling events
    # ------------------------------------------------------------------

    def _spawn_final(self, event: Event) -> None:
        """Run ``_on_final`` as a per-(session, user) task.

        Decouples ingestion from processing: the bus dispatcher returns
        immediately after handing off, so a fresh ``activity.start``
        can call ``prior.cancel()`` and ``tts.stop()`` while this task
        is still parked at an LLM/TTS await point. Without this, the
        soft scope-cancel never fires until the prior reply has fully
        played, and the user hears it lag behind.
        """
        user_id = self._user_id_from_event(event)
        scope_key = self._scope_key(event.session_id, user_id)
        prior_task = self._inflight.get(scope_key)
        if prior_task is not None and not prior_task.done():
            # A newer final from the same speaker without an intervening
            # activity.start is unusual but defendable: cancel the prior
            # so we don't double-speak. Cross-user finals don't collide
            # because they live under different scope keys.
            prior_task.cancel()
        task = asyncio.create_task(
            self._run_final(event),
            name=f"voice-final-{event.turn_id}",
        )
        self._inflight[scope_key] = task
        task.add_done_callback(lambda t, sid=scope_key: self._on_final_done(sid, t))

    def _on_final_done(self, scope_key: str, task: asyncio.Task[None]) -> None:
        # Only clear the slot if we still own it — a newer turn may
        # have already replaced our entry.
        if self._inflight.get(scope_key) is task:
            self._inflight.pop(scope_key, None)

    async def _run_final(self, event: Event) -> None:
        try:
            await self._on_final(event)
        except asyncio.CancelledError:
            # Expected on barge-in: a newer final hard-cancelled us.
            return

    # ------------------------------------------------------------------
    # Final transcript — run the reply pipeline
    # ------------------------------------------------------------------

    async def _on_final(self, event: Event) -> None:
        user_id = self._user_id_from_event(event)
        scope_key = self._scope_key(event.session_id, user_id)
        scope = self._router.active_scope(scope_key)
        if scope is None or scope.turn_id != event.turn_id:
            # stale final — a newer utterance already started
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

        # Cold-cache signals — instrumentation only (no action yet).
        # Runs before the user turn is appended so ``prev_turn_at``
        # reflects the real gap.
        self._emit_cold_cache_signals(
            channel_id=channel_id, turn_id=scope.turn_id, text=text
        )

        # Record the user turn so RecentHistoryLayer picks it up next time.
        self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="user",
            content=text,
            author=author,
        )

        # Seed retrieval cue for RagContextLayer (if wired).
        self._assembler.set_rag_cue(text)

        reply = await self._stream_and_speak(scope, channel_id)
        if reply is None or scope.is_cancelled():
            return
        # Cartesia rejects empty/whitespace ``transcript`` with HTTP 400.
        # An empty reply usually means the LLM stream emitted no deltas —
        # bad model name, content filter, or upstream error frame the
        # parser silently dropped. Mirrors ``TextResponder``'s guard.
        # _stream_and_speak gates TTS on this too, so we just skip the
        # assistant-turn write here.
        if not reply.strip():
            _logger.warning(
                f"{ls.tag('Voice', ls.Y)} "
                f"{ls.kv('skip', 'empty_reply', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return

        self._history.append_turn(
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
        empty / cancelled / stream error. ``<silent>`` sentinel is
        decided before any sentence reaches TTS — buffered sentences
        wait until :class:`SilentDetector` rules in/out.
        """
        ctx = AssemblyContext(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            viewer_mode="voice",
        )
        prompt = await self._assembler.assemble(ctx)
        reminder = build_final_reminder(viewer_mode="voice")
        system = "\n\n".join(s for s in (prompt.system_prompt, reminder) if s)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.extend(prompt.recent_history)

        accumulated: list[str] = []
        streamer = SentenceStreamer()
        silent = SilentDetector()
        # sentences buffered while the silent gate is still pending.
        # drained in arrival order once gate opens; dropped on ``True``.
        pending: list[str] = []
        gate_open = False  # SilentDetector returned False — speak path live
        budget = get_voice_budget_recorder()
        first_delta_seen = False
        # exactly one decision line per turn — silent | respond | preempted.
        # tracked so a cancel-mid-speak after gate_open doesn't double-log.
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

            # Stream ended undecided (very short / whitespace-only reply).
            # Treat non-empty content as speak path so _on_final's
            # empty-reply guard runs symmetrically with the streaming case.
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

    def _log_respond(self, turn_id: str) -> None:
        """One ``decision=respond`` line per turn, matching the silent log."""
        _logger.info(
            f"{ls.tag('Voice', ls.G)} "
            f"{ls.kv('decision', 'respond', vc=ls.LG)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)}"
        )

    def _log_preempted(self, turn_id: str) -> None:
        """Cancel-before-decision marker.

        Emitted when ``scope.is_cancelled()`` short-circuits the stream
        loop before silent/respond latched. Without this line a
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
        # First call per turn marks tts_first_audio. Recorder dedupes,
        # so subsequent sentences don't overwrite.
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

        Best-effort — instrumentation must never block the reply path.
        """
        try:
            summary = self._history.get_summary(
                familiar_id=self._familiar_id, channel_id=channel_id
            )
            prior_context = summary.summary_text if summary is not None else ""
            # Pull the most recent turn's timestamp for silence-gap.
            recent = self._history.recent(
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
    """``voice:123`` -> ``123``. Returns ``None`` for non-voice sessions."""
    if not session_id.startswith("voice:"):
        return None
    try:
        return int(session_id.split(":", 1)[1])
    except ValueError:
        return None
