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
from familiar_connect.diagnostics.cold_cache import log_signals
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import Event, TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.llm import LLMClient
    from familiar_connect.tts_player.protocol import TTSPlayer

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
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._tts = tts_player
        self._history = history_store
        self._router = router
        self._familiar_id = familiar_id

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        if event.topic == TOPIC_VOICE_ACTIVITY_START:
            self._on_activity_start(event)
            return
        if event.topic == TOPIC_VOICE_TRANSCRIPT_FINAL:
            await self._on_final(event)
            return

    # ------------------------------------------------------------------
    # Activity start — cancel prior, install new scope + stop TTS
    # ------------------------------------------------------------------

    def _on_activity_start(self, event: Event) -> None:
        prior = self._router.active_scope(event.session_id)
        self._router.begin_turn(session_id=event.session_id, turn_id=event.turn_id)
        # Tell the current player to flush whatever it's speaking so
        # the cut-point is immediate. The player is stateless across
        # turns, so calling stop() is safe even with no speech in
        # flight.
        if prior is not None:
            # fire-and-forget: tts.stop is expected to be cheap
            asyncio.create_task(  # noqa: RUF006 — best-effort flush
                self._tts.stop(),
                name="voice-responder-tts-stop",
            )

    # ------------------------------------------------------------------
    # Final transcript — run the reply pipeline
    # ------------------------------------------------------------------

    async def _on_final(self, event: Event) -> None:
        scope = self._router.active_scope(event.session_id)
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
            author=None,
        )

        # Seed retrieval cue for RagContextLayer (if wired).
        self._assembler.set_rag_cue(text)

        reply = await self._stream_reply(scope, channel_id)
        if reply is None or scope.is_cancelled():
            return

        # Speak — respects scope cancellation via the TTSPlayer contract.
        await self._tts.speak(reply, scope=scope)

        if scope.is_cancelled():
            return

        self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="assistant",
            content=reply,
            author=None,
        )
        self._router.end_turn(scope)

    async def _stream_reply(self, scope: TurnScope, channel_id: int) -> str | None:
        ctx = AssemblyContext(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            viewer_mode="voice",
        )
        prompt = await self._assembler.assemble(ctx)
        messages: list[Message] = []
        if prompt.system_prompt:
            messages.append(Message(role="system", content=prompt.system_prompt))
        messages.extend(prompt.recent_history)

        accumulated: list[str] = []
        try:
            async for delta in self._llm.chat_stream(messages):
                if scope.is_cancelled():
                    return None
                accumulated.append(delta)
        except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
            _logger.warning(
                f"{ls.tag('Voice', ls.R)} "
                f"{ls.kv('llm_stream_error', repr(exc), vc=ls.R)}"
            )
            return None
        return "".join(accumulated)

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
