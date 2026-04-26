"""Text reply orchestrator.

Consumes ``discord.text`` events and produces an LLM reply, posted via
the injected ``send_text`` callback. Mirrors :class:`VoiceResponder`
but skips TTS — text channels render the assistant string directly.

Owns user-turn writes for ``discord.text`` (single-writer per channel
keeps read-after-write consistency for ``RecentHistoryLayer`` in the
same task — a separate :class:`HistoryWriter` task would race with
the responder's ``assemble`` call). Cancellation flows through
:class:`TurnRouter` so future barge-in (e.g. user sends a follow-up
mid-stream) cancels in-flight LLM work.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.identity import Author
from familiar_connect.llm import Message
from familiar_connect.silence import SilentDetector

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.bus.envelope import Event, TurnScope
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.context.assembler import Assembler
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.text_responder")


class TextResponder:
    """Stream LLM replies for ``discord.text`` events; post via ``send_text``."""

    name: str = "text-responder"
    topics: tuple[str, ...] = (TOPIC_DISCORD_TEXT,)

    def __init__(
        self,
        *,
        assembler: Assembler,
        llm_client: LLMClient,
        send_text: Callable[[int, str], Awaitable[None]],
        history_store: HistoryStore,
        router: TurnRouter,
        familiar_id: str,
    ) -> None:
        self._assembler = assembler
        self._llm = llm_client
        self._send_text = send_text
        self._history = history_store
        self._router = router
        self._familiar_id = familiar_id
        # in-process dedup; bus does not republish today, but cheap insurance
        self._seen: set[str] = set()

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        if event.topic != TOPIC_DISCORD_TEXT:
            return
        if event.event_id in self._seen:
            return
        payload = event.payload
        if not isinstance(payload, dict):
            return
        if payload.get("familiar_id") != self._familiar_id:
            return
        channel_id = payload.get("channel_id")
        content = payload.get("content") or ""
        if not isinstance(channel_id, int) or not content:
            return
        raw_author = payload.get("author")
        author = raw_author if isinstance(raw_author, Author) else None
        raw_guild = payload.get("guild_id")
        guild_id = raw_guild if isinstance(raw_guild, int) else None

        self._seen.add(event.event_id)
        scope = self._router.begin_turn(
            session_id=event.session_id, turn_id=event.turn_id
        )

        # Persist user turn *before* streaming so RecentHistoryLayer
        # in the same task sees it. Mirrors VoiceResponder.
        self._history.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="user",
            content=content,
            author=author,
            guild_id=guild_id,
        )
        # seed retrieval before assembly so RagContextLayer sees the cue
        self._assembler.set_rag_cue(content)

        reply = await self._stream_reply(scope, channel_id)
        if reply is None or scope.is_cancelled():
            return
        # Discord rejects empty / whitespace-only messages (HTTP 400,
        # error code 50006). An empty reply usually means the LLM
        # stream emitted no deltas — bad model name, content filter,
        # or upstream error frame the parser silently dropped.
        if not reply.strip():
            _logger.warning(
                f"{ls.tag('Text', ls.Y)} "
                f"{ls.kv('skip', 'empty_reply', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return

        try:
            await self._send_text(channel_id, reply)
        except Exception as exc:  # noqa: BLE001 — surface but don't crash loop
            _logger.warning(
                f"{ls.tag('Text', ls.R)} {ls.kv('send_error', repr(exc), vc=ls.R)}"
            )
            return
        _logger.info(
            f"{ls.tag('💬 Text', ls.G)} "
            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
            f"{ls.kv('chars', str(len(reply)), vc=ls.LW)} "
            f"{ls.kv('text', ls.trunc(reply, 200), vc=ls.LW)}"
        )

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
            viewer_mode="text",
        )
        prompt = await self._assembler.assemble(ctx)
        messages: list[Message] = []
        if prompt.system_prompt:
            messages.append(Message(role="system", content=prompt.system_prompt))
        messages.extend(prompt.recent_history)

        accumulated: list[str] = []
        silent = SilentDetector()
        try:
            async for delta in self._llm.chat_stream(messages):
                if scope.is_cancelled():
                    return None
                accumulated.append(delta)
                if silent.feed(delta) is True:
                    # Mirror cancellation: no send, no assistant turn.
                    # The 'decision=silent' log replaces the would-be
                    # empty-reply warning (which targets bad model /
                    # content-filter cases, not deliberate silence).
                    _logger.info(
                        f"{ls.tag('💤 Text', ls.B)} "
                        f"{ls.kv('decision', 'silent', vc=ls.LB)} "
                        f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                    )
                    return None
        except Exception as exc:  # noqa: BLE001 — stream errors shouldn't crash loop
            _logger.warning(
                f"{ls.tag('Text', ls.R)} "
                f"{ls.kv('llm_stream_error', repr(exc), vc=ls.R)}"
            )
            return None
        return "".join(accumulated)
