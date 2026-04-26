"""Tests for :class:`familiar_connect.processors.text_responder.TextResponder`.

End-to-end text turn: ``discord.text`` event in → assemble prompt →
stream LLM → call ``send_text(channel_id, reply)`` → append assistant
turn to history. Mirrors :class:`VoiceResponder` but without TTS.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

import asyncio

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context import (
    Assembler,
    CoreInstructionsLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.text_responder import TextResponder


class _ScriptedLLM(LLMClient):
    """Yield a fixed delta sequence; optional per-delta delay."""

    def __init__(self, *, deltas: list[str], delay_ms: int = 0) -> None:
        super().__init__(api_key="k", model="m")
        self._deltas = list(deltas)
        self._delay_ms = delay_ms

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        return Message(role="assistant", content="".join(self._deltas))

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],  # noqa: ARG002
    ) -> AsyncIterator[str]:
        for d in self._deltas:
            if self._delay_ms:
                await asyncio.sleep(self._delay_ms / 1000.0)
            yield d


class _CapturingSend:
    """Records ``send_text`` invocations for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, channel_id: int, content: str) -> None:
        self.calls.append((channel_id, content))


def _discord_text_event(
    *,
    event_id: str = "e-1",
    channel_id: int = 42,
    content: str = "hi there",
    seq: int = 1,
) -> Event:
    return Event(
        event_id=event_id,
        turn_id=event_id,
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "familiar_id": "fam",
            "channel_id": channel_id,
            "guild_id": 99,
            "author": Author(
                platform="discord",
                user_id="1",
                username="alice",
                display_name="Alice",
            ),
            "content": content,
        },
    )


def _make_responder(
    *,
    llm: LLMClient,
    send: _CapturingSend,
    tmp_path: Path,
    router: TurnRouter | None = None,
    store: HistoryStore | None = None,
) -> tuple[TextResponder, TurnRouter, HistoryStore]:
    core = tmp_path / "core.md"
    core.write_text("You are a familiar.\n")
    store = store or HistoryStore(":memory:")
    assembler = Assembler(
        layers=[
            CoreInstructionsLayer(path=core),
            RecentHistoryLayer(store=store, window_size=20),
        ]
    )
    router = router or TurnRouter()
    responder = TextResponder(
        assembler=assembler,
        llm_client=llm,
        send_text=send,
        history_store=store,
        router=router,
        familiar_id="fam",
    )
    return responder, router, store


class TestProcessorSurface:
    def test_topic_is_discord_text(self, tmp_path: Path) -> None:
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=_ScriptedLLM(deltas=["x"]),
            send=send,
            tmp_path=tmp_path,
        )
        assert responder.name == "text-responder"
        assert TOPIC_DISCORD_TEXT in responder.topics


class TestReply:
    @pytest.mark.asyncio
    async def test_streams_reply_and_calls_send_text(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        send = _CapturingSend()
        responder, _, store = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_discord_text_event(content="hi there"), bus)
        await bus.shutdown()

        assert send.calls == [(42, "Hello, world.")]
        # assistant turn appended (user-turn write is HistoryWriter's job)
        turns = store.recent(familiar_id="fam", channel_id=42, limit=10)
        roles_contents = [(t.role, t.content) for t in turns]
        assert ("assistant", "Hello, world.") in roles_contents

    @pytest.mark.asyncio
    async def test_skips_when_payload_missing_content(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["nope"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content=""), bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_skips_other_familiars(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["nope"])
        send = _CapturingSend()
        responder, _, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        ev = _discord_text_event()
        payload = dict(ev.payload)
        payload["familiar_id"] = "other"
        other = Event(
            event_id="other",
            turn_id="other",
            session_id=ev.session_id,
            parent_event_ids=(),
            topic=ev.topic,
            timestamp=ev.timestamp,
            sequence_number=ev.sequence_number,
            payload=payload,
        )
        await responder.handle(other, bus)
        await bus.shutdown()
        assert send.calls == []

    @pytest.mark.asyncio
    async def test_begins_turn_for_session(self, tmp_path: Path) -> None:
        """Router should track an active scope for the discord session."""
        llm = _ScriptedLLM(deltas=["ok"])
        send = _CapturingSend()
        responder, router, _ = _make_responder(llm=llm, send=send, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_discord_text_event(content="hi"), bus)
        await bus.shutdown()
        # turn ended cleanly — no active scope after completion
        assert router.active_scope("discord:42") is None
        assert send.calls
