"""TDD tests for focus-aware responder behavior (Phase 4, Issue #107).

Covers:
- TextResponder with focus_manager: unfocused message staged, no reply
- TextResponder with focus_manager: focused message generates reply, end_turn called
- TextResponder without focus_manager: backward compat (existing behavior)
- VoiceResponder with focus_manager: end_turn called after turn completes
- VoiceResponder without focus_manager: backward compat
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import (
    TOPIC_DISCORD_TEXT,
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from familiar_connect.context import (
    Assembler,
    CharacterCardLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.text_responder import TextResponder
from familiar_connect.processors.voice_responder import VoiceResponder
from familiar_connect.tts_player import MockTTSPlayer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Scripted LLM helpers
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLMClient):
    """Returns a fixed reply string."""

    def __init__(self, reply: str = "hello") -> None:
        super().__init__(api_key="k", model="m")
        self._reply = reply
        self.call_count = 0

    async def chat(self, messages: list[Message]) -> Message:
        self.call_count += 1
        return Message(role="assistant", content=self._reply)

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],  # noqa: ARG002
    ) -> AsyncIterator[str]:
        self.call_count += 1
        yield self._reply


class _CapturingSend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str | None, tuple[int, ...]]] = []

    async def __call__(
        self,
        channel_id: int,
        content: str,
        reply_to_message_id: str | None = None,
        mention_user_ids: tuple[int, ...] = (),
    ) -> str | None:
        self.calls.append((channel_id, content, reply_to_message_id, mention_user_ids))
        return "msg-sent"


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def _text_event(
    *,
    channel_id: int = 100,
    content: str = "hello",
    event_id: str = "e-1",
) -> Event:
    return Event(
        event_id=event_id,
        turn_id=event_id,
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=1,
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


def _activity_start(
    *,
    session_id: str = "voice:200",
    turn_id: str = "t-1",
) -> Event:
    return Event(
        event_id=f"act-{turn_id}",
        turn_id=turn_id,
        session_id=session_id,
        parent_event_ids=(),
        topic=TOPIC_VOICE_ACTIVITY_START,
        timestamp=datetime.now(tz=UTC),
        sequence_number=1,
        payload=None,
    )


def _final(
    text: str,
    *,
    session_id: str = "voice:200",
    turn_id: str = "t-1",
) -> Event:
    return Event(
        event_id=f"fin-{turn_id}",
        turn_id=turn_id,
        session_id=session_id,
        parent_event_ids=(),
        topic=TOPIC_VOICE_TRANSCRIPT_FINAL,
        timestamp=datetime.now(tz=UTC),
        sequence_number=2,
        payload={
            "text": text,
            "user_id": None,
        },
    )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_text_responder(
    *,
    tmp_path: "Path",
    llm: LLMClient | None = None,
    send: _CapturingSend | None = None,
    focus_manager: object = None,
) -> tuple[TextResponder, HistoryStore]:
    card = tmp_path / "character.md"
    card.write_text("You are a familiar.\n")
    store = HistoryStore(":memory:")
    async_store = AsyncHistoryStore(store)
    assembler = Assembler(
        layers=[
            CharacterCardLayer(card_path=card),
            RecentHistoryLayer(store=async_store, window_size=20),
        ]
    )
    router = TurnRouter()
    responder = TextResponder(
        assembler=assembler,
        llm_client=llm or _ScriptedLLM("hi there"),
        send_text=send or _CapturingSend(),
        history_store=async_store,
        router=router,
        familiar_id="fam",
        focus_manager=focus_manager,
    )
    return responder, store


def _make_voice_responder(
    *,
    tmp_path: "Path",
    llm: LLMClient | None = None,
    player: MockTTSPlayer | None = None,
    focus_manager: object = None,
) -> tuple[VoiceResponder, HistoryStore]:
    card = tmp_path / "character.md"
    card.write_text("You are a familiar.\n")
    store = HistoryStore(":memory:")
    async_store = AsyncHistoryStore(store)
    assembler = Assembler(
        layers=[
            CharacterCardLayer(card_path=card),
            RecentHistoryLayer(store=async_store, window_size=20),
        ]
    )
    router = TurnRouter()
    responder = VoiceResponder(
        assembler=assembler,
        llm_client=llm or _ScriptedLLM("hello"),
        tts_player=player or MockTTSPlayer(ms_per_word=1),
        history_store=async_store,
        router=router,
        familiar_id="fam",
        focus_manager=focus_manager,
    )
    return responder, store


# ---------------------------------------------------------------------------
# Mock FocusManager
# ---------------------------------------------------------------------------


def _focused_focus_manager(channel_id: int = 100) -> MagicMock:
    """FocusManager where is_focused returns True for given channel."""
    fm = MagicMock()
    fm.is_focused = MagicMock(return_value=True)
    fm.get_focus = MagicMock(return_value=channel_id)
    fm.end_turn = AsyncMock()
    # staged_channels returns empty dict (no staged turns)
    return fm


def _unfocused_focus_manager() -> MagicMock:
    """FocusManager where is_focused always returns False."""
    fm = MagicMock()
    fm.is_focused = MagicMock(return_value=False)
    fm.get_focus = MagicMock(return_value=999)  # some other channel
    fm.end_turn = AsyncMock()
    return fm


# ---------------------------------------------------------------------------
# TextResponder tests
# ---------------------------------------------------------------------------


class TestTextResponderFocusAware:
    @pytest.mark.asyncio
    async def test_unfocused_message_is_staged_no_reply(self, tmp_path: "Path") -> None:
        """Unfocused channel: turn staged (consumed=False), no reply generated."""
        llm = _ScriptedLLM("should not be called")
        send = _CapturingSend()
        fm = _unfocused_focus_manager()

        responder, store = _make_text_responder(
            tmp_path=tmp_path, llm=llm, send=send, focus_manager=fm
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        # no reply sent
        assert send.calls == []
        # LLM not invoked
        assert llm.call_count == 0
        # user turn was recorded
        turns = store.recent(familiar_id="fam", channel_id=100, limit=10)
        assert any(t.role == "user" and "hello" in t.content for t in turns)
        # turn is staged: consumed_at is None
        raw = store._conn.execute(
            "SELECT consumed_at FROM turns WHERE familiar_id='fam' AND role='user'"
        ).fetchone()
        assert raw is not None
        assert raw["consumed_at"] is None

    @pytest.mark.asyncio
    async def test_focused_message_generates_reply_and_calls_end_turn(
        self, tmp_path: "Path"
    ) -> None:
        """Focused channel: reply generated, end_turn called on focus_manager."""
        llm = _ScriptedLLM("hi there")
        send = _CapturingSend()
        fm = _focused_focus_manager(channel_id=100)

        responder, _ = _make_text_responder(
            tmp_path=tmp_path, llm=llm, send=send, focus_manager=fm
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        # reply sent
        assert len(send.calls) == 1
        assert send.calls[0][1] == "hi there"
        # end_turn called
        fm.end_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_focused_passes_focus_context_to_final_reminder(
        self, tmp_path: "Path"
    ) -> None:
        """Focus context (channel_id, digest) passed to build_final_reminder."""
        llm = _ScriptedLLM("ok")
        send = _CapturingSend()
        fm = _focused_focus_manager(channel_id=100)

        # staged_channels should be called and its result forwarded
        async_mock_store = MagicMock()
        async_mock_store.staged_channels = AsyncMock(return_value={200: 3})
        async_mock_store.sync = MagicMock()

        responder, _ = _make_text_responder(
            tmp_path=tmp_path, llm=llm, send=send, focus_manager=fm
        )

        captured_calls: list[dict] = []

        original_build = __import__(
            "familiar_connect.context.final_reminder",
            fromlist=["build_final_reminder"],
        ).build_final_reminder

        def _capturing_build(**kwargs: object) -> str:
            captured_calls.append(dict(kwargs))
            return original_build(**kwargs)

        with patch(
            "familiar_connect.processors.text_responder.build_final_reminder",
            side_effect=_capturing_build,
        ):
            bus = InProcessEventBus()
            await bus.start()
            try:
                await responder.handle(_text_event(channel_id=100), bus)
            finally:
                await bus.shutdown()

        # at least one call should have focus_channel_id set
        focus_calls = [c for c in captured_calls if c.get("focus_channel_id") == 100]
        assert focus_calls, (
            f"No build_final_reminder call with focus_channel_id=100; calls={captured_calls}"
        )

    @pytest.mark.asyncio
    async def test_no_focus_manager_backward_compat(self, tmp_path: "Path") -> None:
        """Without focus_manager: behaves exactly as before (reply generated)."""
        llm = _ScriptedLLM("compat reply")
        send = _CapturingSend()

        # no focus_manager
        responder, _ = _make_text_responder(
            tmp_path=tmp_path, llm=llm, send=send, focus_manager=None
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        assert len(send.calls) == 1
        assert send.calls[0][1] == "compat reply"

    @pytest.mark.asyncio
    async def test_unfocused_does_not_call_end_turn(self, tmp_path: "Path") -> None:
        """end_turn not called for unfocused (staged) turns."""
        fm = _unfocused_focus_manager()
        responder, _ = _make_text_responder(tmp_path=tmp_path, focus_manager=fm)
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        fm.end_turn.assert_not_awaited()


# ---------------------------------------------------------------------------
# VoiceResponder tests
# ---------------------------------------------------------------------------


class TestVoiceResponderFocusAware:
    @pytest.mark.asyncio
    async def test_voice_turn_calls_end_turn_when_focus_manager_set(
        self, tmp_path: "Path"
    ) -> None:
        """After a completed voice turn, end_turn is called on focus_manager."""
        llm = _ScriptedLLM("sure")
        player = MockTTSPlayer(ms_per_word=1)
        fm = _focused_focus_manager(channel_id=200)

        responder, _ = _make_voice_responder(
            tmp_path=tmp_path, llm=llm, player=player, focus_manager=fm
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_activity_start(session_id="voice:200"), bus)
            await responder.handle(_final("hi there", session_id="voice:200"), bus)
            await responder.wait_until_idle()
        finally:
            await bus.shutdown()

        fm.end_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_voice_no_focus_manager_backward_compat(
        self, tmp_path: "Path"
    ) -> None:
        """Without focus_manager: voice turn proceeds exactly as before."""
        llm = _ScriptedLLM("hello there")
        player = MockTTSPlayer(ms_per_word=1)

        responder, store = _make_voice_responder(
            tmp_path=tmp_path, llm=llm, player=player, focus_manager=None
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_activity_start(session_id="voice:200"), bus)
            await responder.handle(_final("hi there", session_id="voice:200"), bus)
            await responder.wait_until_idle()
        finally:
            await bus.shutdown()

        assert player.calls
        spoken, cancelled = player.calls[0]
        assert "hello there" in spoken
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_voice_end_turn_not_called_on_silent(self, tmp_path: "Path") -> None:
        """When voice reply is silent, end_turn should not be called (no actual turn)."""
        llm = _ScriptedLLM("<silent>")
        player = MockTTSPlayer(ms_per_word=1)
        fm = _focused_focus_manager(channel_id=200)

        responder, _ = _make_voice_responder(
            tmp_path=tmp_path, llm=llm, player=player, focus_manager=fm
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_activity_start(session_id="voice:200"), bus)
            await responder.handle(_final("hi there", session_id="voice:200"), bus)
            await responder.wait_until_idle()
        finally:
            await bus.shutdown()

        # silent: no TTS, no end_turn
        assert player.calls == []
        fm.end_turn.assert_not_awaited()
