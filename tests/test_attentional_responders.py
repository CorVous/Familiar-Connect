"""TDD tests for focus-aware responder behavior (Phase 4, Issue #107).

Covers:
- TextResponder with focus_manager: unfocused message staged, no reply
- TextResponder with focus_manager: focused message generates reply, end_turn called
- TextResponder without focus_manager: backward compat (existing behavior)
- VoiceResponder with focus_manager: end_turn called after turn completes
- VoiceResponder without focus_manager: backward compat
"""

from __future__ import annotations

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
from familiar_connect.focus import FocusManager
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import ChannelUnread, HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.processors.text_responder import TextResponder
from familiar_connect.processors.voice_responder import VoiceResponder
from familiar_connect.subscriptions import SubscriptionKind, SubscriptionRegistry
from familiar_connect.tools.registry import ToolContext, ToolRegistry
from familiar_connect.tools.shift_focus import build_shift_focus_tool
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

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
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
    tmp_path: Path,
    llm: LLMClient | None = None,
    send: _CapturingSend | None = None,
    focus_manager: FocusManager | None = None,
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
    tmp_path: Path,
    llm: LLMClient | None = None,
    player: MockTTSPlayer | None = None,
    focus_manager: FocusManager | None = None,
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
    fm.should_wake = MagicMock(return_value=False)
    fm.mark_nudge_pending = MagicMock()
    # staged_channels returns empty dict (no staged turns)
    return fm


def _unfocused_focus_manager() -> MagicMock:
    """FocusManager where is_focused always returns False."""
    fm = MagicMock()
    fm.is_focused = MagicMock(return_value=False)
    fm.get_focus = MagicMock(return_value=999)  # some other channel
    fm.end_turn = AsyncMock()
    fm.should_wake = MagicMock(return_value=False)
    fm.mark_nudge_pending = MagicMock()
    return fm


# ---------------------------------------------------------------------------
# TextResponder tests
# ---------------------------------------------------------------------------


class TestTextResponderFocusAware:
    @pytest.mark.asyncio
    async def test_unfocused_message_is_staged_no_reply(self, tmp_path: Path) -> None:
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
        self, tmp_path: Path
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
        self, tmp_path: Path
    ) -> None:
        """Focus context (channel_id, digest) passed to build_final_reminder."""
        llm = _ScriptedLLM("ok")
        send = _CapturingSend()
        fm = _focused_focus_manager(channel_id=100)

        # staged_channels should be called and its result forwarded
        async_mock_store = MagicMock()
        async_mock_store.staged_channels = AsyncMock(
            return_value={200: ChannelUnread(3, 0)}
        )
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
            "No build_final_reminder call with focus_channel_id=100; "
            f"calls={captured_calls}"
        )

    @pytest.mark.asyncio
    async def test_no_focus_manager_backward_compat(self, tmp_path: Path) -> None:
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
    async def test_unfocused_does_not_call_end_turn(self, tmp_path: Path) -> None:
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


class TestTextResponderIdleNudge:
    @pytest.mark.asyncio
    async def test_unfocused_arrival_publishes_wake_to_focused_channel(
        self, tmp_path: Path
    ) -> None:
        """should_wake → publish wake event routed at the focused channel."""
        fm = _unfocused_focus_manager()
        fm.should_wake = MagicMock(return_value=True)
        fm.get_focus = MagicMock(return_value=555)  # focused text channel
        responder, _ = _make_text_responder(tmp_path=tmp_path, focus_manager=fm)

        bus = InProcessEventBus()
        await bus.start()
        published: list[Event] = []

        async def _capture(event: Event) -> None:  # noqa: RUF029
            published.append(event)

        bus.publish = _capture  # ty: ignore[invalid-assignment]
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        fm.mark_nudge_pending.assert_called_once()
        wakes = [e for e in published if e.payload.get("wake") is True]
        assert len(wakes) == 1
        assert wakes[0].payload["channel_id"] == 555

    @pytest.mark.asyncio
    async def test_unfocused_arrival_publishes_nothing(self, tmp_path: Path) -> None:
        """should_wake False → no nudge, plain staging."""
        fm = _unfocused_focus_manager()  # should_wake defaults False
        responder, _ = _make_text_responder(tmp_path=tmp_path, focus_manager=fm)

        bus = InProcessEventBus()
        await bus.start()
        published: list[Event] = []

        async def _capture(event: Event) -> None:  # noqa: RUF029
            published.append(event)

        bus.publish = _capture  # ty: ignore[invalid-assignment]
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        fm.mark_nudge_pending.assert_not_called()
        assert [e for e in published if e.payload.get("wake")] == []

    @pytest.mark.asyncio
    async def test_wake_event_replies_without_persisting_user_turn(
        self, tmp_path: Path
    ) -> None:
        """Wake event: focused turn runs, but no synthetic user turn stored."""
        llm = _ScriptedLLM("checking in")
        send = _CapturingSend()
        fm = _focused_focus_manager(channel_id=555)
        responder, store = _make_text_responder(
            tmp_path=tmp_path, llm=llm, send=send, focus_manager=fm
        )

        wake = Event(
            event_id="wake-1",
            turn_id="wake-1",
            session_id="discord:555",
            parent_event_ids=(),
            topic=TOPIC_DISCORD_TEXT,
            timestamp=datetime.now(tz=UTC),
            sequence_number=0,
            payload={
                "familiar_id": "fam",
                "channel_id": 555,
                "content": "[idle check]",
                "author": None,
                "wake": True,
            },
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(wake, bus)
        finally:
            await bus.shutdown()

        # reply generated + turn finalized
        assert len(send.calls) == 1
        fm.end_turn.assert_awaited_once()
        # no synthetic user turn persisted (no transcript pollution)
        user_turns = [
            t
            for t in store.recent(familiar_id="fam", channel_id=555, limit=10)
            if t.role == "user"
        ]
        assert user_turns == []


# ---------------------------------------------------------------------------
# VoiceResponder tests
# ---------------------------------------------------------------------------


class TestVoiceResponderFocusAware:
    @pytest.mark.asyncio
    async def test_voice_turn_calls_end_turn_when_focus_manager_set(
        self, tmp_path: Path
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
    async def test_voice_no_focus_manager_backward_compat(self, tmp_path: Path) -> None:
        """Without focus_manager: voice turn proceeds exactly as before."""
        llm = _ScriptedLLM("hello there")
        player = MockTTSPlayer(ms_per_word=1)

        responder, _store = _make_voice_responder(
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
    async def test_voice_end_turn_not_called_on_silent(self, tmp_path: Path) -> None:
        """Voice reply is silent: end_turn not called (no actual turn)."""
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


# ---------------------------------------------------------------------------
# Immediate shift_focus (no deferral): peek = real move
# ---------------------------------------------------------------------------


class _ScriptedToolLLM(LLMClient):
    """LLM stand-in yielding LLMDeltas from a per-call script (tool mode)."""

    def __init__(self, scripts: list[list[LLMDelta]]) -> None:
        super().__init__(api_key="k", model="m", tool_calling=True)
        self._scripts = list(scripts)
        self.calls: list[list[Message]] = []

    async def stream_completion(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,  # noqa: ARG002
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append(list(messages))
        script = self._scripts.pop(0) if self._scripts else []
        for d in script:
            yield d


def _shift_tc(channel_id: int) -> LLMDelta:
    return LLMDelta(
        tool_calls=[
            {
                "index": 0,
                "id": "sf-1",
                "type": "function",
                "function": {
                    "name": "shift_focus",
                    "arguments": f'{{"channel_id": {channel_id}}}',
                },
            }
        ]
    )


def _real_focus_responder(
    *,
    tmp_path: Path,
    llm: LLMClient,
    send: _CapturingSend,
) -> tuple[TextResponder, FocusManager, HistoryStore]:
    """Responder wired to a *real* FocusManager + shift_focus tool.

    Channels 100 + 200 subscribed (text); focus starts on 100.
    """
    subs = SubscriptionRegistry(tmp_path / "subscriptions.toml")
    subs.add(channel_id=100, kind=SubscriptionKind.text, guild_id=99)
    subs.add(channel_id=200, kind=SubscriptionKind.text, guild_id=99)

    store = HistoryStore(":memory:")
    async_store = AsyncHistoryStore(store)
    fm = FocusManager(
        familiar_id="fam", store=async_store, subscriptions=subs, unread_nudge_enabled=False
    )
    fm.set_focus_immediately(100, "text")

    card = tmp_path / "character.md"
    card.write_text("You are a familiar.\n")
    assembler = Assembler(
        layers=[
            CharacterCardLayer(card_path=card),
            RecentHistoryLayer(store=async_store, window_size=20),
        ]
    )
    registry = ToolRegistry()
    registry.register(build_shift_focus_tool())

    def _ctx_factory(
        channel_id: int, turn_id: str, images: dict | None = None
    ) -> ToolContext:
        return ToolContext(
            familiar_id="fam",
            channel_id=channel_id,
            channel_kind="text",
            turn_id=turn_id,
            history=async_store,
            bus=InProcessEventBus(),
            images=images or {},
            focus_manager=fm,
            store=async_store,
        )

    responder = TextResponder(
        assembler=assembler,
        llm_client=llm,
        send_text=send,
        history_store=async_store,
        router=TurnRouter(),
        familiar_id="fam",
        focus_manager=fm,
        tool_registry=registry,
        tool_context_factory=_ctx_factory,
    )
    return responder, fm, store


class TestImmediateShiftFocus:
    @pytest.mark.asyncio
    async def test_silent_shift_focus_moves_immediately(self, tmp_path: Path) -> None:
        """shift_focus applies at tool-call time, even on a silent turn.

        Model peeks #200 then stays silent. Because the shift is immediate
        (not deferred to a reply that never comes), she is now focused on
        #200 — the peek was a real move.
        """
        llm = _ScriptedToolLLM([
            [_shift_tc(200), LLMDelta(finish_reason="tool_calls")],
            [LLMDelta(content="<silent>"), LLMDelta(finish_reason="stop")],
        ])
        send = _CapturingSend()
        responder, fm, _ = _real_focus_responder(tmp_path=tmp_path, llm=llm, send=send)
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        assert send.calls == []  # silent: nothing posted
        assert fm.get_focus("text") == 200  # moved at tool-call time

    @pytest.mark.asyncio
    async def test_silent_peek_then_old_channel_message_stages(
        self, tmp_path: Path
    ) -> None:
        """The #general→#media bug cannot occur under immediate shift.

        Turn 1: peek #200 + silent → she's now in #200. Turn 2: a message
        in #100 (her *old* channel) is unfocused, so it stages — no reply,
        no misroute.
        """
        llm = _ScriptedToolLLM([
            [_shift_tc(200), LLMDelta(finish_reason="tool_calls")],
            [LLMDelta(content="<silent>"), LLMDelta(finish_reason="stop")],
            # turn 2 should never reach the LLM (message stages)
            [LLMDelta(content="should not send"), LLMDelta(finish_reason="stop")],
        ])
        send = _CapturingSend()
        responder, fm, store = _real_focus_responder(
            tmp_path=tmp_path, llm=llm, send=send
        )
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(
                _text_event(channel_id=100, content="peek", event_id="e-1"), bus
            )
            await responder.handle(
                _text_event(channel_id=100, content="ping", event_id="e-2"), bus
            )
        finally:
            await bus.shutdown()

        assert fm.get_focus("text") == 200
        assert send.calls == []  # turn 2 staged, not replied/misrouted
        # the #100 message was staged (consumed_at NULL)
        raw = store._conn.execute(
            "SELECT consumed_at FROM turns "
            "WHERE familiar_id='fam' AND role='user' AND content='ping'"
        ).fetchone()
        assert raw is not None
        assert raw["consumed_at"] is None

    @pytest.mark.asyncio
    async def test_shift_focus_with_reply_posts_to_new_channel(
        self, tmp_path: Path
    ) -> None:
        """Behavior preserved: shift + actual reply posts to the new channel."""
        llm = _ScriptedToolLLM([
            [_shift_tc(200), LLMDelta(finish_reason="tool_calls")],
            [LLMDelta(content="hello over here"), LLMDelta(finish_reason="stop")],
        ])
        send = _CapturingSend()
        responder, fm, _ = _real_focus_responder(tmp_path=tmp_path, llm=llm, send=send)
        bus = InProcessEventBus()
        await bus.start()
        try:
            await responder.handle(_text_event(channel_id=100), bus)
        finally:
            await bus.shutdown()

        assert fm.get_focus("text") == 200
        assert len(send.calls) == 1
        assert send.calls[0][0] == 200
        assert send.calls[0][1] == "hello over here"
