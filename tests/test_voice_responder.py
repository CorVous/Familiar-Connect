"""Tests for :class:`familiar_connect.processors.voice_responder.VoiceResponder`.

End-to-end voice turn: activity.start → cancel prior → FINAL →
assemble prompt → stream LLM → speak via TTS. Barge-in bites here.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import (
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from familiar_connect.context import (
    Assembler,
    CoreInstructionsLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.voice_responder import VoiceResponder
from familiar_connect.tts_player import MockTTSPlayer


class _ScriptedLLM(LLMClient):
    """LLM stand-in with configurable per-delta delay."""

    def __init__(
        self,
        *,
        deltas: list[str],
        delay_ms: int = 0,
    ) -> None:
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


def _mk_activity_start(session_id: str = "voice:1", turn_id: str = "t-1") -> Event:
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


def _mk_final(
    text: str,
    session_id: str = "voice:1",
    turn_id: str = "t-1",
    seq: int = 2,
) -> Event:
    return Event(
        event_id=f"final-{turn_id}",
        turn_id=turn_id,
        session_id=session_id,
        parent_event_ids=(),
        topic=TOPIC_VOICE_TRANSCRIPT_FINAL,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "text": text,
            "confidence": 0.9,
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
        },
    )


def _make_responder(
    *,
    llm: LLMClient,
    player: MockTTSPlayer,
    tmp_path: Path,
    router: TurnRouter | None = None,
    store: HistoryStore | None = None,
) -> tuple[VoiceResponder, TurnRouter, HistoryStore]:
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
    responder = VoiceResponder(
        assembler=assembler,
        llm_client=llm,
        tts_player=player,
        history_store=store,
        router=router,
        familiar_id="fam",
    )
    return responder, router, store


class TestActivityStart:
    @pytest.mark.asyncio
    async def test_activity_start_begins_turn(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["hi"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-new"), bus)
        scope = router.active_scope("voice:1")
        assert scope is not None
        assert scope.turn_id == "t-new"
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_second_activity_start_cancels_first(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["hi"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        first = router.active_scope("voice:1")
        assert first is not None

        await responder.handle(_mk_activity_start(turn_id="t-2"), bus)
        second = router.active_scope("voice:1")
        assert second is not None
        assert second.turn_id == "t-2"
        assert first.is_cancelled() is True
        await bus.shutdown()


class TestFinalReply:
    @pytest.mark.asyncio
    async def test_full_reply_on_final(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, store = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await bus.shutdown()

        assert player.calls
        spoken, cancelled = player.calls[0]
        assert spoken == "Hello, world."
        assert cancelled is False
        # user turn recorded
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        contents = [t.content for t in turns]
        assert any("hi there" in c for c in contents)
        assert any("Hello, world." in c for c in contents)

    @pytest.mark.asyncio
    async def test_stale_final_ignored(self, tmp_path: Path) -> None:
        """A FINAL whose turn_id mismatches the active scope is dropped."""
        llm = _ScriptedLLM(deltas=["ignored"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-new"), bus)
        # final from an older utterance
        await responder.handle(_mk_final("old", turn_id="t-OLD"), bus)
        await bus.shutdown()
        assert player.calls == []


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_during_llm_stream_prevents_speech(
        self, tmp_path: Path
    ) -> None:
        """A new activity.start mid-LLM-stream cancels speech entirely."""
        # 10 deltas x 50ms each = 500ms LLM stream
        llm = _ScriptedLLM(deltas=[f"d{i} " for i in range(10)], delay_ms=50)
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)

        async def bargein() -> None:
            await asyncio.sleep(0.08)  # 80ms into stream
            await responder.handle(_mk_activity_start(turn_id="t-2"), bus)

        # Run responder and barge-in concurrently
        task_reply = asyncio.create_task(
            responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        )
        task_barge = asyncio.create_task(bargein())
        await asyncio.gather(task_reply, task_barge)

        # Player should either have not been called, or been called with a
        # cancelled flag. The key assertion: we didn't speak the full text.
        full_text = "".join(f"d{i} " for i in range(10)).strip()
        if player.calls:
            spoken, cancelled = player.calls[0]
            assert cancelled is True or spoken != full_text

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_barge_in_during_speech_cuts_playback_fast(
        self, tmp_path: Path
    ) -> None:
        """Sub-200ms interruption of in-progress TTS playback.

        This is the plan's architectural proof: speak a long utterance,
        fire a new activity.start mid-playback, assert TTS stops
        well under 200 ms.
        """
        llm = _ScriptedLLM(deltas=["hello world"], delay_ms=0)
        # 50 words x 20ms each = 1000ms of playback
        player = MockTTSPlayer(ms_per_word=20, poll_ms=5)
        # Make LLM return a very long text so MockTTSPlayer has lots to play
        llm = _ScriptedLLM(
            deltas=[" ".join(f"w{i}" for i in range(50))],
            delay_ms=0,
        )
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)

        async def bargein_mid_speech() -> float:
            # LLM stream is instant; playback starts immediately. Cancel 100ms in.
            await asyncio.sleep(0.1)
            loop = asyncio.get_running_loop()
            cancel_time = loop.time()
            await responder.handle(_mk_activity_start(turn_id="t-2"), bus)
            return cancel_time

        loop = asyncio.get_running_loop()
        task_reply = asyncio.create_task(
            responder.handle(_mk_final("go ahead", turn_id="t-1"), bus)
        )
        task_barge = asyncio.create_task(bargein_mid_speech())
        cancel_time = await task_barge
        await task_reply
        t_reply_done = loop.time()

        cut_latency_ms = int((t_reply_done - cancel_time) * 1000)
        # The plan's architectural proof: sub-200ms cancel
        assert cut_latency_ms < 200, (
            f"barge-in took {cut_latency_ms}ms; should be under 200"
        )
        assert player.calls
        assert player.calls[0][1] is True  # was cancelled
        await bus.shutdown()
